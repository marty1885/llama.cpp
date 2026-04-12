// experiments/hybrid-edit.cpp — benchmark edit methods with aggregate metrics.
//
// Compares 5 methods on 5-entity editing:
//   UNIFORM   — full-prompt delta, uniform alpha (oracle, not reusable)
//   ISOLATE   — isolated delta, R1-WT weighting
//   FULL-R1   — full-prompt delta, R1-WT weighting
//   HYB-UV    — offline u₁ + online v₁, offline σ
//   HYB-U     — offline u₁ + online v₁, online σ
//
// Metrics per method (aggregated across entities and alpha sweep):
//   flip_rate  — fraction of target entities that flip (P < 0.5)
//   best_alpha — lowest alpha at which target flips with no leak
//   selectivity— mean (1 - max_control_drop) across successful edits
//   mean_P_tgt — mean P(old) for target at best alpha
//
// Usage:
//   llama-rwkv-hybrid-edit -m model.gguf [--layers 16-31] [--alpha 1.0]

#include "rwkv_probe/state.h"
#include "rwkv_probe/pipeline.h"
#include "rwkv_probe/edit.h"
#include "rwkv_probe/head_decomp.h"
#include "rwkv_probe/experiment.h"

#include "rwkv_probe/numeric.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

namespace rp = rwkv_probe;

// Single-token probe: full HeadDecomp per head from single-token delta.
static std::vector<rp::HeadDecomp> online_probe_decomps(
        rp::Context & ctx, const rp::ModelGeometry & geom,
        const llama_vocab * vocab,
        const std::string & prefix,
        llama_token tok_a, llama_token tok_b,
        const std::vector<int> & layers) {

    auto state_prefix = rp::capture(ctx, geom, vocab, prefix);

    rp::StateBuf s_after_a(geom);
    state_prefix.store_to(ctx.raw());
    ctx.decode_one(tok_a);
    s_after_a.load_from(ctx.raw());

    rp::StateBuf s_after_b(geom);
    state_prefix.store_to(ctx.raw());
    ctx.decode_one(tok_b);
    s_after_b.load_from(ctx.raw());

    return rp::decompose(s_after_a, s_after_b, layers);
}

// HYBRID: offline u₁, online v₁.
static void apply_hybrid_uv(rp::StateBuf & dst,
                            const std::vector<rp::HeadDecomp> & offline,
                            const std::vector<rp::HeadDecomp> & online,
                            double alpha, bool use_online_sigma) {
    int hs = dst.geom().head_size;
    for (std::size_t i = 0; i < offline.size(); ++i) {
        const auto & off = offline[i];
        const auto & on  = online[i];
        if (off.sigma < 1e-10 || on.sigma < 1e-10) continue;
        double sigma = use_online_sigma ? on.sigma : off.sigma;
        auto dd = dst.s_head(off.layer, off.head);
        for (int r = 0; r < hs; ++r) {
            for (int c = 0; c < hs; ++c) {
                dd[r * hs + c] += (float)(alpha * sigma * off.u[r] * on.v[c]);
            }
        }
    }
}

// ONLINE: full rank-1 from online probe (no offline component)
static void apply_online_r1(rp::StateBuf & dst,
                            const std::vector<rp::HeadDecomp> & online,
                            double alpha) {
    int hs = dst.geom().head_size;
    for (const auto & on : online) {
        if (on.sigma < 1e-10) continue;
        auto dd = dst.s_head(on.layer, on.head);
        for (int r = 0; r < hs; ++r) {
            for (int c = 0; c < hs; ++c) {
                dd[r * hs + c] += (float)(alpha * on.sigma * on.u[r] * on.v[c]);
            }
        }
    }
}

// ONLINE-ADAP: ONLINE with adaptive scaling (ratio * state_norm / edit_norm)
static void apply_online_adaptive(rp::StateBuf & dst,
                                  const std::vector<rp::HeadDecomp> & online,
                                  double ratio) {
    int hs = dst.geom().head_size;
    const auto & geom = dst.geom();

    // compute state norm
    double s_norm_sq = 0;
    for (int l = 0; l < geom.n_layer; ++l) {
        for (int h = 0; h < geom.n_head; ++h) {
            auto s = dst.s_head(l, h);
            for (int i = 0; i < hs * hs; ++i) s_norm_sq += (double)s[i] * s[i];
        }
    }

    // compute edit norm from online sigmas
    double e_norm_sq = 0;
    for (const auto & on : online) e_norm_sq += on.sigma * on.sigma;

    double alpha = ratio;
    if (e_norm_sq > 1e-20) alpha *= std::sqrt(s_norm_sq) / std::sqrt(e_norm_sq);

    apply_online_r1(dst, online, alpha);
}

// ONLINE-ADAP2: ONLINE with adaptive scaling using full delta Frobenius norm
static void apply_online_adaptive2(rp::StateBuf & dst,
                                   const rp::StateBuf & sa, const rp::StateBuf & sb,
                                   const std::vector<rp::HeadDecomp> & online,
                                   const std::vector<int> & layers,
                                   double ratio) {
    int hs = dst.geom().head_size;
    int n_head = dst.geom().n_head;
    const auto & geom = dst.geom();

    // state norm
    double s_norm_sq = 0;
    for (int l = 0; l < geom.n_layer; ++l) {
        for (int h = 0; h < geom.n_head; ++h) {
            auto s = dst.s_head(l, h);
            for (int i = 0; i < hs * hs; ++i) s_norm_sq += (double)s[i] * s[i];
        }
    }

    // full delta Frobenius norm (not rank-1 sigma sum)
    double d_norm_sq = 0;
    for (int layer : layers) {
        for (int h = 0; h < n_head; ++h) {
            auto da = sa.s_head(layer, h);
            auto db = sb.s_head(layer, h);
            for (int i = 0; i < hs * hs; ++i) {
                double d = (double)db[i] - (double)da[i];
                d_norm_sq += d * d;
            }
        }
    }

    double alpha = ratio;
    if (d_norm_sq > 1e-20) alpha *= std::sqrt(s_norm_sq) / std::sqrt(d_norm_sq);

    apply_online_r1(dst, online, alpha);
}

// R1-WT: weight by σ₁/Σσ of each head's own delta.
static void apply_r1wt(rp::StateBuf & dst,
                       const rp::StateBuf & sa, const rp::StateBuf & sb,
                       const std::vector<int> & layers, int n_head, int hs,
                       double alpha) {
    for (int layer : layers) {
        for (int h = 0; h < n_head; ++h) {
            auto da = sa.s_head(layer, h);
            auto db = sb.s_head(layer, h);
            auto dd = dst.s_head(layer, h);
            std::vector<double> delta(hs * hs);
            for (int r = 0; r < hs; ++r) {
                for (int c = 0; c < hs; ++c) {
                    delta[r * hs + c] = (double)db[r * hs + c] - (double)da[r * hs + c];
                }
            }
            auto svdr = rp::numeric::svd(delta.data(), hs);
            double s1 = svdr.sigma[0];
            double ssum = 0.0;
            for (int si = 0; si < (int)svdr.sigma.size(); ++si) ssum += svdr.sigma[si];
            if (ssum < 1e-10) continue;
            double w = (s1 / ssum) * alpha;
            for (int idx = 0; idx < hs * hs; ++idx) {
                dd[idx] += (float)(w * ((double)db[idx] - (double)da[idx]));
            }
        }
    }
}

// ── per-edit result ──────────────────────────────────────────────────────────
struct EditResult {
    float p_target;          // P(old) for target entity
    float max_control_drop;  // max P-drop among non-target entities (1.0 - min_control_P)
    bool  flipped;           // p_target < 0.5
    bool  leaked;            // any control P < 0.5
};

int main(int argc, char ** argv) {
    auto args = rp::parse_common_args(argc, argv);
    args.defaults(/*ctx=*/2048, /*predict=*/15);
    if (args.layer_range.empty()) args.layer_range = "16-31";

    // parse experiment-specific args from extras
    std::string method_filter;
    double alpha_fixed = -1.0;
    {
        std::vector<std::string> remaining;
        for (std::size_t i = 0; i < args.extra.size(); ++i) {
            const auto & a = args.extra[i];
            if (a == "--alpha" && i+1 < args.extra.size()) {
                alpha_fixed = std::atof(args.extra[++i].c_str());
            } else if (a == "--filter" && i+1 < args.extra.size()) {
                method_filter = args.extra[++i];
            } else {
                remaining.push_back(a);
            }
        }
        args.extra = std::move(remaining);
    }

    if (args.help_requested) {
        std::printf("usage: %s -m MODEL [--layers 16-31] [--alpha 1.0] [--filter AD]\n", argv[0]);
        return 0;
    }
    if (!rp::require_model(args)) return 1;
    if (!rp::reject_extra(args)) return 1;

    std::vector<double> alphas;
    if (alpha_fixed >= 0.0) {
        alphas = {alpha_fixed};
    } else {
        alphas = {0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20, 0.75, 1.0, 1.5};
    }

    return rp::run_experiment([&] {
        auto env = rp::make_env(args);
        auto & model = env->model;
        auto & ctx   = env->ctx;
        const auto & geom  = env->geom;
        const auto * vocab = env->vocab;
        int n_head = geom.n_head;
        int hs = geom.head_size;

        auto layers = rp::parse_layer_range(args.layer_range, geom.n_layer);

        struct Entity {
            std::string name, city, new_city;
            llama_token tok_city, tok_new;
        };
        Entity entities[] = {
            {"Alice", "Paris",  "Madrid", rp::find_token(vocab, "Paris"),  rp::find_token(vocab, "Madrid")},
            {"Bob",   "Tokyo",  "Seoul",  rp::find_token(vocab, "Tokyo"),  rp::find_token(vocab, "Seoul")},
            {"Carol", "London", "Sydney", rp::find_token(vocab, "London"), rp::find_token(vocab, "Sydney")},
            {"Dave",  "Rome",   "Vienna", rp::find_token(vocab, "Rome"),   rp::find_token(vocab, "Vienna")},
            {"Eve",   "Berlin", "Oslo",   rp::find_token(vocab, "Berlin"), rp::find_token(vocab, "Oslo")},
        };
        int n_ent = 5;

        std::string context;
        for (int i = 0; i < n_ent; ++i) {
            if (i > 0) context += " ";
            context += entities[i].name + " lives in " + entities[i].city + ".";
        }
        auto state_base = rp::capture(ctx, geom, vocab, context);

        std::printf("CONTEXT: %s\n\n", context.c_str());

        // baselines
        std::printf("BASELINES:\n");
        for (int i = 0; i < n_ent; ++i) {
            std::string q = " Where does " + entities[i].name + " live? " +
                           entities[i].name + " lives in";
            auto pr = rp::probe(model, ctx, state_base, context, q,
                                entities[i].tok_city, entities[i].tok_new,
                                vocab, 0, args.seed);
            std::printf("  %s: P(%s)=%.4f\n",
                        entities[i].name.c_str(), entities[i].city.c_str(), pr.p_binary);
        }

        // offline calibration
        std::fprintf(stderr, "computing offline calibrations...\n");
        struct OfflineCal {
            rp::StateBuf sa;
            rp::StateBuf sb;
            std::vector<rp::HeadDecomp> decomps;
            OfflineCal(const rp::ModelGeometry & g) : sa(g), sb(g) {}
        };
        std::vector<OfflineCal> cals;
        for (int ei = 0; ei < n_ent; ++ei) {
            cals.emplace_back(geom);
            auto & cal = cals.back();
            std::string iso_a = entities[ei].name + " lives in " + entities[ei].city + ".";
            std::string iso_b = entities[ei].name + " lives in " + entities[ei].new_city + ".";
            cal.sa = rp::capture(ctx, geom, vocab, iso_a);
            cal.sb = rp::capture(ctx, geom, vocab, iso_b);
            cal.decomps = rp::decompose(cal.sa, cal.sb, layers);
        }

        // method names
        const char * method_names[] = {"UNIFORM", "ISOLATE", "FULL-R1", "HYB-UV", "HYB-U", "ONLINE", "ONL-ADP", "ONL-AD2"};
        int n_methods = 8;

        // results[method][entity][alpha_idx] = EditResult
        std::vector<std::vector<std::vector<EditResult>>> results(
            n_methods, std::vector<std::vector<EditResult>>(
                n_ent, std::vector<EditResult>(alphas.size())));

        // ── run all edits ────────────────────────────────────────────────
        for (int ei = 0; ei < n_ent; ++ei) {
            const auto & ent = entities[ei];

            // full-prompt pair
            std::string full_a = context;
            std::string full_b;
            for (int i = 0; i < n_ent; ++i) {
                if (i > 0) full_b += " ";
                full_b += entities[i].name + " lives in " +
                          (i == ei ? ent.new_city : entities[i].city) + ".";
            }
            auto state_full_a = state_base;
            auto state_full_b = rp::capture(ctx, geom, vocab, full_b);

            // online probe
            std::string probe_prefix;
            for (int i = 0; i <= ei; ++i) {
                if (i > 0) probe_prefix += " ";
                probe_prefix += entities[i].name + " lives in";
                if (i < ei) probe_prefix += " " + entities[i].city + ".";
                else probe_prefix += " ";
            }
            std::fprintf(stderr, "  probing %s...\n", ent.name.c_str());
            // capture online probe states (needed for ONL-ADP2 full delta norm)
            auto state_probe_prefix = rp::capture(ctx, geom, vocab, probe_prefix);
            rp::StateBuf s_probe_a(geom);
            state_probe_prefix.store_to(ctx.raw());
            ctx.decode_one(ent.tok_city);
            s_probe_a.load_from(ctx.raw());
            rp::StateBuf s_probe_b(geom);
            state_probe_prefix.store_to(ctx.raw());
            ctx.decode_one(ent.tok_new);
            s_probe_b.load_from(ctx.raw());
            auto online_decomps = rp::decompose(s_probe_a, s_probe_b, layers);

            for (int ai = 0; ai < (int)alphas.size(); ++ai) {
                double a = alphas[ai];
                std::fprintf(stderr, "    %s a=%.2f ...\n", ent.name.c_str(), a);

                // build edited states for each method
                struct MethodState { rp::StateBuf state; };
                std::vector<MethodState> mstates;

                // 0: UNIFORM
                { rp::StateBuf e(state_base);
                  rp::apply_full_delta(e, state_full_a, state_full_b, layers, a);
                  mstates.push_back({std::move(e)}); }

                // 1: ISOLATE
                { rp::StateBuf e(state_base);
                  apply_r1wt(e, cals[ei].sa, cals[ei].sb, layers, n_head, hs, a);
                  mstates.push_back({std::move(e)}); }

                // 2: FULL-R1
                { rp::StateBuf e(state_base);
                  apply_r1wt(e, state_full_a, state_full_b, layers, n_head, hs, a);
                  mstates.push_back({std::move(e)}); }

                // 3: HYB-UV
                { rp::StateBuf e(state_base);
                  apply_hybrid_uv(e, cals[ei].decomps, online_decomps, a, false);
                  mstates.push_back({std::move(e)}); }

                // 4: HYB-U
                { rp::StateBuf e(state_base);
                  apply_hybrid_uv(e, cals[ei].decomps, online_decomps, a, true);
                  mstates.push_back({std::move(e)}); }

                // 5: ONLINE (full rank-1 from online probe, no offline)
                { rp::StateBuf e(state_base);
                  apply_online_r1(e, online_decomps, a);
                  mstates.push_back({std::move(e)}); }

                // 6: ONL-ADP (ONLINE + adaptive, edit_norm = rank-1 sigma sum)
                { rp::StateBuf e(state_base);
                  apply_online_adaptive(e, online_decomps, a);
                  mstates.push_back({std::move(e)}); }

                // 7: ONL-AD2 (ONLINE + adaptive, edit_norm = full delta Frobenius)
                { rp::StateBuf e(state_base);
                  apply_online_adaptive2(e, s_probe_a, s_probe_b, online_decomps, layers, a);
                  mstates.push_back({std::move(e)}); }

                // evaluate each method
                for (int mi = 0; mi < n_methods; ++mi) {
                    // skip methods that don't match filter
                    if (!method_filter.empty() &&
                        std::string(method_names[mi]).find(method_filter) == std::string::npos) {
                        continue;
                    }

                    float p_target = 1.0f;
                    float min_control = 1.0f;
                    bool leaked = false;

                    for (int qi = 0; qi < n_ent; ++qi) {
                        std::string q = " Where does " + entities[qi].name + " live? " +
                                       entities[qi].name + " lives in";
                        auto pr = rp::probe(model, ctx, mstates[mi].state, context, q,
                                            entities[qi].tok_city, ent.tok_new,
                                            vocab, 0, args.seed);
                        if (qi == ei) {
                            p_target = pr.p_binary;
                        } else {
                            if (pr.p_binary < min_control) min_control = pr.p_binary;
                            if (pr.p_binary < 0.5f) leaked = true;
                        }
                    }

                    results[mi][ei][ai] = {
                        p_target,
                        1.0f - min_control,
                        p_target < 0.5f,
                        leaked,
                    };
                }
            }
        }

        // ── per-entity detail table ──────────────────────────────────────
        std::printf("\n══════════════════════════════════════════════════════════\n");
        std::printf("PER-ENTITY RESULTS\n");
        std::printf("══════════════════════════════════════════════════════════\n");

        for (int ei = 0; ei < n_ent; ++ei) {
            std::printf("\n%s (%s->%s):\n", entities[ei].name.c_str(),
                        entities[ei].city.c_str(), entities[ei].new_city.c_str());
            std::printf("  %-9s", "method");
            for (double a : alphas) std::printf("  a=%-5.2f", a);
            std::printf("  best_a\n");

            for (int mi = 0; mi < n_methods; ++mi) {
                std::printf("  %-9s", method_names[mi]);
                double best_a = -1;
                for (int ai = 0; ai < (int)alphas.size(); ++ai) {
                    const auto & r = results[mi][ei][ai];
                    char mark = r.flipped ? (r.leaked ? '!' : '*') : ' ';
                    std::printf("  %5.3f%c ", r.p_target, mark);
                    if (r.flipped && !r.leaked && best_a < 0) best_a = alphas[ai];
                }
                if (best_a > 0) std::printf("  %.2f", best_a);
                else std::printf("  -");
                std::printf("\n");
            }
        }

        // ── aggregate summary ────────────────────────────────────────────
        std::printf("\n══════════════════════════════════════════════════════════\n");
        std::printf("AGGREGATE METRICS (across %d entities)\n", n_ent);
        std::printf("══════════════════════════════════════════════════════════\n\n");
        std::printf("  %-9s  %6s  %8s  %11s  %8s\n",
                    "method", "flip", "best_a", "selectivity", "P(tgt)");
        std::printf("  %s\n", std::string(52, '-').c_str());

        for (int mi = 0; mi < n_methods; ++mi) {
            int flipped_count = 0;
            double sum_best_a = 0;
            double sum_selectivity = 0;
            double sum_p_target = 0;
            int clean_count = 0;

            for (int ei = 0; ei < n_ent; ++ei) {
                // find best alpha: lowest where flipped && !leaked
                double best_a = -1;
                int best_ai = -1;
                for (int ai = 0; ai < (int)alphas.size(); ++ai) {
                    const auto & r = results[mi][ei][ai];
                    if (r.flipped && !r.leaked) {
                        best_a = alphas[ai];
                        best_ai = ai;
                        break;
                    }
                }
                if (best_ai >= 0) {
                    flipped_count++;
                    sum_best_a += best_a;
                    sum_selectivity += (1.0 - results[mi][ei][best_ai].max_control_drop);
                    sum_p_target += results[mi][ei][best_ai].p_target;
                    clean_count++;
                }
            }

            std::printf("  %-9s  %d/%d    ", method_names[mi], flipped_count, n_ent);
            if (clean_count > 0) {
                std::printf("%8.2f  %11.4f  %8.4f",
                            sum_best_a / clean_count,
                            sum_selectivity / clean_count,
                            sum_p_target / clean_count);
            } else {
                std::printf("%8s  %11s  %8s", "-", "-", "-");
            }
            std::printf("\n");
        }

        // ── per-alpha flip rate ──────────────────────────────────────────
        std::printf("\n  Flip rate by alpha (flipped && !leaked):\n");
        std::printf("  %-9s", "method");
        for (double a : alphas) std::printf("  a=%-5.2f", a);
        std::printf("\n");
        for (int mi = 0; mi < n_methods; ++mi) {
            std::printf("  %-9s", method_names[mi]);
            for (int ai = 0; ai < (int)alphas.size(); ++ai) {
                int ok = 0;
                for (int ei = 0; ei < n_ent; ++ei) {
                    const auto & r = results[mi][ei][ai];
                    if (r.flipped && !r.leaked) ok++;
                }
                std::printf("  %d/%-5d ", ok, n_ent);
            }
            std::printf("\n");
        }

        std::printf("\n");
        std::fprintf(stderr, "done\n");
    });
}
