// experiments/calibrated-multi.cpp — CALIB/ISOLATE methods on multi-entity
// and ambiguous-entity scenarios.
//
// Part 1: 5 entities (Alice/Bob/Carol/Dave/Eve with cities), edit each one.
// Part 2: Ambiguous entities (Alice Anderson / Alice White with hat colors).
//
// Methods compared per edit:
//   UNIFORM  — full-prompt delta, uniform alpha
//   CALIB    — full-prompt delta, weights from single-token probe at shared prefix
//   ISOLATE  — isolated sentence delta, R1-WT self-calibrated
//   ISO-FREE — freeform concept fragment delta, R1-WT self-calibrated
//
// Usage:
//   llama-rwkv-calibrated-multi -m model.gguf [--layers 16-31] [-n 15]
//       [--filter five|ambig] [--alpha 0.5]

#include "rwkv_probe/state.h"
#include "rwkv_probe/generate.h"
#include "rwkv_probe/experiment.h"
#include "rwkv_probe/pipeline.h"
#include "rwkv_probe/edit.h"
#include "rwkv_probe/head_decomp.h"

#include "rwkv_probe/numeric.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace rp = rwkv_probe;

// CALIB: compute per-head weights from single-token probe, then apply full-prompt delta
static std::vector<std::vector<double>> compute_calib_weights(
        rp::Context & ctx,
        const rp::StateBuf & state_prefix,
        llama_token tok_a, llama_token tok_b,
        const std::vector<int> & layers, int n_head, int hs,
        const rp::ModelGeometry & geom) {
    int n_layers = (int)layers.size();

    rp::StateBuf s_after_a(geom);
    state_prefix.store_to(ctx.raw());
    ctx.decode_one(tok_a);
    s_after_a.load_from(ctx.raw());

    rp::StateBuf s_after_b(geom);
    state_prefix.store_to(ctx.raw());
    ctx.decode_one(tok_b);
    s_after_b.load_from(ctx.raw());

    std::vector<std::vector<double>> weights(n_layers, std::vector<double>(n_head, 0.0));
    for (int li = 0; li < n_layers; ++li) {
        int layer = layers[li];
        double max_sigma = 0.0;
        for (int h = 0; h < n_head; ++h) {
            auto ha = s_after_a.s_head(layer, h);
            auto hb = s_after_b.s_head(layer, h);
            std::vector<double> delta(hs * hs);
            for (int r = 0; r < hs; ++r) {
                for (int c = 0; c < hs; ++c) {
                    delta[r * hs + c] = (double)hb[r * hs + c] - (double)ha[r * hs + c];
                }
            }
            auto svdr = rp::numeric::svd(delta.data(), hs);
            {
                double s1 = svdr.sigma[0];
                weights[li][h] = s1;
                if (s1 > max_sigma) max_sigma = s1;
            }
        }
        if (max_sigma > 0.0) {
            for (int h = 0; h < n_head; ++h) weights[li][h] /= max_sigma;
        }
    }
    return weights;
}

// NOTE: apply_weighted replaced by rp::apply_weighted_delta from the framework.

// R1-WT: weight by σ₁/Σσ of each head's own delta
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
            for (int i = 0; i < hs * hs; ++i) {
                dd[i] += (float)(w * ((double)db[i] - (double)da[i]));
            }
        }
    }
}

// Online probe: single-token decomposition for hybrid methods
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

// HYBRID: offline u1, online v1
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

// escape control chars and newlines for clean terminal output
static std::string escape_gen(const std::string & s, size_t max_len = 35) {
    std::string out;
    out.reserve(max_len);
    for (size_t i = 0; i < s.size() && out.size() < max_len; ++i) {
        char c = s[i];
        if (c == '\n') { out += "\\n"; }
        else if (c == '\r') { out += "\\r"; }
        else if (c == '\t') { out += "\\t"; }
        else if (c < 0x20) { out += '?'; }
        else { out += c; }
    }
    return out;
}

// ── query helper ───────────────────────────────────────────────────────────

struct QResult {
    float p;
    std::string gen;
};

static QResult do_query(rp::Model & model, rp::Context & ctx,
                        const rp::StateBuf & state,
                        const std::string & prefix, const std::string & qstr,
                        llama_token tok_a, llama_token tok_b,
                        const llama_vocab * vocab, int n_predict, uint32_t seed) {
    auto pr = rp::probe(model, ctx, state, prefix, qstr,
                        tok_a, tok_b, vocab, n_predict, seed);
    return {pr.p_binary, pr.generation};
}

int main(int argc, char ** argv) {
    auto args = rp::parse_common_args(argc, argv);
    args.defaults(/*ctx=*/2048, /*predict=*/15);
    if (args.layer_range.empty()) args.layer_range = "16-31";

    // parse experiment-specific args from extras
    std::string filter;
    double alpha = -1.0;
    {
        std::vector<std::string> remaining;
        for (std::size_t i = 0; i < args.extra.size(); ++i) {
            const auto & a = args.extra[i];
            if (a == "--filter" && i+1 < args.extra.size()) {
                filter = args.extra[++i];
            } else if (a == "--alpha" && i+1 < args.extra.size()) {
                alpha = std::atof(args.extra[++i].c_str());
            } else {
                remaining.push_back(a);
            }
        }
        args.extra = std::move(remaining);
    }

    if (args.help_requested) {
        std::printf("usage: %s -m MODEL [--layers 16-31] [-n 15] [--filter five|ambig] [--alpha 0.5]\n", argv[0]);
        return 0;
    }
    if (!rp::require_model(args)) return 1;
    if (!rp::reject_extra(args)) return 1;

    std::vector<double> alphas;
    if (alpha >= 0.0) {
        alphas = {alpha};
    } else {
        alphas = {0.5, 0.75, 1.0, 1.5};
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

        std::printf("layers: %d-%d (%zu)  n_head=%d  head_size=%d\n\n",
                    layers.front(), layers.back(), layers.size(), n_head, hs);

        // ════════════════════════════════════════════════════════════════
        // PART 1: Five entities with cities
        // ════════════════════════════════════════════════════════════════
        if (filter.empty() || filter.find("five") != std::string::npos) {
            struct Entity {
                std::string name;
                std::string city;
                std::string new_city;
                llama_token tok_city;
                llama_token tok_new;
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

            std::printf("======================================================================\n");
            std::printf("FIVE ENTITIES\n");
            std::printf("  CONTEXT: \"%s\"\n\n", context.c_str());

            auto state_base = rp::capture(ctx, geom, vocab,context);

            // baselines
            std::printf("  BASELINES:\n");
            for (int i = 0; i < n_ent; ++i) {
                std::string q = " Where does " + entities[i].name + " live? " +
                               entities[i].name + " lives in";
                auto r = do_query(model, ctx, state_base, context, q,
                                  entities[i].tok_city, entities[i].tok_new,
                                  vocab, args.n_predict, args.seed);
                std::printf("    %s: P(%s)=%.4f  %s\n",
                            entities[i].name.c_str(), entities[i].city.c_str(),
                            r.p, escape_gen(r.gen, 40).c_str());
            }

            // method names for summary
            const char * method_names[] = {
                "UNIFORM", "CALIB", "ISOLATE", "ISO-FREE", "FULL-R1", "HYB-UV", "HYB-U", "ONLINE"
            };
            int n_methods = 8;

            // results[method][entity][alpha_idx]
            struct EditRes {
                float p_target;        // P(old) for target
                float min_control;     // lowest P(old) among non-targets
                bool  flipped;         // p_target < 0.5
                bool  leaked;          // any control < 0.5
            };
            std::vector<std::vector<std::vector<EditRes>>> results(
                n_methods, std::vector<std::vector<EditRes>>(
                    n_ent, std::vector<EditRes>(alphas.size(), {1.0f, 1.0f, false, false})));

            // edit each entity with each method
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

                auto state_full_a = state_base; // same as base
                auto state_full_b = rp::capture(ctx, geom, vocab,full_b);

                // CALIB: shared prefix up to the edited entity's city
                std::string cal_prefix;
                for (int i = 0; i <= ei; ++i) {
                    if (i > 0) cal_prefix += " ";
                    cal_prefix += entities[i].name + " lives in";
                    if (i < ei) cal_prefix += " " + entities[i].city + ".";
                    else cal_prefix += " "; // stop before the city token
                }
                auto state_cal_prefix = rp::capture(ctx, geom, vocab,cal_prefix);
                auto calib_weights = compute_calib_weights(ctx, state_cal_prefix,
                    ent.tok_city, ent.tok_new, layers, n_head, hs, geom);

                // ISOLATE: just the entity's sentence
                std::string iso_a = ent.name + " lives in " + ent.city + ".";
                std::string iso_b = ent.name + " lives in " + ent.new_city + ".";
                auto state_iso_a = rp::capture(ctx, geom, vocab,iso_a);
                auto state_iso_b = rp::capture(ctx, geom, vocab,iso_b);

                // ISO-FREE: just "lives in <city>"
                std::string free_a = ent.name + " lives in " + ent.city;
                std::string free_b = ent.name + " lives in " + ent.new_city;
                auto state_free_a = rp::capture(ctx, geom, vocab,free_a);
                auto state_free_b = rp::capture(ctx, geom, vocab,free_b);

                // HYBRID: offline decomps from isolated + online probe at entity position
                auto iso_decomps = rp::decompose(state_iso_a, state_iso_b, layers);
                std::string probe_prefix;
                for (int i = 0; i <= ei; ++i) {
                    if (i > 0) probe_prefix += " ";
                    probe_prefix += entities[i].name + " lives in";
                    if (i < ei) probe_prefix += " " + entities[i].city + ".";
                    else probe_prefix += " ";
                }
                auto online_decomps = online_probe_decomps(ctx, geom, vocab, probe_prefix,
                    ent.tok_city, ent.tok_new, layers);

                for (int ai = 0; ai < (int)alphas.size(); ++ai) {
                    double a = alphas[ai];
                    std::printf("\n  EDIT: %s %s->%s  alpha=%.2f\n",
                                ent.name.c_str(), ent.city.c_str(), ent.new_city.c_str(), a);
                    std::printf("  %-8s  |  %-8s  %8s  %8s  %s\n",
                                "method", "entity", "P(old)", "status", "gen");
                    std::printf("  %s\n", std::string(8 + 3 + 8 + 2 + 8 + 2 + 8 + 2 + 40, '-').c_str());

                    struct Method {
                        const char * name;
                        rp::StateBuf edited;
                    };

                    // build edited states
                    rp::StateBuf e_uniform(state_base);
                    rp::apply_full_delta(e_uniform, state_full_a, state_full_b, layers, a);

                    rp::StateBuf e_calib(state_base);
                    rp::apply_weighted_delta(e_calib, state_full_a, state_full_b, layers, calib_weights, a);

                    rp::StateBuf e_isolate(state_base);
                    apply_r1wt(e_isolate, state_iso_a, state_iso_b, layers, n_head, hs, a);

                    rp::StateBuf e_isofree(state_base);
                    apply_r1wt(e_isofree, state_free_a, state_free_b, layers, n_head, hs, a);

                    // FULL-R1WT: full-prompt delta with R1-WT self-calibration
                    rp::StateBuf e_fullr1(state_base);
                    apply_r1wt(e_fullr1, state_full_a, state_full_b, layers, n_head, hs, a);

                    // HYB-UV: offline u1 + online v1, offline sigma
                    rp::StateBuf e_hybuv(state_base);
                    apply_hybrid_uv(e_hybuv, iso_decomps, online_decomps, a, false);

                    // HYB-U: offline u1 + online v1, online sigma
                    rp::StateBuf e_hybu(state_base);
                    apply_hybrid_uv(e_hybu, iso_decomps, online_decomps, a, true);

                    // ONLINE: full rank-1 from online probe, no offline
                    rp::StateBuf e_online(state_base);
                    apply_online_r1(e_online, online_decomps, a);

                    Method methods[] = {
                        {"UNIFORM",  std::move(e_uniform)},
                        {"CALIB",    std::move(e_calib)},
                        {"ISOLATE",  std::move(e_isolate)},
                        {"ISO-FREE", std::move(e_isofree)},
                        {"FULL-R1",  std::move(e_fullr1)},
                        {"HYB-UV",   std::move(e_hybuv)},
                        {"HYB-U",    std::move(e_hybu)},
                        {"ONLINE",   std::move(e_online)},
                    };

                    for (int mi = 0; mi < n_methods; ++mi) {
                        auto & m = methods[mi];
                        float p_tgt = 1.0f;
                        float min_ctrl = 1.0f;
                        bool any_leak = false;

                        for (int qi = 0; qi < n_ent; ++qi) {
                            std::string q = " Where does " + entities[qi].name + " live? " +
                                           entities[qi].name + " lives in";
                            auto r = do_query(model, ctx, m.edited, context, q,
                                              entities[qi].tok_city, ent.tok_new,
                                              vocab, args.n_predict, args.seed);

                            bool is_target = (qi == ei);
                            bool flipped = (r.p < 0.5);
                            const char * status = is_target ?
                                (flipped ? "YES" : "FAIL") :
                                (flipped ? "LEAK!" : "ok");

                            std::printf("  %-8s  |  %-8s  %8.4f  %8s  %s%s\n",
                                        m.name, entities[qi].name.c_str(),
                                        r.p, status,
                                        escape_gen(r.gen).c_str(),
                                        is_target ? "  <-TGT" : "");

                            if (is_target) {
                                p_tgt = r.p;
                            } else {
                                if (r.p < min_ctrl) min_ctrl = r.p;
                                if (r.p < 0.5f) any_leak = true;
                            }
                        }
                        results[mi][ei][ai] = {p_tgt, min_ctrl, p_tgt < 0.5f, any_leak};
                    }

                    std::fflush(stdout);
                }
            }

            // ── SUMMARY ──────────────────────────────────────────────────
            std::printf("\n======================================================================\n");
            std::printf("SUMMARY: 5 ENTITIES\n");
            std::printf("======================================================================\n\n");

            // per-entity: P(target) at each alpha, * = clean flip, ! = leak, X = fail
            for (int ei = 0; ei < n_ent; ++ei) {
                std::printf("  %s (%s->%s):\n", entities[ei].name.c_str(),
                            entities[ei].city.c_str(), entities[ei].new_city.c_str());
                std::printf("  %-8s", "method");
                for (double a : alphas) std::printf("  a=%-5.2f", a);
                std::printf("  best_a\n");

                for (int mi = 0; mi < n_methods; ++mi) {
                    std::printf("  %-8s", method_names[mi]);
                    double best = -1;
                    for (int ai = 0; ai < (int)alphas.size(); ++ai) {
                        const auto & r = results[mi][ei][ai];
                        char mark = r.flipped ? (r.leaked ? '!' : '*') : ' ';
                        std::printf("  %5.3f%c ", r.p_target, mark);
                        if (r.flipped && !r.leaked && best < 0) best = alphas[ai];
                    }
                    if (best > 0) std::printf("  %.2f", best);
                    else std::printf("  -");
                    std::printf("\n");
                }
                std::printf("\n");
            }

            // aggregate
            std::printf("  %-8s  %5s  %7s  %11s  %8s\n",
                        "method", "flip", "best_a", "selectivity", "P(tgt)");
            std::printf("  %s\n", std::string(48, '-').c_str());

            for (int mi = 0; mi < n_methods; ++mi) {
                int ok = 0;
                double sum_a = 0, sum_sel = 0, sum_p = 0;
                for (int ei = 0; ei < n_ent; ++ei) {
                    for (int ai = 0; ai < (int)alphas.size(); ++ai) {
                        const auto & r = results[mi][ei][ai];
                        if (r.flipped && !r.leaked) {
                            ok++;
                            sum_a += alphas[ai];
                            sum_sel += r.min_control;
                            sum_p += r.p_target;
                            break;
                        }
                    }
                }
                std::printf("  %-8s  %d/%d  ", method_names[mi], ok, n_ent);
                if (ok > 0) {
                    std::printf("  %5.2f  %11.4f  %8.4f",
                                sum_a / ok, sum_sel / ok, sum_p / ok);
                } else {
                    std::printf("  %5s  %11s  %8s", "-", "-", "-");
                }
                std::printf("\n");
            }

            // flip rate by alpha
            std::printf("\n  Flip rate by alpha (flipped && !leaked):\n");
            std::printf("  %-8s", "method");
            for (double a : alphas) std::printf("  a=%-5.2f", a);
            std::printf("\n");
            for (int mi = 0; mi < n_methods; ++mi) {
                std::printf("  %-8s", method_names[mi]);
                for (int ai = 0; ai < (int)alphas.size(); ++ai) {
                    int c = 0;
                    for (int ei = 0; ei < n_ent; ++ei) {
                        const auto & r = results[mi][ei][ai];
                        if (r.flipped && !r.leaked) c++;
                    }
                    std::printf("  %d/%-5d ", c, n_ent);
                }
                std::printf("\n");
            }
            std::printf("\n");
        }

        // ════════════════════════════════════════════════════════════════
        // PART 2: Ambiguous entities — Alice Anderson / Alice White
        // ════════════════════════════════════════════════════════════════
        if (filter.empty() || filter.find("ambig") != std::string::npos) {
            llama_token tok_red   = rp::find_token(vocab, "red");
            llama_token tok_green = rp::find_token(vocab, "green");
            llama_token tok_blue  = rp::find_token(vocab, "blue");

            std::string context = "Alice Anderson has a red hat. Alice White has a blue hat.";
            auto state_base = rp::capture(ctx, geom, vocab,context);

            std::string q_anderson = " What color is Alice Anderson's hat? Alice Anderson's hat is";
            std::string q_white    = " What color is Alice White's hat? Alice White's hat is";

            std::printf("\n======================================================================\n");
            std::printf("AMBIGUOUS ENTITIES\n");
            std::printf("  CONTEXT: \"%s\"\n\n", context.c_str());

            auto bl_and = do_query(model, ctx, state_base, context, q_anderson,
                                   tok_red, tok_green, vocab, args.n_predict, args.seed);
            auto bl_wht = do_query(model, ctx, state_base, context, q_white,
                                   tok_blue, tok_green, vocab, args.n_predict, args.seed);
            std::printf("  BASELINES:\n");
            std::printf("    Anderson: P(red)=%.4f   %s\n", bl_and.p, escape_gen(bl_and.gen, 40).c_str());
            std::printf("    White:    P(blue)=%.4f  %s\n\n", bl_wht.p, escape_gen(bl_wht.gen, 40).c_str());

            // edit targets: try editing Anderson's hat red→green
            struct AmbigEdit {
                const char * name;
                // full prompt pair
                std::string full_a;
                std::string full_b;
                // calib prefix and tokens
                std::string cal_prefix;
                llama_token cal_tok_a;
                llama_token cal_tok_b;
                // isolate sentence pair
                std::string iso_a;
                std::string iso_b;
                // freeform
                std::string free_a;
                std::string free_b;
                // which query is target, which is control
                llama_token tok_tgt_old; // what the target should flip from
                llama_token tok_tgt_new; // what the target should flip to
            };

            AmbigEdit edits[] = {
                {
                    "Anderson red→green (specific cal)",
                    // full pair
                    "Alice Anderson has a red hat. Alice White has a blue hat.",
                    "Alice Anderson has a green hat. Alice White has a blue hat.",
                    // calib
                    "Alice Anderson has a ",
                    tok_red, tok_green,
                    // isolate
                    "Alice Anderson has a red hat.",
                    "Alice Anderson has a green hat.",
                    // freeform — specific name, no period, no other entity
                    "Alice Anderson has a red hat",
                    "Alice Anderson has a green hat",
                    tok_red, tok_green,
                },
                {
                    "White blue→green (specific cal)",
                    // full pair
                    "Alice Anderson has a red hat. Alice White has a blue hat.",
                    "Alice Anderson has a red hat. Alice White has a green hat.",
                    // calib
                    "Alice Anderson has a red hat. Alice White has a ",
                    tok_blue, tok_green,
                    // isolate
                    "Alice White has a blue hat.",
                    "Alice White has a green hat.",
                    // freeform — specific name, no period, no other entity
                    "Alice White has a blue hat",
                    "Alice White has a green hat",
                    tok_blue, tok_green,
                },
            };

            for (const auto & ed : edits) {
                auto state_full_a = rp::capture(ctx, geom, vocab,ed.full_a);
                auto state_full_b = rp::capture(ctx, geom, vocab,ed.full_b);

                auto state_cal_prefix = rp::capture(ctx, geom, vocab,ed.cal_prefix);
                auto calib_weights = compute_calib_weights(ctx, state_cal_prefix,
                    ed.cal_tok_a, ed.cal_tok_b, layers, n_head, hs, geom);

                auto state_iso_a = rp::capture(ctx, geom, vocab,ed.iso_a);
                auto state_iso_b = rp::capture(ctx, geom, vocab,ed.iso_b);

                auto state_free_a = rp::capture(ctx, geom, vocab,ed.free_a);
                auto state_free_b = rp::capture(ctx, geom, vocab,ed.free_b);

                // HYBRID: offline decomps from isolated + online probe at cal_prefix
                auto iso_decomps = rp::decompose(state_iso_a, state_iso_b, layers);
                auto online_decomps = online_probe_decomps(ctx, geom, vocab, ed.cal_prefix,
                    ed.cal_tok_a, ed.cal_tok_b, layers);

                for (double a : alphas) {
                    std::printf("  EDIT: %s  alpha=%.2f\n", ed.name, a);
                    std::printf("  %-8s  |  %-10s  %8s  %8s  %s\n",
                                "method", "entity", "P(old)", "status", "gen");
                    std::printf("  %s\n", std::string(8 + 3 + 10 + 2 + 8 + 2 + 8 + 2 + 40, '-').c_str());

                    rp::StateBuf e_uniform(state_base);
                    rp::apply_full_delta(e_uniform, state_full_a, state_full_b, layers, a);

                    rp::StateBuf e_calib(state_base);
                    rp::apply_weighted_delta(e_calib, state_full_a, state_full_b, layers, calib_weights, a);

                    rp::StateBuf e_isolate(state_base);
                    apply_r1wt(e_isolate, state_iso_a, state_iso_b, layers, n_head, hs, a);

                    rp::StateBuf e_isofree(state_base);
                    apply_r1wt(e_isofree, state_free_a, state_free_b, layers, n_head, hs, a);

                    rp::StateBuf e_fullr1(state_base);
                    apply_r1wt(e_fullr1, state_full_a, state_full_b, layers, n_head, hs, a);

                    rp::StateBuf e_hybuv(state_base);
                    apply_hybrid_uv(e_hybuv, iso_decomps, online_decomps, a, false);

                    rp::StateBuf e_hybu(state_base);
                    apply_hybrid_uv(e_hybu, iso_decomps, online_decomps, a, true);

                    rp::StateBuf e_online(state_base);
                    apply_online_r1(e_online, online_decomps, a);

                    struct Method { const char * name; rp::StateBuf * edited; };
                    Method methods[] = {
                        {"UNIFORM",  &e_uniform},
                        {"CALIB",    &e_calib},
                        {"ISOLATE",  &e_isolate},
                        {"ISO-FREE", &e_isofree},
                        {"FULL-R1",  &e_fullr1},
                        {"HYB-UV",   &e_hybuv},
                        {"HYB-U",    &e_hybu},
                        {"ONLINE",   &e_online},
                    };

                    // determine which edit this is
                    bool editing_anderson = (ed.cal_tok_a == tok_red);

                    for (auto & m : methods) {
                        // query Anderson
                        auto r_and = do_query(model, ctx, *m.edited, context, q_anderson,
                                              tok_red, tok_green, vocab, args.n_predict, args.seed);
                        bool and_flipped = (r_and.p < 0.5);
                        const char * and_status = editing_anderson ?
                            (and_flipped ? "YES" : "FAIL") :
                            (and_flipped ? "LEAK!" : "ok");

                        std::printf("  %-8s  |  %-10s  %8.4f  %8s  %s%s\n",
                                    m.name, "Anderson", r_and.p, and_status,
                                    escape_gen(r_and.gen).c_str(),
                                    editing_anderson ? "  <-TGT" : "");

                        // query White
                        auto r_wht = do_query(model, ctx, *m.edited, context, q_white,
                                              tok_blue, tok_green, vocab, args.n_predict, args.seed);
                        bool wht_flipped = (r_wht.p < 0.5);
                        const char * wht_status = !editing_anderson ?
                            (wht_flipped ? "YES" : "FAIL") :
                            (wht_flipped ? "LEAK!" : "ok");

                        std::printf("  %-8s  |  %-10s  %8.4f  %8s  %s%s\n",
                                    m.name, "White", r_wht.p, wht_status,
                                    escape_gen(r_wht.gen).c_str(),
                                    !editing_anderson ? "  <-TGT" : "");
                    }

                    std::printf("\n");
                    std::fflush(stdout);
                }
            }
        }

        std::fprintf(stderr, "done\n");
    });
}
