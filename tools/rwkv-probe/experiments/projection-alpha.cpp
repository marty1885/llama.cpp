// experiments/projection-alpha.cpp — test projection-based adaptive alpha.
//
// Instead of a global norm ratio, compute per-head projection coefficients:
//   proj_h = dot(state_h, delta_h) / dot(delta_h, delta_h)
// This tells us how much of the edit delta is present in each head's state.
// Use |proj_h| as the per-head alpha — heads where the fact is strongly
// encoded get a strong edit, heads where it's absent get none.
//
// Also tests global projection (single scalar over entire state) and
// norm-ratio (current approach) for comparison.
//
// Usage:
//   llama-rwkv-projection-alpha -m model.gguf [--layers 16-31] [-n 15]

#include "rwkv_probe/state.h"
#include "rwkv_probe/experiment.h"
#include "rwkv_probe/generate.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace rp = rwkv_probe;

// ── edit methods ──────────────────────────────────────────────────────

// norm-ratio adaptive (current approach)
static void apply_norm_ratio(rp::StateBuf & dst,
                             const rp::StateBuf & sa, const rp::StateBuf & sb,
                             const std::vector<int> & layers, int n_head, int hs,
                             double ratio) {
    int n = hs * hs;
    // compute global norms
    double s_norm_sq = 0.0, d_norm_sq = 0.0;
    for (int layer : layers) {
        for (int h = 0; h < n_head; ++h) {
            auto dd = dst.s_head(layer, h);
            auto da = sa.s_head(layer, h);
            auto db = sb.s_head(layer, h);
            for (int i = 0; i < n; ++i) {
                s_norm_sq += (double)dd[i] * (double)dd[i];
                double d = (double)db[i] - (double)da[i];
                d_norm_sq += d * d;
            }
        }
    }
    double s_norm = std::sqrt(s_norm_sq);
    double d_norm = std::sqrt(d_norm_sq);
    double alpha = (d_norm > 1e-10) ? ratio * s_norm / d_norm : ratio;

    for (int layer : layers) {
        for (int h = 0; h < n_head; ++h) {
            auto da = sa.s_head(layer, h);
            auto db = sb.s_head(layer, h);
            auto dd = dst.s_head(layer, h);
            for (int i = 0; i < n; ++i) {
                dd[i] += (float)(alpha * ((double)db[i] - (double)da[i]));
            }
        }
    }
}

// global projection: single proj_coeff over entire state
static void apply_global_projection(rp::StateBuf & dst,
                                    const rp::StateBuf & sa, const rp::StateBuf & sb,
                                    const std::vector<int> & layers, int n_head, int hs,
                                    double strength) {
    int n = hs * hs;
    double dot_sd = 0.0, dot_dd = 0.0;
    for (int layer : layers) {
        for (int h = 0; h < n_head; ++h) {
            auto dd = dst.s_head(layer, h);
            auto da = sa.s_head(layer, h);
            auto db = sb.s_head(layer, h);
            for (int i = 0; i < n; ++i) {
                double d = (double)db[i] - (double)da[i];
                dot_sd += (double)dd[i] * d;
                dot_dd += d * d;
            }
        }
    }
    double proj_coeff = (dot_dd > 1e-20) ? dot_sd / dot_dd : 0.0;
    double alpha = strength * std::abs(proj_coeff);

    for (int layer : layers) {
        for (int h = 0; h < n_head; ++h) {
            auto da = sa.s_head(layer, h);
            auto db = sb.s_head(layer, h);
            auto dd = dst.s_head(layer, h);
            for (int i = 0; i < n; ++i) {
                dd[i] += (float)(alpha * ((double)db[i] - (double)da[i]));
            }
        }
    }
}

// per-head projection: each head gets its own proj_coeff
static void apply_perhead_projection(rp::StateBuf & dst,
                                     const rp::StateBuf & sa, const rp::StateBuf & sb,
                                     const std::vector<int> & layers, int n_head, int hs,
                                     double strength) {
    int n = hs * hs;
    for (int layer : layers) {
        for (int h = 0; h < n_head; ++h) {
            auto da = sa.s_head(layer, h);
            auto db = sb.s_head(layer, h);
            auto dd = dst.s_head(layer, h);

            double dot_sd = 0.0, dot_dd = 0.0;
            for (int i = 0; i < n; ++i) {
                double d = (double)db[i] - (double)da[i];
                dot_sd += (double)dd[i] * d;
                dot_dd += d * d;
            }
            if (dot_dd < 1e-20) continue;

            double proj_coeff = dot_sd / dot_dd;
            double alpha = strength * std::abs(proj_coeff);

            for (int i = 0; i < n; ++i) {
                dd[i] += (float)(alpha * ((double)db[i] - (double)da[i]));
            }
        }
    }
}

// per-head signed projection: use sign of proj to decide direction
// positive proj = delta aligns with state -> amplify
// negative proj = delta opposes state -> the fact is there, subtract it
static void apply_perhead_signed_projection(rp::StateBuf & dst,
                                            const rp::StateBuf & sa, const rp::StateBuf & sb,
                                            const std::vector<int> & layers, int n_head, int hs,
                                            double strength) {
    int n = hs * hs;
    for (int layer : layers) {
        for (int h = 0; h < n_head; ++h) {
            auto da = sa.s_head(layer, h);
            auto db = sb.s_head(layer, h);
            auto dd = dst.s_head(layer, h);

            double dot_sd = 0.0, dot_dd = 0.0;
            for (int i = 0; i < n; ++i) {
                double d = (double)db[i] - (double)da[i];
                dot_sd += (double)dd[i] * d;
                dot_dd += d * d;
            }
            if (dot_dd < 1e-20) continue;

            double proj_coeff = dot_sd / dot_dd;
            // use the actual projection coefficient (signed)
            double alpha = strength * proj_coeff;

            for (int i = 0; i < n; ++i) {
                dd[i] += (float)(alpha * ((double)db[i] - (double)da[i]));
            }
        }
    }
}

// NORM-RATIO + HEAD-PROJ hybrid: use norm-ratio for global alpha magnitude,
// then distribute non-uniformly across heads weighted by their projection coefficients.
// Heads where the delta aligns with the state get more edit, others get less.
static void apply_proj_weighted_norm(rp::StateBuf & dst,
                                     const rp::StateBuf & sa, const rp::StateBuf & sb,
                                     const std::vector<int> & layers, int n_head, int hs,
                                     double ratio) {
    int n = hs * hs;

    // first pass: compute global alpha via norm-ratio, and per-head projection coefficients
    double s_norm_sq = 0.0, d_norm_sq = 0.0;
    struct HeadInfo { int layer; int head; double proj; double delta_norm; };
    std::vector<HeadInfo> heads;

    for (int layer : layers) {
        for (int h = 0; h < n_head; ++h) {
            auto dd = dst.s_head(layer, h);
            auto da = sa.s_head(layer, h);
            auto db = sb.s_head(layer, h);

            double dot_sd = 0.0, dot_dd = 0.0;
            for (int i = 0; i < n; ++i) {
                double d = (double)db[i] - (double)da[i];
                s_norm_sq += (double)dd[i] * (double)dd[i];
                d_norm_sq += d * d;
                dot_sd += (double)dd[i] * d;
                dot_dd += d * d;
            }

            double proj = (dot_dd > 1e-20) ? std::abs(dot_sd / dot_dd) : 0.0;
            heads.push_back({layer, h, proj, std::sqrt(dot_dd)});
        }
    }

    double s_norm = std::sqrt(s_norm_sq);
    double d_norm = std::sqrt(d_norm_sq);
    double global_alpha = (d_norm > 1e-10) ? ratio * s_norm / d_norm : ratio;

    // compute mean projection for normalization
    double proj_sum = 0.0;
    int proj_count = 0;
    for (const auto & hi : heads) {
        if (hi.delta_norm > 1e-10) {
            proj_sum += hi.proj;
            proj_count++;
        }
    }
    double proj_mean = (proj_count > 0) ? proj_sum / proj_count : 1.0;

    // second pass: apply with per-head weighting
    for (const auto & hi : heads) {
        auto da = sa.s_head(hi.layer, hi.head);
        auto db = sb.s_head(hi.layer, hi.head);
        auto dd = dst.s_head(hi.layer, hi.head);

        if (hi.delta_norm < 1e-10) continue;

        // weight = projection / mean_projection, so mean weight = 1.0
        // heads with above-average projection get amplified, below-average get suppressed
        double weight = (proj_mean > 1e-10) ? hi.proj / proj_mean : 1.0;
        double alpha = global_alpha * weight;

        for (int i = 0; i < n; ++i) {
            dd[i] += (float)(alpha * ((double)db[i] - (double)da[i]));
        }
    }
}

// Same as above but cap the weight at a maximum to prevent outlier heads
// from getting too much edit strength
static void apply_proj_weighted_capped(rp::StateBuf & dst,
                                       const rp::StateBuf & sa, const rp::StateBuf & sb,
                                       const std::vector<int> & layers, int n_head, int hs,
                                       double ratio) {
    int n = hs * hs;

    double s_norm_sq = 0.0, d_norm_sq = 0.0;
    struct HeadInfo { int layer; int head; double proj; double delta_norm; };
    std::vector<HeadInfo> heads;

    for (int layer : layers) {
        for (int h = 0; h < n_head; ++h) {
            auto dd = dst.s_head(layer, h);
            auto da = sa.s_head(layer, h);
            auto db = sb.s_head(layer, h);

            double dot_sd = 0.0, dot_dd = 0.0;
            for (int i = 0; i < n; ++i) {
                double d = (double)db[i] - (double)da[i];
                s_norm_sq += (double)dd[i] * (double)dd[i];
                d_norm_sq += d * d;
                dot_sd += (double)dd[i] * d;
                dot_dd += d * d;
            }

            double proj = (dot_dd > 1e-20) ? std::abs(dot_sd / dot_dd) : 0.0;
            heads.push_back({layer, h, proj, std::sqrt(dot_dd)});
        }
    }

    double s_norm = std::sqrt(s_norm_sq);
    double d_norm = std::sqrt(d_norm_sq);
    double global_alpha = (d_norm > 1e-10) ? ratio * s_norm / d_norm : ratio;

    double proj_sum = 0.0;
    int proj_count = 0;
    for (const auto & hi : heads) {
        if (hi.delta_norm > 1e-10) {
            proj_sum += hi.proj;
            proj_count++;
        }
    }
    double proj_mean = (proj_count > 0) ? proj_sum / proj_count : 1.0;

    for (const auto & hi : heads) {
        auto da = sa.s_head(hi.layer, hi.head);
        auto db = sb.s_head(hi.layer, hi.head);
        auto dd = dst.s_head(hi.layer, hi.head);

        if (hi.delta_norm < 1e-10) continue;

        double weight = (proj_mean > 1e-10) ? hi.proj / proj_mean : 1.0;
        // cap weight at 3x mean to prevent outlier heads from dominating
        weight = std::min(weight, 3.0);
        double alpha = global_alpha * weight;

        for (int i = 0; i < n; ++i) {
            dd[i] += (float)(alpha * ((double)db[i] - (double)da[i]));
        }
    }
}

// ── query ─────────────────────────────────────────────────────────────

struct QResult {
    float p;
    std::string gen;
};

static QResult do_query(rp::Model & model, rp::Context & ctx,
                        const rp::StateBuf & state,
                        const std::string & prefix, const std::string & qstr,
                        llama_token tok_a, llama_token tok_b,
                        const llama_vocab * vocab, int n_predict, uint32_t seed) {
    auto ptoks = rp::tokenize(vocab, prefix);
    auto qtoks = rp::tokenize(vocab, qstr, false, false);
    ctx.clear_memory();
    ctx.decode(rp::span<const llama_token>(ptoks.data(), ptoks.size()));
    state.store_to(ctx.raw());
    for (const auto & t : qtoks) ctx.decode_one(t);

    const float * logits = llama_get_logits(ctx.raw());
    QResult qr;
    qr.p = rp::prob_of(logits[tok_a], logits[tok_b]);
    if (n_predict > 0) {
        common_params_sampling sp;
        sp.seed = seed;
        rp::Generator gen(model, ctx, sp);
        llama_token last = qtoks.back();
        gen.accept_prompt(rp::span<const llama_token>(&last, 1));
        qr.gen = gen.run(n_predict).text;
        for (auto & c : qr.gen) if (c == '\n') c = ' ';
    }
    return qr;
}

int main(int argc, char ** argv) {
    auto args = rp::parse_common_args(argc, argv);
    args.defaults(4096, 15);

    if (args.help_requested) {
        std::printf("usage: %s -m MODEL [--layers 16-31] [-n 15]\n", argv[0]);
        return 0;
    }

    if (!rp::require_model(args)) return 1;
    if (!rp::reject_extra(args))  return 1;

    return rp::run_experiment([&] {
        auto env = rp::make_env(args);
        auto & model = env->model;
        auto & ctx   = env->ctx;
        const auto & geom  = env->geom;
        const auto * vocab = env->vocab;
        int n_head = geom.n_head;
        int hs = geom.head_size;

        std::string layer_range = args.layer_range.empty() ? "16-31" : args.layer_range;
        auto layers = rp::parse_layer_range(layer_range, geom.n_layer);

        llama_token tok_austin  = rp::find_token(vocab, "Austin");
        llama_token tok_chicago = rp::find_token(vocab, "Chicago");
        llama_token tok_paris   = rp::find_token(vocab, "Paris");
        llama_token tok_madrid  = rp::find_token(vocab, "Madrid");
        llama_token tok_car     = rp::find_token(vocab, "car");
        llama_token tok_house   = rp::find_token(vocab, "house");

        // ── edit pairs ──
        struct EditPair {
            std::string from, to;
        };

        // ── test contexts of increasing length ──
        struct TestCase {
            std::string name;
            std::string context;
            std::string query;
            llama_token tok_old, tok_new;
            EditPair edit;
            // control query (should stay unchanged)
            std::string ctrl_query;
            llama_token tok_ctrl_a, tok_ctrl_b;
        };

        TestCase tests[] = {
            {
                "short: Bob Austin→Chicago",
                "Alice lives in Madrid. Bob lives in Austin.",
                " Where does Bob live? Bob lives in",
                tok_austin, tok_chicago,
                {"Bob lives in Austin", "Bob lives in Chicago"},
                " Where does Alice live? Alice lives in",
                tok_madrid, tok_chicago,
            },
            {
                "medium: Bob Austin→Chicago (4 entities)",
                "Alice lives in Madrid. Bob lives in Austin. Carol lives in London. Dave lives in Rome.",
                " Where does Bob live? Bob lives in",
                tok_austin, tok_chicago,
                {"Bob lives in Austin", "Bob lives in Chicago"},
                " Where does Alice live? Alice lives in",
                tok_madrid, tok_chicago,
            },
            {
                "long: Bob Austin→Chicago (detailed)",
                "Alice lives in Madrid and works as a teacher. Bob lives in Austin and has a car. "
                "Carol lives in London and plays piano. Dave lives in Rome and likes cooking. "
                "Eve lives in Berlin and studies physics. Frank lives in Tokyo and rides a bicycle.",
                " Where does Bob live? Bob lives in",
                tok_austin, tok_chicago,
                {"Bob lives in Austin", "Bob lives in Chicago"},
                " Where does Alice live? Alice lives in",
                tok_madrid, tok_chicago,
            },
            {
                "erase: Alice Madrid (short)",
                "Alice lives in Madrid. Bob lives in Austin.",
                " Where does Alice live? Alice lives in",
                tok_madrid, tok_paris,
                {"Alice lives in Madrid", "The weather is nice today"},
                " Where does Bob live? Bob lives in",
                tok_austin, tok_paris,
            },
            {
                "erase: Alice Madrid (long)",
                "Alice lives in Madrid and works as a teacher. Bob lives in Austin and has a car. "
                "Carol lives in London and plays piano. Dave lives in Rome and likes cooking.",
                " Where does Alice live? Alice lives in",
                tok_madrid, tok_paris,
                {"Alice lives in Madrid", "The weather is nice today"},
                " Where does Bob live? Bob lives in",
                tok_austin, tok_paris,
            },
            {
                "ownership: Bob car→house",
                "Alice has a red hat. Bob has a car.",
                " What does Bob have? Bob has a",
                tok_car, tok_house,
                {"Bob has a car", "Bob has a house"},
                " Alice's hat is",
                rp::find_token(vocab, "red"), tok_house,
            },
        };

        // capture edit pair states
        auto capture = [&](const std::string & prompt) -> rp::StateBuf {
            auto toks = rp::tokenize(vocab, prompt);
            ctx.clear_memory();
            ctx.decode(rp::span<const llama_token>(toks.data(), toks.size()));
            rp::StateBuf s(geom);
            s.load_from(ctx.raw());
            return s;
        };

        // header
        std::printf("%-35s  %-12s  %-6s  |  %8s  %8s  %s\n",
                    "test", "method", "param", "P(old)", "P(ctrl)", "gen");
        std::printf("%s\n", std::string(35+2+12+2+6+3+3+8+2+8+2+30, '-').c_str());

        for (const auto & tc : tests) {
            auto state_base = capture(tc.context);
            auto s_edit_a   = capture(tc.edit.from);
            auto s_edit_b   = capture(tc.edit.to);

            // baseline
            auto bl = do_query(model, ctx, state_base, tc.context, tc.query,
                               tc.tok_old, tc.tok_new, vocab, args.n_predict, args.seed);
            auto bl_ctrl = do_query(model, ctx, state_base, tc.context, tc.ctrl_query,
                                    tc.tok_ctrl_a, tc.tok_ctrl_b, vocab, args.n_predict, args.seed);
            std::printf("%-35s  %-12s  %-6s  |  %8.4f  %8.4f  %s\n",
                        tc.name.substr(0, 35).c_str(), "BASE", "-",
                        bl.p, bl_ctrl.p, bl.gen.substr(0, 28).c_str());

            // norm-ratio at various ratios
            for (double ratio : {0.15, 0.20, 0.30, 0.50}) {
                rp::StateBuf edited(state_base);
                apply_norm_ratio(edited, s_edit_a, s_edit_b, layers, n_head, hs, ratio);
                auto r = do_query(model, ctx, edited, tc.context, tc.query,
                                  tc.tok_old, tc.tok_new, vocab, args.n_predict, args.seed);
                auto rc = do_query(model, ctx, edited, tc.context, tc.ctrl_query,
                                   tc.tok_ctrl_a, tc.tok_ctrl_b, vocab, args.n_predict, args.seed);
                char p[16]; std::snprintf(p, sizeof p, "r=%.2f", ratio);
                std::printf("%-35s  %-12s  %-6s  |  %8.4f  %8.4f  %s\n",
                            "", "NORM-RATIO", p, r.p, rc.p, r.gen.substr(0, 28).c_str());
            }

            // global projection at various strengths
            for (double s : {0.5, 1.0, 2.0, 5.0}) {
                rp::StateBuf edited(state_base);
                apply_global_projection(edited, s_edit_a, s_edit_b, layers, n_head, hs, s);
                auto r = do_query(model, ctx, edited, tc.context, tc.query,
                                  tc.tok_old, tc.tok_new, vocab, args.n_predict, args.seed);
                auto rc = do_query(model, ctx, edited, tc.context, tc.ctrl_query,
                                   tc.tok_ctrl_a, tc.tok_ctrl_b, vocab, args.n_predict, args.seed);
                char p[16]; std::snprintf(p, sizeof p, "s=%.1f", s);
                std::printf("%-35s  %-12s  %-6s  |  %8.4f  %8.4f  %s\n",
                            "", "GLOB-PROJ", p, r.p, rc.p, r.gen.substr(0, 28).c_str());
            }

            // per-head projection at various strengths
            for (double s : {0.5, 1.0, 2.0, 5.0}) {
                rp::StateBuf edited(state_base);
                apply_perhead_projection(edited, s_edit_a, s_edit_b, layers, n_head, hs, s);
                auto r = do_query(model, ctx, edited, tc.context, tc.query,
                                  tc.tok_old, tc.tok_new, vocab, args.n_predict, args.seed);
                auto rc = do_query(model, ctx, edited, tc.context, tc.ctrl_query,
                                   tc.tok_ctrl_a, tc.tok_ctrl_b, vocab, args.n_predict, args.seed);
                char p[16]; std::snprintf(p, sizeof p, "s=%.1f", s);
                std::printf("%-35s  %-12s  %-6s  |  %8.4f  %8.4f  %s\n",
                            "", "HEAD-PROJ", p, r.p, rc.p, r.gen.substr(0, 28).c_str());
            }

            // per-head signed projection
            for (double s : {0.5, 1.0, 2.0, 5.0}) {
                rp::StateBuf edited(state_base);
                apply_perhead_signed_projection(edited, s_edit_a, s_edit_b, layers, n_head, hs, s);
                auto r = do_query(model, ctx, edited, tc.context, tc.query,
                                  tc.tok_old, tc.tok_new, vocab, args.n_predict, args.seed);
                auto rc = do_query(model, ctx, edited, tc.context, tc.ctrl_query,
                                   tc.tok_ctrl_a, tc.tok_ctrl_b, vocab, args.n_predict, args.seed);
                char p[16]; std::snprintf(p, sizeof p, "s=%.1f", s);
                std::printf("%-35s  %-12s  %-6s  |  %8.4f  %8.4f  %s\n",
                            "", "HEAD-SIGN", p, r.p, rc.p, r.gen.substr(0, 28).c_str());
            }

            // proj-weighted norm-ratio (hybrid)
            for (double ratio : {0.15, 0.20, 0.30, 0.50}) {
                rp::StateBuf edited(state_base);
                apply_proj_weighted_norm(edited, s_edit_a, s_edit_b, layers, n_head, hs, ratio);
                auto r = do_query(model, ctx, edited, tc.context, tc.query,
                                  tc.tok_old, tc.tok_new, vocab, args.n_predict, args.seed);
                auto rc = do_query(model, ctx, edited, tc.context, tc.ctrl_query,
                                   tc.tok_ctrl_a, tc.tok_ctrl_b, vocab, args.n_predict, args.seed);
                char p[16]; std::snprintf(p, sizeof p, "r=%.2f", ratio);
                std::printf("%-35s  %-12s  %-6s  |  %8.4f  %8.4f  %s\n",
                            "", "PROJ-NORM", p, r.p, rc.p, r.gen.substr(0, 28).c_str());
            }

            // proj-weighted capped
            for (double ratio : {0.15, 0.20, 0.30, 0.50}) {
                rp::StateBuf edited(state_base);
                apply_proj_weighted_capped(edited, s_edit_a, s_edit_b, layers, n_head, hs, ratio);
                auto r = do_query(model, ctx, edited, tc.context, tc.query,
                                  tc.tok_old, tc.tok_new, vocab, args.n_predict, args.seed);
                auto rc = do_query(model, ctx, edited, tc.context, tc.ctrl_query,
                                   tc.tok_ctrl_a, tc.tok_ctrl_b, vocab, args.n_predict, args.seed);
                char p[16]; std::snprintf(p, sizeof p, "r=%.2f", ratio);
                std::printf("%-35s  %-12s  %-6s  |  %8.4f  %8.4f  %s\n",
                            "", "PROJ-CAP", p, r.p, rc.p, r.gen.substr(0, 28).c_str());
            }

            std::printf("\n");
            std::fflush(stdout);
        }

        std::fprintf(stderr, "done\n");
    });
}
