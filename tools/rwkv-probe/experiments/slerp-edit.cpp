// experiments/slerp-edit.cpp — compare linear vs spherical interpolation for
// RWKV state editing.
//
// Instead of dst = a + alpha*(b - a)  (linear addition of the delta),
// try SLERP(a, b, t) on each head's flattened WKV vector, which preserves
// the norm and interpolates along the great circle on the hypersphere.
//
// Sweeps interpolation strength t and compares both methods head-to-head.
//
// Usage:
//   llama-rwkv-slerp-edit -m model.gguf --tests entity_edit_tests.json
//       [--layers 16-31] [-n 20]

#include "rwkv_probe/state.h"
#include "rwkv_probe/generate.h"
#include "rwkv_probe/experiment.h"
#include "rwkv_probe/pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace rp = rwkv_probe;
using json = nlohmann::json;

struct ControlQuery {
    std::string query;
    std::string expect;
};

struct EntityTest {
    std::string name;
    std::string prompt_a, prompt_b;
    std::string query_target;
    std::string expect_a_target, expect_b_target;
    std::vector<ControlQuery> controls;
};

// -- interpolation methods --------------------------------------------------

// Linear: dst = a + t*(b - a)
static void apply_lerp(rp::StateBuf & dst,
                       const rp::StateBuf & sa, const rp::StateBuf & sb,
                       const std::vector<int> & layers, int n_head, int hs,
                       double t) {
    for (int layer : layers) {
        for (int h = 0; h < n_head; ++h) {
            auto da = sa.s_head(layer, h);
            auto db = sb.s_head(layer, h);
            auto dd = dst.s_head(layer, h);
            int n = hs * hs;
            for (int i = 0; i < n; ++i) {
                dd[i] = (float)((1.0 - t) * (double)da[i] + t * (double)db[i]);
            }
        }
    }
}

// SLERP: spherical linear interpolation on the flattened head vector.
// slerp(a, b, t) = sin((1-t)*omega)/sin(omega) * a + sin(t*omega)/sin(omega) * b
// where omega = arccos(cos_sim(a, b)).
// Falls back to lerp when vectors are nearly parallel or anti-parallel.
static void apply_slerp(rp::StateBuf & dst,
                        const rp::StateBuf & sa, const rp::StateBuf & sb,
                        const std::vector<int> & layers, int n_head, int hs,
                        double t) {
    for (int layer : layers) {
        for (int h = 0; h < n_head; ++h) {
            auto da = sa.s_head(layer, h);
            auto db = sb.s_head(layer, h);
            auto dd = dst.s_head(layer, h);
            int n = hs * hs;

            // compute norms and dot product
            double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
            for (int i = 0; i < n; ++i) {
                dot    += (double)da[i] * (double)db[i];
                norm_a += (double)da[i] * (double)da[i];
                norm_b += (double)db[i] * (double)db[i];
            }
            norm_a = std::sqrt(norm_a);
            norm_b = std::sqrt(norm_b);

            // degenerate: zero-norm vector -> just lerp
            if (norm_a < 1e-12 || norm_b < 1e-12) {
                for (int i = 0; i < n; ++i) {
                    dd[i] = (float)((1.0 - t) * (double)da[i] + t * (double)db[i]);
                }
                continue;
            }

            double cos_omega = dot / (norm_a * norm_b);
            // clamp for numerical safety
            cos_omega = std::max(-1.0, std::min(1.0, cos_omega));
            double omega = std::acos(cos_omega);

            // nearly parallel or anti-parallel -> lerp fallback
            if (std::abs(omega) < 1e-6 || std::abs(omega - M_PI) < 1e-6) {
                for (int i = 0; i < n; ++i) {
                    dd[i] = (float)((1.0 - t) * (double)da[i] + t * (double)db[i]);
                }
                continue;
            }

            double sin_omega = std::sin(omega);
            double w_a = std::sin((1.0 - t) * omega) / sin_omega;
            double w_b = std::sin(t * omega) / sin_omega;

            for (int i = 0; i < n; ++i) {
                dd[i] = (float)(w_a * (double)da[i] + w_b * (double)db[i]);
            }
        }
    }
}

// -- query helper -----------------------------------------------------------

struct QueryResult {
    float p_target;
    std::string gen;
};

static QueryResult query_with_state(
        rp::Model & model, rp::Context & ctx,
        const rp::StateBuf & state,
        const std::string & prefix,
        const std::string & query,
        llama_token tok_expected, llama_token tok_alt,
        const llama_vocab * vocab,
        int n_predict, uint32_t seed) {
    auto pr = rp::probe(model, ctx, state, prefix, query,
                        tok_expected, tok_alt, vocab, n_predict, seed);
    return {pr.p_binary, pr.generation};
}

int main(int argc, char ** argv) {
    auto args = rp::parse_common_args(argc, argv);
    if (args.help_requested) {
        std::printf("usage: %s -m MODEL --tests FILE [--layers 16-31] [-n 20]\n", argv[0]);
        return 0;
    }
    args.defaults(2048, 20);
    if (args.layer_range.empty()) args.layer_range = "16-31";
    if (!rp::require_model(args)) return 1;
    if (!rp::require_tests(args)) return 1;
    if (!rp::reject_extra(args))  return 1;

    // load tests from JSON
    json arr = rp::load_tests(args);
    std::vector<EntityTest> tests;
    for (const auto & e : arr) {
        EntityTest tc;
        tc.name           = e.at("name").get<std::string>();
        tc.prompt_a       = e.at("prompt_a").get<std::string>();
        tc.prompt_b       = e.at("prompt_b").get<std::string>();
        tc.query_target   = e.at("query_target").get<std::string>();
        tc.expect_a_target = e.at("expect_a_target").get<std::string>();
        tc.expect_b_target = e.at("expect_b_target").get<std::string>();

        if (e.contains("query_controls")) {
            for (const auto & ctrl : e.at("query_controls")) {
                for (auto it = ctrl.begin(); it != ctrl.end(); ++it) {
                    tc.controls.push_back({it.key(), it.value().get<std::string>()});
                }
            }
        } else if (e.contains("query_control")) {
            tc.controls.push_back({
                e.at("query_control").get<std::string>(),
                e.at("expect_control").get<std::string>(),
            });
        }
        tests.push_back(std::move(tc));
    }

    return rp::run_experiment([&] {
        auto env = rp::make_env(args);
        auto & model = env->model;
        auto & ctx   = env->ctx;
        const auto & geom  = env->geom;
        const auto * vocab  = env->vocab;
        int n_head = geom.n_head;
        int hs = geom.head_size;

        auto layers = rp::parse_layer_range(args.layer_range, geom.n_layer);

        std::printf("layers: %d-%d (%zu)  n_head=%d  head_size=%d\n\n",
                    layers.front(), layers.back(), layers.size(), n_head, hs);

        for (const auto & tc : tests) {
            llama_token tok_a_target = rp::find_token(vocab, tc.expect_a_target);
            llama_token tok_b_target = rp::find_token(vocab, tc.expect_b_target);

            struct ResolvedControl {
                std::string query;
                std::string expect_str;
                llama_token tok;
            };
            std::vector<ResolvedControl> controls;
            for (const auto & c : tc.controls) {
                controls.push_back({c.query, c.expect, rp::find_token(vocab, c.expect)});
            }

            // capture states
            auto toks_a = rp::tokenize(vocab, tc.prompt_a);
            auto toks_b = rp::tokenize(vocab, tc.prompt_b);

            ctx.clear_memory();
            ctx.decode(rp::span<const llama_token>(toks_a.data(), toks_a.size()));
            rp::StateBuf state_a(geom);
            state_a.load_from(ctx.raw());

            ctx.clear_memory();
            ctx.decode(rp::span<const llama_token>(toks_b.data(), toks_b.size()));
            rp::StateBuf state_b(geom);
            state_b.load_from(ctx.raw());

            std::printf("======================================================================\n");
            std::printf("TEST: %s\n", tc.name.c_str());
            std::printf("  A: \"%s\"\n", tc.prompt_a.c_str());
            std::printf("  B: \"%s\"\n", tc.prompt_b.c_str());
            std::printf("  target query: \"%s\"  (expect: %s->%s)\n",
                        tc.query_target.c_str(), tc.expect_a_target.c_str(), tc.expect_b_target.c_str());
            for (const auto & c : controls) {
                std::printf("  control query: \"%s\"  (expect: %s)\n",
                            c.query.c_str(), c.expect_str.c_str());
            }

            // -- baselines --------------------------------------------------
            auto bl_a = query_with_state(model, ctx, state_a,
                tc.prompt_a, tc.query_target,
                tok_a_target, tok_b_target, vocab, args.n_predict, args.seed);
            auto bl_b = query_with_state(model, ctx, state_b,
                tc.prompt_b, tc.query_target,
                tok_a_target, tok_b_target, vocab, args.n_predict, args.seed);

            std::printf("\n  BASELINES:\n");
            std::printf("    A target P(exp)=%.4f  %s\n", bl_a.p_target,
                        bl_a.gen.substr(0, 50).c_str());
            std::printf("    B target P(exp)=%.4f  %s\n", bl_b.p_target,
                        bl_b.gen.substr(0, 50).c_str());

            // -- sweep t values, compare LERP vs SLERP ----------------------
            std::vector<double> t_vals = {0.25, 0.5, 0.75, 1.0, 1.25, 1.5};

            // header
            std::printf("\n  %-6s  %-6s  %8s", "method", "t", "P(tgt)");
            for (const auto & c : controls) {
                char hdr[32];
                std::snprintf(hdr, sizeof hdr, "P(%s)", c.expect_str.substr(0, 6).c_str());
                std::printf("  %8s", hdr);
            }
            std::printf("  %s\n", "gen");
            std::printf("  %s\n", std::string(8 + 8 + 10 + controls.size() * 10 + 42, '-').c_str());

            for (double t : t_vals) {
                // LERP
                {
                    rp::StateBuf edited(state_a);
                    apply_lerp(edited, state_a, state_b, layers, n_head, hs, t);

                    auto r = query_with_state(model, ctx, edited,
                        tc.prompt_a, tc.query_target,
                        tok_a_target, tok_b_target, vocab, args.n_predict, args.seed);

                    std::printf("  %-6s  %-6.2f  %8.4f", "LERP", t, r.p_target);

                    for (const auto & c : controls) {
                        auto rc = query_with_state(model, ctx, edited,
                            tc.prompt_a, c.query,
                            c.tok, tok_b_target, vocab, args.n_predict, args.seed);
                        std::printf("  %8.4f", rc.p_target);
                    }

                    std::string tgen = r.gen.substr(0, 40);
                    for (auto & ch : tgen) if (ch == '\n') ch = ' ';
                    std::printf("  %s\n", tgen.c_str());
                }

                // SLERP
                {
                    rp::StateBuf edited(state_a);
                    apply_slerp(edited, state_a, state_b, layers, n_head, hs, t);

                    auto r = query_with_state(model, ctx, edited,
                        tc.prompt_a, tc.query_target,
                        tok_a_target, tok_b_target, vocab, args.n_predict, args.seed);

                    std::printf("  %-6s  %-6.2f  %8.4f", "SLERP", t, r.p_target);

                    for (const auto & c : controls) {
                        auto rc = query_with_state(model, ctx, edited,
                            tc.prompt_a, c.query,
                            c.tok, tok_b_target, vocab, args.n_predict, args.seed);
                        std::printf("  %8.4f", rc.p_target);
                    }

                    std::string tgen = r.gen.substr(0, 40);
                    for (auto & ch : tgen) if (ch == '\n') ch = ' ';
                    std::printf("  %s\n", tgen.c_str());
                }

                std::fflush(stdout);
            }

            // -- norm diagnostic: show how each method affects head norms ----
            std::printf("\n  NORM DIAGNOSTIC (t=1.0, first 5 affected heads):\n");
            std::printf("  %-8s  %-8s  %12s  %12s  %12s  %10s\n",
                        "layer", "head", "norm_A", "norm_LERP", "norm_SLERP", "norm_B");
            {
                rp::StateBuf edited_lerp(state_a);
                rp::StateBuf edited_slerp(state_a);
                apply_lerp(edited_lerp, state_a, state_b, layers, n_head, hs, 1.0);
                apply_slerp(edited_slerp, state_a, state_b, layers, n_head, hs, 1.0);

                int shown = 0;
                for (int layer : layers) {
                    for (int h = 0; h < n_head && shown < 5; ++h) {
                        auto ha = state_a.s_head(layer, h);
                        auto hb = state_b.s_head(layer, h);
                        auto hl = edited_lerp.s_head(layer, h);
                        auto hs_vec = edited_slerp.s_head(layer, h);
                        int nn = hs * hs;

                        // check if this head actually differs
                        double diff = 0.0;
                        for (int i = 0; i < nn; ++i) {
                            double d = (double)ha[i] - (double)hb[i];
                            diff += d * d;
                        }
                        if (diff < 1e-10) continue;

                        auto norm = [&](rp::span<const float> v) {
                            double s = 0.0;
                            for (int i = 0; i < nn; ++i) s += (double)v[i] * (double)v[i];
                            return std::sqrt(s);
                        };

                        std::printf("  L%-6d  H%-6d  %12.4f  %12.4f  %12.4f  %10.4f\n",
                                    layer, h, norm(ha), norm(hl), norm(hs_vec), norm(hb));
                        ++shown;
                    }
                }
            }

            std::printf("\n");
        }

        std::fprintf(stderr, "done\n");
    });
}
