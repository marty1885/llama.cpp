// experiments/state-svd.cpp — SVD of per-head state deltas.
//
// For each test case, captures recurrent state from prompt A and B,
// computes delta = state_B - state_A at every head, then SVDs the
// 64×64 delta matrix. Reports the singular value spectrum.
//
// Key question: is the fact delta rank-1? If sigma_1 >> sigma_2, surgical
// editing via s_new = s - sigma_1 u_1 v_1^T + sigma_1 u_1_new v_1^T is tractable.
//
// Usage:
//   llama-rwkv-state-svd -m model.gguf --tests causal_trace_tests.json
//       [--layers 16-31] [--top-sv 5]

#include "rwkv_probe/state.h"
#include "rwkv_probe/generate.h"
#include "rwkv_probe/experiment.h"

#include "rwkv_probe/numeric.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace rp = rwkv_probe;

struct TestCase {
    std::string prompt_a;
    std::string prompt_b;
    std::string expect_a;
    std::string expect_b;
    std::string name;
};

int main(int argc, char ** argv) {
    auto args = rp::parse_common_args(argc, argv);
    args.defaults(2048, 15);

    // experiment-specific args
    int  top_sv    = 10;     // how many singular values to print
    int  top_heads = 10;     // how many heads to show per layer (0 = all)
    double r1_min  = 0.0;    // minimum r1_ratio to print

    for (std::size_t i = 0; i < args.extra.size(); ++i) {
        const std::string & a = args.extra[i];
        if      (a == "--top-sv"    && i+1 < args.extra.size()) top_sv    = std::atoi(args.extra[++i].c_str());
        else if (a == "--top-heads" && i+1 < args.extra.size()) top_heads = std::atoi(args.extra[++i].c_str());
        else if (a == "--r1-min"    && i+1 < args.extra.size()) r1_min    = std::atof(args.extra[++i].c_str());
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 1; }
    }

    if (args.help_requested) {
        std::printf("usage: %s -m MODEL --tests FILE [--layers 16-31] [--top-sv 5]\n",
                    argv[0]);
        return 0;
    }

    if (!rp::require_model(args)) return 1;
    if (!rp::require_tests(args)) return 1;

    auto jarr = rp::load_tests(args);
    std::vector<TestCase> tests;
    for (const auto & e : jarr) {
        tests.push_back({
            e.at("prompt_a").get<std::string>(),
            e.at("prompt_b").get<std::string>(),
            e.at("expect_a").get<std::string>(),
            e.at("expect_b").get<std::string>(),
            e.value("name", ""),
        });
    }

    return rp::run_experiment([&] {
        auto env = rp::make_env(args);
        auto & ctx   = env->ctx;
        const auto & geom  = env->geom;
        const auto * vocab = env->vocab;
        int n_head  = geom.n_head;
        int hs      = geom.head_size;  // 64

        std::string layer_range = args.layer_range.empty() ? "0-31" : args.layer_range;
        auto layers = rp::parse_layer_range(layer_range, geom.n_layer);
        if (layers.empty()) {
            throw std::runtime_error("no valid layers in range '" + layer_range + "'");
        }

        std::fprintf(stderr, "layers: %d-%d  n_head=%d  head_size=%d\n",
                     layers.front(), layers.back(), n_head, hs);

        for (std::size_t ti = 0; ti < tests.size(); ++ti) {
            const auto & tc = tests[ti];
            std::string label = tc.name.empty() ? ("test_" + std::to_string(ti)) : tc.name;

            llama_token tok_a = rp::find_token(vocab, tc.expect_a);
            llama_token tok_b = rp::find_token(vocab, tc.expect_b);

            auto toks_a = rp::tokenize(vocab, tc.prompt_a);
            auto toks_b = rp::tokenize(vocab, tc.prompt_b);

            if (toks_a.size() < 2 || toks_b.size() < 2) {
                std::printf("  need at least 2 tokens per prompt, skipping\n");
                continue;
            }

            // ── capture states at prefix (all-but-last token) ───────────
            ctx.clear_memory();
            if (ctx.decode(rp::span<const llama_token>(toks_a.data(), toks_a.size() - 1)) != 0) {
                std::fprintf(stderr, "  decode A prefix failed\n"); continue;
            }
            rp::StateBuf state_a(geom);
            state_a.load_from(ctx.raw());

            ctx.clear_memory();
            if (ctx.decode(rp::span<const llama_token>(toks_b.data(), toks_b.size() - 1)) != 0) {
                std::fprintf(stderr, "  decode B prefix failed\n"); continue;
            }
            rp::StateBuf state_b(geom);
            state_b.load_from(ctx.raw());

            // ── baselines ──────────────────────────────────────────────
            ctx.clear_memory();
            if (ctx.decode(rp::span<const llama_token>(toks_a.data(), toks_a.size())) != 0) {
                std::fprintf(stderr, "  decode A full failed\n"); continue;
            }
            const float * logits_full_a = llama_get_logits(ctx.raw());
            float bl_la = logits_full_a[tok_a];
            float bl_lb = logits_full_a[tok_b];

            ctx.clear_memory();
            if (ctx.decode(rp::span<const llama_token>(toks_b.data(), toks_b.size())) != 0) {
                std::fprintf(stderr, "  decode B full failed\n"); continue;
            }
            const float * logits_full_b = llama_get_logits(ctx.raw());
            float bl_ba = logits_full_b[tok_a];
            float bl_bb = logits_full_b[tok_b];

            std::printf("\n======================================================================\n");
            std::printf("TEST: %s\n", label.c_str());
            std::printf("  A: %.80s\n", tc.prompt_a.c_str());
            std::printf("  B: %.80s\n", tc.prompt_b.c_str());
            std::printf("  BASELINE A: P(%s)=%.4f   BASELINE B: P(%s)=%.4f\n",
                        tc.expect_a.c_str(), rp::prob_of(bl_la, bl_lb),
                        tc.expect_b.c_str(), rp::prob_of(bl_bb, bl_ba));

            // ── SVD per layer per head — collect all results ─────────
            struct HeadSVD {
                int layer, head;
                double sigma1, sigma2, sigma3, r1_ratio, l2;
            };
            std::vector<HeadSVD> all_results;

            for (int layer : layers) {
                for (int h = 0; h < n_head; ++h) {
                    auto sa = state_a.s_head(layer, h);
                    auto sb = state_b.s_head(layer, h);

                    std::vector<double> delta(hs * hs);
                    double l2_sq = 0.0;
                    for (int r = 0; r < hs; ++r) {
                        for (int c = 0; c < hs; ++c) {
                            double d = (double)sb[r * hs + c] - (double)sa[r * hs + c];
                            delta[r * hs + c] = d;
                            l2_sq += d * d;
                        }
                    }
                    double l2_n = std::sqrt(l2_sq);
                    if (l2_n < 1e-10) continue;

                    auto svdr = rp::numeric::svd(delta.data(), hs);
                    int nsv = (int)svdr.sigma.size();

                    double sum_sq = 0.0;
                    for (int i = 0; i < nsv; ++i) sum_sq += svdr.sigma[i] * svdr.sigma[i];
                    double r1 = (sum_sq > 0) ? (svdr.sigma[0] * svdr.sigma[0] / sum_sq) : 0.0;

                    all_results.push_back({layer, h,
                        (nsv > 0) ? svdr.sigma[0] : 0.0,
                        (nsv > 1) ? svdr.sigma[1] : 0.0,
                        (nsv > 2) ? svdr.sigma[2] : 0.0,
                        r1, l2_n});
                }
            }

            // ── per-layer summary ─────────────────────────────────────
            std::printf("\n  PER-LAYER SUMMARY (mean r1_ratio, top heads by r1):\n");
            std::printf("  %-5s  %8s  %8s  %s\n",
                        "layer", "mean_r1", "max_r1", "top heads (r1 > 0.8)");
            std::printf("  %s\n", std::string(70, '-').c_str());

            for (int layer : layers) {
                double r1_sum = 0.0;
                double r1_max = 0.0;
                int count = 0;
                std::vector<std::pair<double, int>> layer_heads;  // (r1, head)

                for (const auto & r : all_results) {
                    if (r.layer != layer) continue;
                    r1_sum += r.r1_ratio;
                    if (r.r1_ratio > r1_max) r1_max = r.r1_ratio;
                    count++;
                    layer_heads.push_back({r.r1_ratio, r.head});
                }
                std::sort(layer_heads.begin(), layer_heads.end(),
                          [](const auto & a, const auto & b) { return a.first > b.first; });

                std::printf("  L%-4d  %8.4f  %8.4f  ", layer,
                            count > 0 ? r1_sum / count : 0.0, r1_max);
                int shown = 0;
                for (const auto & lh : layer_heads) {
                    if (lh.first < 0.8) break;
                    if (shown > 0) std::printf(", ");
                    std::printf("H%d(%.3f)", lh.second, lh.first);
                    shown++;
                }
                if (shown == 0) std::printf("(none)");
                std::printf("\n");
            }

            // ── top heads across all layers sorted by r1 ──────────────
            std::sort(all_results.begin(), all_results.end(),
                      [](const HeadSVD & a, const HeadSVD & b) {
                          return a.r1_ratio > b.r1_ratio;
                      });

            int n_show = (top_heads > 0) ? std::min(top_heads, (int)all_results.size())
                                         : (int)all_results.size();
            std::printf("\n  TOP %d HEADS BY r1_ratio%s:\n", n_show,
                        r1_min > 0 ? " (filtered)" : "");
            std::printf("  %-5s %-4s  %10s  %10s  %10s  %8s  %8s\n",
                        "layer", "head", "sigma_1", "sigma_2", "sigma_3",
                        "r1_ratio", "l2_norm");
            std::printf("  %s\n", std::string(70, '-').c_str());

            for (int i = 0; i < n_show; ++i) {
                const auto & r = all_results[i];
                if (r.r1_ratio < r1_min) break;
                std::printf("  L%-3d  H%-3d  %10.4f  %10.4f  %10.4f  %8.4f  %8.4f\n",
                            r.layer, r.head, r.sigma1, r.sigma2, r.sigma3,
                            r.r1_ratio, r.l2);
            }
        }

        std::fprintf(stderr, "\ndone\n");
    });
}
