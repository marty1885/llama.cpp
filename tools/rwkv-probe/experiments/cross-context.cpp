// experiments/cross-context.cpp — cross-context delta generalization.
//
// Captures state deltas from multiple prompt phrasings of the same fact,
// then cross-applies each delta to every other phrasing's state.
// Tests whether a delta from "Alice lives in Paris/London" (simple)
// can edit a state from "Last year Alice moved to Paris..." (narrative).
//
// Usage:
//   llama-rwkv-cross-context -m model.gguf --tests cross_context_tests.json
//       [--layers 16-31] [-n 20]

#include "rwkv_probe/state.h"
#include "rwkv_probe/experiment.h"
#include "rwkv_probe/pipeline.h"
#include "rwkv_probe/edit.h"

#include <cstdio>
#include <string>
#include <vector>

namespace rp = rwkv_probe;
using json = nlohmann::json;

struct TestCase {
    std::string name, prompt_a, prompt_b, expect_a, expect_b;
};

int main(int argc, char ** argv) {
    auto args = rp::parse_common_args(argc, argv);
    args.defaults(/*ctx=*/2048, /*predict=*/15);
    if (args.layer_range.empty()) args.layer_range = "16-31";

    if (args.help_requested) {
        std::printf("usage: %s -m MODEL --tests FILE [--layers 16-31] [-n 15]\n", argv[0]);
        return 0;
    }
    if (!rp::require_model(args) || !rp::require_tests(args)) return 1;
    if (!rp::reject_extra(args)) return 1;

    auto arr = rp::load_tests(args);
    std::vector<TestCase> tests;
    for (const auto & e : arr) {
        tests.push_back({
            e.at("name").get<std::string>(),
            e.at("prompt_a").get<std::string>(),
            e.at("prompt_b").get<std::string>(),
            e.at("expect_a").get<std::string>(),
            e.at("expect_b").get<std::string>(),
        });
    }

    return rp::run_experiment([&] {
        auto env = rp::make_env(args);
        auto & model = env->model;
        auto & ctx   = env->ctx;
        const auto & geom  = env->geom;
        const auto * vocab = env->vocab;

        auto layers = rp::parse_layer_range(args.layer_range, geom.n_layer);

        // group tests by fact (same expect_a + expect_b)
        struct FactGroup {
            std::string fact_id;  // e.g. "Paris→London"
            std::vector<std::size_t> test_indices;
        };
        std::vector<FactGroup> groups;
        for (std::size_t i = 0; i < tests.size(); ++i) {
            std::string fid = tests[i].expect_a + "→" + tests[i].expect_b;
            bool found = false;
            for (auto & g : groups) {
                if (g.fact_id == fid) {
                    g.test_indices.push_back(i);
                    found = true;
                    break;
                }
            }
            if (!found) {
                groups.push_back({fid, {i}});
            }
        }

        // capture all states and baselines
        struct CapturedState {
            rp::StateBuf state_a;
            rp::StateBuf state_b;
            std::vector<llama_token> toks_a;
            std::vector<llama_token> toks_b;
            llama_token tok_a, tok_b;
            float baseline_a, baseline_b;

            CapturedState(const rp::ModelGeometry & g) : state_a(g), state_b(g) {}
        };

        std::vector<CapturedState> captured;
        captured.reserve(tests.size());

        std::fprintf(stderr, "capturing %zu states...\n", tests.size());

        for (std::size_t i = 0; i < tests.size(); ++i) {
            const auto & tc = tests[i];

            captured.emplace_back(geom);
            auto & cap = captured.back();

            cap.tok_a = rp::find_token(vocab, tc.expect_a);
            cap.tok_b = rp::find_token(vocab, tc.expect_b);
            cap.toks_a = rp::tokenize(vocab, tc.prompt_a);
            cap.toks_b = rp::tokenize(vocab, tc.prompt_b);

            // state A
            ctx.clear_memory();
            ctx.decode(rp::span<const llama_token>(cap.toks_a.data(), cap.toks_a.size() - 1));
            cap.state_a.load_from(ctx.raw());

            // state B
            ctx.clear_memory();
            ctx.decode(rp::span<const llama_token>(cap.toks_b.data(), cap.toks_b.size() - 1));
            cap.state_b.load_from(ctx.raw());

            // baselines
            cap.baseline_a = rp::measure(ctx, cap.state_a, cap.toks_a, cap.tok_a, cap.tok_b);
            cap.baseline_b = rp::measure(ctx, cap.state_b, cap.toks_b, cap.tok_a, cap.tok_b);

            std::fprintf(stderr, "  %s: P(%s)=%.4f  P(%s)=%.4f %s\n",
                         tc.name.c_str(),
                         tc.expect_a.c_str(), cap.baseline_a,
                         tc.expect_b.c_str(), 1.0f - cap.baseline_b,
                         (cap.baseline_a > 0.8 && cap.baseline_b < 0.2) ? "OK" : "WEAK");
        }

        // cross-application matrix
        for (const auto & group : groups) {
            std::printf("\n======================================================================\n");
            std::printf("FACT GROUP: %s  (%zu phrasings)\n",
                        group.fact_id.c_str(), group.test_indices.size());
            std::printf("======================================================================\n");

            // print baselines
            std::printf("\n  BASELINES:\n");
            for (std::size_t idx : group.test_indices) {
                std::printf("  %-20s  P(%s)=%.4f  P(%s)=%.4f\n",
                            tests[idx].name.c_str(),
                            tests[idx].expect_a.c_str(), captured[idx].baseline_a,
                            tests[idx].expect_b.c_str(), 1.0f - captured[idx].baseline_b);
            }

            // cross-apply: delta from donor → target state_A
            std::printf("\n  CROSS-APPLICATION MATRIX — P(%s) after edit (lower = better flip):\n",
                        tests[group.test_indices[0]].expect_a.c_str());
            // determine column width from longest test name
            int cw = 10;
            for (std::size_t ti : group.test_indices) {
                int len = (int)tests[ti].name.size();
                if (len > cw) cw = len;
            }
            cw += 2;  // padding

            std::printf("  %-*s", cw, "donor \\ target");
            for (std::size_t ti : group.test_indices) {
                std::printf("  %*s", cw, tests[ti].name.c_str());
            }
            std::printf("\n");
            std::printf("  %s\n",
                std::string(cw + group.test_indices.size() * (cw + 2), '-').c_str());

            for (std::size_t di : group.test_indices) {
                const auto & donor = captured[di];

                std::printf("  %-*s", cw, tests[di].name.c_str());

                // cross-apply to each target (includes self)
                for (std::size_t ti : group.test_indices) {
                    const auto & target = captured[ti];

                    rp::StateBuf edited(target.state_a);
                    rp::apply_rank_delta(edited, donor.state_a, donor.state_b, layers, 0);
                    float p = rp::measure(ctx, edited, target.toks_a, target.tok_a, target.tok_b);

                    if (di == ti) {
                        // self-edit: mark with *
                        char buf[32];
                        std::snprintf(buf, sizeof buf, "%.4f*", p);
                        std::printf("  %*s", cw, buf);
                    } else {
                        std::printf("  %*.4f", cw, p);
                    }
                }
                std::printf("\n");
            }

            // also show generation for a few interesting cross-applications
            if (args.n_predict > 0 && group.test_indices.size() >= 2) {
                std::printf("\n  CROSS-CONTEXT GENERATION EXAMPLES:\n");

                // use first phrasing's delta on second phrasing, and vice versa
                for (std::size_t pass = 0; pass < 2 && pass < group.test_indices.size() - 1; ++pass) {
                    std::size_t di = group.test_indices[pass];
                    std::size_t ti = group.test_indices[(pass + 1) % group.test_indices.size()];

                    const auto & donor = captured[di];
                    const auto & target = captured[ti];

                    rp::StateBuf edited(target.state_a);
                    rp::apply_rank_delta(edited, donor.state_a, donor.state_b, layers, 0);

                    std::string gen = rp::generate_from(model, ctx, edited,
                                                    target.toks_a, args.n_predict, args.seed);

                    std::printf("\n  donor: %s delta  →  target: %s state\n",
                                tests[di].name.c_str(), tests[ti].name.c_str());
                    std::printf("  prompt: \"%s\"\n", tests[ti].prompt_a.c_str());
                    std::printf("  output: %s\n", gen.c_str());
                }
            }
        }

        std::fprintf(stderr, "\ndone\n");
    });
}
