// experiments/rank1-validate.cpp — validate that rank-1 editing works.
//
// Applies the rank-1 approximation of the full state delta (all heads,
// configurable layers) and checks whether logits flip. No generation,
// just logit measurement — fast.
//
// Compares three conditions:
//   1. FULL DELTA:  state_A + full (state_B - state_A)  (= state swap, should flip)
//   2. RANK-1:      state_A + rank1_approx(delta)       (should ~flip if rank-1 works)
//   3. RANK-2:      state_A + rank2_approx(delta)       (should be closer to full)
//   4. ERASE:       state_B - rank1_approx(delta)       (should weaken B's fact)
//
// Usage:
//   llama-rwkv-rank1-validate -m model.gguf --tests tests.json [--layers 16-31]

#include "rwkv_probe/experiment.h"
#include "rwkv_probe/state.h"
#include "rwkv_probe/generate.h"
#include "rwkv_probe/pipeline.h"
#include "rwkv_probe/edit.h"

#include <cmath>
#include <string>
#include <vector>

namespace rp = rwkv_probe;
using json = nlohmann::json;

struct TestCase {
    std::string prompt_a, prompt_b, expect_a, expect_b, name;
};

int main(int argc, char ** argv) {
    auto args = rp::parse_common_args(argc, argv);
    args.defaults(2048, 0);

    bool do_gen = false;

    // Check if -n was explicitly provided (n_predict will have been set by
    // parse_common_args before defaults() zeroed it — but defaults only
    // applies when < 0, so if user passed -n, it stays).  We set default to
    // 0 above; if user passed -n it overrides.
    if (args.n_predict > 0) do_gen = true;

    if (args.help_requested) {
        std::printf("usage: %s -m MODEL --tests FILE [--layers 16-31] [-n N]\n", argv[0]);
        return 0;
    }

    if (!rp::require_model(args)) return 1;
    if (!rp::require_tests(args)) return 1;
    if (!rp::reject_extra(args)) return 1;

    // --layers is handled by parse_common_args into args.layer_range
    std::string layer_range = args.layer_range.empty() ? "16-31" : args.layer_range;

    std::vector<TestCase> tests;
    {
        json arr = rp::load_tests(args);
        for (const auto & e : arr) {
            tests.push_back({
                e.at("prompt_a").get<std::string>(),
                e.at("prompt_b").get<std::string>(),
                e.at("expect_a").get<std::string>(),
                e.at("expect_b").get<std::string>(),
                e.value("name", ""),
            });
        }
    }

    return rp::run_experiment([&] {
        auto env = rp::make_env(args);
        auto & model = env->model;
        auto & ctx   = env->ctx;
        const auto & geom  = env->geom;
        const auto * vocab  = env->vocab;
        int hs = geom.head_size;
        int n_head = geom.n_head;

        auto layers = rp::parse_layer_range(layer_range, geom.n_layer);

        std::printf("layers: %d-%d (%zu layers)  n_head=%d  head_size=%d  total_heads=%zu\n",
                    layers.front(), layers.back(), layers.size(), n_head, hs,
                    layers.size() * n_head);

        std::printf("\n%-12s  %8s  %8s  %8s  %8s  %8s  %8s  %8s  %8s\n",
                    "test", "base_A", "base_B", "full_Δ", "rank1", "rank2", "rank4",
                    "erase_r1", "erase_full");
        std::printf("%s\n", std::string(100, '-').c_str());

        for (const auto & tc : tests) {
            std::string label = tc.name.empty() ? "?" : tc.name;

            llama_token tok_a = rp::find_token(vocab, tc.expect_a);
            llama_token tok_b = rp::find_token(vocab, tc.expect_b);

            auto toks_a = rp::tokenize(vocab, tc.prompt_a);
            auto toks_b = rp::tokenize(vocab, tc.prompt_b);

            if (toks_a.size() < 2 || toks_b.size() < 2) continue;

            // ── capture prefix states ───────────────────────────────
            ctx.clear_memory();
            ctx.decode(rp::span<const llama_token>(toks_a.data(), toks_a.size() - 1));
            rp::StateBuf state_a(geom);
            state_a.load_from(ctx.raw());

            ctx.clear_memory();
            ctx.decode(rp::span<const llama_token>(toks_b.data(), toks_b.size() - 1));
            rp::StateBuf state_b(geom);
            state_b.load_from(ctx.raw());

            // ── baselines (full decode) ─────────────────────────────
            ctx.clear_memory();
            ctx.decode(rp::span<const llama_token>(toks_a.data(), toks_a.size()));
            const float * la = llama_get_logits(ctx.raw());
            float pa_base = rp::prob_of(la[tok_a], la[tok_b]);

            ctx.clear_memory();
            ctx.decode(rp::span<const llama_token>(toks_b.data(), toks_b.size()));
            const float * lb = llama_get_logits(ctx.raw());
            float pb_base = rp::prob_of(lb[tok_a], lb[tok_b]);

            // ── REWRITE: state_A + delta -> should match state_B ─────
            // full delta (rank=0)
            rp::StateBuf full_edit(state_a);
            rp::apply_full_delta(full_edit, state_a, state_b, layers, 1.0);
            float pa_full = rp::measure(ctx, full_edit, toks_a, tok_a, tok_b);

            // rank-1
            rp::StateBuf r1_edit(state_a);
            rp::apply_rank_delta(r1_edit, state_a, state_b, layers, 1, 1.0);
            float pa_r1 = rp::measure(ctx, r1_edit, toks_a, tok_a, tok_b);

            // rank-2
            rp::StateBuf r2_edit(state_a);
            rp::apply_rank_delta(r2_edit, state_a, state_b, layers, 2, 1.0);
            float pa_r2 = rp::measure(ctx, r2_edit, toks_a, tok_a, tok_b);

            // rank-4
            rp::StateBuf r4_edit(state_a);
            rp::apply_rank_delta(r4_edit, state_a, state_b, layers, 4, 1.0);
            float pa_r4 = rp::measure(ctx, r4_edit, toks_a, tok_a, tok_b);

            // ── ERASE: state_B - delta -> should weaken B ────────────
            rp::StateBuf erase_r1(state_b);
            rp::apply_rank_delta(erase_r1, state_a, state_b, layers, 1, -1.0);
            float pa_erase_r1 = rp::measure(ctx, erase_r1, toks_b, tok_a, tok_b);

            rp::StateBuf erase_full(state_b);
            rp::apply_full_delta(erase_full, state_a, state_b, layers, -1.0);
            float pa_erase_full = rp::measure(ctx, erase_full, toks_b, tok_a, tok_b);

            std::printf("%-12s  %8.4f  %8.4f  %8.4f  %8.4f  %8.4f  %8.4f  %8.4f  %8.4f\n",
                        label.c_str(),
                        pa_base, pb_base,
                        pa_full, pa_r1, pa_r2, pa_r4,
                        pa_erase_r1, pa_erase_full);
            std::fflush(stdout);

            // ── generation if requested ─────────────────────────────
            if (do_gen && args.n_predict > 0) {
                std::printf("\n");
                std::printf("======================================================================\n");
                std::printf("  TEST: %s  —  rewriting \"%s\" → \"%s\"\n",
                            label.c_str(), tc.expect_a.c_str(), tc.expect_b.c_str());
                std::printf("  Prompt A: \"%s\"\n", tc.prompt_a.c_str());
                std::printf("  Prompt B: \"%s\"\n", tc.prompt_b.c_str());
                std::printf("  P(%s): base_A=%.4f  base_B=%.4f\n",
                            tc.expect_a.c_str(), pa_base, pb_base);
                std::printf("======================================================================\n\n");

                auto show = [](const char * label, const char * prompt,
                              const char * expect, float pa,
                              const std::string & gen) {
                    std::printf("  [%s]  P(%s)=%.4f\n", label, expect, pa);
                    std::printf("  prompt: \"%s...\"\n", prompt);
                    std::printf("  output: %s\n\n", gen.c_str());
                };

                show("BASELINE A — unedited", tc.prompt_a.c_str(),
                     tc.expect_a.c_str(), pa_base,
                     rp::generate_from(model, ctx, state_a, toks_a, args.n_predict, args.seed));

                show("BASELINE B — unedited", tc.prompt_b.c_str(),
                     tc.expect_a.c_str(), pb_base,
                     rp::generate_from(model, ctx, state_b, toks_b, args.n_predict, args.seed));

                show("RANK-1 REWRITE — prompt A state edited toward B",
                     tc.prompt_a.c_str(), tc.expect_a.c_str(), pa_r1,
                     rp::generate_from(model, ctx, r1_edit, toks_a, args.n_predict, args.seed));

                show("RANK-2 REWRITE — prompt A state edited toward B",
                     tc.prompt_a.c_str(), tc.expect_a.c_str(), pa_r2,
                     rp::generate_from(model, ctx, r2_edit, toks_a, args.n_predict, args.seed));

                show("RANK-4 REWRITE — prompt A state edited toward B",
                     tc.prompt_a.c_str(), tc.expect_a.c_str(), pa_r4,
                     rp::generate_from(model, ctx, r4_edit, toks_a, args.n_predict, args.seed));

                show("FULL REWRITE — prompt A + exact state delta toward B",
                     tc.prompt_a.c_str(), tc.expect_a.c_str(), pa_full,
                     rp::generate_from(model, ctx, full_edit, toks_a, args.n_predict, args.seed));

                show("RANK-1 ERASE — prompt B with rank-1 fact component removed",
                     tc.prompt_b.c_str(), tc.expect_a.c_str(), pa_erase_r1,
                     rp::generate_from(model, ctx, erase_r1, toks_b, args.n_predict, args.seed));

                std::fflush(stdout);
            }
        }

        std::printf("\nP(a) values. base_A should be high, base_B low.\n");
        std::printf("full_delta = full state swap at target layers (gold standard).\n");
        std::printf("rank1/2/4 = rank-k approximation of per-head delta.\n");
        std::printf("erase = state_B minus delta (should push toward A).\n");

        std::fprintf(stderr, "\ndone\n");
    });
}
