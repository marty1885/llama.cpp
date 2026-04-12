// experiments/state-causal-trace.cpp — causal tracing on recurrent state.
//
// Tests whether facts are localized or distributed across layers in the
// recurrent state. Runs two prompts that differ in one fact (e.g., "red"
// vs "green"), captures their states, then for each layer L patches layer L
// from state_B into state_A and checks how the logits and generation change.
//
// For each patched layer, prints:
//   - Logit lens: top-5 predicted tokens + logits for expect_a / expect_b
//   - Full generated text for manual inspection
//
// Usage:
//   llama-rwkv-state-causal-trace -m model.gguf \
//       --tests causal_trace_tests.json -n 30 [--seed 42] [--top-k 10]
//
// Single-pair mode:
//   llama-rwkv-state-causal-trace -m model.gguf \
//       --prompt-a "..." --prompt-b "..." --expect-a red --expect-b green -n 30
//
// JSON format:
//   [{"prompt_a": "...", "prompt_b": "...", "expect_a": "red", "expect_b": "green",
//     "name": "hat_color"}]

#include "rwkv_probe/state.h"
#include "rwkv_probe/generate.h"
#include "rwkv_probe/experiment.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
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

struct TokenLogit {
    llama_token tok;
    float       logit;
};

// get top-k tokens from logits
static std::vector<TokenLogit> top_k_logits(const float * logits, int n_vocab, int k) {
    std::vector<int> indices(n_vocab);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                      [logits](int a, int b) { return logits[a] > logits[b]; });

    std::vector<TokenLogit> result;
    for (int i = 0; i < k && i < n_vocab; ++i) {
        result.push_back({(llama_token) indices[i], logits[indices[i]]});
    }
    return result;
}

// print logit lens for a given state
static void print_logit_lens(llama_context * ctx, const llama_vocab * /* vocab */,
                             int n_vocab, int top_k,
                             llama_token tok_a, llama_token tok_b,
                             const std::string & expect_a_str,
                             const std::string & expect_b_str) {
    const float * logits = llama_get_logits(ctx);

    auto top = top_k_logits(logits, n_vocab, top_k);

    float logit_a = (tok_a >= 0 && tok_a < n_vocab) ? logits[tok_a] : -999.0f;
    float logit_b = (tok_b >= 0 && tok_b < n_vocab) ? logits[tok_b] : -999.0f;
    float p_a = rp::prob_of(logit_a, logit_b);

    std::printf("    logit(%s)=%.2f  logit(%s)=%.2f  P(%s|{%s,%s})=%.3f\n",
                expect_a_str.c_str(), logit_a,
                expect_b_str.c_str(), logit_b,
                expect_a_str.c_str(),
                expect_a_str.c_str(), expect_b_str.c_str(), p_a);

    std::printf("    top-%d:", top_k);
    for (const auto & tl : top) {
        std::string piece = rp::piece(ctx, tl.tok);
        // escape newlines/tabs for display
        for (auto & c : piece) {
            if (c == '\n') c = ' ';
            if (c == '\t') c = ' ';
        }
        std::printf("  [%s %.2f]", piece.c_str(), tl.logit);
    }
    std::printf("\n");
}

int main(int argc, char ** argv) {
    auto args = rp::parse_common_args(argc, argv);
    args.defaults(2048, 30);

    // experiment-specific args
    std::string prompt_a, prompt_b, expect_a, expect_b;
    int  top_k     = 5;
    bool include_r = true;

    for (std::size_t i = 0; i < args.extra.size(); ++i) {
        const std::string & a = args.extra[i];
        if      (a == "--prompt-a"  && i+1 < args.extra.size()) prompt_a   = args.extra[++i];
        else if (a == "--prompt-b"  && i+1 < args.extra.size()) prompt_b   = args.extra[++i];
        else if (a == "--expect-a"  && i+1 < args.extra.size()) expect_a   = args.extra[++i];
        else if (a == "--expect-b"  && i+1 < args.extra.size()) expect_b   = args.extra[++i];
        else if (a == "--top-k"     && i+1 < args.extra.size()) top_k      = std::atoi(args.extra[++i].c_str());
        else if (a == "--s-only")                                include_r  = false;
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 1; }
    }

    if (args.help_requested) {
        std::printf(
            "usage: %s -m MODEL [--tests FILE | --prompt-a A --prompt-b B --expect-a EA --expect-b EB]\n"
            "       [-n N] [--seed S] [--top-k K] [--s-only]\n", argv[0]);
        return 0;
    }

    if (!rp::require_model(args)) return 1;

    std::vector<TestCase> tests;
    if (!args.tests_path.empty()) {
        auto arr = rp::load_tests(args);
        for (const auto & e : arr) {
            tests.push_back({
                e.at("prompt_a").get<std::string>(),
                e.at("prompt_b").get<std::string>(),
                e.at("expect_a").get<std::string>(),
                e.at("expect_b").get<std::string>(),
                e.value("name", ""),
            });
        }
    } else if (!prompt_a.empty() && !prompt_b.empty()) {
        tests.push_back({prompt_a, prompt_b, expect_a, expect_b, "cli"});
    } else {
        std::fprintf(stderr, "error: provide --prompt-a/--prompt-b or --tests\n");
        return 1;
    }

    return rp::run_experiment([&] {
        auto env = rp::make_env(args);
        auto & model = env->model;
        auto & ctx   = env->ctx;
        const auto & geom  = env->geom;
        const auto * vocab = env->vocab;
        int n_vocab = geom.n_vocab;

        std::fprintf(stderr, "n_layer=%d  n_embd=%d  n_vocab=%d  include_r=%d  top_k=%d\n",
                     geom.n_layer, geom.n_embd, n_vocab, (int) include_r, top_k);

        for (std::size_t ti = 0; ti < tests.size(); ++ti) {
            const auto & tc = tests[ti];
            std::string label = tc.name.empty()
                ? ("test_" + std::to_string(ti))
                : tc.name;

            // find token IDs for expect strings
            llama_token tok_a = rp::find_token(vocab, tc.expect_a);
            llama_token tok_b = rp::find_token(vocab, tc.expect_b);

            std::printf("\n======================================================================\n");
            std::printf("TEST: %s\n", label.c_str());
            std::printf("  A: %s\n", tc.prompt_a.c_str());
            std::printf("  B: %s\n", tc.prompt_b.c_str());
            std::printf("  expect_a='%s' (tok %d)  expect_b='%s' (tok %d)\n",
                        tc.expect_a.c_str(), tok_a, tc.expect_b.c_str(), tok_b);

            if (tok_a < 0 || tok_b < 0) {
                std::printf("  WARNING: could not find token for expect string\n");
            }

            auto toks_a = rp::tokenize(vocab, tc.prompt_a);
            auto toks_b = rp::tokenize(vocab, tc.prompt_b);

            if (toks_a.size() < 2 || toks_b.size() < 2) {
                std::printf("  need at least 2 tokens per prompt, skipping\n");
                continue;
            }

            std::fprintf(stderr, "  toks_a=%zu  toks_b=%zu\n", toks_a.size(), toks_b.size());

            // ── baseline A ──────────────────────────────────────────────
            ctx.clear_memory();
            if (ctx.decode(rp::span<const llama_token>(toks_a.data(), toks_a.size())) != 0) {
                std::fprintf(stderr, "  decode A failed\n"); continue;
            }
            rp::StateBuf state_a(geom);
            state_a.load_from(ctx.raw());

            std::printf("\n  BASELINE A:\n");
            print_logit_lens(ctx.raw(), vocab, n_vocab, top_k,
                             tok_a, tok_b, tc.expect_a, tc.expect_b);

            std::string gen_a;
            {
                common_params_sampling sp;
                sp.seed = args.seed;
                rp::Generator gen(model, ctx, sp);
                gen.accept_prompt(rp::span<const llama_token>(toks_a.data(), toks_a.size()));
                auto r = gen.run(args.n_predict);
                gen_a = r.text;
            }
            std::printf("    gen: %s\n", gen_a.substr(0, 200).c_str());

            // ── control 1: pure split (no state roundtrip) ──
            {
                ctx.clear_memory();
                if (ctx.decode(rp::span<const llama_token>(
                        toks_a.data(), toks_a.size() - 1)) != 0) {
                    std::fprintf(stderr, "  pure split: prefix decode failed\n");
                } else if (ctx.decode_one(toks_a.back()) != 0) {
                    std::fprintf(stderr, "  pure split: decode_one failed\n");
                } else {
                    std::printf("\n  PURE SPLIT (A[0:%zu] then A[%zu], no roundtrip):\n",
                                 toks_a.size() - 1, toks_a.size() - 1);
                    print_logit_lens(ctx.raw(), vocab, n_vocab, top_k,
                                     tok_a, tok_b, tc.expect_a, tc.expect_b);
                }
            }

            // ── control 2: split + state roundtrip ──
            {
                ctx.clear_memory();
                if (ctx.decode(rp::span<const llama_token>(
                        toks_a.data(), toks_a.size() - 1)) != 0) {
                    std::fprintf(stderr, "  roundtrip: prefix decode failed\n");
                } else {
                    rp::StateBuf roundtrip(geom);
                    roundtrip.load_from(ctx.raw());

                    // verify roundtrip fidelity: compare loaded state to itself
                    rp::StateBuf verify(geom);
                    verify.load_from(ctx.raw());

                    double max_diff_r = 0.0, max_diff_s = 0.0;
                    auto r1 = roundtrip.r_flat();
                    auto r2 = verify.r_flat();
                    for (std::size_t j = 0; j < r1.size(); ++j) {
                        double d = std::fabs((double)r1[j] - (double)r2[j]);
                        if (d > max_diff_r) max_diff_r = d;
                    }
                    auto s1 = roundtrip.s_flat();
                    auto s2 = verify.s_flat();
                    for (std::size_t j = 0; j < s1.size(); ++j) {
                        double d = std::fabs((double)s1[j] - (double)s2[j]);
                        if (d > max_diff_s) max_diff_s = d;
                    }
                    std::fprintf(stderr, "  load consistency: max_diff r=%e s=%e\n",
                                 max_diff_r, max_diff_s);

                    roundtrip.store_to(ctx.raw());

                    // verify store_to fidelity: load again and compare
                    rp::StateBuf after_store(geom);
                    after_store.load_from(ctx.raw());
                    double max_diff_r2 = 0.0, max_diff_s2 = 0.0;
                    auto r3 = roundtrip.r_flat();
                    auto r4 = after_store.r_flat();
                    for (std::size_t j = 0; j < r3.size(); ++j) {
                        double d = std::fabs((double)r3[j] - (double)r4[j]);
                        if (d > max_diff_r2) max_diff_r2 = d;
                    }
                    auto s3 = roundtrip.s_flat();
                    auto s4 = after_store.s_flat();
                    for (std::size_t j = 0; j < s3.size(); ++j) {
                        double d = std::fabs((double)s3[j] - (double)s4[j]);
                        if (d > max_diff_s2) max_diff_s2 = d;
                    }
                    std::fprintf(stderr, "  store roundtrip: max_diff r=%e s=%e\n",
                                 max_diff_r2, max_diff_s2);

                    if (ctx.decode_one(toks_a.back()) != 0) {
                        std::fprintf(stderr, "  roundtrip: decode_one failed\n");
                    } else {
                        std::printf("\n  ROUNDTRIP (A[0:%zu] + load/store + A[%zu]):\n",
                                     toks_a.size() - 1, toks_a.size() - 1);
                        print_logit_lens(ctx.raw(), vocab, n_vocab, top_k,
                                         tok_a, tok_b, tc.expect_a, tc.expect_b);
                    }
                }
            }

            // ── baseline B (full, for display) ────────────────────────
            ctx.clear_memory();
            if (ctx.decode(rp::span<const llama_token>(toks_b.data(), toks_b.size())) != 0) {
                std::fprintf(stderr, "  decode B failed\n"); continue;
            }

            std::printf("\n  BASELINE B:\n");
            print_logit_lens(ctx.raw(), vocab, n_vocab, top_k,
                             tok_a, tok_b, tc.expect_a, tc.expect_b);

            std::string gen_b;
            {
                common_params_sampling sp;
                sp.seed = args.seed;
                rp::Generator gen(model, ctx, sp);
                gen.accept_prompt(rp::span<const llama_token>(toks_b.data(), toks_b.size()));
                auto r = gen.run(args.n_predict);
                gen_b = r.text;
            }
            std::printf("    gen: %s\n", gen_b.substr(0, 200).c_str());

            // ── state_b for patching: captured at all-but-last of prompt B ──
            // This way state_b and the per-layer patching states are at the
            // same time point (just before the last shared token).
            ctx.clear_memory();
            if (ctx.decode(rp::span<const llama_token>(
                    toks_b.data(), toks_b.size() - 1)) != 0) {
                std::fprintf(stderr, "  decode B-prefix failed\n"); continue;
            }
            rp::StateBuf state_b(geom);
            state_b.load_from(ctx.raw());
            std::fprintf(stderr, "  state_b captured at prefix (toks_b[0:%zu])\n",
                         toks_b.size() - 1);

            // ── per-layer patching: A base + layer L from B ─────────────
            std::printf("\n  LAYER SWEEP (patching A <- B[layer]):\n");
            std::printf("  %5s  %8s  %8s  %7s  %s\n",
                        "layer", "logit_a", "logit_b", "P(a)",  "generation");
            std::printf("  %s\n", std::string(80, '-').c_str());

            for (int il = 0; il < geom.n_layer; ++il) {
                // To get logits from a patched state, we need to:
                //   1. Decode all-but-last tokens of prompt A (creates sequence)
                //   2. Patch the state in-place
                //   3. Decode last token (computes logits using patched state)
                ctx.clear_memory();

                // decode all but last token
                if (toks_a.size() > 1) {
                    if (ctx.decode(rp::span<const llama_token>(
                            toks_a.data(), toks_a.size() - 1)) != 0) {
                        std::fprintf(stderr, "  L%d: decode prefix failed\n", il);
                        continue;
                    }
                }

                // patch state in-place: load current, overwrite layer il from B, store back
                rp::StateBuf current(geom);
                current.load_from(ctx.raw());
                int layer_arr[] = {il};
                current.copy_layers_from(state_b,
                    rp::span<const int>(layer_arr, 1), include_r);
                current.store_to(ctx.raw());

                // decode last token with patched state → fresh logits
                llama_token last_tok = toks_a.back();
                if (ctx.decode_one(last_tok) != 0) {
                    std::fprintf(stderr, "  L%d: decode_one failed\n", il);
                    continue;
                }

                // read logits
                const float * logits = llama_get_logits(ctx.raw());
                float la = (tok_a >= 0 && tok_a < n_vocab) ? logits[tok_a] : -999.0f;
                float lb = (tok_b >= 0 && tok_b < n_vocab) ? logits[tok_b] : -999.0f;
                float pa = rp::prob_of(la, lb);

                // generate
                std::string gen_text;
                {
                    common_params_sampling sp;
                    sp.seed = args.seed;
                    rp::Generator gen(model, ctx, sp);
                    llama_token arr[] = {last_tok};
                    gen.accept_prompt(rp::span<const llama_token>(arr, 1));
                    auto r = gen.run(args.n_predict);
                    gen_text = r.text;
                }

                // one-line summary
                std::string short_gen = gen_text.substr(0, 60);
                for (auto & c : short_gen) {
                    if (c == '\n') c = ' ';
                }

                std::printf("  L%-4d  %8.2f  %8.2f  %6.3f  %s\n",
                            il, la, lb, pa, short_gen.c_str());
                std::fflush(stdout);
            }

            // ── block patching: patch contiguous layer ranges ─────────
            // answers: how many layers must be patched to flip the answer?
            struct Block { const char * name; int lo; int hi; };  // [lo, hi)
            int nl = geom.n_layer;
            int q1 = nl / 4, q2 = nl / 2, q3 = 3 * nl / 4;
            Block blocks[] = {
                // growing from start
                {"L0-7",        0,   q1},
                {"L0-15",       0,   q2},
                {"L0-23",       0,   q3},
                // growing from end
                {"L24-31",      q3,  nl},
                {"L16-31",      q2,  nl},
                {"L8-31",       q1,  nl},
                // quarters
                {"L8-15",       q1,  q2},
                {"L16-23",      q2,  q3},
                // halves
                {"even",        -1,  -1},  // sentinel: even layers only
                {"odd",         -2,  -2},  // sentinel: odd layers only
                // full swap (sanity check)
                {"ALL",         0,   nl},
            };

            std::printf("\n  BLOCK PATCHING (A <- B[block]):\n");
            std::printf("  %-10s  %8s  %8s  %7s  %s\n",
                        "block", "logit_a", "logit_b", "P(a)", "generation");
            std::printf("  %s\n", std::string(80, '-').c_str());

            for (const auto & blk : blocks) {
                // build layer list
                std::vector<int> layers;
                if (blk.lo == -1) {
                    // even layers
                    for (int l = 0; l < nl; l += 2) layers.push_back(l);
                } else if (blk.lo == -2) {
                    // odd layers
                    for (int l = 1; l < nl; l += 2) layers.push_back(l);
                } else {
                    for (int l = blk.lo; l < blk.hi; ++l) layers.push_back(l);
                }

                ctx.clear_memory();
                if (toks_a.size() > 1) {
                    if (ctx.decode(rp::span<const llama_token>(
                            toks_a.data(), toks_a.size() - 1)) != 0) continue;
                }

                rp::StateBuf current(geom);
                current.load_from(ctx.raw());
                current.copy_layers_from(state_b,
                    rp::span<const int>(layers.data(), layers.size()), include_r);
                current.store_to(ctx.raw());

                llama_token last_tok = toks_a.back();
                if (ctx.decode_one(last_tok) != 0) continue;

                const float * logits = llama_get_logits(ctx.raw());
                float la = (tok_a >= 0 && tok_a < n_vocab) ? logits[tok_a] : -999.0f;
                float lb = (tok_b >= 0 && tok_b < n_vocab) ? logits[tok_b] : -999.0f;
                float pa = rp::prob_of(la, lb);

                std::string gen_text;
                {
                    common_params_sampling sp;
                    sp.seed = args.seed;
                    rp::Generator gen(model, ctx, sp);
                    llama_token arr[] = {last_tok};
                    gen.accept_prompt(rp::span<const llama_token>(arr, 1));
                    auto r = gen.run(args.n_predict);
                    gen_text = r.text;
                }

                std::string short_gen = gen_text.substr(0, 60);
                for (auto & c : short_gen) {
                    if (c == '\n') c = ' ';
                }

                std::printf("  %-10s  %8.2f  %8.2f  %6.3f  %s\n",
                            blk.name, la, lb, pa, short_gen.c_str());

                // print top-k for this block
                print_logit_lens(ctx.raw(), vocab, n_vocab, top_k,
                                 tok_a, tok_b, tc.expect_a, tc.expect_b);
                std::fflush(stdout);
            }
        }

        std::fprintf(stderr, "\ndone\n");
    });
}
