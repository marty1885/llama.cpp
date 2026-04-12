// experiments/state-head-trace.cpp — per-head causal tracing on recurrent state.
//
// For a given layer (default L29), patches individual WKV heads from
// prompt B's state into prompt A's state, then checks logits.
// Answers: which heads encode which facts?
//
// Usage:
//   llama-rwkv-state-head-trace -m model.gguf \
//       --tests causal_trace_tests.json --layer 29 -n 30 --top-k 10
//
// Also tests two-fact disambiguation:
//   For prompts with two entities (e.g. Alice=red, Bob=blue vs Alice=green, Bob=yellow),
//   checks whether patching a head flips one entity without the other.

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

static std::vector<TokenLogit> top_k_logits(const float * logits, int n_vocab, int k) {
    std::vector<int> idx(n_vocab);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [logits](int a, int b) { return logits[a] > logits[b]; });
    std::vector<TokenLogit> out;
    for (int i = 0; i < k && i < n_vocab; ++i) {
        out.push_back({(llama_token) idx[i], logits[idx[i]]});
    }
    return out;
}

// patch a single head's S state from src into dst
static void copy_head(rp::StateBuf & dst, const rp::StateBuf & src, int layer, int head) {
    auto d = dst.s_head(layer, head);
    auto s = src.s_head(layer, head);
    for (std::size_t i = 0; i < d.size(); ++i) {
        d[i] = s[i];
    }
}

// patch a set of heads
static void copy_heads(rp::StateBuf & dst, const rp::StateBuf & src,
                       int layer, const std::vector<int> & heads) {
    for (int h : heads) {
        copy_head(dst, src, layer, h);
    }
}

// L2 distance between two head states
static double head_distance(const rp::StateBuf & a, const rp::StateBuf & b,
                            int layer, int head) {
    auto ha = a.s_head(layer, head);
    auto hb = b.s_head(layer, head);
    double sum = 0.0;
    for (std::size_t i = 0; i < ha.size(); ++i) {
        double d = (double)ha[i] - (double)hb[i];
        sum += d * d;
    }
    return std::sqrt(sum);
}

int main(int argc, char ** argv) {
    auto args = rp::parse_common_args(argc, argv);
    args.defaults(2048, 30);

    // experiment-specific args
    std::string prompt_a, prompt_b, expect_a, expect_b;
    int  target_layer = 29;
    int  top_k        = 10;

    for (std::size_t i = 0; i < args.extra.size(); ++i) {
        const std::string & a = args.extra[i];
        if      (a == "--prompt-a" && i+1 < args.extra.size()) prompt_a     = args.extra[++i];
        else if (a == "--prompt-b" && i+1 < args.extra.size()) prompt_b     = args.extra[++i];
        else if (a == "--expect-a" && i+1 < args.extra.size()) expect_a     = args.extra[++i];
        else if (a == "--expect-b" && i+1 < args.extra.size()) expect_b     = args.extra[++i];
        else if (a == "--layer"    && i+1 < args.extra.size()) target_layer = std::atoi(args.extra[++i].c_str());
        else if (a == "--top-k"    && i+1 < args.extra.size()) top_k        = std::atoi(args.extra[++i].c_str());
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 1; }
    }

    if (args.help_requested) {
        std::printf("usage: %s -m MODEL [--tests FILE | --prompt-a/b --expect-a/b]\n"
                    "       [--layer L] [-n N] [--seed S] [--top-k K]\n", argv[0]);
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
        std::fprintf(stderr, "error: provide --tests or --prompt-a/--prompt-b\n");
        return 1;
    }

    return rp::run_experiment([&] {
        auto env = rp::make_env(args);
        auto & model = env->model;
        auto & ctx   = env->ctx;
        const auto & geom  = env->geom;
        const auto * vocab = env->vocab;
        int n_vocab = geom.n_vocab;
        int n_head  = geom.n_head;
        int head_sz = geom.head_size;

        if (target_layer < 0 || target_layer >= geom.n_layer) {
            throw std::runtime_error("layer " + std::to_string(target_layer) +
                                     " out of range [0, " + std::to_string(geom.n_layer) + ")");
        }

        std::fprintf(stderr, "target layer: L%d  n_head=%d  head_size=%d  head_state=%d floats\n",
                     target_layer, n_head, head_sz, head_sz * head_sz);

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

            std::printf("\n======================================================================\n");
            std::printf("TEST: %s  (L%d, %d heads × %d×%d)\n",
                        label.c_str(), target_layer, n_head, head_sz, head_sz);
            std::printf("  A: %s\n", tc.prompt_a.c_str());
            std::printf("  B: %s\n", tc.prompt_b.c_str());
            std::printf("  expect_a='%s' (tok %d)  expect_b='%s' (tok %d)\n",
                        tc.expect_a.c_str(), tok_a, tc.expect_b.c_str(), tok_b);

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

            // ── baselines (full prompt decode) ──────────────────────────
            ctx.clear_memory();
            if (ctx.decode(rp::span<const llama_token>(toks_a.data(), toks_a.size())) != 0) {
                std::fprintf(stderr, "  decode A full failed\n"); continue;
            }
            const float * logits_a = llama_get_logits(ctx.raw());
            float base_la = (tok_a >= 0) ? logits_a[tok_a] : -999.f;
            float base_lb = (tok_b >= 0) ? logits_a[tok_b] : -999.f;

            std::string gen_a;
            {
                common_params_sampling sp; sp.seed = args.seed;
                rp::Generator gen(model, ctx, sp);
                gen.accept_prompt(rp::span<const llama_token>(toks_a.data(), toks_a.size()));
                gen_a = gen.run(args.n_predict).text;
            }

            std::printf("\n  BASELINE A: logit(%s)=%.2f  logit(%s)=%.2f  P(%s)=%.3f\n",
                        tc.expect_a.c_str(), base_la, tc.expect_b.c_str(), base_lb,
                        tc.expect_a.c_str(), rp::prob_of(base_la, base_lb));
            std::printf("    gen: %s\n", gen_a.substr(0, 120).c_str());

            ctx.clear_memory();
            if (ctx.decode(rp::span<const llama_token>(toks_b.data(), toks_b.size())) != 0) {
                std::fprintf(stderr, "  decode B full failed\n"); continue;
            }
            const float * logits_b = llama_get_logits(ctx.raw());
            float bbase_la = (tok_a >= 0) ? logits_b[tok_a] : -999.f;
            float bbase_lb = (tok_b >= 0) ? logits_b[tok_b] : -999.f;

            std::string gen_b;
            {
                common_params_sampling sp; sp.seed = args.seed;
                rp::Generator gen(model, ctx, sp);
                gen.accept_prompt(rp::span<const llama_token>(toks_b.data(), toks_b.size()));
                gen_b = gen.run(args.n_predict).text;
            }

            std::printf("  BASELINE B: logit(%s)=%.2f  logit(%s)=%.2f  P(%s)=%.3f\n",
                        tc.expect_a.c_str(), bbase_la, tc.expect_b.c_str(), bbase_lb,
                        tc.expect_a.c_str(), rp::prob_of(bbase_la, bbase_lb));
            std::printf("    gen: %s\n", gen_b.substr(0, 120).c_str());

            // ── per-head distances ──────────────────────────────────────
            std::printf("\n  HEAD DISTANCES (L%d, state_a vs state_b):\n", target_layer);
            std::printf("  %4s  %12s\n", "head", "L2_dist");
            for (int h = 0; h < n_head; ++h) {
                double d = head_distance(state_a, state_b, target_layer, h);
                std::printf("  H%-3d  %12.4f\n", h, d);
            }

            // ── per-head patching ───────────────────────────────────────
            std::printf("\n  PER-HEAD PATCHING (L%d, A <- B[head]):\n", target_layer);
            std::printf("  %4s  %8s  %8s  %7s  %s\n",
                        "head", "logit_a", "logit_b", "P(a)", "generation");
            std::printf("  %s\n", std::string(80, '-').c_str());

            for (int h = 0; h < n_head; ++h) {
                // patch single head
                rp::StateBuf patched(state_a);
                copy_head(patched, state_b, target_layer, h);

                ctx.clear_memory();
                if (ctx.decode(rp::span<const llama_token>(
                        toks_a.data(), toks_a.size() - 1)) != 0) continue;
                patched.store_to(ctx.raw());
                if (ctx.decode_one(toks_a.back()) != 0) continue;

                const float * logits = llama_get_logits(ctx.raw());
                float la = (tok_a >= 0) ? logits[tok_a] : -999.f;
                float lb = (tok_b >= 0) ? logits[tok_b] : -999.f;
                float pa = rp::prob_of(la, lb);

                std::string gen_text;
                {
                    common_params_sampling sp; sp.seed = args.seed;
                    rp::Generator gen(model, ctx, sp);
                    llama_token arr[] = {toks_a.back()};
                    gen.accept_prompt(rp::span<const llama_token>(arr, 1));
                    gen_text = gen.run(args.n_predict).text;
                }

                std::string short_gen = gen_text.substr(0, 60);
                for (auto & c : short_gen) if (c == '\n') c = ' ';

                std::printf("  H%-3d  %8.2f  %8.2f  %6.3f  %s\n",
                            h, la, lb, pa, short_gen.c_str());
                std::fflush(stdout);
            }

            // ── cumulative head patching (sorted by distance) ───────────
            // patch heads one at a time in decreasing order of L2 distance
            std::vector<int> head_order(n_head);
            std::iota(head_order.begin(), head_order.end(), 0);
            std::sort(head_order.begin(), head_order.end(),
                      [&](int a, int b) {
                          return head_distance(state_a, state_b, target_layer, a)
                               > head_distance(state_a, state_b, target_layer, b);
                      });

            std::printf("\n  CUMULATIVE PATCHING (L%d, heads added by decreasing distance):\n",
                        target_layer);
            std::printf("  %4s  %12s  %8s  %8s  %7s  %s\n",
                        "n", "added_head", "logit_a", "logit_b", "P(a)", "generation");
            std::printf("  %s\n", std::string(85, '-').c_str());

            std::vector<int> patched_heads;
            for (int hi = 0; hi < n_head; ++hi) {
                int h = head_order[hi];
                patched_heads.push_back(h);

                rp::StateBuf patched(state_a);
                copy_heads(patched, state_b, target_layer, patched_heads);

                ctx.clear_memory();
                if (ctx.decode(rp::span<const llama_token>(
                        toks_a.data(), toks_a.size() - 1)) != 0) continue;
                patched.store_to(ctx.raw());
                if (ctx.decode_one(toks_a.back()) != 0) continue;

                const float * logits = llama_get_logits(ctx.raw());
                float la = (tok_a >= 0) ? logits[tok_a] : -999.f;
                float lb = (tok_b >= 0) ? logits[tok_b] : -999.f;
                float pa = rp::prob_of(la, lb);

                std::string gen_text;
                {
                    common_params_sampling sp; sp.seed = args.seed;
                    rp::Generator gen(model, ctx, sp);
                    llama_token arr[] = {toks_a.back()};
                    gen.accept_prompt(rp::span<const llama_token>(arr, 1));
                    gen_text = gen.run(args.n_predict).text;
                }

                std::string short_gen = gen_text.substr(0, 55);
                for (auto & c : short_gen) if (c == '\n') c = ' ';

                double dist = head_distance(state_a, state_b, target_layer, h);
                std::printf("  %2d    H%-3d (%6.2f)  %8.2f  %8.2f  %6.3f  %s\n",
                            hi + 1, h, dist, la, lb, pa, short_gen.c_str());
                std::fflush(stdout);
            }
        }

        std::fprintf(stderr, "\ndone\n");
    });
}
