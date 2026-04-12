// experiments/token-delta.cpp — per-token state delta editing.
//
// Instead of full-prompt deltas, captures the state change from a SINGLE
// diverging token. This uses the model's own k⊗v write mechanism with
// its learned gating — the k projection naturally handles entity×property
// addressing.
//
// Method:
//   1. Run shared prefix "Alice has a" → capture state_prefix
//   2. Decode "red" from state_prefix → capture state_after_red
//   3. Decode "blue" from state_prefix → capture state_after_blue
//   4. token_delta = state_after_blue - state_after_red
//   5. Apply token_delta to a full prompt's state
//
// Compares token-level delta vs full-prompt delta for selectivity.
//
// Usage:
//   llama-rwkv-token-delta -m model.gguf [--layers 16-31] [-n 15]

#include "rwkv_probe/state.h"
#include "rwkv_probe/experiment.h"
#include "rwkv_probe/edit.h"
#include "rwkv_probe/generate.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace rp = rwkv_probe;

// query: decode prefix, load state, decode query tokens, return P and gen
struct QResult {
    float p;
    std::string gen;
};

static QResult query(rp::Model & model, rp::Context & ctx,
                     const rp::StateBuf & state,
                     const std::string & prefix, const std::string & query_str,
                     llama_token tok_a, llama_token tok_b,
                     const llama_vocab * vocab, int n_predict, uint32_t seed) {
    auto ptoks = rp::tokenize(vocab, prefix);
    auto qtoks = rp::tokenize(vocab, query_str, false, false);

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
    args.defaults(/*ctx=*/2048, /*predict=*/15);
    if (args.layer_range.empty()) args.layer_range = "16-31";

    if (args.help_requested) {
        std::printf("usage: %s -m MODEL [--layers 16-31] [-n 15]\n", argv[0]);
        return 0;
    }
    if (!rp::require_model(args)) return 1;
    if (!rp::reject_extra(args)) return 1;

    return rp::run_experiment([&] {
        auto env = rp::make_env(args);
        auto & model = env->model;
        auto & ctx   = env->ctx;
        const auto & geom  = env->geom;
        const auto * vocab = env->vocab;
        auto layers = rp::parse_layer_range(args.layer_range, geom.n_layer);

        llama_token tok_red    = rp::find_token(vocab, "red");
        llama_token tok_yellow = rp::find_token(vocab, "yellow");
        llama_token tok_blue   = rp::find_token(vocab, "blue");
        llama_token tok_green  = rp::find_token(vocab, "green");
        llama_token tok_brown  = rp::find_token(vocab, "brown");
        llama_token tok_gray   = rp::find_token(vocab, "gray");

        // ════════════════════════════════════════════════════════════
        // TEST 1: hat color edit — token-level delta for "red" → "yellow"
        // ════════════════════════════════════════════════════════════
        {
            std::printf("======================================================================\n");
            std::printf("TEST 1: Alice hat red→yellow (token-level delta)\n");
            std::printf("  Context: 'Alice has a red hat and blue eyes. Bob has a green hat and brown eyes.'\n\n");

            // the prompts diverge at the hat-color token
            // shared prefix up to the color: "Alice has a "
            std::string shared_prefix = "Alice has a";
            auto shared_toks = rp::tokenize(vocab, shared_prefix);

            // capture state at the shared prefix
            ctx.clear_memory();
            ctx.decode(rp::span<const llama_token>(shared_toks.data(), shared_toks.size()));
            rp::StateBuf state_before_color(geom);
            state_before_color.load_from(ctx.raw());

            // decode " red" from this state
            rp::StateBuf state_after_red(geom);
            {
                // reload state to ensure clean start
                state_before_color.store_to(ctx.raw());
                llama_token t = rp::find_token(vocab, "red");
                ctx.decode_one(t);
                state_after_red.load_from(ctx.raw());
            }

            // decode " yellow" from same state
            rp::StateBuf state_after_yellow(geom);
            {
                state_before_color.store_to(ctx.raw());
                llama_token t = rp::find_token(vocab, "yellow");
                ctx.decode_one(t);
                state_after_yellow.load_from(ctx.raw());
            }

            // full-prompt states (for comparison)
            std::string prompt_a = "Alice has a red hat and blue eyes. Bob has a green hat and brown eyes.";
            std::string prompt_b = "Alice has a yellow hat and blue eyes. Bob has a green hat and brown eyes.";
            auto toks_a = rp::tokenize(vocab, prompt_a);
            auto toks_b = rp::tokenize(vocab, prompt_b);

            ctx.clear_memory();
            ctx.decode(rp::span<const llama_token>(toks_a.data(), toks_a.size()));
            rp::StateBuf state_full_a(geom);
            state_full_a.load_from(ctx.raw());

            ctx.clear_memory();
            ctx.decode(rp::span<const llama_token>(toks_b.data(), toks_b.size()));
            rp::StateBuf state_full_b(geom);
            state_full_b.load_from(ctx.raw());

            // queries
            std::string q_hat  = " What color is Alice's hat? Alice's hat is";
            std::string q_eyes = " What color are Alice's eyes? Alice's eyes are";

            // baselines
            auto bl_hat  = query(model, ctx, state_full_a, prompt_a, q_hat, tok_red, tok_yellow, vocab, args.n_predict, args.seed);
            auto bl_eyes = query(model, ctx, state_full_a, prompt_a, q_eyes, tok_blue, tok_green, vocab, args.n_predict, args.seed);

            std::printf("  BASELINES (unedited state_A):\n");
            std::printf("    hat:  P(red)=%.4f  gen: %s\n", bl_hat.p, bl_hat.gen.substr(0, 50).c_str());
            std::printf("    eyes: P(blue)=%.4f  gen: %s\n", bl_eyes.p, bl_eyes.gen.substr(0, 50).c_str());

            // full-prompt delta edit
            rp::StateBuf edited_full(state_full_a);
            rp::apply_full_delta(edited_full, state_full_a, state_full_b, layers);

            auto ef_hat  = query(model, ctx, edited_full, prompt_a, q_hat, tok_red, tok_yellow, vocab, args.n_predict, args.seed);
            auto ef_eyes = query(model, ctx, edited_full, prompt_a, q_eyes, tok_blue, tok_green, vocab, args.n_predict, args.seed);

            std::printf("\n  FULL-PROMPT DELTA (state_A + full_delta):\n");
            std::printf("    hat:  P(red)=%.4f  gen: %s\n", ef_hat.p, ef_hat.gen.substr(0, 50).c_str());
            std::printf("    eyes: P(blue)=%.4f  selectivity=%.4f\n", ef_eyes.p, ef_eyes.p - ef_hat.p);

            // token-level delta edit
            rp::StateBuf edited_token(state_full_a);
            rp::apply_full_delta(edited_token, state_after_red, state_after_yellow, layers);

            auto et_hat  = query(model, ctx, edited_token, prompt_a, q_hat, tok_red, tok_yellow, vocab, args.n_predict, args.seed);
            auto et_eyes = query(model, ctx, edited_token, prompt_a, q_eyes, tok_blue, tok_green, vocab, args.n_predict, args.seed);

            std::printf("\n  TOKEN-LEVEL DELTA (state_A + token_delta[red→yellow]):\n");
            std::printf("    hat:  P(red)=%.4f  gen: %s\n", et_hat.p, et_hat.gen.substr(0, 50).c_str());
            std::printf("    eyes: P(blue)=%.4f  selectivity=%.4f\n", et_eyes.p, et_eyes.p - et_hat.p);
        }

        // ════════════════════════════════════════════════════════════
        // TEST 2: eye color edit — token-level delta for "blue" → "green"
        // ════════════════════════════════════════════════════════════
        {
            std::printf("\n======================================================================\n");
            std::printf("TEST 2: Alice eyes blue→green (token-level delta)\n");
            std::printf("  Context: 'Alice has a red hat and blue eyes. Bob has a green hat and brown eyes.'\n\n");

            // shared prefix up to eye color: "Alice has a red hat and"
            std::string shared_prefix = "Alice has a red hat and";
            auto shared_toks = rp::tokenize(vocab, shared_prefix);

            ctx.clear_memory();
            ctx.decode(rp::span<const llama_token>(shared_toks.data(), shared_toks.size()));
            rp::StateBuf state_before_eyes(geom);
            state_before_eyes.load_from(ctx.raw());

            // decode " blue" from this state
            rp::StateBuf state_after_blue(geom);
            {
                state_before_eyes.store_to(ctx.raw());
                ctx.decode_one(rp::find_token(vocab, "blue"));
                state_after_blue.load_from(ctx.raw());
            }

            // decode " green" from same state
            rp::StateBuf state_after_green(geom);
            {
                state_before_eyes.store_to(ctx.raw());
                ctx.decode_one(rp::find_token(vocab, "green"));
                state_after_green.load_from(ctx.raw());
            }

            // full-prompt states
            std::string prompt_a = "Alice has a red hat and blue eyes. Bob has a green hat and brown eyes.";
            std::string prompt_b = "Alice has a red hat and green eyes. Bob has a green hat and brown eyes.";
            auto toks_a = rp::tokenize(vocab, prompt_a);
            auto toks_b = rp::tokenize(vocab, prompt_b);

            ctx.clear_memory();
            ctx.decode(rp::span<const llama_token>(toks_a.data(), toks_a.size()));
            rp::StateBuf state_full_a(geom);
            state_full_a.load_from(ctx.raw());

            ctx.clear_memory();
            ctx.decode(rp::span<const llama_token>(toks_b.data(), toks_b.size()));
            rp::StateBuf state_full_b(geom);
            state_full_b.load_from(ctx.raw());

            std::string q_hat  = " What color is Alice's hat? Alice's hat is";
            std::string q_eyes = " What color are Alice's eyes? Alice's eyes are";

            auto bl_hat  = query(model, ctx, state_full_a, prompt_a, q_hat, tok_red, tok_green, vocab, args.n_predict, args.seed);
            auto bl_eyes = query(model, ctx, state_full_a, prompt_a, q_eyes, tok_blue, tok_green, vocab, args.n_predict, args.seed);

            std::printf("  BASELINES (unedited state_A):\n");
            std::printf("    hat:  P(red)=%.4f  gen: %s\n", bl_hat.p, bl_hat.gen.substr(0, 50).c_str());
            std::printf("    eyes: P(blue)=%.4f  gen: %s\n", bl_eyes.p, bl_eyes.gen.substr(0, 50).c_str());

            // full-prompt delta
            rp::StateBuf edited_full(state_full_a);
            rp::apply_full_delta(edited_full, state_full_a, state_full_b, layers);

            auto ef_hat  = query(model, ctx, edited_full, prompt_a, q_hat, tok_red, tok_green, vocab, args.n_predict, args.seed);
            auto ef_eyes = query(model, ctx, edited_full, prompt_a, q_eyes, tok_blue, tok_green, vocab, args.n_predict, args.seed);

            std::printf("\n  FULL-PROMPT DELTA (state_A + full_delta):\n");
            std::printf("    eyes: P(blue)=%.4f  gen: %s\n", ef_eyes.p, ef_eyes.gen.substr(0, 50).c_str());
            std::printf("    hat:  P(red)=%.4f  selectivity=%.4f\n", ef_hat.p, ef_hat.p - ef_eyes.p);

            // token-level delta
            rp::StateBuf edited_token(state_full_a);
            rp::apply_full_delta(edited_token, state_after_blue, state_after_green, layers);

            auto et_hat  = query(model, ctx, edited_token, prompt_a, q_hat, tok_red, tok_green, vocab, args.n_predict, args.seed);
            auto et_eyes = query(model, ctx, edited_token, prompt_a, q_eyes, tok_blue, tok_green, vocab, args.n_predict, args.seed);

            std::printf("\n  TOKEN-LEVEL DELTA (state_A + token_delta[blue→green]):\n");
            std::printf("    eyes: P(blue)=%.4f  gen: %s\n", et_eyes.p, et_eyes.gen.substr(0, 50).c_str());
            std::printf("    hat:  P(red)=%.4f  selectivity=%.4f\n", et_hat.p, et_hat.p - et_eyes.p);
        }

        std::fprintf(stderr, "\ndone\n");
    });
}
