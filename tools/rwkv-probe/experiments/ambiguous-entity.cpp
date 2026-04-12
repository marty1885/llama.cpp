// experiments/ambiguous-entity.cpp — test editing with ambiguous entities.
//
// Two Alices (Alice Anderson, Alice White) with different properties.
// Tests:
//   1. Ambiguous calibration: "Alice has red/green hat" — which Alice gets edited?
//   2. Specific calibration: "Alice Anderson has red/green hat" — selective?
//   3. Wrong calibration: "Alice White has red/green hat" — does it edit Anderson?
//
// Usage:
//   llama-rwkv-ambiguous-entity -m model.gguf [--layers 16-31] [-n 15]

#include "rwkv_probe/state.h"
#include "rwkv_probe/generate.h"
#include "rwkv_probe/experiment.h"
#include "rwkv_probe/pipeline.h"
#include "rwkv_probe/edit.h"

#include <cstdio>
#include <string>
#include <vector>

namespace rp = rwkv_probe;

struct CalPair {
    rp::StateBuf state_a;
    rp::StateBuf state_b;
    std::string label;
    CalPair(const rp::ModelGeometry & g) : state_a(g), state_b(g) {}
};

static CalPair calibrate(rp::Context & ctx,
                         const rp::ModelGeometry & geom, const llama_vocab * vocab,
                         const std::string & prompt_a, const std::string & prompt_b,
                         const std::string & label) {
    CalPair cp(geom);
    cp.label = label;

    auto ta = rp::tokenize(vocab, prompt_a);
    auto tb = rp::tokenize(vocab, prompt_b);

    ctx.clear_memory();
    ctx.decode(rp::span<const llama_token>(ta.data(), ta.size() - 1));
    cp.state_a.load_from(ctx.raw());

    ctx.clear_memory();
    ctx.decode(rp::span<const llama_token>(tb.data(), tb.size() - 1));
    cp.state_b.load_from(ctx.raw());

    return cp;
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

        llama_token tok_red   = rp::find_token(vocab, "red");
        llama_token tok_green = rp::find_token(vocab, "green");
        llama_token tok_blue  = rp::find_token(vocab, "blue");

        // context with two Alices
        std::string context = "Alice Anderson has a red hat. Alice White has a blue hat.";
        auto ctx_toks = rp::tokenize(vocab, context);

        ctx.clear_memory();
        ctx.decode(rp::span<const llama_token>(ctx_toks.data(), ctx_toks.size()));
        rp::StateBuf state_base(geom);
        state_base.load_from(ctx.raw());

        std::string q_anderson = " What color is Alice Anderson's hat? Alice Anderson's hat is";
        std::string q_white    = " What color is Alice White's hat? Alice White's hat is";

        // baselines
        auto bl_and = rp::probe(model, ctx, state_base, context, q_anderson,
                               tok_red, tok_green, vocab, args.n_predict, args.seed);
        auto bl_wht = rp::probe(model, ctx, state_base, context, q_white,
                               tok_blue, tok_green, vocab, args.n_predict, args.seed);

        std::printf("CONTEXT: \"%s\"\n\n", context.c_str());
        std::printf("BASELINES:\n");
        std::printf("  Anderson: P(red)=%.4f  gen: %s\n", bl_and.p_binary, bl_and.generation.substr(0, 50).c_str());
        std::printf("  White:    P(blue)=%.4f  gen: %s\n\n", bl_wht.p_binary, bl_wht.generation.substr(0, 50).c_str());

        // calibration variants
        auto cal_ambig = calibrate(ctx, geom, vocab,
            "Alice has a red hat. What color is Alice's hat? Alice's hat is",
            "Alice has a green hat. What color is Alice's hat? Alice's hat is",
            "ambiguous 'Alice'");

        auto cal_anderson = calibrate(ctx, geom, vocab,
            "Alice Anderson has a red hat. What color is Alice Anderson's hat? Alice Anderson's hat is",
            "Alice Anderson has a green hat. What color is Alice Anderson's hat? Alice Anderson's hat is",
            "specific 'Alice Anderson'");

        auto cal_white = calibrate(ctx, geom, vocab,
            "Alice White has a blue hat. What color is Alice White's hat? Alice White's hat is",
            "Alice White has a green hat. What color is Alice White's hat? Alice White's hat is",
            "specific 'Alice White'");

        // apply each calibration and check both Alices
        struct TestCal {
            CalPair * cal;
            const char * name;
        };
        TestCal cals[] = {
            {&cal_ambig,    "ambiguous 'Alice' (red→green)"},
            {&cal_anderson, "specific 'Alice Anderson' (red→green)"},
            {&cal_white,    "specific 'Alice White' (blue→green)"},
        };

        for (const auto & tc : cals) {
            rp::StateBuf edited(state_base);
            rp::apply_full_delta(edited, tc.cal->state_a, tc.cal->state_b, layers);

            auto r_and = rp::probe(model, ctx, edited, context, q_anderson,
                                  tok_red, tok_green, vocab, args.n_predict, args.seed);
            auto r_wht = rp::probe(model, ctx, edited, context, q_white,
                                  tok_blue, tok_green, vocab, args.n_predict, args.seed);

            std::printf("  EDIT: %s\n", tc.name);
            std::printf("    Anderson: P(red)=%.4f  %s  gen: %s\n",
                        r_and.p_binary,
                        (r_and.p_binary < 0.5) ? "FLIPPED" : "stayed",
                        r_and.generation.substr(0, 45).c_str());
            std::printf("    White:    P(blue)=%.4f  %s  gen: %s\n\n",
                        r_wht.p_binary,
                        (r_wht.p_binary < 0.5) ? "FLIPPED" : "stayed",
                        r_wht.generation.substr(0, 45).c_str());
        }

        std::fprintf(stderr, "done\n");
    });
}
