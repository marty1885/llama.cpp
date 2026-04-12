// experiments/distance-test.cpp — test editing faded memories.
//
// Inserts N tokens of filler between the fact and the query, then
// applies a calibration delta (from a short prompt). Tests whether
// edits still work as facts decay in the recurrent state.
//
// Usage:
//   llama-rwkv-distance-test -m model.gguf [--layers 16-31] [-n 15]
//       [--distances 0,10,25,50,100,200]

#include "rwkv_probe/state.h"
#include "rwkv_probe/generate.h"
#include "rwkv_probe/experiment.h"
#include "rwkv_probe/edit.h"

#include <cstdio>
#include <string>
#include <vector>

namespace rp = rwkv_probe;

int main(int argc, char ** argv) {
    auto args = rp::parse_common_args(argc, argv);
    args.defaults(/*ctx=*/2048, /*predict=*/15);
    if (args.layer_range.empty()) args.layer_range = "16-31";

    std::string dist_str = "0,10,25,50,100,200";

    for (std::size_t i = 0; i < args.extra.size(); ++i) {
        const auto & a = args.extra[i];
        if (a == "--distances" && i + 1 < args.extra.size()) {
            dist_str = args.extra[++i];
        } else {
            std::fprintf(stderr, "unknown arg: %s\n", a.c_str());
            return 1;
        }
    }

    if (args.help_requested) {
        std::printf("usage: %s -m MODEL [--distances 0,10,50,100,200] [--layers 16-31]\n", argv[0]);
        return 0;
    }
    if (!rp::require_model(args)) return 1;

    auto distances = rp::parse_int_list(dist_str);

    return rp::run_experiment([&] {
        auto env = rp::make_env(args);
        auto & model = env->model;
        auto & ctx   = env->ctx;
        const auto & geom  = env->geom;
        const auto * vocab = env->vocab;

        auto layers = rp::parse_layer_range(args.layer_range, geom.n_layer);

        llama_token tok_paris  = rp::find_token(vocab, "Paris");
        llama_token tok_london = rp::find_token(vocab, "London");
        llama_token tok_red    = rp::find_token(vocab, "red");
        llama_token tok_green  = rp::find_token(vocab, "green");

        // filler sentences (unrelated to the facts)
        const char * fillers[] = {
            " The weather is nice today.",
            " Birds sing in the morning.",
            " The river flows to the sea.",
            " Mountains are tall and cold.",
            " Fish swim in the ocean.",
            " The sun sets in the west.",
            " Trees grow in the forest.",
            " Stars shine at night.",
            " Clouds float in the sky.",
            " Rain falls from above.",
        };
        int n_fillers = 10;

        // ── calibration: short prompt delta ──────────────────────
        std::printf("CALIBRATION (short prompts, no filler):\n\n");

        auto cal_a_city = rp::tokenize(vocab, "Alice lives in Paris. Where does Alice live? Alice lives in");
        auto cal_b_city = rp::tokenize(vocab, "Alice lives in London. Where does Alice live? Alice lives in");

        ctx.clear_memory();
        ctx.decode(rp::span<const llama_token>(cal_a_city.data(), cal_a_city.size() - 1));
        rp::StateBuf cal_state_paris(geom);
        cal_state_paris.load_from(ctx.raw());

        ctx.clear_memory();
        ctx.decode(rp::span<const llama_token>(cal_b_city.data(), cal_b_city.size() - 1));
        rp::StateBuf cal_state_london(geom);
        cal_state_london.load_from(ctx.raw());

        auto cal_a_hat = rp::tokenize(vocab, "Alice has a red hat. What color is Alice's hat? Alice's hat is");
        auto cal_b_hat = rp::tokenize(vocab, "Alice has a green hat. What color is Alice's hat? Alice's hat is");

        ctx.clear_memory();
        ctx.decode(rp::span<const llama_token>(cal_a_hat.data(), cal_a_hat.size() - 1));
        rp::StateBuf cal_state_red(geom);
        cal_state_red.load_from(ctx.raw());

        ctx.clear_memory();
        ctx.decode(rp::span<const llama_token>(cal_b_hat.data(), cal_b_hat.size() - 1));
        rp::StateBuf cal_state_green(geom);
        cal_state_green.load_from(ctx.raw());

        std::printf("  calibration deltas captured\n\n");

        // ── distance sweep ───────────────────────────────────────
        struct FactTest {
            const char * name;
            const char * fact_a;
            const char * fact_b;
            const char * query;
            llama_token tok_a, tok_b;
            rp::StateBuf * cal_a;
            rp::StateBuf * cal_b;
        };

        FactTest tests[] = {
            {"city Paris→London",
             "Alice lives in Paris.",
             "Alice lives in London.",
             " Where does Alice live? Alice lives in",
             tok_paris, tok_london, &cal_state_paris, &cal_state_london},
            {"hat red→green",
             "Alice has a red hat.",
             "Alice has a green hat.",
             " What color is Alice's hat? Alice's hat is",
             tok_red, tok_green, &cal_state_red, &cal_state_green},
        };

        for (const auto & ft : tests) {
            std::printf("======================================================================\n");
            std::printf("TEST: %s\n", ft.name);
            std::printf("======================================================================\n\n");

            std::printf("  %-6s  %8s  %8s  %8s  %8s  %s\n",
                        "dist", "base_P", "edit_P", "flipped?", "n_toks", "gen");
            std::printf("  %s\n", std::string(80, '-').c_str());

            for (int dist : distances) {
                // build prompt: fact + filler + query
                std::string prompt = ft.fact_a;
                int filler_toks = 0;
                {
                    std::string filler;
                    int fi = 0;
                    while (true) {
                        std::string candidate = filler + fillers[fi % n_fillers];
                        auto toks = rp::tokenize(vocab, candidate, false, false);
                        if ((int)toks.size() >= dist) break;
                        filler = candidate;
                        fi++;
                        if (fi > 100) break;  // safety
                    }
                    prompt += filler;
                    filler_toks = (int)rp::tokenize(vocab, filler, false, false).size();
                }
                (void)filler_toks;

                std::string full_a = prompt + ft.query;
                auto toks_full = rp::tokenize(vocab, full_a);

                if ((int)toks_full.size() >= args.n_ctx - args.n_predict - 10) {
                    std::printf("  %-6d  (skipped — exceeds context)\n", dist);
                    continue;
                }

                // baseline (unedited)
                ctx.clear_memory();
                ctx.decode(rp::span<const llama_token>(toks_full.data(), toks_full.size()));
                const float * logits = llama_get_logits(ctx.raw());
                float p_base = rp::prob_of(logits[ft.tok_a], logits[ft.tok_b]);

                // edited: capture state at full prompt, apply calibration delta
                ctx.clear_memory();
                ctx.decode(rp::span<const llama_token>(toks_full.data(), toks_full.size() - 1));
                rp::StateBuf state(geom);
                state.load_from(ctx.raw());

                rp::StateBuf edited(state);
                rp::apply_full_delta(edited, *ft.cal_a, *ft.cal_b, layers);

                edited.store_to(ctx.raw());
                ctx.decode_one(toks_full.back());
                logits = llama_get_logits(ctx.raw());
                float p_edit = rp::prob_of(logits[ft.tok_a], logits[ft.tok_b]);

                // generate
                std::string gen;
                {
                    common_params_sampling sp;
                    sp.seed = args.seed;
                    rp::Generator g(model, ctx, sp);
                    llama_token last = toks_full.back();
                    g.accept_prompt(rp::span<const llama_token>(&last, 1));
                    gen = g.run(args.n_predict).text;
                    for (auto & c : gen) {
                        if (c == '\n') c = ' ';
                    }
                }

                bool flipped = (p_base > 0.5 && p_edit < 0.5);
                std::printf("  %-6d  %8.4f  %8.4f  %8s  %8d  %s\n",
                            dist, p_base, p_edit,
                            flipped ? "YES" : "no",
                            (int)toks_full.size(),
                            gen.substr(0, 50).c_str());
                std::fflush(stdout);
            }
            std::printf("\n");
        }

        std::fprintf(stderr, "done\n");
    });
}
