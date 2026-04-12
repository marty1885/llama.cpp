// experiments/donor-free.cpp — donor-free state editing via calibrated directions.
//
// Phase 1 (calibrate): extract per-head v1 (address) and u1 (value) from
//   a calibration pair (e.g., alice_red vs alice_green).
// Phase 2 (edit): given a NEW prompt's state (never seen in calibration),
//   construct a synthetic edit using calibrated directions. No donor needed.
//
// Edit method: for each head at each layer,
//   projection = S . v1  (64-dim vector: current value along the address)
//   S_new = S - projection . v1^T + (sigma_target . u1_target) . v1^T
//
// Usage:
//   llama-rwkv-donor-free -m model.gguf --tests donor_free_tests.json
//       [--layers 16-31] [-n 15] [--sigma-min 0.5] [--alpha 1.0]

#include "rwkv_probe/state.h"
#include "rwkv_probe/generate.h"
#include "rwkv_probe/experiment.h"
#include "rwkv_probe/pipeline.h"
#include "rwkv_probe/edit.h"
#include "rwkv_probe/head_decomp.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace rp = rwkv_probe;
using json = nlohmann::json;

int main(int argc, char ** argv) {
    auto args = rp::parse_common_args(argc, argv);
    if (args.help_requested) {
        std::printf("usage: %s -m MODEL [--layers 16-31] [-n 15] [--sigma-min 0.5] [--alpha 1.0]\n", argv[0]);
        return 0;
    }
    args.defaults(2048, 15);
    if (args.layer_range.empty()) args.layer_range = "16-31";
    if (!rp::require_model(args)) return 1;

    // experiment-specific args
    double sigma_min = -1.0;  // <0 = sweep multiple values
    double alpha = 1.0;

    for (std::size_t i = 0; i < args.extra.size(); ++i) {
        const auto & a = args.extra[i];
        if      (a == "--sigma-min" && i+1 < args.extra.size()) sigma_min = std::atof(args.extra[++i].c_str());
        else if (a == "--alpha"     && i+1 < args.extra.size()) alpha     = std::atof(args.extra[++i].c_str());
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 1; }
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

        llama_token tok_paris  = rp::find_token(vocab, "Paris");
        llama_token tok_london = rp::find_token(vocab, "London");
        llama_token tok_tokyo  = rp::find_token(vocab, "Tokyo");
        llama_token tok_red    = rp::find_token(vocab, "red");
        llama_token tok_green  = rp::find_token(vocab, "green");
        llama_token tok_blue   = rp::find_token(vocab, "blue");

        // ════════════════════════════════════════════════════════════
        // PHASE 1: CALIBRATION
        // Run simple calibration prompts, extract per-head directions
        // ════════════════════════════════════════════════════════════
        std::printf("PHASE 1: CALIBRATION\n");
        std::printf("====================\n\n");

        // city calibration: Paris <-> London, Paris <-> Tokyo
        auto cal_paris  = rp::tokenize(vocab, "Alice lives in Paris. Where does Alice live? Alice lives in");
        auto cal_london = rp::tokenize(vocab, "Alice lives in London. Where does Alice live? Alice lives in");
        auto cal_tokyo  = rp::tokenize(vocab, "Alice lives in Tokyo. Where does Alice live? Alice lives in");

        rp::StateBuf st_paris(geom), st_london(geom), st_tokyo(geom);

        ctx.clear_memory();
        ctx.decode(rp::span<const llama_token>(cal_paris.data(), cal_paris.size() - 1));
        st_paris.load_from(ctx.raw());

        ctx.clear_memory();
        ctx.decode(rp::span<const llama_token>(cal_london.data(), cal_london.size() - 1));
        st_london.load_from(ctx.raw());

        ctx.clear_memory();
        ctx.decode(rp::span<const llama_token>(cal_tokyo.data(), cal_tokyo.size() - 1));
        st_tokyo.load_from(ctx.raw());

        // extract per-head directions for Paris->London and Paris->Tokyo
        std::vector<rp::HeadDecomp> dirs_paris_london, dirs_paris_tokyo;
        for (int layer : layers) {
            for (int h = 0; h < n_head; ++h) {
                dirs_paris_london.push_back(rp::decompose_head(st_paris, st_london, layer, h));
                dirs_paris_tokyo.push_back(rp::decompose_head(st_paris, st_tokyo, layer, h));
            }
        }

        std::printf("  city calibration: %zu heads x %zu layers = %zu directions\n",
                    (std::size_t)n_head, layers.size(), dirs_paris_london.size());

        // hat calibration: red <-> green, red <-> blue
        auto cal_red   = rp::tokenize(vocab, "Alice has a red hat. What color is Alice's hat? Alice's hat is");
        auto cal_green = rp::tokenize(vocab, "Alice has a green hat. What color is Alice's hat? Alice's hat is");
        auto cal_blue  = rp::tokenize(vocab, "Alice has a blue hat. What color is Alice's hat? Alice's hat is");

        rp::StateBuf st_red(geom), st_green(geom), st_blue(geom);

        ctx.clear_memory();
        ctx.decode(rp::span<const llama_token>(cal_red.data(), cal_red.size() - 1));
        st_red.load_from(ctx.raw());

        ctx.clear_memory();
        ctx.decode(rp::span<const llama_token>(cal_green.data(), cal_green.size() - 1));
        st_green.load_from(ctx.raw());

        ctx.clear_memory();
        ctx.decode(rp::span<const llama_token>(cal_blue.data(), cal_blue.size() - 1));
        st_blue.load_from(ctx.raw());

        std::vector<rp::HeadDecomp> dirs_red_green, dirs_red_blue;
        for (int layer : layers) {
            for (int h = 0; h < n_head; ++h) {
                dirs_red_green.push_back(rp::decompose_head(st_red, st_green, layer, h));
                dirs_red_blue.push_back(rp::decompose_head(st_red, st_blue, layer, h));
            }
        }

        std::printf("  hat calibration: %zu directions\n\n", dirs_red_green.size());

        // ════════════════════════════════════════════════════════════
        // PHASE 2: DONOR-FREE EDITING ON NEW PROMPTS
        // ════════════════════════════════════════════════════════════
        std::printf("PHASE 2: DONOR-FREE EDITING\n");
        std::printf("===========================\n\n");

        struct EditTest {
            const char * name;
            const char * prompt;
            llama_token tok_from, tok_to, tok_alt;
            const char * expect_from;
            const char * expect_to;
            std::vector<rp::HeadDecomp> * dirs_from;
            std::vector<rp::HeadDecomp> * dirs_to;
            // for delta comparison
            rp::StateBuf * cal_from;
            rp::StateBuf * cal_to;
        };

        EditTest edit_tests[] = {
            {"city: Paris->London (narrative)",
             "Last year Alice moved to Paris. She enjoys living there. Alice currently lives in",
             tok_paris, tok_london, tok_tokyo, "Paris", "London",
             &dirs_paris_london, &dirs_paris_london,
             &st_paris, &st_london},

            {"city: Paris->Tokyo (narrative)",
             "Last year Alice moved to Paris. She enjoys living there. Alice currently lives in",
             tok_paris, tok_tokyo, tok_london, "Paris", "Tokyo",
             &dirs_paris_tokyo, &dirs_paris_tokyo,
             &st_paris, &st_tokyo},

            {"city: Paris->London (QA)",
             "Q: Where does Alice live? A: Paris. Q: What city is Alice in? A:",
             tok_paris, tok_london, tok_tokyo, "Paris", "London",
             &dirs_paris_london, &dirs_paris_london,
             &st_paris, &st_london},

            {"hat: red->green (narrative)",
             "Alice bought a beautiful red hat yesterday. She wore it today. The color of Alice's hat is",
             tok_red, tok_green, tok_blue, "red", "green",
             &dirs_red_green, &dirs_red_green,
             &st_red, &st_green},

            {"hat: red->blue (third person)",
             "We know Alice wears a red hat. If asked about Alice's hat color, the answer is",
             tok_red, tok_blue, tok_green, "red", "blue",
             &dirs_red_blue, &dirs_red_blue,
             &st_red, &st_blue},
        };

        // determine sigma_min values to sweep
        std::vector<double> sigma_mins;
        if (sigma_min >= 0.0) {
            sigma_mins.push_back(sigma_min);
        } else {
            // auto-sweep: find sigma distribution and pick percentiles
            std::vector<double> all_sigmas;
            for (const auto & d : dirs_paris_london) all_sigmas.push_back(d.sigma);
            for (const auto & d : dirs_red_green) all_sigmas.push_back(d.sigma);
            std::sort(all_sigmas.begin(), all_sigmas.end());
            int n = (int)all_sigmas.size();
            sigma_mins = {
                0.0,
                all_sigmas[n / 2],       // median
                all_sigmas[n * 3 / 4],   // p75
                all_sigmas[n * 9 / 10],  // p90
                all_sigmas[n * 19 / 20], // p95
            };
            std::printf("  sigma distribution: min=%.4f  median=%.4f  p75=%.4f  p90=%.4f  p95=%.4f  max=%.4f\n\n",
                        all_sigmas[0], all_sigmas[n/2], all_sigmas[n*3/4],
                        all_sigmas[n*9/10], all_sigmas[n*19/20], all_sigmas[n-1]);
        }

        for (const auto & et : edit_tests) {
            auto toks = rp::tokenize(vocab, et.prompt);

            ctx.clear_memory();
            ctx.decode(rp::span<const llama_token>(toks.data(), toks.size() - 1));
            rp::StateBuf state_orig(geom);
            state_orig.load_from(ctx.raw());

            float p_base = rp::measure(ctx, state_orig, toks, et.tok_from, et.tok_to);

            // delta-based (reference)
            rp::StateBuf state_delta(state_orig);
            rp::apply_full_delta(state_delta, *et.cal_from, *et.cal_to, layers);
            float p_delta = rp::measure(ctx, state_delta, toks, et.tok_from, et.tok_to);
            std::string gen_delta = rp::generate_from(model, ctx, state_delta, toks, args.n_predict, args.seed);

            std::printf("  %s\n", et.name);
            std::printf("    prompt: \"%.70s\"\n", et.prompt);
            std::printf("    baseline:     P(%s)=%.4f\n", et.expect_from, p_base);
            std::printf("    delta (ref):  P(%s)=%.4f  gen: %s\n", et.expect_from, p_delta, gen_delta.c_str());

            for (double sm : sigma_mins) {
                // count how many heads pass the threshold
                int n_active = 0;
                for (const auto & d : *et.dirs_from) {
                    if (d.sigma >= sm) n_active++;
                }

                rp::StateBuf state_df(state_orig);
                rp::apply_donor_free(state_df, *et.dirs_from, *et.dirs_to, sm, alpha);
                float p_df = rp::measure(ctx, state_df, toks, et.tok_from, et.tok_to);
                std::string gen_df = rp::generate_from(model, ctx, state_df, toks, args.n_predict, args.seed);

                std::printf("    s>%-8.4f  P(%s)=%.4f  heads=%4d  gen: %s\n",
                            sm, et.expect_from, p_df, n_active, gen_df.c_str());
            }
            std::printf("\n");
            std::fflush(stdout);
        }

        std::fprintf(stderr, "done\n");
    });
}
