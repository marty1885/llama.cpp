// experiments/entity-edit.cpp — test entity-selective state editing.
//
// Prompts A and B differ in ONE entity's fact only. Captures the delta,
// applies it (full and rank-k), then queries BOTH entities to check:
//   - target entity: did the fact change?
//   - control entity: did it stay the same?
//
// If control stays unchanged -> entity-selective editing works.
//
// Usage:
//   llama-rwkv-entity-edit -m model.gguf --tests entity_edit_tests.json
//       [--layers 16-31] [-n 20]

#include "rwkv_probe/experiment.h"
#include "rwkv_probe/state.h"
#include "rwkv_probe/generate.h"
#include "rwkv_probe/pipeline.h"
#include "rwkv_probe/edit.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
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

int main(int argc, char ** argv) {
    auto args = rp::parse_common_args(argc, argv);
    args.defaults(2048, 20);

    double sigma_min = -1.0;  // <0 = sweep

    for (std::size_t i = 0; i < args.extra.size(); ++i) {
        const auto & a = args.extra[i];
        if (a == "--sigma-min" && i+1 < args.extra.size()) sigma_min = std::atof(args.extra[++i].c_str());
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 1; }
    }

    if (args.help_requested) {
        std::printf("usage: %s -m MODEL --tests FILE [--layers 16-31] [-n 20] [--sigma-min 0.5]\n", argv[0]);
        return 0;
    }

    if (!rp::require_model(args)) return 1;
    if (!rp::require_tests(args)) return 1;

    std::string layer_range = args.layer_range.empty() ? "16-31" : args.layer_range;

    std::vector<EntityTest> tests;
    {
        json arr = rp::load_tests(args);
        for (const auto & e : arr) {
            EntityTest tc;
            tc.name           = e.at("name").get<std::string>();
            tc.prompt_a       = e.at("prompt_a").get<std::string>();
            tc.prompt_b       = e.at("prompt_b").get<std::string>();
            tc.query_target   = e.at("query_target").get<std::string>();
            tc.expect_a_target = e.at("expect_a_target").get<std::string>();
            tc.expect_b_target = e.at("expect_b_target").get<std::string>();

            // support both single control and multi-control formats
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
    }

    return rp::run_experiment([&] {
        auto env = rp::make_env(args);
        auto & model = env->model;
        auto & ctx   = env->ctx;
        const auto & geom  = env->geom;
        const auto * vocab  = env->vocab;

        auto layers = rp::parse_layer_range(layer_range, geom.n_layer);

        std::printf("layers: %d-%d (%zu)  n_head=%d\n\n",
                    layers.front(), layers.back(), layers.size(), geom.n_head);

        for (const auto & tc : tests) {
            llama_token tok_a_target = rp::find_token(vocab, tc.expect_a_target);
            llama_token tok_b_target = rp::find_token(vocab, tc.expect_b_target);

            // resolve control tokens
            struct ResolvedControl {
                std::string query;
                std::string expect_str;
                llama_token tok;
            };
            std::vector<ResolvedControl> controls;
            for (const auto & c : tc.controls) {
                controls.push_back({c.query, c.expect, rp::find_token(vocab, c.expect)});
            }

            // capture states after the fact-stating prefix
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

            // ── baselines ───────────────────────────────────────────
            auto bl_a_target = rp::probe(model, ctx, state_a,
                tc.prompt_a, tc.query_target,
                tok_a_target, tok_b_target, vocab, args.n_predict, args.seed);

            auto bl_b_target = rp::probe(model, ctx, state_b,
                tc.prompt_b, tc.query_target,
                tok_a_target, tok_b_target, vocab, args.n_predict, args.seed);

            std::printf("\n  BASELINES:\n");
            std::printf("  %-30s  %8s  %s\n", "", "P(exp)", "gen");
            std::printf("  %-30s  %8.4f  %s\n",
                        "A target", bl_a_target.p_binary,
                        bl_a_target.generation.substr(0, 50).c_str());
            std::printf("  %-30s  %8.4f  %s\n",
                        "B target", bl_b_target.p_binary,
                        bl_b_target.generation.substr(0, 50).c_str());
            for (const auto & c : controls) {
                auto bl = rp::probe(model, ctx, state_a,
                    tc.prompt_a, c.query,
                    c.tok, tok_b_target, vocab, args.n_predict, args.seed);
                std::printf("  A ctrl %-22s  %8.4f  %s\n",
                            c.expect_str.c_str(), bl.p_binary,
                            bl.generation.substr(0, 50).c_str());
            }

            // ── sigma sweep ──────────────────────────────────────────
            std::vector<double> sigma_vals;
            if (sigma_min >= 0.0) {
                sigma_vals = {sigma_min};
            } else {
                sigma_vals = {0.0, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0};
            }

            std::printf("\n  EDITS (state_A + full delta, varying sigma threshold):\n");

            // build header
            std::printf("  %-10s  %8s", "sigma_min", "P(a_tgt)");
            for (const auto & c : controls) {
                char hdr[32];
                std::snprintf(hdr, sizeof hdr, "P(%s)", c.expect_str.substr(0, 6).c_str());
                std::printf("  %8s", hdr);
            }
            std::printf("  %s\n", "gen_target");
            std::printf("  %s\n", std::string(12 + 10 + controls.size() * 10 + 40, '-').c_str());

            for (double sm : sigma_vals) {
                rp::StateBuf edited(state_a);
                rp::apply_rank_delta(edited, state_a, state_b, layers, 0, 1.0, sm);

                auto r_target = rp::probe(model, ctx, edited,
                    tc.prompt_a, tc.query_target,
                    tok_a_target, tok_b_target, vocab, args.n_predict, args.seed);

                std::printf("  s>=%-7.1f  %8.4f", sm, r_target.p_binary);

                for (const auto & c : controls) {
                    auto r_ctrl = rp::probe(model, ctx, edited,
                        tc.prompt_a, c.query,
                        c.tok, tok_b_target, vocab, args.n_predict, args.seed);
                    std::printf("  %8.4f", r_ctrl.p_binary);
                }

                std::string tgen = r_target.generation.substr(0, 40);
                for (auto & ch : tgen) if (ch == '\n') ch = ' ';
                std::printf("  %s\n", tgen.c_str());
                std::fflush(stdout);
            }

            std::printf("\n");
        }

        std::fprintf(stderr, "done\n");
    });
}
