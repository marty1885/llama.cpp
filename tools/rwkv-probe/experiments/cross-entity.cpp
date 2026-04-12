// experiments/cross-entity.cpp — cross-context + entity-selective editing.
//
// Takes a delta from a "donor" prompt pair and applies it to a different
// "target" prompt pair's state, then queries both entities.
// Tests: does a simple calibration delta selectively edit one entity
// in a completely different prompt structure?
//
// Groups tests by fact (same expect values), then cross-applies
// every donor's delta to every target's state.
//
// Usage:
//   llama-rwkv-cross-entity -m model.gguf --tests cross_entity_tests.json
//       [--layers 16-31] [-n 15]

#include "rwkv_probe/state.h"
#include "rwkv_probe/generate.h"
#include "rwkv_probe/experiment.h"
#include "rwkv_probe/edit.h"

#include <cstdio>
#include <string>
#include <vector>

namespace rp = rwkv_probe;
using json = nlohmann::json;

struct EntityTest {
    std::string name;
    std::string prompt_a, prompt_b;
    std::string query_target, query_control;
    std::string expect_a_target, expect_b_target, expect_control;
};

// query state with a given prompt prefix + query suffix
static float query_prob(rp::Context & ctx, const rp::StateBuf & state,
                        const std::vector<llama_token> & prefix_toks,
                        const std::vector<llama_token> & query_toks,
                        llama_token tok_a, llama_token tok_b) {
    ctx.clear_memory();
    ctx.decode(rp::span<const llama_token>(prefix_toks.data(), prefix_toks.size()));
    state.store_to(ctx.raw());
    for (const auto & t : query_toks) {
        ctx.decode_one(t);
    }
    const float * logits = llama_get_logits(ctx.raw());
    return rp::prob_of(logits[tok_a], logits[tok_b]);
}

static std::string query_gen(rp::Model & model, rp::Context & ctx,
                             const rp::StateBuf & state,
                             const std::vector<llama_token> & prefix_toks,
                             const std::vector<llama_token> & query_toks,
                             int n_predict, uint32_t seed) {
    ctx.clear_memory();
    ctx.decode(rp::span<const llama_token>(prefix_toks.data(), prefix_toks.size()));
    state.store_to(ctx.raw());
    for (const auto & t : query_toks) {
        ctx.decode_one(t);
    }
    common_params_sampling sp;
    sp.seed = seed;
    rp::Generator gen(model, ctx, sp);
    llama_token last = query_toks.back();
    gen.accept_prompt(rp::span<const llama_token>(&last, 1));
    std::string text = gen.run(n_predict).text;
    for (auto & c : text) {
        if (c == '\n') c = ' ';
    }
    return text;
}

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
    std::vector<EntityTest> tests;
    for (const auto & e : arr) {
        tests.push_back({
            e.at("name").get<std::string>(),
            e.at("prompt_a").get<std::string>(),
            e.at("prompt_b").get<std::string>(),
            e.at("query_target").get<std::string>(),
            e.at("query_control").get<std::string>(),
            e.at("expect_a_target").get<std::string>(),
            e.at("expect_b_target").get<std::string>(),
            e.at("expect_control").get<std::string>(),
        });
    }

    return rp::run_experiment([&] {
        auto env = rp::make_env(args);
        auto & model = env->model;
        auto & ctx   = env->ctx;
        const auto & geom  = env->geom;
        const auto * vocab = env->vocab;

        auto layers = rp::parse_layer_range(args.layer_range, geom.n_layer);

        // group tests by fact type (same expect values)
        struct FactGroup {
            std::string id;
            std::vector<std::size_t> indices;
        };
        std::vector<FactGroup> groups;
        for (std::size_t i = 0; i < tests.size(); ++i) {
            std::string fid = tests[i].expect_a_target + "→" + tests[i].expect_b_target +
                              " (ctrl=" + tests[i].expect_control + ")";
            bool found = false;
            for (auto & g : groups) {
                if (g.id == fid) { g.indices.push_back(i); found = true; break; }
            }
            if (!found) groups.push_back({fid, {i}});
        }

        // capture all states
        struct Captured {
            rp::StateBuf state_a, state_b;
            std::vector<llama_token> prefix_a, prefix_b;
            std::vector<llama_token> q_target, q_control;
            llama_token tok_a_target, tok_b_target, tok_control;
            Captured(const rp::ModelGeometry & g) : state_a(g), state_b(g) {}
        };

        std::vector<Captured> cap;
        cap.reserve(tests.size());

        for (std::size_t i = 0; i < tests.size(); ++i) {
            const auto & tc = tests[i];
            cap.emplace_back(geom);
            auto & c = cap.back();

            c.tok_a_target = rp::find_token(vocab, tc.expect_a_target);
            c.tok_b_target = rp::find_token(vocab, tc.expect_b_target);
            c.tok_control = rp::find_token(vocab, tc.expect_control);

            c.prefix_a = rp::tokenize(vocab, tc.prompt_a);
            c.prefix_b = rp::tokenize(vocab, tc.prompt_b);
            c.q_target = rp::tokenize(vocab, tc.query_target, false, false);
            c.q_control = rp::tokenize(vocab, tc.query_control, false, false);

            ctx.clear_memory();
            ctx.decode(rp::span<const llama_token>(c.prefix_a.data(), c.prefix_a.size()));
            c.state_a.load_from(ctx.raw());

            ctx.clear_memory();
            ctx.decode(rp::span<const llama_token>(c.prefix_b.data(), c.prefix_b.size()));
            c.state_b.load_from(ctx.raw());

            // verify baselines
            float pt = query_prob(ctx, c.state_a, c.prefix_a, c.q_target,
                                  c.tok_a_target, c.tok_b_target);
            float pc = query_prob(ctx, c.state_a, c.prefix_a, c.q_control,
                                  c.tok_control, c.tok_b_target);
            std::fprintf(stderr, "  %s: P(%s)=%.3f  P(%s)=%.3f\n",
                         tc.name.c_str(), tc.expect_a_target.c_str(), pt,
                         tc.expect_control.c_str(), pc);
        }

        // cross-apply within each group
        for (const auto & group : groups) {
            std::printf("======================================================================\n");
            std::printf("FACT: %s  (%zu phrasings)\n", group.id.c_str(), group.indices.size());
            std::printf("======================================================================\n\n");

            for (std::size_t di : group.indices) {
                for (std::size_t ti : group.indices) {
                    const auto & donor = cap[di];
                    const auto & target = cap[ti];
                    const auto & tc_target = tests[ti];

                    bool is_self = (di == ti);

                    // apply donor's delta to target's state_a
                    rp::StateBuf edited(target.state_a);
                    rp::apply_full_delta(edited, donor.state_a, donor.state_b,
                                    layers);

                    float pt = query_prob(ctx, edited, target.prefix_a, target.q_target,
                                          target.tok_a_target, target.tok_b_target);
                    float pc = query_prob(ctx, edited, target.prefix_a, target.q_control,
                                          target.tok_control, target.tok_b_target);

                    std::string t_gen = query_gen(model, ctx, edited,
                                                 target.prefix_a, target.q_target,
                                                 args.n_predict, args.seed);
                    std::string c_gen = query_gen(model, ctx, edited,
                                                 target.prefix_a, target.q_control,
                                                 args.n_predict, args.seed);

                    std::printf("  donor: %-22s → target: %-22s %s\n",
                                tests[di].name.c_str(), tests[ti].name.c_str(),
                                is_self ? "(SELF)" : "(CROSS)");
                    std::printf("    target entity: P(%s)=%.4f  gen: %s\n",
                                tc_target.expect_a_target.c_str(), pt,
                                t_gen.substr(0, 60).c_str());
                    std::printf("    control entity: P(%s)=%.4f  gen: %s\n",
                                tc_target.expect_control.c_str(), pc,
                                c_gen.substr(0, 60).c_str());
                    std::printf("    selectivity: %.4f\n\n", pc - pt);
                    std::fflush(stdout);
                }
            }
        }

        std::fprintf(stderr, "done\n");
    });
}
