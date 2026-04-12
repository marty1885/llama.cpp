// experiments/multi-entity.cpp — test editing with 5 entities.
//
// 5 entities each with a city. Edit one entity's city, check all 5.
// Tests whether edits scale to realistic multi-fact contexts.
//
// Usage:
//   llama-rwkv-multi-entity -m model.gguf [--layers 16-31] [-n 15]

#include "rwkv_probe/state.h"
#include "rwkv_probe/generate.h"
#include "rwkv_probe/experiment.h"
#include "rwkv_probe/edit.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace rp = rwkv_probe;

struct Entity {
    std::string name;
    std::string city;
    llama_token tok_city;
};

int main(int argc, char ** argv) {
    auto args = rp::parse_common_args(argc, argv);
    if (args.help_requested) {
        std::printf("usage: %s -m MODEL [--layers 16-31] [-n 15]\n", argv[0]);
        return 0;
    }
    args.defaults(2048, 15);
    if (args.layer_range.empty()) args.layer_range = "16-31";
    if (!rp::require_model(args))  return 1;
    if (!rp::reject_extra(args))   return 1;

    return rp::run_experiment([&] {
        auto env = rp::make_env(args);
        auto & model = env->model;
        auto & ctx   = env->ctx;
        const auto & geom  = env->geom;
        const auto * vocab  = env->vocab;

        auto layers = rp::parse_layer_range(args.layer_range, geom.n_layer);

        // 5 entities with cities
        Entity entities[] = {
            {"Alice", "Paris",  rp::find_token(vocab, "Paris")},
            {"Bob",   "Tokyo",  rp::find_token(vocab, "Tokyo")},
            {"Carol", "London", rp::find_token(vocab, "London")},
            {"Dave",  "Rome",   rp::find_token(vocab, "Rome")},
            {"Eve",   "Berlin", rp::find_token(vocab, "Berlin")},
        };
        int n_ent = 5;

        // target edits: change each entity's city to a different one
        struct EditTarget {
            int entity_idx;
            std::string new_city;
            llama_token tok_new;
        };
        EditTarget edits[] = {
            {0, "Madrid",  rp::find_token(vocab, "Madrid")},   // Alice: Paris->Madrid
            {1, "Seoul",   rp::find_token(vocab, "Seoul")},    // Bob: Tokyo->Seoul
            {2, "Sydney",  rp::find_token(vocab, "Sydney")},   // Carol: London->Sydney
            {3, "Vienna",  rp::find_token(vocab, "Vienna")},   // Dave: Rome->Vienna
            {4, "Oslo",    rp::find_token(vocab, "Oslo")},     // Eve: Berlin->Oslo
        };

        // build the context prompt with all 5 entities
        std::string context;
        for (int i = 0; i < n_ent; ++i) {
            if (i > 0) context += " ";
            context += entities[i].name + " lives in " + entities[i].city + ".";
        }

        std::printf("CONTEXT: \"%s\"\n\n", context.c_str());

        // capture base state
        auto ctx_toks = rp::tokenize(vocab, context);
        ctx.clear_memory();
        ctx.decode(rp::span<const llama_token>(ctx_toks.data(), ctx_toks.size()));
        rp::StateBuf state_base(geom);
        state_base.load_from(ctx.raw());

        // verify baselines
        std::printf("BASELINES (unedited):\n");
        for (int i = 0; i < n_ent; ++i) {
            std::string q = " Where does " + entities[i].name + " live? " +
                           entities[i].name + " lives in";
            auto qtoks = rp::tokenize(vocab, q, false, false);

            ctx.clear_memory();
            ctx.decode(rp::span<const llama_token>(ctx_toks.data(), ctx_toks.size()));
            state_base.store_to(ctx.raw());
            for (const auto & t : qtoks) ctx.decode_one(t);

            const float * logits = llama_get_logits(ctx.raw());
            // find top token
            int top = 0;
            for (int v = 1; v < geom.n_vocab; ++v) {
                if (logits[v] > logits[top]) top = v;
            }

            std::string top_str = rp::piece(ctx.raw(), (llama_token)top);
            float p_correct = logits[entities[i].tok_city];

            std::printf("  %s: logit(%s)=%.2f  top='%s'(%.2f)  %s\n",
                        entities[i].name.c_str(), entities[i].city.c_str(),
                        p_correct, top_str.c_str(), logits[top],
                        top == entities[i].tok_city ? "OK" : "WARN");
        }

        // calibration: simple prompt pairs for each edit
        std::printf("\nCALIBRATION:\n");
        struct CalPair {
            rp::StateBuf state_a;
            rp::StateBuf state_b;
            CalPair(const rp::ModelGeometry & g) : state_a(g), state_b(g) {}
        };

        std::vector<CalPair> cals;
        for (int ei = 0; ei < n_ent; ++ei) {
            const auto & ent = entities[ei];
            const auto & edit = edits[ei];

            std::string cal_a = ent.name + " lives in " + ent.city +
                               ". Where does " + ent.name + " live? " +
                               ent.name + " lives in";
            std::string cal_b = ent.name + " lives in " + edit.new_city +
                               ". Where does " + ent.name + " live? " +
                               ent.name + " lives in";

            auto ta = rp::tokenize(vocab, cal_a);
            auto tb = rp::tokenize(vocab, cal_b);

            cals.emplace_back(geom);
            auto & cp = cals.back();

            ctx.clear_memory();
            ctx.decode(rp::span<const llama_token>(ta.data(), ta.size() - 1));
            cp.state_a.load_from(ctx.raw());

            ctx.clear_memory();
            ctx.decode(rp::span<const llama_token>(tb.data(), tb.size() - 1));
            cp.state_b.load_from(ctx.raw());

            std::fprintf(stderr, "  %s: %s->%s calibrated\n",
                         ent.name.c_str(), ent.city.c_str(), edit.new_city.c_str());
        }

        // apply each edit and check all entities
        std::printf("\nEDITS (one at a time, check all 5):\n\n");

        for (int ei = 0; ei < n_ent; ++ei) {
            const auto & edit = edits[ei];

            std::printf("  EDIT: %s %s->%s\n",
                        entities[ei].name.c_str(), entities[ei].city.c_str(),
                        edit.new_city.c_str());

            rp::StateBuf edited(state_base);
            rp::apply_full_delta(edited, cals[ei].state_a, cals[ei].state_b, layers);

            std::printf("  %-8s  %10s  %10s  %8s  %s\n",
                        "entity", "old_city", "P(old)", "flipped?", "gen");
            std::printf("  %s\n", std::string(65, '-').c_str());

            for (int qi = 0; qi < n_ent; ++qi) {
                std::string q = " Where does " + entities[qi].name + " live? " +
                               entities[qi].name + " lives in";
                auto qtoks = rp::tokenize(vocab, q, false, false);

                ctx.clear_memory();
                ctx.decode(rp::span<const llama_token>(ctx_toks.data(), ctx_toks.size()));
                edited.store_to(ctx.raw());
                for (const auto & t : qtoks) ctx.decode_one(t);

                const float * logits = llama_get_logits(ctx.raw());
                float p_old = rp::prob_of(logits[entities[qi].tok_city], logits[edit.tok_new]);

                // generate
                std::string gen;
                {
                    common_params_sampling sp;
                    sp.seed = args.seed;
                    rp::Generator g(model, ctx, sp);
                    llama_token last = qtoks.back();
                    g.accept_prompt(rp::span<const llama_token>(&last, 1));
                    gen = g.run(args.n_predict).text;
                    for (auto & c : gen) if (c == '\n') c = ' ';
                }

                bool is_target = (qi == ei);
                bool flipped = (p_old < 0.5);
                std::printf("  %-8s  %10s  %10.4f  %8s  %s %s\n",
                            entities[qi].name.c_str(),
                            entities[qi].city.c_str(),
                            p_old,
                            is_target ? (flipped ? "YES" : "FAIL") :
                                       (flipped ? "LEAK!" : "ok"),
                            gen.substr(0, 40).c_str(),
                            is_target ? "<-- TARGET" : "");
            }
            std::printf("\n");
            std::fflush(stdout);
        }

        std::fprintf(stderr, "done\n");
    });
}
