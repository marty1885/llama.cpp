// experiments/state-edit.cpp — general-purpose state editing tool.
//
// Accepts a JSON test file with context, questions, and edits.
// Runs baselines, applies edits, queries all questions, reports results.
//
// JSON format:
// [
//   {
//     "name": "test_name",
//     "context": "Alice has a red hat. Bob has a blue hat.",
//     "questions": [
//       {"query": " Alice's hat is", "expect": "red", "label": "alice_hat"},
//       {"query": " Bob's hat is", "expect": "blue", "label": "bob_hat"}
//     ],
//     "edits": [
//       {
//         "op": "change",
//         "from": "Alice has a red hat. Alice's hat is",
//         "to": "Alice has a green hat. Alice's hat is",
//         "label": "alice red->green"
//       },
//       {
//         "op": "wipe",
//         "from": "Alice has a red hat. Alice's hat is",
//         "to": "Alice has a green hat. Alice's hat is",
//         "label": "wipe alice hat"
//       }
//     ]
//   }
// ]
//
// Operations:
//   "change"  — apply full delta (from->to) to context state
//   "wipe"    — subtract delta from context state (erase the fact)
//   "donor-free" — projection-based edit using calibrated v1/u1
//
// Usage:
//   llama-rwkv-state-edit -m model.gguf --tests FILE [--layers 16-31] [-n 15]

#include "rwkv_probe/experiment.h"
#include "rwkv_probe/state.h"
#include "rwkv_probe/generate.h"
#include "rwkv_probe/pipeline.h"
#include "rwkv_probe/edit.h"
#include "rwkv_probe/head_decomp.h"

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

namespace rp = rwkv_probe;
using json = nlohmann::json;

// ── query ────────────────────────────────────────────────────────────

struct QResult {
    float logit_expect;
    int   rank;          // rank of expected token (0 = top-1)
    std::string top_tok; // actual top token text
    float top_logit;
    std::string gen;
};

static QResult do_query(rp::Model & model, rp::Context & ctx,
                        const rp::StateBuf & state,
                        const std::string & context,
                        const std::string & query_str,
                        llama_token tok_expect,
                        const llama_vocab * vocab, int n_vocab,
                        int n_predict, uint32_t seed) {
    auto ctx_toks = rp::tokenize(vocab, context);
    auto q_toks = rp::tokenize(vocab, query_str, false, false);

    ctx.clear_memory();
    ctx.decode(rp::span<const llama_token>(ctx_toks.data(), ctx_toks.size()));
    state.store_to(ctx.raw());
    for (const auto & t : q_toks) {
        ctx.decode_one(t);
    }

    const float * logits = llama_get_logits(ctx.raw());

    QResult qr;
    qr.logit_expect = (tok_expect >= 0) ? logits[tok_expect] : -999.0f;

    // find top token and rank
    int top = 0;
    qr.rank = 0;
    for (int v = 1; v < n_vocab; ++v) {
        if (logits[v] > logits[top]) top = v;
        if (tok_expect >= 0 && logits[v] > qr.logit_expect) qr.rank++;
    }
    qr.top_logit = logits[top];
    qr.top_tok = rp::piece(ctx.raw(), (llama_token)top);

    if (n_predict > 0) {
        common_params_sampling sp;
        sp.seed = seed;
        rp::Generator gen(model, ctx, sp);
        llama_token last = q_toks.back();
        gen.accept_prompt(rp::span<const llama_token>(&last, 1));
        qr.gen = gen.run(n_predict).text;
        for (auto & c : qr.gen) if (c == '\n') c = ' ';
    }
    return qr;
}

// ── main ─────────────────────────────────────────────────────────────

int main(int argc, char ** argv) {
    auto args = rp::parse_common_args(argc, argv);
    args.defaults(2048, 15);

    double sigma_min = 1.0;

    for (std::size_t i = 0; i < args.extra.size(); ++i) {
        const auto & a = args.extra[i];
        if (a == "--sigma-min" && i+1 < args.extra.size()) sigma_min = std::atof(args.extra[++i].c_str());
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 1; }
    }

    if (args.help_requested) {
        std::printf(
            "usage: %s -m MODEL --tests FILE [--layers 16-31] [-n 15] [--sigma-min 1.0]\n\n"
            "JSON format: array of test objects, each with:\n"
            "  name:      test label\n"
            "  context:   the factual prompt\n"
            "  questions: [{query, expect, label}]\n"
            "  edits:     [{op, from, to, label}]\n"
            "    op: \"change\" | \"wipe\" | \"donor-free\"\n",
            argv[0]);
        return 0;
    }

    if (!rp::require_model(args)) return 1;
    if (!rp::require_tests(args)) return 1;

    std::string layer_range = args.layer_range.empty() ? "16-31" : args.layer_range;

    // parse test file
    json test_array = rp::load_tests(args);

    return rp::run_experiment([&] {
        auto env = rp::make_env(args);
        auto & model = env->model;
        auto & ctx   = env->ctx;
        const auto & geom  = env->geom;
        const auto * vocab  = env->vocab;
        int n_head  = geom.n_head;
        int n_vocab = geom.n_vocab;

        auto layers = rp::parse_layer_range(layer_range, geom.n_layer);

        std::printf("model: %s  layers: %d-%d  n_head=%d  sigma_min=%.1f\n\n",
                    args.model_path.c_str(), layers.front(), layers.back(), n_head, sigma_min);

        for (const auto & test : test_array) {
            std::string name = test.at("name").get<std::string>();
            std::string context = test.at("context").get<std::string>();

            std::printf("======================================================================\n");
            std::printf("TEST: %s\n", name.c_str());
            std::printf("  context: \"%s\"\n", context.c_str());

            // parse questions
            struct Question {
                std::string query, expect, label;
                llama_token tok;
            };
            std::vector<Question> questions;
            for (const auto & q : test.at("questions")) {
                Question qn;
                qn.query  = q.at("query").get<std::string>();
                qn.expect = q.at("expect").get<std::string>();
                qn.label  = q.value("label", qn.expect);
                qn.tok    = rp::find_token(vocab, qn.expect);
                questions.push_back(qn);
            }

            // capture context state
            auto ctx_toks = rp::tokenize(vocab, context);
            ctx.clear_memory();
            ctx.decode(rp::span<const llama_token>(ctx_toks.data(), ctx_toks.size()));
            rp::StateBuf state_base(geom);
            state_base.load_from(ctx.raw());

            // ── baselines ────────────────────────────────────────
            std::printf("\n  BASELINES:\n");
            std::printf("  %-16s  %6s  %8s  %10s  %s\n",
                        "question", "rank", "logit", "top_tok", "gen");
            std::printf("  %s\n", std::string(75, '-').c_str());

            for (const auto & q : questions) {
                auto r = do_query(model, ctx, state_base, context, q.query,
                                  q.tok, vocab, n_vocab, args.n_predict, args.seed);
                std::printf("  %-16s  %6d  %8.2f  %10s  %s\n",
                            q.label.c_str(), r.rank, r.logit_expect,
                            r.top_tok.c_str(),
                            r.gen.substr(0, 45).c_str());
            }

            // ── edits ────────────────────────────────────────────
            if (!test.contains("edits")) continue;

            for (const auto & edit : test.at("edits")) {
                std::string op    = edit.at("op").get<std::string>();
                std::string from  = edit.at("from").get<std::string>();
                std::string to    = edit.at("to").get<std::string>();
                std::string label = edit.value("label", op);

                // calibrate
                auto from_toks = rp::tokenize(vocab, from);
                auto to_toks   = rp::tokenize(vocab, to);

                ctx.clear_memory();
                ctx.decode(rp::span<const llama_token>(from_toks.data(), from_toks.size() - 1));
                rp::StateBuf state_from(geom);
                state_from.load_from(ctx.raw());

                ctx.clear_memory();
                ctx.decode(rp::span<const llama_token>(to_toks.data(), to_toks.size() - 1));
                rp::StateBuf state_to(geom);
                state_to.load_from(ctx.raw());

                // apply edit
                rp::StateBuf edited(state_base);

                if (op == "change") {
                    rp::apply_full_delta(edited, state_from, state_to, layers, 1.0);
                } else if (op == "wipe") {
                    rp::apply_full_delta(edited, state_from, state_to, layers, -1.0);
                } else if (op == "donor-free") {
                    auto dirs = rp::decompose(state_from, state_to, layers);
                    rp::apply_donor_free(edited, dirs, dirs, sigma_min);
                } else {
                    std::fprintf(stderr, "  unknown op: %s\n", op.c_str());
                    continue;
                }

                // query all questions with edited state
                std::printf("\n  EDIT: %s [%s]\n", label.c_str(), op.c_str());
                std::printf("    from: \"%.60s\"\n", from.c_str());
                std::printf("    to:   \"%.60s\"\n", to.c_str());
                std::printf("  %-16s  %6s  %8s  %10s  %s\n",
                            "question", "rank", "logit", "top_tok", "gen");
                std::printf("  %s\n", std::string(75, '-').c_str());

                for (const auto & q : questions) {
                    auto r = do_query(model, ctx, edited, context, q.query,
                                      q.tok, vocab, n_vocab, args.n_predict, args.seed);

                    std::printf("  %-16s  %6d  %8.2f  %10s  %s\n",
                                q.label.c_str(), r.rank, r.logit_expect,
                                r.top_tok.c_str(),
                                r.gen.substr(0, 45).c_str());
                }
            }
            std::printf("\n");
            std::fflush(stdout);
        }

        std::fprintf(stderr, "done\n");
    });
}
