// experiments/edit-decompose.cpp — decompose donor-free edit into components.
//
// Tests whether the donor-free edit is genuinely writing a new value
// or just drowning out the original signal.
//
// Three components:
//   REMOVE-ONLY:  S - projection . v1^T                    (erase old)
//   ADD-ONLY:     S + sigma . u1_target . v1^T              (write new)
//   FULL:         S - projection . v1^T + sigma . u1 . v1^T (both)
//
// If remove-only flips to target -> just erasing, add is noise.
// If remove-only -> random and only full -> target -> add is genuine writing.

#include "rwkv_probe/state.h"
#include "rwkv_probe/experiment.h"
#include "rwkv_probe/generate.h"

#include "rwkv_probe/numeric.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace rp = rwkv_probe;

enum EditMode { FULL, REMOVE_ONLY, ADD_ONLY };

static void apply_decomposed(rp::StateBuf & dst,
                             const rp::StateBuf & sa, const rp::StateBuf & sb,
                             const std::vector<int> & layers, int n_head, int hs,
                             double sigma_min, EditMode mode) {
    for (int layer : layers) {
        for (int h = 0; h < n_head; ++h) {
            auto da = sa.s_head(layer, h);
            auto db = sb.s_head(layer, h);
            auto dd = dst.s_head(layer, h);

            std::vector<double> delta(hs * hs);
            for (int r = 0; r < hs; ++r) {
                for (int c = 0; c < hs; ++c) {
                    delta[r * hs + c] = (double)db[r * hs + c] - (double)da[r * hs + c];
                }
            }

            auto svdr = rp::numeric::svd(delta.data(), hs);

            if (svdr.sigma[0] < sigma_min) continue;

            // project current state onto v1
            // V(c, 0) in ROOT = Vt row 0, col c = svdr.Vt[0 + c*hs]
            std::vector<double> proj(hs, 0.0);
            for (int r = 0; r < hs; ++r) {
                for (int c = 0; c < hs; ++c) {
                    proj[r] += (double)dd[r * hs + c] * svdr.Vt[0 + c * hs];
                }
            }

            for (int r = 0; r < hs; ++r) {
                for (int c = 0; c < hs; ++c) {
                    double change = 0.0;
                    if (mode == REMOVE_ONLY || mode == FULL) {
                        change -= proj[r] * svdr.Vt[0 + c * hs];
                    }
                    if (mode == ADD_ONLY || mode == FULL) {
                        change += svdr.sigma[0] * svdr.U[r + 0 * hs] * svdr.Vt[0 + c * hs];
                    }
                    dd[r * hs + c] += (float)change;
                }
            }
        }
    }
}

struct QResult {
    float p;
    int rank;
    std::string top_tok;
    std::string gen;
};

static QResult do_query(rp::Model & model, rp::Context & ctx,
                        const rp::StateBuf & state,
                        const std::vector<llama_token> & toks,
                        llama_token tok_a, llama_token tok_b,
                        int n_vocab, int n_predict, uint32_t seed) {
    ctx.clear_memory();
    ctx.decode(rp::span<const llama_token>(toks.data(), toks.size() - 1));
    state.store_to(ctx.raw());
    ctx.decode_one(toks.back());

    const float * logits = llama_get_logits(ctx.raw());

    QResult qr;
    qr.p = rp::prob_of(logits[tok_a], logits[tok_b]);

    int top = 0;
    qr.rank = 0;
    for (int v = 1; v < n_vocab; ++v) {
        if (logits[v] > logits[top]) top = v;
        if (logits[v] > logits[tok_a]) qr.rank++;
    }
    qr.top_tok = rp::piece(ctx.raw(), (llama_token)top);

    if (n_predict > 0) {
        common_params_sampling sp;
        sp.seed = seed;
        rp::Generator gen(model, ctx, sp);
        llama_token last = toks.back();
        gen.accept_prompt(rp::span<const llama_token>(&last, 1));
        qr.gen = gen.run(n_predict).text;
        for (auto & c : qr.gen) if (c == '\n') c = ' ';
    }
    return qr;
}

int main(int argc, char ** argv) {
    auto args = rp::parse_common_args(argc, argv);
    if (args.help_requested) {
        std::printf("usage: %s -m MODEL [--layers 16-31] [--sigma-min 1.0] [-n 15]\n", argv[0]);
        return 0;
    }
    args.defaults(2048, 15);
    if (args.layer_range.empty()) args.layer_range = "16-31";
    if (!rp::require_model(args)) return 1;

    // experiment-specific arg: --sigma-min
    double sigma_min = 1.0;
    for (std::size_t i = 0; i < args.extra.size(); ++i) {
        const auto & a = args.extra[i];
        if (a == "--sigma-min" && i+1 < args.extra.size()) sigma_min = std::atof(args.extra[++i].c_str());
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
        int n_vocab = geom.n_vocab;

        auto layers = rp::parse_layer_range(args.layer_range, geom.n_layer);

        struct TestCase {
            const char * name;
            const char * prompt;
            const char * cal_from;
            const char * cal_to;
            const char * expect_from;
            const char * expect_to;
        };

        TestCase tests[] = {
            {"city: Paris->London",
             "Alice lives in Paris. Where does Alice live? Alice lives in",
             "Alice lives in Paris. Where does Alice live? Alice lives in",
             "Alice lives in London. Where does Alice live? Alice lives in",
             "Paris", "London"},
            {"city: Paris->London (narrative)",
             "Last year Alice moved to Paris. She enjoys living there. Alice currently lives in",
             "Alice lives in Paris. Where does Alice live? Alice lives in",
             "Alice lives in London. Where does Alice live? Alice lives in",
             "Paris", "London"},
            {"hat: red->green",
             "Alice has a red hat. What color is Alice's hat? Alice's hat is",
             "Alice has a red hat. What color is Alice's hat? Alice's hat is",
             "Alice has a green hat. What color is Alice's hat? Alice's hat is",
             "red", "green"},
            {"hat: red->green (narrative)",
             "Alice bought a beautiful red hat yesterday. She wore it today. The color of Alice's hat is",
             "Alice has a red hat. What color is Alice's hat? Alice's hat is",
             "Alice has a green hat. What color is Alice's hat? Alice's hat is",
             "red", "green"},
        };

        for (const auto & tc : tests) {
            llama_token tok_from = rp::find_token(vocab, tc.expect_from);
            llama_token tok_to   = rp::find_token(vocab, tc.expect_to);

            auto toks = rp::tokenize(vocab, tc.prompt);

            // capture target state
            ctx.clear_memory();
            ctx.decode(rp::span<const llama_token>(toks.data(), toks.size() - 1));
            rp::StateBuf state_orig(geom);
            state_orig.load_from(ctx.raw());

            // calibrate
            auto cal_from_toks = rp::tokenize(vocab, tc.cal_from);
            auto cal_to_toks   = rp::tokenize(vocab, tc.cal_to);

            ctx.clear_memory();
            ctx.decode(rp::span<const llama_token>(cal_from_toks.data(), cal_from_toks.size() - 1));
            rp::StateBuf state_cal_from(geom);
            state_cal_from.load_from(ctx.raw());

            ctx.clear_memory();
            ctx.decode(rp::span<const llama_token>(cal_to_toks.data(), cal_to_toks.size() - 1));
            rp::StateBuf state_cal_to(geom);
            state_cal_to.load_from(ctx.raw());

            // baseline
            auto bl = do_query(model, ctx, state_orig, toks, tok_from, tok_to,
                               n_vocab, args.n_predict, args.seed);

            std::printf("======================================================================\n");
            std::printf("TEST: %s\n", tc.name);
            std::printf("  prompt: \"%.70s\"\n", tc.prompt);
            std::printf("  baseline:     P(%s)=%.4f  rank=%d  top=%s  gen: %s\n",
                        tc.expect_from, bl.p, bl.rank, bl.top_tok.c_str(),
                        bl.gen.substr(0, 45).c_str());

            struct Mode {
                const char * label;
                EditMode mode;
            };
            Mode modes[] = {
                {"REMOVE-ONLY", REMOVE_ONLY},
                {"ADD-ONLY",    ADD_ONLY},
                {"FULL",        FULL},
            };

            for (const auto & m : modes) {
                rp::StateBuf edited(state_orig);
                apply_decomposed(edited, state_cal_from, state_cal_to,
                                layers, n_head, hs, sigma_min, m.mode);

                auto r = do_query(model, ctx, edited, toks, tok_from, tok_to,
                                  n_vocab, args.n_predict, args.seed);

                std::printf("  %-14s  P(%s)=%.4f  rank=%d  top=%s  gen: %s\n",
                            m.label, tc.expect_from, r.p, r.rank,
                            r.top_tok.c_str(), r.gen.substr(0, 45).c_str());
            }
            std::printf("\n");
            std::fflush(stdout);
        }

        std::fprintf(stderr, "done\n");
    });
}
