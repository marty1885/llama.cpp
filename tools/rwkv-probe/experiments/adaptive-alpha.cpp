// experiments/adaptive-alpha.cpp — test adaptive alpha scaling for state edits.
//
// The problem: a fixed alpha works for 2-entity prompts in experiments but
// fails in the CLI where the conversation state is much larger (chat template,
// system prompt, multi-turn history). The edit delta from a short prompt pair
// is tiny relative to the full state.
//
// This experiment:
//   1. Tests the same edit across increasing context lengths
//   2. Compares fixed alpha vs adaptive alpha (scaled by state_norm/delta_norm)
//   3. Tests SLERP-like norm-preserving application
//
// Usage:
//   llama-rwkv-adaptive-alpha -m model.gguf [--layers 16-31] [-n 15]

#include "rwkv_probe/state.h"
#include "rwkv_probe/generate.h"
#include "rwkv_probe/experiment.h"
#include "rwkv_probe/pipeline.h"
#include "rwkv_probe/edit.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace rp = rwkv_probe;

static double state_norm(const rp::StateBuf & s, const std::vector<int> & layers,
                         int n_head, int hs) {
    double sum = 0.0;
    for (int layer : layers) {
        for (int h = 0; h < n_head; ++h) {
            auto hv = s.s_head(layer, h);
            for (int i = 0; i < hs * hs; ++i) {
                sum += (double)hv[i] * (double)hv[i];
            }
        }
    }
    return std::sqrt(sum);
}

static double delta_norm(const rp::StateBuf & sa, const rp::StateBuf & sb,
                         const std::vector<int> & layers, int n_head, int hs) {
    double sum = 0.0;
    for (int layer : layers) {
        for (int h = 0; h < n_head; ++h) {
            auto ha = sa.s_head(layer, h);
            auto hb = sb.s_head(layer, h);
            for (int i = 0; i < hs * hs; ++i) {
                double d = (double)hb[i] - (double)ha[i];
                sum += d * d;
            }
        }
    }
    return std::sqrt(sum);
}

// NOTE: apply_delta replaced by rp::apply_full_delta from the framework.

// SLERP-based edit: instead of dst += alpha * delta, interpolate each head
// on the sphere between dst_h and (dst_h + delta_h), preserving the head's norm.
static void apply_slerp_edit(rp::StateBuf & dst,
                             const rp::StateBuf & sa, const rp::StateBuf & sb,
                             const std::vector<int> & layers, int n_head, int hs,
                             double t) {
    int n = hs * hs;
    for (int layer : layers) {
        for (int h = 0; h < n_head; ++h) {
            auto da = sa.s_head(layer, h);
            auto db = sb.s_head(layer, h);
            auto dd = dst.s_head(layer, h);

            // target = dst + delta
            std::vector<double> target(n);
            for (int i = 0; i < n; ++i) {
                target[i] = (double)dd[i] + (double)db[i] - (double)da[i];
            }

            // compute norms and dot
            double norm_src = 0.0, norm_tgt = 0.0, dot = 0.0;
            for (int i = 0; i < n; ++i) {
                norm_src += (double)dd[i] * (double)dd[i];
                norm_tgt += target[i] * target[i];
                dot      += (double)dd[i] * target[i];
            }
            norm_src = std::sqrt(norm_src);
            norm_tgt = std::sqrt(norm_tgt);

            if (norm_src < 1e-12 || norm_tgt < 1e-12) {
                // degenerate — fall back to linear
                for (int i = 0; i < n; ++i) {
                    dd[i] += (float)(t * ((double)db[i] - (double)da[i]));
                }
                continue;
            }

            double cos_omega = dot / (norm_src * norm_tgt);
            cos_omega = std::max(-1.0, std::min(1.0, cos_omega));
            double omega = std::acos(cos_omega);

            if (std::abs(omega) < 1e-6) {
                // nearly parallel — linear is fine
                for (int i = 0; i < n; ++i) {
                    dd[i] += (float)(t * ((double)db[i] - (double)da[i]));
                }
                continue;
            }

            double sin_omega = std::sin(omega);
            double w_src = std::sin((1.0 - t) * omega) / sin_omega;
            double w_tgt = std::sin(t * omega) / sin_omega;

            for (int i = 0; i < n; ++i) {
                dd[i] = (float)(w_src * (double)dd[i] + w_tgt * target[i]);
            }
        }
    }
}

struct QResult {
    float p;
    std::string gen;
};

static QResult do_query(rp::Model & model, rp::Context & ctx,
                        const rp::StateBuf & state,
                        const std::string & prefix, const std::string & qstr,
                        llama_token tok_a, llama_token tok_b,
                        const llama_vocab * vocab, int n_predict, uint32_t seed) {
    auto pr = rp::probe(model, ctx, state, prefix, qstr,
                        tok_a, tok_b, vocab, n_predict, seed);
    return {pr.p_binary, pr.generation};
}

int main(int argc, char ** argv) {
    auto args = rp::parse_common_args(argc, argv);
    args.defaults(4096, 15);

    if (args.help_requested) {
        std::printf("usage: %s -m MODEL [--layers 16-31] [-n 15]\n", argv[0]);
        return 0;
    }

    if (!rp::require_model(args)) return 1;
    if (!rp::reject_extra(args))  return 1;

    return rp::run_experiment([&] {
        auto env = rp::make_env(args);
        auto & model = env->model;
        auto & ctx   = env->ctx;
        const auto & geom  = env->geom;
        const auto * vocab = env->vocab;
        int n_head = geom.n_head;
        int hs = geom.head_size;

        std::string layer_range = args.layer_range.empty() ? "16-31" : args.layer_range;
        auto layers = rp::parse_layer_range(layer_range, geom.n_layer);

        llama_token tok_austin  = rp::find_token(vocab, "Austin");
        llama_token tok_chicago = rp::find_token(vocab, "Chicago");

        // ── edit pair (short, isolated) ──
        std::string edit_a = "Bob lives in Austin";
        std::string edit_b = "Bob lives in Chicago";

        auto toks_ea = rp::tokenize(vocab, edit_a);
        auto toks_eb = rp::tokenize(vocab, edit_b);

        ctx.clear_memory();
        ctx.decode(rp::span<const llama_token>(toks_ea.data(), toks_ea.size()));
        rp::StateBuf s_edit_a(geom);
        s_edit_a.load_from(ctx.raw());

        ctx.clear_memory();
        ctx.decode(rp::span<const llama_token>(toks_eb.data(), toks_eb.size()));
        rp::StateBuf s_edit_b(geom);
        s_edit_b.load_from(ctx.raw());

        double d_norm = delta_norm(s_edit_a, s_edit_b, layers, n_head, hs);
        std::printf("edit delta norm: %.2f\n\n", d_norm);

        // ── contexts of increasing length ──
        struct TestContext {
            std::string name;
            std::string prompt;
        };

        TestContext contexts[] = {
            {"short (2 entities)",
             "Alice lives in Madrid. Bob lives in Austin."},
            {"medium (4 entities)",
             "Alice lives in Madrid. Bob lives in Austin. Carol lives in London. Dave lives in Rome."},
            {"long (6 entities + details)",
             "Alice lives in Madrid and works as a teacher. Bob lives in Austin and has a car. "
             "Carol lives in London and plays piano. Dave lives in Rome and likes cooking. "
             "Eve lives in Berlin and studies physics. Frank lives in Tokyo and rides a bicycle."},
            {"very long (repeated context)",
             "Alice lives in Madrid and works as a teacher. She enjoys painting on weekends. "
             "Bob lives in Austin and has a car. He drives to work every day. "
             "Carol lives in London and plays piano. She performs at local venues. "
             "Dave lives in Rome and likes cooking. He makes pasta from scratch. "
             "Eve lives in Berlin and studies physics. She is working on her thesis. "
             "Frank lives in Tokyo and rides a bicycle. He commutes 30 minutes each way. "
             "Grace lives in Sydney and loves surfing. She goes to the beach every morning."},
        };

        std::string query = " Where does Bob live? Bob lives in";

        // header
        std::printf("%-8s  %-8s  %-6s  |  %8s  %10s  %10s  %8s  %s\n",
                    "context", "method", "alpha", "P(old)", "state_nrm", "delta_nrm", "ratio", "gen");
        std::printf("%s\n", std::string(8+2+8+2+6+3+3+8+2+10+2+10+2+8+2+30, '-').c_str());

        for (const auto & tc : contexts) {
            // capture base state
            auto toks = rp::tokenize(vocab, tc.prompt);
            ctx.clear_memory();
            ctx.decode(rp::span<const llama_token>(toks.data(), toks.size()));
            rp::StateBuf state_base(geom);
            state_base.load_from(ctx.raw());

            double s_norm = state_norm(state_base, layers, n_head, hs);

            // baseline
            auto bl = do_query(model, ctx, state_base, tc.prompt, query,
                               tok_austin, tok_chicago, vocab, args.n_predict, args.seed);
            std::printf("%-8s  %-8s  %-6s  |  %8.4f  %10.2f  %10s  %8s  %s\n",
                        tc.name.substr(0, 8).c_str(), "BASE", "-",
                        bl.p, s_norm, "-", "-",
                        bl.gen.substr(0, 28).c_str());

            // fixed alphas
            std::vector<double> fixed_alphas = {0.5, 1.0, 1.5, 2.0, 3.0, 5.0};
            for (double alpha : fixed_alphas) {
                rp::StateBuf edited(state_base);
                rp::apply_full_delta(edited, s_edit_a, s_edit_b, layers, alpha);

                double e_norm = state_norm(edited, layers, n_head, hs);
                double ratio = (alpha * d_norm) / s_norm;

                auto r = do_query(model, ctx, edited, tc.prompt, query,
                                  tok_austin, tok_chicago, vocab, args.n_predict, args.seed);

                char alpha_str[16];
                std::snprintf(alpha_str, sizeof alpha_str, "%.1f", alpha);
                std::printf("%-8s  %-8s  %-6s  |  %8.4f  %10.2f  %10.2f  %8.4f  %s\n",
                            tc.name.substr(0, 8).c_str(), "UNIFORM", alpha_str,
                            r.p, e_norm, alpha * d_norm, ratio,
                            r.gen.substr(0, 28).c_str());
            }

            // adaptive alpha: target ratio = 0.20
            for (double target_ratio : {0.15, 0.20, 0.25, 0.30}) {
                double adaptive_alpha = target_ratio * s_norm / d_norm;

                rp::StateBuf edited(state_base);
                rp::apply_full_delta(edited, s_edit_a, s_edit_b, layers, adaptive_alpha);

                auto r = do_query(model, ctx, edited, tc.prompt, query,
                                  tok_austin, tok_chicago, vocab, args.n_predict, args.seed);

                char alpha_str[16];
                std::snprintf(alpha_str, sizeof alpha_str, "%.2f", adaptive_alpha);
                char method_str[16];
                std::snprintf(method_str, sizeof method_str, "ADAPr%.2f", target_ratio);
                std::printf("%-8s  %-8s  %-6s  |  %8.4f  %10.2f  %10.2f  %8.4f  %s\n",
                            tc.name.substr(0, 8).c_str(), method_str, alpha_str,
                            r.p, state_norm(edited, layers, n_head, hs),
                            adaptive_alpha * d_norm, target_ratio,
                            r.gen.substr(0, 28).c_str());
            }

            // SLERP at fixed t values
            for (double t : {0.5, 1.0, 1.5}) {
                rp::StateBuf edited(state_base);
                apply_slerp_edit(edited, s_edit_a, s_edit_b, layers, n_head, hs, t);

                double e_norm = state_norm(edited, layers, n_head, hs);

                auto r = do_query(model, ctx, edited, tc.prompt, query,
                                  tok_austin, tok_chicago, vocab, args.n_predict, args.seed);

                char t_str[16];
                std::snprintf(t_str, sizeof t_str, "%.1f", t);
                std::printf("%-8s  %-8s  %-6s  |  %8.4f  %10.2f  %10s  %8s  %s\n",
                            tc.name.substr(0, 8).c_str(), "SLERP", t_str,
                            r.p, e_norm, "-", "-",
                            r.gen.substr(0, 28).c_str());
            }

            // SLERP + adaptive: scale t so the angular displacement is consistent
            // t_adaptive = target_ratio * state_norm / delta_norm
            // (same formula as adaptive alpha, but applied as SLERP parameter)
            for (double target_ratio : {0.15, 0.20, 0.25, 0.30}) {
                double adaptive_t = target_ratio * s_norm / d_norm;

                rp::StateBuf edited(state_base);
                apply_slerp_edit(edited, s_edit_a, s_edit_b, layers, n_head, hs, adaptive_t);

                double e_norm = state_norm(edited, layers, n_head, hs);

                auto r = do_query(model, ctx, edited, tc.prompt, query,
                                  tok_austin, tok_chicago, vocab, args.n_predict, args.seed);

                char t_str[16];
                std::snprintf(t_str, sizeof t_str, "%.2f", adaptive_t);
                char method_str[16];
                std::snprintf(method_str, sizeof method_str, "SL+A%.2f", target_ratio);
                std::printf("%-8s  %-8s  %-6s  |  %8.4f  %10.2f  %10.2f  %8.4f  %s\n",
                            tc.name.substr(0, 8).c_str(), method_str, t_str,
                            r.p, e_norm, adaptive_t * d_norm, target_ratio,
                            r.gen.substr(0, 28).c_str());
            }

            std::printf("\n");
            std::fflush(stdout);
        }

        std::fprintf(stderr, "done\n");
    });
}
