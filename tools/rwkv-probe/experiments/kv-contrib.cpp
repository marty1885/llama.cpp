// experiments/kv-contrib.cpp — measure per-token k*v contributions to WKV state.
//
// Processes a prompt token-by-token, capturing the WKV state after each token.
// For each token, computes the state delta (S_t - S_{t-1}) at every layer/head.
// This delta = decay_effect + k_t^T * v_t, i.e. it's the net state change
// that one token caused.
//
// Reports:
//   1. Per-token delta magnitude (Frobenius norm) at each layer — which tokens
//      cause the biggest state writes?
//   2. Rank structure of each delta — is it rank-1 as expected from k^T*v,
//      or does the decay make it higher rank?
//   3. Survival fraction — how much of token t's delta is still present in
//      the final state? Measured as ||project(S_final, delta_t)|| / ||delta_t||
//   4. Per-token breakdown correlated with actual token text.
//
// This tells us whether surgical rank-1 replacement of a specific token's
// contribution is feasible, or whether decay has made it unrecoverable.
//
// Usage:
//   llama-rwkv-kv-contrib -m model.gguf --prompt "Alice has a red hat..."
//       [--layers 16-31] [--top-k 10]

#include "rwkv_probe/state.h"
#include "rwkv_probe/experiment.h"
#include "rwkv_probe/generate.h"

#include "rwkv_probe/numeric.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

namespace rp = rwkv_probe;
using json = nlohmann::json;

// Frobenius norm of a head-sized matrix stored flat
static double frob_norm(rp::span<const float> v) {
    double s = 0.0;
    for (std::size_t i = 0; i < v.size(); ++i) {
        s += (double)v[i] * (double)v[i];
    }
    return std::sqrt(s);
}

// compute delta = b - a for a single head, return as vector<double>
static std::vector<double> head_delta(rp::span<const float> a, rp::span<const float> b) {
    std::vector<double> d(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        d[i] = (double)b[i] - (double)a[i];
    }
    return d;
}

// project vector x onto direction d: returns ||proj_d(x)|| / ||d||
// measures how much of d survives in x
static double projection_fraction(const std::vector<double> & d, rp::span<const float> x_raw) {
    // how much of delta d is present in the final state x_raw?
    // |<x,d>| / ||d||^2 = the scalar coefficient if you wrote x = alpha*d + orthogonal
    double dot = 0.0, norm_d = 0.0;
    for (std::size_t i = 0; i < d.size(); ++i) {
        dot    += d[i] * (double)x_raw[i];
        norm_d += d[i] * d[i];
    }
    if (norm_d < 1e-20) return 0.0;
    return std::abs(dot) / norm_d;
}

struct TokenContrib {
    int         pos;
    std::string token_text;
    // per-layer stats (indexed by layer offset within the selected range)
    std::vector<double> delta_norm;     // ||S_t - S_{t-1}|| per layer
    std::vector<double> sigma_ratio;    // sigma_1 / sum(sigma) — how rank-1 is the delta
    std::vector<double> survival;       // projection of delta onto final state
};

int main(int argc, char ** argv) {
    auto args = rp::parse_common_args(argc, argv);
    args.defaults(2048, 15);

    // experiment-specific args
    std::string prompt;
    int top_k = 10;

    for (std::size_t i = 0; i < args.extra.size(); ++i) {
        const std::string & a = args.extra[i];
        if      (a == "--prompt" && i+1 < args.extra.size()) prompt = args.extra[++i];
        else if (a == "--top-k"  && i+1 < args.extra.size()) top_k  = std::atoi(args.extra[++i].c_str());
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 1; }
    }

    if (args.help_requested) {
        std::printf("usage: %s -m MODEL --prompt TEXT|--tests FILE [--layers 0-31] [--top-k 10]\n", argv[0]);
        return 0;
    }

    if (!rp::require_model(args)) return 1;

    // collect prompts to analyze
    std::vector<std::pair<std::string, std::string>> prompts; // (name, text)
    if (!args.tests_path.empty()) {
        auto arr = rp::load_tests(args);
        for (const auto & e : arr) {
            std::string name = e.value("name", "unnamed");
            prompts.push_back({name, e.at("prompt_a").get<std::string>()});
        }
    } else if (!prompt.empty()) {
        prompts.push_back({"cli", prompt});
    } else {
        std::fprintf(stderr, "error: --prompt or --tests required\n");
        return 1;
    }

    return rp::run_experiment([&] {
        auto env = rp::make_env(args);
        auto & ctx   = env->ctx;
        const auto & geom  = env->geom;
        const auto * vocab = env->vocab;
        int n_head = geom.n_head;
        int hs = geom.head_size;

        std::string layer_range = args.layer_range.empty() ? "0-31" : args.layer_range;
        auto layers = rp::parse_layer_range(layer_range, geom.n_layer);
        int n_layers = (int)layers.size();

        std::printf("n_layer=%d  n_head=%d  head_size=%d  analyzing layers %d-%d\n\n",
                    geom.n_layer, n_head, hs, layers.front(), layers.back());

        for (const auto & [pname, ptext] : prompts) {
            auto tokens = rp::tokenize(vocab, ptext);
            int n_tokens = (int)tokens.size();

            std::printf("======================================================================\n");
            std::printf("PROMPT: \"%s\" (%d tokens)\n", pname.c_str(), n_tokens);
            std::printf("  \"%s\"\n\n", ptext.substr(0, 80).c_str());

            // ── step 1: process token by token, capture state after each ──
            std::vector<rp::StateBuf> states;
            states.reserve(n_tokens + 1);

            // initial state (before any tokens)
            ctx.clear_memory();
            // decode first token to initialize, then capture
            // actually, capture the zero state first
            {
                rp::StateBuf s0(geom);
                s0.load_from(ctx.raw());
                states.push_back(std::move(s0));
            }

            std::fprintf(stderr, "  processing %d tokens one-by-one...\n", n_tokens);
            for (int t = 0; t < n_tokens; ++t) {
                if (t == 0) {
                    ctx.clear_memory();
                }
                ctx.decode_one(tokens[t]);
                rp::StateBuf st(geom);
                st.load_from(ctx.raw());
                states.push_back(std::move(st));
            }

            // final state reference
            const auto & state_final = states.back();

            // ── step 2: compute per-token contributions ──
            std::vector<TokenContrib> contribs(n_tokens);

            for (int t = 0; t < n_tokens; ++t) {
                auto & tc = contribs[t];
                tc.pos = t;

                // get token text
                char buf[128];
                int len = llama_token_to_piece(vocab, tokens[t], buf, sizeof(buf) - 1, 0, true);
                if (len < 0) len = 0;
                buf[len] = '\0';
                tc.token_text = buf;

                tc.delta_norm.resize(n_layers);
                tc.sigma_ratio.resize(n_layers);
                tc.survival.resize(n_layers);

                for (int li = 0; li < n_layers; ++li) {
                    int layer = layers[li];
                    double total_delta_norm = 0.0;
                    double total_sigma1 = 0.0;
                    double total_sigma_sum = 0.0;
                    double total_survival = 0.0;
                    int    heads_with_signal = 0;

                    for (int h = 0; h < n_head; ++h) {
                        auto h_prev  = states[t].s_head(layer, h);
                        auto h_cur   = states[t + 1].s_head(layer, h);
                        auto h_final = state_final.s_head(layer, h);

                        auto delta = head_delta(h_prev, h_cur);

                        double dnorm = 0.0;
                        for (double d : delta) dnorm += d * d;
                        dnorm = std::sqrt(dnorm);

                        if (dnorm < 1e-10) continue;
                        heads_with_signal++;

                        total_delta_norm += dnorm;

                        // SVD of delta reshaped as hs x hs matrix
                        auto svdr = rp::numeric::svd(delta.data(), hs);
                        {
                            double s1 = svdr.sigma[0];
                            double ssum = 0.0;
                            for (int si = 0; si < (int)svdr.sigma.size(); ++si) ssum += svdr.sigma[si];
                            total_sigma1 += s1;
                            total_sigma_sum += ssum;
                        }

                        // survival: how much of this delta is in final state
                        total_survival += projection_fraction(delta, h_final);
                    }

                    tc.delta_norm[li] = total_delta_norm;
                    tc.sigma_ratio[li] = (total_sigma_sum > 0) ?
                        total_sigma1 / total_sigma_sum : 0.0;
                    tc.survival[li] = (heads_with_signal > 0) ?
                        total_survival / heads_with_signal : 0.0;
                }
            }

            // ── step 3: report — all tokens, summed across layers ──
            std::printf("  ALL TOKENS (summed across layers %d-%d):\n",
                        layers.front(), layers.back());
            std::printf("  %4s  %-20s  %12s  %8s  %8s\n",
                        "pos", "token", "Σ||delta||", "σ₁/Σσ", "survival");
            std::printf("  %s\n", std::string(4 + 2 + 20 + 2 + 12 + 2 + 8 + 2 + 8, '-').c_str());

            for (int t = 0; t < n_tokens; ++t) {
                const auto & tc = contribs[t];
                double sum_delta = 0.0, sum_sr = 0.0, sum_surv = 0.0;
                for (int li = 0; li < n_layers; ++li) {
                    sum_delta += tc.delta_norm[li];
                    sum_sr    += tc.sigma_ratio[li];
                    sum_surv  += tc.survival[li];
                }
                double avg_sr   = n_layers > 0 ? sum_sr / n_layers : 0.0;
                double avg_surv = n_layers > 0 ? sum_surv / n_layers : 0.0;

                // escape the token text for display
                std::string disp = tc.token_text;
                for (auto & ch : disp) if (ch == '\n') ch = '\\';
                if (disp.size() > 18) disp = disp.substr(0, 18) + "..";

                std::printf("  %4d  %-20s  %12.4f  %8.4f  %8.4f\n",
                            t, disp.c_str(), sum_delta, avg_sr, avg_surv);
            }

            // ── step 4: top-k tokens by delta norm, per layer ──
            std::printf("\n  TOP-%d TOKENS BY DELTA NORM (per layer):\n", top_k);

            // pick a few representative layers to show detail
            std::vector<int> detail_layers;
            if (n_layers <= 6) {
                detail_layers = layers;
            } else {
                // show first, 1/4, 1/2, 3/4, last
                detail_layers = {
                    layers[0],
                    layers[n_layers / 4],
                    layers[n_layers / 2],
                    layers[3 * n_layers / 4],
                    layers.back()
                };
            }

            for (int layer : detail_layers) {
                int li = layer - layers[0]; // offset into our range

                // sort by delta norm at this layer
                std::vector<int> order(n_tokens);
                std::iota(order.begin(), order.end(), 0);
                std::sort(order.begin(), order.end(), [&](int a, int b) {
                    return contribs[a].delta_norm[li] > contribs[b].delta_norm[li];
                });

                std::printf("\n  Layer %d:\n", layer);
                std::printf("    %4s  %-20s  %12s  %8s  %8s\n",
                            "pos", "token", "||delta||", "σ₁/Σσ", "survival");

                int shown = std::min(top_k, n_tokens);
                for (int i = 0; i < shown; ++i) {
                    int t = order[i];
                    const auto & tc = contribs[t];

                    std::string disp = tc.token_text;
                    for (auto & ch : disp) if (ch == '\n') ch = '\\';
                    if (disp.size() > 18) disp = disp.substr(0, 18) + "..";

                    std::printf("    %4d  %-20s  %12.4f  %8.4f  %8.4f\n",
                                t, disp.c_str(),
                                tc.delta_norm[li], tc.sigma_ratio[li], tc.survival[li]);
                }
            }

            // ── step 5: survival heatmap — does position matter? ──
            std::printf("\n  SURVIVAL BY POSITION (mean across heads, selected layers):\n");
            std::printf("  %4s  %-20s", "pos", "token");
            for (int layer : detail_layers) {
                char hdr[16];
                std::snprintf(hdr, sizeof hdr, "L%d", layer);
                std::printf("  %8s", hdr);
            }
            std::printf("\n");
            std::printf("  %s\n", std::string(4 + 2 + 20 + detail_layers.size() * 10, '-').c_str());

            for (int t = 0; t < n_tokens; ++t) {
                const auto & tc = contribs[t];
                std::string disp = tc.token_text;
                for (auto & ch : disp) if (ch == '\n') ch = '\\';
                if (disp.size() > 18) disp = disp.substr(0, 18) + "..";

                std::printf("  %4d  %-20s", t, disp.c_str());
                for (int layer : detail_layers) {
                    int li = layer - layers[0];
                    std::printf("  %8.4f", tc.survival[li]);
                }
                std::printf("\n");
            }

            std::printf("\n");
            std::fflush(stdout);
        }

        std::fprintf(stderr, "done\n");
    });
}
