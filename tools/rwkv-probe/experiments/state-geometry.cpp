// experiments/state-geometry.cpp — measure how safe/dangerous recurrent states
// diverge across layers.
//
// For each safe/dangerous pair, captures the S state at all layers after the
// prompt pass and computes per-layer distance metrics. Answers the basic
// question: are the states at layers 1+ actually different, or do they converge?
//
// Usage:
//   llama-rwkv-state-geometry -m model.gguf --prompts paired_prompts.json
//
// Output: TSV table to stdout with per-layer, per-pair distances.

#include "rwkv_probe/state.h"
#include "rwkv_probe/generate.h"
#include "rwkv_probe/experiment.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

namespace rp = rwkv_probe;
using json = nlohmann::json;

// -- distance metrics --------------------------------------------------------

static double l2_distance(rp::span<const float> a, rp::span<const float> b) {
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        double d = (double) a[i] - (double) b[i];
        sum += d * d;
    }
    return std::sqrt(sum);
}

static double cosine_similarity(rp::span<const float> a, rp::span<const float> b) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        dot += (double) a[i] * (double) b[i];
        na  += (double) a[i] * (double) a[i];
        nb  += (double) b[i] * (double) b[i];
    }
    double denom = std::sqrt(na) * std::sqrt(nb);
    return denom > 0.0 ? dot / denom : 0.0;
}

static double l2_norm(rp::span<const float> v) {
    double sum = 0.0;
    for (std::size_t i = 0; i < v.size(); ++i) {
        sum += (double) v[i] * (double) v[i];
    }
    return std::sqrt(sum);
}

int main(int argc, char ** argv) {
    auto args = rp::parse_common_args(argc, argv);
    if (args.help_requested) {
        std::printf("usage: %s -m MODEL --prompts FILE [--n-ctx N]\n", argv[0]);
        return 0;
    }
    args.defaults(2048, 0);
    if (!rp::require_model(args)) return 1;

    // experiment-specific arg: --prompts FILE
    std::string prompts_path;
    for (std::size_t i = 0; i < args.extra.size(); ++i) {
        const auto & a = args.extra[i];
        if (a == "--prompts" && i+1 < args.extra.size()) prompts_path = args.extra[++i];
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 1; }
    }
    if (prompts_path.empty()) {
        std::fprintf(stderr, "error: --prompts required\n");
        return 1;
    }

    // -- load prompts and group into pairs ------------------------------------
    json raw = rp::load_json_file(prompts_path);

    struct Prompt {
        std::string id;
        std::string pair_id;
        std::string label;
        std::string text;
    };

    std::vector<Prompt> prompts;
    for (const auto & e : raw) {
        prompts.push_back({
            e.value("id", ""),
            e.value("pair_id", ""),
            e.value("label", ""),
            e.at("prompt").get<std::string>(),
        });
    }
    std::fprintf(stderr, "loaded %zu prompts\n", prompts.size());

    return rp::run_experiment([&] {
        auto env = rp::make_env(args);
        auto & ctx   = env->ctx;
        const auto & geom  = env->geom;

        std::fprintf(stderr, "n_layer=%d  n_embd=%d  n_embd_s=%d  head_size=%d  n_head=%d\n",
                     geom.n_layer, geom.n_embd, geom.n_embd_s, geom.head_size, geom.n_head);

        // -- run each prompt, capture state ----------------------------------
        std::map<std::string, rp::StateBuf> states;  // id -> captured state

        for (std::size_t pi = 0; pi < prompts.size(); ++pi) {
            const auto & p = prompts[pi];
            std::fprintf(stderr, "[%zu/%zu] %s (%s): %s\n",
                         pi + 1, prompts.size(), p.id.c_str(), p.label.c_str(),
                         p.text.substr(0, 60).c_str());

            ctx.clear_memory();
            auto tokens = rp::tokenize(env->model.vocab(), p.text);
            if (ctx.decode(rp::span<const llama_token>(tokens.data(), tokens.size())) != 0) {
                std::fprintf(stderr, "  decode failed, skipping\n");
                continue;
            }

            rp::StateBuf state(geom);
            state.load_from(ctx.raw());
            states.emplace(p.id, std::move(state));
        }

        // -- group pairs -----------------------------------------------------
        struct Pair {
            std::string pair_id;
            std::string safe_id;
            std::string dang_id;
        };
        std::vector<Pair> pairs;
        std::map<std::string, std::string> pair_safe, pair_dang;  // pair_id -> prompt id
        for (const auto & p : prompts) {
            if (p.label == "safe")      pair_safe[p.pair_id] = p.id;
            if (p.label == "dangerous") pair_dang[p.pair_id] = p.id;
        }
        for (const auto & kv : pair_safe) {
            auto it = pair_dang.find(kv.first);
            if (it != pair_dang.end() &&
                states.count(kv.second) && states.count(it->second)) {
                pairs.push_back({kv.first, kv.second, it->second});
            }
        }
        std::sort(pairs.begin(), pairs.end(),
                  [](const Pair & a, const Pair & b) { return a.pair_id < b.pair_id; });

        std::fprintf(stderr, "\n%zu valid pairs\n\n", pairs.size());

        // -- per-layer, per-pair distances (TSV) -----------------------------
        std::printf("%-5s  %-12s  %10s  %12s  %12s  %10s  %11s  %s\n",
                    "layer", "pair", "l2_dist", "l2_norm_safe", "l2_norm_dang",
                    "cosine_sim", "max_head_l2", "max_head_idx");

        // accumulators for per-layer mean
        std::vector<double> mean_l2(geom.n_layer, 0.0);
        std::vector<double> mean_cos(geom.n_layer, 0.0);

        for (int il = 0; il < geom.n_layer; ++il) {
            for (const auto & pair : pairs) {
                const auto & s_safe = states.at(pair.safe_id);
                const auto & s_dang = states.at(pair.dang_id);

                auto wkv_safe = s_safe.s_wkv(il);
                auto wkv_dang = s_dang.s_wkv(il);

                double dist = l2_distance(wkv_safe, wkv_dang);
                double ns   = l2_norm(wkv_safe);
                double nd   = l2_norm(wkv_dang);
                double cos   = cosine_similarity(wkv_safe, wkv_dang);

                // per-head breakdown: find the head with max L2 distance
                double max_head_l2 = 0.0;
                int    max_head_idx = 0;
                for (int h = 0; h < geom.n_head; ++h) {
                    auto hs = s_safe.s_head(il, h);
                    auto hd = s_dang.s_head(il, h);
                    double hd_l2 = l2_distance(hs, hd);
                    if (hd_l2 > max_head_l2) {
                        max_head_l2 = hd_l2;
                        max_head_idx = h;
                    }
                }

                std::printf("L%-4d  %-12s  %10.4f  %12.4f  %12.4f  %10.6f  %11.4f  %d\n",
                            il, pair.pair_id.c_str(), dist, ns, nd, cos,
                            max_head_l2, max_head_idx);

                mean_l2[il]  += dist;
                mean_cos[il] += cos;
            }
        }

        // -- per-layer summary -----------------------------------------------
        std::fprintf(stderr, "\n=== PER-LAYER MEAN (across %zu pairs) ===\n", pairs.size());
        std::fprintf(stderr, "%-5s  %12s  %12s\n", "layer", "mean_l2", "mean_cosine");
        std::fprintf(stderr, "-----  ------------  ------------\n");
        for (int il = 0; il < geom.n_layer; ++il) {
            double ml2 = mean_l2[il]  / (double) pairs.size();
            double mcs = mean_cos[il] / (double) pairs.size();
            std::fprintf(stderr, "L%-4d  %12.4f  %12.6f\n", il, ml2, mcs);
        }

        // -- cross-pair distances (control) ----------------------------------
        // compare within-pair distance to across-pair distance at a few layers
        std::fprintf(stderr, "\n=== WITHIN-PAIR vs CROSS-PAIR distances (layers 0, 8, 16, 24, 31) ===\n");
        const int check_layers[] = {0, 8, 16, 24, geom.n_layer - 1};
        for (int il : check_layers) {
            if (il >= geom.n_layer) continue;
            double within_sum = 0.0;
            int    within_n   = 0;
            double cross_sum  = 0.0;
            int    cross_n    = 0;

            // within-pair: safe vs dangerous in same pair
            for (const auto & pair : pairs) {
                auto ws = states.at(pair.safe_id).s_wkv(il);
                auto wd = states.at(pair.dang_id).s_wkv(il);
                within_sum += l2_distance(ws, wd);
                within_n++;
            }

            // cross-pair: safe of pair A vs dangerous of pair B (all combos)
            for (std::size_t a = 0; a < pairs.size(); ++a) {
                for (std::size_t b = 0; b < pairs.size(); ++b) {
                    if (a == b) continue;
                    auto ws = states.at(pairs[a].safe_id).s_wkv(il);
                    auto wd = states.at(pairs[b].dang_id).s_wkv(il);
                    cross_sum += l2_distance(ws, wd);
                    cross_n++;
                }
            }

            double within_mean = within_n > 0 ? within_sum / within_n : 0.0;
            double cross_mean  = cross_n  > 0 ? cross_sum  / cross_n  : 0.0;
            std::fprintf(stderr, "  L%-4d  within=%.4f  cross=%.4f  ratio=%.3f\n",
                         il, within_mean, cross_mean,
                         cross_mean > 0 ? within_mean / cross_mean : 0.0);
        }

        // -- all-vs-all cosine at layer 0 (sanity check for known PCA separation) --
        std::fprintf(stderr, "\n=== LAYER 0 ALL-VS-ALL COSINE ===\n");
        std::fprintf(stderr, "%14s", "");
        for (const auto & p : prompts) {
            std::fprintf(stderr, "  %6s", p.id.c_str());
        }
        std::fprintf(stderr, "\n");
        for (const auto & pa : prompts) {
            if (!states.count(pa.id)) continue;
            std::fprintf(stderr, "%14s", pa.id.c_str());
            for (const auto & pb : prompts) {
                if (!states.count(pb.id)) continue;
                auto a = states.at(pa.id).s_wkv(0);
                auto b = states.at(pb.id).s_wkv(0);
                std::fprintf(stderr, "  %6.3f", cosine_similarity(a, b));
            }
            std::fprintf(stderr, "\n");
        }

        std::fprintf(stderr, "\ndone\n");
    });
}
