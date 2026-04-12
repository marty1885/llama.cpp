// experiments/order-rotation.cpp — test whether entity introduction order
// rotates the SVD address/value directions.
//
// Hypothesis: RWKV rotates the u₁/v₁ directions of a fact delta depending
// on when the entity was introduced in the sequence. If true, a calibration
// delta from position 1 would be misaligned with the same entity at position 5,
// explaining multi-entity edit failures.
//
// Method: Place the SAME entity (Alice, Paris→Madrid) at positions 1-5 among
// filler entities. SVD the delta at each position. Report cosine similarity
// of u₁ and v₁ across all position pairs and against the isolated (no-filler)
// calibration delta.
//
// Usage:
//   llama-rwkv-order-rotation -m model.gguf [--layers 16-31]

#include "rwkv_probe/state.h"
#include "rwkv_probe/pipeline.h"
#include "rwkv_probe/head_decomp.h"
#include "rwkv_probe/experiment.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace rp = rwkv_probe;

static double cosine_sim(const std::vector<double> & a, const std::vector<double> & b) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    if (na < 1e-20 || nb < 1e-20) return 0.0;
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

int main(int argc, char ** argv) {
    auto args = rp::parse_common_args(argc, argv);
    args.defaults(/*ctx=*/2048, /*predict=*/15);
    if (args.layer_range.empty()) args.layer_range = "16-31";

    if (args.help_requested) {
        std::printf("usage: %s -m MODEL [--layers 16-31]\n", argv[0]);
        return 0;
    }
    if (!rp::require_model(args)) return 1;
    if (!rp::reject_extra(args)) return 1;

    return rp::run_experiment([&] {
        auto env = rp::make_env(args);
        auto & model = env->model;
        auto & ctx   = env->ctx;
        const auto & geom  = env->geom;
        const auto * vocab = env->vocab;
        int n_head = geom.n_head;

        auto layers = rp::parse_layer_range(args.layer_range, geom.n_layer);

        std::printf("layers: %d-%d (%zu)  n_head=%d  head_size=%d\n\n",
                    layers.front(), layers.back(), layers.size(), n_head, geom.head_size);

        // ── The target entity and its edit ──────────────────────────────
        const std::string target_name = "Alice";
        const std::string old_city    = "Paris";
        const std::string new_city    = "Madrid";

        // Filler entities (enough for 4 slots around the target)
        struct Filler {
            std::string name;
            std::string city;
        };
        Filler fillers[] = {
            {"Bob",   "Tokyo"},
            {"Carol", "London"},
            {"Dave",  "Rome"},
            {"Eve",   "Berlin"},
        };
        constexpr int n_fillers = 4;

        // ── Position conditions ─────────────────────────────────────────
        // pos 0: Alice is 1st of 5
        // pos 1: Alice is 2nd of 5
        // ...
        // pos 4: Alice is 5th of 5
        // pos 5: Alice alone (no fillers) — the calibration baseline

        int n_positions = 6; // 0..4 = position among 5, 5 = isolated

        // For each position, build prompt_a (old city) and prompt_b (new city)
        struct Condition {
            std::string label;
            std::string prompt_a;
            std::string prompt_b;
        };
        std::vector<Condition> conditions;

        for (int pos = 0; pos < 5; ++pos) {
            std::string pa, pb;
            int filler_idx = 0;
            for (int slot = 0; slot < 5; ++slot) {
                if (slot > 0) { pa += " "; pb += " "; }
                if (slot == pos) {
                    pa += target_name + " lives in " + old_city + ".";
                    pb += target_name + " lives in " + new_city + ".";
                } else {
                    pa += fillers[filler_idx].name + " lives in " + fillers[filler_idx].city + ".";
                    pb += fillers[filler_idx].name + " lives in " + fillers[filler_idx].city + ".";
                    filler_idx++;
                }
            }
            conditions.push_back({
                "pos" + std::to_string(pos+1) + "_of_5",
                pa, pb
            });
        }

        // isolated: just Alice
        conditions.push_back({
            "isolated",
            target_name + " lives in " + old_city + ".",
            target_name + " lives in " + new_city + ".",
        });

        // ── Capture states and decompose ─────────────────────────────────
        // decomps[cond_idx] = vector of HeadDecomp for that condition
        std::vector<std::vector<rp::HeadDecomp>> all_decomps;

        for (int ci = 0; ci < n_positions; ++ci) {
            const auto & cond = conditions[ci];
            std::fprintf(stderr, "capturing %s ...\n", cond.label.c_str());

            auto sa = rp::capture(ctx, geom, vocab, cond.prompt_a);
            auto sb = rp::capture(ctx, geom, vocab, cond.prompt_b);

            auto decomps = rp::decompose(sa, sb, layers);
            all_decomps.push_back(std::move(decomps));

            std::printf("  %s:\n", cond.label.c_str());
            std::printf("    A: %s\n", cond.prompt_a.c_str());
            std::printf("    B: %s\n", cond.prompt_b.c_str());
        }

        // ── Identify top heads by sigma (use isolated condition as reference) ──
        const auto & ref_decomps = all_decomps.back(); // isolated
        struct HeadRef { int idx; double sigma; double r1; };
        std::vector<HeadRef> ranked;
        for (int i = 0; i < (int)ref_decomps.size(); ++i) {
            ranked.push_back({i, ref_decomps[i].sigma, ref_decomps[i].r1_ratio});
        }
        std::sort(ranked.begin(), ranked.end(),
                  [](const HeadRef & a, const HeadRef & b) { return a.sigma > b.sigma; });

        // Show top 20 heads
        int n_show = std::min(20, (int)ranked.size());
        std::printf("\n══════════════════════════════════════════════════════════\n");
        std::printf("TOP %d HEADS BY SIGMA (from isolated condition)\n", n_show);
        std::printf("══════════════════════════════════════════════════════════\n\n");

        for (int hi = 0; hi < n_show; ++hi) {
            int idx = ranked[hi].idx;
            const auto & ref = ref_decomps[idx];

            std::printf("── L%d H%d  sigma=%.4f  r1=%.4f ──\n",
                        ref.layer, ref.head, ref.sigma, ref.r1_ratio);

            // Print sigma and r1 for each condition at this head
            std::printf("  %-12s  %8s  %8s\n", "condition", "sigma", "r1");
            for (int ci = 0; ci < n_positions; ++ci) {
                const auto & d = all_decomps[ci][idx];
                std::printf("  %-12s  %8.4f  %8.4f\n",
                            conditions[ci].label.c_str(), d.sigma, d.r1_ratio);
            }

            // Cosine similarity matrix for u₁ (value direction)
            std::printf("\n  u₁ cosine similarity (value direction):\n");
            std::printf("  %12s", "");
            for (int ci = 0; ci < n_positions; ++ci)
                std::printf("  %8s", conditions[ci].label.c_str());
            std::printf("\n");

            for (int ci = 0; ci < n_positions; ++ci) {
                std::printf("  %-12s", conditions[ci].label.c_str());
                for (int cj = 0; cj < n_positions; ++cj) {
                    double sim = cosine_sim(all_decomps[ci][idx].u,
                                           all_decomps[cj][idx].u);
                    std::printf("  %8.4f", sim);
                }
                std::printf("\n");
            }

            // Cosine similarity matrix for v₁ (address direction)
            std::printf("\n  v₁ cosine similarity (address direction):\n");
            std::printf("  %12s", "");
            for (int ci = 0; ci < n_positions; ++ci)
                std::printf("  %8s", conditions[ci].label.c_str());
            std::printf("\n");

            for (int ci = 0; ci < n_positions; ++ci) {
                std::printf("  %-12s", conditions[ci].label.c_str());
                for (int cj = 0; cj < n_positions; ++cj) {
                    double sim = cosine_sim(all_decomps[ci][idx].v,
                                           all_decomps[cj][idx].v);
                    std::printf("  %8.4f", sim);
                }
                std::printf("\n");
            }

            std::printf("\n");
        }

        // ── Summary: mean |cos_sim| of isolated vs each position ──────────
        std::printf("══════════════════════════════════════════════════════════\n");
        std::printf("SUMMARY: mean |cos_sim| with isolated (top %d heads)\n", n_show);
        std::printf("══════════════════════════════════════════════════════════\n\n");
        std::printf("  %-12s  %10s  %10s\n", "condition", "mean|u₁|", "mean|v₁|");
        std::printf("  %s\n", std::string(36, '-').c_str());

        for (int ci = 0; ci < n_positions - 1; ++ci) { // skip self (isolated)
            double u_sum = 0, v_sum = 0;
            int count = 0;
            for (int hi = 0; hi < n_show; ++hi) {
                int idx = ranked[hi].idx;
                if (all_decomps[ci][idx].sigma < 1e-6) continue;
                if (ref_decomps[idx].sigma < 1e-6) continue;
                u_sum += std::abs(cosine_sim(all_decomps[ci][idx].u,
                                             ref_decomps[idx].u));
                v_sum += std::abs(cosine_sim(all_decomps[ci][idx].v,
                                             ref_decomps[idx].v));
                count++;
            }
            std::printf("  %-12s  %10.4f  %10.4f  (n=%d)\n",
                        conditions[ci].label.c_str(),
                        count > 0 ? u_sum / count : 0.0,
                        count > 0 ? v_sum / count : 0.0,
                        count);
        }

        // Also show adjacent position similarities
        std::printf("\n  Adjacent position similarity (pos N vs pos N+1):\n");
        std::printf("  %-16s  %10s  %10s\n", "pair", "mean|u₁|", "mean|v₁|");
        std::printf("  %s\n", std::string(40, '-').c_str());

        for (int ci = 0; ci < 4; ++ci) { // pos1-2, 2-3, 3-4, 4-5
            double u_sum = 0, v_sum = 0;
            int count = 0;
            for (int hi = 0; hi < n_show; ++hi) {
                int idx = ranked[hi].idx;
                if (all_decomps[ci][idx].sigma < 1e-6) continue;
                if (all_decomps[ci+1][idx].sigma < 1e-6) continue;
                u_sum += std::abs(cosine_sim(all_decomps[ci][idx].u,
                                             all_decomps[ci+1][idx].u));
                v_sum += std::abs(cosine_sim(all_decomps[ci][idx].v,
                                             all_decomps[ci+1][idx].v));
                count++;
            }
            std::printf("  pos%d vs pos%d     %10.4f  %10.4f  (n=%d)\n",
                        ci+1, ci+2,
                        count > 0 ? u_sum / count : 0.0,
                        count > 0 ? v_sum / count : 0.0,
                        count);
        }

        std::printf("\n");
        std::fprintf(stderr, "done\n");
    });
}
