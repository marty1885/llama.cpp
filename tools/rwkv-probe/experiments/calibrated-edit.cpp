// experiments/calibrated-edit.cpp — per-head calibrated state editing.
//
// Instead of applying the delta uniformly across all heads, calibrate
// per-head edit strength using a single-token probe:
//
//   1. Process shared prefix up to the diverging word → capture S_prefix
//   2. Decode token A (e.g. "red") from S_prefix → S_after_A
//   3. Decode token B (e.g. "yellow") from S_prefix → S_after_B
//   4. Per head: SVD(S_after_B - S_after_A) → σ₁ tells how strongly that
//      head participated in writing the color fact
//   5. Normalize σ₁ across heads → per-head weight w_h
//   6. Apply full-prompt delta with per-head weight: dst_h += w_h * alpha * delta_h
//
// Heads that the model uses to write the target property get the full edit;
// heads that encode structural/other info get scaled down → less spillover.
//
// Usage:
//   llama-rwkv-calibrated-edit -m model.gguf --tests entity_edit_tests.json
//       [--layers 16-31] [-n 20]

#include "rwkv_probe/state.h"
#include "rwkv_probe/generate.h"
#include "rwkv_probe/experiment.h"
#include "rwkv_probe/pipeline.h"
#include "rwkv_probe/edit.h"
#include "rwkv_probe/head_decomp.h"

#include "rwkv_probe/numeric.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace rp = rwkv_probe;
using json = nlohmann::json;

struct ControlQuery {
    std::string query;
    std::string expect;
};

struct EditTest {
    std::string name;
    std::string prompt_a, prompt_b;
    std::string query_target;
    std::string expect_a_target, expect_b_target;
    // the shared prefix and the two diverging words for calibration
    std::string cal_prefix;
    std::string cal_word_a, cal_word_b;
    // isolated edit sentences — just the target entity's sentence
    // e.g. "Bob has a green hat." / "Bob has a yellow hat."
    std::string edit_sentence_a, edit_sentence_b;
    std::vector<ControlQuery> controls;
};

// compute per-head calibration weights from single-token probe.
// returns weights[layer_idx][head] normalized so max = 1.0 within each layer.
static std::vector<std::vector<double>> compute_calibration(
        rp::Context & ctx,
        const rp::StateBuf & state_prefix,
        llama_token tok_a, llama_token tok_b,
        const std::vector<int> & layers, int n_head, int hs,
        const rp::ModelGeometry & geom) {
    int n_layers = (int)layers.size();

    // decode tok_a from prefix state
    rp::StateBuf s_after_a(geom);
    {
        state_prefix.store_to(ctx.raw());
        ctx.decode_one(tok_a);
        s_after_a.load_from(ctx.raw());
    }

    // decode tok_b from prefix state
    rp::StateBuf s_after_b(geom);
    {
        state_prefix.store_to(ctx.raw());
        ctx.decode_one(tok_b);
        s_after_b.load_from(ctx.raw());
    }

    // per-head: SVD of (s_after_b - s_after_a), extract σ₁
    std::vector<std::vector<double>> weights(n_layers, std::vector<double>(n_head, 0.0));

    for (int li = 0; li < n_layers; ++li) {
        int layer = layers[li];
        double max_sigma = 0.0;

        for (int h = 0; h < n_head; ++h) {
            auto ha = s_after_a.s_head(layer, h);
            auto hb = s_after_b.s_head(layer, h);

            std::vector<double> delta(hs * hs);
            for (int r = 0; r < hs; ++r) {
                for (int c = 0; c < hs; ++c) {
                    delta[r * hs + c] = (double)hb[r * hs + c] - (double)ha[r * hs + c];
                }
            }

            auto svdr = rp::numeric::svd(delta.data(), hs);
            {
                double s1 = svdr.sigma[0];
                weights[li][h] = s1;
                if (s1 > max_sigma) max_sigma = s1;
            }
        }

        // normalize so max weight = 1.0 within each layer
        if (max_sigma > 0.0) {
            for (int h = 0; h < n_head; ++h) {
                weights[li][h] /= max_sigma;
            }
        }
    }

    return weights;
}

// NOTE: apply_weighted_delta and apply_uniform_delta replaced by
// rp::apply_weighted_delta and rp::apply_full_delta from the framework.

// thin wrapper: maps rp::ProbeResult to the (p_target, gen) pair used below.
struct QueryResult {
    float p_target;
    std::string gen;
};

static QueryResult query_with_state(
        rp::Model & model, rp::Context & ctx,
        const rp::StateBuf & state,
        const std::string & prefix,
        const std::string & query,
        llama_token tok_expected, llama_token tok_alt,
        const llama_vocab * vocab,
        int n_predict, uint32_t seed) {
    auto pr = rp::probe(model, ctx, state, prefix, query,
                        tok_expected, tok_alt, vocab, n_predict, seed);
    return {pr.p_binary, pr.generation};
}

// Self-calibrated edit: weight each head by the rank-1-ness (σ₁/Σσ) of its
// own delta. Heads with clean rank-1 deltas (coherent k^T*v writes) get the
// full edit; heads with diffuse/noisy deltas get suppressed.
// No separate calibration pass needed — the weights come from the edit itself.
static void apply_rank1_weighted_delta(rp::StateBuf & dst,
                                       const rp::StateBuf & sa, const rp::StateBuf & sb,
                                       const std::vector<int> & layers, int n_head, int hs,
                                       double alpha) {
    for (int layer : layers) {
        for (int h = 0; h < n_head; ++h) {
            auto da = sa.s_head(layer, h);
            auto db = sb.s_head(layer, h);
            auto dd = dst.s_head(layer, h);

            // SVD of this head's delta
            std::vector<double> delta(hs * hs);
            for (int r = 0; r < hs; ++r) {
                for (int c = 0; c < hs; ++c) {
                    delta[r * hs + c] = (double)db[r * hs + c] - (double)da[r * hs + c];
                }
            }

            auto svdr = rp::numeric::svd(delta.data(), hs);

            double s1 = svdr.sigma[0];
            double ssum = 0.0;
            for (int si = 0; si < (int)svdr.sigma.size(); ++si) ssum += svdr.sigma[si];
            if (ssum < 1e-10) continue;

            double rank1ness = s1 / ssum;  // ∈ [0, 1]

            // weight = rank1ness, so clean writes get full alpha, noise gets suppressed
            double w = rank1ness * alpha;
            for (int i = 0; i < hs * hs; ++i) {
                dd[i] += (float)(w * ((double)db[i] - (double)da[i]));
            }
        }
    }
}

// Self-calibrated rank-1 projection: for each head, SVD the delta, then apply
// ONLY the rank-1 component (σ₁ · u₁ · v₁ᵀ). Strips out noise entirely,
// keeps only the coherent write signal.
static void apply_rank1_projected_delta(rp::StateBuf & dst,
                                        const rp::StateBuf & sa, const rp::StateBuf & sb,
                                        const std::vector<int> & layers, int n_head, int hs,
                                        double alpha) {
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

            if (svdr.sigma[0] < 1e-10) continue;

            // apply only σ₁ · u₁ · v₁ᵀ
            for (int r = 0; r < hs; ++r) {
                for (int c = 0; c < hs; ++c) {
                    dd[r * hs + c] += (float)(alpha * svdr.sigma[0] * svdr.U[r + 0 * hs] * svdr.Vt[0 + c * hs]);
                }
            }
        }
    }
}

// compute state norm over selected layers
static double compute_state_norm(const rp::StateBuf & s,
                                 const std::vector<int> & layers, int n_head, int hs) {
    double sum = 0.0;
    for (int layer : layers) {
        for (int h = 0; h < n_head; ++h) {
            auto hv = s.s_head(layer, h);
            for (int i = 0; i < hs * hs; ++i) sum += (double)hv[i] * (double)hv[i];
        }
    }
    return std::sqrt(sum);
}

// compute delta norm over selected layers
static double compute_delta_norm(const rp::StateBuf & sa, const rp::StateBuf & sb,
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

// Adaptive uniform: auto-scale alpha so alpha * delta_norm / state_norm = target_ratio
static void apply_adaptive_delta(rp::StateBuf & dst,
                                 const rp::StateBuf & sa, const rp::StateBuf & sb,
                                 const std::vector<int> & layers, int n_head, int hs,
                                 double target_ratio) {
    double s_norm = compute_state_norm(dst, layers, n_head, hs);
    double d_norm = compute_delta_norm(sa, sb, layers, n_head, hs);
    double alpha = (d_norm > 1e-10) ? target_ratio * s_norm / d_norm : target_ratio;

    for (int layer : layers) {
        for (int h = 0; h < n_head; ++h) {
            auto da = sa.s_head(layer, h);
            auto db = sb.s_head(layer, h);
            auto dd = dst.s_head(layer, h);
            for (int i = 0; i < hs * hs; ++i) {
                dd[i] += (float)(alpha * ((double)db[i] - (double)da[i]));
            }
        }
    }
}

// SLERP + adaptive: auto-scale, then SLERP each head to preserve norm
static void apply_slerp_adaptive_delta(rp::StateBuf & dst,
                                       const rp::StateBuf & sa, const rp::StateBuf & sb,
                                       const std::vector<int> & layers, int n_head, int hs,
                                       double target_ratio) {
    double s_norm = compute_state_norm(dst, layers, n_head, hs);
    double d_norm = compute_delta_norm(sa, sb, layers, n_head, hs);
    double alpha = (d_norm > 1e-10) ? target_ratio * s_norm / d_norm : target_ratio;

    int n = hs * hs;
    for (int layer : layers) {
        for (int h = 0; h < n_head; ++h) {
            auto da = sa.s_head(layer, h);
            auto db = sb.s_head(layer, h);
            auto dd = dst.s_head(layer, h);

            // target = dd + alpha * (db - da)
            std::vector<double> target(n);
            for (int i = 0; i < n; ++i) {
                target[i] = (double)dd[i] + alpha * ((double)db[i] - (double)da[i]);
            }

            double norm_src = 0.0, norm_tgt = 0.0, dot = 0.0;
            for (int i = 0; i < n; ++i) {
                norm_src += (double)dd[i] * (double)dd[i];
                norm_tgt += target[i] * target[i];
                dot      += (double)dd[i] * target[i];
            }
            norm_src = std::sqrt(norm_src);
            norm_tgt = std::sqrt(norm_tgt);

            if (norm_src < 1e-12 || norm_tgt < 1e-12) {
                for (int i = 0; i < n; ++i) dd[i] = (float)target[i];
                continue;
            }

            double cos_omega = dot / (norm_src * norm_tgt);
            cos_omega = std::max(-1.0, std::min(1.0, cos_omega));
            double omega = std::acos(cos_omega);

            if (std::abs(omega) < 1e-6) {
                for (int i = 0; i < n; ++i) dd[i] = (float)target[i];
                continue;
            }

            // SLERP at t=1: fully move to target direction, but preserve source norm
            double sin_omega = std::sin(omega);
            double w_tgt = std::sin(omega) / sin_omega; // = 1.0
            (void)w_tgt;

            // just take target direction, scale to source norm
            for (int i = 0; i < n; ++i) dd[i] = (float)(target[i] / norm_tgt * norm_src);
        }
    }
}

// R1-WT + SLERP + adaptive: per-head R1-WT weighting, then adaptive scaling,
// then SLERP norm preservation. Combines selectivity with magnitude scaling.
static void apply_r1wt_slerp_adaptive(rp::StateBuf & dst,
                                      const rp::StateBuf & sa, const rp::StateBuf & sb,
                                      const std::vector<int> & layers, int n_head, int hs,
                                      double target_ratio) {
    int n = hs * hs;

    // first pass: compute R1-WT weighted delta and its norm
    // (we need the weighted delta norm for adaptive scaling)
    struct HeadEdit {
        int layer, head;
        std::vector<double> delta;  // R1-WT weighted delta for this head
    };
    std::vector<HeadEdit> edits;

    for (int layer : layers) {
        for (int h = 0; h < n_head; ++h) {
            auto ha = sa.s_head(layer, h);
            auto hb = sb.s_head(layer, h);

            std::vector<double> dm(hs * hs);
            for (int r = 0; r < hs; ++r) {
                for (int c = 0; c < hs; ++c) {
                    dm[r * hs + c] = (double)hb[r * hs + c] - (double)ha[r * hs + c];
                }
            }

            auto svdr = rp::numeric::svd(dm.data(), hs);
            double s1 = svdr.sigma[0];
            double ssum = 0.0;
            for (int si = 0; si < (int)svdr.sigma.size(); ++si) ssum += svdr.sigma[si];
            if (ssum < 1e-10) continue;

            double rank1ness = s1 / ssum;
            if (rank1ness < 1e-6) continue;

            HeadEdit he;
            he.layer = layer;
            he.head = h;
            he.delta.resize(n);
            for (int i = 0; i < n; ++i) {
                he.delta[i] = rank1ness * ((double)hb[i] - (double)ha[i]);
            }
            edits.push_back(std::move(he));
        }
    }

    // compute norms for adaptive scaling
    double s_norm_sq = 0.0;
    for (int layer : layers) {
        for (int h = 0; h < n_head; ++h) {
            auto hv = dst.s_head(layer, h);
            for (int i = 0; i < n; ++i) s_norm_sq += (double)hv[i] * (double)hv[i];
        }
    }
    double s_norm = std::sqrt(s_norm_sq);

    double d_norm_sq = 0.0;
    for (const auto & he : edits) {
        for (int i = 0; i < n; ++i) d_norm_sq += he.delta[i] * he.delta[i];
    }
    double d_norm = std::sqrt(d_norm_sq);

    double alpha = (d_norm > 1e-10) ? target_ratio * s_norm / d_norm : target_ratio;

    // apply with SLERP
    for (const auto & he : edits) {
        auto dd = dst.s_head(he.layer, he.head);

        std::vector<double> target(n);
        for (int i = 0; i < n; ++i) {
            target[i] = (double)dd[i] + alpha * he.delta[i];
        }

        double norm_src = 0.0, norm_tgt = 0.0;
        for (int i = 0; i < n; ++i) {
            norm_src += (double)dd[i] * (double)dd[i];
            norm_tgt += target[i] * target[i];
        }
        norm_src = std::sqrt(norm_src);
        norm_tgt = std::sqrt(norm_tgt);

        if (norm_src < 1e-12 || norm_tgt < 1e-12) {
            for (int i = 0; i < n; ++i) dd[i] = (float)target[i];
            continue;
        }

        // SLERP t=1: take target direction, scale to source norm
        for (int i = 0; i < n; ++i) {
            dd[i] = (float)(target[i] / norm_tgt * norm_src);
        }
    }
}

int main(int argc, char ** argv) {
    auto args = rp::parse_common_args(argc, argv);
    args.defaults(/*ctx=*/2048, /*predict=*/20);
    if (args.layer_range.empty()) args.layer_range = "16-31";

    // parse experiment-specific args from extras
    std::string test_filter;
    {
        std::vector<std::string> remaining;
        for (std::size_t i = 0; i < args.extra.size(); ++i) {
            const auto & a = args.extra[i];
            if (a == "--filter" && i+1 < args.extra.size()) {
                test_filter = args.extra[++i];
            } else {
                remaining.push_back(a);
            }
        }
        args.extra = std::move(remaining);
    }

    if (args.help_requested) {
        std::printf("usage: %s -m MODEL --tests FILE [--layers 16-31] [-n 20] [--filter NAME]\n", argv[0]);
        return 0;
    }
    if (!rp::require_model(args) || !rp::require_tests(args)) return 1;
    if (!rp::reject_extra(args)) return 1;

    // ── load tests ──
    std::vector<EditTest> tests;
    {
        json arr = rp::load_tests(args);
        for (const auto & e : arr) {
            EditTest tc;
            tc.name           = e.at("name").get<std::string>();
            tc.prompt_a       = e.at("prompt_a").get<std::string>();
            tc.prompt_b       = e.at("prompt_b").get<std::string>();
            tc.query_target   = e.at("query_target").get<std::string>();
            tc.expect_a_target = e.at("expect_a_target").get<std::string>();
            tc.expect_b_target = e.at("expect_b_target").get<std::string>();
            tc.cal_prefix     = e.value("cal_prefix", "");
            tc.cal_word_a     = e.value("cal_word_a", tc.expect_a_target);
            tc.cal_word_b     = e.value("cal_word_b", tc.expect_b_target);
            tc.edit_sentence_a = e.value("edit_sentence_a", "");
            tc.edit_sentence_b = e.value("edit_sentence_b", "");

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
        const auto * vocab = env->vocab;
        int n_head = geom.n_head;
        int hs = geom.head_size;

        auto layers = rp::parse_layer_range(args.layer_range, geom.n_layer);

        std::printf("layers: %d-%d (%zu)  n_head=%d  head_size=%d\n\n",
                    layers.front(), layers.back(), layers.size(), n_head, hs);

        for (const auto & tc : tests) {
            if (!test_filter.empty() && tc.name.find(test_filter) == std::string::npos) continue;

            llama_token tok_a_target = rp::find_token(vocab, tc.expect_a_target);
            llama_token tok_b_target = rp::find_token(vocab, tc.expect_b_target);

            struct ResolvedControl {
                std::string query;
                std::string expect_str;
                llama_token tok;
            };
            std::vector<ResolvedControl> controls;
            for (const auto & c : tc.controls) {
                controls.push_back({c.query, c.expect, rp::find_token(vocab, c.expect)});
            }

            // ── capture full-prompt states ──
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

            // ── calibration: process shared prefix, probe single token ──
            std::string cal_prefix = tc.cal_prefix;
            if (cal_prefix.empty()) {
                // auto-detect: find longest common prefix of prompt_a and prompt_b
                std::size_t i = 0;
                while (i < tc.prompt_a.size() && i < tc.prompt_b.size()
                       && tc.prompt_a[i] == tc.prompt_b[i]) {
                    ++i;
                }
                // back up to last space
                while (i > 0 && tc.prompt_a[i - 1] != ' ') --i;
                cal_prefix = tc.prompt_a.substr(0, i);
            }

            llama_token cal_tok_a = rp::find_token(vocab, tc.cal_word_a);
            llama_token cal_tok_b = rp::find_token(vocab, tc.cal_word_b);

            std::printf("======================================================================\n");
            std::printf("TEST: %s\n", tc.name.c_str());
            std::printf("  A: \"%s\"\n", tc.prompt_a.c_str());
            std::printf("  B: \"%s\"\n", tc.prompt_b.c_str());
            std::printf("  cal_prefix: \"%s\"\n", cal_prefix.c_str());
            std::printf("  cal_words: \"%s\" → \"%s\"\n",
                        tc.cal_word_a.c_str(), tc.cal_word_b.c_str());

            // capture prefix state for calibration
            auto cal_toks = rp::tokenize(vocab, cal_prefix);
            ctx.clear_memory();
            ctx.decode(rp::span<const llama_token>(cal_toks.data(), cal_toks.size()));
            rp::StateBuf state_prefix(geom);
            state_prefix.load_from(ctx.raw());

            auto weights = compute_calibration(ctx, state_prefix,
                cal_tok_a, cal_tok_b, layers, n_head, hs, geom);

            // ── show calibration weights (top heads per layer) ──
            std::printf("\n  CALIBRATION WEIGHTS (top 5 heads per layer):\n");
            for (int li = 0; li < (int)layers.size(); li += 4) {
                int layer = layers[li];
                std::vector<std::pair<double, int>> sorted_heads;
                for (int h = 0; h < n_head; ++h) {
                    sorted_heads.push_back({weights[li][h], h});
                }
                std::sort(sorted_heads.rbegin(), sorted_heads.rend());
                std::printf("    L%-3d: ", layer);
                for (int i = 0; i < std::min(5, n_head); ++i) {
                    std::printf("H%d=%.3f ", sorted_heads[i].second, sorted_heads[i].first);
                }
                std::printf("\n");
            }

            // ── isolated edit pair: just the target entity's sentence ──
            // auto-detect by finding the sentence in prompt_a/b that differs
            std::string iso_a = tc.edit_sentence_a;
            std::string iso_b = tc.edit_sentence_b;
            if (iso_a.empty()) {
                // split prompt_a and prompt_b into sentences (by ". ")
                // find the first sentence pair that differs
                auto split = [](const std::string & s) {
                    std::vector<std::string> out;
                    std::size_t pos = 0;
                    while (pos < s.size()) {
                        auto dot = s.find(". ", pos);
                        if (dot == std::string::npos) {
                            out.push_back(s.substr(pos));
                            break;
                        }
                        out.push_back(s.substr(pos, dot - pos + 1));
                        pos = dot + 2;
                    }
                    return out;
                };
                auto sents_a = split(tc.prompt_a);
                auto sents_b = split(tc.prompt_b);
                for (std::size_t si = 0; si < sents_a.size() && si < sents_b.size(); ++si) {
                    if (sents_a[si] != sents_b[si]) {
                        iso_a = sents_a[si];
                        iso_b = sents_b[si];
                        break;
                    }
                }
            }

            rp::StateBuf state_iso_a(geom);
            rp::StateBuf state_iso_b(geom);
            bool have_iso = !iso_a.empty() && !iso_b.empty();
            if (have_iso) {
                std::printf("  isolated: \"%s\" → \"%s\"\n", iso_a.c_str(), iso_b.c_str());

                auto iso_toks_a = rp::tokenize(vocab, iso_a);
                ctx.clear_memory();
                ctx.decode(rp::span<const llama_token>(iso_toks_a.data(), iso_toks_a.size()));
                state_iso_a.load_from(ctx.raw());

                auto iso_toks_b = rp::tokenize(vocab, iso_b);
                ctx.clear_memory();
                ctx.decode(rp::span<const llama_token>(iso_toks_b.data(), iso_toks_b.size()));
                state_iso_b.load_from(ctx.raw());
            }

            // ── baselines ──
            auto bl_a = query_with_state(model, ctx, state_a,
                tc.prompt_a, tc.query_target,
                tok_a_target, tok_b_target, vocab, args.n_predict, args.seed);

            std::printf("\n  BASELINE A: P(exp)=%.4f  %s\n",
                        bl_a.p_target, bl_a.gen.substr(0, 50).c_str());

            // ── sweep alpha, compare uniform vs calibrated ──
            std::printf("\n  %-8s  %-6s  |  %8s", "method", "alpha", "P(tgt)");
            for (const auto & c : controls) {
                char hdr[32];
                std::snprintf(hdr, sizeof hdr, "P(%s)", c.expect_str.substr(0, 6).c_str());
                std::printf("  %8s", hdr);
            }
            std::printf("  %s\n", "gen");
            std::printf("  %s\n", std::string(10 + 8 + 3 + 10 + controls.size() * 10 + 42, '-').c_str());

            std::vector<double> alphas = {0.5, 0.75, 1.0, 1.25, 1.5};

            for (double alpha : alphas) {
                // uniform edit
                {
                    rp::StateBuf edited(state_a);
                    rp::apply_full_delta(edited, state_a, state_b, layers, alpha);

                    auto r = query_with_state(model, ctx, edited,
                        tc.prompt_a, tc.query_target,
                        tok_a_target, tok_b_target, vocab, args.n_predict, args.seed);

                    std::printf("  %-8s  %-6.2f  |  %8.4f", "UNIFORM", alpha, r.p_target);

                    for (const auto & c : controls) {
                        auto rc = query_with_state(model, ctx, edited,
                            tc.prompt_a, c.query,
                            c.tok, tok_b_target, vocab, args.n_predict, args.seed);
                        std::printf("  %8.4f", rc.p_target);
                    }

                    std::string tgen = r.gen.substr(0, 40);
                    for (auto & ch : tgen) if (ch == '\n') ch = ' ';
                    std::printf("  %s\n", tgen.c_str());
                }

                // calibrated edit (separate probe — for reference)
                {
                    rp::StateBuf edited(state_a);
                    rp::apply_weighted_delta(edited, state_a, state_b, layers,
                                         weights, alpha);

                    auto r = query_with_state(model, ctx, edited,
                        tc.prompt_a, tc.query_target,
                        tok_a_target, tok_b_target, vocab, args.n_predict, args.seed);

                    std::printf("  %-8s  %-6.2f  |  %8.4f", "CALIB", alpha, r.p_target);

                    for (const auto & c : controls) {
                        auto rc = query_with_state(model, ctx, edited,
                            tc.prompt_a, c.query,
                            c.tok, tok_b_target, vocab, args.n_predict, args.seed);
                        std::printf("  %8.4f", rc.p_target);
                    }

                    std::string tgen = r.gen.substr(0, 40);
                    for (auto & ch : tgen) if (ch == '\n') ch = ' ';
                    std::printf("  %s\n", tgen.c_str());
                }

                // self-calibrated: weight by rank-1-ness of each head's delta
                {
                    rp::StateBuf edited(state_a);
                    apply_rank1_weighted_delta(edited, state_a, state_b, layers, n_head, hs, alpha);

                    auto r = query_with_state(model, ctx, edited,
                        tc.prompt_a, tc.query_target,
                        tok_a_target, tok_b_target, vocab, args.n_predict, args.seed);

                    std::printf("  %-8s  %-6.2f  |  %8.4f", "R1-WT", alpha, r.p_target);

                    for (const auto & c : controls) {
                        auto rc = query_with_state(model, ctx, edited,
                            tc.prompt_a, c.query,
                            c.tok, tok_b_target, vocab, args.n_predict, args.seed);
                        std::printf("  %8.4f", rc.p_target);
                    }

                    std::string tgen = r.gen.substr(0, 40);
                    for (auto & ch : tgen) if (ch == '\n') ch = ' ';
                    std::printf("  %s\n", tgen.c_str());
                }

                // self-calibrated: apply only rank-1 component of each head's delta
                {
                    rp::StateBuf edited(state_a);
                    apply_rank1_projected_delta(edited, state_a, state_b, layers, n_head, hs, alpha);

                    auto r = query_with_state(model, ctx, edited,
                        tc.prompt_a, tc.query_target,
                        tok_a_target, tok_b_target, vocab, args.n_predict, args.seed);

                    std::printf("  %-8s  %-6.2f  |  %8.4f", "R1-PROJ", alpha, r.p_target);

                    for (const auto & c : controls) {
                        auto rc = query_with_state(model, ctx, edited,
                            tc.prompt_a, c.query,
                            c.tok, tok_b_target, vocab, args.n_predict, args.seed);
                        std::printf("  %8.4f", rc.p_target);
                    }

                    std::string tgen = r.gen.substr(0, 40);
                    for (auto & ch : tgen) if (ch == '\n') ch = ' ';
                    std::printf("  %s\n", tgen.c_str());
                }

                // ISOLATE: use short isolated sentence pair as delta source,
                // with R1-WT self-calibration. Target entity is first (and only)
                // entity so sigma weights are clean.
                if (have_iso) {
                    rp::StateBuf edited(state_a);
                    apply_rank1_weighted_delta(edited, state_iso_a, state_iso_b,
                                               layers, n_head, hs, alpha);

                    auto r = query_with_state(model, ctx, edited,
                        tc.prompt_a, tc.query_target,
                        tok_a_target, tok_b_target, vocab, args.n_predict, args.seed);

                    std::printf("  %-8s  %-6.2f  |  %8.4f", "ISOLATE", alpha, r.p_target);

                    for (const auto & c : controls) {
                        auto rc = query_with_state(model, ctx, edited,
                            tc.prompt_a, c.query,
                            c.tok, tok_b_target, vocab, args.n_predict, args.seed);
                        std::printf("  %8.4f", rc.p_target);
                    }

                    std::string tgen = r.gen.substr(0, 40);
                    for (auto & ch : tgen) if (ch == '\n') ch = ' ';
                    std::printf("  %s\n", tgen.c_str());
                }

                // ISO-UNI: isolated sentence pair, uniform weights (no R1-WT).
                // Tests whether the isolation alone helps, without the weighting.
                if (have_iso) {
                    rp::StateBuf edited(state_a);
                    rp::apply_full_delta(edited, state_iso_a, state_iso_b,
                                         layers, alpha);

                    auto r = query_with_state(model, ctx, edited,
                        tc.prompt_a, tc.query_target,
                        tok_a_target, tok_b_target, vocab, args.n_predict, args.seed);

                    std::printf("  %-8s  %-6.2f  |  %8.4f", "ISO-UNI", alpha, r.p_target);

                    for (const auto & c : controls) {
                        auto rc = query_with_state(model, ctx, edited,
                            tc.prompt_a, c.query,
                            c.tok, tok_b_target, vocab, args.n_predict, args.seed);
                        std::printf("  %8.4f", rc.p_target);
                    }

                    std::string tgen = r.gen.substr(0, 40);
                    for (auto & ch : tgen) if (ch == '\n') ch = ' ';
                    std::printf("  %s\n", tgen.c_str());
                }

                std::fflush(stdout);
            }

            // ── adaptive methods (ratio-based, not alpha-based) ──
            std::printf("\n  Adaptive methods (ratio = edit_norm/state_norm):\n");
            std::printf("  %-8s  %-6s  |  %8s", "method", "ratio", "P(tgt)");
            for (const auto & c : controls) {
                char hdr[32];
                std::snprintf(hdr, sizeof hdr, "P(%s)", c.expect_str.substr(0, 6).c_str());
                std::printf("  %8s", hdr);
            }
            std::printf("  %s\n", "gen");
            std::printf("  %s\n", std::string(10 + 8 + 3 + 10 + controls.size() * 10 + 42, '-').c_str());

            for (double ratio : {0.15, 0.20, 0.25, 0.30}) {
                // ADAP: adaptive uniform (full prompt delta)
                {
                    rp::StateBuf edited(state_a);
                    apply_adaptive_delta(edited, state_a, state_b, layers, n_head, hs, ratio);

                    auto r = query_with_state(model, ctx, edited,
                        tc.prompt_a, tc.query_target,
                        tok_a_target, tok_b_target, vocab, args.n_predict, args.seed);

                    std::printf("  %-8s  %-6.2f  |  %8.4f", "ADAP", ratio, r.p_target);
                    for (const auto & c : controls) {
                        auto rc = query_with_state(model, ctx, edited,
                            tc.prompt_a, c.query,
                            c.tok, tok_b_target, vocab, args.n_predict, args.seed);
                        std::printf("  %8.4f", rc.p_target);
                    }
                    std::string tgen = r.gen.substr(0, 40);
                    for (auto & ch : tgen) if (ch == '\n') ch = ' ';
                    std::printf("  %s\n", tgen.c_str());
                }

                // SL+A: SLERP + adaptive (full prompt delta)
                {
                    rp::StateBuf edited(state_a);
                    apply_slerp_adaptive_delta(edited, state_a, state_b, layers, n_head, hs, ratio);

                    auto r = query_with_state(model, ctx, edited,
                        tc.prompt_a, tc.query_target,
                        tok_a_target, tok_b_target, vocab, args.n_predict, args.seed);

                    std::printf("  %-8s  %-6.2f  |  %8.4f", "SL+A", ratio, r.p_target);
                    for (const auto & c : controls) {
                        auto rc = query_with_state(model, ctx, edited,
                            tc.prompt_a, c.query,
                            c.tok, tok_b_target, vocab, args.n_predict, args.seed);
                        std::printf("  %8.4f", rc.p_target);
                    }
                    std::string tgen = r.gen.substr(0, 40);
                    for (auto & ch : tgen) if (ch == '\n') ch = ' ';
                    std::printf("  %s\n", tgen.c_str());
                }

                // SL+A-ISO: SLERP + adaptive with isolated sentence delta
                if (have_iso) {
                    rp::StateBuf edited(state_a);
                    apply_slerp_adaptive_delta(edited, state_iso_a, state_iso_b, layers, n_head, hs, ratio);

                    auto r = query_with_state(model, ctx, edited,
                        tc.prompt_a, tc.query_target,
                        tok_a_target, tok_b_target, vocab, args.n_predict, args.seed);

                    std::printf("  %-8s  %-6.2f  |  %8.4f", "SL+A-IS", ratio, r.p_target);
                    for (const auto & c : controls) {
                        auto rc = query_with_state(model, ctx, edited,
                            tc.prompt_a, c.query,
                            c.tok, tok_b_target, vocab, args.n_predict, args.seed);
                        std::printf("  %8.4f", rc.p_target);
                    }
                    std::string tgen = r.gen.substr(0, 40);
                    for (auto & ch : tgen) if (ch == '\n') ch = ' ';
                    std::printf("  %s\n", tgen.c_str());
                }

                // R1+SL+A: R1-WT weighting + SLERP + adaptive (full prompt delta)
                {
                    rp::StateBuf edited(state_a);
                    apply_r1wt_slerp_adaptive(edited, state_a, state_b, layers, n_head, hs, ratio);

                    auto r = query_with_state(model, ctx, edited,
                        tc.prompt_a, tc.query_target,
                        tok_a_target, tok_b_target, vocab, args.n_predict, args.seed);

                    std::printf("  %-8s  %-6.2f  |  %8.4f", "R1+SL+A", ratio, r.p_target);
                    for (const auto & c : controls) {
                        auto rc = query_with_state(model, ctx, edited,
                            tc.prompt_a, c.query,
                            c.tok, tok_b_target, vocab, args.n_predict, args.seed);
                        std::printf("  %8.4f", rc.p_target);
                    }
                    std::string tgen = r.gen.substr(0, 40);
                    for (auto & ch : tgen) if (ch == '\n') ch = ' ';
                    std::printf("  %s\n", tgen.c_str());
                }

                std::fflush(stdout);
            }

            std::printf("\n");
        }

        std::fprintf(stderr, "done\n");
    });
}
