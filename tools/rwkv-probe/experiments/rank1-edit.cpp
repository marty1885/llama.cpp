// experiments/rank1-edit.cpp — test rank-1 state editing.
//
// Given prompt A and B, computes the rank-1 approximation of their
// per-head state delta, then applies it as an edit to state A.
// Tests whether rank-1 editing can shift logits toward B.
//
// Three modes:
//   REWRITE:  state_A + alpha·delta_r1  -> should shift toward B
//   ERASE:    state_B - alpha·delta_r1  -> should lose the fact
//   CONTROL:  state_A + alpha·random_r1 -> should NOT shift toward B
//
// Usage:
//   llama-rwkv-rank1-edit -m model.gguf --tests causal_trace_tests.json
//       [--layer 29] [--heads 31,15,26,16,27]
//       [--alphas 0.25,0.5,0.75,1.0,1.5,2.0]

#include "rwkv_probe/experiment.h"
#include "rwkv_probe/state.h"
#include "rwkv_probe/generate.h"

#include "rwkv_probe/numeric.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace rp = rwkv_probe;
using json = nlohmann::json;

struct TestCase {
    std::string prompt_a, prompt_b, expect_a, expect_b, name;
};

// rank-1 component of a head's delta: sigma * u * v^T
struct Rank1 {
    double sigma;
    std::vector<double> u;  // left singular vector (hs)
    std::vector<double> v;  // right singular vector (hs)
    double r1_ratio;
    int layer, head;
};

static std::vector<double> parse_double_list(const std::string & s) {
    std::vector<double> out;
    std::istringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        out.push_back(std::atof(tok.c_str()));
    }
    return out;
}

// compute rank-1 decomposition of delta between two head states
static Rank1 compute_rank1(const rp::StateBuf & sa, const rp::StateBuf & sb,
                           int layer, int head, int hs) {
    auto a = sa.s_head(layer, head);
    auto b = sb.s_head(layer, head);

    std::vector<double> delta(hs * hs);
    double l2_sq = 0.0;
    for (int r = 0; r < hs; ++r) {
        for (int c = 0; c < hs; ++c) {
            double d = (double)b[r * hs + c] - (double)a[r * hs + c];
            delta[r * hs + c] = d;
            l2_sq += d * d;
        }
    }

    Rank1 result;
    result.layer = layer;
    result.head = head;
    result.sigma = 0;
    result.r1_ratio = 0;
    result.u.resize(hs, 0.0);
    result.v.resize(hs, 0.0);

    if (l2_sq < 1e-20) return result;

    auto svdr = rp::numeric::svd(delta.data(), hs);

    result.sigma = svdr.sigma[0];

    for (int i = 0; i < hs; ++i) {
        result.u[i] = svdr.U[i + 0 * hs];      // U column 0
        result.v[i] = svdr.Vt[0 + i * hs];      // Vt row 0, col i
    }

    double sum_sq = 0.0;
    for (int i = 0; i < (int)svdr.sigma.size(); ++i) sum_sq += svdr.sigma[i] * svdr.sigma[i];
    result.r1_ratio = (sum_sq > 0) ? (svdr.sigma[0] * svdr.sigma[0] / sum_sq) : 0.0;

    return result;
}

// apply rank-1 edit: state[head] += alpha * sigma * u * v^T
static void apply_rank1(rp::StateBuf & state, const Rank1 & r1, double alpha, int hs) {
    auto s = state.s_head(r1.layer, r1.head);
    for (int r = 0; r < hs; ++r) {
        for (int c = 0; c < hs; ++c) {
            s[r * hs + c] += (float)(alpha * r1.sigma * r1.u[r] * r1.v[c]);
        }
    }
}

int main(int argc, char ** argv) {
    auto args = rp::parse_common_args(argc, argv);
    args.defaults(2048, 20);

    int target_layer = 29;
    std::string heads_str = "31,15,26,16,27";
    std::string alphas_str = "0.25,0.5,0.75,1.0,1.5,2.0";
    bool all_layers = false;

    for (std::size_t i = 0; i < args.extra.size(); ++i) {
        const auto & a = args.extra[i];
        if      (a == "--layer"      && i+1 < args.extra.size()) target_layer = std::atoi(args.extra[++i].c_str());
        else if (a == "--heads"      && i+1 < args.extra.size()) heads_str    = args.extra[++i];
        else if (a == "--alphas"     && i+1 < args.extra.size()) alphas_str   = args.extra[++i];
        else if (a == "--all-layers") all_layers = true;
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 1; }
    }

    if (args.help_requested) {
        std::printf("usage: %s -m MODEL --tests FILE [--layer L] [--heads 31,15,27]\n"
                    "       [--alphas 0.5,1.0,2.0] [--all-layers] [-n N]\n", argv[0]);
        return 0;
    }

    if (!rp::require_model(args)) return 1;
    if (!rp::require_tests(args)) return 1;

    auto heads = rp::parse_int_list(heads_str);
    auto alphas = parse_double_list(alphas_str);

    std::vector<TestCase> tests;
    {
        json arr = rp::load_tests(args);
        for (const auto & e : arr) {
            tests.push_back({
                e.at("prompt_a").get<std::string>(),
                e.at("prompt_b").get<std::string>(),
                e.at("expect_a").get<std::string>(),
                e.at("expect_b").get<std::string>(),
                e.value("name", ""),
            });
        }
    }

    return rp::run_experiment([&] {
        auto env = rp::make_env(args);
        auto & model = env->model;
        auto & ctx   = env->ctx;
        const auto & geom  = env->geom;
        const auto * vocab  = env->vocab;
        int hs = geom.head_size;

        // determine which layers to operate on
        std::vector<int> layers;
        if (all_layers) {
            for (int l = 16; l < geom.n_layer; ++l) layers.push_back(l);
        } else {
            layers.push_back(target_layer);
        }

        std::fprintf(stderr, "heads: %s  layers: %d-%d  alphas: %s\n",
                     heads_str.c_str(), layers.front(), layers.back(), alphas_str.c_str());

        for (std::size_t ti = 0; ti < tests.size(); ++ti) {
            const auto & tc = tests[ti];
            std::string label = tc.name.empty() ? ("test_" + std::to_string(ti)) : tc.name;

            llama_token tok_a = rp::find_token(vocab, tc.expect_a);
            llama_token tok_b = rp::find_token(vocab, tc.expect_b);

            auto toks_a = rp::tokenize(vocab, tc.prompt_a);
            auto toks_b = rp::tokenize(vocab, tc.prompt_b);

            if (toks_a.size() < 2 || toks_b.size() < 2) continue;

            // ── capture states at prefix ────────────────────────────
            ctx.clear_memory();
            ctx.decode(rp::span<const llama_token>(toks_a.data(), toks_a.size() - 1));
            rp::StateBuf state_a(geom);
            state_a.load_from(ctx.raw());

            ctx.clear_memory();
            ctx.decode(rp::span<const llama_token>(toks_b.data(), toks_b.size() - 1));
            rp::StateBuf state_b(geom);
            state_b.load_from(ctx.raw());

            // ── baselines ───────────────────────────────────────────
            ctx.clear_memory();
            ctx.decode(rp::span<const llama_token>(toks_a.data(), toks_a.size()));
            const float * logits = llama_get_logits(ctx.raw());
            float bl_a_la = logits[tok_a], bl_a_lb = logits[tok_b];

            ctx.clear_memory();
            ctx.decode(rp::span<const llama_token>(toks_b.data(), toks_b.size()));
            logits = llama_get_logits(ctx.raw());
            float bl_b_la = logits[tok_a], bl_b_lb = logits[tok_b];

            std::printf("\n======================================================================\n");
            std::printf("TEST: %s\n", label.c_str());
            std::printf("  A: %.80s\n", tc.prompt_a.c_str());
            std::printf("  B: %.80s\n", tc.prompt_b.c_str());
            std::printf("  BASELINE A: P(%s)=%.4f  BASELINE B: P(%s)=%.4f\n",
                        tc.expect_a.c_str(), rp::prob_of(bl_a_la, bl_a_lb),
                        tc.expect_b.c_str(), rp::prob_of(bl_b_lb, bl_b_la));

            // ── compute rank-1 decompositions ───────────────────────
            std::vector<Rank1> rank1s;
            for (int layer : layers) {
                for (int h : heads) {
                    if (h >= geom.n_head) continue;
                    Rank1 r1 = compute_rank1(state_a, state_b, layer, h, hs);
                    rank1s.push_back(r1);
                }
            }

            // print rank-1 info
            std::printf("\n  RANK-1 DECOMPOSITIONS:\n");
            std::printf("  %4s %4s  %10s  %8s\n", "L", "H", "sigma_1", "r1_ratio");
            for (const auto & r1 : rank1s) {
                std::printf("  L%-3d H%-3d  %10.4f  %8.4f\n",
                            r1.layer, r1.head, r1.sigma, r1.r1_ratio);
            }

            // ── REWRITE test: state_A + alpha·delta_r1 -> shift toward B? ──
            std::printf("\n  REWRITE: state_A + alpha·sum(rank1_deltas):\n");
            std::printf("  %6s  %8s  %8s  %7s  %s\n",
                        "alpha", "logit_a", "logit_b", "P(a)", "gen");
            std::printf("  %s\n", std::string(75, '-').c_str());

            for (double alpha : alphas) {
                rp::StateBuf edited(state_a);
                for (const auto & r1 : rank1s) {
                    if (r1.sigma < 1e-10) continue;
                    apply_rank1(edited, r1, alpha, hs);
                }

                ctx.clear_memory();
                ctx.decode(rp::span<const llama_token>(toks_a.data(), toks_a.size() - 1));
                edited.store_to(ctx.raw());
                ctx.decode_one(toks_a.back());

                logits = llama_get_logits(ctx.raw());
                float la = logits[tok_a], lb = logits[tok_b];
                float pa = rp::prob_of(la, lb);

                std::string gen_text;
                {
                    common_params_sampling sp; sp.seed = args.seed;
                    rp::Generator gen(model, ctx, sp);
                    llama_token arr[] = {toks_a.back()};
                    gen.accept_prompt(rp::span<const llama_token>(arr, 1));
                    gen_text = gen.run(args.n_predict).text;
                }
                for (auto & c : gen_text) if (c == '\n') c = ' ';

                std::printf("  %6.2f  %8.2f  %8.2f  %6.3f  %.55s\n",
                            alpha, la, lb, pa, gen_text.c_str());
                std::fflush(stdout);
            }

            // ── ERASE test: state_B - alpha·delta_r1 -> lose the fact? ──────
            std::printf("\n  ERASE: state_B - alpha·sum(rank1_deltas):\n");
            std::printf("  %6s  %8s  %8s  %7s  %s\n",
                        "alpha", "logit_a", "logit_b", "P(a)", "gen");
            std::printf("  %s\n", std::string(75, '-').c_str());

            for (double alpha : alphas) {
                rp::StateBuf edited(state_b);
                for (const auto & r1 : rank1s) {
                    if (r1.sigma < 1e-10) continue;
                    apply_rank1(edited, r1, -alpha, hs);  // subtract
                }

                ctx.clear_memory();
                ctx.decode(rp::span<const llama_token>(toks_b.data(), toks_b.size() - 1));
                edited.store_to(ctx.raw());
                ctx.decode_one(toks_b.back());

                logits = llama_get_logits(ctx.raw());
                float la = logits[tok_a], lb = logits[tok_b];
                float pa = rp::prob_of(la, lb);

                std::string gen_text;
                {
                    common_params_sampling sp; sp.seed = args.seed;
                    rp::Generator gen(model, ctx, sp);
                    llama_token arr[] = {toks_b.back()};
                    gen.accept_prompt(rp::span<const llama_token>(arr, 1));
                    gen_text = gen.run(args.n_predict).text;
                }
                for (auto & c : gen_text) if (c == '\n') c = ' ';

                std::printf("  %6.2f  %8.2f  %8.2f  %6.3f  %.55s\n",
                            alpha, la, lb, pa, gen_text.c_str());
                std::fflush(stdout);
            }

            // ── CONTROL: random rank-1 perturbation, same magnitude ────
            std::printf("\n  CONTROL: state_A + alpha·sum(random_r1, same sigma):\n");
            std::printf("  %6s  %8s  %8s  %7s  %s\n",
                        "alpha", "logit_a", "logit_b", "P(a)", "gen");
            std::printf("  %s\n", std::string(75, '-').c_str());

            // generate random rank-1 perturbations with same sigma
            std::mt19937 rng(args.seed + 12345);
            std::normal_distribution<double> normal(0.0, 1.0);
            std::vector<Rank1> random_r1s;
            for (const auto & r1 : rank1s) {
                Rank1 rr;
                rr.layer = r1.layer;
                rr.head = r1.head;
                rr.sigma = r1.sigma;
                rr.r1_ratio = 0;
                rr.u.resize(hs);
                rr.v.resize(hs);
                // random unit vectors
                double norm_u = 0, norm_v = 0;
                for (int i = 0; i < hs; ++i) {
                    rr.u[i] = normal(rng);
                    rr.v[i] = normal(rng);
                    norm_u += rr.u[i] * rr.u[i];
                    norm_v += rr.v[i] * rr.v[i];
                }
                norm_u = std::sqrt(norm_u);
                norm_v = std::sqrt(norm_v);
                for (int i = 0; i < hs; ++i) {
                    rr.u[i] /= norm_u;
                    rr.v[i] /= norm_v;
                }
                random_r1s.push_back(rr);
            }

            for (double alpha : alphas) {
                rp::StateBuf edited(state_a);
                for (const auto & rr : random_r1s) {
                    if (rr.sigma < 1e-10) continue;
                    apply_rank1(edited, rr, alpha, hs);
                }

                ctx.clear_memory();
                ctx.decode(rp::span<const llama_token>(toks_a.data(), toks_a.size() - 1));
                edited.store_to(ctx.raw());
                ctx.decode_one(toks_a.back());

                logits = llama_get_logits(ctx.raw());
                float la = logits[tok_a], lb = logits[tok_b];
                float pa = rp::prob_of(la, lb);

                std::string gen_text;
                {
                    common_params_sampling sp; sp.seed = args.seed;
                    rp::Generator gen(model, ctx, sp);
                    llama_token arr[] = {toks_a.back()};
                    gen.accept_prompt(rp::span<const llama_token>(arr, 1));
                    gen_text = gen.run(args.n_predict).text;
                }
                for (auto & c : gen_text) if (c == '\n') c = ' ';

                std::printf("  %6.2f  %8.2f  %8.2f  %6.3f  %.55s\n",
                            alpha, la, lb, pa, gen_text.c_str());
                std::fflush(stdout);
            }

            // ── Per-head REWRITE at alpha=1.0 to see individual contributions ──
            std::printf("\n  PER-HEAD REWRITE (alpha=1.0, single head at a time):\n");
            std::printf("  %4s %4s  %8s  %8s  %7s  %8s  %s\n",
                        "L", "H", "logit_a", "logit_b", "P(a)", "r1_ratio", "gen");
            std::printf("  %s\n", std::string(80, '-').c_str());

            for (const auto & r1 : rank1s) {
                if (r1.sigma < 1e-10) continue;

                rp::StateBuf edited(state_a);
                apply_rank1(edited, r1, 1.0, hs);

                ctx.clear_memory();
                ctx.decode(rp::span<const llama_token>(toks_a.data(), toks_a.size() - 1));
                edited.store_to(ctx.raw());
                ctx.decode_one(toks_a.back());

                logits = llama_get_logits(ctx.raw());
                float la = logits[tok_a], lb = logits[tok_b];
                float pa = rp::prob_of(la, lb);

                std::string gen_text;
                {
                    common_params_sampling sp; sp.seed = args.seed;
                    rp::Generator gen(model, ctx, sp);
                    llama_token arr[] = {toks_a.back()};
                    gen.accept_prompt(rp::span<const llama_token>(arr, 1));
                    gen_text = gen.run(args.n_predict).text;
                }
                for (auto & c : gen_text) if (c == '\n') c = ' ';

                std::printf("  L%-3d H%-3d  %8.2f  %8.2f  %6.3f  %8.4f  %.50s\n",
                            r1.layer, r1.head, la, lb, pa, r1.r1_ratio, gen_text.c_str());
                std::fflush(stdout);
            }

            // ── Cumulative head REWRITE at alpha=1.0 ───────────────────────
            // Add heads one at a time, sorted by r1_ratio (highest first)
            std::vector<std::size_t> order(rank1s.size());
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(),
                      [&](std::size_t a, std::size_t b) {
                          return rank1s[a].r1_ratio > rank1s[b].r1_ratio;
                      });

            std::printf("\n  CUMULATIVE REWRITE (alpha=1.0, heads added by r1_ratio):\n");
            std::printf("  %3s  %4s %4s  %8s  %8s  %7s  %8s\n",
                        "n", "L", "H", "logit_a", "logit_b", "P(a)", "r1_ratio");
            std::printf("  %s\n", std::string(65, '-').c_str());

            std::vector<std::size_t> added;
            for (std::size_t oi = 0; oi < order.size(); ++oi) {
                std::size_t idx = order[oi];
                if (rank1s[idx].sigma < 1e-10) continue;
                added.push_back(idx);

                rp::StateBuf edited(state_a);
                for (std::size_t ai : added) {
                    apply_rank1(edited, rank1s[ai], 1.0, hs);
                }

                ctx.clear_memory();
                ctx.decode(rp::span<const llama_token>(toks_a.data(), toks_a.size() - 1));
                edited.store_to(ctx.raw());
                ctx.decode_one(toks_a.back());

                logits = llama_get_logits(ctx.raw());
                float la = logits[tok_a], lb = logits[tok_b];
                float pa = rp::prob_of(la, lb);

                std::printf("  %3zu  L%-3d H%-3d  %8.2f  %8.2f  %6.3f  %8.4f\n",
                            added.size(), rank1s[idx].layer, rank1s[idx].head,
                            la, lb, pa, rank1s[idx].r1_ratio);
                std::fflush(stdout);
            }
        }

        std::fprintf(stderr, "\ndone\n");
    });
}
