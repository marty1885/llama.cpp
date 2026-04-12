// experiments/decompose-epv.cpp — entity x property x value decomposition.
//
// Captures state from many single-fact prompts that vary one axis at a time,
// computes pairwise deltas, SVDs them, and checks whether the u1 (value)
// and v1 (key/address) directions are consistent.
//
// If varying value keeps v1 stable -> entity x property address is separable.
// If varying entity keeps u1 stable -> value encoding is separable.
// Together -> surgical editing without a donor prompt.
//
// Usage:
//   llama-rwkv-decompose-epv -m model.gguf --tests decompose_tests.json
//       [--layers 29] [--heads 31,27,15,26,16]

#include "rwkv_probe/state.h"
#include "rwkv_probe/experiment.h"
#include "rwkv_probe/head_decomp.h"
#include "rwkv_probe/generate.h"

#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

namespace rp = rwkv_probe;
using json = nlohmann::json;

struct FactPrompt {
    std::string name;
    std::string prompt;
    std::string entity;
    std::string property;
    std::string value;
    std::string expect;
};

static double cosine_sim(const std::vector<double> & a, const std::vector<double> & b) {
    double dot = 0, na = 0, nb = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if (na < 1e-30 || nb < 1e-30) return 0.0;
    return dot / (std::sqrt(na) * std::sqrt(nb));
}


int main(int argc, char ** argv) {
    auto args = rp::parse_common_args(argc, argv);
    if (args.help_requested) {
        std::printf("usage: %s -m MODEL --tests FILE [--layers 29] [--heads 31,27]\n", argv[0]);
        return 0;
    }
    args.defaults(2048, 0);
    if (args.layer_range.empty()) args.layer_range = "29";
    if (!rp::require_model(args)) return 1;
    if (!rp::require_tests(args)) return 1;

    // experiment-specific arg: --heads
    std::string heads_str = "31,27,15,26,16";
    for (std::size_t i = 0; i < args.extra.size(); ++i) {
        const auto & a = args.extra[i];
        if (a == "--heads" && i+1 < args.extra.size()) heads_str = args.extra[++i];
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 1; }
    }

    auto heads = rp::parse_int_list(heads_str);

    // load test prompts
    json arr = rp::load_tests(args);
    std::vector<FactPrompt> prompts;
    for (const auto & e : arr) {
        prompts.push_back({
            e.at("name").get<std::string>(),
            e.at("prompt").get<std::string>(),
            e.at("entity").get<std::string>(),
            e.at("property").get<std::string>(),
            e.at("value").get<std::string>(),
            e.at("expect").get<std::string>(),
        });
    }

    return rp::run_experiment([&] {
        auto env = rp::make_env(args);
        auto & ctx   = env->ctx;
        const auto & geom  = env->geom;
        const auto * vocab  = env->vocab;

        auto layers = rp::parse_layer_range(args.layer_range, geom.n_layer);

        std::fprintf(stderr, "prompts: %zu  layers: %zu  heads: %zu\n",
                     prompts.size(), layers.size(), heads.size());

        // -- capture all states ----------------------------------------------
        std::vector<rp::StateBuf> states;
        states.reserve(prompts.size());

        for (std::size_t i = 0; i < prompts.size(); ++i) {
            const auto & fp = prompts[i];
            auto toks = rp::tokenize(vocab, fp.prompt);

            ctx.clear_memory();
            ctx.decode(rp::span<const llama_token>(toks.data(), toks.size() - 1));

            states.emplace_back(geom);
            states.back().load_from(ctx.raw());

            // verify the model gets the right answer
            ctx.decode_one(toks.back());
            const float * logits = llama_get_logits(ctx.raw());
            llama_token tok_exp = rp::find_token(vocab, fp.expect);
            if (tok_exp >= 0) {
                // find rank of expected token
                float exp_logit = logits[tok_exp];
                int rank = 0;
                for (int v = 0; v < geom.n_vocab; ++v) {
                    if (logits[v] > exp_logit) rank++;
                }
                std::fprintf(stderr, "  %s: expect='%s' rank=%d logit=%.2f %s\n",
                             fp.name.c_str(), fp.expect.c_str(), rank, exp_logit,
                             rank == 0 ? "OK" : "WARN");
            }
        }

        // -- pairwise SVD analysis -------------------------------------------
        // For each head, compute SVD of delta between all pairs,
        // then check cosine similarity of u1 and v1 directions.

        for (int layer : layers) {
            for (int h : heads) {
                std::printf("\n======================================================================\n");
                std::printf("LAYER %d  HEAD %d\n", layer, h);
                std::printf("======================================================================\n");

                // compute all pairwise SVDs
                struct PairSVD {
                    std::size_t i, j;
                    rp::HeadDecomp svd;
                };
                std::vector<PairSVD> pairs;

                for (std::size_t i = 0; i < prompts.size(); ++i) {
                    for (std::size_t j = i + 1; j < prompts.size(); ++j) {
                        rp::HeadDecomp s = rp::decompose_head(states[i], states[j], layer, h);
                        pairs.push_back({i, j, s});
                    }
                }

                // -- collect pair groups ------------------------------------
                struct LabeledPair {
                    std::size_t i, j;
                    rp::HeadDecomp svd;
                    std::string label;
                };
                std::vector<LabeledPair> vv_pairs;  // vary value
                std::vector<LabeledPair> ve_pairs;  // vary entity

                for (const auto & p : pairs) {
                    const auto & a = prompts[p.i];
                    const auto & b = prompts[p.j];
                    std::string label = a.name + " vs " + b.name;
                    if (a.entity == b.entity && a.property == b.property && a.value != b.value) {
                        vv_pairs.push_back({p.i, p.j, p.svd, label});
                    }
                    if (a.entity != b.entity && a.property == b.property && a.value == b.value) {
                        ve_pairs.push_back({p.i, p.j, p.svd, label});
                    }
                }

                // helper: print a list of pairs with sigma1 and r1
                auto print_pair_list = [](const char * title,
                                          const std::vector<LabeledPair> & plist) {
                    std::printf("\n  %s:\n", title);
                    std::printf("  %-30s  %8s  %8s\n", "pair", "sigma_1", "r1");
                    std::printf("  %s\n", std::string(50, '-').c_str());
                    for (const auto & p : plist) {
                        std::printf("  %-30s  %8.4f  %8.4f\n",
                                    p.label.c_str(), p.svd.sigma, p.svd.r1_ratio);
                    }
                };

                // helper: print cosine matrix (rows x cols)
                auto print_cosine_matrix = [](const char * title,
                                              const char * what,  // "u1" or "v1"
                                              const std::vector<LabeledPair> & rows,
                                              const std::vector<LabeledPair> & cols,
                                              bool use_u) {
                    std::printf("\n  %s cos(%s) matrix:\n", title, what);
                    // legend
                    for (std::size_t k = 0; k < cols.size(); ++k) {
                        std::printf("    [%zu] %s\n", k, cols[k].label.c_str());
                    }
                    // column headers
                    std::printf("  %-30s", "");
                    for (std::size_t k = 0; k < cols.size(); ++k) {
                        std::printf("  %6zu", k);
                    }
                    std::printf("\n");
                    std::printf("  %s\n",
                        std::string(30 + 2 + cols.size() * 8, '-').c_str());
                    for (const auto & r : rows) {
                        std::printf("  %-30s", r.label.c_str());
                        for (const auto & c : cols) {
                            double cs = use_u ? cosine_sim(r.svd.u, c.svd.u)
                                              : cosine_sim(r.svd.v, c.svd.v);
                            std::printf("  %6.3f", cs);
                        }
                        std::printf("\n");
                    }
                };

                // -- VARY VALUE ---------------------------------------------
                print_pair_list("VARY VALUE (same entity+property, different value)", vv_pairs);
                print_cosine_matrix("VARY VALUE", "v1 address", vv_pairs, vv_pairs, false);
                print_cosine_matrix("VARY VALUE", "u1 value",   vv_pairs, vv_pairs, true);

                // -- VARY ENTITY --------------------------------------------
                print_pair_list("VARY ENTITY (same value+property, different entity)", ve_pairs);
                print_cosine_matrix("VARY ENTITY", "v1", ve_pairs, ve_pairs, false);

                // -- CROSS-AXIS ---------------------------------------------
                if (!vv_pairs.empty() && !ve_pairs.empty()) {
                    print_cosine_matrix("CROSS-AXIS (value vs entity)", "u1",
                                        vv_pairs, ve_pairs, true);
                }
            }
        }

        std::fprintf(stderr, "\ndone\n");
    });
}
