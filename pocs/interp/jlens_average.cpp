#include "interp.hpp"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

struct pilot_result {
    std::vector<std::vector<ggml_fp16_t>> directions;
    std::vector<std::vector<float>> responses;
};

static void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s -m MODEL --output PREFIX [-ngl N] [--layer N] [--epsilon E] [--rank N]\n"
        "\n"
        "Builds a compact rank-K average current-token Jacobian estimate over four fixed prompts.\n",
        argv0);
}

static std::string residual_name(int layer) {
    return "rwkv.layer." + std::to_string(layer) + ".resid.out";
}

static const llama_interp_activation & require_capture(
        const llama_interp::activation_set & caps,
        const std::string & name) {
    const llama_interp_activation * found = nullptr;
    for (const auto & cap : caps) {
        if (cap.name == name) {
            if (found) {
                throw std::runtime_error("duplicate activation capture: " + name);
            }
            found = &cap;
        }
    }
    if (!found) {
        throw std::runtime_error("missing activation capture: " + name);
    }
    return *found;
}

static uint64_t splitmix64(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

static std::vector<ggml_fp16_t> rademacher_perturbation(size_t n, float epsilon, uint64_t seed) {
    std::vector<ggml_fp16_t> out(n);
    const float scale = epsilon / std::sqrt((float) n);
    for (size_t i = 0; i < n; ++i) {
        out[i] = ggml_fp32_to_fp16((splitmix64(seed + i) & 1) ? scale : -scale);
    }
    return out;
}

static std::vector<ggml_fp16_t> negate(const std::vector<ggml_fp16_t> & values) {
    std::vector<ggml_fp16_t> out;
    out.reserve(values.size());
    for (ggml_fp16_t value : values) {
        out.push_back(ggml_fp32_to_fp16(-ggml_fp16_to_fp32(value)));
    }
    return out;
}

static double l2_norm(const std::vector<ggml_fp16_t> & values) {
    double sum = 0.0;
    for (ggml_fp16_t value : values) {
        const double x = ggml_fp16_to_fp32(value);
        sum += x*x;
    }
    return std::sqrt(sum);
}

static double l2_norm(const std::vector<float> & values) {
    double sum = 0.0;
    for (float value : values) {
        sum += (double) value * value;
    }
    return std::sqrt(sum);
}

static llama_interp::task<> run_pilot(
        llama_interp::runtime & rt,
        const llama_interp::rwkv_state & initial_state,
        const std::vector<std::string> & prompts,
        int source_layer,
        float epsilon,
        int rank,
        pilot_result & out) {
    using namespace llama_interp;

    const std::string source = residual_name(source_layer);
    const std::string final = residual_name((int) initial_state.n_layer - 1);

    for (size_t iprompt = 0; iprompt < prompts.size(); ++iprompt) {
        const rwkv_state prompt_state = co_await rt.prefill(initial_state, prompts[iprompt]);

        if (out.directions.empty()) {
            activation_set source_caps;
            auto clean = rt.decode(prompt_state, 1);
            clean.capture_f16("^" + source + "\\.pre$", source_caps);
            co_await clean;

            const size_t n_embd = require_capture(source_caps, source + ".pre").data.size();
            out.directions.reserve(rank);
            out.responses.assign(rank, std::vector<float>(n_embd, 0.0f));
            for (int idir = 0; idir < rank; ++idir) {
                out.directions.push_back(rademacher_perturbation(n_embd, epsilon, (uint64_t) idir));
            }
        }

        for (int idir = 0; idir < rank; ++idir) {
            activation_set plus_caps;
            auto plus = rt.decode(prompt_state, 1);
            plus.perturb("^" + source + "$", runtime::add_head(-1, out.directions[idir]));
            plus.capture_f16("^" + final + "$", plus_caps);
            co_await plus;

            activation_set minus_caps;
            auto minus = rt.decode(prompt_state, 1);
            minus.perturb("^" + source + "$", runtime::add_head(-1, negate(out.directions[idir])));
            minus.capture_f16("^" + final + "$", minus_caps);
            co_await minus;

            const auto & plus_data = require_capture(plus_caps, final).data;
            const auto & minus_data = require_capture(minus_caps, final).data;
            if (plus_data.size() != out.responses[idir].size() || minus_data.size() != out.responses[idir].size()) {
                throw std::runtime_error("inconsistent target activation dimensions");
            }

            const double actual_epsilon = l2_norm(out.directions[idir]);
            for (size_t i = 0; i < plus_data.size(); ++i) {
                out.responses[idir][i] += (float) ((ggml_fp16_to_fp32(plus_data[i]) - ggml_fp16_to_fp32(minus_data[i])) /
                    (2.0 * actual_epsilon));
            }
        }

        std::printf("averaged prompt %zu/%zu\n", iprompt + 1, prompts.size());
    }

    for (auto & response : out.responses) {
        for (float & value : response) {
            value /= (float) prompts.size();
        }
    }
}

static void write_f16_rows(const std::string & path, const std::vector<std::vector<ggml_fp16_t>> & rows) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open output: " + path);
    }
    for (const auto & row : rows) {
        out.write((const char *) row.data(), (std::streamsize) (row.size() * sizeof(ggml_fp16_t)));
    }
    if (!out) {
        throw std::runtime_error("failed to write output: " + path);
    }
}

static std::vector<std::vector<ggml_fp16_t>> unit_directions(
        const std::vector<std::vector<ggml_fp16_t>> & perturbations) {
    std::vector<std::vector<ggml_fp16_t>> out;
    out.reserve(perturbations.size());
    for (const auto & perturbation : perturbations) {
        const double norm = l2_norm(perturbation);
        std::vector<ggml_fp16_t> direction;
        direction.reserve(perturbation.size());
        for (ggml_fp16_t value : perturbation) {
            direction.push_back(ggml_fp32_to_fp16((float) (ggml_fp16_to_fp32(value) / norm)));
        }
        out.push_back(std::move(direction));
    }
    return out;
}

static void write_f16_rows(const std::string & path, const std::vector<std::vector<float>> & rows) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open output: " + path);
    }
    std::vector<ggml_fp16_t> row;
    for (const auto & values : rows) {
        row.resize(values.size());
        for (size_t i = 0; i < values.size(); ++i) {
            row[i] = ggml_fp32_to_fp16(values[i]);
        }
        out.write((const char *) row.data(), (std::streamsize) (row.size() * sizeof(ggml_fp16_t)));
    }
    if (!out) {
        throw std::runtime_error("failed to write output: " + path);
    }
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::string model_path;
    std::string output_prefix;
    int n_gpu_layers = 0;
    int source_layer = -1;
    float epsilon = 0.2f;
    int rank = 4;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_prefix = argv[++i];
        } else if (std::strcmp(argv[i], "-ngl") == 0 && i + 1 < argc) {
            n_gpu_layers = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--layer") == 0 && i + 1 < argc) {
            source_layer = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--epsilon") == 0 && i + 1 < argc) {
            epsilon = std::strtof(argv[++i], nullptr);
        } else if (std::strcmp(argv[i], "--rank") == 0 && i + 1 < argc) {
            rank = std::atoi(argv[++i]);
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    if (model_path.empty() || output_prefix.empty() || epsilon <= 0.0f || rank <= 0) {
        usage(argv[0]);
        return 1;
    }

    const std::vector<std::string> prompts = {
        "Paris is the capital of",
        "The Eiffel Tower is located in",
        "The largest planet is",
        "A triangle has",
    };

    ggml_backend_load_all();
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        std::fprintf(stderr, "failed to load model: %s\n", model_path.c_str());
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    int n_prompt = 0;
    for (const auto & prompt : prompts) {
        n_prompt = std::max(n_prompt, -llama_tokenize(vocab, prompt.c_str(), (int) prompt.size(), nullptr, 0, false, true));
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = std::max(128, n_prompt + 8);
    cparams.n_batch = n_prompt;
    cparams.n_ubatch = n_prompt;
    cparams.n_seq_max = 1;
    cparams.no_perf = false;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::fprintf(stderr, "failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    llama_interp::runtime rt(ctx, 1);
    const llama_interp::rwkv_state initial_state = rt.make_state();
    if (source_layer < 0) {
        source_layer = (int) initial_state.n_layer / 2;
    }
    if (source_layer < 0 || source_layer >= (int) initial_state.n_layer) {
        std::fprintf(stderr, "invalid source layer %d for %u-layer model\n", source_layer, initial_state.n_layer);
        return 1;
    }

    pilot_result results;
    auto task = run_pilot(rt, initial_state, prompts, source_layer, epsilon, rank, results);
    rt.run();
    task.rethrow_if_failed();

    const std::string directions_path = output_prefix + ".directions.f16";
    const std::string responses_path = output_prefix + ".responses.f16";
    const std::string manifest_path = output_prefix + ".txt";
    const auto directions = unit_directions(results.directions);
    write_f16_rows(directions_path, directions);
    write_f16_rows(responses_path, results.responses);

    std::ofstream manifest(manifest_path);
    if (!manifest) {
        throw std::runtime_error("failed to open manifest: " + manifest_path);
    }
    manifest << "source_layer=" << source_layer << '\n';
    manifest << "target_layer=" << initial_state.n_layer - 1 << '\n';
    manifest << "dimension=" << results.directions.front().size() << '\n';
    manifest << "rank=" << rank << '\n';
    manifest << "n_prompts=" << prompts.size() << '\n';
    manifest << "epsilon_requested=" << epsilon << '\n';
    manifest << "directions_f16=" << directions_path << '\n';
    manifest << "responses_f16=" << responses_path << '\n';
    manifest << "directions=unit_rademacher_rows\n";
    manifest << "estimator=Jhat=(dimension/rank)*sum_k(response[k] outer direction[k])\n";

    std::printf("source=%s target=%s rank=%d prompts=%zu\n",
        residual_name(source_layer).c_str(), residual_name((int) initial_state.n_layer - 1).c_str(), rank, prompts.size());
    for (size_t idir = 0; idir < results.directions.size(); ++idir) {
        std::printf("direction %zu input_l2=%g mean_response_l2=%g\n",
            idir, l2_norm(results.directions[idir]), l2_norm(results.responses[idir]));
    }
    std::printf("wrote %s\n", manifest_path.c_str());

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
