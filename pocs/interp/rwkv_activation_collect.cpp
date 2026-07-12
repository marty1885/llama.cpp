#include "interp.hpp"
#include "rwkv_activation_store.h"

#include <algorithm>
#include <chrono>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

std::string activation_name(int layer, const char * name) {
    return "rwkv.layer." + std::to_string(layer) + "." + name;
}

uint64_t splitmix64(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

llama_interp::task<> prefill_state(
        llama_interp::runtime & rt,
        const llama_interp::rwkv_state & initial,
        const std::vector<llama_token> & tokens,
        llama_interp::rwkv_state & out) {
    out = co_await rt.prefill_tokens(initial, tokens);
}

llama_interp::task<> capture_step(
        llama_interp::runtime & rt,
        const llama_interp::rwkv_state & before,
        llama_token token,
        const std::vector<std::string> * names,
        llama_interp::rwkv_state & out,
        llama_interp::activation_set & captures) {
    auto path = rt.prefill_tokens(before, { token });
    if (names) {
        for (const auto & name : *names) {
            path.capture_f16("^" + name + "$", captures);
        }
    }
    out = co_await path;
}

struct pending_capture {
    llama_interp::rwkv_state before;
    llama_token token;
    rwkv_activation_store::sample_metadata metadata;
};

void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s -m MODEL --corpus FILE --output FILE [--max-samples N] [--capture-batch N] [--seed N] [-ngl N]\n"
        "\n"
        "Streams one deterministic, context-bearing token position from each corpus line into\n"
        "a ROOT RNTuple. All 488 raw RWKV time-mix sources and the final residual are stored\n"
        "as native FP16; no corpus-sized activation buffer is retained in memory.\n",
        argv0);
}

} // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::string model_path;
    std::string corpus_path;
    std::string output_path;
    int n_gpu_layers = 0;
    uint64_t max_samples = 1500;
    uint32_t capture_batch = 4;
    uint64_t seed = 17;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (std::strcmp(argv[i], "--corpus") == 0 && i + 1 < argc) {
            corpus_path = argv[++i];
        } else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_path = argv[++i];
        } else if (std::strcmp(argv[i], "--max-samples") == 0 && i + 1 < argc) {
            max_samples = std::strtoull(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "--capture-batch") == 0 && i + 1 < argc) {
            capture_batch = std::strtoul(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = std::strtoull(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "-ngl") == 0 && i + 1 < argc) {
            n_gpu_layers = std::atoi(argv[++i]);
        } else {
            usage(argv[0]);
            return 1;
        }
    }
    if (model_path.empty() || corpus_path.empty() || output_path.empty() || max_samples == 0 || capture_batch == 0) {
        usage(argv[0]);
        return 1;
    }
    std::ifstream corpus(corpus_path);
    if (!corpus) {
        throw std::runtime_error("failed to open corpus: " + corpus_path);
    }
    const std::filesystem::path output(output_path);
    if (!output.parent_path().empty()) {
        std::filesystem::create_directories(output.parent_path());
    }

    ggml_backend_load_all();
    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        throw std::runtime_error("failed to load model");
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = 512;
    context_params.n_batch = 512;
    context_params.n_ubatch = 512;
    context_params.n_seq_max = capture_batch;
    llama_context * ctx = llama_init_from_model(model, context_params);
    if (!ctx) {
        llama_model_free(model);
        throw std::runtime_error("failed to create context");
    }

    llama_interp::runtime rt(ctx, capture_batch);
    const auto initial = rt.make_state();
    if (initial.n_embd_r / 2 != rwkv_activation_store::k_default_dimension) {
        throw std::runtime_error("collector currently requires 4096-dimensional RWKV activations");
    }
    const std::vector<const char *> time_mix = {
        "time.r", "time.w", "time.k", "time.v", "time.a", "time.g", "time.wkv", "time.rkv",
    };
    std::vector<std::string> names;
    for (int layer = 0; layer < (int) initial.n_layer; ++layer) {
        for (const char * activation : time_mix) {
            names.push_back(activation_name(layer, activation));
        }
    }
    const std::string final_residual = activation_name((int) initial.n_layer - 1, "resid.out");
    names.push_back(final_residual);

    auto writer = rwkv_activation_store::activation_dataset_writer::create(output_path, names);
    const auto started = std::chrono::steady_clock::now();
    uint64_t corpus_line = 0;
    uint64_t sample_id = 0;
    std::vector<pending_capture> pending;
    pending.reserve(capture_batch);
    std::string prompt;
    auto flush_pending = [&] {
        if (pending.empty()) {
            return;
        }
        llama_interp::activation_set captures;
        std::vector<llama_interp::rwkv_state> after(pending.size());
        std::vector<llama_interp::task<>> tasks;
        tasks.reserve(pending.size());
        for (size_t i = 0; i < pending.size(); ++i) {
            tasks.emplace_back(capture_step(rt, pending[i].before, pending[i].token, i == 0 ? &names : nullptr, after[i], captures));
        }
        rt.run();
        for (const auto & task : tasks) {
            task.rethrow_if_failed();
        }

        std::unordered_map<std::string, const llama_interp_activation *> by_name;
        by_name.reserve(captures.size());
        for (const auto & item : captures) {
            if (!item.data.empty()) {
                by_name[item.name] = &item;
            }
        }
        for (size_t sample = 0; sample < pending.size(); ++sample) {
            std::vector<std::span<const ggml_fp16_t>> activations;
            activations.reserve(names.size());
            for (const auto & name : names) {
                const auto found = by_name.find(name);
                if (found == by_name.end()) {
                    throw std::runtime_error("missing batched activation capture: " + name);
                }
                const auto & captured = *found->second;
                if (captured.data.size() != pending.size() * rwkv_activation_store::k_default_dimension) {
                    throw std::runtime_error("malformed batched activation capture: " + name +
                        " shape=" + std::to_string(captured.shape[0]) + "x" + std::to_string(captured.shape[1]) +
                        " values=" + std::to_string(captured.data.size()));
                }
                const auto & data = captured.data;
                activations.emplace_back(data.data() + sample * rwkv_activation_store::k_default_dimension,
                    rwkv_activation_store::k_default_dimension);
            }
            writer.append(pending[sample].metadata, activations);
            ++sample_id;
        }
        pending.clear();
        if (sample_id % 16 == 0 || sample_id == max_samples) {
            writer.commit();
            const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            const double rate = sample_id / std::max(elapsed, 1e-9);
            const double remaining = (max_samples - sample_id) / rate;
            std::printf("progress=%llu/%llu %.1f%% corpus_line=%llu rate=%.3f samples/s elapsed=%.0fs eta=%.0fs\n",
                (unsigned long long) sample_id,
                (unsigned long long) max_samples,
                100.0 * sample_id / max_samples,
                (unsigned long long) corpus_line,
                rate,
                elapsed,
                remaining);
            std::fflush(stdout);
        }
    };
    while (sample_id + pending.size() < max_samples && std::getline(corpus, prompt)) {
        ++corpus_line;
        if (prompt.empty()) {
            continue;
        }
        const auto tokens = common_tokenize(vocab, prompt, false, true);
        if (tokens.empty()) {
            continue;
        }
        const size_t first_context_position = std::min<size_t>(8, tokens.size() - 1);
        const size_t range = tokens.size() - first_context_position;
        const size_t position = first_context_position + splitmix64(seed ^ corpus_line) % range;

        llama_interp::rwkv_state before = initial;
        if (position > 0) {
            auto prefix = prefill_state(rt, initial, std::vector<llama_token>(tokens.begin(), tokens.begin() + position), before);
            rt.run();
            prefix.rethrow_if_failed();
        }
        pending.push_back({
            std::move(before),
            tokens[position],
            { sample_id + pending.size(), corpus_line, (int32_t) position, tokens[position], (int32_t) tokens.size() },
        });
        if (pending.size() == capture_batch) {
            flush_pending();
        }
    }
    flush_pending();
    writer.commit();
    std::printf("wrote=%s samples=%llu\n", output_path.c_str(), (unsigned long long) sample_id);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return sample_id == max_samples ? 0 : 2;
}
