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
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

void quiet_llama_logs(ggml_log_level, const char *, void *) {}

uint64_t splitmix64(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

llama_interp::task<> capture_token(
        llama_interp::runtime & runtime,
        const llama_interp::rwkv_state & before,
        llama_token token,
        const std::vector<std::string> & taps,
        bool request_captures,
        llama_interp::rwkv_state & after,
        llama_interp::activation_set & captures) {
    auto call = runtime.prefill_tokens(before, { token });
    if (request_captures) {
        for (const std::string & tap : taps) call.capture_f16("^" + tap + "$", captures);
    }
    after = co_await call;
}

llama_interp::task<> prefill_state(
        llama_interp::runtime & runtime,
        const llama_interp::rwkv_state & initial,
        const std::vector<llama_token> & tokens,
        llama_interp::rwkv_state & output) {
    output = co_await runtime.prefill_tokens(initial, tokens);
}

struct pending_sample {
    llama_interp::rwkv_state state;
    llama_token token;
    rwkv_activation_store::sample_metadata metadata;
};

void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s -m MODEL --corpus FILE --output FILE --tap FULL_NAME [--tap FULL_NAME ...] [--max-samples N] [--positions-per-line N] [--prompt-stride N] [--capture-batch N] [--commit-rows N] [--seed N] [-ngl N]\n"
        "\n"
        "Captures named, same-width RWKV graph taps at context-bearing corpus positions into a\n"
        "ROOT RNTuple. Taps must be exact names such as rwkv.layer.60.resid.out. Captures are\n"
        "materialized snapshots in the production GGML graph before host readback.\n",
        argv0);
}

} // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    llama_log_set(quiet_llama_logs, nullptr);

    std::string model_path;
    std::string corpus_path;
    std::string output_path;
    std::vector<std::string> taps;
    int n_gpu_layers = 0;
    uint64_t max_samples = 1500;
    uint32_t positions_per_line = 1;
    uint32_t prompt_stride = 1;
    uint32_t capture_batch = 4;
    uint64_t commit_rows = 16;
    uint64_t seed = 17;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (std::strcmp(argv[i], "--corpus") == 0 && i + 1 < argc) corpus_path = argv[++i];
        else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) output_path = argv[++i];
        else if (std::strcmp(argv[i], "--tap") == 0 && i + 1 < argc) taps.emplace_back(argv[++i]);
        else if (std::strcmp(argv[i], "--max-samples") == 0 && i + 1 < argc) max_samples = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "--positions-per-line") == 0 && i + 1 < argc) positions_per_line = std::strtoul(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "--prompt-stride") == 0 && i + 1 < argc) prompt_stride = std::strtoul(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "--capture-batch") == 0 && i + 1 < argc) capture_batch = std::strtoul(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "--commit-rows") == 0 && i + 1 < argc) commit_rows = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "-ngl") == 0 && i + 1 < argc) n_gpu_layers = std::atoi(argv[++i]);
        else {
            usage(argv[0]);
            return 1;
        }
    }
    std::sort(taps.begin(), taps.end());
    const bool duplicate_taps = std::adjacent_find(taps.begin(), taps.end()) != taps.end();
    if (model_path.empty() || corpus_path.empty() || output_path.empty() || taps.empty() || duplicate_taps ||
        max_samples == 0 || positions_per_line == 0 || prompt_stride == 0 || capture_batch == 0 || commit_rows == 0) {
        usage(argv[0]);
        return 1;
    }

    std::ifstream corpus(corpus_path);
    if (!corpus) throw std::runtime_error("failed to open corpus: " + corpus_path);
    const std::filesystem::path output(output_path);
    if (!output.parent_path().empty()) std::filesystem::create_directories(output.parent_path());

    ggml_backend_load_all();
    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) throw std::runtime_error("failed to load model");
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = 512;
    context_params.n_batch = 512;
    context_params.n_ubatch = 512;
    context_params.n_seq_max = capture_batch;
    llama_context * ctx = llama_init_from_model(model, context_params);
    if (!ctx) throw std::runtime_error("failed to create context");

    llama_interp::runtime runtime(ctx, capture_batch);
    const llama_interp::rwkv_state initial = runtime.make_state();
    std::unique_ptr<rwkv_activation_store::activation_dataset_writer> writer;
    std::vector<pending_sample> pending;
    pending.reserve(capture_batch);
    const auto started = std::chrono::steady_clock::now();
    uint64_t corpus_line = 0;
    uint64_t sample_id = 0;
    uint64_t since_commit = 0;

    const auto flush = [&] {
        if (pending.empty()) return;
        llama_interp::activation_set captures;
        std::vector<llama_interp::rwkv_state> after(pending.size());
        std::vector<llama_interp::task<>> calls;
        calls.reserve(pending.size());
        for (size_t i = 0; i < pending.size(); ++i) {
            calls.emplace_back(capture_token(runtime, pending[i].state, pending[i].token, taps, i == 0, after[i], captures));
        }
        runtime.run();
        for (const auto & call : calls) call.rethrow_if_failed();

        std::unordered_map<std::string, const llama_interp_activation *> by_name;
        for (const auto & capture : captures) {
            if (!capture.data.empty() && !by_name.emplace(capture.name, &capture).second) {
                throw std::runtime_error("duplicate capture: " + capture.name);
            }
        }
        size_t dimension = 0;
        for (const std::string & tap : taps) {
            const auto found = by_name.find(tap);
            if (found == by_name.end() || found->second->data.size() % pending.size() != 0) {
                throw std::runtime_error("missing or malformed capture: " + tap);
            }
            const size_t tap_dimension = found->second->data.size() / pending.size();
            if (tap_dimension == 0 || (dimension != 0 && tap_dimension != dimension)) {
                throw std::runtime_error("selected taps do not have one shared vector width");
            }
            dimension = tap_dimension;
        }
        if (!writer) writer = std::make_unique<rwkv_activation_store::activation_dataset_writer>(
            rwkv_activation_store::activation_dataset_writer::create(output_path, taps, dimension));

        for (size_t sample = 0; sample < pending.size(); ++sample) {
            std::vector<std::span<const ggml_fp16_t>> activations;
            activations.reserve(taps.size());
            for (const std::string & tap : taps) {
                const auto & values = by_name.at(tap)->data;
                activations.emplace_back(values.data() + sample * dimension, dimension);
            }
            writer->append(pending[sample].metadata, activations);
            ++sample_id;
            ++since_commit;
        }
        pending.clear();
        if (since_commit >= commit_rows || sample_id == max_samples) {
            writer->commit();
            since_commit = 0;
            const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            const double rate = sample_id / std::max(elapsed, 1e-9);
            std::printf("progress=%llu/%llu %.1f%% corpus_line=%llu rate=%.3f samples/s\n",
                (unsigned long long) sample_id, (unsigned long long) max_samples, 100.0 * sample_id / max_samples,
                (unsigned long long) corpus_line, rate);
            std::fflush(stdout);
        }
    };

    std::string prompt;
    while (sample_id + pending.size() < max_samples && std::getline(corpus, prompt)) {
        ++corpus_line;
        if (prompt.empty() || corpus_line % prompt_stride != 1 % prompt_stride) continue;
        const std::vector<llama_token> tokens = common_tokenize(vocab, prompt, false, true);
        if (tokens.empty()) continue;
        const size_t first_position = std::min<size_t>(8, tokens.size() - 1);
        const size_t range = tokens.size() - first_position;
        const size_t positions = std::min<size_t>(positions_per_line, range);
        const size_t first_offset = splitmix64(seed ^ corpus_line) % range;
        for (size_t offset = 0; offset < positions && sample_id + pending.size() < max_samples; ++offset) {
            const size_t position = first_position + (first_offset + offset) % range;
            llama_interp::rwkv_state state = initial;
            if (position > 0) {
                auto prefix = prefill_state(runtime, initial, std::vector<llama_token>(tokens.begin(), tokens.begin() + position), state);
                runtime.run();
                prefix.rethrow_if_failed();
            }
            pending.push_back({ std::move(state), tokens[position],
                { sample_id + pending.size(), corpus_line, (int32_t) position, tokens[position], (int32_t) tokens.size() } });
            if (pending.size() == capture_batch) flush();
        }
    }
    flush();
    if (!writer) throw std::runtime_error("corpus produced no capture samples");
    writer->commit();
    writer.reset();
    const auto saved = rwkv_activation_store::activation_dataset_reader::open(output_path);
    if (saved.entries() != sample_id || saved.sources() != taps || saved.dimension() == 0) {
        throw std::runtime_error("ROOT capture verification failed");
    }
    std::printf("wrote=%s samples=%llu\n", output_path.c_str(), (unsigned long long) sample_id);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return sample_id == max_samples ? 0 : 2;
}
