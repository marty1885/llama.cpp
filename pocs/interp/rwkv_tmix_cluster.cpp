#include "rwkv_experiment.hpp"

#include "ggml-backend.h"
#include "llama-model.h"

#include <algorithm>
#include <cmath>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void quiet_llama_logs(ggml_log_level, const char *, void *) {}

struct sample {
    int32_t position;
    int32_t layer;
    llama_token token;
    std::vector<float> value;
};

float dot(const float * a, const float * b, int32_t width) {
    float result = 0.0f;
    for (int32_t i = 0; i < width; ++i) result += a[i] * b[i];
    return result;
}

float normalize(std::vector<float> & value) {
    const float length = std::sqrt(dot(value.data(), value.data(), (int32_t) value.size()));
    if (length != 0.0f) for (float & x : value) x /= length;
    return length;
}

llama_interp::task<> collect(
        llama_interp::runtime & runtime,
        const llama_interp::rwkv_state & initial,
        const std::vector<llama_token> & prompt,
        int32_t n_generate,
        const std::vector<std::string> & taps,
        std::vector<sample> & output) {
    llama_interp::rwkv_state state = initial;
    const int32_t total = (int32_t) prompt.size() + n_generate;
    for (int32_t position = 0; position < total; ++position) {
        const llama_token token = position < (int32_t) prompt.size() ? prompt[position] : state.next_token;
        if (position >= (int32_t) prompt.size() && !state.has_next) throw std::runtime_error("generation state has no next token");
        llama_interp::activation_set activations;
        auto call = runtime.prefill_tokens(state, { token });
        for (const std::string & tap : taps) call.capture_f32("^" + tap + "$", activations);
        state = co_await call;
        for (int32_t layer = 0; layer < (int32_t) taps.size(); ++layer) {
            const auto & activation = rwkv_experiment::require_capture(activations, taps[layer]);
            output.push_back({ position, layer, token, activation.data_f32 });
        }
    }
}

void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s -m MODEL -p PROMPT --output FILE [--generate N] [--clusters K] [--top-n N] [-ngl N]\n",
        argv0);
}

} // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    llama_log_set(quiet_llama_logs, nullptr);
    std::string model_path, prompt, output_path;
    int32_t n_generate = 32;
    int32_t k = 8;
    int32_t top_n = 10;
    int n_gpu_layers = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (std::strcmp(argv[i], "-p") == 0 && i + 1 < argc) prompt = argv[++i];
        else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) output_path = argv[++i];
        else if (std::strcmp(argv[i], "--generate") == 0 && i + 1 < argc) n_generate = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--clusters") == 0 && i + 1 < argc) k = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--top-n") == 0 && i + 1 < argc) top_n = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-ngl") == 0 && i + 1 < argc) n_gpu_layers = std::atoi(argv[++i]);
        else { usage(argv[0]); return 1; }
    }
    if (model_path.empty() || prompt.empty() || output_path.empty() || n_generate < 0 || k <= 0 || top_n <= 0) {
        usage(argv[0]);
        return 1;
    }

    ggml_backend_load_all();
    llama_backend_init();
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = n_gpu_layers;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), params);
    if (!model || model->arch != LLM_ARCH_RWKV7) throw std::runtime_error("an RWKV-7 model is required");
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const std::vector<llama_token> tokens = common_tokenize(vocab, prompt, false, true);
    if (tokens.empty()) throw std::runtime_error("prompt tokenized to zero tokens");
    const int32_t n_layer = model->hparams.n_layer();

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = std::max(128, (int) tokens.size() + n_generate + 8);
    context_params.n_batch = context_params.n_ctx;
    context_params.n_ubatch = context_params.n_ctx;
    llama_context * ctx = llama_init_from_model(model, context_params);
    if (!ctx) throw std::runtime_error("failed to create context");
    llama_interp::runtime runtime(ctx, 1);
    const int32_t width = (int32_t) runtime.make_state().n_embd_r / 2;
    std::vector<std::string> taps;
    for (int32_t layer = 0; layer < n_layer; ++layer) taps.push_back("rwkv.layer." + std::to_string(layer) + ".time.out");
    std::vector<sample> samples;
    samples.reserve(((size_t) tokens.size() + n_generate) * n_layer);
    std::fprintf(stderr, "stage=collect layers=%d prompt_tokens=%zu generate=%d\n", n_layer, tokens.size(), n_generate);
    auto run = collect(runtime, runtime.make_state(), tokens, n_generate, taps, samples);
    runtime.run();
    run.rethrow_if_failed();
    if (samples.empty()) throw std::runtime_error("no time.out samples captured");
    k = std::min<int32_t>(k, (int32_t) samples.size());

    std::vector<float> magnitudes(samples.size());
    for (size_t i = 0; i < samples.size(); ++i) magnitudes[i] = normalize(samples[i].value);

    // Farthest-point initialization avoids choosing several nearly identical late-layer vectors.
    std::vector<std::vector<float>> centroids;
    centroids.push_back(samples[0].value);
    while ((int32_t) centroids.size() < k) {
        size_t best = 0;
        float best_distance = -1.0f;
        for (size_t i = 0; i < samples.size(); ++i) {
            float nearest = 1.0f;
            for (const auto & centroid : centroids) nearest = std::min(nearest, 1.0f - dot(samples[i].value.data(), centroid.data(), width));
            if (nearest > best_distance) { best_distance = nearest; best = i; }
        }
        centroids.push_back(samples[best].value);
    }

    std::vector<int32_t> assignment(samples.size(), -1);
    for (int iteration = 0; iteration < 12; ++iteration) {
        bool changed = false;
        std::vector<std::vector<float>> sums(k, std::vector<float>(width));
        std::vector<int32_t> counts(k);
        for (size_t i = 0; i < samples.size(); ++i) {
            int32_t best = 0;
            float best_score = dot(samples[i].value.data(), centroids[0].data(), width);
            for (int32_t cluster = 1; cluster < k; ++cluster) {
                const float score = dot(samples[i].value.data(), centroids[cluster].data(), width);
                if (score > best_score) { best_score = score; best = cluster; }
            }
            changed |= assignment[i] != best;
            assignment[i] = best;
            ++counts[best];
            for (int32_t d = 0; d < width; ++d) sums[best][d] += samples[i].value[d];
        }
        for (int32_t cluster = 0; cluster < k; ++cluster) if (counts[cluster] != 0) {
            normalize(sums[cluster]);
            centroids[cluster] = std::move(sums[cluster]);
        }
        std::fprintf(stderr, "stage=kmeans iteration=%d changed=%d\n", iteration + 1, changed ? 1 : 0);
        if (!changed) break;
    }

    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<float> centroid_rows((size_t) k * width);
    for (int32_t cluster = 0; cluster < k; ++cluster) {
        std::copy(centroids[cluster].begin(), centroids[cluster].end(), centroid_rows.begin() + (size_t) cluster * width);
    }
    std::vector<float> centroid_logits((size_t) k * n_vocab);
    if (!llama_interp_rwkv_final_readout(ctx, centroid_rows.data(), (uint32_t) k, centroid_logits.data())) {
        throw std::runtime_error("centroid final readout failed");
    }
    std::vector<int32_t> counts(k);
    std::vector<float> mean_cosine(k), mean_magnitude(k);
    std::vector<std::vector<int32_t>> layer_counts(k, std::vector<int32_t>(n_layer));
    for (size_t i = 0; i < samples.size(); ++i) {
        const int32_t cluster = assignment[i];
        ++counts[cluster];
        mean_cosine[cluster] += dot(samples[i].value.data(), centroids[cluster].data(), width);
        mean_magnitude[cluster] += magnitudes[i];
        ++layer_counts[cluster][samples[i].layer];
    }
    for (int32_t cluster = 0; cluster < k; ++cluster) if (counts[cluster]) {
        mean_cosine[cluster] /= counts[cluster];
        mean_magnitude[cluster] /= counts[cluster];
    }

    std::ofstream output(output_path);
    if (!output) throw std::runtime_error("failed to open output: " + output_path);
    output << std::setprecision(9) << "{\n  \"model\": ";
    rwkv_experiment::write_json_string(output, model_path);
    output << ",\n  \"prompt\": ";
    rwkv_experiment::write_json_string(output, prompt);
    output << ",\n  \"cluster_method\": \"cosine_kmeans\",\n  \"clusters\": [";
    for (int32_t cluster = 0; cluster < k; ++cluster) {
        if (cluster) output << ',';
        std::vector<int32_t> indices(n_vocab);
        std::iota(indices.begin(), indices.end(), 0);
        const int32_t count = std::min(top_n, n_vocab);
        const float * logits = centroid_logits.data() + (size_t) cluster * n_vocab;
        std::partial_sort(indices.begin(), indices.begin() + count, indices.end(),
            [logits](int32_t a, int32_t b) { return logits[a] > logits[b]; });
        output << "\n    {\"id\": " << cluster << ", \"count\": " << counts[cluster]
               << ", \"mean_cosine\": " << mean_cosine[cluster] << ", \"mean_norm\": " << mean_magnitude[cluster]
               << ", \"layer_counts\": [";
        for (int32_t layer = 0; layer < n_layer; ++layer) { if (layer) output << ','; output << layer_counts[cluster][layer]; }
        output << "], \"top_logits\": [";
        for (int32_t i = 0; i < count; ++i) {
            if (i) output << ',';
            output << "{\"token_id\": " << indices[i] << ", \"token\": ";
            rwkv_experiment::write_json_string(output, common_token_to_piece(ctx, indices[i], true));
            output << ", \"logit\": " << logits[indices[i]] << '}';
        }
        output << "]}";
    }
    output << "\n  ],\n  \"assignments\": [";
    for (size_t i = 0; i < samples.size(); ++i) {
        if (i) output << ',';
        output << "\n    {\"position\": " << samples[i].position << ", \"layer\": " << samples[i].layer
               << ", \"cluster\": " << assignment[i] << ", \"token_id\": " << samples[i].token << '}';
    }
    output << "\n  ]\n}\n";
    std::printf("wrote=%s samples=%zu clusters=%d\n", output_path.c_str(), samples.size(), k);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
