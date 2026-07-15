#include "rwkv_experiment.hpp"

#include "ggml-backend.h"
#include "llama-model.h"

#include <cmath>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void quiet_llama_logs(ggml_log_level, const char *, void *) {}

float dot(const float * a, const float * b, int32_t width) {
    float result = 0.0f;
    for (int32_t i = 0; i < width; ++i) result += a[i] * b[i];
    return result;
}

float norm(const float * value, int32_t width) {
    return std::sqrt(dot(value, value, width));
}

llama_interp::task<> collect(
        llama_interp::runtime & runtime,
        const llama_interp::rwkv_state & initial,
        const std::vector<llama_token> & prompt,
        int32_t n_generate,
        const std::vector<std::string> & taps,
        std::vector<llama_interp::activation_set> & output) {
    llama_interp::rwkv_state state = initial;
    const int32_t total = (int32_t) prompt.size() + n_generate;
    for (int32_t position = 0; position < total; ++position) {
        const llama_token token = position < (int32_t) prompt.size() ? prompt[position] : state.next_token;
        if (position >= (int32_t) prompt.size() && !state.has_next) throw std::runtime_error("generation state has no next token");
        llama_interp::activation_set activations;
        auto call = runtime.prefill_tokens(state, { token });
        for (const std::string & tap : taps) call.capture_f32("^" + tap + "$", activations);
        state = co_await call;
        output.push_back(std::move(activations));
    }
}

void usage(const char * argv0) {
    std::fprintf(stderr, "usage: %s -m MODEL -p PROMPT --output FILE [--generate N] [--layer L] [-ngl N]\n", argv0);
}

} // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    llama_log_set(quiet_llama_logs, nullptr);
    std::string model_path, prompt, output_path;
    int32_t n_generate = 32;
    int32_t layer = -1;
    int n_gpu_layers = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (std::strcmp(argv[i], "-p") == 0 && i + 1 < argc) prompt = argv[++i];
        else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) output_path = argv[++i];
        else if (std::strcmp(argv[i], "--generate") == 0 && i + 1 < argc) n_generate = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--layer") == 0 && i + 1 < argc) layer = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-ngl") == 0 && i + 1 < argc) n_gpu_layers = std::atoi(argv[++i]);
        else { usage(argv[0]); return 1; }
    }
    if (model_path.empty() || prompt.empty() || output_path.empty() || n_generate < 0) { usage(argv[0]); return 1; }

    ggml_backend_load_all();
    llama_backend_init();
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = n_gpu_layers;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), params);
    if (!model || model->arch != LLM_ARCH_RWKV7) throw std::runtime_error("an RWKV-7 model is required");
    const int32_t n_layer = model->hparams.n_layer();
    if (layer < 0) layer = n_layer - 1;
    if (layer < 0 || layer >= n_layer) throw std::runtime_error("invalid layer");
    const int32_t width = (int32_t) model->layers[layer].time_mix_output->ne[0];
    const int32_t head_size = (int32_t) model->hparams.wkv_head_size;
    if (head_size <= 0 || width % head_size != 0) throw std::runtime_error("invalid RWKV head geometry");
    const int32_t n_head = width / head_size;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const std::vector<llama_token> tokens = common_tokenize(vocab, prompt, false, true);
    if (tokens.empty()) throw std::runtime_error("prompt tokenized to zero tokens");

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = std::max(128, (int) tokens.size() + n_generate + 8);
    context_params.n_batch = context_params.n_ctx;
    context_params.n_ubatch = context_params.n_ctx;
    llama_context * ctx = llama_init_from_model(model, context_params);
    if (!ctx) throw std::runtime_error("failed to create context");
    llama_interp::runtime runtime(ctx, 1);
    const std::string prefix = "rwkv.layer." + std::to_string(layer) + ".time.";
    const std::string pre_output_tap = prefix + "pre_output";
    const std::string out_tap = prefix + "out";
    const std::vector<std::string> taps = { pre_output_tap, out_tap };
    std::vector<llama_interp::activation_set> captures;
    std::fprintf(stderr, "stage=collect layer=%d heads=%d head_size=%d generate=%d\n", layer, n_head, head_size, n_generate);
    auto run = collect(runtime, runtime.make_state(), tokens, n_generate, taps, captures);
    runtime.run();
    run.rethrow_if_failed();

    std::vector<float> head_energy(n_head), head_coherence(n_head), head_sum((size_t) n_head * head_size);
    float pre_output_pairwise_cosine = 0.0f;
    float out_pairwise_cosine = 0.0f;
    for (size_t sample = 0; sample < captures.size(); ++sample) {
        const auto & pre_output = rwkv_experiment::require_capture(captures[sample], pre_output_tap).data_f32;
        const auto & out = rwkv_experiment::require_capture(captures[sample], out_tap).data_f32;
        const float pre_norm = norm(pre_output.data(), width);
        const float out_norm = norm(out.data(), width);
        for (size_t earlier = 0; earlier < sample; ++earlier) {
            const auto & earlier_pre = rwkv_experiment::require_capture(captures[earlier], pre_output_tap).data_f32;
            const auto & earlier_out = rwkv_experiment::require_capture(captures[earlier], out_tap).data_f32;
            pre_output_pairwise_cosine += dot(pre_output.data(), earlier_pre.data(), width) / (pre_norm * norm(earlier_pre.data(), width));
            out_pairwise_cosine += dot(out.data(), earlier_out.data(), width) / (out_norm * norm(earlier_out.data(), width));
        }
        for (int32_t head = 0; head < n_head; ++head) {
            const float * block = pre_output.data() + (size_t) head * head_size;
            const float block_norm = norm(block, head_size);
            head_energy[head] += block_norm * block_norm / (pre_norm * pre_norm);
            if (block_norm != 0.0f) for (int32_t d = 0; d < head_size; ++d) head_sum[(size_t) head * head_size + d] += block[d] / block_norm;
        }
    }
    const float pair_count = (float) captures.size() * (captures.size() - 1) / 2.0f;
    if (pair_count > 0.0f) { pre_output_pairwise_cosine /= pair_count; out_pairwise_cosine /= pair_count; }
    for (int32_t head = 0; head < n_head; ++head) {
        head_energy[head] /= captures.size();
        head_coherence[head] = norm(head_sum.data() + (size_t) head * head_size, head_size) / captures.size();
    }

    std::ofstream output(output_path);
    if (!output) throw std::runtime_error("failed to open output: " + output_path);
    output << std::setprecision(9) << "{\n  \"layer\": " << layer << ",\n  \"samples\": " << captures.size()
           << ",\n  \"n_head\": " << n_head << ",\n  \"head_size\": " << head_size
           << ",\n  \"pre_output_mean_pairwise_cosine\": " << pre_output_pairwise_cosine
           << ",\n  \"time_out_mean_pairwise_cosine\": " << out_pairwise_cosine << ",\n  \"heads\": [";
    for (int32_t head = 0; head < n_head; ++head) {
        if (head) output << ',';
        output << "\n    {\"head\": " << head << ", \"mean_energy_fraction\": " << head_energy[head]
               << ", \"directional_coherence\": " << head_coherence[head] << '}';
    }
    output << "\n  ]\n}\n";
    std::printf("wrote=%s layer=%d samples=%zu pre_output_pairwise_cosine=%g time_out_pairwise_cosine=%g\n",
                output_path.c_str(), layer, captures.size(), pre_output_pairwise_cosine, out_pairwise_cosine);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
