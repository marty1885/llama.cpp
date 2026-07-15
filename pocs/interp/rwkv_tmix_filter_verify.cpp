#include "rwkv_experiment.hpp"

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

struct step_capture {
    llama_interp::rwkv_state before;
    llama_token token;
    llama_interp::activation_set activations;
};

const llama_interp_activation & capture(const llama_interp::activation_set & activations, const std::string & name) {
    return rwkv_experiment::require_capture(activations, name);
}

float dot(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) throw std::runtime_error("vector size mismatch");
    float result = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) result += a[i] * b[i];
    return result;
}

float norm(const std::vector<float> & value) {
    return std::sqrt(dot(value, value));
}

float cosine(const std::vector<float> & a, const std::vector<float> & b) {
    const float denom = norm(a) * norm(b);
    return denom == 0.0f ? 0.0f : dot(a, b) / denom;
}

std::vector<float> subtract(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) throw std::runtime_error("vector size mismatch");
    std::vector<float> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) result[i] = a[i] - b[i];
    return result;
}

llama_interp::task<> collect_steps(
        llama_interp::runtime & runtime,
        const llama_interp::rwkv_state & initial,
        const std::vector<llama_token> & prompt,
        int32_t n_generate,
        const std::vector<std::string> & taps,
        std::vector<step_capture> & output) {
    llama_interp::rwkv_state state = initial;
    const int32_t total = (int32_t) prompt.size() + n_generate;
    for (int32_t position = 0; position < total; ++position) {
        const llama_token token = position < (int32_t) prompt.size() ? prompt[position] : state.next_token;
        if (position >= (int32_t) prompt.size() && !state.has_next) throw std::runtime_error("generation state has no next token");
        llama_interp::activation_set activations;
        auto call = runtime.prefill_tokens(state, { token });
        for (const std::string & tap : taps) call.capture_f32("^" + tap + "$", activations);
        const llama_interp::rwkv_state before = state;
        state = co_await call;
        output.push_back({ before, token, std::move(activations) });
    }
}

llama_interp::task<> perturb_step(
        llama_interp::runtime & runtime,
        const step_capture & source,
        const std::string & perturb_tap,
        std::vector<ggml_fp16_t> perturbation,
        const std::vector<std::string> & taps,
        llama_interp::activation_set & output) {
    auto call = runtime.prefill_tokens(source.before, { source.token });
    call.perturb("^" + perturb_tap + "$", llama_interp::runtime::add_head(-1, std::move(perturbation)));
    for (const std::string & tap : taps) call.capture_f32("^" + tap + "$", output);
    (void) co_await call;
}

llama_token argmax(const std::vector<float> & logits) {
    return (llama_token) std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()));
}

void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s -m MODEL -p PROMPT --output FILE [--generate N] [--mean-first-layer L] [--target-layer L] [-ngl N]\n",
        argv0);
}

} // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    llama_log_set(quiet_llama_logs, nullptr);
    std::string model_path, prompt, output_path;
    int32_t n_generate = 16;
    int32_t mean_first_layer = 48;
    int32_t target_layer = -1;
    int n_gpu_layers = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (std::strcmp(argv[i], "-p") == 0 && i + 1 < argc) prompt = argv[++i];
        else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) output_path = argv[++i];
        else if (std::strcmp(argv[i], "--generate") == 0 && i + 1 < argc) n_generate = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--mean-first-layer") == 0 && i + 1 < argc) mean_first_layer = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--target-layer") == 0 && i + 1 < argc) target_layer = std::atoi(argv[++i]);
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
    if (target_layer < 0) target_layer = n_layer - 1;
    if (mean_first_layer < 0 || mean_first_layer > target_layer || target_layer >= n_layer) {
        throw std::runtime_error("invalid layer range");
    }
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
    const int32_t width = (int32_t) runtime.make_state().n_embd_r / 2;
    std::vector<std::string> mean_taps;
    for (int32_t layer = mean_first_layer; layer <= target_layer; ++layer) {
        mean_taps.push_back("rwkv.layer." + std::to_string(layer) + ".time.out");
    }
    const std::string target_prefix = "rwkv.layer." + std::to_string(target_layer) + ".";
    const std::string target_time_out = target_prefix + "time.out";
    const std::string target_resid_time = target_prefix + "resid.time";
    const std::string target_ffn_norm = target_prefix + "ffn.norm";
    const std::string target_key_preact = target_prefix + "channel.key_preact";
    const std::string target_key_relu_sq = target_prefix + "channel.key_relu_sq";
    const std::string target_channel = target_prefix + "channel.out";
    const std::string target_resid = target_prefix + "resid.out";
    std::vector<std::string> taps = mean_taps;
    taps.push_back(target_resid_time);
    taps.push_back(target_ffn_norm);
    taps.push_back(target_key_preact);
    taps.push_back(target_key_relu_sq);
    taps.push_back(target_channel);
    taps.push_back(target_resid);

    std::vector<step_capture> steps;
    std::fprintf(stderr, "stage=collect prompt_tokens=%zu generate=%d layers=%d..%d\n",
                 tokens.size(), n_generate, mean_first_layer, target_layer);
    auto collection = collect_steps(runtime, runtime.make_state(), tokens, n_generate, taps, steps);
    runtime.run();
    collection.rethrow_if_failed();

    std::vector<float> direction(width);
    int32_t direction_count = 0;
    float pairwise_alignment_sum = 0.0f;
    std::vector<std::vector<float>> normalized;
    for (const auto & step : steps) {
        for (const std::string & tap : mean_taps) {
            std::vector<float> value = capture(step.activations, tap).data_f32;
            const float value_norm = norm(value);
            if (value_norm == 0.0f) continue;
            for (float & x : value) x /= value_norm;
            for (const auto & previous : normalized) pairwise_alignment_sum += dot(value, previous);
            normalized.push_back(value);
            for (int32_t i = 0; i < width; ++i) direction[i] += value[i];
            ++direction_count;
        }
    }
    const float direction_norm = norm(direction);
    if (direction_count == 0 || direction_norm == 0.0f) throw std::runtime_error("could not estimate a shared direction");
    for (float & x : direction) x /= direction_norm;
    const float pairwise_alignment = normalized.size() < 2 ? 0.0f :
        pairwise_alignment_sum / ((float) normalized.size() * (normalized.size() - 1) / 2.0f);

    const step_capture & probe = steps[tokens.size() - 1];
    const std::vector<float> native_time = capture(probe.activations, target_time_out).data_f32;
    const std::vector<float> native_resid_time = capture(probe.activations, target_resid_time).data_f32;
    const std::vector<float> native_ffn_norm = capture(probe.activations, target_ffn_norm).data_f32;
    const std::vector<float> native_key_preact = capture(probe.activations, target_key_preact).data_f32;
    const std::vector<float> native_key_relu_sq = capture(probe.activations, target_key_relu_sq).data_f32;
    const std::vector<float> native_channel = capture(probe.activations, target_channel).data_f32;
    const std::vector<float> native_resid = capture(probe.activations, target_resid).data_f32;
    const float alpha = dot(native_time, direction);
    std::vector<float> full_component(width);
    for (int32_t i = 0; i < width; ++i) full_component[i] = alpha * direction[i];
    const float component_mean = std::accumulate(full_component.begin(), full_component.end(), 0.0f) / width;
    const float residual_mean = std::accumulate(native_resid_time.begin(), native_resid_time.end(), 0.0f) / width;
    std::vector<float> mean_component(width), centered_component(width), centered_residual(width);
    for (int32_t i = 0; i < width; ++i) {
        mean_component[i] = component_mean;
        centered_component[i] = full_component[i] - component_mean;
        centered_residual[i] = native_resid_time[i] - residual_mean;
    }
    const float centered_residual_norm = norm(centered_residual);
    if (centered_residual_norm == 0.0f) throw std::runtime_error("zero centered residual");
    for (float & x : centered_residual) x /= centered_residual_norm;
    const float radial_scale = dot(centered_component, centered_residual);
    std::vector<float> radial_component(width), tangent_component(width);
    for (int32_t i = 0; i < width; ++i) {
        radial_component[i] = radial_scale * centered_residual[i];
        tangent_component[i] = centered_component[i] - radial_component[i];
    }

    const std::vector<std::string> ablation_taps = {
        target_time_out, target_resid_time, target_ffn_norm, target_key_preact, target_key_relu_sq, target_channel, target_resid,
    };

    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<float> native_logits(n_vocab);
    if (!llama_interp_rwkv_final_readout(ctx, native_resid.data(), 1, native_logits.data())) throw std::runtime_error("final residual readout failed");

    struct component_result {
        std::string name;
        float fraction;
        float ffn_norm_relative_delta;
        float ffn_norm_to_resid_time_ratio;
        float ffn_opposes_time_cosine;
        float residual_survival_fraction;
        llama_token next_token;
        float max_logit_delta;
    };
    std::vector<component_result> results;
    std::vector<float> tangent_ffn_norm_delta;
    std::vector<float> tangent_key_preact_delta;
    std::vector<float> tangent_key_relu_sq_delta;
    const std::vector<std::pair<std::string, const std::vector<float> *>> components = {
        { "mean", &mean_component }, { "radial", &radial_component },
        { "tangent", &tangent_component }, { "full", &full_component },
    };
    for (const auto & [name, component] : components) {
        std::vector<ggml_fp16_t> perturbation(width);
        for (int32_t i = 0; i < width; ++i) perturbation[i] = ggml_fp32_to_fp16(-(*component)[i]);
        llama_interp::activation_set ablated;
        auto ablation = perturb_step(runtime, probe, target_time_out, std::move(perturbation), ablation_taps, ablated);
        runtime.run();
        ablation.rethrow_if_failed();
        const std::vector<float> delta_time = subtract(native_time, capture(ablated, target_time_out).data_f32);
        const std::vector<float> delta_resid_time = subtract(native_resid_time, capture(ablated, target_resid_time).data_f32);
        const std::vector<float> delta_ffn_norm = subtract(native_ffn_norm, capture(ablated, target_ffn_norm).data_f32);
        const std::vector<float> delta_key_preact = subtract(native_key_preact, capture(ablated, target_key_preact).data_f32);
        const std::vector<float> delta_key_relu_sq = subtract(native_key_relu_sq, capture(ablated, target_key_relu_sq).data_f32);
        const std::vector<float> delta_channel = subtract(native_channel, capture(ablated, target_channel).data_f32);
        const std::vector<float> delta_resid = subtract(native_resid, capture(ablated, target_resid).data_f32);
        std::vector<float> negative_delta_channel = delta_channel;
        for (float & x : negative_delta_channel) x = -x;
        std::vector<float> ablated_logits(n_vocab);
        if (!llama_interp_rwkv_final_readout(ctx, capture(ablated, target_resid).data_f32.data(), 1, ablated_logits.data())) {
            throw std::runtime_error("final residual readout failed");
        }
        float max_logit_delta = 0.0f;
        for (int32_t i = 0; i < n_vocab; ++i) max_logit_delta = std::max(max_logit_delta, std::abs(native_logits[i] - ablated_logits[i]));
        const float full_norm = norm(full_component);
        results.push_back({
            name,
            full_norm == 0.0f ? 0.0f : norm(*component) / full_norm,
            norm(delta_ffn_norm) / norm(native_ffn_norm),
            norm(delta_resid_time) == 0.0f ? 0.0f : norm(delta_ffn_norm) / norm(delta_resid_time),
            cosine(negative_delta_channel, delta_time),
            norm(delta_time) == 0.0f ? 0.0f : norm(delta_resid) / norm(delta_time),
            argmax(ablated_logits),
            max_logit_delta,
        });
        if (name == "tangent") {
            tangent_ffn_norm_delta = delta_ffn_norm;
            tangent_key_preact_delta = delta_key_preact;
            tangent_key_relu_sq_delta = delta_key_relu_sq;
        }
    }

    std::vector<float> tangent_lens_logits(n_vocab);
    if (!llama_interp_rwkv_final_readout(ctx, tangent_ffn_norm_delta.data(), 1, tangent_lens_logits.data())) {
        throw std::runtime_error("tangent FFN-normalized lens failed");
    }
    auto top_absolute = [](const std::vector<float> & values, int32_t count) {
        std::vector<int32_t> indices(values.size());
        std::iota(indices.begin(), indices.end(), 0);
        count = std::min<int32_t>(count, (int32_t) indices.size());
        std::partial_sort(indices.begin(), indices.begin() + count, indices.end(), [&values](int32_t a, int32_t b) {
            return std::abs(values[a]) > std::abs(values[b]);
        });
        indices.resize(count);
        return indices;
    };
    auto top_positive = [&top_absolute](const std::vector<float> & values, int32_t count) {
        std::vector<int32_t> indices(values.size());
        std::iota(indices.begin(), indices.end(), 0);
        count = std::min<int32_t>(count, (int32_t) indices.size());
        std::partial_sort(indices.begin(), indices.begin() + count, indices.end(), [&values](int32_t a, int32_t b) {
            return values[a] > values[b];
        });
        indices.resize(count);
        return indices;
    };
    const std::vector<int32_t> tangent_lens_top = top_positive(tangent_lens_logits, 16);
    const std::vector<int32_t> key_preact_top = top_absolute(tangent_key_preact_delta, 16);
    const std::vector<int32_t> key_relu_sq_top = top_absolute(tangent_key_relu_sq_delta, 16);

    std::ofstream output(output_path);
    if (!output) throw std::runtime_error("failed to open output: " + output_path);
    output << std::setprecision(9) << "{\n"
           << "  \"model\": \"" << model_path << "\",\n"
           << "  \"probe_token\": {\"id\": " << probe.token << ", \"text\": \""
           << common_token_to_piece(ctx, probe.token, true) << "\"},\n"
           << "  \"target_layer\": " << target_layer << ",\n"
           << "  \"mean_layers\": [" << mean_first_layer << ", " << target_layer << "],\n"
           << "  \"direction_samples\": " << direction_count << ",\n"
           << "  \"mean_pairwise_cosine\": " << pairwise_alignment << ",\n"
           << "  \"native_next_token\": {\"id\": " << argmax(native_logits) << ", \"text\": \""
           << common_token_to_piece(ctx, argmax(native_logits), true) << "\"},\n"
           << "  \"components\": [";
    for (size_t i = 0; i < results.size(); ++i) {
        const auto & result = results[i];
        if (i) output << ',';
        output << "\n    {\"name\": \"" << result.name << "\", \"norm_fraction\": " << result.fraction
               << ", \"ffn_norm_relative_delta\": " << result.ffn_norm_relative_delta
               << ", \"ffn_norm_to_resid_time_ratio\": " << result.ffn_norm_to_resid_time_ratio
               << ", \"ffn_opposes_time_cosine\": " << result.ffn_opposes_time_cosine
               << ", \"residual_survival_fraction\": " << result.residual_survival_fraction
               << ", \"next_token\": {\"id\": " << result.next_token << ", \"text\": \""
               << common_token_to_piece(ctx, result.next_token, true) << "\"}, \"max_logit_delta\": " << result.max_logit_delta << '}';
    }
    output << "\n  ],\n  \"tangent_ffn_norm_lens\": [";
    for (size_t i = 0; i < tangent_lens_top.size(); ++i) {
        const int32_t token = tangent_lens_top[i];
        if (i) output << ',';
        output << "\n    {\"token_id\": " << token << ", \"token\": ";
        rwkv_experiment::write_json_string(output, common_token_to_piece(ctx, token, true));
        output << ", \"logit\": " << tangent_lens_logits[token] << '}';
    }
    output << "\n  ],\n  \"tangent_key_features\": {\n    \"preact\": [";
    for (size_t i = 0; i < key_preact_top.size(); ++i) {
        const int32_t feature = key_preact_top[i];
        if (i) output << ',';
        output << "\n      {\"feature\": " << feature << ", \"delta\": " << tangent_key_preact_delta[feature]
               << ", \"native\": " << native_key_preact[feature] << '}';
    }
    output << "\n    ],\n    \"relu_sq\": [";
    for (size_t i = 0; i < key_relu_sq_top.size(); ++i) {
        const int32_t feature = key_relu_sq_top[i];
        if (i) output << ',';
        output << "\n      {\"feature\": " << feature << ", \"delta\": " << tangent_key_relu_sq_delta[feature]
               << ", \"native\": " << native_key_relu_sq[feature] << '}';
    }
    output << "\n    ]\n  }\n}\n";
    std::printf("wrote=%s samples=%d mean_pairwise_cosine=%g components=%zu\n",
                output_path.c_str(), direction_count, pairwise_alignment, results.size());
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
