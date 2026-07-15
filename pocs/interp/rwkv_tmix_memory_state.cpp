#include "rwkv_experiment.hpp"

#include "llama-model.h"

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

llama_interp::task<> evaluate_step(
        llama_interp::runtime & runtime,
        const llama_interp::rwkv_state & state,
        llama_token token,
        const std::vector<std::string> & taps,
        llama_interp::activation_set & output) {
    auto call = runtime.prefill_tokens(state, { token });
    for (const std::string & tap : taps) call.capture_f32("^" + tap + "$", output);
    call.discard_state();
    (void) co_await call;
}

std::vector<float> tangent_projection(
        const std::vector<float> & time_out,
        const std::vector<float> & resid_time,
        const std::vector<float> & direction) {
    if (time_out.size() != resid_time.size() || time_out.size() != direction.size()) {
        throw std::runtime_error("vector size mismatch");
    }
    const int32_t width = (int32_t) time_out.size();
    const float alpha = dot(time_out, direction);
    const float component_mean = alpha * std::accumulate(direction.begin(), direction.end(), 0.0f) / width;
    const float residual_mean = std::accumulate(resid_time.begin(), resid_time.end(), 0.0f) / width;
    std::vector<float> centered_residual(width), tangent(width);
    for (int32_t i = 0; i < width; ++i) centered_residual[i] = resid_time[i] - residual_mean;
    const float centered_residual_norm = norm(centered_residual);
    if (centered_residual_norm == 0.0f) throw std::runtime_error("zero centered residual");
    for (float & value : centered_residual) value /= centered_residual_norm;
    const float radial_scale = alpha * dot(direction, centered_residual);
    for (int32_t i = 0; i < width; ++i) {
        tangent[i] = alpha * direction[i] - component_mean - radial_scale * centered_residual[i];
    }
    return tangent;
}

void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s -m MODEL -p PROMPT --output FILE [--generate N] [--target-layer L] [--probe-position N] [-ngl N]\n",
        argv0);
}

} // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    llama_log_set(quiet_llama_logs, nullptr);
    std::string model_path, prompt, output_path;
    int32_t n_generate = 16;
    int32_t target_layer = -1;
    int32_t probe_position = -1;
    int n_gpu_layers = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (std::strcmp(argv[i], "-p") == 0 && i + 1 < argc) prompt = argv[++i];
        else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) output_path = argv[++i];
        else if (std::strcmp(argv[i], "--generate") == 0 && i + 1 < argc) n_generate = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--target-layer") == 0 && i + 1 < argc) target_layer = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--probe-position") == 0 && i + 1 < argc) probe_position = std::atoi(argv[++i]);
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
    if (target_layer < 0 || target_layer >= n_layer) throw std::runtime_error("invalid target layer");
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const std::vector<llama_token> tokens = common_tokenize(vocab, prompt, false, true);
    if (tokens.empty()) throw std::runtime_error("prompt tokenized to zero tokens");
    if (probe_position < 0) probe_position = (int32_t) tokens.size() - 1;
    if (probe_position < 0 || probe_position >= (int32_t) tokens.size() + n_generate) throw std::runtime_error("invalid probe position");

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = std::max(128, (int) tokens.size() + n_generate + 8);
    context_params.n_batch = context_params.n_ctx;
    context_params.n_ubatch = context_params.n_ctx;
    llama_context * ctx = llama_init_from_model(model, context_params);
    if (!ctx) throw std::runtime_error("failed to create context");

    llama_interp::runtime runtime(ctx, 1);
    const int32_t width = (int32_t) runtime.make_state().n_embd_r / 2;
    const std::string prefix = "rwkv.layer." + std::to_string(target_layer) + ".";
    const std::string time_wkv = prefix + "time.wkv";
    const std::string time_rkv = prefix + "time.rkv";
    const std::string time_pre_output = prefix + "time.pre_output";
    const std::string time_out = prefix + "time.out";
    const std::string resid_time = prefix + "resid.time";
    const std::string ffn_norm = prefix + "ffn.norm";
    const std::vector<std::string> taps = { time_wkv, time_rkv, time_pre_output, time_out, resid_time, ffn_norm };

    std::vector<step_capture> steps;
    auto collection = collect_steps(runtime, runtime.make_state(), tokens, n_generate, taps, steps);
    runtime.run();
    collection.rethrow_if_failed();
    if (steps.size() < 2) throw std::runtime_error("need a direction sample distinct from the probe");

    std::vector<float> direction(width);
    int32_t direction_samples = 0;
    for (int32_t position = 0; position < (int32_t) steps.size(); ++position) {
        if (position == probe_position) continue;
        std::vector<float> value = capture(steps[position].activations, time_out).data_f32;
        const float value_norm = norm(value);
        if (value_norm == 0.0f) continue;
        for (int32_t i = 0; i < width; ++i) direction[i] += value[i] / value_norm;
        ++direction_samples;
    }
    const float direction_norm = norm(direction);
    if (direction_samples == 0 || direction_norm == 0.0f) throw std::runtime_error("could not estimate direction");
    for (float & value : direction) value /= direction_norm;

    const step_capture & probe = steps[probe_position];
    llama_interp::activation_set native, zero_memory;
    auto native_run = evaluate_step(runtime, probe.before, probe.token, taps, native);
    runtime.run();
    native_run.rethrow_if_failed();

    llama_interp::rwkv_state zero_state = probe.before;
    if ((size_t) target_layer >= zero_state.layers.size() || zero_state.layers[target_layer].s.empty()) {
        throw std::runtime_error("missing layer-local WKV state");
    }
    const float native_s_norm = norm(zero_state.layers[target_layer].s);
    std::fill(zero_state.layers[target_layer].s.begin(), zero_state.layers[target_layer].s.end(), 0.0f);
    auto zero_run = evaluate_step(runtime, zero_state, probe.token, taps, zero_memory);
    runtime.run();
    zero_run.rethrow_if_failed();

    struct tap_result {
        std::string name;
        float native_norm;
        float zero_memory_norm;
        float delta_norm;
        float native_zero_cosine;
    };
    std::vector<tap_result> results;
    for (const std::string & tap : taps) {
        const std::vector<float> & native_value = capture(native, tap).data_f32;
        const std::vector<float> & zero_value = capture(zero_memory, tap).data_f32;
        results.push_back({ tap, norm(native_value), norm(zero_value), norm(subtract(native_value, zero_value)), cosine(native_value, zero_value) });
    }

    const std::vector<float> & native_time_out = capture(native, time_out).data_f32;
    const std::vector<float> & zero_time_out = capture(zero_memory, time_out).data_f32;
    const std::vector<float> & native_resid_time = capture(native, resid_time).data_f32;
    const std::vector<float> & zero_resid_time = capture(zero_memory, resid_time).data_f32;
    const std::vector<float> native_tangent = tangent_projection(native_time_out, native_resid_time, direction);
    const std::vector<float> zero_tangent = tangent_projection(zero_time_out, zero_resid_time, direction);

    std::ofstream output(output_path);
    if (!output) throw std::runtime_error("failed to open output: " + output_path);
    output << std::setprecision(9) << "{\n"
           << "  \"experiment\": \"rwkv_tmix_layer_local_wkv_state_necessity\",\n"
           << "  \"counterfactual\": \"Set only the target layer s state, passed to ggml_rwkv_wkv7 as wkv_state, to zero. Keep every r state, every other s state, token, and model weight native.\",\n"
           << "  \"limitation\": \"The zero s state is off-manifold. This measures state-dependence relative to that baseline, not semantic memory content or a decomposition valid under arbitrary baselines.\",\n"
           << "  \"model\": ";
    rwkv_experiment::write_json_string(output, model_path);
    output << ",\n  \"prompt\": ";
    rwkv_experiment::write_json_string(output, prompt);
    output << ",\n  \"target_layer\": " << target_layer
           << ",\n  \"probe_position\": " << probe_position
           << ",\n  \"probe_token\": {\"id\": " << probe.token << ", \"text\": ";
    rwkv_experiment::write_json_string(output, common_token_to_piece(ctx, probe.token, true));
    output << "},\n  \"target_layer_native_s_norm\": " << native_s_norm
           << ",\n  \"direction_samples_excluding_probe\": " << direction_samples
           << ",\n  \"tap_results\": [";
    for (size_t i = 0; i < results.size(); ++i) {
        const tap_result & value = results[i];
        if (i) output << ',';
        output << "\n    {\"tap\": ";
        rwkv_experiment::write_json_string(output, value.name);
        output << ", \"native_norm\": " << value.native_norm
               << ", \"zero_memory_norm\": " << value.zero_memory_norm
               << ", \"native_minus_zero_norm\": " << value.delta_norm
               << ", \"native_zero_cosine\": " << value.native_zero_cosine << '}';
    }
    output << "\n  ],\n  \"shared_direction\": {\"native_time_out_projection\": " << dot(native_time_out, direction)
           << ", \"zero_memory_time_out_projection\": " << dot(zero_time_out, direction)
           << ", \"native_tangent_norm\": " << norm(native_tangent)
           << ", \"zero_memory_tangent_norm\": " << norm(zero_tangent)
           << ", \"native_zero_tangent_cosine\": " << cosine(native_tangent, zero_tangent)
           << ", \"native_minus_zero_tangent_norm\": " << norm(subtract(native_tangent, zero_tangent)) << "}\n}\n";
    if (!output) throw std::runtime_error("failed to write output: " + output_path);
    std::printf("wrote=%s probe_position=%d direction_samples=%d\n", output_path.c_str(), probe_position, direction_samples);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
