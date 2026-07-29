#include "rwkv_experiment.hpp"

#include "llama-model.h"

#include <algorithm>
#include <array>
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

float dot(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) throw std::runtime_error("vector size mismatch");
    double value = 0.0;
    for (size_t i = 0; i < a.size(); ++i) value += (double) a[i] * b[i];
    return (float) value;
}

float norm(const std::vector<float> & value) { return std::sqrt(dot(value, value)); }

float cosine(const std::vector<float> & a, const std::vector<float> & b) {
    const float denominator = norm(a) * norm(b);
    return denominator == 0.0f ? 0.0f : dot(a, b) / denominator;
}

std::vector<float> difference(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) throw std::runtime_error("vector size mismatch");
    std::vector<float> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) result[i] = a[i] - b[i];
    return result;
}

const llama_interp_activation & capture(const llama_interp::activation_set & captures, const std::string & name) {
    return rwkv_experiment::require_capture(captures, name);
}

struct decomposition {
    std::vector<float> mean;
    std::vector<float> radial;
    std::vector<float> tangent;
    float input_centered_norm;
    float output_centered_norm;
    float rotation_radians;
};

decomposition decompose_time_write(const std::vector<float> & resid_in, const std::vector<float> & time_out) {
    if (resid_in.size() != time_out.size()) throw std::runtime_error("time write dimension differs from residual");
    const size_t width = resid_in.size();
    float input_mean = 0.0f, write_mean = 0.0f;
    for (size_t i = 0; i < width; ++i) {
        input_mean += resid_in[i] / width;
        write_mean += time_out[i] / width;
    }
    std::vector<float> centered_input(width), centered_write(width), mean(width, write_mean);
    for (size_t i = 0; i < width; ++i) {
        centered_input[i] = resid_in[i] - input_mean;
        centered_write[i] = time_out[i] - write_mean;
    }
    const float input_norm = norm(centered_input);
    if (input_norm == 0.0f) throw std::runtime_error("constant residual input");
    const float radial_scale = dot(centered_write, centered_input) / input_norm;
    std::vector<float> radial(width), tangent(width), centered_output(width);
    for (size_t i = 0; i < width; ++i) {
        radial[i] = radial_scale * centered_input[i] / input_norm;
        tangent[i] = centered_write[i] - radial[i];
        centered_output[i] = centered_input[i] + centered_write[i];
    }
    const float output_norm = norm(centered_output);
    const float angle_cosine = std::clamp(dot(centered_input, centered_output) / (input_norm * output_norm), -1.0f, 1.0f);
    return { std::move(mean), std::move(radial), std::move(tangent), input_norm, output_norm, std::acos(angle_cosine) };
}

llama_interp::task<> run_perturbation(
        llama_interp::runtime & runtime,
        const llama_interp::rwkv_state & before,
        llama_token token,
        const std::string & time_out_tap,
        const std::vector<float> & removed_component,
        const std::vector<std::string> & taps,
        llama_interp::activation_set & output) {
    std::vector<ggml_fp16_t> perturbation(removed_component.size());
    for (size_t i = 0; i < perturbation.size(); ++i) perturbation[i] = ggml_fp32_to_fp16(-removed_component[i]);
    auto step = runtime.prefill_tokens(before, { token });
    step.perturb("^" + time_out_tap + "$", llama_interp::runtime::add_head(-1, std::move(perturbation)));
    for (const std::string & tap : taps) step.capture_f32("^" + tap + "$", output);
    step.discard_state();
    (void) co_await step;
}

void usage(const char * argv0) {
    std::fprintf(stderr, "usage: %s -m MODEL -p PROMPT --output FILE [--probe-position N] [-ngl N]\n", argv0);
}

} // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    llama_log_set(quiet_llama_logs, nullptr);
    std::string model_path, prompt, output_path;
    int32_t probe_position = -1;
    int n_gpu_layers = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (std::strcmp(argv[i], "-p") == 0 && i + 1 < argc) prompt = argv[++i];
        else if (std::strcmp(argv[i], "--probe-position") == 0 && i + 1 < argc) probe_position = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) output_path = argv[++i];
        else if (std::strcmp(argv[i], "-ngl") == 0 && i + 1 < argc) n_gpu_layers = std::atoi(argv[++i]);
        else { usage(argv[0]); return 1; }
    }
    if (model_path.empty() || prompt.empty() || output_path.empty()) { usage(argv[0]); return 1; }

    ggml_backend_load_all();
    llama_backend_init();
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = n_gpu_layers;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), params);
    if (!model || model->arch != LLM_ARCH_RWKV7) throw std::runtime_error("an RWKV-7 model is required");
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const std::vector<llama_token> tokens = common_tokenize(vocab, prompt, false, true);
    if (tokens.empty()) throw std::runtime_error("prompt tokenized to zero tokens");
    if (probe_position < 0) probe_position = (int32_t) tokens.size() - 1;
    if (probe_position < 0 || probe_position >= (int32_t) tokens.size()) throw std::runtime_error("invalid probe position");
    const int32_t n_layer = model->hparams.n_layer();

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = std::max(128, (int) tokens.size() + 8);
    context_params.n_batch = context_params.n_ctx;
    context_params.n_ubatch = context_params.n_ctx;
    llama_context * ctx = llama_init_from_model(model, context_params);
    if (!ctx) throw std::runtime_error("failed to create context");
    llama_interp::runtime runtime(ctx, 1);
    llama_interp::rwkv_state before = runtime.make_state();
    if (probe_position > 0) {
        const std::vector<llama_token> prefix(tokens.begin(), tokens.begin() + probe_position);
        auto prefix_task = [&]() -> llama_interp::task<> { before = co_await runtime.prefill_tokens(before, prefix); }();
        runtime.run();
        prefix_task.rethrow_if_failed();
    }

    llama_interp::activation_set native;
    auto native_step = runtime.prefill_tokens(before, { tokens[probe_position] });
    for (int32_t layer = 0; layer < n_layer; ++layer) {
        const std::string prefix = "rwkv.layer." + std::to_string(layer) + ".";
        for (const std::string & suffix : { "resid.in", "time.out", "resid.time", "ffn.norm", "channel.out" }) {
            native_step.capture_f32("^" + prefix + suffix + "$", native);
        }
    }
    native_step.discard_state();
    auto native_task = [&]() -> llama_interp::task<> { (void) co_await native_step; }();
    runtime.run();
    native_task.rethrow_if_failed();

    std::ofstream json(output_path);
    if (!json) throw std::runtime_error("failed to open output");
    json << std::setprecision(9) << "{\n  \"experiment\": \"rwkv_tmix_residual_rotation\",\n"
         << "  \"definition\": \"The native time.out write is split into mean, radial, and tangent components relative to centered resid.in. Each component is removed separately at time.out; ffn.norm and channel.out are then recaptured from the production graph.\",\n  \"model\": ";
    rwkv_experiment::write_json_string(json, model_path);
    json << ",\n  \"prompt\": "; rwkv_experiment::write_json_string(json, prompt);
    json << ",\n  \"probe_position\": " << probe_position << ",\n  \"probe_token\": {\"id\": " << tokens[probe_position] << ", \"text\": ";
    rwkv_experiment::write_json_string(json, common_token_to_piece(ctx, tokens[probe_position], true));
    json << "},\n  \"layers\": [";
    const std::array<const char *, 3> names = {{ "mean", "radial", "tangent" }};
    for (int32_t layer = 0; layer < n_layer; ++layer) {
        const std::string prefix = "rwkv.layer." + std::to_string(layer) + ".";
        const std::string time_out = prefix + "time.out";
        const std::string resid_in = prefix + "resid.in";
        const std::string resid_time = prefix + "resid.time";
        const std::string ffn_norm = prefix + "ffn.norm";
        const std::string channel_out = prefix + "channel.out";
        const decomposition parts = decompose_time_write(capture(native, resid_in).data_f32, capture(native, time_out).data_f32);
        const std::array<const std::vector<float> *, 3> components = {{ &parts.mean, &parts.radial, &parts.tangent }};
        if (layer) json << ',';
        json << "\n    {\"layer\": " << layer
             << ", \"native\": {\"time_out_norm\": " << norm(capture(native, time_out).data_f32)
             << ", \"resid_in_centered_norm\": " << parts.input_centered_norm
             << ", \"resid_time_centered_norm\": " << parts.output_centered_norm
             << ", \"rotation_radians\": " << parts.rotation_radians
             << ", \"rotation_degrees\": " << parts.rotation_radians * 180.0f / (float) M_PI
             << ", \"mean_norm\": " << norm(parts.mean) << ", \"radial_norm\": " << norm(parts.radial) << ", \"tangent_norm\": " << norm(parts.tangent) << "}, \"remove_component\": [";
        for (size_t component_index = 0; component_index < components.size(); ++component_index) {
            llama_interp::activation_set ablated;
            const std::vector<std::string> taps = { time_out, resid_time, ffn_norm, channel_out };
            auto task = run_perturbation(runtime, before, tokens[probe_position], time_out, *components[component_index], taps, ablated);
            runtime.run();
            task.rethrow_if_failed();
            const std::vector<float> ffn_delta = difference(capture(ablated, ffn_norm).data_f32, capture(native, ffn_norm).data_f32);
            const std::vector<float> channel_delta = difference(capture(ablated, channel_out).data_f32, capture(native, channel_out).data_f32);
            const std::vector<float> applied = difference(capture(native, time_out).data_f32, capture(ablated, time_out).data_f32);
            if (component_index) json << ',';
            json << "{\"component\": \"" << names[component_index] << "\", \"component_norm\": " << norm(*components[component_index])
                 << ", \"applied_time_out_norm\": " << norm(applied)
                 << ", \"ffn_norm_delta_norm\": " << norm(ffn_delta)
                 << ", \"ffn_norm_native_cosine\": " << cosine(capture(ablated, ffn_norm).data_f32, capture(native, ffn_norm).data_f32)
                 << ", \"channel_out_delta_norm\": " << norm(channel_delta)
                 << ", \"channel_out_native_cosine\": " << cosine(capture(ablated, channel_out).data_f32, capture(native, channel_out).data_f32) << '}';
        }
        json << "]}";
        std::fprintf(stderr, "completed_layer=%d/%d\n", layer + 1, n_layer);
    }
    json << "\n  ]\n}\n";
    if (!json) throw std::runtime_error("failed to write output");
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
