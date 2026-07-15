#include "rwkv_experiment.hpp"

#include "llama-model.h"

#include <array>
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
        const std::string & capture_tap,
        llama_interp::activation_set & output) {
    auto call = runtime.prefill_tokens(source.before, { source.token });
    call.perturb("^" + perturb_tap + "$", llama_interp::runtime::add_head(-1, std::move(perturbation)));
    call.capture_f32("^" + perturb_tap + "$", output);
    call.capture_f32("^" + capture_tap + "$", output);
    (void) co_await call;
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
    const std::string time_out = prefix + "time.out";
    const std::string resid_time = prefix + "resid.time";
    const std::string ffn_norm = prefix + "ffn.norm";
    const std::vector<std::string> taps = { time_out, resid_time, ffn_norm };

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
    if (direction_samples == 0 || direction_norm == 0.0f) throw std::runtime_error("could not estimate held-out direction");
    for (float & value : direction) value /= direction_norm;

    const step_capture & probe = steps[probe_position];
    const std::vector<float> native_time = capture(probe.activations, time_out).data_f32;
    const std::vector<float> native_resid = capture(probe.activations, resid_time).data_f32;
    const std::vector<float> native_ffn = capture(probe.activations, ffn_norm).data_f32;
    const float alpha = dot(native_time, direction);
    std::vector<float> full_component(width);
    for (int32_t i = 0; i < width; ++i) full_component[i] = alpha * direction[i];
    const float component_mean = std::accumulate(full_component.begin(), full_component.end(), 0.0f) / width;
    const float residual_mean = std::accumulate(native_resid.begin(), native_resid.end(), 0.0f) / width;
    std::vector<float> mean_component(width), centered_component(width), centered_residual(width);
    for (int32_t i = 0; i < width; ++i) {
        mean_component[i] = component_mean;
        centered_component[i] = full_component[i] - component_mean;
        centered_residual[i] = native_resid[i] - residual_mean;
    }
    const float centered_residual_norm = norm(centered_residual);
    if (centered_residual_norm == 0.0f) throw std::runtime_error("zero centered residual");
    for (float & value : centered_residual) value /= centered_residual_norm;
    const float radial_scale = dot(centered_component, centered_residual);
    std::vector<float> radial_component(width), tangent_component(width);
    for (int32_t i = 0; i < width; ++i) {
        radial_component[i] = radial_scale * centered_residual[i];
        tangent_component[i] = centered_component[i] - radial_component[i];
    }

    struct observation {
        const char * component;
        float scale;
        float requested_norm;
        float applied_norm;
        float ffn_delta_norm;
        std::vector<float> ffn_delta;
    };
    const std::array<std::pair<const char *, const std::vector<float> *>, 3> components = {{
        { "mean", &mean_component },
        { "radial", &radial_component },
        { "tangent", &tangent_component },
    }};
    static constexpr std::array<float, 4> scales = {{ -0.25f, -0.125f, 0.125f, 0.25f }};
    std::vector<observation> observations;
    for (const auto & [name, component] : components) {
        for (const float scale : scales) {
            std::vector<ggml_fp16_t> perturbation(width);
            for (int32_t i = 0; i < width; ++i) perturbation[i] = ggml_fp32_to_fp16(-scale * (*component)[i]);
            llama_interp::activation_set ablated;
            auto run = perturb_step(runtime, probe, time_out, std::move(perturbation), ffn_norm, ablated);
            runtime.run();
            run.rethrow_if_failed();
            const std::vector<float> applied = subtract(native_time, capture(ablated, time_out).data_f32);
            const std::vector<float> delta = subtract(capture(ablated, ffn_norm).data_f32, native_ffn);
            observations.push_back({ name, scale, std::abs(scale) * norm(*component), norm(applied), norm(delta), delta });
        }
    }

    auto observation_for = [&observations](const char * component, float scale) -> const observation & {
        for (const observation & value : observations) {
            if (std::strcmp(value.component, component) == 0 && value.scale == scale) return value;
        }
        throw std::runtime_error("missing observation");
    };

    std::ofstream output(output_path);
    if (!output) throw std::runtime_error("failed to open output: " + output_path);
    output << std::setprecision(9) << "{\n"
           << "  \"experiment\": \"rwkv_tmix_layernorm_geometry\",\n"
           << "  \"hypothesis\": \"At small signed scales, tangent time.out interventions change production ffn.norm more than radial interventions of the same component scale.\",\n"
           << "  \"model\": ";
    rwkv_experiment::write_json_string(output, model_path);
    output << ",\n  \"prompt\": ";
    rwkv_experiment::write_json_string(output, prompt);
    output << ",\n  \"target_layer\": " << target_layer
           << ",\n  \"probe_position\": " << probe_position
           << ",\n  \"probe_token\": {\"id\": " << probe.token << ", \"text\": ";
    rwkv_experiment::write_json_string(output, common_token_to_piece(ctx, probe.token, true));
    output << "},\n  \"direction_samples_excluding_probe\": " << direction_samples
           << ",\n  \"component_norms\": {\"mean\": " << norm(mean_component)
           << ", \"radial\": " << norm(radial_component)
           << ", \"tangent\": " << norm(tangent_component) << "},\n"
           << "  \"observations\": [";
    for (size_t i = 0; i < observations.size(); ++i) {
        const observation & value = observations[i];
        if (i) output << ',';
        output << "\n    {\"component\": \"" << value.component << "\", \"scale\": " << value.scale
               << ", \"requested_time_out_norm\": " << value.requested_norm
               << ", \"applied_time_out_norm\": " << value.applied_norm
               << ", \"ffn_norm_delta_norm\": " << value.ffn_delta_norm << '}';
    }
    output << "\n  ],\n  \"signed_scale_checks\": [";
    bool first_check = true;
    for (const auto & [name, component] : components) {
        (void) component;
        for (const float magnitude : { 0.125f, 0.25f }) {
            const observation & negative = observation_for(name, -magnitude);
            const observation & positive = observation_for(name, magnitude);
            if (!first_check) output << ',';
            first_check = false;
            output << "\n    {\"component\": \"" << name << "\", \"magnitude\": " << magnitude
                   << ", \"opposite_sign_cosine\": " << cosine(negative.ffn_delta, positive.ffn_delta)
                   << ", \"opposite_sign_norm_ratio\": "
                   << (positive.ffn_delta_norm == 0.0f ? 0.0f : negative.ffn_delta_norm / positive.ffn_delta_norm) << '}';
        }
    }
    output << "\n  ],\n  \"scale_linearity_checks\": [";
    first_check = true;
    for (const auto & [name, component] : components) {
        (void) component;
        for (const float sign : { -1.0f, 1.0f }) {
            const observation & small = observation_for(name, sign * 0.125f);
            const observation & large = observation_for(name, sign * 0.25f);
            std::vector<float> predicted = small.ffn_delta;
            for (float & value : predicted) value *= 2.0f;
            const std::vector<float> error = subtract(large.ffn_delta, predicted);
            if (!first_check) output << ',';
            first_check = false;
            output << "\n    {\"component\": \"" << name << "\", \"sign\": " << sign
                   << ", \"small_to_large_cosine\": " << cosine(small.ffn_delta, large.ffn_delta)
                   << ", \"double_small_relative_error\": "
                   << (large.ffn_delta_norm == 0.0f ? 0.0f : norm(error) / large.ffn_delta_norm) << '}';
        }
    }
    output << "\n  ]\n}\n";
    if (!output) throw std::runtime_error("failed to write output: " + output_path);
    std::printf("wrote=%s probe_position=%d direction_samples=%d\n", output_path.c_str(), probe_position, direction_samples);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
