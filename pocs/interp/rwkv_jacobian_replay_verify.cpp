#include "rwkv_jacobian_lens.hpp"

#include "llama-model.h"

#include <algorithm>
#include <cmath>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void quiet_llama_logs(ggml_log_level, const char *, void *) {}

struct branch_result {
    llama_interp::activation_set source_capture;
    std::vector<llama_interp::activation_set> target_captures;
    llama_interp::rwkv_state final_state;
};

struct scale_result {
    float multiplier;
    float epsilon;
    branch_result plus;
    branch_result minus;
};

float max_abs_difference(const std::vector<float> & left, const std::vector<float> & right) {
    if (left.size() != right.size()) throw std::runtime_error("vector size mismatch");
    float result = 0.0f;
    for (size_t i = 0; i < left.size(); ++i) result = std::max(result, std::abs(left[i] - right[i]));
    return result;
}

float l2_norm(const std::vector<float> & values) {
    double sum = 0.0;
    for (const float value : values) sum += (double) value * value;
    return (float) std::sqrt(sum);
}

float cosine(const std::vector<float> & left, const std::vector<float> & right) {
    if (left.size() != right.size()) throw std::runtime_error("vector size mismatch");
    double dot = 0.0;
    for (size_t i = 0; i < left.size(); ++i) dot += (double) left[i] * right[i];
    const float denom = l2_norm(left) * l2_norm(right);
    return denom == 0.0f ? 0.0f : (float) (dot / denom);
}

float agreement_cosine(const std::vector<float> & left, const std::vector<float> & right) {
    const float left_norm = l2_norm(left);
    const float right_norm = l2_norm(right);
    if (left_norm == 0.0f && right_norm == 0.0f) return 1.0f;
    if (left_norm == 0.0f || right_norm == 0.0f) return 0.0f;
    return cosine(left, right);
}

float safe_ratio(float numerator, float denominator) {
    if (denominator != 0.0f) return numerator / denominator;
    return numerator == 0.0f ? 0.0f : std::numeric_limits<float>::max();
}

float state_max_abs_difference(const llama_interp::rwkv_state & left, const llama_interp::rwkv_state & right) {
    if (left.pos != right.pos || left.n_layer != right.n_layer || left.layers.size() != right.layers.size()) {
        return INFINITY;
    }
    float result = 0.0f;
    for (size_t layer = 0; layer < left.layers.size(); ++layer) {
        result = std::max(result, max_abs_difference(left.layers[layer].r, right.layers[layer].r));
        result = std::max(result, max_abs_difference(left.layers[layer].s, right.layers[layer].s));
    }
    return result;
}

const llama_interp_activation & capture(const llama_interp::activation_set & captures, const std::string & name) {
    return rwkv_experiment::require_capture(captures, name);
}

llama_interp::task<> run_branch(
        llama_interp::runtime & runtime,
        const llama_interp::rwkv_state & before_source,
        llama_token source_token,
        const std::vector<llama_token> & suffix,
        const rwkv_jacobian_lens::lens_spec & spec,
        const std::vector<ggml_fp16_t> * perturbation,
        branch_result & output) {
    auto source = runtime.prefill_tokens(before_source, { source_token });
    source.capture_f32(rwkv_jacobian_lens::exact_regex(spec.source_tap), output.source_capture);
    if (std::find(spec.offsets.begin(), spec.offsets.end(), 0) != spec.offsets.end()) {
        source.capture_f32(rwkv_jacobian_lens::exact_regex(spec.target_tap), output.target_captures[0]);
    }
    if (perturbation) {
        source.perturb(rwkv_jacobian_lens::exact_regex(spec.source_tap),
            llama_interp::runtime::add_head(-1, *perturbation));
    }
    llama_interp::rwkv_state state = co_await source;

    for (size_t offset = 1; offset <= suffix.size(); ++offset) {
        auto step = runtime.prefill_tokens(state, { suffix[offset - 1] });
        if (std::find(spec.offsets.begin(), spec.offsets.end(), (int32_t) offset) != spec.offsets.end()) {
            step.capture_f32(rwkv_jacobian_lens::exact_regex(spec.target_tap), output.target_captures[offset]);
        }
        state = co_await step;
    }
    output.final_state = std::move(state);
}

std::vector<ggml_fp16_t> fp16_perturbation(const std::vector<float> & direction, float epsilon) {
    std::vector<ggml_fp16_t> result(direction.size());
    for (size_t i = 0; i < direction.size(); ++i) result[i] = ggml_fp32_to_fp16(epsilon * direction[i]);
    return result;
}

void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s -m MODEL -p PROMPT --source-tap FULL_NAME [--target-tap FULL_NAME]\n"
        "       [--source-position N] [--offset N ...] [--seed N] [--relative-epsilon E]\n"
        "       [--epsilon-scale S ...]\n"
        "       [--min-scale-cosine C] [--max-scale-norm-spread R]\n"
        "       [--max-center-error-ratio R] [--min-repeat-cosine C]\n"
        "       --output FILE [--tolerance E] [-ngl N]\n"
        "\n"
        "Validates a pluggable named-source, strict-future RWKV finite-difference response.\n"
        "Offsets index teacher-forced input tokens after the source token; offset zero is\n"
        "the source token's target activation. Captures are production graph snapshots.\n",
        argv0);
}

} // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    llama_log_set(quiet_llama_logs, nullptr);

    std::string model_path, prompt, output_path;
    rwkv_jacobian_lens::lens_spec spec;
    int32_t source_position = -1;
    std::vector<float> epsilon_scales;
    int n_gpu_layers = 0;
    float tolerance = 1e-4f;
    float min_scale_cosine = 0.995f;
    float max_scale_norm_spread = 0.05f;
    float max_center_error_ratio = 0.05f;
    float min_repeat_cosine = 0.999f;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (std::strcmp(argv[i], "-p") == 0 && i + 1 < argc) prompt = argv[++i];
        else if (std::strcmp(argv[i], "--source-tap") == 0 && i + 1 < argc) spec.source_tap = argv[++i];
        else if (std::strcmp(argv[i], "--target-tap") == 0 && i + 1 < argc) spec.target_tap = argv[++i];
        else if (std::strcmp(argv[i], "--source-position") == 0 && i + 1 < argc) source_position = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--offset") == 0 && i + 1 < argc) spec.offsets.push_back(std::atoi(argv[++i]));
        else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) spec.direction_seed = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "--relative-epsilon") == 0 && i + 1 < argc) spec.relative_epsilon = std::strtof(argv[++i], nullptr);
        else if (std::strcmp(argv[i], "--epsilon-scale") == 0 && i + 1 < argc) epsilon_scales.push_back(std::strtof(argv[++i], nullptr));
        else if (std::strcmp(argv[i], "--min-scale-cosine") == 0 && i + 1 < argc) min_scale_cosine = std::strtof(argv[++i], nullptr);
        else if (std::strcmp(argv[i], "--max-scale-norm-spread") == 0 && i + 1 < argc) max_scale_norm_spread = std::strtof(argv[++i], nullptr);
        else if (std::strcmp(argv[i], "--max-center-error-ratio") == 0 && i + 1 < argc) max_center_error_ratio = std::strtof(argv[++i], nullptr);
        else if (std::strcmp(argv[i], "--min-repeat-cosine") == 0 && i + 1 < argc) min_repeat_cosine = std::strtof(argv[++i], nullptr);
        else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) output_path = argv[++i];
        else if (std::strcmp(argv[i], "--tolerance") == 0 && i + 1 < argc) tolerance = std::strtof(argv[++i], nullptr);
        else if (std::strcmp(argv[i], "-ngl") == 0 && i + 1 < argc) n_gpu_layers = std::atoi(argv[++i]);
        else { usage(argv[0]); return 1; }
    }
    if (model_path.empty() || prompt.empty() || spec.source_tap.empty() || output_path.empty() || tolerance <= 0.0f) {
        usage(argv[0]);
        return 1;
    }

    ggml_backend_load_all();
    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model || model->arch != LLM_ARCH_RWKV7) throw std::runtime_error("an RWKV-7 model is required");
    if (spec.target_tap.empty()) spec.target_tap = "rwkv.layer." + std::to_string(model->hparams.n_layer() - 1) + ".resid.out";
    if (spec.offsets.empty()) spec.offsets = { 0, 1, 2, 4 };
    rwkv_jacobian_lens::normalize_offsets(spec);
    rwkv_jacobian_lens::validate(spec);
    if (epsilon_scales.empty()) epsilon_scales = { 0.5f, 1.0f, 2.0f };
    if (std::find(epsilon_scales.begin(), epsilon_scales.end(), 1.0f) == epsilon_scales.end()) epsilon_scales.push_back(1.0f);
    std::sort(epsilon_scales.begin(), epsilon_scales.end());
    epsilon_scales.erase(std::unique(epsilon_scales.begin(), epsilon_scales.end()), epsilon_scales.end());
    for (const float scale : epsilon_scales) {
        if (scale <= 0.0f) throw std::runtime_error("epsilon scales must be positive");
    }
    if (min_scale_cosine < -1.0f || min_scale_cosine > 1.0f || min_repeat_cosine < -1.0f || min_repeat_cosine > 1.0f ||
        max_scale_norm_spread < 0.0f || max_center_error_ratio < 0.0f) {
        throw std::runtime_error("invalid linearity gate threshold");
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const std::vector<llama_token> tokens = common_tokenize(vocab, prompt, false, true);
    const int32_t max_offset = spec.offsets.back();
    if (source_position < 0) source_position = (int32_t) tokens.size() - 1 - max_offset;
    if (source_position < 0 || source_position >= (int32_t) tokens.size() || source_position + max_offset >= (int32_t) tokens.size()) {
        throw std::runtime_error("source position needs all requested teacher-forced future offsets in the prompt");
    }
    const std::vector<llama_token> prefix(tokens.begin(), tokens.begin() + source_position);
    const llama_token source_token = tokens[source_position];
    const std::vector<llama_token> suffix(tokens.begin() + source_position + 1, tokens.begin() + source_position + 1 + max_offset);

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = std::max(128, (int) tokens.size() + 8);
    context_params.n_batch = context_params.n_ctx;
    context_params.n_ubatch = context_params.n_ctx;
    llama_context * ctx = llama_init_from_model(model, context_params);
    if (!ctx) throw std::runtime_error("failed to create context");
    llama_interp::runtime runtime(ctx, 1);
    llama_interp::rwkv_state before_source = runtime.make_state();
    // Drive prefix separately so the saved state is precisely before the perturbed source token.
    if (!prefix.empty()) {
        llama_interp::rwkv_state initial = runtime.make_state();
        auto prefix_run = [&]() -> llama_interp::task<> {
            before_source = co_await runtime.prefill_tokens(initial, prefix);
        }();
        runtime.run();
        prefix_run.rethrow_if_failed();
    }

    branch_result native, zero;
    native.target_captures.resize(max_offset + 1);
    zero.target_captures.resize(max_offset + 1);
    auto native_run = run_branch(runtime, before_source, source_token, suffix, spec, nullptr, native);
    runtime.run();
    native_run.rethrow_if_failed();

    const std::vector<float> native_source = capture(native.source_capture, spec.source_tap).data_f32;
    const float source_norm = l2_norm(native_source);
    if (source_norm == 0.0f) throw std::runtime_error("source activation has zero L2 norm");
    std::mt19937_64 generator(spec.direction_seed);
    std::uniform_int_distribution<int> sign(0, 1);
    std::vector<float> direction(native_source.size());
    const float scale = 1.0f / std::sqrt((float) direction.size());
    for (float & value : direction) value = sign(generator) ? scale : -scale;
    const std::vector<ggml_fp16_t> zero_perturbation(direction.size(), ggml_fp32_to_fp16(0.0f));

    auto zero_run = run_branch(runtime, before_source, source_token, suffix, spec, &zero_perturbation, zero);
    runtime.run();
    zero_run.rethrow_if_failed();
    std::vector<scale_result> scale_results;
    for (const float multiplier : epsilon_scales) {
        const float epsilon = multiplier * spec.relative_epsilon * source_norm;
        scale_result result { multiplier, epsilon };
        result.plus.target_captures.resize(max_offset + 1);
        result.minus.target_captures.resize(max_offset + 1);
        const std::vector<ggml_fp16_t> plus_perturbation = fp16_perturbation(direction, epsilon);
        const std::vector<ggml_fp16_t> minus_perturbation = fp16_perturbation(direction, -epsilon);
        auto plus_run = run_branch(runtime, before_source, source_token, suffix, spec, &plus_perturbation, result.plus);
        runtime.run();
        plus_run.rethrow_if_failed();
        auto minus_run = run_branch(runtime, before_source, source_token, suffix, spec, &minus_perturbation, result.minus);
        runtime.run();
        minus_run.rethrow_if_failed();
        scale_results.push_back(std::move(result));
    }
    const size_t base_scale_index = (size_t) std::distance(epsilon_scales.begin(), std::find(epsilon_scales.begin(), epsilon_scales.end(), 1.0f));
    scale_result repeat_result { 1.0f, spec.relative_epsilon * source_norm };
    repeat_result.plus.target_captures.resize(max_offset + 1);
    repeat_result.minus.target_captures.resize(max_offset + 1);
    const std::vector<ggml_fp16_t> repeat_plus = fp16_perturbation(direction, repeat_result.epsilon);
    const std::vector<ggml_fp16_t> repeat_minus = fp16_perturbation(direction, -repeat_result.epsilon);
    auto repeat_plus_run = run_branch(runtime, before_source, source_token, suffix, spec, &repeat_plus, repeat_result.plus);
    runtime.run();
    repeat_plus_run.rethrow_if_failed();
    auto repeat_minus_run = run_branch(runtime, before_source, source_token, suffix, spec, &repeat_minus, repeat_result.minus);
    runtime.run();
    repeat_minus_run.rethrow_if_failed();

    const float zero_state_error = state_max_abs_difference(native.final_state, zero.final_state);
    std::ofstream output(output_path);
    if (!output) throw std::runtime_error("failed to open output: " + output_path);
    output << std::setprecision(9) << "{\n  \"experiment\": \"rwkv_jacobian_replay_verify\",\n  \"lens_spec\": ";
    rwkv_jacobian_lens::write_json(output, spec);
    output << ",\n  \"model\": ";
    rwkv_experiment::write_json_string(output, model_path);
    output << ",\n  \"source_position\": " << source_position
           << ",\n  \"source_token\": {\"id\": " << source_token << ", \"text\": ";
    rwkv_experiment::write_json_string(output, common_token_to_piece(ctx, source_token, true));
    output << "},\n  \"source_dimension\": " << direction.size()
           << ",\n  \"source_l2_norm\": " << source_norm
           << ",\n  \"zero_branch_final_state_max_abs_error\": " << zero_state_error
           << ",\n  \"epsilon_scale_results\": [";
    float max_zero_target_error = 0.0f;
    float max_identity_error = 0.0f;
    bool linearity_gate_passed = true;
    for (size_t scale_index = 0; scale_index < scale_results.size(); ++scale_index) {
        const scale_result & scale_result = scale_results[scale_index];
        const auto applied_norm = [&](const branch_result & branch) {
            const std::vector<float> & value = capture(branch.source_capture, spec.source_tap).data_f32;
            std::vector<float> difference(value.size());
            for (size_t i = 0; i < value.size(); ++i) difference[i] = value[i] - native_source[i];
            return l2_norm(difference);
        };
        if (scale_index) output << ',';
        output << "\n    {\"multiplier\": " << scale_result.multiplier
               << ", \"epsilon\": " << scale_result.epsilon
               << ", \"plus_applied_l2_norm\": " << applied_norm(scale_result.plus)
               << ", \"minus_applied_l2_norm\": " << applied_norm(scale_result.minus)
               << ", \"offsets\": [";
        for (size_t i = 0; i < spec.offsets.size(); ++i) {
            const int32_t offset = spec.offsets[i];
            const std::vector<float> & native_target = capture(native.target_captures[offset], spec.target_tap).data_f32;
            const std::vector<float> & zero_target = capture(zero.target_captures[offset], spec.target_tap).data_f32;
            const std::vector<float> & plus_target = capture(scale_result.plus.target_captures[offset], spec.target_tap).data_f32;
            const std::vector<float> & minus_target = capture(scale_result.minus.target_captures[offset], spec.target_tap).data_f32;
            const float zero_target_error = max_abs_difference(native_target, zero_target);
            max_zero_target_error = std::max(max_zero_target_error, zero_target_error);
            std::vector<float> response(native_target.size());
            std::vector<float> center_error(native_target.size());
            std::vector<float> odd_displacement(native_target.size());
            for (size_t j = 0; j < response.size(); ++j) {
                response[j] = (plus_target[j] - minus_target[j]) / (2.0f * scale_result.epsilon);
                center_error[j] = 0.5f * (plus_target[j] + minus_target[j]) - native_target[j];
                odd_displacement[j] = 0.5f * (plus_target[j] - minus_target[j]);
            }
            if (i) output << ',';
            output << "\n      {\"offset\": " << offset
                   << ", \"target_dimension\": " << native_target.size()
                   << ", \"zero_branch_target_max_abs_error\": " << zero_target_error
                   << ", \"central_difference_response_l2_norm\": " << l2_norm(response)
                   << ", \"central_difference_center_error_l2_norm\": " << l2_norm(center_error)
                   << ", \"center_error_ratio\": " << safe_ratio(l2_norm(center_error), l2_norm(odd_displacement));
            if (offset == 0 && spec.source_tap == spec.target_tap && native_target.size() == direction.size()) {
                const float identity_error = max_abs_difference(response, direction);
                max_identity_error = std::max(max_identity_error, identity_error);
                output << ", \"identity_control_direction_cosine\": " << cosine(response, direction)
                       << ", \"identity_control_max_abs_error\": " << identity_error;
            }
            output << '}';
        }
        output << "\n    ]}";
    }
    output << "\n  ],\n  \"linearity_gate\": {\n    \"thresholds\": {\"min_scale_cosine\": " << min_scale_cosine
           << ", \"max_scale_norm_spread\": " << max_scale_norm_spread
           << ", \"max_center_error_ratio\": " << max_center_error_ratio
           << ", \"min_repeat_cosine\": " << min_repeat_cosine
           << "},\n    \"offsets\": [";
    for (size_t offset_index = 0; offset_index < spec.offsets.size(); ++offset_index) {
        const int32_t offset = spec.offsets[offset_index];
        std::vector<std::vector<float>> responses;
        std::vector<float> response_norms;
        float max_center_ratio = 0.0f;
        for (const scale_result & scale_result : scale_results) {
            const std::vector<float> & native_target = capture(native.target_captures[offset], spec.target_tap).data_f32;
            const std::vector<float> & plus_target = capture(scale_result.plus.target_captures[offset], spec.target_tap).data_f32;
            const std::vector<float> & minus_target = capture(scale_result.minus.target_captures[offset], spec.target_tap).data_f32;
            std::vector<float> response(native_target.size());
            std::vector<float> center_error(native_target.size());
            std::vector<float> odd_displacement(native_target.size());
            for (size_t j = 0; j < response.size(); ++j) {
                response[j] = (plus_target[j] - minus_target[j]) / (2.0f * scale_result.epsilon);
                center_error[j] = 0.5f * (plus_target[j] + minus_target[j]) - native_target[j];
                odd_displacement[j] = 0.5f * (plus_target[j] - minus_target[j]);
            }
            const float odd_norm = l2_norm(odd_displacement);
            const float center_norm = l2_norm(center_error);
            max_center_ratio = std::max(max_center_ratio, safe_ratio(center_norm, odd_norm));
            response_norms.push_back(l2_norm(response));
            responses.push_back(std::move(response));
        }
        const std::vector<float> & native_target = capture(native.target_captures[offset], spec.target_tap).data_f32;
        const std::vector<float> & repeat_plus_target = capture(repeat_result.plus.target_captures[offset], spec.target_tap).data_f32;
        const std::vector<float> & repeat_minus_target = capture(repeat_result.minus.target_captures[offset], spec.target_tap).data_f32;
        std::vector<float> repeat_response(native_target.size());
        for (size_t j = 0; j < repeat_response.size(); ++j) {
            repeat_response[j] = (repeat_plus_target[j] - repeat_minus_target[j]) / (2.0f * repeat_result.epsilon);
        }
        float min_pairwise_cosine = 1.0f;
        for (size_t i = 0; i < responses.size(); ++i) {
            for (size_t j = i + 1; j < responses.size(); ++j) {
                min_pairwise_cosine = std::min(min_pairwise_cosine, agreement_cosine(responses[i], responses[j]));
            }
        }
        const auto [min_norm, max_norm] = std::minmax_element(response_norms.begin(), response_norms.end());
        const float norm_spread = *max_norm == 0.0f ? 0.0f : (*max_norm - *min_norm) / *max_norm;
        const float repeat_cosine = agreement_cosine(responses[base_scale_index], repeat_response);
        const bool passed = min_pairwise_cosine >= min_scale_cosine && norm_spread <= max_scale_norm_spread &&
                            max_center_ratio <= max_center_error_ratio && repeat_cosine >= min_repeat_cosine;
        linearity_gate_passed = linearity_gate_passed && passed;
        if (offset_index) output << ',';
        output << "\n      {\"offset\": " << offset
               << ", \"min_pairwise_scale_cosine\": " << min_pairwise_cosine
               << ", \"scale_response_norm_relative_spread\": " << norm_spread
               << ", \"max_center_error_ratio\": " << max_center_ratio
               << ", \"repeat_1x_cosine\": " << repeat_cosine
               << ", \"passed\": " << (passed ? "true" : "false") << '}';
    }
    output << "\n    ],\n    \"passed\": " << (linearity_gate_passed ? "true" : "false") << "\n  }\n}\n";
    if (!output) throw std::runtime_error("failed to write output: " + output_path);
    output.close();

    if (zero_state_error > tolerance || max_zero_target_error > tolerance) {
        std::fprintf(stderr, "verification failed: zero perturbation changed the target activation or exported recurrent state\n");
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 2;
    }
    if (spec.source_tap == spec.target_tap &&
        std::find(spec.offsets.begin(), spec.offsets.end(), 0) != spec.offsets.end() &&
        max_identity_error > tolerance) {
        std::fprintf(stderr, "verification failed: same-tap offset-zero identity control failed\n");
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 2;
    }
    if (!linearity_gate_passed) {
        std::fprintf(stderr, "linearity gate failed; do not collect corpus Jacobian responses\n");
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 2;
    }
    std::printf("verified source=%s target=%s position=%d epsilon_scales=%zu output=%s\n",
        spec.source_tap.c_str(), spec.target_tap.c_str(), source_position, scale_results.size(), output_path.c_str());
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
