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
#include <numeric>
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

std::vector<float> make_direction(const std::vector<float> & source, uint64_t seed, bool layer_norm_tangent) {
    std::mt19937_64 generator(seed);
    std::uniform_int_distribution<int> sign(0, 1);
    std::vector<float> direction(source.size());
    const float component = 1.0f / std::sqrt((float) direction.size());
    for (float & value : direction) value = sign(generator) ? component : -component;
    if (!layer_norm_tangent) return direction;

    float source_mean = 0.0f, direction_mean = 0.0f;
    for (size_t i = 0; i < source.size(); ++i) {
        source_mean += source[i] / source.size();
        direction_mean += direction[i] / direction.size();
    }
    float radial_norm_sq = 0.0f, radial_dot = 0.0f;
    for (size_t i = 0; i < source.size(); ++i) {
        direction[i] -= direction_mean;
        const float radial = source[i] - source_mean;
        radial_norm_sq += radial * radial;
        radial_dot += direction[i] * radial;
    }
    if (radial_norm_sq == 0.0f) throw std::runtime_error("cannot construct LayerNorm tangent at constant source activation");
    for (size_t i = 0; i < source.size(); ++i) direction[i] -= radial_dot / radial_norm_sq * (source[i] - source_mean);
    const float direction_norm = l2_norm(direction);
    if (direction_norm == 0.0f) throw std::runtime_error("LayerNorm tangent projection produced a zero direction");
    for (float & value : direction) value /= direction_norm;
    return direction;
}

std::vector<float> project_layer_norm_tangent(const std::vector<float> & source, std::vector<float> direction) {
    if (source.size() != direction.size()) throw std::runtime_error("rotor basis dimension differs from source");
    float source_mean = 0.0f, direction_mean = 0.0f;
    for (size_t i = 0; i < source.size(); ++i) {
        source_mean += source[i] / source.size();
        direction_mean += direction[i] / direction.size();
    }
    float radial_norm_sq = 0.0f, radial_dot = 0.0f;
    for (size_t i = 0; i < source.size(); ++i) {
        direction[i] -= direction_mean;
        const float radial = source[i] - source_mean;
        radial_norm_sq += radial * radial;
        radial_dot += direction[i] * radial;
    }
    if (radial_norm_sq == 0.0f) throw std::runtime_error("cannot construct LayerNorm tangent at constant source activation");
    for (size_t i = 0; i < source.size(); ++i) direction[i] -= radial_dot / radial_norm_sq * (source[i] - source_mean);
    const float direction_norm = l2_norm(direction);
    if (direction_norm == 0.0f) throw std::runtime_error("LayerNorm tangent projection produced a zero direction");
    for (float & value : direction) value /= direction_norm;
    return direction;
}

std::vector<ggml_fp16_t> rotor_perturbation(const std::vector<float> & source, const std::vector<float> & tangent, float theta) {
    if (source.size() != tangent.size()) throw std::runtime_error("rotor tangent dimension differs from source");
    float mean = 0.0f, radius_sq = 0.0f;
    for (const float value : source) mean += value / source.size();
    for (const float value : source) radius_sq += (value - mean) * (value - mean);
    const float radius = std::sqrt(radius_sq);
    std::vector<ggml_fp16_t> result(source.size());
    for (size_t i = 0; i < source.size(); ++i) {
        const float radial_unit = (source[i] - mean) / radius;
        const float delta = radius * ((std::cos(theta) - 1.0f) * radial_unit + std::sin(theta) * tangent[i]);
        result[i] = ggml_fp32_to_fp16(delta);
    }
    return result;
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

std::vector<int32_t> top_indices(const std::vector<float> & values, int32_t count, bool positive) {
    std::vector<int32_t> indices(values.size());
    std::iota(indices.begin(), indices.end(), 0);
    count = std::min<int32_t>(count, (int32_t) indices.size());
    std::partial_sort(indices.begin(), indices.begin() + count, indices.end(), [&values, positive](int32_t left, int32_t right) {
        return positive ? values[left] > values[right] : values[left] < values[right];
    });
    indices.resize(count);
    return indices;
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
        const std::vector<std::string> * basis_taps,
        branch_result & output) {
    auto source = runtime.prefill_tokens(before_source, { source_token });
    source.capture_f32(rwkv_jacobian_lens::exact_regex(spec.source_tap), output.source_capture);
    if (basis_taps) {
        for (const std::string & basis_tap : *basis_taps) {
            if (basis_tap != spec.source_tap) source.capture_f32(rwkv_jacobian_lens::exact_regex(basis_tap), output.source_capture);
        }
    }
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
        "       [--direction rademacher|ln-tangent|ln-rotor] [--rotor-basis-tap FULL_NAME]\n"
        "       [--rotor-basis-difference FROM_TAP TO_TAP]\n"
        "       [--epsilon-scale S ...]\n"
        "       [--min-scale-cosine C] [--max-scale-norm-spread R]\n"
        "       [--max-center-error-ratio R] [--min-repeat-cosine C]\n"
        "       [--top-logits N]\n"
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
    int32_t top_logit_count = 0;
    bool layer_norm_tangent = false;
    bool layer_norm_rotor = false;
    std::string rotor_basis_tap;
    std::string rotor_basis_from_tap, rotor_basis_to_tap;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (std::strcmp(argv[i], "-p") == 0 && i + 1 < argc) prompt = argv[++i];
        else if (std::strcmp(argv[i], "--source-tap") == 0 && i + 1 < argc) spec.source_tap = argv[++i];
        else if (std::strcmp(argv[i], "--target-tap") == 0 && i + 1 < argc) spec.target_tap = argv[++i];
        else if (std::strcmp(argv[i], "--source-position") == 0 && i + 1 < argc) source_position = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--offset") == 0 && i + 1 < argc) spec.offsets.push_back(std::atoi(argv[++i]));
        else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) spec.direction_seed = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "--direction") == 0 && i + 1 < argc) {
            const std::string value = argv[++i];
            if (value == "rademacher") {
                layer_norm_tangent = false;
                spec.direction = "seeded_unit_l2_rademacher";
            } else if (value == "ln-tangent") {
                layer_norm_tangent = true;
                spec.direction = "seeded_unit_l2_rademacher_projected_to_layernorm_input_tangent";
            } else if (value == "ln-rotor") {
                layer_norm_rotor = true;
                spec.direction = "layernorm_sphere_rotor";
            } else {
                throw std::runtime_error("--direction must be rademacher, ln-tangent, or ln-rotor");
            }
        }
        else if (std::strcmp(argv[i], "--rotor-basis-tap") == 0 && i + 1 < argc) rotor_basis_tap = argv[++i];
        else if (std::strcmp(argv[i], "--rotor-basis-difference") == 0 && i + 2 < argc) {
            rotor_basis_from_tap = argv[++i];
            rotor_basis_to_tap = argv[++i];
        }
        else if (std::strcmp(argv[i], "--relative-epsilon") == 0 && i + 1 < argc) spec.relative_epsilon = std::strtof(argv[++i], nullptr);
        else if (std::strcmp(argv[i], "--epsilon-scale") == 0 && i + 1 < argc) epsilon_scales.push_back(std::strtof(argv[++i], nullptr));
        else if (std::strcmp(argv[i], "--min-scale-cosine") == 0 && i + 1 < argc) min_scale_cosine = std::strtof(argv[++i], nullptr);
        else if (std::strcmp(argv[i], "--max-scale-norm-spread") == 0 && i + 1 < argc) max_scale_norm_spread = std::strtof(argv[++i], nullptr);
        else if (std::strcmp(argv[i], "--max-center-error-ratio") == 0 && i + 1 < argc) max_center_error_ratio = std::strtof(argv[++i], nullptr);
        else if (std::strcmp(argv[i], "--min-repeat-cosine") == 0 && i + 1 < argc) min_repeat_cosine = std::strtof(argv[++i], nullptr);
        else if (std::strcmp(argv[i], "--top-logits") == 0 && i + 1 < argc) top_logit_count = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) output_path = argv[++i];
        else if (std::strcmp(argv[i], "--tolerance") == 0 && i + 1 < argc) tolerance = std::strtof(argv[++i], nullptr);
        else if (std::strcmp(argv[i], "-ngl") == 0 && i + 1 < argc) n_gpu_layers = std::atoi(argv[++i]);
        else { usage(argv[0]); return 1; }
    }
    if (model_path.empty() || prompt.empty() || spec.source_tap.empty() || output_path.empty() || tolerance <= 0.0f) {
        usage(argv[0]);
        return 1;
    }
    if (top_logit_count < 0) throw std::runtime_error("top-logits must be non-negative");

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
    if (layer_norm_rotor && rotor_basis_tap.empty() && (rotor_basis_from_tap.empty() || rotor_basis_to_tap.empty())) {
        throw std::runtime_error("ln-rotor requires --rotor-basis-tap or --rotor-basis-difference FROM_TAP TO_TAP");
    }
    if (!rotor_basis_tap.empty() && (!rotor_basis_from_tap.empty() || !rotor_basis_to_tap.empty())) {
        throw std::runtime_error("use either --rotor-basis-tap or --rotor-basis-difference, not both");
    }
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
    std::vector<std::string> basis_taps;
    if (layer_norm_rotor) {
        if (!rotor_basis_tap.empty()) basis_taps = { rotor_basis_tap };
        else basis_taps = { rotor_basis_from_tap, rotor_basis_to_tap };
    }
    const std::vector<std::string> * captured_basis_taps = layer_norm_rotor ? &basis_taps : nullptr;
    auto native_run = run_branch(runtime, before_source, source_token, suffix, spec, nullptr, captured_basis_taps, native);
    runtime.run();
    native_run.rethrow_if_failed();

    const std::vector<float> native_source = capture(native.source_capture, spec.source_tap).data_f32;
    const float source_norm = l2_norm(native_source);
    if (source_norm == 0.0f) throw std::runtime_error("source activation has zero L2 norm");
    std::vector<float> direction;
    if (layer_norm_rotor) {
        if (!rotor_basis_tap.empty()) {
            direction = project_layer_norm_tangent(native_source, capture(native.source_capture, rotor_basis_tap).data_f32);
            spec.direction += "_basis_" + rotor_basis_tap;
        } else {
            const std::vector<float> & from = capture(native.source_capture, rotor_basis_from_tap).data_f32;
            const std::vector<float> & to = capture(native.source_capture, rotor_basis_to_tap).data_f32;
            if (from.size() != to.size()) throw std::runtime_error("rotor basis difference dimensions differ");
            std::vector<float> update(from.size());
            for (size_t i = 0; i < update.size(); ++i) update[i] = to[i] - from[i];
            direction = project_layer_norm_tangent(native_source, std::move(update));
            spec.direction += "_basis_difference_" + rotor_basis_to_tap + "_minus_" + rotor_basis_from_tap;
        }
    } else {
        direction = make_direction(native_source, spec.direction_seed, layer_norm_tangent);
    }
    const std::vector<ggml_fp16_t> zero_perturbation(direction.size(), ggml_fp32_to_fp16(0.0f));

    auto zero_run = run_branch(runtime, before_source, source_token, suffix, spec, &zero_perturbation, nullptr, zero);
    runtime.run();
    zero_run.rethrow_if_failed();
    std::vector<scale_result> scale_results;
    for (const float multiplier : epsilon_scales) {
        const float epsilon = layer_norm_rotor ? multiplier * spec.relative_epsilon : multiplier * spec.relative_epsilon * source_norm;
        scale_result result { multiplier, epsilon };
        result.plus.target_captures.resize(max_offset + 1);
        result.minus.target_captures.resize(max_offset + 1);
        const std::vector<ggml_fp16_t> plus_perturbation = layer_norm_rotor ? rotor_perturbation(native_source, direction, epsilon) : fp16_perturbation(direction, epsilon);
        const std::vector<ggml_fp16_t> minus_perturbation = layer_norm_rotor ? rotor_perturbation(native_source, direction, -epsilon) : fp16_perturbation(direction, -epsilon);
        auto plus_run = run_branch(runtime, before_source, source_token, suffix, spec, &plus_perturbation, nullptr, result.plus);
        runtime.run();
        plus_run.rethrow_if_failed();
        auto minus_run = run_branch(runtime, before_source, source_token, suffix, spec, &minus_perturbation, nullptr, result.minus);
        runtime.run();
        minus_run.rethrow_if_failed();
        scale_results.push_back(std::move(result));
    }
    const size_t base_scale_index = (size_t) std::distance(epsilon_scales.begin(), std::find(epsilon_scales.begin(), epsilon_scales.end(), 1.0f));
    scale_result repeat_result { 1.0f, layer_norm_rotor ? spec.relative_epsilon : spec.relative_epsilon * source_norm };
    repeat_result.plus.target_captures.resize(max_offset + 1);
    repeat_result.minus.target_captures.resize(max_offset + 1);
    const std::vector<ggml_fp16_t> repeat_plus = layer_norm_rotor ? rotor_perturbation(native_source, direction, repeat_result.epsilon) : fp16_perturbation(direction, repeat_result.epsilon);
    const std::vector<ggml_fp16_t> repeat_minus = layer_norm_rotor ? rotor_perturbation(native_source, direction, -repeat_result.epsilon) : fp16_perturbation(direction, -repeat_result.epsilon);
    auto repeat_plus_run = run_branch(runtime, before_source, source_token, suffix, spec, &repeat_plus, nullptr, repeat_result.plus);
    runtime.run();
    repeat_plus_run.rethrow_if_failed();
    auto repeat_minus_run = run_branch(runtime, before_source, source_token, suffix, spec, &repeat_minus, nullptr, repeat_result.minus);
    runtime.run();
    repeat_minus_run.rethrow_if_failed();

    const float zero_state_error = state_max_abs_difference(native.final_state, zero.final_state);
    std::ofstream output(output_path);
    if (!output) throw std::runtime_error("failed to open output: " + output_path);
    output << std::setprecision(9) << "{\n  \"schema_version\": 2,\n  \"experiment\": \"rwkv_jacobian_replay_verify\",\n  \"lens_spec\": ";
    rwkv_jacobian_lens::write_json(output, spec);
    output << ",\n  \"model\": ";
    rwkv_experiment::write_json_string(output, model_path);
    output << ",\n  \"prompt\": ";
    rwkv_experiment::write_json_string(output, prompt);
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
    output << "\n    ],\n    \"passed\": " << (linearity_gate_passed ? "true" : "false") << "\n  }";
    output << ",\n  \"inputs\": [";
    if (top_logit_count > 0) {
        if (spec.target_tap != "rwkv.layer." + std::to_string(model->hparams.n_layer() - 1) + ".resid.out") {
            throw std::runtime_error("top-logits requires the final residual target tap");
        }
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);
        int32_t source_layer = -1;
        (void) std::sscanf(spec.source_tap.c_str(), "rwkv.layer.%d.", &source_layer);
        for (size_t offset_index = 0; offset_index < spec.offsets.size(); ++offset_index) {
            const int32_t offset = spec.offsets[offset_index];
            const std::vector<float> & plus_target = capture(scale_results[base_scale_index].plus.target_captures[offset], spec.target_tap).data_f32;
            const std::vector<float> & minus_target = capture(scale_results[base_scale_index].minus.target_captures[offset], spec.target_tap).data_f32;
            std::vector<float> plus_logits(n_vocab), minus_logits(n_vocab), response(n_vocab);
            if (!llama_interp_rwkv_final_readout(ctx, plus_target.data(), 1, plus_logits.data()) ||
                !llama_interp_rwkv_final_readout(ctx, minus_target.data(), 1, minus_logits.data())) {
                throw std::runtime_error("final residual logit evaluation failed");
            }
            for (int32_t token = 0; token < n_vocab; ++token) {
                response[token] = (plus_logits[token] - minus_logits[token]) / (2.0f * scale_results[base_scale_index].epsilon);
            }
            const std::vector<int32_t> top_positive = top_indices(response, top_logit_count, true);
            const std::vector<int32_t> top_negative = top_indices(response, top_logit_count, false);
            const llama_token target_token = offset == 0 ? source_token : suffix[offset - 1];
            if (offset_index) output << ',';
            output << "\n    {\"position\": " << source_position + offset << ", \"input_token\": {\"id\": " << target_token << ", \"text\": ";
            rwkv_experiment::write_json_string(output, common_token_to_piece(ctx, target_token, true));
            output << "}, \"layers\": [{\"layer\": " << source_layer << ", \"lenses\": [{\"lens\": ";
            rwkv_experiment::write_json_string(output, std::string("exploratory.base_logit_derivative.") + (layer_norm_rotor ? "rotor.positive" : "direction.positive"));
            output << ", \"result\": {\"top_logits\": [";
            for (size_t i = 0; i < top_positive.size(); ++i) {
                const int32_t token = top_positive[i];
                if (i) output << ',';
                output << "\n              {\"token_id\": " << token << ", \"token\": ";
                rwkv_experiment::write_json_string(output, common_token_to_piece(ctx, token, true));
                output << ", \"logit\": " << response[token] << '}';
            }
            output << "\n            ]}}, {\"lens\": ";
            rwkv_experiment::write_json_string(output, std::string("exploratory.base_logit_derivative.") + (layer_norm_rotor ? "rotor.negative" : "direction.negative"));
            output << ", \"result\": {\"top_logits\": [";
            for (size_t i = 0; i < top_negative.size(); ++i) {
                const int32_t token = top_negative[i];
                if (i) output << ',';
                output << "\n              {\"token_id\": " << token << ", \"token\": ";
                rwkv_experiment::write_json_string(output, common_token_to_piece(ctx, token, true));
                output << ", \"logit\": " << response[token] << '}';
            }
            output << "\n            ]}}]}]}";
        }
    }
    output << "\n  ]\n}\n";
    if (!output) throw std::runtime_error("failed to write output: " + output_path);
    output.close();

    if (zero_state_error > tolerance || max_zero_target_error > tolerance) {
        std::fprintf(stderr, "verification failed: zero perturbation changed the target activation or exported recurrent state\n");
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 2;
    }
    if (!layer_norm_rotor && spec.source_tap == spec.target_tap &&
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
