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

struct branch_result {
    std::vector<llama_interp::activation_set> targets;
};

struct lens_result {
    std::vector<int32_t> positive;
    std::vector<int32_t> negative;
    std::vector<float> derivative;
};

const llama_interp_activation & capture(const llama_interp::activation_set & captures, const std::string & name) {
    return rwkv_experiment::require_capture(captures, name);
}

float l2_norm(const std::vector<float> & values) {
    double sum = 0.0;
    for (const float value : values) sum += (double) value * value;
    return (float) std::sqrt(sum);
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
    if (radius == 0.0f) throw std::runtime_error("cannot rotate constant LayerNorm source activation");
    std::vector<ggml_fp16_t> result(source.size());
    for (size_t i = 0; i < source.size(); ++i) {
        const float radial_unit = (source[i] - mean) / radius;
        const float delta = radius * ((std::cos(theta) - 1.0f) * radial_unit + std::sin(theta) * tangent[i]);
        result[i] = ggml_fp32_to_fp16(delta);
    }
    return result;
}

std::vector<int32_t> top_indices(const std::vector<float> & values, int32_t count, bool descending) {
    std::vector<int32_t> indices(values.size());
    std::iota(indices.begin(), indices.end(), 0);
    count = std::min<int32_t>(count, (int32_t) indices.size());
    std::partial_sort(indices.begin(), indices.begin() + count, indices.end(), [&values, descending](int32_t a, int32_t b) {
        return descending ? values[a] > values[b] : values[a] < values[b];
    });
    indices.resize(count);
    return indices;
}

llama_interp::task<> run_branch(
        llama_interp::runtime & runtime,
        const llama_interp::rwkv_state & before_source,
        llama_token source_token,
        const std::vector<llama_token> & suffix,
        const std::string & source_tap,
        const std::vector<ggml_fp16_t> * perturbation,
        const std::string & target_tap,
        const std::vector<int32_t> & offsets,
        branch_result & output) {
    auto source = runtime.prefill_tokens(before_source, { source_token });
    if (perturbation) source.perturb("^" + source_tap + "$", llama_interp::runtime::add_head(-1, *perturbation));
    if (std::binary_search(offsets.begin(), offsets.end(), 0)) source.capture_f32("^" + target_tap + "$", output.targets[0]);
    llama_interp::rwkv_state state = co_await source;
    for (size_t offset = 1; offset <= suffix.size(); ++offset) {
        auto step = runtime.prefill_tokens(state, { suffix[offset - 1] });
        if (std::binary_search(offsets.begin(), offsets.end(), (int32_t) offset)) {
            step.capture_f32("^" + target_tap + "$", output.targets[offset]);
        }
        state = co_await step;
    }
}

void write_lens(std::ostream & output, const llama_context * ctx, const std::string & name, const std::vector<int32_t> & tokens, const std::vector<float> & derivative) {
    output << "{\"lens\": ";
    rwkv_experiment::write_json_string(output, name);
    output << ", \"result\": {\"top_logits\": [";
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i) output << ',';
        const int32_t token = tokens[i];
        output << "\n                {\"token_id\": " << token << ", \"token\": ";
        rwkv_experiment::write_json_string(output, common_token_to_piece(ctx, token, true));
        output << ", \"logit\": " << derivative[token] << '}';
    }
    output << "\n              ]}}";
}

void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s -m MODEL -p PROMPT --output FILE [--source-position N] [--offset N ...]\n"
        "       [--relative-epsilon E] [--top-logits N] [-ngl N]\n", argv0);
}

} // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    llama_log_set(quiet_llama_logs, nullptr);
    std::string model_path, prompt, output_path;
    int32_t source_position = -1, top_logit_count = 20;
    int n_gpu_layers = 0;
    float relative_epsilon = 0.01f;
    std::vector<int32_t> offsets;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (std::strcmp(argv[i], "-p") == 0 && i + 1 < argc) prompt = argv[++i];
        else if (std::strcmp(argv[i], "--source-position") == 0 && i + 1 < argc) source_position = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--offset") == 0 && i + 1 < argc) offsets.push_back(std::atoi(argv[++i]));
        else if (std::strcmp(argv[i], "--relative-epsilon") == 0 && i + 1 < argc) relative_epsilon = std::strtof(argv[++i], nullptr);
        else if (std::strcmp(argv[i], "--top-logits") == 0 && i + 1 < argc) top_logit_count = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) output_path = argv[++i];
        else if (std::strcmp(argv[i], "-ngl") == 0 && i + 1 < argc) n_gpu_layers = std::atoi(argv[++i]);
        else { usage(argv[0]); return 1; }
    }
    if (model_path.empty() || prompt.empty() || output_path.empty() || relative_epsilon <= 0.0f || top_logit_count <= 0) {
        usage(argv[0]);
        return 1;
    }
    if (offsets.empty()) offsets = { 0, 1, 2, 4 };
    std::sort(offsets.begin(), offsets.end());
    offsets.erase(std::unique(offsets.begin(), offsets.end()), offsets.end());
    if (offsets.front() < 0) throw std::runtime_error("offsets must be non-negative");

    ggml_backend_load_all();
    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model || model->arch != LLM_ARCH_RWKV7) throw std::runtime_error("an RWKV-7 model is required");
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const std::vector<llama_token> tokens = common_tokenize(vocab, prompt, false, true);
    const int32_t max_offset = offsets.back();
    if (source_position < 0) source_position = (int32_t) tokens.size() - 1 - max_offset;
    if (source_position < 0 || source_position + max_offset >= (int32_t) tokens.size()) {
        throw std::runtime_error("source position needs all requested teacher-forced future offsets in the prompt");
    }

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = std::max(128, (int) tokens.size() + 8);
    context_params.n_batch = context_params.n_ctx;
    context_params.n_ubatch = context_params.n_ctx;
    llama_context * ctx = llama_init_from_model(model, context_params);
    if (!ctx) throw std::runtime_error("failed to create context");
    llama_interp::runtime runtime(ctx, 1);
    llama_interp::rwkv_state before_source = runtime.make_state();
    if (source_position > 0) {
        const std::vector<llama_token> prefix_tokens(tokens.begin(), tokens.begin() + source_position);
        auto prefix = [&]() -> llama_interp::task<> {
            before_source = co_await runtime.prefill_tokens(before_source, prefix_tokens);
        }();
        runtime.run();
        prefix.rethrow_if_failed();
    }

    const int32_t n_layer = model->hparams.n_layer();
    const llama_token source_token = tokens[source_position];
    const std::vector<llama_token> suffix(tokens.begin() + source_position + 1, tokens.begin() + source_position + 1 + max_offset);
    const std::string target_tap = "rwkv.layer." + std::to_string(n_layer - 1) + ".resid.out";
    const std::vector<std::string> basis_names = { "time.k", "time.v", "time.wkv", "time.wkv_norm", "channel.out" };
    llama_interp::activation_set native_captures;
    auto native = runtime.prefill_tokens(before_source, { source_token });
    for (int32_t layer = 0; layer < n_layer; ++layer) {
        const std::string prefix = "rwkv.layer." + std::to_string(layer) + ".";
        native.capture_f32("^" + prefix + "resid.in$", native_captures);
        for (const std::string & basis : basis_names) native.capture_f32("^" + prefix + basis + "$", native_captures);
    }
    native.discard_state();
    auto native_task = [&]() -> llama_interp::task<> { (void) co_await native; }();
    runtime.run();
    native_task.rethrow_if_failed();

    std::vector<std::vector<std::vector<lens_result>>> results(offsets.size(), std::vector<std::vector<lens_result>>(n_layer, std::vector<lens_result>(basis_names.size())));
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    for (int32_t layer = 0; layer < n_layer; ++layer) {
        const std::string prefix = "rwkv.layer." + std::to_string(layer) + ".";
        const std::string source_tap = prefix + "resid.in";
        const std::vector<float> & source = capture(native_captures, source_tap).data_f32;
        for (size_t basis_index = 0; basis_index < basis_names.size(); ++basis_index) {
            const std::vector<float> tangent = project_layer_norm_tangent(source, capture(native_captures, prefix + basis_names[basis_index]).data_f32);
            branch_result plus, minus;
            plus.targets.resize(max_offset + 1);
            minus.targets.resize(max_offset + 1);
            const std::vector<ggml_fp16_t> plus_delta = rotor_perturbation(source, tangent, relative_epsilon);
            const std::vector<ggml_fp16_t> minus_delta = rotor_perturbation(source, tangent, -relative_epsilon);
            auto plus_task = run_branch(runtime, before_source, source_token, suffix, source_tap, &plus_delta, target_tap, offsets, plus);
            runtime.run();
            plus_task.rethrow_if_failed();
            auto minus_task = run_branch(runtime, before_source, source_token, suffix, source_tap, &minus_delta, target_tap, offsets, minus);
            runtime.run();
            minus_task.rethrow_if_failed();
            for (size_t offset_index = 0; offset_index < offsets.size(); ++offset_index) {
                const int32_t offset = offsets[offset_index];
                lens_result & result = results[offset_index][layer][basis_index];
                result.derivative.resize(n_vocab);
                std::vector<float> plus_logits(n_vocab), minus_logits(n_vocab);
                if (!llama_interp_rwkv_final_readout(ctx, capture(plus.targets[offset], target_tap).data_f32.data(), 1, plus_logits.data()) ||
                    !llama_interp_rwkv_final_readout(ctx, capture(minus.targets[offset], target_tap).data_f32.data(), 1, minus_logits.data())) {
                    throw std::runtime_error("final residual logit evaluation failed");
                }
                for (int32_t token = 0; token < n_vocab; ++token) result.derivative[token] = (plus_logits[token] - minus_logits[token]) / (2.0f * relative_epsilon);
                result.positive = top_indices(result.derivative, top_logit_count, true);
                result.negative = top_indices(result.derivative, top_logit_count, false);
            }
        }
        std::fprintf(stderr, "completed_layer=%d/%d\n", layer + 1, n_layer);
    }

    std::ofstream output(output_path);
    if (!output) throw std::runtime_error("failed to open output: " + output_path);
    output << std::setprecision(9) << "{\n  \"schema_version\": 2,\n  \"experiment\": \"rwkv-rotor-layers\",\n  \"model\": ";
    rwkv_experiment::write_json_string(output, model_path);
    output << ",\n  \"prompt\": ";
    rwkv_experiment::write_json_string(output, prompt);
    output << ",\n  \"top_n\": " << top_logit_count
           << ",\n  \"metadata\": {\n    \"method\": \"exploratory LayerNorm-sphere rotor central derivatives through the production RWKV graph\",\n    \"source\": \"rwkv.layer.<L>.resid.in at one teacher-forced source token\",\n    \"bases\": \"time.k, time.v, time.wkv, time.wkv_norm; each is projected onto the source LayerNorm tangent sphere\",\n    \"target\": \"final rwkv.layer.<last>.resid.out read out with llama_interp_rwkv_final_readout\",\n    \"responses\": \"positive and negative lenses are respectively descending and ascending signed central logit derivatives; no result gating or suppression\",\n    \"relative_epsilon\": " << relative_epsilon
           << ",\n    \"source_position\": " << source_position << "\n  },\n  \"inputs\": [";
    for (size_t offset_index = 0; offset_index < offsets.size(); ++offset_index) {
        const int32_t position = source_position + offsets[offset_index];
        if (offset_index) output << ',';
        output << "\n    {\n      \"position\": " << position << ",\n      \"input_token\": {\"id\": " << tokens[position] << ", \"text\": ";
        rwkv_experiment::write_json_string(output, common_token_to_piece(ctx, tokens[position], true));
        output << "},\n      \"layers\": [";
        for (int32_t layer = 0; layer < n_layer; ++layer) {
            if (layer) output << ',';
            output << "\n        {\n          \"layer\": " << layer << ",\n          \"lenses\": [";
            for (size_t basis_index = 0; basis_index < basis_names.size(); ++basis_index) {
                if (basis_index) output << ',';
                const lens_result & result = results[offset_index][layer][basis_index];
                output << "\n            ";
                write_lens(output, ctx, basis_names[basis_index] + ".rotor.positive", result.positive, result.derivative);
                output << ",\n            ";
                write_lens(output, ctx, basis_names[basis_index] + ".rotor.negative", result.negative, result.derivative);
            }
            output << "\n          ]\n        }";
        }
        output << "\n      ]\n    }";
    }
    output << "\n  ]\n}\n";
    if (!output) throw std::runtime_error("failed to write JSON output: " + output_path);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
