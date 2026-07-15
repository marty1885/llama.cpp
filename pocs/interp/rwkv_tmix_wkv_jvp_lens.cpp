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
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void quiet_llama_logs(ggml_log_level, const char *, void *) {}

struct step_capture {
    llama_interp::rwkv_state before;
    llama_token token;
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
        const std::vector<llama_token> & tokens,
        std::vector<step_capture> & output) {
    llama_interp::rwkv_state state = initial;
    for (const llama_token token : tokens) {
        const llama_interp::rwkv_state before = state;
        state = co_await runtime.prefill_tokens(state, { token });
        output.push_back({ before, token });
    }
}

llama_interp::task<> evaluate_step(
        llama_interp::runtime & runtime,
        const llama_interp::rwkv_state & state,
        llama_token token,
        const std::vector<std::string> & taps,
        const std::string * perturb_tap,
        std::vector<ggml_fp16_t> perturbation,
        llama_interp::activation_set & output) {
    auto call = runtime.prefill_tokens(state, { token });
    if (perturb_tap) call.perturb("^" + *perturb_tap + "$", llama_interp::runtime::add_head(-1, std::move(perturbation)));
    for (const std::string & tap : taps) call.capture_f32("^" + tap + "$", output);
    call.discard_state();
    (void) co_await call;
}

std::vector<int32_t> top_indices(const std::vector<float> & values, int32_t count, bool positive) {
    std::vector<int32_t> indices(values.size());
    std::iota(indices.begin(), indices.end(), 0);
    count = std::min<int32_t>(count, (int32_t) indices.size());
    std::partial_sort(indices.begin(), indices.begin() + count, indices.end(), [&values, positive](int32_t a, int32_t b) {
        return positive ? values[a] > values[b] : values[a] < values[b];
    });
    indices.resize(count);
    return indices;
}

void usage(const char * argv0) {
    std::fprintf(stderr, "usage: %s -m MODEL -p PROMPT --output FILE [--target-layer L] [--probe-position N] [-ngl N]\n", argv0);
}

} // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    llama_log_set(quiet_llama_logs, nullptr);
    std::string model_path, prompt, output_path;
    int32_t target_layer = -1;
    int32_t probe_position = -1;
    int n_gpu_layers = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (std::strcmp(argv[i], "-p") == 0 && i + 1 < argc) prompt = argv[++i];
        else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) output_path = argv[++i];
        else if (std::strcmp(argv[i], "--target-layer") == 0 && i + 1 < argc) target_layer = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--probe-position") == 0 && i + 1 < argc) probe_position = std::atoi(argv[++i]);
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
    const int32_t n_layer = model->hparams.n_layer();
    if (target_layer < 0) target_layer = n_layer - 1;
    if (target_layer != n_layer - 1) throw std::runtime_error("the JVP logit lens currently requires the final layer");
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const std::vector<llama_token> tokens = common_tokenize(vocab, prompt, false, true);
    if (tokens.empty()) throw std::runtime_error("prompt tokenized to zero tokens");
    if (probe_position < 0) probe_position = (int32_t) tokens.size() - 1;
    if (probe_position < 0 || probe_position >= (int32_t) tokens.size()) throw std::runtime_error("invalid probe position");

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = std::max(128, (int) tokens.size() + 8);
    context_params.n_batch = context_params.n_ctx;
    context_params.n_ubatch = context_params.n_ctx;
    llama_context * ctx = llama_init_from_model(model, context_params);
    if (!ctx) throw std::runtime_error("failed to create context");
    llama_interp::runtime runtime(ctx, 1);

    std::vector<step_capture> steps;
    auto collection = collect_steps(runtime, runtime.make_state(), tokens, steps);
    runtime.run();
    collection.rethrow_if_failed();
    const step_capture & probe = steps[probe_position];
    const std::string prefix = "rwkv.layer." + std::to_string(target_layer) + ".";
    const std::string wkv_norm = prefix + "time.wkv_norm";
    const std::string resid_out = prefix + "resid.out";
    const std::vector<std::string> taps = { wkv_norm, resid_out };

    llama_interp::activation_set native;
    auto native_run = evaluate_step(runtime, probe.before, probe.token, taps, nullptr, {}, native);
    runtime.run();
    native_run.rethrow_if_failed();
    const std::vector<float> native_wkv_norm = capture(native, wkv_norm).data_f32;
    const float native_wkv_norm_norm = norm(native_wkv_norm);
    if (native_wkv_norm_norm == 0.0f) throw std::runtime_error("zero normalized WKV branch");
    std::vector<float> unit_direction = native_wkv_norm;
    for (float & value : unit_direction) value /= native_wkv_norm_norm;

    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const std::array<float, 3> relative_scales = {{ 0.005f, 0.01f, 0.02f }};
    struct result {
        float relative_scale;
        float epsilon;
        float plus_applied_norm;
        float minus_applied_norm;
        std::vector<float> jvp;
    };
    std::vector<result> results;
    for (const float relative_scale : relative_scales) {
        const float epsilon = relative_scale * native_wkv_norm_norm;
        std::vector<ggml_fp16_t> plus_perturbation(unit_direction.size()), minus_perturbation(unit_direction.size());
        for (size_t i = 0; i < unit_direction.size(); ++i) {
            plus_perturbation[i] = ggml_fp32_to_fp16(epsilon * unit_direction[i]);
            minus_perturbation[i] = ggml_fp32_to_fp16(-epsilon * unit_direction[i]);
        }
        llama_interp::activation_set plus, minus;
        auto plus_run = evaluate_step(runtime, probe.before, probe.token, taps, &wkv_norm, std::move(plus_perturbation), plus);
        runtime.run();
        plus_run.rethrow_if_failed();
        auto minus_run = evaluate_step(runtime, probe.before, probe.token, taps, &wkv_norm, std::move(minus_perturbation), minus);
        runtime.run();
        minus_run.rethrow_if_failed();

        std::vector<float> plus_logits(n_vocab), minus_logits(n_vocab);
        if (!llama_interp_rwkv_final_readout(ctx, capture(plus, resid_out).data_f32.data(), 1, plus_logits.data()) ||
            !llama_interp_rwkv_final_readout(ctx, capture(minus, resid_out).data_f32.data(), 1, minus_logits.data())) {
            throw std::runtime_error("final residual logit evaluation failed");
        }
        std::vector<float> jvp(n_vocab);
        for (int32_t i = 0; i < n_vocab; ++i) jvp[i] = (plus_logits[i] - minus_logits[i]) / (2.0f * epsilon);
        results.push_back({
            relative_scale,
            epsilon,
            norm(subtract(capture(plus, wkv_norm).data_f32, native_wkv_norm)),
            norm(subtract(capture(minus, wkv_norm).data_f32, native_wkv_norm)),
            std::move(jvp),
        });
    }

    const result & reporting_result = results[1];
    const std::vector<int32_t> top_positive = top_indices(reporting_result.jvp, 20, true);
    const std::vector<int32_t> top_negative = top_indices(reporting_result.jvp, 20, false);
    std::ofstream output(output_path);
    if (!output) throw std::runtime_error("failed to open output: " + output_path);
    output << std::setprecision(9) << "{\n"
           << "  \"experiment\": \"rwkv_tmix_normalized_wkv_jvp_logit_lens\",\n"
           << "  \"definition\": \"For the unit direction of native time.wkv_norm, estimate d(final logits)/d(time.wkv_norm) with symmetric production-graph perturbations.\",\n"
           << "  \"limitation\": \"This is a context-conditioned local logit lens, not a baseline-free causal attribution of prior-token memory.\",\n"
           << "  \"model\": ";
    rwkv_experiment::write_json_string(output, model_path);
    output << ",\n  \"prompt\": ";
    rwkv_experiment::write_json_string(output, prompt);
    output << ",\n  \"target_layer\": " << target_layer
           << ",\n  \"probe_position\": " << probe_position
           << ",\n  \"probe_token\": {\"id\": " << probe.token << ", \"text\": ";
    rwkv_experiment::write_json_string(output, common_token_to_piece(ctx, probe.token, true));
    output << "},\n  \"native_wkv_norm_norm\": " << native_wkv_norm_norm
           << ",\n  \"scale_results\": [";
    for (size_t i = 0; i < results.size(); ++i) {
        const result & value = results[i];
        if (i) output << ',';
        output << "\n    {\"relative_scale\": " << value.relative_scale
               << ", \"epsilon\": " << value.epsilon
               << ", \"plus_applied_norm\": " << value.plus_applied_norm
               << ", \"minus_applied_norm\": " << value.minus_applied_norm;
        if (i != 0) output << ", \"jvp_cosine_to_previous\": " << cosine(value.jvp, results[i - 1].jvp);
        output << '}';
    }
    output << "\n  ],\n  \"top_positive_logit_derivative\": [";
    for (size_t i = 0; i < top_positive.size(); ++i) {
        if (i) output << ',';
        const int32_t token = top_positive[i];
        output << "\n    {\"token_id\": " << token << ", \"token\": ";
        rwkv_experiment::write_json_string(output, common_token_to_piece(ctx, token, true));
        output << ", \"d_logit_per_unit\": " << reporting_result.jvp[token] << '}';
    }
    output << "\n  ],\n  \"top_negative_logit_derivative\": [";
    for (size_t i = 0; i < top_negative.size(); ++i) {
        if (i) output << ',';
        const int32_t token = top_negative[i];
        output << "\n    {\"token_id\": " << token << ", \"token\": ";
        rwkv_experiment::write_json_string(output, common_token_to_piece(ctx, token, true));
        output << ", \"d_logit_per_unit\": " << reporting_result.jvp[token] << '}';
    }
    output << "\n  ]\n}\n";
    if (!output) throw std::runtime_error("failed to write output: " + output_path);
    std::printf("wrote=%s probe_position=%d native_wkv_norm=%g\n", output_path.c_str(), probe_position, native_wkv_norm_norm);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
