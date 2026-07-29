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
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void quiet_llama_logs(ggml_log_level, const char *, void *) {}

struct point {
    float theta;
    float teacher_logit_delta;
    float teacher_logprob_delta;
    float native_argmax_logit_delta;
    float native_argmax_logprob_delta;
    int32_t argmax;
};

struct layer_result { std::vector<point> points; };

float l2_norm(const std::vector<float> & values) {
    double sum = 0.0;
    for (const float value : values) sum += (double) value * value;
    return (float) std::sqrt(sum);
}

const llama_interp_activation & capture(const llama_interp::activation_set & captures, const std::string & name) {
    return rwkv_experiment::require_capture(captures, name);
}

std::vector<float> tangent(const std::vector<float> & source, std::vector<float> direction) {
    if (source.size() != direction.size()) throw std::runtime_error("residual update dimension differs from source");
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
    if (radial_norm_sq == 0.0f) throw std::runtime_error("cannot rotate a constant source activation");
    for (size_t i = 0; i < source.size(); ++i) direction[i] -= radial_dot / radial_norm_sq * (source[i] - source_mean);
    const float norm = l2_norm(direction);
    if (norm == 0.0f) throw std::runtime_error("residual update has no LayerNorm tangent component");
    for (float & value : direction) value /= norm;
    return direction;
}

std::vector<ggml_fp16_t> rotor_delta(const std::vector<float> & source, const std::vector<float> & tangent_vector, float theta) {
    float mean = 0.0f, radius_sq = 0.0f;
    for (const float value : source) mean += value / source.size();
    for (const float value : source) radius_sq += (value - mean) * (value - mean);
    const float radius = std::sqrt(radius_sq);
    std::vector<ggml_fp16_t> result(source.size());
    for (size_t i = 0; i < source.size(); ++i) {
        const float radial_unit = (source[i] - mean) / radius;
        result[i] = ggml_fp32_to_fp16(radius * ((std::cos(theta) - 1.0f) * radial_unit + std::sin(theta) * tangent_vector[i]));
    }
    return result;
}

llama_interp::task<> run_perturbed(
        llama_interp::runtime & runtime,
        const llama_interp::rwkv_state & before_source,
        llama_token source_token,
        const std::string & source_tap,
        const std::string & target_tap,
        const std::vector<ggml_fp16_t> & delta,
        llama_interp::activation_set & captures) {
    auto step = runtime.prefill_tokens(before_source, { source_token });
    step.perturb("^" + source_tap + "$", llama_interp::runtime::add_head(-1, delta));
    step.capture_f32("^" + target_tap + "$", captures);
    (void) co_await step;
}

float log_probability(const std::vector<float> & logits, int32_t token) {
    const float maximum = *std::max_element(logits.begin(), logits.end());
    double sum = 0.0;
    for (const float value : logits) sum += std::exp((double) value - maximum);
    return logits[token] - maximum - (float) std::log(sum);
}

int32_t argmax(const std::vector<float> & logits) {
    return (int32_t) std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()));
}

void usage(const char * argv0) {
    std::fprintf(stderr, "usage: %s -m MODEL -p PROMPT --output FILE [--source-position N] [--theta R ...] [-ngl N]\n", argv0);
}

} // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    llama_log_set(quiet_llama_logs, nullptr);
    std::string model_path, prompt, output_path;
    int32_t source_position = -1;
    int n_gpu_layers = 0;
    std::vector<float> angles;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (std::strcmp(argv[i], "-p") == 0 && i + 1 < argc) prompt = argv[++i];
        else if (std::strcmp(argv[i], "--source-position") == 0 && i + 1 < argc) source_position = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--theta") == 0 && i + 1 < argc) angles.push_back(std::strtof(argv[++i], nullptr));
        else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) output_path = argv[++i];
        else if (std::strcmp(argv[i], "-ngl") == 0 && i + 1 < argc) n_gpu_layers = std::atoi(argv[++i]);
        else { usage(argv[0]); return 1; }
    }
    if (model_path.empty() || prompt.empty() || output_path.empty()) { usage(argv[0]); return 1; }
    if (angles.empty()) angles = { -0.10f, -0.05f, -0.02f, -0.01f, 0.0f, 0.01f, 0.02f, 0.05f, 0.10f };

    ggml_backend_load_all();
    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model || model->arch != LLM_ARCH_RWKV7) throw std::runtime_error("an RWKV-7 model is required");
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const std::vector<llama_token> tokens = common_tokenize(vocab, prompt, false, true);
    if (source_position < 0) source_position = (int32_t) tokens.size() - 2;
    if (source_position < 0 || source_position + 1 >= (int32_t) tokens.size()) throw std::runtime_error("source position needs a teacher-forced next token");
    const int32_t n_layer = model->hparams.n_layer();

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = std::max(128, (int) tokens.size() + 8);
    context_params.n_batch = context_params.n_ctx;
    context_params.n_ubatch = context_params.n_ctx;
    llama_context * ctx = llama_init_from_model(model, context_params);
    if (!ctx) throw std::runtime_error("failed to create context");
    llama_interp::runtime runtime(ctx, 1);
    llama_interp::rwkv_state before_source = runtime.make_state();
    if (source_position > 0) {
        const std::vector<llama_token> prefix(tokens.begin(), tokens.begin() + source_position);
        auto prefix_task = [&]() -> llama_interp::task<> { before_source = co_await runtime.prefill_tokens(before_source, prefix); }();
        runtime.run();
        prefix_task.rethrow_if_failed();
    }

    const llama_token source_token = tokens[source_position];
    const llama_token teacher_token = tokens[source_position + 1];
    const std::string target_tap = "rwkv.layer." + std::to_string(n_layer - 1) + ".resid.out";
    llama_interp::activation_set native;
    auto native_step = runtime.prefill_tokens(before_source, { source_token });
    native_step.capture_f32("^" + target_tap + "$", native);
    for (int32_t layer = 0; layer < n_layer; ++layer) {
        const std::string prefix = "rwkv.layer." + std::to_string(layer) + ".";
        native_step.capture_f32("^" + prefix + "resid.in$", native);
        native_step.capture_f32("^" + prefix + "resid.time$", native);
        if (prefix + "resid.out" != target_tap) native_step.capture_f32("^" + prefix + "resid.out$", native);
    }
    auto native_task = [&]() -> llama_interp::task<> { (void) co_await native_step; }();
    runtime.run();
    native_task.rethrow_if_failed();

    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<float> native_logits(n_vocab);
    if (!llama_interp_rwkv_final_readout(ctx, capture(native, target_tap).data_f32.data(), 1, native_logits.data())) throw std::runtime_error("native final readout failed");
    const int32_t native_argmax = argmax(native_logits);
    const float native_teacher_logit = native_logits[teacher_token];
    const float native_teacher_logprob = log_probability(native_logits, teacher_token);
    const float native_argmax_logit = native_logits[native_argmax];
    const float native_argmax_logprob = log_probability(native_logits, native_argmax);

    std::vector<layer_result> results(n_layer);
    for (int32_t layer = 0; layer < n_layer; ++layer) {
        const std::string prefix = "rwkv.layer." + std::to_string(layer) + ".";
        const std::vector<float> & source = capture(native, prefix + "resid.in").data_f32;
        const std::vector<float> & time = capture(native, prefix + "resid.time").data_f32;
        const std::vector<float> & output = capture(native, prefix + "resid.out").data_f32;
        std::vector<float> update(source.size());
        for (size_t i = 0; i < update.size(); ++i) update[i] = output[i] - time[i];
        const std::vector<float> direction = tangent(source, std::move(update));
        for (const float theta : angles) {
            const std::vector<ggml_fp16_t> delta = rotor_delta(source, direction, theta);
            llama_interp::activation_set perturbed;
            auto task = run_perturbed(runtime, before_source, source_token, prefix + "resid.in", target_tap, delta, perturbed);
            runtime.run();
            task.rethrow_if_failed();
            std::vector<float> logits(n_vocab);
            if (!llama_interp_rwkv_final_readout(ctx, capture(perturbed, target_tap).data_f32.data(), 1, logits.data())) throw std::runtime_error("rotor final readout failed");
            results[layer].points.push_back({ theta, logits[teacher_token] - native_teacher_logit,
                log_probability(logits, teacher_token) - native_teacher_logprob, logits[native_argmax] - native_argmax_logit,
                log_probability(logits, native_argmax) - native_argmax_logprob, argmax(logits) });
        }
        std::fprintf(stderr, "completed_layer=%d/%d\n", layer + 1, n_layer);
    }

    struct ranked_point { int32_t layer; const point * value; };
    std::vector<ranked_point> ranked;
    for (int32_t layer = 0; layer < n_layer; ++layer) for (const point & value : results[layer].points) {
        if (value.theta != 0.0f) ranked.push_back({ layer, &value });
    }
    const auto write_ranked = [&](std::ostream & stream, bool positive) {
        std::sort(ranked.begin(), ranked.end(), [positive](const ranked_point & a, const ranked_point & b) {
            return positive ? a.value->teacher_logprob_delta > b.value->teacher_logprob_delta : a.value->teacher_logprob_delta < b.value->teacher_logprob_delta;
        });
        stream << '[';
        for (size_t i = 0; i < std::min<size_t>(5, ranked.size()); ++i) {
            if (i) stream << ',';
            stream << "{\"layer\": " << ranked[i].layer << ", \"theta\": " << ranked[i].value->theta
                   << ", \"teacher_logprob_delta\": " << ranked[i].value->teacher_logprob_delta
                   << ", \"teacher_logit_delta\": " << ranked[i].value->teacher_logit_delta << '}';
        }
        stream << ']';
    };

    std::ofstream json(output_path);
    if (!json) throw std::runtime_error("failed to open output");
    json << std::setprecision(9) << "{\n  \"experiment\": \"rwkv_rotor_response_layers\",\n  \"model\": ";
    rwkv_experiment::write_json_string(json, model_path);
    json << ",\n  \"prompt\": "; rwkv_experiment::write_json_string(json, prompt);
    json << ",\n  \"source_position\": " << source_position << ",\n  \"basis\": \"resid.out - resid.time, projected onto each resid.in LayerNorm tangent sphere\""
         << ",\n  \"teacher_next_token\": {\"id\": " << teacher_token << ", \"text\": "; rwkv_experiment::write_json_string(json, common_token_to_piece(ctx, teacher_token, true));
    json << "},\n  \"native_argmax\": {\"id\": " << native_argmax << ", \"text\": "; rwkv_experiment::write_json_string(json, common_token_to_piece(ctx, native_argmax, true));
    json << ", \"logit\": " << native_argmax_logit << ", \"logprob\": " << native_argmax_logprob << "},\n  \"layers\": [";
    for (int32_t layer = 0; layer < n_layer; ++layer) {
        if (layer) json << ',';
        json << "\n    {\"layer\": " << layer << ", \"points\": [";
        for (size_t i = 0; i < results[layer].points.size(); ++i) {
            const point & value = results[layer].points[i];
            if (i) json << ',';
            json << "{\"theta\": " << value.theta << ", \"teacher_logit_delta\": " << value.teacher_logit_delta
                 << ", \"teacher_logprob_delta\": " << value.teacher_logprob_delta
                 << ", \"native_argmax_logit_delta\": " << value.native_argmax_logit_delta
                 << ", \"native_argmax_logprob_delta\": " << value.native_argmax_logprob_delta
                 << ", \"argmax\": {\"id\": " << value.argmax << ", \"text\": ";
            rwkv_experiment::write_json_string(json, common_token_to_piece(ctx, value.argmax, true));
            json << "}}";
        }
        json << "]}";
    }
    json << "\n  ],\n  \"top_five_teacher_logprob_effects\": {\"positive\": ";
    write_ranked(json, true);
    json << ", \"negative\": ";
    write_ranked(json, false);
    json << "}\n}\n";
    if (!json) throw std::runtime_error("failed to write output");
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
