#include "rwkv_experiment.hpp"

#include "llama-model.h"

#include <algorithm>
#include <cmath>
#include <cctype>
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

struct token_state {
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

llama_interp::task<> advance_tokens(
        llama_interp::runtime & runtime,
        const llama_interp::rwkv_state & initial,
        const std::vector<llama_token> & tokens,
        std::vector<token_state> & output) {
    llama_interp::rwkv_state state = initial;
    for (const llama_token token : tokens) {
        const llama_interp::rwkv_state before = state;
        state = co_await runtime.prefill_tokens(state, { token });
        output.push_back({ before, token });
    }
}

llama_interp::task<> evaluate(
        llama_interp::runtime & runtime,
        const token_state & probe,
        const std::vector<std::string> & taps,
        const std::string * perturb_tap,
        std::vector<ggml_fp16_t> perturbation,
        llama_interp::activation_set & output) {
    auto call = runtime.prefill_tokens(probe.before, { probe.token });
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

std::string trim(std::string value) {
    const auto first = std::find_if_not(value.begin(), value.end(), [](unsigned char ch) { return std::isspace(ch); });
    const auto last = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char ch) { return std::isspace(ch); }).base();
    return first >= last ? "" : std::string(first, last);
}

void usage(const char * argv0) {
    std::fprintf(stderr, "usage: %s -m MODEL --prompts-file FILE --output FILE [--target-layer L] [-ngl N]\n", argv0);
}

} // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    llama_log_set(quiet_llama_logs, nullptr);
    std::string model_path, prompts_path, output_path;
    int32_t target_layer = -1;
    int n_gpu_layers = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (std::strcmp(argv[i], "--prompts-file") == 0 && i + 1 < argc) prompts_path = argv[++i];
        else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) output_path = argv[++i];
        else if (std::strcmp(argv[i], "--target-layer") == 0 && i + 1 < argc) target_layer = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-ngl") == 0 && i + 1 < argc) n_gpu_layers = std::atoi(argv[++i]);
        else { usage(argv[0]); return 1; }
    }
    if (model_path.empty() || prompts_path.empty() || output_path.empty()) { usage(argv[0]); return 1; }
    std::ifstream prompts_input(prompts_path);
    if (!prompts_input) throw std::runtime_error("failed to open prompts file: " + prompts_path);
    std::vector<std::string> prompts;
    for (std::string line; std::getline(prompts_input, line);) {
        line = trim(line);
        if (!line.empty() && line[0] != '#') prompts.push_back(line);
    }
    if (prompts.empty()) throw std::runtime_error("prompts file contains no prompts");

    ggml_backend_load_all();
    llama_backend_init();
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = n_gpu_layers;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), params);
    if (!model || model->arch != LLM_ARCH_RWKV7) throw std::runtime_error("an RWKV-7 model is required");
    const int32_t n_layer = model->hparams.n_layer();
    if (target_layer < 0) target_layer = n_layer - 1;
    if (target_layer != n_layer - 1) throw std::runtime_error("the corpus JVP lens currently requires the final layer");
    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = 128;
    context_params.n_batch = 128;
    context_params.n_ubatch = 128;
    llama_context * ctx = llama_init_from_model(model, context_params);
    if (!ctx) throw std::runtime_error("failed to create context");
    llama_interp::runtime runtime(ctx, 1);
    const std::string prefix = "rwkv.layer." + std::to_string(target_layer) + ".";
    const std::string wkv_norm = prefix + "time.wkv_norm";
    const std::string resid_out = prefix + "resid.out";
    const std::vector<std::string> taps = { wkv_norm, resid_out };
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    static constexpr float relative_scale = 0.01f;

    struct result {
        std::string prompt;
        llama_token token;
        float branch_norm;
        float epsilon;
        float plus_applied_norm;
        float minus_applied_norm;
        std::vector<float> jvp;
    };
    std::vector<result> results;
    for (size_t prompt_index = 0; prompt_index < prompts.size(); ++prompt_index) {
        const std::vector<llama_token> tokens = common_tokenize(vocab, prompts[prompt_index], false, true);
        if (tokens.empty()) throw std::runtime_error("prompt tokenized to zero tokens: " + prompts[prompt_index]);
        std::vector<token_state> states;
        auto advance = advance_tokens(runtime, runtime.make_state(), tokens, states);
        runtime.run();
        advance.rethrow_if_failed();
        const token_state & probe = states.back();
        llama_interp::activation_set native;
        auto native_run = evaluate(runtime, probe, taps, nullptr, {}, native);
        runtime.run();
        native_run.rethrow_if_failed();
        const std::vector<float> native_branch = capture(native, wkv_norm).data_f32;
        const float branch_norm = norm(native_branch);
        if (branch_norm == 0.0f) throw std::runtime_error("zero normalized WKV branch");
        const float epsilon = relative_scale * branch_norm;
        std::vector<ggml_fp16_t> plus(native_branch.size()), minus(native_branch.size());
        for (size_t i = 0; i < native_branch.size(); ++i) {
            plus[i] = ggml_fp32_to_fp16(relative_scale * native_branch[i]);
            minus[i] = ggml_fp32_to_fp16(-relative_scale * native_branch[i]);
        }
        llama_interp::activation_set plus_capture, minus_capture;
        auto plus_run = evaluate(runtime, probe, taps, &wkv_norm, std::move(plus), plus_capture);
        runtime.run();
        plus_run.rethrow_if_failed();
        auto minus_run = evaluate(runtime, probe, taps, &wkv_norm, std::move(minus), minus_capture);
        runtime.run();
        minus_run.rethrow_if_failed();
        std::vector<float> plus_logits(n_vocab), minus_logits(n_vocab), jvp(n_vocab);
        if (!llama_interp_rwkv_final_readout(ctx, capture(plus_capture, resid_out).data_f32.data(), 1, plus_logits.data()) ||
            !llama_interp_rwkv_final_readout(ctx, capture(minus_capture, resid_out).data_f32.data(), 1, minus_logits.data())) {
            throw std::runtime_error("final residual logit evaluation failed");
        }
        for (int32_t i = 0; i < n_vocab; ++i) jvp[i] = (plus_logits[i] - minus_logits[i]) / (2.0f * epsilon);
        results.push_back({
            prompts[prompt_index], probe.token, branch_norm, epsilon,
            norm(subtract(capture(plus_capture, wkv_norm).data_f32, native_branch)),
            norm(subtract(capture(minus_capture, wkv_norm).data_f32, native_branch)), std::move(jvp),
        });
        std::fprintf(stderr, "completed=%zu/%zu\n", prompt_index + 1, prompts.size());
    }

    std::vector<float> mean_jvp(n_vocab);
    for (const result & value : results) {
        for (int32_t i = 0; i < n_vocab; ++i) mean_jvp[i] += value.jvp[i] / results.size();
    }
    float pairwise_cosine_sum = 0.0f;
    int32_t pairwise_count = 0;
    for (size_t i = 0; i < results.size(); ++i) {
        for (size_t j = 0; j < i; ++j) {
            pairwise_cosine_sum += cosine(results[i].jvp, results[j].jvp);
            ++pairwise_count;
        }
    }
    const std::vector<int32_t> aggregate_positive = top_indices(mean_jvp, 20, true);
    const std::vector<int32_t> aggregate_negative = top_indices(mean_jvp, 20, false);

    std::ofstream output(output_path);
    if (!output) throw std::runtime_error("failed to open output: " + output_path);
    output << std::setprecision(9) << "{\n"
           << "  \"experiment\": \"rwkv_tmix_normalized_wkv_jvp_logit_lens_corpus\",\n"
           << "  \"definition\": \"For each independently prefixed prompt, estimate d(final logits)/d(time.wkv_norm) in the native time.wkv_norm direction with a 1% symmetric production-graph perturbation.\",\n"
           << "  \"limitation\": \"This hand-audited corpus is exploratory. The result is a local branch logit lens, not a semantic or prior-memory attribution.\",\n"
           << "  \"model\": ";
    rwkv_experiment::write_json_string(output, model_path);
    output << ",\n  \"prompts_file\": ";
    rwkv_experiment::write_json_string(output, prompts_path);
    output << ",\n  \"target_layer\": " << target_layer
           << ",\n  \"relative_scale\": " << relative_scale
           << ",\n  \"mean_pairwise_jvp_cosine\": " << (pairwise_count == 0 ? 0.0f : pairwise_cosine_sum / pairwise_count)
           << ",\n  \"contexts\": [";
    for (size_t i = 0; i < results.size(); ++i) {
        const result & value = results[i];
        if (i) output << ',';
        const std::vector<int32_t> positive = top_indices(value.jvp, 8, true);
        const std::vector<int32_t> negative = top_indices(value.jvp, 8, false);
        output << "\n    {\"prompt\": ";
        rwkv_experiment::write_json_string(output, value.prompt);
        output << ", \"probe_token\": {\"id\": " << value.token << ", \"text\": ";
        rwkv_experiment::write_json_string(output, common_token_to_piece(ctx, value.token, true));
        output << "}, \"native_wkv_norm_norm\": " << value.branch_norm
               << ", \"epsilon\": " << value.epsilon
               << ", \"plus_applied_norm\": " << value.plus_applied_norm
               << ", \"minus_applied_norm\": " << value.minus_applied_norm
               << ", \"jvp_cosine_to_mean\": " << cosine(value.jvp, mean_jvp)
               << ", \"top_positive\": [";
        for (size_t j = 0; j < positive.size(); ++j) {
            if (j) output << ',';
            const int32_t token = positive[j];
            output << "{\"token_id\": " << token << ", \"token\": ";
            rwkv_experiment::write_json_string(output, common_token_to_piece(ctx, token, true));
            output << ", \"d_logit_per_unit\": " << value.jvp[token] << '}';
        }
        output << "], \"top_negative\": [";
        for (size_t j = 0; j < negative.size(); ++j) {
            if (j) output << ',';
            const int32_t token = negative[j];
            output << "{\"token_id\": " << token << ", \"token\": ";
            rwkv_experiment::write_json_string(output, common_token_to_piece(ctx, token, true));
            output << ", \"d_logit_per_unit\": " << value.jvp[token] << '}';
        }
        output << "]}";
    }
    auto write_aggregate = [&output, ctx, &mean_jvp](const char * name, const std::vector<int32_t> & indices) {
        output << "\n  \"" << name << "\": [";
        for (size_t i = 0; i < indices.size(); ++i) {
            if (i) output << ',';
            const int32_t token = indices[i];
            output << "\n    {\"token_id\": " << token << ", \"token\": ";
            rwkv_experiment::write_json_string(output, common_token_to_piece(ctx, token, true));
            output << ", \"mean_d_logit_per_unit\": " << mean_jvp[token] << '}';
        }
        output << "\n  ]";
    };
    output << "\n  ],";
    write_aggregate("mean_top_positive", aggregate_positive);
    output << ',';
    write_aggregate("mean_top_negative", aggregate_negative);
    output << "\n}\n";
    if (!output) throw std::runtime_error("failed to write output: " + output_path);
    std::printf("wrote=%s contexts=%zu mean_pairwise_jvp_cosine=%g\n", output_path.c_str(), results.size(),
                pairwise_count == 0 ? 0.0f : pairwise_cosine_sum / pairwise_count);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
