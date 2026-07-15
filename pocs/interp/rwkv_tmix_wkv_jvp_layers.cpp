#include "rwkv_experiment.hpp"

#include "llama-model.h"

#include <TFile.h>
#include <TTree.h>

#include <algorithm>
#include <cmath>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
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

float norm(const std::vector<float> & value) { return std::sqrt(dot(value, value)); }

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
        llama_interp::rwkv_state & state) {
    state = initial;
    for (const llama_token token : tokens) state = co_await runtime.prefill_tokens(state, { token });
}

llama_interp::task<> advance_generated(
        llama_interp::runtime & runtime,
        llama_interp::rwkv_state & state,
        int32_t count) {
    for (int32_t i = 0; i < count; ++i) {
        if (!state.has_next) throw std::runtime_error("generation state has no next token");
        state = co_await runtime.prefill_tokens(state, { state.next_token });
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

void usage(const char * argv0) {
    std::fprintf(stderr, "usage: %s -m MODEL --prompt-file FILE --output FILE --json-output FILE --generated-prefix N [-ngl N]\n", argv0);
}

} // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    llama_log_set(quiet_llama_logs, nullptr);
    std::string model_path, prompt_path, output_path, json_output_path;
    int32_t generated_prefix = -1;
    int n_gpu_layers = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (std::strcmp(argv[i], "--prompt-file") == 0 && i + 1 < argc) prompt_path = argv[++i];
        else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) output_path = argv[++i];
        else if (std::strcmp(argv[i], "--json-output") == 0 && i + 1 < argc) json_output_path = argv[++i];
        else if (std::strcmp(argv[i], "--generated-prefix") == 0 && i + 1 < argc) generated_prefix = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-ngl") == 0 && i + 1 < argc) n_gpu_layers = std::atoi(argv[++i]);
        else { usage(argv[0]); return 1; }
    }
    if (model_path.empty() || prompt_path.empty() || output_path.empty() || json_output_path.empty() || generated_prefix < 0) { usage(argv[0]); return 1; }
    std::ifstream prompt_input(prompt_path);
    if (!prompt_input) throw std::runtime_error("failed to open prompt file: " + prompt_path);
    const std::string prompt((std::istreambuf_iterator<char>(prompt_input)), std::istreambuf_iterator<char>());
    if (prompt.empty()) throw std::runtime_error("prompt file is empty");

    ggml_backend_load_all();
    llama_backend_init();
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = n_gpu_layers;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), params);
    if (!model || model->arch != LLM_ARCH_RWKV7) throw std::runtime_error("an RWKV-7 model is required");
    const int32_t n_layer = model->hparams.n_layer();
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const std::vector<llama_token> tokens = common_tokenize(vocab, prompt, false, true);
    if (tokens.empty()) throw std::runtime_error("prompt tokenized to zero tokens");

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = std::max(128, (int) tokens.size() + 8);
    context_params.n_batch = context_params.n_ctx;
    context_params.n_ubatch = context_params.n_ctx;
    llama_context * ctx = llama_init_from_model(model, context_params);
    if (!ctx) throw std::runtime_error("failed to create context");
    llama_interp::runtime runtime(ctx, 1);
    llama_interp::rwkv_state state;
    auto advance = advance_tokens(runtime, runtime.make_state(), tokens, state);
    runtime.run();
    advance.rethrow_if_failed();
    auto generated = advance_generated(runtime, state, generated_prefix);
    runtime.run();
    generated.rethrow_if_failed();
    if (!state.has_next) throw std::runtime_error("generation state has no next token");
    const token_state probe = { state, state.next_token };
    const std::string final_resid = "rwkv.layer." + std::to_string(n_layer - 1) + ".resid.out";
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    static constexpr float relative_scale = 0.01f;
    std::vector<std::pair<std::string, int32_t>> tracked;
    for (const std::string & piece : { " Paris", " France", " Berlin", " Rome", " Tokyo", " Jupiter", "Focus", "Forms", "Force", "Forum", "Flush", "Floor", "Float", "Found" }) {
        for (int32_t token = 0; token < n_vocab; ++token) {
            if (common_token_to_piece(ctx, token, true) == piece) {
                tracked.push_back({ piece, token });
                break;
            }
        }
    }
    TFile output(output_path.c_str(), "RECREATE");
    if (output.IsZombie()) throw std::runtime_error("failed to create ROOT output: " + output_path);
    TTree metadata("metadata", "all-layer normalized-WKV JVP metadata");
    std::string metadata_model = model_path, metadata_prompt_path = prompt_path, metadata_prompt = prompt;
    int32_t metadata_probe_token = probe.token, metadata_prompt_token_count = (int32_t) tokens.size();
    int32_t metadata_generated_prefix = generated_prefix;
    float metadata_relative_scale = relative_scale;
    metadata.Branch("model", &metadata_model);
    metadata.Branch("prompt_path", &metadata_prompt_path);
    metadata.Branch("prompt", &metadata_prompt);
    metadata.Branch("probe_token", &metadata_probe_token);
    metadata.Branch("prompt_token_count", &metadata_prompt_token_count);
    metadata.Branch("relative_scale", &metadata_relative_scale);
    metadata.Branch("generated_prefix", &metadata_generated_prefix);
    metadata.Fill();
    TTree tree("rwkv_wkv_jvp", "one streamed normalized-WKV JVP per layer");
    int32_t result_layer = 0;
    float result_branch_norm = 0.0f, result_epsilon = 0.0f, result_plus_applied_norm = 0.0f, result_minus_applied_norm = 0.0f;
    std::vector<float> result_jvp, result_tracked;
    std::vector<int32_t> result_top_positive, result_top_negative, tracked_token_ids;
    for (const auto & [piece, token] : tracked) {
        (void) piece;
        tracked_token_ids.push_back(token);
    }
    tree.Branch("layer", &result_layer);
    tree.Branch("native_wkv_norm_norm", &result_branch_norm);
    tree.Branch("epsilon", &result_epsilon);
    tree.Branch("plus_applied_norm", &result_plus_applied_norm);
    tree.Branch("minus_applied_norm", &result_minus_applied_norm);
    tree.Branch("logit_jvp", &result_jvp);
    tree.Branch("tracked_token_ids", &tracked_token_ids);
    tree.Branch("tracked_d_logit_per_unit", &result_tracked);
    tree.Branch("top_positive_token_ids", &result_top_positive);
    tree.Branch("top_negative_token_ids", &result_top_negative);
    std::ofstream json(json_output_path);
    if (!json) throw std::runtime_error("failed to open JSON output: " + json_output_path);
    json << std::setprecision(9) << "{\n  \"schema_version\": 2,\n  \"experiment\": \"rwkv-normalized-wkv-jvp-generation-lens\",\n  \"model\": ";
    rwkv_experiment::write_json_string(json, model_path);
    json << ",\n  \"prompt\": ";
    rwkv_experiment::write_json_string(json, prompt);
    json << ",\n  \"top_n\": 10,\n  \"metadata\": {\n    \"generation\": \"greedy production decode; fresh all-layer local JVP at the selected generated token\",\n    \"generated_prefix\": \"" << generated_prefix
         << "\",\n    \"jvp\": \"symmetric 1% production-graph derivative at time.wkv_norm\",\n    \"negative_lens\": \"top logits are the largest negative derivatives, reported as positive magnitudes\"\n  },\n  \"inputs\": [\n    {\n      \"position\": " << generated_prefix << ",\n      \"input_token\": {\"id\": " << probe.token << ", \"text\": ";
    rwkv_experiment::write_json_string(json, common_token_to_piece(ctx, probe.token, true));
    json << "},\n      \"layers\": [";
    for (int32_t layer = 0; layer < n_layer; ++layer) {
        const std::string tap = "rwkv.layer." + std::to_string(layer) + ".time.wkv_norm";
        const std::vector<std::string> taps = { tap, final_resid };
        llama_interp::activation_set native;
        auto native_run = evaluate(runtime, probe, taps, nullptr, {}, native);
        runtime.run();
        native_run.rethrow_if_failed();
        const std::vector<float> native_branch = capture(native, tap).data_f32;
        const float branch_norm = norm(native_branch);
        if (branch_norm == 0.0f) throw std::runtime_error("zero normalized WKV branch");
        const float epsilon = relative_scale * branch_norm;
        std::vector<ggml_fp16_t> plus(native_branch.size()), minus(native_branch.size());
        for (size_t i = 0; i < native_branch.size(); ++i) {
            plus[i] = ggml_fp32_to_fp16(relative_scale * native_branch[i]);
            minus[i] = ggml_fp32_to_fp16(-relative_scale * native_branch[i]);
        }
        llama_interp::activation_set plus_capture, minus_capture;
        auto plus_run = evaluate(runtime, probe, taps, &tap, std::move(plus), plus_capture);
        runtime.run();
        plus_run.rethrow_if_failed();
        auto minus_run = evaluate(runtime, probe, taps, &tap, std::move(minus), minus_capture);
        runtime.run();
        minus_run.rethrow_if_failed();
        std::vector<float> plus_logits(n_vocab), minus_logits(n_vocab), jvp(n_vocab);
        if (!llama_interp_rwkv_final_readout(ctx, capture(plus_capture, final_resid).data_f32.data(), 1, plus_logits.data()) ||
            !llama_interp_rwkv_final_readout(ctx, capture(minus_capture, final_resid).data_f32.data(), 1, minus_logits.data())) {
            throw std::runtime_error("final residual logit evaluation failed");
        }
        for (int32_t i = 0; i < n_vocab; ++i) jvp[i] = (plus_logits[i] - minus_logits[i]) / (2.0f * epsilon);
        result_layer = layer;
        result_branch_norm = branch_norm;
        result_epsilon = epsilon;
        result_plus_applied_norm = norm(subtract(capture(plus_capture, tap).data_f32, native_branch));
        result_minus_applied_norm = norm(subtract(capture(minus_capture, tap).data_f32, native_branch));
        result_jvp = std::move(jvp);
        result_top_positive = top_indices(result_jvp, 10, true);
        result_top_negative = top_indices(result_jvp, 10, false);
        result_tracked.clear();
        for (const int32_t token : tracked_token_ids) result_tracked.push_back(result_jvp[token]);
        tree.Fill();
        if (layer) json << ',';
        json << "\n        {\n          \"layer\": " << layer << ",\n          \"lenses\": [\n            {\n              \"lens\": \"time.wkv_norm.jvp.positive\",\n              \"result\": {\"top_logits\": [";
        for (size_t i = 0; i < result_top_positive.size(); ++i) {
            if (i) json << ',';
            const int32_t token = result_top_positive[i];
            json << "\n                {\"token_id\": " << token << ", \"token\": ";
            rwkv_experiment::write_json_string(json, common_token_to_piece(ctx, token, true));
            json << ", \"logit\": " << result_jvp[token] << '}';
        }
        json << "\n              ]}} ,\n            {\n              \"lens\": \"time.wkv_norm.jvp.negative\",\n              \"result\": {\"top_logits\": [";
        for (size_t i = 0; i < result_top_negative.size(); ++i) {
            if (i) json << ',';
            const int32_t token = result_top_negative[i];
            json << "\n                {\"token_id\": " << token << ", \"token\": ";
            rwkv_experiment::write_json_string(json, common_token_to_piece(ctx, token, true));
            json << ", \"logit\": " << -result_jvp[token] << '}';
        }
        json << "\n              ]}}\n          ]\n        }";
        std::printf("layer=%d positive=", layer);
        for (const int32_t token : result_top_positive) std::printf(" %s", common_token_to_piece(ctx, token, true).c_str());
        std::printf("\nlayer=%d negative=", layer);
        for (const int32_t token : result_top_negative) std::printf(" %s", common_token_to_piece(ctx, token, true).c_str());
        std::printf("\n");
        result_jvp.clear();
        result_tracked.clear();
        result_top_positive.clear();
        result_top_negative.clear();
        std::fprintf(stderr, "completed_layer=%d/%d\n", layer + 1, n_layer);
    }
    output.Write();
    output.Close();
    json << "\n      ]\n    }\n  ]\n}\n";
    if (!json) throw std::runtime_error("failed to write JSON output: " + json_output_path);
    std::printf("wrote=%s layers=%d prompt_tokens=%zu\n", output_path.c_str(), n_layer, tokens.size());
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
