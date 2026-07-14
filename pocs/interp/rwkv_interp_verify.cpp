#include "interp.hpp"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void quiet_llama_logs(ggml_log_level, const char *, void *) {}

llama_interp::task<> prefill_capture(
        llama_interp::runtime & runtime,
        const llama_interp::rwkv_state & state,
        const std::vector<llama_token> & tokens,
        const std::vector<std::string> & taps,
        llama_interp::rwkv_state & output,
        llama_interp::activation_set & captures) {
    auto call = runtime.prefill_tokens(state, tokens);
    for (const std::string & tap : taps) call.capture_f32("^" + tap + "$", captures);
    output = co_await call;
}

const llama_interp_activation & require_capture(const llama_interp::activation_set & captures, const std::string & name) {
    const llama_interp_activation * result = nullptr;
    for (const auto & capture : captures) {
        if (capture.name != name || capture.data_f32.empty()) continue;
        if (result) throw std::runtime_error("duplicate capture: " + name);
        result = &capture;
    }
    if (!result) throw std::runtime_error("missing FP32 capture: " + name);
    return *result;
}

std::vector<float> native_logits(
        llama_context * ctx,
        const llama_interp::rwkv_state & initial,
        const std::vector<llama_token> & tokens) {
    if (!llama_interp_rwkv_state_import(ctx, 0, &initial)) {
        throw std::runtime_error("failed to import initial RWKV state for native decode");
    }
    llama_batch batch = llama_batch_init((int32_t) tokens.size(), 0, 1);
    common_batch_clear(batch);
    for (size_t i = 0; i < tokens.size(); ++i) {
        common_batch_add(batch, tokens[i], (llama_pos) i, { 0 }, true);
    }
    if (llama_decode(ctx, batch) != 0) throw std::runtime_error("native decode failed");
    llama_batch_free(batch);
    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
    const float * values = llama_get_logits_ith(ctx, (int32_t) tokens.size() - 1);
    if (!values) throw std::runtime_error("missing logits");
    return { values, values + llama_vocab_n_tokens(vocab) };
}

llama_token greedy_token(const std::vector<float> & logits) {
    return (llama_token) std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()));
}

float max_abs_difference(const std::vector<float> & left, const std::vector<float> & right) {
    if (left.size() != right.size()) throw std::runtime_error("vector size mismatch");
    float result = 0.0f;
    for (size_t i = 0; i < left.size(); ++i) result = std::max(result, std::abs(left[i] - right[i]));
    return result;
}

void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s -m MODEL [-p PROMPT] [--tap FULL_NAME] [--tolerance E] [-ngl N]\n"
        "\n"
        "Verifies production-graph capture snapshots, RWKV state handoff, and final-residual\n"
        "readout. A capture cannot be read unless the core graph registered a ggml_dup snapshot.\n",
        argv0);
}

} // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    llama_log_set(quiet_llama_logs, nullptr);
    std::string model_path;
    std::string prompt = "The capital of France is";
    std::string tap;
    int n_gpu_layers = 0;
    float tolerance = 1e-4f;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (std::strcmp(argv[i], "-p") == 0 && i + 1 < argc) prompt = argv[++i];
        else if (std::strcmp(argv[i], "--tap") == 0 && i + 1 < argc) tap = argv[++i];
        else if (std::strcmp(argv[i], "--tolerance") == 0 && i + 1 < argc) tolerance = std::strtof(argv[++i], nullptr);
        else if (std::strcmp(argv[i], "-ngl") == 0 && i + 1 < argc) n_gpu_layers = std::atoi(argv[++i]);
        else {
            usage(argv[0]);
            return 1;
        }
    }
    if (model_path.empty() || tolerance <= 0.0f) {
        usage(argv[0]);
        return 1;
    }

    ggml_backend_load_all();
    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) throw std::runtime_error("failed to load model");
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const std::vector<llama_token> tokens = common_tokenize(vocab, prompt, false, true);
    if (tokens.size() < 2) throw std::runtime_error("prompt needs at least two tokens");
    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = std::max(128, (int) tokens.size() + 8);
    context_params.n_batch = context_params.n_ctx;
    context_params.n_ubatch = context_params.n_ctx;
    llama_context * ctx = llama_init_from_model(model, context_params);
    if (!ctx) throw std::runtime_error("failed to create context");

    llama_interp::runtime runtime(ctx, 1);
    const llama_interp::rwkv_state initial = runtime.make_state();
    const std::string final_residual = "rwkv.layer." + std::to_string(initial.n_layer - 1) + ".resid.out";
    if (tap.empty()) tap = final_residual;
    std::vector<std::string> taps = { tap };
    if (tap != final_residual) taps.push_back(final_residual);

    llama_interp::activation_set full_captures;
    llama_interp::rwkv_state full_state;
    auto full_call = prefill_capture(runtime, initial, tokens, taps, full_state, full_captures);
    runtime.run();
    full_call.rethrow_if_failed();
    const auto & captured_final = require_capture(full_captures, final_residual);
    const std::vector<float> native_full_logits = native_logits(ctx, initial, tokens);
    std::vector<float> readout_logits(native_full_logits.size());
    if (!llama_interp_rwkv_final_readout(ctx, captured_final.data_f32.data(), 1, readout_logits.data())) {
        throw std::runtime_error("final residual readout failed");
    }
    const float final_readout_error = max_abs_difference(native_full_logits, readout_logits);

    const std::vector<llama_token> prefix(tokens.begin(), tokens.end() - 1);
    const std::vector<llama_token> suffix(tokens.end() - 1, tokens.end());
    llama_interp::rwkv_state prefix_state;
    llama_interp::activation_set prefix_captures;
    auto prefix_call = prefill_capture(runtime, initial, prefix, {}, prefix_state, prefix_captures);
    runtime.run();
    prefix_call.rethrow_if_failed();
    llama_interp::activation_set split_captures;
    llama_interp::rwkv_state split_state;
    auto split_call = prefill_capture(runtime, prefix_state, suffix, taps, split_state, split_captures);
    runtime.run();
    split_call.rethrow_if_failed();
    const auto & split_final = require_capture(split_captures, final_residual);
    std::vector<float> split_readout_logits(native_full_logits.size());
    if (!llama_interp_rwkv_final_readout(ctx, split_final.data_f32.data(), 1, split_readout_logits.data())) {
        throw std::runtime_error("split final residual readout failed");
    }
    const float state_handoff_error = max_abs_difference(readout_logits, split_readout_logits);
    const float capture_handoff_error = max_abs_difference(
        require_capture(full_captures, tap).data_f32, require_capture(split_captures, tap).data_f32);
    if (final_readout_error > tolerance || state_handoff_error > tolerance || capture_handoff_error > tolerance ||
        full_state.pos != split_state.pos || greedy_token(readout_logits) != full_state.next_token) {
        throw std::runtime_error("RWKV instrumentation verification failed");
    }

    std::printf("verified_tap=%s backend=%s final_readout_max_abs_error=%g state_handoff_logit_max_abs_error=%g capture_handoff_max_abs_error=%g\n",
        tap.c_str(), require_capture(full_captures, tap).backend.c_str(), final_readout_error, state_handoff_error, capture_handoff_error);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
