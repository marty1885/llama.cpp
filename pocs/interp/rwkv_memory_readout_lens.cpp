#include "rwkv_experiment.hpp"

#include "ggml-backend.h"
#include "llama-model.h"

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void quiet_llama_logs(ggml_log_level, const char *, void *) {}

llama_interp::task<> capture_readouts(
        llama_interp::runtime & runtime,
        const llama_interp::rwkv_state & initial,
        const std::vector<llama_token> & tokens,
        int32_t n_generate,
        const std::vector<std::string> & taps,
        std::vector<rwkv_experiment::token_capture> & output,
        std::vector<llama_token> & generated) {
    llama_interp::rwkv_state state = initial;
    for (size_t position = 0; position < tokens.size(); ++position) {
        llama_interp::activation_set activations;
        auto call = runtime.prefill_tokens(state, { tokens[position] });
        for (const std::string & tap : taps) {
            call.capture_f32("^" + tap + "$", activations);
        }
        state = co_await call;
        output.push_back({ (int32_t) position, tokens[position], std::move(activations) });
    }
    for (int32_t step = 0; step < n_generate; ++step) {
        if (!state.has_next) throw std::runtime_error("generation state has no next token");
        const llama_token token = state.next_token;
        llama_interp::activation_set activations;
        auto call = runtime.prefill_tokens(state, { token });
        for (const std::string & tap : taps) {
            call.capture_f32("^" + tap + "$", activations);
        }
        state = co_await call;
        generated.push_back(token);
        output.push_back({ (int32_t) tokens.size() + step, token, std::move(activations) });
    }
}

void append_lenses(
        llama_context * ctx,
        const std::vector<float> & rows,
        const std::vector<std::string> & names,
        const std::vector<rwkv_experiment::token_capture> & captures,
        int32_t n_embd,
        int32_t n_vocab,
        rwkv_experiment::logit_document & document,
        int32_t fixed_layer = -1) {
    const size_t n_rows = names.size();
    static constexpr size_t k_readout_rows = 64;
    for (size_t first = 0; first < n_rows; first += k_readout_rows) {
        const size_t count = std::min(k_readout_rows, n_rows - first);
        std::vector<float> logits((size_t) n_vocab * count);
        if (!llama_interp_rwkv_final_readout(
                ctx, rows.data() + first * n_embd, (uint32_t) count, logits.data())) {
            throw std::runtime_error("batched final readout failed");
        }
        for (size_t offset = 0; offset < count; ++offset) {
            const size_t row = first + offset;
            const size_t position = fixed_layer >= 0 ? row : row / (names.size() / captures.size());
            const int32_t layer = fixed_layer >= 0 ? fixed_layer : (int32_t) (row % (names.size() / captures.size()));
            document.outputs.push_back({
                names[row], layer, captures[position].position, captures[position].input_token,
                { logits.begin() + offset * n_vocab, logits.begin() + (offset + 1) * n_vocab },
            });
        }
    }
}

void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s -m MODEL -p PROMPT --output FILE [--generate N] [--top-n N] [-ngl N]\n"
        "\n"
        "Lenses direct RWKV-7 time.xv, time.pre_output, and time.out activations.\n",
        argv0);
}

} // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    llama_log_set(quiet_llama_logs, nullptr);

    std::string model_path, prompt, output_path;
    int32_t top_n = 20;
    int32_t n_generate = 0;
    int n_gpu_layers = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (std::strcmp(argv[i], "-p") == 0 && i + 1 < argc) prompt = argv[++i];
        else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) output_path = argv[++i];
        else if (std::strcmp(argv[i], "--generate") == 0 && i + 1 < argc) n_generate = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--top-n") == 0 && i + 1 < argc) top_n = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-ngl") == 0 && i + 1 < argc) n_gpu_layers = std::atoi(argv[++i]);
        else { usage(argv[0]); return 1; }
    }
    if (model_path.empty() || prompt.empty() || output_path.empty() || top_n <= 0 || n_generate < 0) {
        usage(argv[0]);
        return 1;
    }

    ggml_backend_load_all();
    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model || model->arch != LLM_ARCH_RWKV7) {
        throw std::runtime_error("an RWKV-7 model is required");
    }
    const int32_t n_layer = model->hparams.n_layer();
    const int32_t n_embd = (int32_t) model->layers[0].time_mix_value->ne[0];

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
    std::vector<std::string> xv_taps, pre_output_taps, out_taps, taps;
    for (int32_t layer = 0; layer < n_layer; ++layer) {
        const std::string prefix = "rwkv.layer." + std::to_string(layer) + ".time.";
        xv_taps.push_back(prefix + "xv");
        pre_output_taps.push_back(prefix + "pre_output");
        out_taps.push_back(prefix + "out");
    }
    taps = xv_taps;
    taps.insert(taps.end(), pre_output_taps.begin(), pre_output_taps.end());
    taps.insert(taps.end(), out_taps.begin(), out_taps.end());
    std::vector<rwkv_experiment::token_capture> captures;
    std::vector<llama_token> generated;
    std::fprintf(stderr, "stage=capture prompt_tokens=%zu generate=%d layers=%d\n", tokens.size(), n_generate, n_layer);
    auto run = capture_readouts(runtime, runtime.make_state(), tokens, n_generate, taps, captures, generated);
    runtime.run();
    run.rethrow_if_failed();

    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    std::string generated_text;
    for (llama_token token : generated) generated_text += common_token_to_piece(ctx, token, true);
    rwkv_experiment::logit_document document;
    document.experiment = "rwkv-value-input-and-pre-output-logit-lens";
    document.model = model_path;
    document.prompt = prompt;
    document.metadata = {
        { "generation", "greedy production decode, generated tokens are appended to inputs" },
        { "generated_text", generated_text },
        { "xv_tap", "rwkv.layer.<L>.time.xv" },
        { "pre_output_tap", "rwkv.layer.<L>.time.pre_output" },
        { "out_tap", "rwkv.layer.<L>.time.out" },
        { "readout", "native_final_cvec_norm_output_head" },
    };
    document.outputs.reserve(captures.size() * n_layer * 3);

    std::vector<float> rows;
    std::vector<std::string> names;
    rows.reserve(captures.size() * n_layer * n_embd);
    names.reserve(captures.size() * n_layer);
    for (const auto & captured : captures) {
        for (int32_t layer = 0; layer < n_layer; ++layer) {
            const auto & activation = rwkv_experiment::require_capture(captured.activations, xv_taps[layer]);
            if (activation.data_f32.size() != (size_t) n_embd) throw std::runtime_error("unexpected xv tap width");
            rows.insert(rows.end(), activation.data_f32.begin(), activation.data_f32.end());
            names.push_back(xv_taps[layer] + ".lens");
        }
    }
    append_lenses(ctx, rows, names, captures, n_embd, n_vocab, document);

    rows.clear();
    names.clear();
    for (const auto & captured : captures) {
        for (int32_t layer = 0; layer < n_layer; ++layer) {
            const auto & activation = rwkv_experiment::require_capture(captured.activations, out_taps[layer]);
            if (activation.data_f32.size() != (size_t) n_embd) throw std::runtime_error("unexpected output tap width");
            rows.insert(rows.end(), activation.data_f32.begin(), activation.data_f32.end());
            names.push_back(out_taps[layer] + ".lens");
        }
    }
    append_lenses(ctx, rows, names, captures, n_embd, n_vocab, document);

    rows.clear();
    names.clear();
    for (const auto & captured : captures) {
        for (int32_t layer = 0; layer < n_layer; ++layer) {
            const auto & activation = rwkv_experiment::require_capture(captured.activations, pre_output_taps[layer]);
            if (activation.data_f32.size() != (size_t) n_embd) throw std::runtime_error("unexpected pre-output tap width");
            rows.insert(rows.end(), activation.data_f32.begin(), activation.data_f32.end());
            names.push_back(pre_output_taps[layer] + ".lens");
        }
    }
    append_lenses(ctx, rows, names, captures, n_embd, n_vocab, document);

    rwkv_experiment::write_top_logits_json(output_path, ctx, document, top_n);
    std::printf("wrote=%s prompt_tokens=%zu generated_tokens=%zu generated_text=%s layers=%d outputs=%zu\n",
                output_path.c_str(), tokens.size(), generated.size(), generated_text.c_str(), n_layer, document.outputs.size());
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
