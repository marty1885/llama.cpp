#include "rwkv_experiment.hpp"

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

void quiet_llama_logs(ggml_log_level, const char *, void *) {}

llama_interp::task<> capture_key_inputs(
        llama_interp::runtime & runtime,
        const llama_interp::rwkv_state & initial,
        const std::vector<llama_token> & tokens,
        const std::vector<std::string> & taps,
        std::vector<rwkv_experiment::token_capture> & output) {
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
}

void append_lens_rows(
        llama_context * ctx,
        const rwkv_experiment::token_capture & captured,
        const std::vector<std::string> & taps,
        const std::string & suffix,
        int32_t n_embd,
        int32_t n_vocab,
        rwkv_experiment::logit_document & document) {
    std::vector<float> rows((size_t) n_embd * taps.size());
    for (size_t layer = 0; layer < taps.size(); ++layer) {
        const auto & activation = rwkv_experiment::require_capture(captured.activations, taps[layer]);
        if (activation.data_f32.size() != (size_t) n_embd) {
            throw std::runtime_error("unexpected tap width: " + taps[layer]);
        }
        std::copy(activation.data_f32.begin(), activation.data_f32.end(), rows.begin() + layer * n_embd);
    }

    std::vector<float> logits((size_t) n_vocab * taps.size());
    if (!llama_interp_rwkv_final_readout(ctx, rows.data(), (uint32_t) taps.size(), logits.data())) {
        throw std::runtime_error("batched final readout failed");
    }
    for (size_t layer = 0; layer < taps.size(); ++layer) {
        document.outputs.push_back({
            taps[layer] + suffix,
            (int32_t) layer,
            captured.position,
            captured.input_token,
            { logits.begin() + layer * n_vocab, logits.begin() + (layer + 1) * n_vocab },
        });
    }
}

void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s -m MODEL -p PROMPT --output FILE [--top-n N] [-ngl N]\n"
        "\n"
        "Captures RWKV-7 att.norm and time.xk at every layer and writes final-readout\n"
        "logit lenses for both activations to JSON.\n",
        argv0);
}

} // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    llama_log_set(quiet_llama_logs, nullptr);

    std::string model_path;
    std::string prompt;
    std::string output_path;
    int32_t top_n = 20;
    int n_gpu_layers = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (std::strcmp(argv[i], "-p") == 0 && i + 1 < argc) prompt = argv[++i];
        else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) output_path = argv[++i];
        else if (std::strcmp(argv[i], "--top-n") == 0 && i + 1 < argc) top_n = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-ngl") == 0 && i + 1 < argc) n_gpu_layers = std::atoi(argv[++i]);
        else {
            usage(argv[0]);
            return 1;
        }
    }
    if (model_path.empty() || prompt.empty() || output_path.empty() || top_n <= 0) {
        usage(argv[0]);
        return 1;
    }

    ggml_backend_load_all();
    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        throw std::runtime_error("failed to load model");
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const std::vector<llama_token> tokens = common_tokenize(vocab, prompt, false, true);
    if (tokens.empty()) {
        throw std::runtime_error("prompt tokenized to zero tokens");
    }

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = std::max(128, (int) tokens.size() + 8);
    context_params.n_batch = context_params.n_ctx;
    context_params.n_ubatch = context_params.n_ctx;
    llama_context * ctx = llama_init_from_model(model, context_params);
    if (!ctx) {
        throw std::runtime_error("failed to create context");
    }

    llama_interp::runtime runtime(ctx, 1);
    const llama_interp::rwkv_state initial = runtime.make_state();
    std::vector<std::string> att_norm_taps;
    std::vector<std::string> xk_taps;
    for (uint32_t layer = 0; layer < initial.n_layer; ++layer) {
        const std::string prefix = "rwkv.layer." + std::to_string(layer);
        att_norm_taps.push_back(prefix + ".att.norm");
        xk_taps.push_back(prefix + ".time.xk");
    }
    std::vector<std::string> taps = att_norm_taps;
    taps.insert(taps.end(), xk_taps.begin(), xk_taps.end());
    std::vector<rwkv_experiment::token_capture> captures;
    std::fprintf(stderr, "stage=capture tokens=%zu layers=%u\n", tokens.size(), initial.n_layer);
    std::fflush(stderr);
    auto run = capture_key_inputs(runtime, initial, tokens, taps, captures);
    runtime.run();
    run.rethrow_if_failed();

    const int32_t n_embd = (int32_t) initial.n_embd_r / 2;
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    rwkv_experiment::logit_document document;
    document.experiment = "rwkv-att-norm-and-key-input-logit-lens";
    document.model = model_path;
    document.prompt = prompt;
    document.metadata = {
        { "att_norm_tap", "rwkv.layer.<L>.att.norm" },
        { "xk_tap", "rwkv.layer.<L>.time.xk" },
        { "readout", "native_final_cvec_norm_output_head" },
    };
    document.outputs.reserve(captures.size() * initial.n_layer * 2);
    for (const auto & captured : captures) {
        append_lens_rows(ctx, captured, att_norm_taps, ".lens", n_embd, n_vocab, document);
        append_lens_rows(ctx, captured, xk_taps, ".lens", n_embd, n_vocab, document);
    }

    rwkv_experiment::write_top_logits_json(output_path, ctx, document, top_n);
    std::printf("wrote=%s tokens=%zu layers=%u outputs=%zu\n", output_path.c_str(), tokens.size(), initial.n_layer, document.outputs.size());
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
