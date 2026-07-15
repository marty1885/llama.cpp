#include "interp.hpp"

#include "llama-model.h"

#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void quiet_llama_logs(ggml_log_level, const char *, void *) {}

llama_interp::task<> generate(
        llama_interp::runtime & runtime,
        const std::vector<llama_token> & prompt,
        int32_t n_generate,
        std::vector<llama_token> & output) {
    llama_interp::rwkv_state state = runtime.make_state();
    for (const llama_token token : prompt) state = co_await runtime.prefill_tokens(state, { token });
    for (int32_t i = 0; i < n_generate; ++i) {
        if (!state.has_next) throw std::runtime_error("generation state has no next token");
        const llama_token token = state.next_token;
        output.push_back(token);
        state = co_await runtime.prefill_tokens(state, { token });
    }
}

void usage(const char * argv0) {
    std::fprintf(stderr, "usage: %s -m MODEL --prompt-file FILE --generate N [-ngl N]\n", argv0);
}

} // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    llama_log_set(quiet_llama_logs, nullptr);
    std::string model_path, prompt_path;
    int32_t n_generate = 0;
    int n_gpu_layers = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (std::strcmp(argv[i], "--prompt-file") == 0 && i + 1 < argc) prompt_path = argv[++i];
        else if (std::strcmp(argv[i], "--generate") == 0 && i + 1 < argc) n_generate = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-ngl") == 0 && i + 1 < argc) n_gpu_layers = std::atoi(argv[++i]);
        else { usage(argv[0]); return 1; }
    }
    if (model_path.empty() || prompt_path.empty() || n_generate <= 0) { usage(argv[0]); return 1; }
    std::ifstream input(prompt_path);
    if (!input) throw std::runtime_error("failed to open prompt file: " + prompt_path);
    const std::string prompt((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    ggml_backend_load_all();
    llama_backend_init();
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = n_gpu_layers;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), params);
    if (!model || model->arch != LLM_ARCH_RWKV7) throw std::runtime_error("an RWKV-7 model is required");
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const std::vector<llama_token> tokens = common_tokenize(vocab, prompt, false, true);
    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = std::max(256, (int) tokens.size() + n_generate + 8);
    context_params.n_batch = context_params.n_ctx;
    context_params.n_ubatch = context_params.n_ctx;
    llama_context * ctx = llama_init_from_model(model, context_params);
    if (!ctx) throw std::runtime_error("failed to create context");
    llama_interp::runtime runtime(ctx, 1);
    std::vector<llama_token> generated;
    auto run = generate(runtime, tokens, n_generate, generated);
    runtime.run();
    run.rethrow_if_failed();
    for (const llama_token token : generated) std::fputs(common_token_to_piece(ctx, token, true).c_str(), stdout);
    std::fputc('\n', stdout);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
