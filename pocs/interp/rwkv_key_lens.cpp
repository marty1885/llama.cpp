#include "rwkv_experiment.hpp"
#include "rwkv_inverse_cache.hpp"

#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "llama-model.h"

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void quiet_llama_logs(ggml_log_level, const char *, void *) {}

class layerwise_bank {
public:
    layerwise_bank(const llama_model & model, int32_t width, int32_t n_layer, std::vector<float> values) :
            metadata(ggml_tensor_overhead() * 2), width(width), n_layer(n_layer) {
        const size_t expected = (size_t) width * width * n_layer;
        if (values.size() != expected) {
            throw std::runtime_error("invalid aggregated inverse size");
        }
        ggml_init_params params = {
            /*.mem_size   =*/ metadata.size(),
            /*.mem_buffer =*/ metadata.data(),
            /*.no_alloc   =*/ true,
        };
        ctx.reset(ggml_init(params));
        if (!ctx) {
            throw std::runtime_error("failed to create layerwise inverse context");
        }
        projection = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, width, width, n_layer, 1);
        aggregate = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, width, 1, n_layer, 1);
        const ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(model.dev_output());
        buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft));
        if (!buffer) {
            throw std::runtime_error("failed to allocate persistent layerwise inverse tensors");
        }
        ggml_backend_tensor_set(projection, values.data(), 0, values.size() * sizeof(float));
    }

    llama_interp_layerwise_readout_spec spec() const {
        return { "key_lens.logits", projection, aggregate };
    }

    const char * backend_name() const {
        return ggml_backend_dev_name(ggml_backend_buft_get_device(ggml_backend_buffer_get_type(buffer.get())));
    }

private:
    std::vector<uint8_t> metadata;
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buffer;
    int32_t width;
    int32_t n_layer;
    ggml_tensor * projection = nullptr;
    ggml_tensor * aggregate = nullptr;
};

llama_interp::task<> capture_layerwise_logits(
        llama_interp::runtime & runtime,
        const llama_interp::rwkv_state & initial,
        const std::vector<llama_token> & tokens,
        const llama_interp_layerwise_readout_spec & spec,
        std::vector<rwkv_experiment::token_capture> & output) {
    llama_interp::rwkv_state state = initial;
    for (size_t position = 0; position < tokens.size(); ++position) {
        llama_interp::activation_set activations;
        auto call = runtime.prefill_tokens(state, { tokens[position] });
        call.layerwise_readout(spec);
        call.capture_f32("^rwkv.layerwise.key_lens.logits$", activations);
        state = co_await call;
        output.push_back({ (int32_t) position, tokens[position], std::move(activations) });
    }
}

void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s -m MODEL -p PROMPT --inverse-cache CACHE.root --output FILE [--top-n N] [-ngl N]\n"
        "\n"
        "Loads cached inverse RWKV-7 W_K matrices, uploads the [K,M,L,1] bank once,\n"
        "and writes graph-produced layerwise lens logits as JSON.\n",
        argv0);
}

} // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    llama_log_set(quiet_llama_logs, nullptr);

    std::string model_path, inverse_cache_path;
    std::string prompt;
    std::string output_path;
    int32_t top_n = 20;
    int n_gpu_layers = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (std::strcmp(argv[i], "-p") == 0 && i + 1 < argc) prompt = argv[++i];
        else if (std::strcmp(argv[i], "--inverse-cache") == 0 && i + 1 < argc) inverse_cache_path = argv[++i];
        else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) output_path = argv[++i];
        else if (std::strcmp(argv[i], "--top-n") == 0 && i + 1 < argc) top_n = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-ngl") == 0 && i + 1 < argc) n_gpu_layers = std::atoi(argv[++i]);
        else {
            usage(argv[0]);
            return 1;
        }
    }
    if (model_path.empty() || prompt.empty() || output_path.empty() || inverse_cache_path.empty() || top_n <= 0) {
        usage(argv[0]);
        return 1;
    }

    ggml_backend_load_all();
    llama_backend_init();
    std::fprintf(stderr, "stage=load_model path=%s gpu_layers=%d\n", model_path.c_str(), n_gpu_layers);
    std::fflush(stderr);
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model || model->arch != LLM_ARCH_RWKV7) {
        throw std::runtime_error("an RWKV-7 model is required");
    }

    const int32_t n_layer = model->hparams.n_layer();
    const int32_t width = (int32_t) model->layers[0].time_mix_key->ne[0];
    rwkv_inverse_cache::reader cache(inverse_cache_path, model_path, n_layer, width);
    std::fprintf(stderr, "stage=load_inverse_bank layers=%d width=%d cache=%s\n", n_layer, width, inverse_cache_path.c_str());
    std::vector<float> inverse_bank;
    inverse_bank.reserve((size_t) width * width * n_layer);
    for (int32_t layer = 0; layer < n_layer; ++layer) {
        const std::vector<float> inverse = cache.load(layer, "time_mix_key");
        inverse_bank.insert(inverse_bank.end(), inverse.begin(), inverse.end());
    }
    std::fprintf(stderr, "stage=upload_inverse_bank bytes=%zu\n", inverse_bank.size() * sizeof(float));
    std::fflush(stderr);
    layerwise_bank bank(*model, width, n_layer, std::move(inverse_bank));
    std::fprintf(stderr, "inverse_bank_shape=[%d,%d,%d,1] backend=%s\n", width, width, n_layer, bank.backend_name());
    std::fflush(stderr);

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
    std::vector<rwkv_experiment::token_capture> captures;
    std::fprintf(stderr, "stage=decode tokens=%zu layerwise_batch=1\n", tokens.size());
    std::fflush(stderr);
    auto run = capture_layerwise_logits(runtime, runtime.make_state(), tokens, bank.spec(), captures);
    runtime.run();
    run.rethrow_if_failed();
    std::fprintf(stderr, "stage=decode status=done captures=%zu\n", captures.size());
    std::fflush(stderr);

    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    rwkv_experiment::logit_document document;
    document.experiment = "rwkv-key-pseudoinverse-logit-lens";
    document.model = model_path;
    document.prompt = prompt;
    document.metadata = {
        { "projection_ggml_shape", "[K,M,L,1]=[" + std::to_string(width) + "," + std::to_string(width) + "," + std::to_string(n_layer) + ",1]" },
        { "aggregate_ggml_shape", "[K,B,L,1]=[" + std::to_string(width) + ",1," + std::to_string(n_layer) + ",1]" },
        { "projection_backend", bank.backend_name() },
        { "inverse_cache", inverse_cache_path },
    };
    document.outputs.reserve(captures.size() * n_layer);
    for (const auto & captured : captures) {
        const auto & logits = rwkv_experiment::require_capture(captured.activations, "rwkv.layerwise.key_lens.logits");
        if (logits.shape[0] != n_vocab || logits.shape[1] != 1 || logits.shape[2] != n_layer || logits.shape[3] != 1) {
            throw std::runtime_error("malformed layerwise logits capture");
        }
        for (int32_t layer = 0; layer < n_layer; ++layer) {
            const float * row = logits.data_f32.data() + (size_t) layer * n_vocab;
            document.outputs.push_back({
                "rwkv.layer." + std::to_string(layer) + ".time.k0.pinv",
                layer,
                captured.position,
                captured.input_token,
                { row, row + n_vocab },
            });
        }
    }

    std::fprintf(stderr, "stage=write_json rows=%zu path=%s\n", document.outputs.size(), output_path.c_str());
    std::fflush(stderr);
    rwkv_experiment::write_top_logits_json(output_path, ctx, document, top_n);
    std::printf("wrote=%s tokens=%zu layers=%d outputs=%zu\n", output_path.c_str(), tokens.size(), n_layer, document.outputs.size());
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
