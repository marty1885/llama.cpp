#include "rwkv_inverse_cache.hpp"

#include "ggml-backend.h"
#include "llama-model.h"

#include <lapacke.h>

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

std::vector<float> invert_projection(const ggml_tensor * weight) {
    if (!weight || weight->ne[0] != weight->ne[1]) throw std::runtime_error("RWKV projection must be square");
    const int32_t width = (int32_t) weight->ne[0];
    std::vector<uint8_t> raw(ggml_nbytes(weight));
    ggml_backend_tensor_get(weight, raw.data(), 0, raw.size());
    const ggml_type_traits * traits = ggml_get_type_traits(weight->type);
    if (!traits || (weight->type != GGML_TYPE_F32 && !traits->to_float)) {
        throw std::runtime_error("RWKV projection cannot be converted to F32");
    }
    std::vector<double> matrix((size_t) width * width);
    std::vector<float> row(width);
    for (int32_t output = 0; output < width; ++output) {
        const uint8_t * source = raw.data() + output * weight->nb[1];
        if (weight->type == GGML_TYPE_F32) std::memcpy(row.data(), source, row.size() * sizeof(float));
        else traits->to_float(source, row.data(), width);
        for (int32_t input = 0; input < width; ++input) matrix[(size_t) output * width + input] = row[input];
    }
    std::vector<lapack_int> pivots(width);
    if (LAPACKE_dgetrf(LAPACK_ROW_MAJOR, width, width, matrix.data(), width, pivots.data()) != 0 ||
        LAPACKE_dgetri(LAPACK_ROW_MAJOR, width, matrix.data(), width, pivots.data()) != 0) {
        throw std::runtime_error("OpenBLAS could not invert an RWKV projection");
    }
    std::vector<float> result((size_t) width * width);
    std::transform(matrix.begin(), matrix.end(), result.begin(), [](double value) { return (float) value; });
    return result;
}

void usage(const char * argv0) {
    std::fprintf(stderr, "usage: %s -m MODEL --output CACHE.root [-ngl N]\n", argv0);
}

} // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    llama_log_set(quiet_llama_logs, nullptr);
    std::string model_path, output_path;
    int n_gpu_layers = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) output_path = argv[++i];
        else if (std::strcmp(argv[i], "-ngl") == 0 && i + 1 < argc) n_gpu_layers = std::atoi(argv[++i]);
        else { usage(argv[0]); return 1; }
    }
    if (model_path.empty() || output_path.empty()) { usage(argv[0]); return 1; }

    ggml_backend_load_all();
    llama_backend_init();
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = n_gpu_layers;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), params);
    if (!model || model->arch != LLM_ARCH_RWKV7) throw std::runtime_error("an RWKV-7 model is required");
    const int32_t n_layer = model->hparams.n_layer();
    const int32_t width = (int32_t) model->layers[0].time_mix_key->ne[0];
    rwkv_inverse_cache::writer cache(output_path, model_path, n_layer, width);
    for (int32_t layer = 0; layer < n_layer; ++layer) {
        std::fprintf(stderr, "stage=invert layer=%d/%d projection=key\n", layer + 1, n_layer);
        cache.append(layer, "time_mix_key", invert_projection(model->layers[layer].time_mix_key));
        std::fprintf(stderr, "stage=invert layer=%d/%d projection=value\n", layer + 1, n_layer);
        cache.append(layer, "time_mix_value", invert_projection(model->layers[layer].time_mix_value));
        std::fprintf(stderr, "stage=invert layer=%d/%d projection=output\n", layer + 1, n_layer);
        cache.append(layer, "time_mix_output", invert_projection(model->layers[layer].time_mix_output));
    }
    cache.close();
    std::printf("wrote=%s layers=%d width=%d projections=3\n", output_path.c_str(), n_layer, width);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
