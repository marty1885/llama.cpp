#include "ggml.h"
#include "ggml-cpu.h"
#include <string.h>
#include <stdio.h>
#include <vector>

int main(void) {
    const int D = 512;
    const int N = 4;
    std::vector<float> input_vec(D * N);
    for(int i = 0; i < D * N; i++) {
        input_vec[i] = (float)i;
    }
    std::vector<int> pos(N);
    for(int i = 0; i < N; i++) {
        pos[i] = i+1;
    }



    // Arbitrary amount of allowed memory as the example is small
    size_t ctx_size = 5 * 1024 * 1024; // 5 MB

    // Allocate `ggml_context` to store tensor data
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(params);

    // 2. Create tensors and set data
    struct ggml_tensor * tensor_a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, 1, N);
    struct ggml_tensor * tensor_b = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, N, 1);
    memcpy(tensor_a->data, input_vec.data(), ggml_nbytes(tensor_a));
    memcpy(tensor_b->data, pos.data(), ggml_nbytes(tensor_b));
    printf("%zu, %zu\n", tensor_a->ne[2], tensor_b->ne[0]);


    struct ggml_cgraph * gf = ggml_new_graph(ctx);

    struct ggml_tensor * result = ggml_rope(ctx, tensor_a, tensor_b, D/4, GGML_ROPE_TYPE_NEOX);

    // Mark the "result" tensor to be computed
    ggml_build_forward_expand(gf, result);

    // 4. Run the computation
    int n_threads = 1; // Optional: number of threads to perform some operations with multi-threading
    ggml_graph_compute_with_ctx(ctx, gf, n_threads);

    // 5. Retrieve results (output tensors)
    float * result_data = (float *) result->data;
    for(size_t i = 0; i < N; i++) {
        printf("vector %zu:\n", i);
        for(size_t j = 0; j < D; j++) {
            printf("%f ", result_data[i * D + j]);
        }
        printf("\n");
    }

    // 6. Free memory and exit
    ggml_free(ctx);
    return 0;
}

