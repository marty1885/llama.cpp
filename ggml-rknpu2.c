#include "ggml-rknpu2.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <rknn_api.h>
#include <rknn_matmul_api.h>

// Must be a power of 2, between 1 and 8. Controls the scale convertion from float to fixed point(8 bit)
// 1 means multiply by 1, therefore 0 bits are used for the integer part and is purely fractional
// 2 means multiply by 2, therefore 1 bit is used for the integer part and 7 bits for the fractional part
// 4 means multiply by 4, 2 bits integer, 6 bits fractional, etc..
#define GGML_RKNPU2_FP2INT_RANGE_MULTIPLIER 2
#define GGML_RKNPU2_FP2INT_WEIGHT_RANGE_MULTIPLIER 2

// Helper macros
#define ggrk_clamp(x, low, high) (x < low ? low : (x > high ? high : x))

struct ggml_rknpu2_data_pack
{
    void* ordered_data;
    int initialized;

    // RKNPU2 API structs
    rknn_tensor_mem* B;
};

struct ggml_rknpu2_matmul_kernel
{
    rknn_matmul_info matmul_info;
    rknn_matmul_ctx matmul_ctx;
    rknn_matmul_io_attr matmul_io_attr;

    rknn_tensor_mem* A;
    rknn_tensor_mem* C;
};

// Pool of RKNPU2 matmul kernels so we can reuse them
#define GGML_RKNPU2_MAX_MATMUL_KERNELS 16
static struct ggml_rknpu2_matmul_kernel matmul_kernels[GGML_RKNPU2_MAX_MATMUL_KERNELS];
static int matmul_kernels_count = 0;

static struct ggml_rknpu2_matmul_kernel* ggml_rknpu2_matmul_kernel_find(int m, int k, int n, rknn_tensor_type type)
{
    for(int i = 0; i < matmul_kernels_count; i++) {
        struct ggml_rknpu2_matmul_kernel* kernel = &matmul_kernels[i];
        if(kernel->matmul_info.M == m && kernel->matmul_info.K == k && kernel->matmul_info.N == n && kernel->matmul_info.type == type)
            return kernel;
    }
    return NULL;
}

static struct ggml_rknpu2_matmul_kernel* ggml_rknpu2_matmul_kernel_create(int m, int k, int n, rknn_tensor_type type)
{
    struct ggml_rknpu2_matmul_kernel* kernel = ggml_rknpu2_matmul_kernel_find(m, k, n, type);
    if(kernel != NULL)
        return kernel;

    GGML_ASSERT(matmul_kernels_count < GGML_RKNPU2_MAX_MATMUL_KERNELS);
    kernel = &matmul_kernels[matmul_kernels_count++];
    memset(kernel, 0, sizeof(struct ggml_rknpu2_matmul_kernel));

    kernel->matmul_info.M = m;
    kernel->matmul_info.K = k;
    kernel->matmul_info.N = n;
    kernel->matmul_info.type = type;
    kernel->matmul_info.native_layout = 1;
    kernel->matmul_info.perf_layout = 0;

    int ret = rknn_matmul_create(&kernel->matmul_ctx, &kernel->matmul_info, &kernel->matmul_io_attr);
    GGML_ASSERT(ret == 0);

    kernel->A = rknn_create_mem(kernel->matmul_ctx, kernel->matmul_io_attr.A.size);
    kernel->C = rknn_create_mem(kernel->matmul_ctx, kernel->matmul_io_attr.C.size);

    ret = rknn_matmul_set_io_mem(kernel->matmul_ctx, kernel->A, &kernel->matmul_io_attr.A);
    GGML_ASSERT(ret == 0);
    ret = rknn_matmul_set_io_mem(kernel->matmul_ctx, kernel->C, &kernel->matmul_io_attr.C);
    GGML_ASSERT(ret == 0);
    return kernel;
}

void ggml_rknpu2_init(void)
{
    // no-op
}

void ggml_rknpu2_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst, void * wdata, size_t wsize)
{
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    struct ggml_rknpu2_data_pack* pack = src0->extra;
    GGML_ASSERT(pack != NULL);

    const int64_t m = src1->ne[1];
    const int64_t k = src0->ne[0];
    const int64_t n = dst->ne[0];

    // First time called. Initialize RKNPU2 API structs
    if(pack->initialized == 0) {
        struct ggml_rknpu2_matmul_kernel* kernel = ggml_rknpu2_matmul_kernel_create(m, k, n, RKNN_TENSOR_INT8);

        pack->B = rknn_create_mem(kernel->matmul_ctx, kernel->matmul_io_attr.B.size);
        memcpy(pack->B->virt_addr, pack->ordered_data, kernel->matmul_io_attr.B.size);
        free(pack->ordered_data);
        pack->ordered_data = NULL;
        pack->initialized = 1;
    }

    struct ggml_rknpu2_matmul_kernel* kernel = ggml_rknpu2_matmul_kernel_find(m, k, n, RKNN_TENSOR_INT8);
    // GGML will switch batch size on the fly. So we need to create a new kernel if the batch size is different
    if(kernel == NULL)
        kernel = ggml_rknpu2_matmul_kernel_create(m, k, n, RKNN_TENSOR_INT8);

    float const* src1_data = src1->data;
    int8_t* A = kernel->A->virt_addr;
    for(size_t i = 0; i < m*k; i++) {
        A[i] = ggrk_clamp(src1_data[i]*128.f/GGML_RKNPU2_FP2INT_RANGE_MULTIPLIER, -128, 127);
    }

    int ret = rknn_matmul_set_io_mem(kernel->matmul_ctx, kernel->A, &kernel->matmul_io_attr.A);
    GGML_ASSERT(ret == 0);
    ret = rknn_matmul_set_io_mem(kernel->matmul_ctx, pack->B, &kernel->matmul_io_attr.B);
    GGML_ASSERT(ret == 0);
    ret = rknn_matmul_run(kernel->matmul_ctx);
    GGML_ASSERT(ret == 0);

    int32_t* C = kernel->C->virt_addr;
    float* dst_data = dst->data;
    const float div_factor = 1.f / (128.f * 128.f / GGML_RKNPU2_FP2INT_RANGE_MULTIPLIER / GGML_RKNPU2_FP2INT_WEIGHT_RANGE_MULTIPLIER);
    for(size_t i = 0; i < m*n; i++) {
        // Round about way to convert int32_t to float to workaround floating point precision issues
        dst_data[i] = (float)C[i] / div_factor;
    }
}

int ggml_rknpu2_can_mul_mat_b(const struct ggml_tensor * tensor)
{
    const int64_t k = tensor->ne[0];
    const int64_t n = tensor->ne[1];
    if(k > 4096)
        return 0;
    
    // k and n size must align to 32 bytes
    if(k % 32 != 0 || n % 32 != 0)
        return 0;

    // make sure the tensor has assosiated data
    if(tensor->backend != GGML_BACKEND_GPU)
        return 0;
    
    if(tensor->type != GGML_TYPE_Q8_0)
        return 0;
    
    // HACK: Only support square matrix for now
    if(k != n)
        return 0;

    return 1;
}

int ggml_rknpu2_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst)
{
    // TODO: Support RK3566/RK3568 NPU. This is only for RK3588
    if(ggml_rknpu2_can_mul_mat_b(src0) == 0)
        return 0;
    
    if(src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32)
        return 0;

    if(src0->extra == NULL)
        return 0;
    
    return 1;
}

static void ggml_rknpu2_transposed_to_native_int8(int8_t* restrict dst, int8_t* restrict src, size_t k, size_t n)
{
    GGML_ASSERT(k % 32 == 0 && n % 16 == 0 && k > 0 && n > 0);

    // RKNN native layout is (N/16, K/32, 16, 32)
    const size_t rknpu_strides[4] = {k / 32 * 16 * 32, 16 * 32, 32, 1};

    // Block copy 32x16 at a time to improve cache locality
    for(size_t j = 0; j < k/32; j++) {
        for(size_t i = 0; i < n/16; i++) {
            for(size_t jj = 0; jj < 32; jj++) {
                size_t partial_src_idx = (j*32+jj) * n + i*16;
                size_t partial_dst_idx = i * rknpu_strides[0] + j * rknpu_strides[1] + jj;

                for(size_t ii=0; ii < 16; ii++) {
                    size_t src_idx = partial_src_idx + ii;
                    size_t dst_idx = partial_dst_idx + ii * rknpu_strides[2];
                    dst[dst_idx] = src[src_idx];
                }
            }
        }
    }
}

void ggml_rknpu2_transform_tensor(void * data, struct ggml_tensor * tensor)
{
    const int64_t ne0 = tensor->ne[0];
    const int64_t ne1 = tensor->ne[1];
    const int64_t ne2 = tensor->ne[2];
    const int64_t ne3 = tensor->ne[3];
    const int64_t nb0 = tensor->nb[0];
    const int64_t nb1 = tensor->nb[1];

    const enum ggml_type type = tensor->type;

    GGML_ASSERT(ne2 == 1 && ne3 == 1 && ne1 > 0 && ne0 > 0);
    GGML_ASSERT(type == GGML_TYPE_Q8_0);
    GGML_ASSERT(ggml_is_quantized(type));

    ggml_type_traits_t traits = ggml_internal_get_type_traits(type);
    GGML_ASSERT(traits.to_float != NULL);

    const size_t nelements = ne0 * ne1;
    float* fdata = malloc(nelements * sizeof(float));
    
    traits.to_float(data, fdata, nelements);

    int8_t* u8data = malloc(nelements * sizeof(int8_t));
    for(int64_t i = 0; i < nelements; i++)
        u8data[i] = ggrk_clamp(fdata[i]*(128.f/GGML_RKNPU2_FP2INT_WEIGHT_RANGE_MULTIPLIER), -128, 127);
    free(fdata);

    int8_t* reorderd_data = malloc(ne0 * ne1 * sizeof(int8_t));
    ggml_rknpu2_transposed_to_native_int8(reorderd_data, u8data, ne1, ne0);
    free(u8data);

    struct ggml_rknpu2_data_pack* pack = malloc(sizeof(struct ggml_rknpu2_data_pack));
    memset(pack, 0, sizeof(struct ggml_rknpu2_data_pack));

    pack->ordered_data = reorderd_data;
    pack->initialized = 0;

    tensor->extra = pack;
}

void ggml_rknpu2_free_data(struct ggml_tensor * tensor)
{
    if(tensor->extra == NULL)
        return;

    struct ggml_rknpu2_data_pack* pack = tensor->extra;
    if(pack->ordered_data != NULL)
        free(pack->ordered_data);
    if(pack->initialized != 0) {
        // HACK: Grab a random kernel to release the memory
        GGML_ASSERT(matmul_kernels_count > 0);
        struct ggml_rknpu2_matmul_kernel* kernel = &matmul_kernels[0];
        rknn_destroy_mem(kernel->matmul_ctx, pack->B);
    }
    free(pack);
    tensor->extra = NULL;
}

void ggml_rknpu2_destroy(void)
{
    for(size_t i = 0; i < matmul_kernels_count; i++) {
        struct ggml_rknpu2_matmul_kernel* kernel = &matmul_kernels[i];
        rknn_destroy_mem(kernel->matmul_ctx, kernel->A);
        rknn_destroy_mem(kernel->matmul_ctx, kernel->C);

        rknn_matmul_destroy(kernel->matmul_ctx);
    }
}