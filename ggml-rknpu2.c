#include "ggml-rknpu2.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rknn_api.h"
#include "rknn_matmul_api.h"

#include <arm_neon.h>


inline uint16_t arm_fp32_to_fp16(float x) {
  float32x4_t tmp = vld1q_dup_f32(&x);
  float16_t res = vget_lane_f16(vcvt_f16_f32(tmp), 0);
  return *(uint16_t *)(&res);
}

rknn_tensor_type rknpu2_matmul_type_to_rknn_type_input(rknn_matmul_type type)
{
    switch(type) {
        case RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32:
            return RKNN_TENSOR_FLOAT16;
        case RKNN_INT8_MM_INT8_TO_INT32:
            return RKNN_TENSOR_INT8;
        case RKNN_INT4_MM_INT4_TO_INT16:
            return RKNN_TENSOR_INT4;
        default:
            GGML_ASSERT(0);
    }
}

rknn_tensor_type rknpu2_matmul_type_to_rknn_type_output(rknn_matmul_type type)
{
    switch(type) {
        case RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32:
            return RKNN_TENSOR_FLOAT32;
        case RKNN_INT8_MM_INT8_TO_INT32:
            return RKNN_TENSOR_INT32;
        case RKNN_INT4_MM_INT4_TO_INT16:
            return RKNN_TENSOR_INT16;
        default:
            GGML_ASSERT(0);
    }
}

rknn_matmul_type rknpu2_matmul_type_from_rknn_type(rknn_tensor_type type)
{
    switch(type) {
        case RKNN_TENSOR_FLOAT16:
            return RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
        case RKNN_TENSOR_INT8:
            return RKNN_INT8_MM_INT8_TO_INT32;
        case RKNN_TENSOR_INT4:
            return RKNN_INT4_MM_INT4_TO_INT16;
        default:
            GGML_ASSERT(0);
    }
}

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

static struct ggml_rknpu2_matmul_kernel *
ggml_rknpu2_matmul_kernel_find(int m, int k, int n, rknn_tensor_type type) {
  for (int i = 0; i < matmul_kernels_count; i++) {
    struct ggml_rknpu2_matmul_kernel *kernel = &matmul_kernels[i];
    if (kernel->matmul_info.M == m && kernel->matmul_info.K == k &&
        kernel->matmul_info.N == n &&
        rknpu2_matmul_type_to_rknn_type_input(kernel->matmul_info.type) == type)
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
    kernel->matmul_info.type = rknpu2_matmul_type_from_rknn_type(type);
    kernel->matmul_info.B_layout = 1; // B use native layout (weight)
    kernel->matmul_info.AC_layout = 0; // A and C use original layout (intermediate)

    int ret = rknn_matmul_create(&kernel->matmul_ctx, &kernel->matmul_info, &kernel->matmul_io_attr);
    GGML_ASSERT(ret == 0);
    rknn_matmul_set_core_mask(kernel->matmul_ctx, RKNN_NPU_CORE_1);
    printf("Created RKNPU2 matmul kernel: src0(%d, %d) x src1(%d, %d) = dst(%d, %d)\n", m, k, k, n, m, n);

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
        struct ggml_rknpu2_matmul_kernel* kernel = ggml_rknpu2_matmul_kernel_create(m, k, n, RKNN_TENSOR_FLOAT16);

        pack->B = rknn_create_mem(kernel->matmul_ctx, kernel->matmul_io_attr.B.size);
        memcpy(pack->B->virt_addr, pack->ordered_data, kernel->matmul_io_attr.B.size);
        free(pack->ordered_data);
        pack->ordered_data = NULL;
        pack->initialized = 1;
    }

    struct ggml_rknpu2_matmul_kernel* kernel = ggml_rknpu2_matmul_kernel_find(m, k, n, RKNN_TENSOR_FLOAT16);
    // GGML will switch batch size on the fly. So we need to create a new kernel if the batch size is different
    if(kernel == NULL)
        kernel = ggml_rknpu2_matmul_kernel_create(m, k, n, RKNN_TENSOR_FLOAT16);

    GGML_ASSERT(kernel->matmul_io_attr.A.type == RKNN_TENSOR_FLOAT16);
    GGML_ASSERT(kernel->matmul_io_attr.C.type == RKNN_TENSOR_FLOAT32);
    //A: fp32 -> fp16
    float const* src1_data = src1->data;
    uint16_t* A = kernel->A->virt_addr;
    #pragma clang loop unroll_count(32)
    for(size_t i = 0; i < m*k; i++) {
        A[i] = arm_fp32_to_fp16(src1_data[i]);
    }

    int ret = rknn_matmul_set_io_mem(kernel->matmul_ctx, kernel->A, &kernel->matmul_io_attr.A);
    GGML_ASSERT(ret == 0);
    ret = rknn_matmul_set_io_mem(kernel->matmul_ctx, pack->B, &kernel->matmul_io_attr.B);
    GGML_ASSERT(ret == 0);
    ret = rknn_matmul_run(kernel->matmul_ctx);
    GGML_ASSERT(ret == 0);

    float* C = kernel->C->virt_addr;
    float* dst_data = kernel->C->virt_addr;
    memcpy(dst_data, C, m*n*sizeof(float));
}

int ggml_rknpu2_can_mul_mat_b(const struct ggml_tensor * tensor)
{
    const int64_t k = tensor->ne[0];
    const int64_t n = tensor->ne[1];
    if(k > 10240 || n > 4096) // RKNPU2 limit
        return 0;
    
    // k and n size must align to 32 bytes
    if(k % 32 != 0 || n % 32 != 0)
        return 0;

    // make sure the tensor has assosiated data
    if(tensor->backend != GGML_BACKEND_GPU)
        return 0;
    
    if(tensor->type != GGML_TYPE_Q8_0)
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

static void ggml_rknpu2_transposed_to_native_fp16(uint16_t* restrict dst, const uint16_t* restrict src, size_t k, size_t n)
{
    GGML_ASSERT(k % 16 == 0 && n % 8 == 0 && k > 0 && n > 0);

    // RKNN native layout is [N/8, K/16, 8, 16]
    const size_t rknpu_strides[4] = {k / 16 * 8 * 16, 8 * 16, 16, 1};

    // Block copy 16x8 at a time to improve cache locality
    for (size_t j = 0; j < k / 16; j++) {
      for (size_t i = 0; i < n / 8; i++) {
        for (size_t jj = 0; jj < 16; jj++) {
          size_t partial_src_idx = (j * 16 + jj) * n + i * 8;
          size_t partial_dst_idx =
              i * rknpu_strides[0] + j * rknpu_strides[1] + jj;

          for (size_t ii = 0; ii < 8; ii++) {
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

    uint16_t *fp16data = malloc(nelements * sizeof(uint16_t));
    for (int64_t i = 0; i < nelements; i++) {
      fp16data[i] = arm_fp32_to_fp16(fdata[i]);
    }

    free(fdata);

    uint16_t* reordered_data = malloc(ne0 * ne1 * sizeof(uint16_t));
    ggml_rknpu2_transposed_to_native_fp16(reordered_data, fp16data, ne1, ne0);
    free(fp16data);

    struct ggml_rknpu2_data_pack* pack = malloc(sizeof(struct ggml_rknpu2_data_pack));
    memset(pack, 0, sizeof(struct ggml_rknpu2_data_pack));

    pack->ordered_data = reordered_data;
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