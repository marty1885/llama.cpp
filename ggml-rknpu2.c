#include "ggml-rknpu2.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "rknn_api.h"
#include "rknn_matmul_api.h"

#include <sys/ioctl.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>

#include <arm_neon.h>

#define GGML_RKNPU2_INPUT_SCALE 1.7f

static __fp16 arm_fp32_to_fp16(float x) {
    return (__fp16)x;
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

rknn_tensor_type rknpu2_matmul_input_type_to_output_type(rknn_tensor_type type)
{
    switch(type) {
        case RKNN_TENSOR_FLOAT16:
            return RKNN_TENSOR_FLOAT32;
        case RKNN_TENSOR_INT8:
            return RKNN_TENSOR_INT32;
        case RKNN_TENSOR_INT4:
            return RKNN_TENSOR_INT16;
        default:
            GGML_ASSERT(0);
    }
}

const char* rknpu2_tensor_type_to_string(rknn_tensor_type type)
{
    switch(type) {
        case RKNN_TENSOR_FLOAT32:
            return "FLOAT32";
        case RKNN_TENSOR_FLOAT16:
            return "FLOAT16";
        case RKNN_TENSOR_INT8:
            return "INT8";
        case RKNN_TENSOR_INT16:
            return "INT16";
        case RKNN_TENSOR_INT32:
            return "INT32";
        case RKNN_TENSOR_UINT8:
            return "UINT8";
        case RKNN_TENSOR_UINT16:
            return "UINT16";
        default:
            GGML_ASSERT(0);
    }
}

struct ggml_rknpu2_data_pack
{
    rknn_tensor_type type;
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

#define GGML_RKNPU2_USE_OUTSIDE_ALLOC 1

#if GGML_RKNPU2_USE_OUTSIDE_ALLOC
struct dma_heap_allocation_data {
	uint64_t len;
	uint32_t fd;
	uint32_t fd_flags;
	uint64_t heap_flags;
};

#define DMA_HEAP_IOC_MAGIC		'H'
#define DMA_HEAP_IOCTL_ALLOC	_IOWR(DMA_HEAP_IOC_MAGIC, 0x0,\
				      struct dma_heap_allocation_data)

#define DMA_BUF_SYNC_READ      (1 << 0)
#define DMA_BUF_SYNC_WRITE     (2 << 0)
#define DMA_BUF_SYNC_RW        (DMA_BUF_SYNC_READ | DMA_BUF_SYNC_WRITE)
#define DMA_BUF_SYNC_START     (0 << 2)
#define DMA_BUF_SYNC_END       (1 << 2)
#define DMA_BUF_BASE		'b'
#define DMA_BUF_IOCTL_SYNC	_IOW(DMA_BUF_BASE, 0, uint64_t)
#define CMA_HEAP_SIZE	(1024 * 1024)

//Helper function to manually allocate buffer from dma_heap for RKNPU2
//The internal RKNPU2 API will allocate buffer from DMA32 heap, which is only 4GiB, not enough for large models.
//WARNING: Memory leak will not be released on exit!! But it will be released on next run...?
int dma_alloc(size_t size, int *fd, void **va) {
    int ret;
    int prot;
    void *mmap_va;
    int dma_heap_fd = -1;
    struct dma_heap_allocation_data buf_data;
    const char* path = "/dev/dma_heap/system";

    /* open dma_heap fd */
    dma_heap_fd = open(path, O_RDWR);
    if (dma_heap_fd < 0) {
        printf("open %s fail!\n", path);
        return dma_heap_fd;
    }

    /* alloc buffer */
    memset(&buf_data, 0x0, sizeof(struct dma_heap_allocation_data));

    buf_data.len = size;
    buf_data.fd_flags = O_CLOEXEC | O_RDWR;
    ret = ioctl(dma_heap_fd, DMA_HEAP_IOCTL_ALLOC, &buf_data);
    if (ret < 0) {
        printf("RK_DMA_HEAP_ALLOC_BUFFER failed\n");
        return ret;
    }

    /* mmap va */
    if (fcntl(buf_data.fd, F_GETFL) & O_RDWR)
        prot = PROT_READ | PROT_WRITE;
    else
        prot = PROT_READ;

    /* mmap contiguors buffer to user */
    mmap_va = (void *)mmap(NULL, buf_data.len, prot, MAP_SHARED, buf_data.fd, 0);
    if (mmap_va == MAP_FAILED) {
        printf("mmap failed: %s\n", strerror(errno));
        return -errno;
    }

    *va = mmap_va;
    *fd = buf_data.fd;

    close(dma_heap_fd);

    return 0;
}

int dma_sync_device_to_cpu(int fd) {
    uint64_t flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_RW;
    return ioctl(fd, DMA_BUF_IOCTL_SYNC, &flags);
}

int dma_sync_cpu_to_device(int fd) {
    uint64_t flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_RW;
    return ioctl(fd, DMA_BUF_IOCTL_SYNC, &flags);
}
void dma_buf_free(size_t size, int *fd, void *va) {
    int len;

    len =  size;
    munmap(va, len);

    close(*fd);
    *fd = -1;
}

#endif

// Pool of RKNPU2 matmul kernels so we can reuse them
#define GGML_RKNPU2_MAX_MATMUL_KERNELS 16
static struct ggml_rknpu2_matmul_kernel matmul_kernels[GGML_RKNPU2_MAX_MATMUL_KERNELS];
static int matmul_kernels_count = 0;

static uint64_t rknpu2_allocated_bytes = 0;

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
    printf("Created RKNPU2 matmul kernel: src0(%d, %d) x src1(%d, %d) = dst(%d, %d) %s\n", m, k, k, n, m, n, rknpu2_tensor_type_to_string(type));

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
        struct ggml_rknpu2_matmul_kernel* kernel = ggml_rknpu2_matmul_kernel_create(m, k, n, pack->type);
        // allocate B
#if GGML_RKNPU2_USE_OUTSIDE_ALLOC
        int fd = -1;
        uint8_t *va = NULL;
        dma_alloc(kernel->matmul_io_attr.B.size, &fd, (void *)&va);
        dma_sync_device_to_cpu(fd);
        pack->B = rknn_create_mem_from_fd(kernel->matmul_ctx, fd, va,
                                          kernel->matmul_io_attr.B.size, 0);
        memcpy(pack->B->virt_addr, pack->ordered_data,
               kernel->matmul_io_attr.B.size);
        dma_sync_cpu_to_device(fd);
#else
        pack->B =
            rknn_create_mem(kernel->matmul_ctx, kernel->matmul_io_attr.B.size);
        memcpy(pack->B->virt_addr, pack->ordered_data,
               kernel->matmul_io_attr.B.size);
#endif
        free(pack->ordered_data);
        rknpu2_allocated_bytes += kernel->matmul_io_attr.B.size;
        printf("RKNPU2 allocated %f MiB\n",
               rknpu2_allocated_bytes / 1024.0F / 1024.0F);
        pack->ordered_data = NULL;
        pack->initialized = 1;
    }

    struct ggml_rknpu2_matmul_kernel* kernel = ggml_rknpu2_matmul_kernel_find(m, k, n, pack->type);
    // GGML will switch batch size on the fly. So we need to create a new kernel if the batch size is different
    if(kernel == NULL)
        kernel = ggml_rknpu2_matmul_kernel_create(m, k, n, pack->type);

    GGML_ASSERT(kernel->matmul_io_attr.A.type == pack->type);
    GGML_ASSERT(kernel->matmul_io_attr.C.type == rknpu2_matmul_input_type_to_output_type(pack->type));
    rknn_tensor_type inference_type = pack->type;
    if(inference_type == RKNN_TENSOR_FLOAT16) {
        //A: fp32 -> fp16
        float const* src1_data = src1->data;
        __fp16* A = kernel->A->virt_addr;
        #pragma clang loop unroll_count(32)
        for(size_t i = 0; i < m*k; i++) {
            A[i] = arm_fp32_to_fp16(src1_data[i]);
        }
    }
    else if(inference_type == RKNN_TENSOR_INT8) {
        //A: fp32 -> int8
        float const* src1_data = src1->data;
        int8_t* A = kernel->A->virt_addr;
        #pragma clang loop unroll_count(32)
        for(size_t i = 0; i < m*k; i++) {
            float val = round(fmin(fmax(src1_data[i]*127.0f/GGML_RKNPU2_INPUT_SCALE, -127.0f), 127.0f));
            A[i] = val;
        }
    }
    else {
        GGML_ASSERT(0 && "Unsupported inference type");
    }

    int ret = rknn_matmul_set_io_mem(kernel->matmul_ctx, kernel->A, &kernel->matmul_io_attr.A);
    GGML_ASSERT(ret == 0);
    ret = rknn_matmul_set_io_mem(kernel->matmul_ctx, pack->B, &kernel->matmul_io_attr.B);
    GGML_ASSERT(ret == 0);
    ret = rknn_matmul_run(kernel->matmul_ctx);
    GGML_ASSERT(ret == 0);

    // dst->data = kernel->C->virt_addr;
    if(inference_type == RKNN_TENSOR_FLOAT16) {
        //C: fp32 -> fp32
        memcpy(dst->data, kernel->C->virt_addr, m * n * sizeof(float));
    }
    else if(inference_type == RKNN_TENSOR_INT8) {
        //C: int32 -> fp32
        float* dst_data = dst->data;
        int32_t* C = kernel->C->virt_addr;
        #pragma clang loop unroll_count(32)
        for(size_t i = 0; i < m*n; i++)
            dst_data[i] = C[i] / 127.0f / 127.f * GGML_RKNPU2_INPUT_SCALE;
    }
    else {
        GGML_ASSERT(0 && "Unsupported inference type");
    }
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

static void ggml_rknpu2_transposed_to_native_fp16(__fp16 *restrict dst,
                                                  const float *restrict src,
                                                  size_t k, size_t n) {
  GGML_ASSERT(k % 32 == 0 && n % 16 == 0 && k > 0 && n > 0);

  // RKNN native layout is (N/16, K/32, 16, 32)
  const size_t rknpu_strides[4] = {k / 32 * 16 * 32, 16 * 32, 32, 1};

  // Block copy 32x16 at a time to improve cache locality
  for (size_t j = 0; j < k / 32; j++) {
    for (size_t i = 0; i < n / 16; i++) {
      for (size_t ii = 0; ii < 16; ii++) {
        size_t partial_src_idx = j * 32 + (i * 16 + ii) * k;
        size_t partial_dst_idx =
            i * rknpu_strides[0] + j * rknpu_strides[1] + ii * rknpu_strides[2];

        for (size_t jj = 0; jj < 32; jj++) {
          size_t src_idx = partial_src_idx + jj;
          size_t dst_idx = partial_dst_idx + jj;
          dst[dst_idx] = src[src_idx];
        }
      }
    }
  }
}

static void ggml_rknpu2_transposed_to_native_int8(int8_t *restrict dst,
                                                  const float *restrict src,
                                                  size_t k, size_t n) {
  GGML_ASSERT(k % 32 == 0 && n % 32 == 0 && k > 0 && n > 0);

  // RKNN native layout is (N/32, K/32, 32, 32)
  const size_t rknpu_strides[4] = {k / 32 * 32 * 32, 32 * 32, 32, 1};

  // Block copy 32x32 at a time to improve cache locality
  for (size_t j = 0; j < k / 32; j++) {
    for (size_t i = 0; i < n / 32; i++) {
      for (size_t ii = 0; ii < 32; ii++) {
        size_t partial_src_idx = j * 32 + (i * 32 + ii) * k;
        size_t partial_dst_idx =
            i * rknpu_strides[0] + j * rknpu_strides[1] + ii * rknpu_strides[2];

        for (size_t jj = 0; jj < 32; jj++) {
          size_t src_idx = partial_src_idx + jj;
          size_t dst_idx = partial_dst_idx + jj;
          dst[dst_idx] = round(fmin(fmax(src[src_idx], -1.0f), 1.0f) * 127.0f);
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

    void* reordered_data = NULL;
    const rknn_tensor_type inference_type = RKNN_TENSOR_INT8;
    if(inference_type == RKNN_TENSOR_FLOAT16) {
        reordered_data = malloc(nelements * sizeof(__fp16));
        ggml_rknpu2_transposed_to_native_fp16((__fp16*)reordered_data, fdata, ne1, ne0);
    }
    else if(inference_type == RKNN_TENSOR_INT8) {
        reordered_data = malloc(nelements * sizeof(int8_t));
        ggml_rknpu2_transposed_to_native_int8((int8_t*)reordered_data, fdata, ne1, ne0);
    }
    else {
        free(fdata);
        GGML_ASSERT(0 && "Unsupported inference type");
    }

    GGML_ASSERT(reordered_data != NULL);
    free(fdata);
    struct ggml_rknpu2_data_pack* pack = malloc(sizeof(struct ggml_rknpu2_data_pack));
    memset(pack, 0, sizeof(struct ggml_rknpu2_data_pack));

    pack->ordered_data = reordered_data;
    pack->initialized = 0;
    pack->type = inference_type;

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
