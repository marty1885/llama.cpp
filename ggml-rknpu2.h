#pragma once

/******************************************************

The experimental RKNPU2 backend for GGML.

LIMITATIONS:
- Only supports Q8_0 GGML quantization
- Only MatMul is supported
- RK3588 only
- Please only use 1 CPU thread for best efficiency. More threads will not run faster.
- Not faster than running on 1x A76 Cores, but saves some CPU time...
*/

#include "ggml.h"

#ifdef  __cplusplus
extern "C" {
#endif

void ggml_rknpu2_init(void);

int ggml_rknpu2_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
int ggml_rknpu2_can_mul_mat_b(const struct ggml_tensor * tensor);
void ggml_rknpu2_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst, void * wdata, size_t wsize);

void ggml_rknpu2_transform_tensor(void * data, struct ggml_tensor * tensor);
void ggml_rknpu2_free_data(struct ggml_tensor * tensor);

// TODO: Find a place to call this. Not big deal since kernel will release all resources.
void ggml_rknpu2_destroy(void);


#ifdef  __cplusplus
}
#endif