#pragma once

/******************************************************

The experimental RKNPU2 backend for GGML.

LIMITATIONS:
- Only supports running in FLOAT16 mode (uses Q8_0 GGML tensors)
- Only MatMul is supported
- RK3588 only
- Not faster then just running on RK3588 CPU (4x A76)
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