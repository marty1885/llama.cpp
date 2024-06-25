#pragma once

#include "ggml.h"
#include "ggml-backend.h"

// GGML backend for using Tenstorrent's tt-Metalium and TTNN libraries


#ifdef  __cplusplus
extern "C" {
#endif

// backend API
GGML_API GGML_CALL ggml_backend_t ggml_backend_metalium_init(void);

GGML_API GGML_CALL bool ggml_backend_is_metalium(ggml_backend_t backend);

GGML_CALL ggml_backend_t ggml_backend_reg_metalium_init(const char * params, void * user_data);

#ifdef  __cplusplus
}
#endif
