#pragma once

#include "ggml.h"
#include "ggml-backend.h"

// GGML backend for using Tenstorrent's tt-Metalium and TTNN libraries


#ifdef  __cplusplus
extern "C" {
#endif

// backend API

GGML_BACKEND_API bool ggml_backend_is_metalium(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_metalium_buffer_type(int device_id);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_metalium_reg();

// TTNN trace (capture/replay) control. Tracing must be enabled BEFORE the device is opened
// (i.e. before the first ggml_backend_reg_by_name("Metalium")) so the device reserves a
// trace-capable region. Also seeded from the GGML_METALIUM_TRACE env var. Default OFF.
GGML_BACKEND_API void ggml_backend_metalium_set_tracing(bool enable);
GGML_BACKEND_API bool ggml_backend_metalium_tracing_enabled(void);

#ifdef  __cplusplus
}
#endif
