#include <unistd.h>
#ifdef __cplusplus
#include "common/bfloat16.hpp"
#include "common/constants.hpp"
#include "device/tt_arch_types.h"
#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "ggml-metalium.h"

#include "host_api.hpp"
#include "impl/dispatch/command_queue.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/normalization/softmax/device/softmax_op.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <optional>
#include <ttnn/core.hpp>
#include <ttnn/device.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding.hpp>
#include <ttnn/operations/matmul/matmul.hpp>
#include <ttnn/operations/kv_cache.hpp>
#include <ttnn/operations/data_movement/slice/slice.hpp>
#include <ttnn/operations/normalization/layernorm/layernorm.hpp>
#include <ttnn/operations/normalization/rmsnorm/rmsnorm.hpp>
#include <ttnn/experimental/tt_dnn/op_library/untilize/untilize_op.hpp>
#include <ttnn/experimental/tt_dnn/op_library/transpose/transpose_op.hpp>
#include <ttnn/experimental/tt_dnn/op_library/nlp_tms/nlp_tms.hpp>
#include <ttnn/experimental/tt_dnn/op_library/composite/composite_ops.hpp>
#include <tt_metal/detail/persistent_kernel_cache.hpp>
#include <tt_dnn/op_library/concat/concat_op.hpp>
#include <ttnn/operations/normalization/softmax/softmax.hpp>


#include <memory>
#include <type_traits>
#include <variant>

#endif
