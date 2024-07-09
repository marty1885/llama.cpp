#ifdef __cplusplus
#include "common/bfloat16.hpp"
#include "common/constants.hpp"
#include "device/tt_arch_types.h"
#include "ggml-backend-impl.h"
#include "ggml.h"
#include "ggml-metalium.h"

#include "host_api.hpp"
#include "hostdevcommon/kernel_structs.h"
#include "impl/dispatch/command_queue.hpp"
#include "tensor/host_buffer/functions.hpp"
#include "tensor/host_buffer/types.hpp"
#include "tensor/types.hpp"
#include "tt_dnn/op_library/auto_format.hpp"
#include "tt_dnn/op_library/unpad/unpad_op.hpp"
#include "tt_dnn/op_library/composite/composite_ops.hpp"
#include "tt_dnn/op_library/tilize/tilize_op.hpp"
#include "tt_dnn/op_library/untilize/untilize_op.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <optional>
#include <tt_eager/tensor/tensor.hpp>
#include <ttnn/core.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <tt_eager/tt_dnn/op_library/transpose/transpose_op.hpp>
#include <ttnn/device.hpp>
#include <tt_dnn/op_library/fully_connected/fully_connected_op.hpp>
#include <tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp>
#include <tt_dnn/op_library/copy/copy_op.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/operations/matmul/matmul.hpp>


#include <memory>
#include <type_traits>
#include <unordered_map>
#include <variant>

#endif