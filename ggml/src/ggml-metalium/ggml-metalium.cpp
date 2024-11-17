#include "common/base_types.hpp"
#include "common/bfloat16.hpp"
#include "common/constants.hpp"
#include "common/logger.hpp"
#include "device/tt_arch_types.h"
#include "distributed/mesh_device.hpp"
#include "distributed/mesh_device_view.hpp"
#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"
#include "ggml-metalium.h"

#include "host_api.hpp"
#include "hostdevcommon/common_values.hpp"
#include "impl/dispatch/dispatch_core_manager.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/data_movement/reshape_on_device/reshape.hpp"
#include "ttnn/operations/data_movement/untilize_with_unpadding/untilize_with_unpadding.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/moreh/moreh_group_norm/moreh_group_norm.hpp"
#include "ttnn/operations/normalization/softmax/device/softmax_op.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <ttnn/core.hpp>
#include <ttnn/device.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding.hpp>
#include <ttnn/operations/matmul/matmul.hpp>
#include <ttnn/operations/moreh/moreh_matmul/moreh_matmul.hpp>
#include <ttnn/operations/kv_cache/kv_cache.hpp>
#include <ttnn/operations/data_movement/slice/slice.hpp>
#include <ttnn/operations/normalization/layernorm/layernorm.hpp>
#include <ttnn/operations/normalization/rmsnorm/rmsnorm.hpp>
#include <ttnn/operations/data_movement/untilize/untilize.hpp>
#include <ttnn/operations/experimental/transformer/nlp_kv_cache_load_slice/nlp_kv_cache_load_slice.hpp>
#include <ttnn/operations/creation.hpp>
#include <ttnn/operations/eltwise/unary/unary_composite.hpp>
#include <ttnn/operations/data_movement/transpose/transpose.hpp>
#include <ttnn/operations/data_movement/permute/permute.hpp>
#include <ttnn/operations/data_movement/repeat/repeat.hpp>
#include <ttnn/operations/data_movement/concat/concat.hpp>
#include <ttnn/operations/experimental/copy/typecast/typecast.hpp>
#include <tt_metal/detail/persistent_kernel_cache.hpp>
#include <ttnn/distributed/api.hpp>
#include <ttnn/operations/normalization/softmax/softmax.hpp>


#include <memory>
#include <type_traits>
#include <variant>
#include <vector>

struct ggml_backend_metalium_context {
    std::shared_ptr<ttnn::distributed::MeshDevice> device;
    int device_id = 0;
    std::string name;
};

struct ggml_backend_metalium_device_context {
    std::shared_ptr<ttnn::distributed::MeshDevice> device;
    int device_id = -1;
    std::string name;
    std::string description;
};

struct ggml_backend_metalium_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

struct TensorWithMetadata;

struct ggml_backend_metalium_buffer_context {

    size_t ggml_buffer_size_bytes = 0;
    std::string name;
    ttnn::distributed::MeshDevice* device = nullptr;
    size_t base_offset = 0;

    // Tracking our own allocations because Metalium limitations and GGML assuming them
    std::vector<std::unique_ptr<TensorWithMetadata>> metadata_to_free;
};

struct TensorWithMetadata
{
    std::shared_ptr<tt::tt_metal::Tensor> tensor;
    ggml_type ggtype = GGML_TYPE_COUNT;
    ggml_backend_metalium_buffer_context* bufctx = nullptr;
};

static bool ggml_tt_tensors_shape_equal(const ggml_tensor* ggtensor, const tt::tt_metal::Tensor& ttensor)
{
    for(int i = 0; i < GGML_MAX_DIMS; i++) {
        if(ggtensor->ne[GGML_MAX_DIMS - i - 1] != ttensor.shape()[i]) {
            return false;
        }
    }

    if(ttensor.shape().size() > GGML_MAX_DIMS) {
        for(size_t i = GGML_MAX_DIMS; i < ttensor.shape().size(); i++) {
            if(ttensor.shape()[i] != 1) {
                return false;
            }
        }
    }
    return true;
}

static void dump_ggml_tensor_meta(const ggml_tensor* ggtensor)
{
    std::cerr << "GGML tensor: " << ggtensor->name << "\n"
        << "  type: " << ggml_type_name(ggtensor->type) << "\n"
        << "  ne: " << ggtensor->ne[0] << " " << ggtensor->ne[1] << " " << ggtensor->ne[2] << " " << ggtensor->ne[3] << "\n"
        << "  nb: " << ggtensor->nb[0] << " " << ggtensor->nb[1] << " " << ggtensor->nb[2] << " " << ggtensor->nb[3] << "\n"
        << "  op: " << ggml_op_name(ggtensor->op) << "\n"
        << "  data: " << ggtensor->data << "\n"
        << "  src0: " << ggtensor->src[0] << "\n";
    if(ggtensor->src[0] != nullptr) {
        std::cerr << "    src0->name: " << ggtensor->src[0]->name << "\n"
            << "    src0->type: " << ggml_type_name(ggtensor->src[0]->type) << "\n"
            << "    src0->ne:   " << ggtensor->src[0]->ne[0] << " " << ggtensor->src[0]->ne[1] << " " << ggtensor->src[0]->ne[2] << " " << ggtensor->src[0]->ne[3] << "\n"
            << "    src0->nb:   " << ggtensor->src[0]->nb[0] << " " << ggtensor->src[0]->nb[1] << " " << ggtensor->src[0]->nb[2] << " " << ggtensor->src[0]->nb[3] << "\n"
            << "    src0->op:   " << ggml_op_name(ggtensor->src[0]->op) << "\n"
            << "    src0->data: " << ggtensor->src[0]->data << "\n";
    }
    std::cerr << "  src1: " << ggtensor->src[1] << "\n";
    if(ggtensor->src[1] != nullptr) {
        std::cerr << "    src1->name: " << ggtensor->src[1]->name << "\n"
            << "    src1->type: " << ggml_type_name(ggtensor->src[1]->type) << "\n"
            << "    src1->ne: " << ggtensor->src[1]->ne[0] << " " << ggtensor->src[1]->ne[1] << " " << ggtensor->src[1]->ne[2] << " " << ggtensor->src[1]->ne[3] << "\n"
            << "    src1->nb: " << ggtensor->src[1]->nb[0] << " " << ggtensor->src[1]->nb[1] << " " << ggtensor->src[1]->nb[2] << " " << ggtensor->src[1]->nb[3] << "\n"
            << "    src1->op: " << ggml_op_name(ggtensor->src[1]->op) << "\n"
            << "    src1->data: " << ggtensor->src[1]->data << "\n";
    }
    std::cerr << "  view_src: " << ggtensor->view_src << "\n";
    if(ggtensor->view_src != nullptr) {
        std::cerr << "    view_src->name: " << ggtensor->view_src->name << "\n"
            << "    view_src->type: " << ggml_type_name(ggtensor->view_src->type) << "\n"
            << "    view_src->ne: " << ggtensor->view_src->ne[0] << " " << ggtensor->view_src->ne[1] << " " << ggtensor->view_src->ne[2] << " " << ggtensor->view_src->ne[3] << "\n"
            << "    view_src->nb: " << ggtensor->view_src->nb[0] << " " << ggtensor->view_src->nb[1] << " " << ggtensor->view_src->nb[2] << " " << ggtensor->view_src->nb[3] << "\n"
            << "    view_src->op: " << ggml_op_name(ggtensor->view_src->op) << "\n"
            << "    view_src->data: " << ggtensor->view_src->data << "\n";
    }
}

static ttnn::DeviceComputeKernelConfig make_compute_kernel_config(ttnn::Device* device)
{
    ttnn::DeviceComputeKernelConfig cfg;
    if(device->arch() == tt::ARCH::GRAYSKULL) {
        cfg = ttnn::GrayskullComputeKernelConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .math_approx_mode = false,
        };
    }
    else if (device->arch() == tt::ARCH::WORMHOLE_B0 || device->arch() == tt::ARCH::BLACKHOLE) {
        cfg = ttnn::WormholeComputeKernelConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .math_approx_mode = false,
            .fp32_dest_acc_en = true,
            .packer_l1_acc = true
        };
    }
    else {
        tt::log_fatal("Unsupported device arch {} in make_compute_kernel_config", device->arch());
        abort();
    }
    return cfg;
}

// Debug flags that can be enabled at runtime. Because recompiling the backend takes forever
// this enables faster iteration on debugging. Eventually these should be removed
// NOTE: DO NOT invent more _hack flags. Else it devolves into a mess like what BUDA did
struct ggml_backend_metalium_debug_flags {
    bool print_rejected_ops = false;        // Print ops that the backend rejects
    bool print_view = false;                // Print details when a VIEW op is being realized
};

static const ggml_backend_metalium_debug_flags g_debug_flags = []() {
    auto func = [](const char* env) -> bool {
        const char* val = std::getenv(env);
        if(val != nullptr) {
            std::string str(val);
            std::transform(str.begin(), str.end(), str.begin(), ::tolower);
            if(str != "0" && str != "false" && str != "no" && str != "off") {
                return true;
            }
        }
        return false;
    };

    return ggml_backend_metalium_debug_flags {
        .print_rejected_ops = func("GGML_METALIUM_PRINT_REJECTED_OPS"),
        .print_view = func("GGML_METALIUM_PRINT_VIEW")
    };
}();

static std::string_view trim_sv(std::string_view sv, std::string_view chars = " \t\n\r")
{
    sv.remove_prefix(std::min(sv.find_first_not_of(chars), sv.size()));
    sv.remove_suffix(sv.size() - std::min(sv.find_last_not_of(chars) + 1, sv.size()));
    return sv;
}

static std::optional<int> try_stoi(std::string_view sv)
{
    try {
        size_t pos = 0;
        int n = std::stoi(std::string(sv), &pos);
        if(pos != sv.size()) {
            return std::nullopt;
        }
        return n;
    } catch(...) {
        return std::nullopt;
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// Backend internal state tracking because GGML API does not allow
///////////////////////////////////////////////////////////////////////////////////////////////////////

// Maintain all base addresses are unique
// TODO: Do we still need this since we already removed the virtual address mapping hack?
static size_t g_metalium_base_offset = 0;

///////////////////////////////////////////////////////////////////////////////////////////////////////
// Actual backend code
///////////////////////////////////////////////////////////////////////////////////////////////////////

static tt::tt_metal::DataType ggml2tt_type_internal(ggml_type ggtype, tt::ARCH arch) {
    // This table is consulted to map GGML types to TT types dueing tensor creation
    // TODO: Separate Wormhole out, it supports more types
    if(arch == tt::ARCH::GRAYSKULL || arch == tt::ARCH::WORMHOLE_B0) {
        static constexpr std::array<tt::tt_metal::DataType, GGML_TYPE_COUNT> table = {
            /*GGML_TYPE_F32 = */ tt::tt_metal::DataType::BFLOAT16,
            /*GGML_TYPE_F16 = */ tt::tt_metal::DataType::BFLOAT16,
            /*GGML_TYPE_Q4_0 = */ tt::tt_metal::DataType::BFLOAT8_B,    // Using BFLOAT8_B for now as BFLOAT4_B is broken on Grayskull
            /*GGML_TYPE_Q4_1 = */ tt::tt_metal::DataType::INVALID,      // Does work but causes issues in unit tests
            tt::tt_metal::DataType::INVALID,
            tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_Q5_0 = */ tt::tt_metal::DataType::BFLOAT8_B,
            /*GGML_TYPE_Q5_1 = */ tt::tt_metal::DataType::BFLOAT8_B,      // Does work but causes issues in unit tests
            /*GGML_TYPE_Q8_0 = */ tt::tt_metal::DataType::BFLOAT8_B,
            /*GGML_TYPE_Q8_1 = */ tt::tt_metal::DataType::BFLOAT8_B,
            /*GGML_TYPE_Q2_K = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_Q3_K = */ tt::tt_metal::DataType::BFLOAT8_B,   // Using BFLOAT8_B for now as BFLOAT4_B is broken on Grayskull
            /*GGML_TYPE_Q4_K = */ tt::tt_metal::DataType::BFLOAT8_B,   // Using BFLOAT8_B for now as BFLOAT4_B is broken on Grayskull
            /*GGML_TYPE_Q5_K = */ tt::tt_metal::DataType::BFLOAT8_B,
            /*GGML_TYPE_Q6_K = */ tt::tt_metal::DataType::BFLOAT8_B,
            /*GGML_TYPE_Q8_K = */ tt::tt_metal::DataType::BFLOAT8_B,
            /*GGML_TYPE_IQ2_XXS = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_IQ2_XS = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_IQ3_XXS = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_IQ1_S = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_IQ4_NL = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_IQ3_S = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_IQ2_S = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_IQ4_XS = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_I8 = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_I16 = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_I32 = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_I64 = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_F64 = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_IQ1_M = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_BF16 = */ tt::tt_metal::DataType::BFLOAT16,
            /*GGML_TYPE_Q4_0_4_4 = */ tt::tt_metal::DataType::INVALID, // Untested from this point on
            /*GGML_TYPE_Q4_0_4_8 = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_Q4_0_8_8 = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_TQ1_0   = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_TQ2_0   = */ tt::tt_metal::DataType::INVALID,
        };
        tt::tt_metal::DataType type = table[ggtype];
        return type;
    }
    GGML_ASSERT(false && "Unsupported Tenstorrent card architecture");
}

static bool numpy_broadcast_rule(const ggml_tensor* t, const ggml_tensor* q)
{
    int tdim = ggml_n_dims(t);
    int qdim = ggml_n_dims(q);

    int min_dim = tdim < qdim ? tdim : qdim;
    for(int i = 0; i < min_dim; i++) {
        if(t->ne[i] != q->ne[i] && t->ne[i] != 1 && q->ne[i] != 1) {
            return false;
        }
    }
    return true;
}

static tt::tt_metal::DataType ggml2tt_type(ggml_type ggtype, tt::ARCH arch)
{
    tt::tt_metal::DataType type = ggml2tt_type_internal(ggtype, arch);
    if(type == tt::tt_metal::DataType::INVALID) {
        tt::log_fatal(tt::LogType::LogAlways, "Unsupported data type: {}", ggml_type_name(ggtype));
        GGML_ASSERT(false && "Unsupported data type");
    }
    return type;

}

static bool is_ggml_type_supported_by_metalium(ggml_type ggtype, tt::ARCH arch) {
    return ggml2tt_type_internal(ggtype, arch) != tt::tt_metal::DataType::INVALID;
}

template <typename SrcType, typename DstType>
tt::tt_metal::OwnedStorage data2owned_storage(const SrcType* src, size_t size) {
    // Converts GGML floating point (FP32, FP16, BF16) to TT floating point (FP32, BF16)
    std::vector<DstType> vec(size);
    using Src = std::remove_cv_t<std::remove_reference_t<SrcType>>;
    using Dst = std::remove_cv_t<std::remove_reference_t<DstType>>;
    // Convert from  GGML types to TT types
    static_assert(std::is_same_v<Src, float> || std::is_same_v<Src, ggml_bf16_t> || std::is_same_v<Src, ggml_fp16_t>);
    static_assert(std::is_same_v<Dst, float> || std::is_same_v<Dst, bfloat16>);

    auto src_adaptor = [](const SrcType& src) -> float {
        if constexpr(std::is_same_v<Src, ggml_fp16_t>) {
            return ggml_fp16_to_fp32(src);
        }
        else if constexpr(std::is_same_v<Src, ggml_bf16_t>) {
            return ggml_bf16_to_fp32(src);
        }
        else if constexpr(std::is_same_v<Src, float>) {
            return src;
        }
        GGML_UNREACHABLE();
    };

    auto dst_adaptor = [](DstType& dst, float val) {
        if constexpr(std::is_same_v<Dst, bfloat16>) {
            dst = bfloat16(val);
        }
        else if constexpr(std::is_same_v<Dst, float>) {
            dst = val;
        }
        else {
            GGML_UNREACHABLE();
        }
    };

    // special case if both GGML and TT types have the same underlying type (e.g. both FP32 or BF16)
    if constexpr(std::is_same_v<Src, Dst> || (std::is_same_v<Src, ggml_fp16_t> && std::is_same_v<Dst, bfloat16>)) {
        // Make GCC shut up about writing into a class like it's flat memory
        memcpy((void*)vec.data(), src, size * sizeof(Src));
    }
    else {
        for(size_t i = 0; i < size; i++) {
            dst_adaptor(vec[i], src_adaptor(src[i]));
        }
    }
    auto owned = tt::tt_metal::owned_buffer::Buffer<DstType>(std::make_shared<std::vector<DstType>>(std::move(vec)));
    return OwnedStorage(std::move(owned));
}

template <typename DstType>
tt::tt_metal::OwnedStorage ggml_quantized2owned_storage(const void* src, ggml_tensor* tensor) {
    const ggml_type_traits* trait = ggml_get_type_traits(tensor->type);
    GGML_ASSERT(trait->to_float != NULL);
    // TODO: Currently we decompress the entire quantized data into float and then convert to bfloat16
    // (So later on we can convert again to TT quantized types). We SHOULD be able to not decompress
    // the entire quantized GGML tensor. Save on memory.
    // TODO: Consider parallelizing this. It is quite slow
    std::vector<float> vec(ggml_nelements(tensor));
    trait->to_float(src, vec.data(), ggml_nelements(tensor));

    return data2owned_storage<float, bfloat16>(vec.data(), vec.size());
}

template <typename SrcType>
void tensor2ggml(const tt::tt_metal::Tensor& tensor, void* dst, ggml_type dst_ggtype) {
    // Converts TT tensors to GGML types
    ttnn::SimpleShape shape = tensor.shape().logical_shape();
    ttnn::SimpleShape padded_shape = tensor.shape().padded_shape();
    static_assert(std::is_same_v<SrcType, float> || std::is_same_v<SrcType, bfloat16>);

    tt::tt_metal::Tensor row_major_tensor = ttnn::untilize(tensor).cpu();
    GGML_ASSERT(row_major_tensor.storage_type() == StorageType::OWNED or row_major_tensor.storage_type() == StorageType::BORROWED);
    GGML_ASSERT(std::holds_alternative<OwnedStorage>(row_major_tensor.storage()) || std::holds_alternative<BorrowedStorage>(row_major_tensor.storage()));

    const SrcType* buf = nullptr;
    size_t buf_size = 0;
    if(std::holds_alternative<OwnedStorage>(row_major_tensor.storage())) {
        const OwnedStorage& owned = std::get<OwnedStorage>(row_major_tensor.storage());
        auto& buffer = std::get<owned_buffer::Buffer<SrcType>>(owned.buffer);
        buf = buffer.begin();
        buf_size = buffer.size();
    }
    else if(std::holds_alternative<BorrowedStorage>(row_major_tensor.storage())) {
        const BorrowedStorage& borrowed = std::get<BorrowedStorage>(row_major_tensor.storage());
        auto& buffer = std::get<borrowed_buffer::Buffer<SrcType>>(borrowed.buffer);
        buf = buffer.begin();
        buf_size = buffer.size();
    } else {
        GGML_ASSERT(false && "Unsupported buffer type");
    }
    GGML_ASSERT(buf != nullptr);
    // TODO: Measure the performance of the following code. This is much simpeer and does untiling on the device
    // But does not work for large tensors
    // row_major_tensor = ttnn::untilize(tensor);
    // GGML_ASSERT(row_major_tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR);
    // tt::tt_metal::memcpy(queue, buf.data(), row_major_tensor);
    // tt::tt_metal::Finish(queue);
    void* intermid = nullptr;
    std::vector<std::byte> intermid_buf;
    bool need_quantized_conversion = false;
    bool src_dst_same = false;
    if(dst_ggtype == GGML_TYPE_F32 && !std::is_same_v<SrcType, float>) {
        intermid = (void*)dst;
        need_quantized_conversion = false;
        src_dst_same = false;
    }
    // Just putting the integer types here to remind me TT tensors can have integer types
    // But not supported on Grayskull.
    else if ((std::is_same_v<SrcType, bfloat16> && dst_ggtype == GGML_TYPE_BF16) ||
             (std::is_same_v<SrcType, int32_t> && dst_ggtype == GGML_TYPE_I32) ||
             (std::is_same_v<SrcType, int16_t> && dst_ggtype == GGML_TYPE_I16) ||
             (std::is_same_v<SrcType, int8_t> && dst_ggtype == GGML_TYPE_I8)) {
        intermid = (void*)dst;
        need_quantized_conversion = false;
        src_dst_same = true;
    }
    else {
        intermid_buf.resize(shape.volume() * sizeof(float));
        intermid = intermid_buf.data();
        need_quantized_conversion = true;
        src_dst_same = false;
    }

    auto src_adaptor = [](const SrcType& src) -> float {
        if constexpr(std::is_same_v<SrcType, bfloat16>) {
            return src.to_float();
        }
        else if (std::is_same_v<SrcType, float>) {
            return src;
        }
        GGML_UNREACHABLE();
    };

    // Tilize to ROW_MAJOR doesn't mean the tensor is contiguous. It still has the underlying 32x32 tiles
    // we need to view into the tensor to get the contiguous data
    const std::array<size_t, 4> stride = {padded_shape[1] * padded_shape[2] * padded_shape[3],
                                    padded_shape[2] * padded_shape[3],
                                    padded_shape[3],
                                    1};
    static_assert(GGML_MAX_DIMS == 4, "Looping depth is hardcoded to 4");
    size_t idx = 0;
    for(size_t w = 0; w < shape[0]; w++) {
        for(size_t z = 0; z < shape[1]; z++) {
            for(size_t y = 0; y < shape[2]; y++) {
                if(src_dst_same) {
                    // optimization: copy a chunk of memory at a time
                    const size_t src_idx = w * stride[0] + z * stride[1] + y * stride[2];
                    memcpy((SrcType*)intermid + idx, buf + src_idx, sizeof(SrcType) * shape[3]);
                    idx += shape[3];
                }
                else {
                    for(size_t x = 0; x < shape[3]; x++) {
                        const size_t src_idx = w * stride[0] + z * stride[1] + y * stride[2] + x * stride[3];
                        GGML_ASSERT(src_idx < buf_size);
                        float val = src_adaptor(buf[src_idx]);
                        ((float*)intermid)[idx] = val;
                        idx++;
                    }
                }
            }
        }
    }

    if (need_quantized_conversion) {
        GGML_ASSERT((ggml_is_quantized(dst_ggtype) || dst_ggtype == GGML_TYPE_F16) && "This block should only reach for quantized data types or FP16");
        GGML_ASSERT(intermid_buf.size() != 0);
        const ggml_type_traits_cpu* trait = ggml_get_type_traits_cpu(dst_ggtype);
        GGML_ASSERT(trait->from_float != NULL);
        trait->from_float((float*)intermid, dst, shape.volume());
    }
}

static bool is_view(const ggml_tensor* tensor)
{
    return tensor->view_src != nullptr ||
        tensor->op == GGML_OP_VIEW ||
        tensor->op == GGML_OP_RESHAPE ||
        tensor->op == GGML_OP_TRANSPOSE ||
        tensor->op == GGML_OP_PERMUTE;
}

static tt::tt_metal::Tensor reshape_tt_tensor_into_ggml(const tt::tt_metal::Tensor& tensor, const struct ggml_tensor * node)
{
    if(ggml_tt_tensors_shape_equal(node, tensor)) {
        return tensor;
    }

    std::array<uint32_t, GGML_MAX_DIMS> target_shape;
    for(int i = 0; i < GGML_MAX_DIMS; i++) {
        target_shape[i] = node->ne[GGML_MAX_DIMS - i - 1];
    }

    // TODO: Remove these checks. see https://github.com/tenstorrent/tt-metal/issues/14922
    //                                               vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    if(tensor.shape()[-1] == (uint32_t)node->ne[0] && tensor.shape()[-2] % 32 == 0 && node->ne[2] % 32 == 0) {
        // Fast path. reshape_on_device() can reshape is both the last dimension is the same 
        return ttnn::reshape_on_device(tensor, ttnn::SimpleShape(target_shape));
    }

    // This MAY trigger a slow path.
    return ttnn::reshape(tensor, ttnn::SimpleShape(target_shape));    
}

static tt::tt_metal::Tensor reshape_host_tt_tensor_into_ggml(const tt::tt_metal::Tensor& tensor, ttnn::Device* device, const struct ggml_tensor * node)
{
    GGML_ASSERT(tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR);
    GGML_ASSERT(tensor.storage_type() == tt::tt_metal::StorageType::OWNED || tensor.storage_type() == tt::tt_metal::StorageType::BORROWED);
    std::array<uint32_t, GGML_MAX_DIMS> target_shape;
    for(int i = 0; i < GGML_MAX_DIMS; i++) {
        target_shape[i] = node->ne[GGML_MAX_DIMS - i - 1];
    }

    return ttnn::tilize_with_zero_padding(tensor.reshape(ttnn::SimpleShape(target_shape)).to(device));
}

static std::shared_ptr<tt::tt_metal::Tensor> realize_ggml_view_impl(const ggml_tensor* tensor);
static std::shared_ptr<tt::tt_metal::Tensor> realize_ggml_view(const ggml_tensor* tensor)
{
    auto res = realize_ggml_view_impl(tensor);
    if(!ggml_tt_tensors_shape_equal(tensor, *res)) {
        std::cout << "FATAL ERROR: Shape mismatch between TTNN and GGML after view op " << ggml_op_name(tensor->op) << "\n"
            << "  Result: " << res->shape() << "\n"
            << "  GGML expecting: " << tensor->ne[3] << " " << tensor->ne[2] << " " << tensor->ne[1] << " " << tensor->ne[0] << "\n";
        GGML_ASSERT(ggml_tt_tensors_shape_equal(tensor, *res));
    }
    return res;
}


static std::shared_ptr<tt::tt_metal::Tensor> realize_ggml_view_impl(const ggml_tensor* tensor)
{
    // Since TTNN does not support the traditional view operation, we had to support it ourselves
    // This function, realize, extracts the data from the source tensor and creates a new tensor
    // that is separate from the source tensor. DO NOT eagerly call this function

    ggml_tensor* src0 = tensor->src[0];
    ggml_op op = tensor->op;


    // Do we really need to lazy evaluate this? Currently transpose is eagerly evaluated
    if(op == GGML_OP_TRANSPOSE) {
        auto patent = realize_ggml_view(src0);
        auto res = ttnn::transpose(*patent, -2, -1);
        return std::make_shared<tt::tt_metal::Tensor>(res);
    }
    if(op == GGML_OP_VIEW) {

        std::shared_ptr<tt::tt_metal::Tensor> parent = realize_ggml_view(tensor->view_src);
        std::array dst_size = std::to_array(tensor->ne);
        std::array dst_stride = std::to_array(tensor->nb);
        std::array src_size = std::to_array(src0->ne);
        std::array src_stride = std::to_array(src0->nb);
        size_t offset = tensor->view_offs;
        ggml_backend_metalium_buffer_context* bufctx = ((TensorWithMetadata*)tensor->extra)->bufctx;

        // TODO: Generalize this to use permute instead of transpose
        // FIXME: This is failing views in test-backend-ops
        // std::optional<std::pair<uint32_t, uint32_t>> axisswap;
        // for (int i = 0; i < ggml_n_dims(tensor); ++i) {
        //     size_t expected_stride = tensor->nb[0];
        //     for (int j = 0; j < i; ++j) {
        //         expected_stride *= tensor->ne[j];
        //     }
        //     // std::cout << "  Axis " << i << " stride: " << tensor->nb[i] << " expected: " << expected_stride << std::endl;
        //     if (tensor->nb[i] != expected_stride) {
        //         if (!axisswap) {
        //             axisswap = std::make_pair(i, 1000);
        //         } else if (axisswap->second == 1000) {
        //             axisswap->second = i;
        //         } else {
        //             GGML_ASSERT(false && "More than one axis swap detected");
        //         }
        //     }
        // }
        // TODO: Do something with axisswap. I think some ops needs this but it haven't crashed yet

        // Fast path if we can just return the parent tensor (view is a no-op)
        if(dst_size == src_size && dst_stride == src_stride && offset == 0) {
            return parent;
        }
        std::array<uint32_t, GGML_MAX_DIMS> start;
        std::array<uint32_t, GGML_MAX_DIMS> end;

        // FIXME: Does not work when we are viewing into a permuted tensor. Sucks
        size_t remaining_offset = offset;
        for(size_t i = GGML_MAX_DIMS - 1; i < GGML_MAX_DIMS; i--) {
            start[i] = remaining_offset / src_stride[i];
            end[i] = dst_size[i] + start[i];
            remaining_offset = remaining_offset % src_stride[i];
        }
        std::reverse(start.begin(), start.end());
        std::reverse(end.begin(), end.end());
        tt::tt_metal::Tensor res;

        if(g_debug_flags.print_view) {
            // Debug prints to help debug complicated view operations
            std::cout << "\nrealize_ggml_view() OP: " << ggml_op_desc(tensor) << std::endl;
            std::cout << "  dst name: " << tensor->name << std::endl;
            std::cout << "  dst shape: " << tensor->ne[0] << " " << tensor->ne[1] << " " << tensor->ne[2] << " " << tensor->ne[3] << std::endl;
            std::cout << "  dst stride: " << tensor->nb[0] << " " << tensor->nb[1] << " " << tensor->nb[2] << " " << tensor->nb[3] << std::endl;
            std::cout << "  dst extra: " << tensor->extra << std::endl;
            if(tensor->extra != nullptr) {
                TensorWithMetadata* meta = (TensorWithMetadata*)tensor->extra;
                std::cout << "  dst tensor: " << meta->tensor << std::endl;
                if(meta->tensor != nullptr) {
                    std::cout << "  dst tensor shape: " << meta->tensor->shape() << std::endl;
                }
            }
            std::cout << "  dst data: " << tensor->data << std::endl;
            std::cout << "  dst view_src: " << tensor->view_src << std::endl;
            std::cout << "  dst view_src shape: " << tensor->view_src->ne[0] << " " << tensor->view_src->ne[1] << " " << tensor->view_src->ne[2] << " " << tensor->view_src->ne[3] << std::endl;
            std::cout << "  dst view_src stride: " << tensor->view_src->nb[0] << " " << tensor->view_src->nb[1] << " " << tensor->view_src->nb[2] << " " << tensor->view_src->nb[3] << std::endl;
            std::cout << "  dst src0: " << src0 << std::endl;
            std::cout << "  dst src1: " << tensor->src[1] << std::endl;
            std::cout << "  src0 shape: " << src0->ne[0] << " " << src0->ne[1] << " " << src0->ne[2] << " " << src0->ne[3] << std::endl;
            std::cout << "  src0 stride: " << src0->nb[0] << " " << src0->nb[1] << " " << src0->nb[2] << " " << src0->nb[3] << std::endl;
            std::cout << "  src0 OP: " << ggml_op_desc(src0) << std::endl;
            std::cout << "  TT parent shape: " << parent->shape() << std::endl;
            std::cout << "  TT slice start: " << start[0] << " " << start[1] << " " << start[2] << " " << start[3] << std::endl;
            std::cout << "  TT slice end: " << end[0] << " " << end[1] << " " << end[2] << " " << end[3] << std::endl;
        }

        // Actually a reshape written as a slice
        if(offset == 0 && ggml_nelements(src0) == ggml_nelements(tensor)) {
            res = reshape_tt_tensor_into_ggml(*parent, tensor);
        }
        // Trying to convert a flat 1D tensor to N-D tensor (potentially with an offset)
        else if(ggml_n_dims(src0) == 1 && ggml_n_dims(tensor) > 1) {
            // slow: grab the source tensor and unpad it
            uint32_t offset_elements = offset / ggml_type_size(src0->type);
            ttnn::SimpleShape start{0, 0, 0, offset_elements};
            auto dst_volume = ggml_nelements(tensor);
            ttnn::SimpleShape end({1, 1, 1, uint32_t(dst_volume) + offset_elements});
            auto t = ttnn::untilize(*parent).cpu().unpad(start, end);
            res = reshape_host_tt_tensor_into_ggml(t, parent->device(), tensor);
        }
        // The fast path, this is what TTNN is designed for
        else if(dst_size[0] % tt::constants::TILE_WIDTH == 0 && dst_size[1] % tt::constants::TILE_HEIGHT == 0 &&
            start[2] % tt::constants::TILE_WIDTH == 0 && start[3] % tt::constants::TILE_HEIGHT == 0) {
            std::array<uint32_t, GGML_MAX_DIMS> step = {1, 1, 1, 1};
            res = ttnn::slice(*parent, start, end, step, tt::tt_metal::MemoryConfig());
        }
        // Unpad on the CPU and then pad back on the device
        else {
            // THIS is EXTREMELY SLOW. But it works
            tt::tt_metal::Tensor tmp = ttnn::untilize(*parent).cpu().unpad(ttnn::SimpleShape(start), ttnn::SimpleShape(end));
            res = ttnn::tilize_with_zero_padding(tmp.to(bufctx->device));
        }
        return std::make_shared<tt::tt_metal::Tensor>(res);
    }
    if(op == GGML_OP_RESHAPE) {
        auto t = realize_ggml_view(src0);
        return std::make_shared<tt::tt_metal::Tensor>(reshape_tt_tensor_into_ggml(*t, tensor));
    }
    if(op == GGML_OP_PERMUTE) {
        std::array<int32_t, GGML_MAX_DIMS> permute;
        memcpy(permute.data(), tensor->op_params, sizeof(permute));

        int ndiff = 0;
        for(int i=0;i<GGML_MAX_DIMS;i++) {
            ndiff += permute[i] != i;
        }
        GGML_ASSERT(ndiff != 1); // Logically impossible

        auto t = realize_ggml_view(src0);
        if(ndiff == 0) {
            return t;
        }

        std::array<int64_t, GGML_MAX_DIMS> permute_tt;
        for(int i=0;i<GGML_MAX_DIMS;i++) {
            permute_tt[i] = GGML_MAX_DIMS - permute[GGML_MAX_DIMS - i - 1] - 1;
        }

        auto res = ttnn::permute(*t, permute_tt);
        return std::make_shared<tt::tt_metal::Tensor>(std::move(res));
    }

    if(TensorWithMetadata* meta = (TensorWithMetadata*)tensor->extra; meta != nullptr && meta->tensor != nullptr) {
        return meta->tensor;
    }

    // TensorWithMetadata* meta = (TensorWithMetadata*)tensor->extra;
    // std::cerr << "meta = " << (void*)meta << std::endl;
    // std::cerr << "meta->tensor = " << (void*)(meta->tensor.get()) << std::endl;
    // std::cout << "tensor->name = " << tensor->name << std::endl;
    // std::cout << "tensor->op = " << ggml_op_name(tensor->op) << std::endl;
    GGML_ASSERT(is_view(tensor));
    GGML_ASSERT(tensor->view_src != nullptr);

    // recursivly resolve the source tensor
    // TODO: Should it even reach here?
    return realize_ggml_view(tensor->view_src);
}

// Sanity check macros to ensure that the tensors are in the correct format and we won't crash
#define GGML_METALIUM_OP_SANITY_CHECK(_node) \
    GGML_ASSERT((_node)->extra != NULL);
// Check if the tensor is on the device (so we wont'e be using the CPU) as well as letting us crash early
#define GGML_METALIUM_OP_SRC_SANITY_CHECK(_node, _idx) \
    do { \
        GGML_ASSERT((_node)->src[_idx] != NULL); \
        GGML_ASSERT((_node)->src[_idx]->extra != NULL); \
        auto _meta = (TensorWithMetadata*)((_node)->src[_idx]->extra); \
        if(_meta->tensor != NULL) { \
        GGML_ASSERT(_meta->tensor->storage_type() == tt::tt_metal::StorageType::DEVICE || _meta->tensor->storage_type() == tt::tt_metal::StorageType::MULTI_DEVICE); \
        GGML_ASSERT(_meta->tensor->layout() == tt::tt_metal::Layout::TILE); }\
    } while(0)
#define GGML_METALIUM_OP_SRC0_SANITY_CHECK(_node) GGML_METALIUM_OP_SRC_SANITY_CHECK(_node, 0)
#define GGML_METALIUM_OP_SRC1_SANITY_CHECK(_node) GGML_METALIUM_OP_SRC_SANITY_CHECK(_node, 1)

static bool ggml_backend_metalium_can_mul_mat(const struct ggml_tensor * dst)
{
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    // TTNN only supports matmul of shape [B, 1, M, K] x [1, 1, K, N] (bcast_batch=True)
    // or [B, 1, M, K] x [B, 1, K, N] (bcast_batch=False)
    // For now we simply only allow those shapes. We transpose the shapes ourselves
    // TODO: Detect when shape[1] can be removed and do that automagically

    return src0->ne[0] == src1->ne[0] && src0->ne[2] == 1 && src1->ne[2] == 1 &&
        (src0->ne[3] == src1->ne[3] || src0->ne[3] == 1);
}

static void ggml_backend_metalium_mul_mat(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst) {
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC1_SANITY_CHECK(dst);

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const enum ggml_type type = src0->type;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(src0->extra != NULL);
    GGML_ASSERT(src1->extra != NULL);
    GGML_ASSERT(dst->extra != NULL);

    auto ap = realize_ggml_view(src0);
    auto bp = realize_ggml_view(src1);
    auto &a = *ap;
    auto &b = *bp;
    TensorWithMetadata* cm = (TensorWithMetadata*)dst->extra;

    GGML_ASSERT(cm != NULL);

    if(a.dtype() == tt::tt_metal::DataType::BFLOAT16 && b.dtype() == tt::tt_metal::DataType::BFLOAT16) {
        // Fast path
        // Need to increase the math fidelity as moreh_matmul by default uses LoFi and won't pass GGML unit tests
        ttnn::DeviceComputeKernelConfig cfg = make_compute_kernel_config(a.device());
        *cm = {
            .tensor = std::make_shared<tt::tt_metal::Tensor>(ttnn::moreh_matmul(b, a, false, true, std::nullopt, std::nullopt, std::nullopt, cfg)),
            .ggtype = dst->type,
            .bufctx = cm->bufctx
        };
    }
    else {
        auto aT = ttnn::transpose(a, -2, -1);
        // TODO: Ask TT to support multiplication of pre-transposed tensors. Calling transpose here is inefficient
        // https://github.com/tenstorrent/tt-metal/issues/9709
        ttnn::operations::matmul::Matmul cfg = ttnn::operations::matmul::Matmul{
            .compute_kernel_config = make_compute_kernel_config(a.device()),
            // XXX: Why output_tile doesn't have a default value?
            .output_tile = std::nullopt
        };
        *cm = {
            .tensor = std::make_shared<tt::tt_metal::Tensor>(ttnn::operations::matmul::matmul(b, aT, std::nullopt, cfg)),
            .ggtype = dst->type,
            .bufctx = cm->bufctx
        };
    }
    GGML_ASSERT(cm->tensor->storage_type() == tt::tt_metal::StorageType::DEVICE || cm->tensor->storage_type() == tt::tt_metal::StorageType::MULTI_DEVICE);
    GGML_UNUSED(ctx);
}

static void ggml_backend_metalium_cpy(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    TensorWithMetadata* dst_meta = (TensorWithMetadata*)dst->extra;
    ggml_tensor* src0 = dst->src[0];

    // TODO: Check we are not writing into a view
    auto res = realize_ggml_view(src0);
    if(!ggml_tt_tensors_shape_equal(dst, *res)) {
        res = std::make_shared<tt::tt_metal::Tensor>(reshape_tt_tensor_into_ggml(*res, dst));
    }

    *dst_meta = {
        // TODO: Type cast to the appropriate type
        .tensor = res,
        .ggtype = dst->type,
        .bufctx = dst_meta->bufctx
    };
}

static bool ggml_backend_metalium_activations(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst, ggml_unary_op op) {
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    GGML_UNUSED(ctx);

    const struct ggml_tensor * src0 = dst->src[0];
    TensorWithMetadata* meta = (TensorWithMetadata*)src0->extra;
    TensorWithMetadata* dst_meta = (TensorWithMetadata*)dst->extra;

    auto src_tensor = realize_ggml_view(src0);

    tt::tt_metal::Tensor ret;
    switch (op) {
        case GGML_UNARY_OP_ABS:
            ret = ttnn::abs(*src_tensor);
            break;
        case GGML_UNARY_OP_SGN:
            ret = ttnn::sign(*src_tensor);
            break;
        case GGML_UNARY_OP_NEG:
            ret = ttnn::neg(*src_tensor);
            break;
        // Not accurate enough to pass unit tests
        case GGML_UNARY_OP_TANH:
            ret = ttnn::tanh(*src_tensor);
            break;
        case GGML_UNARY_OP_ELU:
            ret = ttnn::elu(*src_tensor, 1.0f);
            break;
        case GGML_UNARY_OP_RELU:
            ret = ttnn::relu(*src_tensor);
            break;
        // Not accurate enough to pass unit tests
        case GGML_UNARY_OP_SIGMOID:
            ret = ttnn::sigmoid(*src_tensor);
            break;
        case GGML_UNARY_OP_GELU:
            ret = ttnn::gelu(*src_tensor, false);
            break;
        case GGML_UNARY_OP_GELU_QUICK:
            ret = ttnn::gelu(*src_tensor);
            break;
        case GGML_UNARY_OP_SILU:
            ret = ttnn::silu(*src_tensor);
            break;
        case GGML_UNARY_OP_HARDSWISH:
            ret = ttnn::hardswish(*src_tensor, 1.f/6.f, 0.5);
            break;
        case GGML_UNARY_OP_HARDSIGMOID:
            ret = ttnn::hardsigmoid(*src_tensor, 1.f/6.f, 0.5);
            break;
        case GGML_UNARY_OP_STEP:
            // TODO: Make sure the resulting data type matches the input
            ret = ttnn::experimental::typecast(ttnn::gtz(*src_tensor), ggml2tt_type(dst->type, src_tensor->device()->arch()));
            break;
        case GGML_UNARY_OP_EXP:
            ret = ttnn::exp(*src_tensor);
            break;
        default:
            return false;
    }
    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(std::move(ret)),
        .ggtype = dst->type,
        .bufctx = meta->bufctx
    };
    return true;
}
static void ggml_backend_metalium_leaky_relu(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst) {
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    GGML_UNUSED(ctx);

    const struct ggml_tensor * src0 = dst->src[0];
    TensorWithMetadata* meta = (TensorWithMetadata*)src0->extra;
    TensorWithMetadata* dst_meta = (TensorWithMetadata*)dst->extra;
    auto src_tensor = realize_ggml_view(src0);

    float negative_slope;
    GGML_ASSERT(dst->op_params != NULL);
    memcpy(&negative_slope, dst->op_params, sizeof(float));

    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(ttnn::leaky_relu(*src_tensor, negative_slope)),
        .ggtype = dst->type,
        .bufctx = meta->bufctx
    };
}
static void ggml_backend_metalium_bin_op(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst, ggml_op op) {
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC1_SANITY_CHECK(dst);
    GGML_UNUSED(ctx);

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    TensorWithMetadata* meta0 = (TensorWithMetadata*)src0->extra;
    TensorWithMetadata* dst_meta = (TensorWithMetadata*)dst->extra;

    auto src_tensor0 = realize_ggml_view(src0);
    auto src_tensor1 = realize_ggml_view(src1);

    std::shared_ptr<tt::tt_metal::Tensor> ret;
    switch(op) {
        case GGML_OP_ADD:
            ret = std::make_shared<tt::tt_metal::Tensor>(ttnn::add(*src_tensor0, *src_tensor1));
            break;
        case GGML_OP_MUL:
            ret = std::make_shared<tt::tt_metal::Tensor>(ttnn::multiply(*src_tensor0, *src_tensor1));
            break;
        case GGML_OP_SUB:
            ret = std::make_shared<tt::tt_metal::Tensor>(ttnn::subtract(*src_tensor0, *src_tensor1));
            break;
        case GGML_OP_DIV:
            ret = std::make_shared<tt::tt_metal::Tensor>(ttnn::divide(*src_tensor0, *src_tensor1));
            break;
        default:
            GGML_ASSERT(false && "Unsupported binary operation");
    }
    *dst_meta = {
        .tensor = std::move(ret),
        .ggtype = dst->type,
        .bufctx = meta0->bufctx
    };
}

static bool ggml_backend_metalium_can_set(const struct ggml_tensor * dst)
{
    int32_t params[5];
    memcpy(params, dst->op_params, sizeof(params));
    auto [nb1, nb2, nb3, offset, inplace] = std::to_array(params);

    if(offset >= nb3 || offset % nb1 != 0 || ggml_n_dims(dst->src[0]) < ggml_n_dims(dst->src[1]) ||
        ggml_n_dims(dst->src[1]) != 1) {
        return false;
    }

    return true;
}

static void ggml_backend_metalium_set(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst)
{
    GGML_UNUSED(ctx);
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC1_SANITY_CHECK(dst);

    TensorWithMetadata* dst_meta = (TensorWithMetadata*)dst->extra;
    TensorWithMetadata* src0_meta = (TensorWithMetadata*)dst->src[0]->extra;
    TensorWithMetadata* src1_meta = (TensorWithMetadata*)dst->src[1]->extra;

    int32_t params[5];
    memcpy(params, dst->op_params, sizeof(params));
    auto [nb1, nb2, nb3, offset, inplace] = std::to_array(params);

    int idx = offset / nb1;
    int batch_idx = offset / nb2;
    GGML_ASSERT(offset < nb3);
    GGML_ASSERT(offset % nb1 == 0);
    auto res = ttnn::update_cache(*src0_meta->tensor, *src1_meta->tensor, idx, batch_idx);
    if(!inplace) {
        *dst_meta = {
            .tensor = std::make_shared<tt::tt_metal::Tensor>(res),
            .ggtype = dst->type,
            .bufctx = src0_meta->bufctx
        };
    }
    else {
        std::shared_ptr<tt::tt_metal::Tensor> tensor = std::make_shared<tt::tt_metal::Tensor>(res);
        *src0_meta = {
            .tensor = tensor,
            .ggtype = dst->type,
            .bufctx = src0_meta->bufctx
        };
        *dst_meta = {
            .tensor = tensor,
            .ggtype = dst->type,
            .bufctx = src0_meta->bufctx
        };
    }
}
static void ggml_backend_metalium_clamp(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst)
{
    GGML_UNUSED(ctx);
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);

    TensorWithMetadata* dst_meta = (TensorWithMetadata*)dst->extra;

    float data[2];
    memcpy(data, dst->op_params, sizeof(data));
    auto [min, max] = std::to_array(data);

    auto t = realize_ggml_view(dst->src[0]);
    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(ttnn::clamp(*t, min, max)),
        .ggtype = dst->type,
        .bufctx = ((TensorWithMetadata*)dst->src[0]->extra)->bufctx
    };
}

static void ggml_backend_metalium_scale(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst)
{
    GGML_UNUSED(ctx);
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);

    TensorWithMetadata* dst_meta = (TensorWithMetadata*)dst->extra;

    float scale;
    memcpy(&scale, dst->op_params, sizeof(scale));

    auto t = realize_ggml_view(dst->src[0]);
    auto res = ttnn::multiply(*t, scale);
    // TODO: Support in-place scaling
    GGML_ASSERT(!is_view(dst->src[0]));
    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(std::move(res)),
        .ggtype = dst->type,
        .bufctx = ((TensorWithMetadata*)dst->src[0]->extra)->bufctx
    };
}

static bool ggml_backend_metalium_can_get_row(const struct ggml_tensor * dst)
{
    const ggml_tensor *idxs = dst->src[1];
    if(idxs->ne[0] != 1 || idxs->ne[1] != 1 || idxs->ne[2] != 1 || idxs->ne[3] != 1) {
        return false;
    }
    return true;
}

static void ggml_backend_metalium_get_row(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst)
{
    GGML_UNUSED(ctx);
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);

    TensorWithMetadata* dst_meta = (TensorWithMetadata*)dst->extra;
    uint32_t idx = *(uint32_t*)dst->src[1]->data;

    auto t = realize_ggml_view(dst->src[0]);
    auto res = ttnn::experimental::nlp_kv_cache_load_slice(*t, idx, idx + 1);
    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(res),
        .ggtype = dst->type,
        .bufctx = ((TensorWithMetadata*)dst->src[0]->extra)->bufctx
    };
}

static void ggml_backend_metalium_norm(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst, bool rms)
{
    GGML_UNUSED(ctx);
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);

    TensorWithMetadata* dst_meta = (TensorWithMetadata*)dst->extra;

    float esp = 0;
    memcpy(&esp, dst->op_params, sizeof(esp));

    auto t = realize_ggml_view(dst->src[0]);
    tt::tt_metal::Tensor res;
    if(rms) {
        res = ttnn::rms_norm(*t, esp);
    }
    else {
        res = ttnn::layer_norm(*t, esp);
    }
    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(std::move(res)),
        .ggtype = dst->type,
        .bufctx = ((TensorWithMetadata*)dst->src[0]->extra)->bufctx
    };
}

static void ggml_backend_metalium_add1(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst)
{
    GGML_UNUSED(ctx);
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);

    TensorWithMetadata* dst_meta = (TensorWithMetadata*)dst->extra;

    float esp = 0;
    memcpy(&esp, dst->op_params, sizeof(esp));

    auto t = realize_ggml_view(dst->src[0]);
    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(ttnn::add(*t, 1.f)),
        .ggtype = dst->type,
        .bufctx = ((TensorWithMetadata*)dst->src[0]->extra)->bufctx
    };
}

static void ggml_backend_metalium_sqrt(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst)
{
    GGML_UNUSED(ctx);
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);

    TensorWithMetadata* dst_meta = (TensorWithMetadata*)dst->extra;

    float esp = 0;
    memcpy(&esp, dst->op_params, sizeof(esp));

    auto t = realize_ggml_view(dst->src[0]);
    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(ttnn::sqrt(*t)),
        .ggtype = dst->type,
        .bufctx = ((TensorWithMetadata*)dst->src[0]->extra)->bufctx
    };
}

static void ggml_backend_metalium_sqr(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst)
{
    GGML_UNUSED(ctx);
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);

    TensorWithMetadata* dst_meta = (TensorWithMetadata*)dst->extra;

    float esp = 0;
    memcpy(&esp, dst->op_params, sizeof(esp));

    auto t = realize_ggml_view(dst->src[0]);
    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(ttnn::square(*t)),
        .ggtype = dst->type,
        .bufctx = ((TensorWithMetadata*)dst->src[0]->extra)->bufctx
    };
}

static bool ggml_backend_metalium_can_concat(const struct ggml_tensor * dst)
{
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(dst->op_params != NULL);

    int32_t dim = 0;
    memcpy(&dim, dst->op_params, sizeof(dim));

    // TTNN requires tensors to be tile aligned if concat on the last 2 dimensions
    if(dim == 0 || dim == 1) {
        return src0->ne[dim] % 32 == 0 && src1->ne[dim] % 32 == 0;
    }
    return true;
}

static void ggml_backend_metalium_concat(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst)
{
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC1_SANITY_CHECK(dst);
    GGML_UNUSED(ctx);

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    TensorWithMetadata* dst_meta = (TensorWithMetadata*)dst->extra;

    auto src_tensor0 = realize_ggml_view(src0);
    auto src_tensor1 = realize_ggml_view(src1);

    int32_t axis = 0;
    memcpy(&axis, dst->op_params, sizeof(axis));
    axis = GGML_MAX_DIMS - axis - 1;

    std::vector<tt::tt_metal::Tensor> targets = {*src_tensor0, *src_tensor1};
    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(ttnn::concat(targets, axis)),
        .ggtype = dst->type,
        .bufctx = ((TensorWithMetadata*)dst->src[0]->extra)->bufctx
    };
}

static bool ggml_backend_metalium_can_softmax(const struct ggml_tensor * dst)
{
    float arr[2];
    memcpy(arr, dst->op_params, sizeof(arr));
    auto [scale, max_bias] = arr;
    if(dst->src[1] != nullptr && max_bias != 0.f) {
        return false;
    }
    if(dst->src[1] != nullptr) {
        // TinyLLaMA somehow has x [1, 32, 1, 32] and mask [1, 1, 32, 32]
        // Don't know what's this about
        // FIXME: This masks a problem in RWKV. Need proper fix
        const ggml_tensor *src1 = dst->src[1];
        return numpy_broadcast_rule(src1, dst) && dst->ne[1] == src1->ne[1];
    }
    return true;
}

static void ggml_backend_metalium_softmax(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst)
{
    GGML_UNUSED(ctx);
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);

    TensorWithMetadata* dst_meta = (TensorWithMetadata*)dst->extra;

    std::array<float, 2> params;
    memcpy(&params, dst->op_params, sizeof(params));
    auto [scale, max_bias] = params;

    const ggml_tensor *src1 = dst->src[1];

    auto t = realize_ggml_view(dst->src[0]);
    tt::tt_metal::Tensor x = *t;
    // TODO: use the operimzied op if we can. It only works in certain conidtions
    // if(src1 != nullptr) {
    //     auto mask = realize_ggml_view(src1);
    //     x = ttnn::operations::normalization::scale_mask_softmax(*t, scale, *mask);
    // }
    if(scale != 1.f) {
        x = ttnn::multiply(*t, scale);
    }

    if(src1 != nullptr) {
        auto mask = realize_ggml_view(src1);
        if(max_bias == 0.f) {
            // std::cout << "x: " << x.shape() << " mask: " << mask->shape() << std::endl;
            // std::cout << "x.dtype: " << (int)x.dtype() << " mask.dtype: " << (int)mask->dtype() << std::endl;
            x = ttnn::add(x, *mask);
        }
        else {
            // This path is not used due to bugs
            // TODO: Revive it later
            const uint32_t n_head = t->shape()[1];
            const uint32_t n_head_log2 = 1u << (uint32_t) std::floor(std::log2(n_head));
            const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
            const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);
            auto make_tile = [](const tt::tt_metal::Tensor& t, tt::tt_metal::Device* dev) {
                return ttnn::tilize_with_zero_padding(t.to(dev));
            };

            // const float slope = (max_bias > 0.0f) ? h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_head_log2) + 1) : 1.0f;
            auto *dev = t->device();
            // BUG here. Generating wrong shaped tensor
            // This is a part of the limitation of TTNN can't have odd numbers of elements in the last dimension
            auto idxs = make_tile(ttnn::arange(0, n_head, 1), dev);
            auto slope = ttnn::where(ttnn::lt(idxs, (float)n_head_log2), ttnn::rpow(ttnn::add(idxs, 1.f), m0)
                , ttnn::rpow(ttnn::add(ttnn::multiply(ttnn::subtract(idxs, (float)n_head_log2), 2.f), 1.f), m1));
            auto positional_bias = ttnn::matmul(slope, ttnn::transpose(idxs, -2, -1)); // FIXME: make sure this is correct

            x = ttnn::add(x, ttnn::multiply(*mask, positional_bias));
        }
    }
    ttnn::DeviceComputeKernelConfig cfg = make_compute_kernel_config(x.device());
    x = ttnn::operations::normalization::softmax(x, tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG, cfg, true);
    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(std::move(x)),
        .ggtype = dst->type,
        .bufctx = ((TensorWithMetadata*)dst->src[0]->extra)->bufctx
    };
}

static void ggml_backend_metalium_cos(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst)
{
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    GGML_UNUSED(ctx);

    const struct ggml_tensor * src0 = dst->src[0];
    TensorWithMetadata* dst_meta = (TensorWithMetadata*)dst->extra;

    auto src = realize_ggml_view(src0);
    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(ttnn::cos(*src)),
        .ggtype = dst->type,
        .bufctx = ((TensorWithMetadata*)src0->extra)->bufctx
    };
}

static void ggml_backend_metalium_sin(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst)
{
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    GGML_UNUSED(ctx);

    const struct ggml_tensor * src0 = dst->src[0];
    TensorWithMetadata* dst_meta = (TensorWithMetadata*)dst->extra;

    auto src = realize_ggml_view(src0);
    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(ttnn::sin(*src)),
        .ggtype = dst->type,
        .bufctx = ((TensorWithMetadata*)src0->extra)->bufctx
    };
}

static void ggml_backend_metalium_log(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst)
{
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    GGML_UNUSED(ctx);

    const struct ggml_tensor * src0 = dst->src[0];
    TensorWithMetadata* dst_meta = (TensorWithMetadata*)dst->extra;

    auto src = realize_ggml_view(src0);
    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(ttnn::log(*src)),
        .ggtype = dst->type,
        .bufctx = ((TensorWithMetadata*)src0->extra)->bufctx
    };
}

static void ggml_backend_metalium_arange(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst)
{
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_UNUSED(ctx);

    TensorWithMetadata* dst_meta = (TensorWithMetadata*)dst->extra;
    auto* device = dst_meta->bufctx->device;
    std::array<float, 3> params;
    memcpy(&params, dst->op_params, sizeof(params));
    auto [start, end, step] = params;
    auto dtype = ggml2tt_type(dst->type, device->arch());
    if(dtype == tt::tt_metal::DataType::INVALID) {
        tt::log_fatal(tt::LogType::LogAlways, "Unsupported GGML type {}", ggml_type_name(dst->type));
        GGML_ASSERT(false && "Unsupported GGML type");
    }

    // TODO: Request TT to support arange directly on the device
    auto tensor = ttnn::arange(start, end, step, dtype);
    tensor = ttnn::tilize_with_zero_padding(tensor.to(device));
    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(std::move(tensor)),
        .ggtype = dst->type,
        .bufctx = dst_meta->bufctx
    };
}

static void ggml_backend_metalium_group_norm(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst)
{
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_UNUSED(ctx);

    TensorWithMetadata* dst_meta = (TensorWithMetadata*)dst->extra;
    int n_groups;
    float eps;
    memcpy(&n_groups, dst->op_params, sizeof(n_groups));
    memcpy(&eps, dst->op_params + sizeof(n_groups), sizeof(eps));

    // XXX: Moreh's operators needs some cleanup
    auto tensor = realize_ggml_view(dst->src[0]);
    auto res = ttnn::moreh_group_norm(
        *tensor,
        n_groups,
        eps,
        std::nullopt,
        std::nullopt,
        std::vector<bool>{true, false, false},
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt);
    GGML_ASSERT(res[0].has_value());
    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(*res[0]),
        .ggtype = dst->type,
        .bufctx = dst_meta->bufctx
    };
}

static bool ggml_backend_metalium_can_repeat(const struct ggml_tensor * dst)
{
    ggml_tensor *src0 = dst->src[0];
    for(int i = 0; i < GGML_MAX_DIMS; i++) {
        if(dst->ne[i] % src0->ne[i] != 0) {
            return false;
        }

        // FIXME: TTNN has trouble repeating if the first 2 dimensions are not the same
        if(i < 2 && dst->ne[i] / src0->ne[i] != 1) {
            return false;
        }
    }
    return true;
}

static void ggml_backend_metalium_repeat(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst)
{
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    GGML_UNUSED(ctx);

    TensorWithMetadata* dst_meta = (TensorWithMetadata*)dst->extra;
    ggml_tensor* src0 = dst->src[0];

    auto tensor = realize_ggml_view(dst->src[0]);
    std::array<uint32_t, GGML_MAX_DIMS> repeats;
    int ndiff = 0;
    for(int i = 0; i < GGML_MAX_DIMS; i++) {
        auto repeat = dst->ne[i] / src0->ne[i];
        repeats[GGML_MAX_DIMS - i - 1] = repeat;
        ndiff += (repeat != 1);
    }
    if(ndiff == 0) {
        *dst_meta = {
            .tensor = std::make_shared<tt::tt_metal::Tensor>(*tensor),
            .ggtype = dst->type,
            .bufctx = ((TensorWithMetadata*)src0->extra)->bufctx
        };
        return;
    }

    auto res = ttnn::repeat(*tensor, repeats);
    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(res),
        .ggtype = dst->type,
        .bufctx = ((TensorWithMetadata*)src0->extra)->bufctx
    };
}

static bool ggml_backend_metalium_can_outer_product(const struct ggml_tensor * dst)
{
    auto num_ones_in_shape = [](const ggml_tensor * t) {
        int num_ones = 0;
        for(int i = 0; i < GGML_MAX_DIMS; i++) {
            if(t->ne[i] == 1) {
                num_ones++;
            }
        }
        return num_ones;
    };
    return num_ones_in_shape(dst->src[0]) >= 3 && num_ones_in_shape(dst->src[1]) >= 3;
}

static void ggml_backend_metalium_outer_product(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst)
{
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC1_SANITY_CHECK(dst);
    GGML_UNUSED(ctx);

    TensorWithMetadata* dst_meta = (TensorWithMetadata*)dst->extra;
    TensorWithMetadata* src0_meta = (TensorWithMetadata*)dst->src[0]->extra;

    auto src0 = realize_ggml_view(dst->src[0]);
    auto src1 = realize_ggml_view(dst->src[1]);

    std::cout << "src0: " << src0->shape() << " src1: " << src1->shape() << std::endl;

    auto res = ttnn::outer(*src0, *src1);
    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(res),
        .ggtype = dst->type,
        .bufctx = src0_meta->bufctx
    };
}

// backend interface

static const char * ggml_backend_metalium_name(ggml_backend_t backend) {
    return "Metalium";

    GGML_UNUSED(backend);
}

static void ggml_backend_metalium_free(ggml_backend_t backend) {
    ggml_backend_metalium_context * ctx = (ggml_backend_metalium_context *)backend->context;
    delete ctx;
    delete backend;
}

struct ggml_backend_metalium_buffer_type_context {
    ttnn::distributed::MeshDevice* device = nullptr;
    std::string name;
};

static const char * ggml_backend_metalium_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_metalium_buffer_type_context * ctx = (ggml_backend_metalium_buffer_type_context *)buft->context;

    return ctx->name.c_str();
}

static size_t ggml_backend_metalium_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    // Not using this. Metalium's allication model is not compatible with GGML's allocator
    return 128;
    GGML_UNUSED(buft);
}

// NOTE: I might need to add a metalium tensor wrapper to work around TT tensors have hardware-tagged data types
//       and GGML tensors does not specify the data type during tensor creation.
static size_t ggml_backend_metalium_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_backend_metalium_buffer_type_context * ctx = (ggml_backend_metalium_buffer_type_context *)buft->context;
    auto* device = ctx->device;
    size_t max_size = 0;
    for(size_t i = 0; i < device->num_devices(); i++) {
        auto* dev = device->get_device_index(i);
        max_size += dev->num_dram_channels() * (size_t)dev->dram_size_per_channel();
    }
    return max_size;
}

static size_t ggml_backend_metalium_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    // Not using this. Metalium's allication model is not compatible with GGML's allocator
    return ggml_nbytes(tensor);
    GGML_UNUSED(buft);
}

static void
ggml_backend_metalium_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_metalium_buffer_context * ctx = ( ggml_backend_metalium_buffer_context *)buffer->context;
    delete ctx;
}

static void ggml_backend_metalium_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                ggml_tensor *tensor,
                                                const void *data, size_t offset,
                                                size_t size)
{
    // Here's the general logic of set_tensor
    // 1. Make a flat buffer and copy the data into it
    //    - If the data is quantized, convert it to BFLOAT16
    //    - Try to directly copy the data if it is already in the correct format
    // 2. Create a TT tensor from the flat buffer as ROW_MAJOR. Send it to the device and tile it
    // 3. If the data is quantized, cast down to BFLOAT8_B or BFLOAT4_B
    // There's a lot of things to do here.
    // TODO: On grayskull the best I can do is BFLOAT16 so the final dimension must be a multiple of 2.
    //       But on Wormhole we can use FP32 then the final dimension can be anything. But currently it
    //       is hard coded to BFLOAT16. Use FP32 as intermidate when the hardware supports it and when
    //       it makes sense.
    // TODO: Handle integer data type for Wormhole
    // TODO: Currently FP32 is hard coded to convert to BFLOAT16. Use FP32 when the hardware supports it
    // TODO: Make a scalable way to decide which GGML type casts to TT quantized types
    // TODO: In theory, we can cast BFLOAT16 to FP32, tile it, then cast it back to BFLOAT16, to support
    //       arbitrary tensor dimensions. Do we want to implement this?
    // TODO: Use the simpler tilize() when the final 2 dimensions are both multiples of 32
    // TODO: Check if Metalium tensors can do reuce of there's a way to write to already allocated tensors
    // Must be setting the entire tensor at once
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(tensor->extra != NULL);

    ggml_backend_metalium_buffer_context * bufctx = (ggml_backend_metalium_buffer_context *)buffer->context;
    ggml_type ggtype = tensor->type;
    TensorWithMetadata * meta = (TensorWithMetadata *)tensor->extra;

    // Make sure we are not writing to a view tensor
    if(size != ggml_nbytes(tensor) || (meta->tensor && ggml_tt_tensors_shape_equal(tensor, *meta->tensor) == false)
        || tensor->view_src != NULL) {
        // fprintf(stderr, "Warning: Metalium set_tensor() does not work with tensor views\n");
        return;
    }

    // TODO: Support for FP32 on Wormhole

    // TODO: See if we can use BorrowedStorage to avoid copying the data
    bool source_is_quantized = ggml_is_quantized(ggtype);
    OwnedStorage storage;
    if(ggtype == GGML_TYPE_F32) {
        // For now we cast F32 to BF16. Need a scalable way to handle this as WORMHOLD_B0 have native support for F32
        // TODO: Might want to consider disabling F32 support for Grayskull in the future
        storage = data2owned_storage<float, bfloat16>((const float*)data, size / sizeof(float));
    }
    else if (ggtype == GGML_TYPE_F16) {
        // TT hardware claims to support FP16 but the API does not expose it. For now we use BF16 as it is close enough
        storage = data2owned_storage<ggml_fp16_t, bfloat16>((const ggml_fp16_t*)data, size / sizeof(ggml_fp16_t));
    }
    else if (ggtype == GGML_TYPE_BF16) {
        storage = data2owned_storage<ggml_bf16_t, bfloat16>((const ggml_bf16_t*)data, size / sizeof(ggml_bf16_t));
    }
    else if (source_is_quantized) {
        storage = ggml_quantized2owned_storage<bfloat16>(data, tensor);
    }
    // TODO: Add support for integer data types. Google's Gemma models seems to use them extensively
    else {
        tt::log_fatal(tt::LogType::LogAlways, "Unsupported data type: {}, name '{}', op type: {}\n", ggml_type_name(ggtype), tensor->name, ggml_op_name(tensor->op));
        GGML_ASSERT(false && "Unsupported data type");
    }

    std::vector<uint32_t> shape(GGML_MAX_DIMS, 1);
    for(int i = 0; i < GGML_MAX_DIMS; i++) {
        // GGML stores the shape in reverse order
        shape[i] = tensor->ne[GGML_MAX_DIMS - i - 1];
    }

    std::optional<std::array<int64_t, GGML_MAX_DIMS>> permute;
    // In case GGML sent us a non-contiguous tensor, we need to permute it to make it contiguous
    // We don't care about reshape as that doesn't make a difference in row-major layout
    // TODO: This code does not handle yucky cases like stries of [4, 8, 0, 0] but I assume GGML
    // is decent enough to not send us such tensors
    if(!ggml_is_contiguous(tensor)) {
        // Look at ne (aka strides) and figure out the real underlying shape
        std::array<std::pair<uint64_t, int>, GGML_MAX_DIMS> strides;
        for(int i = 0; i < GGML_MAX_DIMS; i++) {
            strides[i] = {tensor->nb[i], i};
        }
        std::sort(strides.begin(), strides.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });

        std::array<std::pair<uint64_t, int>, GGML_MAX_DIMS> s;
        for(int i = 0; i < GGML_MAX_DIMS; i++) {
            s[i] = {tensor->ne[i], strides[i].second};
        }
        std::sort(s.begin(), s.end(), [](const auto& a, const auto& b) {
            return a.second < b.second;
        });
        for(int i = 0; i < GGML_MAX_DIMS; i++) {
            shape[GGML_MAX_DIMS - i - 1] = s[i].first;
        }

        // Now we can figure out the permutation that we need to apply
        std::array<int64_t, GGML_MAX_DIMS> perm;
        for(int i = 0; i < GGML_MAX_DIMS; i++) {
            perm[strides[i].second] = i;
        }
        permute = perm;
    }

    tt::tt_metal::Tensor t(std::move(storage), ttnn::Shape(shape)
        , tt::tt_metal::DataType::BFLOAT16, tt::tt_metal::Layout::ROW_MAJOR);

    // I think we can allow this.. right?
    // GGML_ASSERT(!bufctx->tensors.contains(offset));
    tt::ARCH processor_class = bufctx->device->arch();
    t = ttnn::tilize_with_zero_padding(t.to(bufctx->device));
    tt::tt_metal::DataType final_type = ggml2tt_type(ggtype, processor_class);
    if(final_type != t.dtype()) {
        t = ttnn::experimental::typecast(t, final_type);
    }
    if(permute.has_value()) {
        t = ttnn::permute(t, permute.value());
    }
    GGML_ASSERT(t.storage_type() == tt::tt_metal::StorageType::DEVICE || t.storage_type() == tt::tt_metal::StorageType::MULTI_DEVICE);
    GGML_ASSERT(t.dtype() == final_type);
    GGML_ASSERT(ggml_tt_tensors_shape_equal(tensor, t));
    *meta = TensorWithMetadata {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(std::move(t)),
        .ggtype = ggtype,
        .bufctx = bufctx
    };
}

static void ggml_backend_metalium_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                const ggml_tensor *tensor,
                                                void *data, size_t offset,
                                                size_t size)
{
    // Here's the general logic of get_tensor
    // 1. Get the TT tensor from the metadata
    // 2. If the TT tensor is quantized, cast it to BFLOAT16
    // 3. Call tensor2ggml to convert the TT tensor to GGML tensor
    //    - tensor2ggml internally handles the data type conversion
    GGML_ASSERT(size == ggml_nbytes(tensor));
    GGML_ASSERT(tensor->extra != NULL);
    GGML_UNUSED(buffer);
    GGML_ASSERT(offset == 0);

    ggml_type dst_ggtype = tensor->type;

    // auto *meta = (TensorWithMetadata*)tensor->extra;
    // auto shape = meta->tensor->shape();
    // std::cout << "get_tensor():\n";
    // std::cout << "  GGML thinks shape: " << tensor->ne[0] << " " << tensor->ne[1] << " " << tensor->ne[2] << " " << tensor->ne[3] << std::endl;
    // std::cout << "  TTNN thinks shape: " << shape << std::endl;
    std::shared_ptr<tt::tt_metal::Tensor> t;
    if(tensor->op == GGML_OP_TRANSPOSE) {
        // HACK: Yeah this one is stupid. GGML as a row-major framework uses lazy evaluation for transpose.
        //      Which means if we try to copy a transposed tensor. We should not transpose it. Else the other
        //      backend would transpose it again.
        ggml_tensor* src = tensor->src[0];
        bool do_transpose = false;
        while(src->op == GGML_OP_TRANSPOSE) {
            do_transpose = !do_transpose;
            src = src->src[0];
            GGML_ASSERT(src != NULL);
        }
        GGML_ASSERT(src != NULL);
        t = realize_ggml_view(src);
        if(do_transpose) {
            *t = ttnn::transpose(*t, -2, -1);
        }
    }
    else if (tensor->op == GGML_OP_PERMUTE) {
        // DITTO above.
        // XXX: This only handles the case where the permute is the only view class operation
        // May broke if there are multiple permutes
        ggml_tensor* src = tensor->src[0];
        t = realize_ggml_view(src);
    }
    else if (tensor->op == GGML_OP_RESHAPE) {
        // No reason to do actual reshaping as it doesn't make a difference in row-major layout
        ggml_tensor* src = tensor->src[0];
        while(src->op == GGML_OP_RESHAPE) {
            src = src->src[0];
            GGML_ASSERT(src != NULL);
        }
        GGML_ASSERT(src != NULL);
        t = realize_ggml_view(src);
    }
    else {
        t = realize_ggml_view(tensor);
        GGML_ASSERT(ggml_tt_tensors_shape_equal(tensor, *t));
    }
    GGML_ASSERT(t->layout() == tt::tt_metal::Layout::TILE);
    if(t->dtype() != tt::tt_metal::DataType::BFLOAT16 || t->dtype() != tt::tt_metal::DataType::FLOAT32) {
        *t = ttnn::experimental::typecast(*t, tt::tt_metal::DataType::BFLOAT16);
    }

    // TODO: Proper handling of data types
    GGML_ASSERT(dst_ggtype != GGML_TYPE_F64 && dst_ggtype != GGML_TYPE_I16 && dst_ggtype != GGML_TYPE_I8 && dst_ggtype != GGML_TYPE_I32);
    switch(t->dtype()) {
        case tt::tt_metal::DataType::BFLOAT16:
            tensor2ggml<bfloat16>(*t, (float*)data, dst_ggtype);
            break;
        case tt::tt_metal::DataType::FLOAT32:
            tensor2ggml<float>(*t, (float*)data, dst_ggtype);
            break;
        default:
            GGML_ASSERT(false && "Unsupported data type in TT tensor when converting to GGML tensor");
            break;
    }
}

static void * ggml_backend_metalium_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_metalium_buffer_context * ctx = (ggml_backend_metalium_buffer_context *)buffer->context;
    return (uint8_t*)0xdeadbeef + ctx->base_offset;
}

static void
ggml_backend_metalium_buffer_init_tensor(ggml_backend_buffer_t buffer,
                                     ggml_tensor *tensor)
{
    ggml_backend_metalium_buffer_context * bufctx = (ggml_backend_metalium_buffer_context *)buffer->context;
    bufctx->metadata_to_free.push_back(std::make_unique<TensorWithMetadata>(TensorWithMetadata{
        .tensor = nullptr,
        .ggtype = GGML_TYPE_COUNT,
        .bufctx = bufctx
    }));
    tensor->extra = bufctx->metadata_to_free.back().get();
    // HACK: Make KV cache work
    std::string name(tensor->name);
    if(name.find("cache") != std::string::npos && tensor->op == GGML_OP_NONE) {
        TensorWithMetadata* meta = (TensorWithMetadata*)tensor->extra;
        std::vector<uint32_t> shape(tensor->ne, tensor->ne + GGML_MAX_DIMS);
        std::reverse(shape.begin(), shape.end());
        // TODO: Check if we can make TILE tensors and not pad on CPU
        auto t = ttnn::zeros(ttnn::Shape(shape), ggml2tt_type(tensor->type, bufctx->device->arch()), tt::tt_metal::Layout::ROW_MAJOR);
        t = ttnn::tilize_with_zero_padding(t.to(bufctx->device));
        meta->tensor = std::make_shared<tt::tt_metal::Tensor>(std::move(t));
    }
    // std::cout << "Creating tensor with address: " << tensor->data << ", shape = " << tensor->ne[0] << " " << tensor->ne[1] << " " << tensor->ne[2] << " " << tensor->ne[3] << ", name " << tensor->name << std::endl;
    GGML_UNUSED(buffer);
}

static void ggml_backend_metalium_buffer_clear(ggml_backend_buffer_t buffer,
                                                        uint8_t value)
{
    // Not using this. Metalium's allication model is not compatible with GGML's allocator
    GGML_UNUSED(buffer);
    GGML_UNUSED(value);
}

static bool
ggml_backend_metalium_buffer_cpy_tensor(ggml_backend_buffer_t buffer,
                                    const ggml_tensor *src,
                                    ggml_tensor *dst)
{
    GGML_UNUSED(buffer);

    GGML_ASSERT(src->extra != NULL);
    GGML_ASSERT(dst->extra != NULL);

    TensorWithMetadata * src_meta = (TensorWithMetadata *)src->extra;
    TensorWithMetadata * dst_meta = (TensorWithMetadata *)dst->extra;

    tt::tt_metal::Tensor& src_tensor = *src_meta->tensor;

    tt::tt_metal::Tensor ret = ttnn::zeros_like(src_tensor);
    ret.deepcopy(src_tensor);
    GGML_ASSERT(ret.storage_type() == tt::tt_metal::StorageType::DEVICE || ret.storage_type() == tt::tt_metal::StorageType::MULTI_DEVICE);
    dst_meta->tensor = std::make_shared<tt::tt_metal::Tensor>(std::move(ret));
    dst_meta->ggtype = dst->type;
    return true;
}

static struct ggml_backend_buffer_i ggml_backend_metalium_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_metalium_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_metalium_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_metalium_buffer_init_tensor,
    /* .memset_tensor   = */ nullptr,
    /* .set_tensor      = */ ggml_backend_metalium_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_metalium_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_metalium_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_metalium_buffer_clear,
    /* .reset           = */ nullptr,
};


static ggml_backend_buffer_t
ggml_backend_metalium_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                           size_t size) {
    ggml_backend_metalium_buffer_type_context * buft_ctx = (ggml_backend_metalium_buffer_type_context *)buft->context;

    // FIXME: GGML unit tests fails if I don't add some additional memory to the buffer beyond the requested size
    size_t alloc_size = size + 4096 * 1024;
    // real allocation is deferred until the first tensor is set because we don't know the underlying tensor type yet
    ggml_backend_metalium_buffer_context* ctx = new ggml_backend_metalium_buffer_context {
        .ggml_buffer_size_bytes = size,
        .name = buft_ctx->name,
        .device = buft_ctx->device,
        .base_offset = g_metalium_base_offset,

        .metadata_to_free = {}
    };
    g_metalium_base_offset += alloc_size;
    // std::cout << "Allocating buffer of size " << size << " bytes\n";
    return ggml_backend_buffer_init(buft, ggml_backend_metalium_buffer_interface, ctx, alloc_size);
}

static bool ggml_backend_metalium_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return false;
}

static ggml_backend_buffer_type_i ggml_backend_metalium_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_metalium_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_metalium_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_metalium_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_metalium_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_metalium_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_metalium_buffer_type_is_host,
};

static ggml_backend_buffer_type_t ggml_backend_metalium_buffer_type(ggml_backend_dev_t dev, ggml_backend_metalium_device_context* dev_ctx) {
    auto device_id = dev_ctx->device_id;
    ggml_backend_metalium_reg_context* regctx = (ggml_backend_metalium_reg_context*)(dev->reg->context);

    GGML_ASSERT((size_t)device_id < tt::tt_metal::GetNumAvailableDevices());
    GGML_ASSERT((size_t)device_id < regctx->devices.size());

    static std::map<int, ggml_backend_buffer_type> buffer_type_map;
    static std::set<std::unique_ptr<ggml_backend_metalium_buffer_type_context>> buffer_type_context_deleter;
    auto it = buffer_type_map.find(device_id);
    if(it != buffer_type_map.end()) {
        return &it->second;
    }

    auto bufctx = std::make_unique<ggml_backend_metalium_buffer_type_context>(
        ggml_backend_metalium_buffer_type_context{
            .device = dev_ctx->device.get(),
            .name = "Metalium " + std::to_string(device_id),
        });
    auto* bufctx_ptr = bufctx.get();
    buffer_type_context_deleter.insert(std::move(bufctx));

    buffer_type_map[device_id] = {
        /* .iface    = */ ggml_backend_metalium_buffer_type_interface,
        /* .device   = */ regctx->devices[device_id],
        /* .context  = */ bufctx_ptr,
    };
    return &buffer_type_map[device_id];
}

static enum ggml_status ggml_backend_metalium_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    ggml_backend_metalium_context * ctx = (ggml_backend_metalium_context *)backend->context;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        // std::cout << "Graph compute " << ggml_op_desc(node) << "\n"
        //     << "  dst addr: " << node->data << "\n"
        //     << "  src0 addr: " << (void*)(node->src[0] ? node->src[0]->data : 0) << "\n"
        //     << "  src1 addr: " << (void*)(node->src[1] ? node->src[1]->data : 0) << "\n";

        // Bypass post conition checks for these ops because they are evaluated lazily
        if(node->op == GGML_OP_VIEW || node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_PERMUTE) {
            continue;
        }

        switch (node->op) {
            case GGML_OP_UNARY: {
                ggml_unary_op unary_op = ggml_get_unary_op(node);
                bool ok = false;
                switch (unary_op) {
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_EXP:
                    ok = ggml_backend_metalium_activations(ctx, node, unary_op);
                    break;
                default:
                    fprintf(stderr, "%s: unsupported unary op %s\n", __func__, ggml_unary_op_name(unary_op));
                }
                GGML_ASSERT(ok && "Failed to execute unary op");
                break;
            }
            case GGML_OP_LEAKY_RELU:
                ggml_backend_metalium_leaky_relu(ctx, node);
                break;
            case GGML_OP_ADD:
            case GGML_OP_SUB:
            case GGML_OP_DIV:
            case GGML_OP_MUL:
                ggml_backend_metalium_bin_op(ctx, node, node->op);
                break;
            case GGML_OP_MUL_MAT:
                ggml_backend_metalium_mul_mat(ctx, node);
                break;
            case GGML_OP_OUT_PROD:
                ggml_backend_metalium_outer_product(ctx, node);
                break;

            case GGML_OP_CONT:
            case GGML_OP_CPY:
            case GGML_OP_DUP:
                ggml_backend_metalium_cpy(ctx, node);
                break;
            case GGML_OP_SET:
                ggml_backend_metalium_set(ctx, node);
                break;

            case GGML_OP_CLAMP:
                ggml_backend_metalium_clamp(ctx, node);
                break;

            case GGML_OP_SCALE:
                ggml_backend_metalium_scale(ctx, node);
                break;

            case GGML_OP_GET_ROWS:
                ggml_backend_metalium_get_row(ctx, node);
                break;

            case GGML_OP_NORM:
                ggml_backend_metalium_norm(ctx, node, false);
                break;

            case GGML_OP_RMS_NORM:
                ggml_backend_metalium_norm(ctx, node, true);
                break;

            case GGML_OP_ADD1:
                ggml_backend_metalium_add1(ctx, node);
                break;

            case GGML_OP_SQRT:
                ggml_backend_metalium_sqrt(ctx, node);
                break;

            case GGML_OP_SQR:
                ggml_backend_metalium_sqr(ctx, node);
                break;

            case GGML_OP_CONCAT:
                ggml_backend_metalium_concat(ctx, node);
                break;

            case GGML_OP_SOFT_MAX:
                ggml_backend_metalium_softmax(ctx, node);
                break;

            case GGML_OP_COS:
                ggml_backend_metalium_cos(ctx, node);
                break;

            case GGML_OP_SIN:
                ggml_backend_metalium_sin(ctx, node);
                break;

            case GGML_OP_LOG:
                ggml_backend_metalium_log(ctx, node);
                break;

            case GGML_OP_ARANGE:
                ggml_backend_metalium_arange(ctx, node);
                break;
            
            case GGML_OP_GROUP_NORM:
                ggml_backend_metalium_group_norm(ctx, node);
                break;
            
            case GGML_OP_REPEAT:
                ggml_backend_metalium_repeat(ctx, node);
                break;

            case GGML_OP_NONE:
                break;

            default:
                fprintf(stderr, "%s: unsupported op %s\n", __func__, ggml_op_desc(node));
                GGML_ASSERT(false);
        }
        TensorWithMetadata* meta = (TensorWithMetadata*)node->extra;
        // std::cout << "Executed " << ggml_op_desc(node) << " with address " << node->data << " and shape " << meta->tensor->shape() << ", GGML wants " << node->ne[0] << " " << node->ne[1] << " " << node->ne[2] << " " << node->ne[3] << std::endl;
        GGML_ASSERT(meta != NULL);
        GGML_ASSERT(meta->tensor != NULL);
        GGML_ASSERT(meta->tensor->storage_type() == tt::tt_metal::StorageType::DEVICE || meta->tensor->storage_type() == tt::tt_metal::StorageType::MULTI_DEVICE);
        if(!ggml_tt_tensors_shape_equal(node, *meta->tensor)) {
            tt::log_fatal(tt::LogType::LogAlways, "Mismatched tensor shapes for node '{}' ({}): GGML wants [{}, {}, {}, {}], TTNN generates {}\n"
                , node->name, ggml_op_name(node->op), node->ne[0], node->ne[1], node->ne[2], node->ne[3], meta->tensor->shape());
            abort();
        }
    }

    return GGML_STATUS_SUCCESS;

    GGML_UNUSED(backend);
}

static bool ggml_backend_metalium_device_supports_op_internal(ggml_backend_dev_t device, const struct ggml_tensor * op);

static bool ggml_backend_metalium_device_supports_op(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    bool ok = ggml_backend_metalium_device_supports_op_internal(device, op);
    // debug print to log rejected ops
    if(!ok && g_debug_flags.print_rejected_ops) {
        fprintf(stderr, "REJECT op %s (%s)\n", ggml_op_name(op->op), op->name);
        if(op->src[0]) {
            fprintf(stderr, "  src0 shape [%ld %ld %ld %ld], dtype = %s\n", op->src[0]->ne[0], op->src[0]->ne[1], op->src[0]->ne[2], op->src[0]->ne[3], ggml_type_name(op->src[0]->type));
        }
        if(op->src[1]) {
            fprintf(stderr, "  src1 shape [%ld %ld %ld %ld], dtype = %s\n", op->src[1]->ne[0], op->src[1]->ne[1], op->src[1]->ne[2], op->src[1]->ne[3], ggml_type_name(op->src[1]->type));
        }
    }
    return ok;
}


static bool ggml_backend_metalium_device_supports_op_internal(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    GGML_ASSERT(op != NULL);
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    ggml_backend_metalium_device_context * ctx = (ggml_backend_metalium_device_context *)device->context;

    // The metalium backend has seperated internal data types from the GGML data types. We really only care about
    // what we can convert to and from.
    auto tensor_supported = [&](const struct ggml_tensor * tensor) {
        if(tensor == NULL || !is_ggml_type_supported_by_metalium(tensor->type, ctx->device->arch())) {
            return false;
        }
        // TTNN requires the tensor to be 4-byte aligned and all quantized tensors must be a multiple of 32

        tt::tt_metal::DataType tt_type = ggml2tt_type(tensor->type, ctx->device->arch());
        switch(tt_type) {
            case tt::tt_metal::DataType::BFLOAT16:
            case tt::tt_metal::DataType::UINT16:
                return tensor->ne[0] % 2 == 0 && tensor->ne[0] != 0;
            case tt::tt_metal::DataType::FLOAT32:
            case tt::tt_metal::DataType::UINT32:
                return true;
            case tt::tt_metal::DataType::UINT8:
                return tensor->ne[0] % 4 == 0 && tensor->ne[0] != 0;
            case tt::tt_metal::DataType::INVALID:
                GGML_ASSERT(false && "Unsupported data type");
                break;
            default:
                return tensor->ne[0] % 32 == 0 && tensor->ne[0] != 0;
        }
        GGML_UNREACHABLE();
    };

    // std::cout << "Checking if op is supported: " << ggml_op_desc(op) << std::endl;
    // std::cout << "Output tensor details:\n"
    //     << "  data: " << op->data << "\n"
    //     << "  ne: " << op->ne[0] << " " << op->ne[1] << " " << op->ne[2] << " " << op->ne[3] << "\n"
    //     << "  nb: " << op->nb[0] << " " << op->nb[1] << " " << op->nb[2] << " " << op->nb[3] << "\n"
    //     << "  type: " << ggml_type_name(op->type) << "\n"
    //     << "  view_src: " << op->view_src << "\n"
    //     << "\n";
    if(!tensor_supported(op)) {
        return false;
    }
    // ARANGE and NONE are special case where src0 is not required
    if(op->op == GGML_OP_NONE || op->op == GGML_OP_ARANGE) {
        return true;
    }
    if(!tensor_supported(src0)) {
        return false;
    }

    switch (op->op) {
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_EXP:
                    return true;
                default:
                    return false;
            }
        case GGML_OP_LEAKY_RELU:
        case GGML_OP_NONE:
        case GGML_OP_CONT:
        case GGML_OP_CPY:
        case GGML_OP_DUP:
        case GGML_OP_RESHAPE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_CLAMP:
        case GGML_OP_SCALE:
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
        case GGML_OP_ADD1:
        case GGML_OP_SQRT:
        case GGML_OP_SQR:
        case GGML_OP_PERMUTE:
        case GGML_OP_LOG:
        case GGML_OP_GROUP_NORM:
        // TTNN can really only do unpad() so the source rank must be greater than or equal to the destination rank
        // and must not be permuted as that's a sign of it being reshaped from another tensor. Which is costly due to
        // TTNN not using row-major layout.
        case GGML_OP_VIEW:
            return true;
        
        case GGML_OP_SIN:     // Sin and Cos disabled on GS due to bug in TTNN until fixed
        case GGML_OP_COS:     // ref: https://github.com/tenstorrent/tt-metal/issues/12753
            return ctx->device->arch() != tt::ARCH::GRAYSKULL;

        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
            return tensor_supported(src1) && numpy_broadcast_rule(src0, src1);
        // DIV does not support broadcasting on TTNN
        case GGML_OP_DIV:
            return tensor_supported(src1) && memcmp(src0->ne, src1->ne, sizeof(src0->ne)) == 0;

        case GGML_OP_MUL_MAT:
            return tensor_supported(src1) && ggml_backend_metalium_can_mul_mat(op);
        case GGML_OP_SET:
            return tensor_supported(src1) && ggml_backend_metalium_can_set(op);
        case GGML_OP_SOFT_MAX:
            return ggml_backend_metalium_can_softmax(op);
        case GGML_OP_GET_ROWS:
            return tensor_supported(src1) && ggml_backend_metalium_can_get_row(op);
        case GGML_OP_CONCAT:
            return tensor_supported(src1) && ggml_backend_metalium_can_concat(op);
        case GGML_OP_REPEAT:
            return ggml_backend_metalium_can_repeat(op);
        case GGML_OP_OUT_PROD:
            return tensor_supported(src1) && ggml_backend_metalium_can_outer_product(op);
        default:
            return false;
    }
}

static bool ggml_backend_metalium_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (buft->iface.get_name != ggml_backend_metalium_buffer_type_name) {
        return false;
    }
    ggml_backend_metalium_buffer_type_context * buft_ctx = (ggml_backend_metalium_buffer_type_context *)buft->context;
    ggml_backend_metalium_device_context * ctx = (ggml_backend_metalium_device_context *)dev->context;
    return buft_ctx->device == ctx->device.get();
}

static void ggml_backend_metalium_synchronize(ggml_backend_t backend)
{
    ggml_backend_metalium_context * ctx = (ggml_backend_metalium_context *)backend->context;
    // tt::tt_metal::Finish(ctx->device->command_queue());
    for(size_t i = 0; i < ctx->device->num_devices(); i++) {
        tt::tt_metal::Finish(ctx->device->get_device_index(i)->command_queue());
    }
}

static struct ggml_backend_i metalium_backend_i = {
    /* .get_name                = */ ggml_backend_metalium_name,
    /* .free                    = */ ggml_backend_metalium_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ ggml_backend_metalium_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_metalium_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL
};

static ggml_guid_t ggml_backend_metalium_guid(void) {
    static ggml_guid guid = { 0x91, 0x69, 0xd5, 0x5f, 0x24, 0xe7, 0x44, 0x00, 0xb4, 0x2a, 0x73, 0x23, 0x48, 0xb0, 0x4e, 0xe7 };
    return &guid;
}

static ggml_backend_t ggml_backend_metalium_init(ggml_backend_metalium_device_context* dev_ctx) {
    int device_id = dev_ctx->device_id;
    auto device = dev_ctx->device;
    GGML_ASSERT(device_id >= 0 && (size_t)device_id < tt::tt_metal::GetNumAvailableDevices());
    GGML_ASSERT(device != nullptr);

    ggml_backend_metalium_context * ctx = new ggml_backend_metalium_context {
        /* device            = */ device,
        /* device_id         = */ device_id,
        /* name              = */ dev_ctx->name,
    };

    ggml_backend_t backend = new ggml_backend {
        /* .guid      = */ ggml_backend_metalium_guid(),
        /* .interface = */ metalium_backend_i,
        /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_metalium_reg(), device_id),
        /* .context   = */ ctx
    };
    return backend;
}

bool ggml_backend_is_metalium(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_metalium_guid());
}

static const char * ggml_backend_metaliium_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return "Metalium";
}

static size_t ggml_backend_metalium_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_metalium_reg_context * ctx = (ggml_backend_metalium_reg_context *)reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_metalium_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_metalium_reg_context * ctx = (ggml_backend_metalium_reg_context *)reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

static const ggml_backend_reg_i ggml_backend_metalium_reg_interface = {
    /* .get_name          = */ ggml_backend_metaliium_reg_get_name,
    /* .get_device_count  = */ ggml_backend_metalium_reg_get_device_count,
    /* .get_device        = */ ggml_backend_metalium_reg_get_device,
    /* .get_proc_address  = */ NULL,
};

static const char* ggml_backend_metalium_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_metalium_device_context * ctx = (ggml_backend_metalium_device_context *)dev->context;
    return ctx->name.c_str();
}

static const char * ggml_backend_metalium_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_metalium_device_context * ctx = (ggml_backend_metalium_device_context *)dev->context;
    return ctx->description.c_str();
}

static void ggml_backend_metalium_get_memory(ggml_backend_dev_t dev, size_t * total, size_t * free) {
    ggml_backend_metalium_device_context * ctx = (ggml_backend_metalium_device_context *)dev->context;

    *total = 0;
    *free = 0;
    for(size_t i = 0; i < ctx->device->num_devices(); i++) {
        auto* dev = ctx->device->get_device_index(i);
        auto stats = dev->get_memory_allocation_statistics(tt::tt_metal::BufferType::DRAM);
        size_t num_dram_channels = dev->num_dram_channels();

        *total += stats.total_allocatable_size_bytes * num_dram_channels;
        *free += stats.total_free_bytes * num_dram_channels;
    }
}

static enum ggml_backend_dev_type ggml_backend_metalium_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static ggml_backend_t ggml_backend_metalium_device_init(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    ggml_backend_metalium_device_context * ctx = (ggml_backend_metalium_device_context *)dev->context;
    ggml_backend_t backend = ggml_backend_metalium_init(ctx);
    GGML_ASSERT(backend != NULL);
    return backend;
}

static void ggml_backend_metalium_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    ggml_backend_metalium_device_context * ctx = (ggml_backend_metalium_device_context *)dev->context;
    size_t free = 0;
    size_t total = 0;
    ggml_backend_metalium_get_memory(dev, &total, &free);
    *props = ggml_backend_dev_props {
        .name = ctx->name.c_str(),
        .description = ctx->description.c_str(),
        .memory_free = free,
        .memory_total = total,
        .type = ggml_backend_metalium_get_type(dev),
        .caps = ggml_backend_dev_caps {
            .async = true,
            .host_buffer = false,
            .buffer_from_host_ptr = false,
            .events = false,
        }
    };
}

static ggml_backend_buffer_type_t ggml_backend_metalium_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_metalium_device_context * ctx = (ggml_backend_metalium_device_context *)dev->context;
    return ggml_backend_metalium_buffer_type(dev, ctx);
}

static const ggml_backend_device_i ggml_backend_metalium_device_interface = {
    /* .get_name                = */ ggml_backend_metalium_device_get_name,
    /* .get_description         = */ ggml_backend_metalium_device_get_description,
    /* .get_memory              = */ ggml_backend_metalium_get_memory,
    /* .get_type                = */ ggml_backend_metalium_get_type,
    /* .get_props               = */ ggml_backend_metalium_device_get_props,
    /* .init_backend            = */ ggml_backend_metalium_device_init,
    /* .get_buffer_type         = */ ggml_backend_metalium_get_buffer_type,
    /* .get_host_buffer_type    = */ NULL,
    /* .buffer_from_host_ptr    = */ NULL,
    /* .supports_op             = */ ggml_backend_metalium_device_supports_op,
    /* .supports_buft           = */ ggml_backend_metalium_device_supports_buft,
    /* .offload_op              = */ NULL,
    /* .event_new               = */ NULL,
    /* .event_free              = */ NULL,
    /* .event_synchronize       = */ NULL,
};

static std::string identify_tensotrrent_device(const ttnn::Device* device)
{
    auto grid_size = device->compute_with_storage_grid_size();
    // TODO: Support mesh configurations
    if(device->arch() == tt::ARCH::GRAYSKULL) {
        if(grid_size.x == 11 && grid_size.y == 8) {
            return "Tenstorrent Grayskull e75";
        }
        return "Tenstorrent Grayskull e150";
    }
    if(device->arch() == tt::ARCH::WORMHOLE_B0) {
        if(grid_size.x == 8 && grid_size.y == 7) {
            return "Tenstorrent Wormhole n300";
        }
        return "Tenstorrent Wormhole n150";
    }

    return "Unknown Tenstorrent device";
}

static std::vector<std::unique_ptr<ggml_backend_device>> g_backend_device_holder;
static std::vector<std::unique_ptr<ggml_backend_metalium_device_context>> g_backend_device_context_holder;
GGML_API ggml_backend_reg_t ggml_backend_metalium_reg()
{
    static ggml_backend_reg reg;
    static std::once_flag once;
    std::call_once(once, [&]() {
        if(getenv("TT_METAL_HOME") == NULL || getenv("ARCH_NAME") == NULL) {
            tt::log_fatal(tt::LogType::LogAlways, "TT_METAL_HOME and ARCH_NAME environment variables must be set to use the Metalium backend");
            abort();
        }

        const char* c_device = getenv("GGML_METALIUM_DEVICE");
        std::vector<int> device_ids = {0};
        bool use_cluster = false;
        std::pair<int, int> offset = {0, 0};
        ttnn::distributed::MeshShape mesh_shape = {1, 1};
        ttnn::distributed::MeshType mesh_type = ttnn::distributed::MeshType::RowMajor;
        if(c_device != NULL && trim_sv(c_device).size() > 0) {
            auto& sys_mesh = tt::tt_metal::distributed::SystemMesh::instance();
            // GGML_METALIUM_DEVICE supports a few configuration syntaxes
            // 1. "0" - Use the device at index 0
            // 2. "0-2,5" - Use the device at index 0, 1, 2 and 5 (without clustering. doesn't work as of now as TTNN needs explicit data transfer between deices. but useful for testing backends)
            // 3. "cluster(x,y, [topology, offset_x, offset_y])" - Use a cluster of devices with x by y configuration and optional topology and offset (default is row-major)

            const std::string_view device_str = c_device;
            // trim leading and trailing whitespaces
            std::string_view remaining = trim_sv(device_str);

            // case 1 is a special case of case 2. So we simply implement case 2
            if(!remaining.starts_with("cluster")) {
                device_ids.clear();
                std::set<int> device_id_set;
                while(remaining.size() > 0) {
                    size_t pos = remaining.find(',');
                    const std::string_view devid_range = trim_sv(remaining.substr(0, pos));
                    remaining = pos == std::string::npos ? std::string_view() : remaining.substr(pos + 1);

                    size_t dash_pos = devid_range.find('-');
                    if(dash_pos == std::string::npos) {
                        auto id = try_stoi(devid_range);
                        if(!id.has_value()) {
                            tt::log_fatal(tt::LogType::LogAlways, "Invalid device id '{}'", devid_range);
                            abort();
                        }
                        if(id < 0 || id >= (int)tt::tt_metal::GetNumAvailableDevices()) {
                            tt::log_fatal(tt::LogType::LogAlways, "Invalid device id '{}'. Device id must be between 0 and {}", devid_range, tt::tt_metal::GetNumAvailableDevices() - 1);
                            abort();
                        }
                        device_id_set.insert(*id);
                    } else {
                        auto start = try_stoi(devid_range.substr(0, dash_pos));
                        if(!start.has_value()) {
                            tt::log_fatal(tt::LogType::LogAlways, "Invalid device id '{}'", devid_range.substr(0, dash_pos));
                            abort();
                        }
                        auto end = try_stoi(devid_range.substr(dash_pos + 1));
                        if(!end.has_value()) {
                            tt::log_fatal(tt::LogType::LogAlways, "Invalid device id '{}'", devid_range.substr(dash_pos + 1));
                            abort();
                        }
                        const std::string range_str = std::to_string(*start) + "-" + std::to_string(*end);
                        if(*start < 0 || *start >= (int)tt::tt_metal::GetNumAvailableDevices() || *end < 0 || *end >= (int)tt::tt_metal::GetNumAvailableDevices()) {
                            tt::log_fatal(tt::LogType::LogAlways, "Invalid device id range '{}'. Device id must be between 0 and {}", range_str, tt::tt_metal::GetNumAvailableDevices() - 1);
                            abort();
                        }
                        if(*start > *end) {
                            tt::log_fatal(tt::LogType::LogAlways, "Invalid device id range '{}'. Start id must be less than or equal to end id", range_str);
                            abort();
                        }
                        for(int i = *start; i <= *end; i++) {
                            device_id_set.insert(i);
                        }
                    }
                    for(auto id : device_id_set) {
                        device_ids.push_back(id);
                    }
                }

                mesh_shape = {1, (int)device_ids.size()};
            } else {
                use_cluster = true;
                remaining = trim_sv(remaining.substr(7));
                if(remaining.size() == 0 || remaining[0] != '(') {
                    tt::log_fatal(tt::LogType::LogAlways, "Syntax errpr: Invalid cluster configuration '{}'. Expecting '(' after 'cluster'", c_device);
                    abort();
                }
                if(remaining[remaining.size() - 1] != ')') {
                    tt::log_fatal(tt::LogType::LogAlways, "Syntax errpr: Invalid cluster configuration '{}'. Expecting ')' at the end", c_device);
                    abort();
                }

                remaining = trim_sv(remaining.substr(1, remaining.size() - 2));
                std::vector<std::string_view> args;
                while(remaining.size() > 0) {
                    size_t pos = remaining.find(',');
                    const std::string_view part = trim_sv(remaining.substr(0, pos));
                    remaining = pos == std::string::npos ? std::string_view() : remaining.substr(pos + 1);
                    args.push_back(part);
                }

                if(args.size() < 2) {
                    tt::log_fatal(tt::LogType::LogAlways, "Syntax error: Invalid cluster configuration '{}'. Expecting at least 2 arguments", c_device);
                    abort();
                }
                
                auto x = try_stoi(args[0]);
                if(!x.has_value()) {
                    tt::log_fatal(tt::LogType::LogAlways, "Invalid cluser x dimension '{}'. Expecting an integer as the first argument", args[0]);
                    abort();
                }
                auto y = try_stoi(args[1]);
                if(!y.has_value()) {
                    tt::log_fatal(tt::LogType::LogAlways, "Invalid cluser y dimension '{}'. Expecting an integer as the second argument", args[1]);
                    abort();
                }
                mesh_shape = {*x, *y};
                if(*x > (int)sys_mesh.get_shape().first || *y > (int)sys_mesh.get_shape().second) {
                    tt::log_fatal(tt::LogType::LogAlways, "Invalid cluster shape of {}x{}. Maximum supported shape is {}x{}", *x, *y, sys_mesh.get_shape().first, sys_mesh.get_shape().second);
                    abort();
                }
                if(*x < 1 || *y < 1) {
                    tt::log_fatal(tt::LogType::LogAlways, "Invalid cluster shape of {}x{}. Minimum supported shape is 1x1", *x, *y);
                    abort();
                }

                for(size_t i = 2; i < args.size(); i++) {
                    const std::string_view arg = args[i];
                    if(i == 2) {
                        // Yeah this is stupid. tt-topology uses different naming compared to what we use in tt-metal
                        if(arg == "ROW_MAJOR" || arg == "MESH") {
                            mesh_type = ttnn::distributed::MeshType::RowMajor;
                        }
                        else if(arg == "LINE" || arg == "LINEAR") {
                            mesh_type = ttnn::distributed::MeshType::Line;
                        }
                        else if(arg == "RING" || arg == "TORUS") {
                            mesh_type = ttnn::distributed::MeshType::Ring;
                        }
                        else {
                            tt::log_fatal(tt::LogType::LogAlways, "Invalid topology '{}'. Supported topologies are ROW_MAJOR (MESH), LINE (LINEAR), RING (TORUS)", arg);
                            abort();
                        }
                    }
                    else if(i == 3) {
                        auto offset_x = try_stoi(arg);
                        if(!offset_x.has_value()) {
                            tt::log_fatal(tt::LogType::LogAlways, "Invalid x offset '{}'. Expecting an integer", arg);
                            abort();
                        }
                        if(*offset_x < 0 || *offset_x >= (int)mesh_shape.first) {
                            tt::log_fatal(tt::LogType::LogAlways, "Invalid x offset '{}'. Offset must be between 0 and {}", arg, mesh_shape.first - 1);
                            abort();
                        }
                        if(*offset_x + offset.first >= (int)mesh_shape.first) {
                            tt::log_fatal(tt::LogType::LogAlways, "Invalid x offset '{}'. Offset must be less than {}", arg, mesh_shape.first - offset.first);
                            abort();
                        }
                        offset.first = *offset_x;
                    }
                    else if(i == 4) {
                        auto offset_y = try_stoi(arg);
                        if(!offset_y.has_value()) {
                            tt::log_fatal(tt::LogType::LogAlways, "Invalid argument '{}' for y offset. Expecting an integer", arg);
                            abort();
                        }
                        if(*offset_y < 0 || *offset_y >= (int)mesh_shape.second) {
                            tt::log_fatal(tt::LogType::LogAlways, "Invalid y offset '{}'. Offset must be between 0 and {}", arg, mesh_shape.second - 1);
                            abort();
                        }
                        if(*offset_y + offset.second >= (int)mesh_shape.second) {
                            tt::log_fatal(tt::LogType::LogAlways, "Invalid y offset '{}'. Offset must be less than {}", arg, mesh_shape.second - offset.second);
                            abort();
                        }
                        offset.second = *offset_y;
                    }
                    else {
                        tt::log_fatal(tt::LogType::LogAlways, "Invalid argument '{}'. More arguments then expected", arg);
                        abort();
                    }
                }
                device_ids.clear(); // Empty. Let TTNN decide the device ids
            }
        }

        tt::tt_metal::detail::EnablePersistentKernelCache();
        static std::unique_ptr<ggml_backend_metalium_reg_context> ctx = std::make_unique<ggml_backend_metalium_reg_context>();
        auto mesh = ttnn::distributed::open_mesh_device(mesh_shape, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, tt::tt_metal::DispatchCoreType::WORKER, mesh_type, offset, device_ids);
        std::vector<std::shared_ptr<ttnn::distributed::MeshDevice>> devices;
        if(use_cluster || (mesh->num_cols() == 1 && mesh->num_rows() == 1)) {
            devices = {mesh};
        }
        else {
            for(size_t x = 0; x < mesh_shape.first; x++) {
                for(size_t y = 0; y < mesh_shape.second; y++) {
                    devices.push_back(mesh->create_submesh({1,1}, {x, y}));
                }
            }
        }
        for(size_t device_id = 0; device_id < devices.size(); device_id++) {
            auto device = devices[device_id];
            ggml_backend_metalium_device_context * dev_ctx = new ggml_backend_metalium_device_context;
            for(size_t i = 0; i < device->num_devices(); i++) {
                auto* dev = device->get_device_index(i);
                ttnn::enable_program_cache(*dev);
            }
            device->enable_program_cache();
            // Limit device support to the ones I own
            GGML_ASSERT(device->arch() == tt::ARCH::GRAYSKULL || device->arch() == tt::ARCH::WORMHOLE_B0);

            dev_ctx->device = device;
            dev_ctx->device_id = device_id;
            dev_ctx->name = "METALIUM" + std::to_string(device_id);
            if(device->num_cols() == 1 && device->num_rows() == 1) {
                auto* dev = device->get_device_index(0);
                dev_ctx->description = identify_tensotrrent_device(dev) + (dev->is_mmio_capable() ? " [Local]" : " [Remote]");
            }
            else {
                std::string_view arch_name;
                switch(device->arch()) {
                    case tt::ARCH::GRAYSKULL:
                        arch_name = "Grayskull";
                        break;
                    case tt::ARCH::WORMHOLE_B0:
                        arch_name = "Wormhole";
                        break;
                    default:
                        arch_name = "Unknown";
                        break;
                }
                dev_ctx->description = fmt::format("Tenstorrent {} cluster of {}x{} devices", arch_name, device->num_rows(), device->num_cols());
            }

            // FIXME: Release the device context when appropriate
            ggml_backend_dev_t dev = new ggml_backend_device {
                .iface = ggml_backend_metalium_device_interface,
                .reg = &reg,
                .context = dev_ctx
            };
            ctx->devices.push_back(dev);
            g_backend_device_context_holder.push_back(std::unique_ptr<ggml_backend_metalium_device_context>(dev_ctx));
            g_backend_device_holder.push_back(std::unique_ptr<ggml_backend_device>(dev));
        }
        
        reg = ggml_backend_reg {
            /* .interface = */ ggml_backend_metalium_reg_interface,
            /* .context   = */ ctx.get()
        };
    });
    return &reg;
}