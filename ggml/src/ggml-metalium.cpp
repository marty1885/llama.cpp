#include "common/bfloat16.hpp"
#include "common/constants.hpp"
#include "device/tt_arch_types.h"
#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-metalium.h"

#include "host_api.hpp"
#include "impl/dispatch/command_queue.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/normalization/softmax/device/softmax_op.hpp"
#include "ttnn/tensor/types.hpp"
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
#include <ttnn/operations/moreh/moreh_matmul/moreh_matmul.hpp>
#include <ttnn/operations/kv_cache/kv_cache.hpp>
#include <ttnn/operations/data_movement/slice/slice.hpp>
#include <ttnn/operations/normalization/layernorm/layernorm.hpp>
#include <ttnn/operations/normalization/rmsnorm/rmsnorm.hpp>
#include <ttnn/operations/data_movement/untilize/untilize.hpp>
#include <ttnn/operations/experimental/transformer/nlp_kv_cache_load_slice/nlp_kv_cache_load_slice.hpp>
#include <ttnn/deprecated/tt_numpy/functions.hpp>
#include <ttnn/operations/eltwise/unary/unary_composite.hpp>
#include <ttnn/operations/data_movement/transpose/transpose.hpp>
#include <ttnn/operations/data_movement/permute/permute.hpp>
#include <ttnn/operations/data_movement/concat/concat.hpp>
#include <ttnn/operations/eltwise/unary/unary.hpp>
#include <ttnn/operations/eltwise/unary/unary_composite.hpp>
#include <ttnn/operations/experimental/copy/typecast/typecast.hpp>
#include <tt_metal/detail/persistent_kernel_cache.hpp>
#include <ttnn/operations/normalization/softmax/softmax.hpp>


#include <memory>
#include <type_traits>
#include <variant>

struct ggml_backend_metalium_context {
    ttnn::device::Device* device = nullptr;
    int device_id = 0;
    std::string name;
};

struct TensorWithMetadata;

static std::string dump_tt_tensor(const tt::tt_metal::Tensor& tensor)
{
    std::stringstream ss;
    auto tmp = ttnn::untilize(tensor);
    std::vector<bfloat16> vec(tmp.shape().volume());
    memcpy(vec.data(), tmp);
    for(size_t i = 0; i < vec.size(); i++) {
        if(i % 32 == 0) {
            ss << std::endl;
        }
        if(i % 1024 == 0) {
            ss << std::endl;
        }
        ss << vec[i].to_float() << " ";
    }
    ss << std::endl;
    return ss.str();
}

struct ggml_backend_metalium_buffer_context {

    size_t ggml_buffer_size_bytes = 0;
    std::string name;
    ttnn::device::Device* device = nullptr;
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
    return true;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////
// Backend internal state tracking because GGML API does not allow
///////////////////////////////////////////////////////////////////////////////////////////////////////

// maps device id to device
static std::map<int, ttnn::Device*> g_device_map;
static std::map<int, ggml_backend_t> g_backend_map;

// Maintain all base addresses are unique
// TODO: Do we still need this since we already removed the virtual address mapping hack?
static size_t g_metalium_base_offset = 0;

///////////////////////////////////////////////////////////////////////////////////////////////////////
// Actual backend code
///////////////////////////////////////////////////////////////////////////////////////////////////////

static tt::tt_metal::DataType ggml2tt_type_internal(ggml_type ggtype, tt::ARCH arch) {
    // This table is consulted to map GGML types to TT types dueing tensor creation
    if(arch == tt::ARCH::GRAYSKULL) {
        static constexpr std::array<tt::tt_metal::DataType, GGML_TYPE_COUNT> table = {
            /*GGML_TYPE_F32 = */ tt::tt_metal::DataType::BFLOAT16,
            /*GGML_TYPE_F16 = */ tt::tt_metal::DataType::BFLOAT16,
            /*GGML_TYPE_Q4_0 = */ tt::tt_metal::DataType::BFLOAT8_B,    // Using BFLOAT8_B for now as BFLOAT4_B is broken on Grayskull
            /*GGML_TYPE_Q4_1 = */ tt::tt_metal::DataType::INVALID,      // Does work but causes issues in unit tests
            tt::tt_metal::DataType::INVALID,
            tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_Q5_0 = */ tt::tt_metal::DataType::BFLOAT8_B,
            /*GGML_TYPE_Q5_1 = */ tt::tt_metal::DataType::INVALID,      // Does work but causes issues in unit tests
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
    GGML_ASSERT(type != tt::tt_metal::DataType::INVALID && "Unsupported data type");
    return type;

}

static bool is_ggml_type_supported_by_metalium(ggml_type ggtype, tt::ARCH arch) {
    return ggml2tt_type_internal(ggtype, arch) != tt::tt_metal::DataType::INVALID;
}

template <typename SrcType, typename DstType>
tt::tt_metal::OwnedStorage data2owned_storage(const SrcType* src, size_t size) {
    // Converts GGML types to TT types
    // TODO: Support quantized data conversion
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
        else {
            dst = val;
        }
    };

    // special case for F32 and BF16 since no conversion is needed
    if constexpr(std::is_same_v<Src, Dst> || (std::is_same_v<Src, ggml_fp16_t> && std::is_same_v<Dst, bfloat16>)) {
        // Make GCC shut up about writing into a class like it's flat memory
        memcpy((void*)vec.data(), src, size * sizeof(Src));
    }
    else {
        for(size_t i = 0; i < size; i++) {
            dst_adaptor(vec[i], src_adaptor(src[i]));
        }
    }
    auto owned = tt::tt_metal::owned_buffer::create(std::move(vec));
    return OwnedStorage(std::move(owned));
}

template <typename DstType>
tt::tt_metal::OwnedStorage ggml_quantized2owned_storage(const void* src, ggml_tensor* tensor) {
    ggml_type_traits_t trait = ggml_internal_get_type_traits(tensor->type);
    GGML_ASSERT(trait.to_float != NULL);
    std::vector<float> vec(ggml_nelements(tensor));
    trait.to_float(src, vec.data(), ggml_nelements(tensor));

    return data2owned_storage<float, bfloat16>(vec.data(), vec.size());
}

template <typename SrcType>
void tensor2ggml(const tt::tt_metal::Tensor& tensor, void* dst, [[maybe_unused]] tt::tt_metal::CommandQueue& queue, ggml_type dst_ggtype) {
    // Converts TT tensors to GGML types
    // TODO: Support reading quantized data
    ttnn::Shape shape = tensor.shape();
    ttnn::Shape padded_shape = tensor.shape().with_tile_padding();
    static_assert(std::is_same_v<SrcType, float> || std::is_same_v<SrcType, bfloat16>);

    tt::tt_metal::Tensor row_major_tensor = tensor.cpu().to(tt::tt_metal::Layout::ROW_MAJOR);
    GGML_ASSERT(row_major_tensor.storage_type() == StorageType::OWNED or row_major_tensor.storage_type() == StorageType::BORROWED);
    GGML_ASSERT(std::holds_alternative<OwnedStorage>(row_major_tensor.storage()) || std::holds_alternative<BorrowedStorage>(row_major_tensor.storage()));

    std::span<SrcType> buf;
    if(std::holds_alternative<OwnedStorage>(row_major_tensor.storage())) {
        const OwnedStorage& owned = std::get<OwnedStorage>(row_major_tensor.storage());
        auto buffer = std::get<owned_buffer::Buffer<SrcType>>(owned.buffer);
        buf = std::span<SrcType>(buffer.begin(), buffer.end());
    }
    else if(std::holds_alternative<BorrowedStorage>(row_major_tensor.storage())) {
        const BorrowedStorage& borrowed = std::get<BorrowedStorage>(row_major_tensor.storage());
        auto buffer = std::get<borrowed_buffer::Buffer<SrcType>>(borrowed.buffer);
        buf = std::span<SrcType>(buffer.begin(), buffer.end());
    } else {
        GGML_ASSERT(false && "Unsupported buffer type");
    }
    GGML_ASSERT(buf.size() != 0);
    // TODO: Measure the performance of the following code. This is much simpeer and does untiling on the device
    // But does not work for large tensors
    // row_major_tensor = ttnn::untilize(tensor);
    // GGML_ASSERT(row_major_tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR);
    // tt::tt_metal::memcpy(queue, buf.data(), row_major_tensor);
    // tt::tt_metal::Finish(queue);
    void* intermid = nullptr;
    std::vector<uint8_t> intermid_buf;
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
    // TODO: Make sure this is correct. As of now not tested for large (>32x32) tensors
    // TODO: There's a lot of optimization that can be done here
    // TODO: Chunk this loop to avoid cache misses
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
                    memcpy((SrcType*)intermid + idx, buf.data() + src_idx, sizeof(SrcType) * shape[3]);
                    idx += shape[3];
                }
                else {
                    for(size_t x = 0; x < shape[3]; x++) {
                        const size_t src_idx = w * stride[0] + z * stride[1] + y * stride[2] + x * stride[3];
                        GGML_ASSERT(src_idx < buf.size());
                        float val = src_adaptor(buf[src_idx]);
                        ((float*)intermid)[idx] = val;
                        idx++;
                    }
                }
            }
        }
    }

    if (need_quantized_conversion) {
        GGML_ASSERT((ggml_is_quantized(dst_ggtype) || dst_ggtype == GGML_TYPE_F16) && "This block should only reach for quantized data types");
        GGML_ASSERT(intermid_buf.size() != 0);
        size_t real_volume = shape[0] * shape[1] * shape[2] * shape[3];
        ggml_type_traits_t trait = ggml_internal_get_type_traits(dst_ggtype);
        GGML_ASSERT(trait.to_float != NULL);
        trait.from_float((float*)intermid, dst, real_volume);
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
    std::vector<uint32_t> target_shape(GGML_MAX_DIMS, 1);
    for(int i = 0; i < GGML_MAX_DIMS; i++) {
        target_shape[i] = node->ne[GGML_MAX_DIMS - i - 1];
    }

    if(node->ne[0] % tt::constants::TILE_WIDTH != 0 || node->ne[1] % tt::constants::TILE_HEIGHT != 0) {
        // This path is SLOW. Reshape on a tilized tensor only works when the last two dimensions are tile aligned
        tt::tt_metal::Tensor row_major_tensor = ttnn::untilize(tensor);
        tt::tt_metal::Tensor reshaped = row_major_tensor.reshape(target_shape);
        tt::tt_metal::Tensor ret = ttnn::tilize_with_zero_padding(reshaped);
        return ret;
    }

    return tensor.reshape(target_shape);
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

    // std::cout << "\nrealize_ggml_view() OP: " << ggml_op_desc(tensor) << std::endl;
    // std::cout << "  dst shape: " << tensor->ne[0] << " " << tensor->ne[1] << " " << tensor->ne[2] << " " << tensor->ne[3] << std::endl;
    // std::cout << "  dst stride: " << tensor->nb[0] << " " << tensor->nb[1] << " " << tensor->nb[2] << " " << tensor->nb[3] << std::endl;
    // std::cout << "  dst extra: " << tensor->extra << std::endl;
    // if(tensor->extra != nullptr) {
    //     TensorWithMetadata* meta = (TensorWithMetadata*)tensor->extra;
    //     std::cout << "  dst tensor: " << meta->tensor << std::endl;
    //     if(meta->tensor != nullptr) {
    //         std::cout << "  dst tensor shape: " << meta->tensor->shape() << std::endl;
    //     }
    // }
    // std::cout << "  dst data: " << tensor->data << std::endl;
    // std::cout << "  dst view_src: " << tensor->view_src << std::endl;
    // std::cout << "  dst src0: " << src0 << std::endl;
    // std::cout << "  dst src1: " << tensor->src[1] << std::endl;

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
            end[i] = dst_size[i] + start[i] - 1;
            remaining_offset = remaining_offset % src_stride[i];
        }
        std::reverse(start.begin(), start.end());
        std::reverse(end.begin(), end.end());
        tt::tt_metal::Tensor res;
        if(dst_size[0] % tt::constants::TILE_WIDTH == 0 && dst_size[1] % tt::constants::TILE_HEIGHT == 0 &&
            start[2] % tt::constants::TILE_WIDTH == 0 && start[3] % tt::constants::TILE_HEIGHT == 0) {
            res = ttnn::slice(*parent, tt::tt_metal::LegacyShape(start), tt::tt_metal::LegacyShape(end), std::nullopt, tt::tt_metal::MemoryConfig());
        }
        else {
            // THIS is EXTREMELY SLOW. But it works
            tt::tt_metal::Tensor tmp = parent->cpu().to(tt::tt_metal::Layout::ROW_MAJOR).unpad(start, end);
            res = ttnn::tilize_with_zero_padding(tmp.to(bufctx->device));
        }
        return std::make_shared<tt::tt_metal::Tensor>(res);
    }
    if(op == GGML_OP_RESHAPE) {
        auto t = realize_ggml_view(src0);
        return std::make_shared<tt::tt_metal::Tensor>(reshape_tt_tensor_into_ggml(*t, tensor));
    }
    if(op == GGML_OP_PERMUTE) {
        int ndiff = 0;
        for(int i=0;i<GGML_MAX_DIMS;i++) {
            ndiff += tensor->nb[i] != src0->nb[i];
        }
        GGML_ASSERT(ndiff == 2);

        auto t = realize_ggml_view(src0);
        if(ndiff == 0) {
            return t;
        }

        std::array<uint32_t, 2> swapaxis = {0, 1};
        uint32_t count = 0;
        for(uint32_t i=0;i<GGML_MAX_DIMS;i++) {
            if(tensor->nb[i] != src0->nb[i]) {
                swapaxis[count] = i;
                count++;
            }
            GGML_ASSERT(count <= swapaxis.size());
        }

        auto res = ttnn::transpose(*t, swapaxis[0], swapaxis[1]);
        return std::make_shared<tt::tt_metal::Tensor>(res);
    }

    if(TensorWithMetadata* meta = (TensorWithMetadata*)tensor->extra; meta != nullptr && meta->tensor != nullptr) {
        return meta->tensor;
    }

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
#define GGML_METALIUM_OP_CHECK_TTTENSOR(_node, _idx) \
    do { \
        auto _meta = (TensorWithMetadata*)((_node)->src[_idx]->extra); \
        GGML_ASSERT(_meta != NULL); \
        GGML_ASSERT(_meta->tensor != NULL); \
    } while(0)
#define GGML_METALIUM_OP_SRC0_CHECK_TTTENSOR(_node) GGML_METALIUM_OP_CHECK_TTTENSOR(_node, 0)
#define GGML_METALIUM_OP_SRC1_CHECK_TTTENSOR(_node) GGML_METALIUM_OP_CHECK_TTTENSOR(_node, 1)

static bool ggml_backend_metalium_can_mul_mat(const struct ggml_tensor * dst)
{
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    // TTNN only supports matmul of shape [B, 1, M, K] x [1, 1, K, N] (bcast_batch=True)
    // or [B, 1, M, K] x [B, 1, K, N] (bcast_batch=False)
    // For now we simply only allow those shapes. We transpose the shapes ourselves
    // TODO: Detect when shape[1] can be removed and do that automagically

    return src0->ne[0] == src1->ne[0] && src0->ne[2] == src1->ne[2] &&
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
    // TODO: Ask TT to support multiplication of pre-transposed tensors. Calling transpose here is inefficient
    // https://github.com/tenstorrent/tt-metal/issues/9709

    if(a.dtype() == tt::tt_metal::DataType::BFLOAT16 && b.dtype() == tt::tt_metal::DataType::BFLOAT16) {
        // Fast path
        *cm = {
            .tensor = std::make_shared<tt::tt_metal::Tensor>(ttnn::moreh_matmul(b, a, false, true, std::nullopt, std::nullopt, std::nullopt, std::nullopt)),
            .ggtype = dst->type,
            .bufctx = cm->bufctx
        };
    }
    else {
        auto aT = ttnn::transpose(a, -2, -1);
        ttnn::operations::matmul::Matmul cfg = ttnn::operations::matmul::Matmul{};
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
    TensorWithMetadata* dst_meta = (TensorWithMetadata*)dst->extra;

    // TODO: Check we are not writing into a view
    auto res = realize_ggml_view(dst->src[0]);
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
            ret = ttnn::experimental::typecast(ttnn::gtz(*src_tensor), ggml2tt_type(dst->type, tt::ARCH::GRAYSKULL));
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

static void ggml_backend_metalium_transpose(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst)
{
    GGML_UNUSED(ctx);
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);

    auto t = realize_ggml_view(dst->src[0]);
    TensorWithMetadata* dst_meta = (TensorWithMetadata*)dst->extra;

    // std::cout << "GGML wants reshape to: " << dst->ne[0] << " " << dst->ne[1] << " " << dst->ne[2] << " " << dst->ne[3] << std::endl;

    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(ttnn::transpose(*t, -2, -1)),
        .ggtype = dst->type,
        .bufctx = ((TensorWithMetadata*)dst->src[0]->extra)->bufctx
    };
    // std::cout << "TT wants reshape to: " << dst_meta->tensor->shape() << std::endl;
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
    if(dst->src[1] != nullptr && arr[1] != 0.f) {
        return false;
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
            x = ttnn::add(x, *mask);
        }
        else {
            // This path is not used due to accuracy issues and bug in TTNN.
            // TODO: Revive it later
            const uint32_t n_head = t->shape()[1];
            const uint32_t n_head_log2 = 1u << (uint32_t) std::ceil(std::log2(n_head));
            const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
            // const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);
            auto make_tile = [](const tt::tt_metal::Tensor& t, tt::tt_metal::Device* dev) {
                return ttnn::tilize_with_zero_padding(t).to(dev);
            };

            // const float slope = (max_bias > 0.0f) ? h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_head_log2) + 1) : 1.0f;
            auto *dev = t->device();
            auto slope = make_tile(tt::numpy::arange<float>(1, n_head+1, 1), dev);
            auto lim = make_tile(tt::numpy::full(slope.legacy_shape(), (float)n_head_log2, tt::tt_metal::DataType::BFLOAT16), dev);
            // BUG: Results in the wrong shape
            // slope = tt::tt_metal::max(slope, lim);
            slope = ttnn::rpow(ttnn::add(slope, 1.f), m0);


            x = ttnn::add(x, ttnn::multiply(*mask, slope));
        }
    }
    x = ttnn::operations::normalization::softmax(x);
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

// backend interface

GGML_CALL static const char * ggml_backend_metalium_name(ggml_backend_t backend) {
    return "Metalium";

    GGML_UNUSED(backend);
}

GGML_CALL static void ggml_backend_metalium_free(ggml_backend_t backend) {
    ggml_backend_metalium_context * ctx = (ggml_backend_metalium_context *)backend->context;
    ctx->device->close();
    delete ctx;
    delete backend;
}

struct ggml_backend_metalium_buffer_type_context {
    ttnn::Device* device = nullptr;
    std::string name;
};

GGML_CALL static const char * ggml_backend_metalium_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_metalium_buffer_type_context * ctx = (ggml_backend_metalium_buffer_type_context *)buft->context;

    return ctx->name.c_str();
}

GGML_CALL static size_t ggml_backend_metalium_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    // Not using this. Metalium's allication model is not compatible with GGML's allocator
    return 128;
    GGML_UNUSED(buft);
}

// NOTE: I might need to add a metalium tensor wrapper to work around TT tensors have hardware-tagged data types
//       and GGML tensors does not specify the data type during tensor creation.
static size_t ggml_backend_metalium_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_backend_metalium_buffer_type_context * ctx = (ggml_backend_metalium_buffer_type_context *)buft->context;
    return ctx->device->num_dram_channels() * (size_t)ctx->device->dram_size_per_channel();
}

GGML_CALL static size_t ggml_backend_metalium_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    // Not using this. Metalium's allication model is not compatible with GGML's allocator
    return ggml_nbytes(tensor);
    GGML_UNUSED(buft);
}

GGML_CALL static const char * ggml_backend_metalium_buffer_get_name(ggml_backend_buffer_t buffer) {
    ggml_backend_metalium_buffer_context * ctx = (ggml_backend_metalium_buffer_context *)buffer->context;
    return ctx->name.c_str();
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

    // std::cout << "Writing to tensor with address: " << tensor->data << std::endl;

    tt::ARCH processor_class = bufctx->device->arch();
    // only grayskull is supported for now.
    // TODO: Wormhole support
    GGML_ASSERT(processor_class == tt::ARCH::GRAYSKULL);

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
    else {
        GGML_ASSERT(false && "Unsupported data type");
    }

    // TODO: Make sure this is correct
    std::vector<uint32_t> shape(GGML_MAX_DIMS, 1);
    for(int i = 0; i < GGML_MAX_DIMS; i++) {
        // GGML stores the shape in reverse order
        shape[i] = tensor->ne[GGML_MAX_DIMS - i - 1];
    }

    tt::tt_metal::Tensor t(std::move(storage), shape
        , tt::tt_metal::DataType::BFLOAT16, tt::tt_metal::Layout::ROW_MAJOR);

    // I think we can allow this.. right?
    // GGML_ASSERT(!bufctx->tensors.contains(offset));

    // TODO: Make sure this is the correct tilize we want to use
    t = ttnn::tilize_with_zero_padding(t.to(bufctx->device));
    tt::tt_metal::DataType final_type = ggml2tt_type(ggtype, processor_class);
    if(final_type != t.dtype()) {
        t = ttnn::experimental::typecast(t, final_type);
    }
    GGML_ASSERT(t.storage_type() == tt::tt_metal::StorageType::DEVICE || t.storage_type() == tt::tt_metal::StorageType::MULTI_DEVICE);
    GGML_ASSERT(t.dtype() == final_type);
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
    GGML_UNUSED(offset);

    ggml_backend_metalium_buffer_context * ctx = (ggml_backend_metalium_buffer_context *)buffer->context;

    ggml_type dst_ggtype = tensor->type;
    tt::tt_metal::CommandQueue& queue = ctx->device->command_queue(0);

    // auto *meta = (TensorWithMetadata*)tensor->extra;
    // auto shape = meta->tensor->shape();
    // std::cout << "get_tensor():\n";
    // std::cout << "  GGML thinks shape: " << tensor->ne[0] << " " << tensor->ne[1] << " " << tensor->ne[2] << " " << tensor->ne[3] << std::endl;
    // std::cout << "  TTNN thinks shape: " << shape << std::endl;
    std::shared_ptr<tt::tt_metal::Tensor> t = realize_ggml_view(tensor);
    GGML_ASSERT(ggml_tt_tensors_shape_equal(tensor, *t));
    tt::tt_metal::Tensor holder;
    if(t->dtype() != tt::tt_metal::DataType::BFLOAT16 || t->dtype() != tt::tt_metal::DataType::FLOAT32) {
        holder = ttnn::experimental::typecast(*t, tt::tt_metal::DataType::BFLOAT16);
        t = std::make_shared<tt::tt_metal::Tensor>(std::move(holder));
    }

    // TODO: Proper handling of data types
    GGML_ASSERT(dst_ggtype != GGML_TYPE_F64 && dst_ggtype != GGML_TYPE_I16 && dst_ggtype != GGML_TYPE_I8 && dst_ggtype != GGML_TYPE_I32);
    switch(t->dtype()) {
        case tt::tt_metal::DataType::BFLOAT16:
            tensor2ggml<bfloat16>(*t, (float*)data, queue, dst_ggtype);
            break;
        case tt::tt_metal::DataType::FLOAT32:
            tensor2ggml<float>(*t, (float*)data, queue, dst_ggtype);
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

GGML_CALL static void
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
    if(std::string_view(tensor->name).find("cache") != std::string::npos) {
        TensorWithMetadata* meta = (TensorWithMetadata*)tensor->extra;
        std::vector<uint32_t> shape(tensor->ne, tensor->ne + GGML_MAX_DIMS);
        std::reverse(shape.begin(), shape.end());
        auto t = tt::numpy::zeros(shape, ggml2tt_type(tensor->type, bufctx->device->arch()));
        t = ttnn::tilize_with_zero_padding(t.to(bufctx->device));
        meta->tensor = std::make_shared<tt::tt_metal::Tensor>(std::move(t));
    }
    // std::cout << "Creating tensor with address: " << tensor->data << ", shape = " << tensor->ne[0] << " " << tensor->ne[1] << " " << tensor->ne[2] << " " << tensor->ne[3] << ", name " << tensor->name << std::endl;
    GGML_UNUSED(buffer);
}

GGML_CALL static void ggml_backend_metalium_buffer_clear(ggml_backend_buffer_t buffer,
                                                        uint8_t value)
{
    // Not using this. Metalium's allication model is not compatible with GGML's allocator
    GGML_UNUSED(buffer);
    GGML_UNUSED(value);
}

GGML_CALL static bool
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

    tt::tt_metal::Tensor ret = tt::numpy::zeros_like(src_tensor);
    ret.deepcopy(src_tensor);
    GGML_ASSERT(ret.storage_type() == tt::tt_metal::StorageType::DEVICE || ret.storage_type() == tt::tt_metal::StorageType::MULTI_DEVICE);
    dst_meta->tensor = std::make_shared<tt::tt_metal::Tensor>(std::move(ret));
    dst_meta->ggtype = dst->type;
    return true;
}

static struct ggml_backend_buffer_i ggml_backend_metalium_buffer_interface = {
    /* .get_name        = */ ggml_backend_metalium_buffer_get_name,
    /* .free_buffer     = */ ggml_backend_metalium_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_metalium_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_metalium_buffer_init_tensor,
    /* .set_tensor      = */ ggml_backend_metalium_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_metalium_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_metalium_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_metalium_buffer_clear,
    /* .reset           = */ nullptr,
};


GGML_CALL static ggml_backend_buffer_t
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

static ggml_backend_buffer_type_i ggml_backend_metalium_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_metalium_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_metalium_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_metalium_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_metalium_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_metalium_buffer_type_get_alloc_size,
    /* .is_host          = */ nullptr,
};

ggml_backend_buffer_type_t ggml_backend_metalium_buffer_type(int device) {
    GGML_ASSERT((size_t)device < tt::tt_metal::GetNumAvailableDevices());
    static std::map<int, ggml_backend_buffer_type> buffer_type_map;
    static std::set<std::unique_ptr<ggml_backend_metalium_buffer_type_context>> buffer_type_context_deleter;

    if(!g_device_map.contains(device)) {
        ggml_backend_metalium_init();
        GGML_ASSERT(g_device_map.contains(device));
    }

    if(buffer_type_map.contains(device)) {
        return &buffer_type_map[device];
    }


    auto bufctx = std::make_unique<ggml_backend_metalium_buffer_type_context>(
        ggml_backend_metalium_buffer_type_context{
            .device = g_device_map[device],
            .name = "Metalium " + std::to_string(device),
        });
    auto* bufctx_ptr = bufctx.get();
    buffer_type_context_deleter.insert(std::move(bufctx));
    buffer_type_map[device] = {
        /* .iface    = */ ggml_backend_metalium_buffer_type_interface,
        /* .context  = */ bufctx_ptr,
    };
    return &buffer_type_map[device];
}

GGML_CALL static ggml_backend_buffer_type_t ggml_backend_metalium_get_default_buffer_type(ggml_backend_t backend) {
    auto* ctx = (ggml_backend_metalium_context *)backend->context;
    return ggml_backend_metalium_buffer_type(ctx->device_id);
}

GGML_CALL static enum ggml_status ggml_backend_metalium_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
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
        GGML_ASSERT(ggml_tt_tensors_shape_equal(node, *meta->tensor));
    }

    return GGML_STATUS_SUCCESS;

    GGML_UNUSED(backend);
}

GGML_CALL static bool ggml_backend_metalium_supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    ggml_backend_metalium_context * ctx = (ggml_backend_metalium_context *)backend->context;

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
                return tensor->ne[0] % 2 == 0;
            case tt::tt_metal::DataType::FLOAT32:
            case tt::tt_metal::DataType::UINT32:
                return true;
            case tt::tt_metal::DataType::UINT8:
                return tensor->ne[0] % 4 == 0;
            case tt::tt_metal::DataType::INVALID:
                GGML_ASSERT(false && "Unsupported data type");
                break;
            default:
                return tensor->ne[0] % 32 == 0;
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
    GGML_ASSERT(op != NULL);
    if(!tensor_supported(op)) {
        return false;
    }
    if(op->op == GGML_OP_NONE) {
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
        // case GGML_OP_PERMUTE: // FIXME: Needs fix https://github.com/tenstorrent/tt-metal/issues/11650
        // case GGML_OP_SIN:     // Sin and Cos disabled due to bug in TTNN until fixed
        // case GGML_OP_COS:     // ref: https://github.com/tenstorrent/tt-metal/issues/12753
        case GGML_OP_LOG:
            return true;

        // TTNN can really only do unpad() so the source rank must be greater than or equal to the destination rank
        // and must not be permuted as that's a sign of it being reshaped from another tensor. Which is costly due to
        // TTNN not using row-major layout.
        case GGML_OP_VIEW:
            return ggml_n_dims(op) <= ggml_n_dims(src0) && ggml_is_permuted(op) == false && ggml_is_permuted(src0) == false;

        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_DIV:
        case GGML_OP_MUL:
            // DIV does not support broadcasting on TTNN
            return tensor_supported(src1) &&
                (memcmp(src0->ne, src1->ne, sizeof(src0->ne)) == 0 || (numpy_broadcast_rule(src0, src1) && op->op != GGML_OP_DIV));
        case GGML_OP_MUL_MAT:
            return tensor_supported(src1) && ggml_backend_metalium_can_mul_mat(op);
        case GGML_OP_SET:
            return tensor_supported(src1) && ggml_backend_metalium_can_set(op);
        case GGML_OP_GET_ROWS:
            return tensor_supported(src1) && ggml_backend_metalium_can_get_row(op);
        case GGML_OP_CONCAT:
            return tensor_supported(src1) && ggml_backend_metalium_can_concat(op);
        case GGML_OP_SOFT_MAX:
            return ggml_backend_metalium_can_softmax(op);
        default:
            return false;
    }
}

GGML_CALL static bool ggml_backend_metalium_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    if (buft->iface.get_name != ggml_backend_metalium_buffer_type_name) {
        return false;
    }
    ggml_backend_metalium_buffer_type_context * buft_ctx = (ggml_backend_metalium_buffer_type_context *)buft->context;
    ggml_backend_metalium_context * ctx = (ggml_backend_metalium_context *)backend->context;
    return buft_ctx->device == ctx->device;
}

static void ggml_backend_metalium_synchronize(ggml_backend_t backend)
{
    ggml_backend_metalium_context * ctx = (ggml_backend_metalium_context *)backend->context;
    tt::tt_metal::Finish(ctx->device->command_queue());
}

static struct ggml_backend_i metalium_backend_i = {
    /* .get_name                = */ ggml_backend_metalium_name,
    /* .free                    = */ ggml_backend_metalium_free,
    /* .get_default_buffer_type = */ ggml_backend_metalium_get_default_buffer_type,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ ggml_backend_metalium_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_metalium_graph_compute,
    /* .supports_op             = */ ggml_backend_metalium_supports_op,
    /* .supports_buft           = */ ggml_backend_metalium_supports_buft,
    /* .offload_op              = */ NULL,
    /* .event_new               = */ NULL,
    /* .event_free              = */ NULL,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .event_synchronize       = */ NULL,
};

static ggml_guid_t ggml_backend_metalium_guid(void) {
    static ggml_guid guid = { 0x91, 0x69, 0xd5, 0x5f, 0x24, 0xe7, 0x44, 0x00, 0xb4, 0x2a, 0x73, 0x23, 0x48, 0xb0, 0x4e, 0xe7 };
    return &guid;
}

ggml_backend_t ggml_backend_metalium_init(void) {
    // TODO: Support multiple devices (do we even need to? TT supports merging diverse devices into a single device, at least the API suggests that)
    const int device_id = 0;
    static std::once_flag once;
    std::call_once(once, [](){
        tt::tt_metal::detail::EnablePersistentKernelCache();
    });

    auto it = g_backend_map.find(device_id);
    if (it != g_backend_map.end()) {
        return it->second;
    }

    ggml_backend_metalium_context * ctx = new ggml_backend_metalium_context {
        /* device            = */ &ttnn::device::open_device(device_id),
        /* device_id         = */ device_id,
        /* name              = */ "Metalium " + std::to_string(device_id),
    };
    ttnn::operations::experimental::auto_format::AutoFormat::SetDefaultDevice(ctx->device);


    // store the device in the global map because tensor creation uses device ID but Metalium disallows opening the same device twice
    g_device_map[device_id] = ctx->device;
    ttnn::enable_program_cache(*ctx->device);

    ggml_backend_t backend = new ggml_backend {
        /* .guid      = */ ggml_backend_metalium_guid(),
        /* .interface = */ metalium_backend_i,
        /* .context   = */ ctx,
    };
    g_backend_map[device_id] = backend;
    return backend;
}

GGML_CALL bool ggml_backend_is_metalium(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_metalium_guid());
}


GGML_CALL ggml_backend_t ggml_backend_reg_metalium_init(const char * params, void * user_data)
{
    // Sanity check for the environment
    static_assert(tt::tt_metal::MAX_NUM_DIMENSIONS >= GGML_MAX_DIMS, "tt::tt_metal::MAX_NUM_DIMENSIONS must be at least GGML_MAX_DIMS");

    GGML_UNUSED(params);
    GGML_UNUSED(user_data);
    return ggml_backend_metalium_init();
}
