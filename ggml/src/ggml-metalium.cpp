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
#include <ttnn/operations/matmul.hpp>


#include <memory>
#include <type_traits>
#include <unordered_map>
#include <variant>

struct ggml_backend_metalium_context {
    ttnn::device::Device* device = nullptr;
    int device_id = 0;
    std::string name;
};

struct TensorWithMetadata;

// GGML views are lazy and it's not 100% of the times we get a tensor with `extra` populated to direct the writes.
// Solution? Look at the tensor's supposed address that WE faked and figure out which tensor it is referring to.
struct FakeMemoryMap
{
    TensorWithMetadata* find(std::ptrdiff_t address)
    {
        // auto it = std::lower_bound(address_tensor_map.begin(), address_tensor_map.end(), address, [](const auto& pair, std::ptrdiff_t address) {
        //     return pair.first < address;
        // });
        // if(it == address_tensor_map.end() || it->first != address) {
        //     return nullptr;
        // }
        // std::cout << "Looking for address: " << (void*)address << std::endl;
        auto it = address_tensor_map.find(address);
        if(it == address_tensor_map.end()) {
            return nullptr;
        }
        return it->second;
    }

    bool contains(std::ptrdiff_t address) const
    {
        return address_tensor_map.contains(address);
    }

    void insert(std::ptrdiff_t address, TensorWithMetadata* tensor)
    {
        // std::cout << "Adding address: " << (void*)address << std::endl;
        GGML_ASSERT(address_tensor_map.contains(address) == false);
        address_tensor_map[address] = tensor;
        // std::cout << "Virtual table size: " << address_tensor_map.size() << std::endl;
    }


    std::unordered_map<std::ptrdiff_t, TensorWithMetadata*> address_tensor_map;

    using MapType = decltype(address_tensor_map);
    MapType::iterator begin() { return address_tensor_map.begin(); }
    MapType::iterator end() { return address_tensor_map.end(); }
    MapType::const_iterator begin() const { return address_tensor_map.begin(); }
    MapType::const_iterator end() const { return address_tensor_map.end(); }
};

struct ggml_backend_metalium_buffer_context {

    size_t ggml_buffer_size_bytes = 0;
    std::string name;
    ttnn::device::Device* device = nullptr;
    size_t base_offset = 0;

    // Tracking our own allocations because Metalium limitations and GGML assuming them
    std::vector<std::unique_ptr<TensorWithMetadata>> metadata_to_free;
    FakeMemoryMap address_tensor_map;
};

struct TensorWithMetadata
{
    std::shared_ptr<tt::tt_metal::Tensor> tensor;
    ggml_type ggtype = (ggml_type)-1;
    ggml_backend_metalium_buffer_context* bufctx = nullptr;
};

static const tt::tt_metal::Tensor& resolve_from_ggml_tensor(const ggml_tensor* tensor, ggml_backend_metalium_buffer_context* bufctx)
{
    GGML_ASSERT(tensor->extra != NULL);
    const auto *meta = (TensorWithMetadata*)tensor->extra;
    if(meta->tensor != nullptr) {
        return *meta->tensor;
    }

    std::ptrdiff_t ptr = 0;
    if(tensor->view_src != nullptr) {
        // std::cout << "View data: " << (void*)tensor->view_src->data << " View offs: " << tensor->view_offs << std::endl;
        ptr = (std::ptrdiff_t)tensor->view_src->data - tensor->view_offs;
    }
    else {
        // std::cout << "Data: " << (void*)tensor->data << std::endl;
        ptr = (std::ptrdiff_t)tensor->data;
    }
    // std::cout << "Virtual table size: " << bufctx->address_tensor_map.address_tensor_map.size() << std::endl;
    // for(auto& [addr, _] : bufctx->address_tensor_map.address_tensor_map) {
    //     std::cout << "  Entry address: " << (void*)addr << std::endl;
    // }
    auto *indirect_meta = bufctx->address_tensor_map.find((std::ptrdiff_t)ptr);
    GGML_ASSERT(indirect_meta != nullptr);
    GGML_ASSERT(indirect_meta->tensor != nullptr);
    return *indirect_meta->tensor;
}

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
            /*GGML_TYPE_Q4_0 = */ tt::tt_metal::DataType::BFLOAT4_B,
            /*GGML_TYPE_Q4_1 = */ tt::tt_metal::DataType::INVALID,      // Does work but causes issues in unit tests
            tt::tt_metal::DataType::INVALID,
            tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_Q5_0 = */ tt::tt_metal::DataType::BFLOAT8_B,
            /*GGML_TYPE_Q5_1 = */ tt::tt_metal::DataType::INVALID,      // Does work but causes issues in unit tests
            /*GGML_TYPE_Q8_0 = */ tt::tt_metal::DataType::BFLOAT8_B,
            /*GGML_TYPE_Q8_1 = */ tt::tt_metal::DataType::BFLOAT8_B,
            /*GGML_TYPE_Q2_K = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_Q3_K = */ tt::tt_metal::DataType::BFLOAT4_B,
            /*GGML_TYPE_Q4_K = */ tt::tt_metal::DataType::BFLOAT4_B,
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

static size_t tttype_size(tt::tt_metal::DataType type) {
    switch(type) {
        case tt::tt_metal::DataType::BFLOAT16:
            return sizeof(bfloat16);
        case tt::tt_metal::DataType::FLOAT32:
            return sizeof(float);
        default:
            GGML_ASSERT(false && "Unsupported data type");
    }
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
    static_assert(std::is_same_v<SrcType, float> || std::is_same_v<SrcType, bfloat16>);

    tt::tt_metal::Tensor row_major_tensor = tensor.cpu().to(tt::tt_metal::Layout::ROW_MAJOR);
    std::vector<SrcType> buf(shape.volume()); // .volume() returns the underlying volume of the tensor not the logical one (TT enforces 32x32 tiles)
    GGML_ASSERT(row_major_tensor.storage_type() == StorageType::OWNED or row_major_tensor.storage_type() == StorageType::BORROWED);
    GGML_ASSERT(std::holds_alternative<OwnedStorage>(row_major_tensor.storage()) || std::holds_alternative<BorrowedStorage>(row_major_tensor.storage()));
    if(std::holds_alternative<OwnedStorage>(row_major_tensor.storage())) {
        const OwnedStorage& owned = std::get<OwnedStorage>(row_major_tensor.storage());
        memcpy(buf.data(), std::get<owned_buffer::Buffer<SrcType>>(owned.buffer).data(), shape.volume() * sizeof(SrcType));
    }
    else if(std::holds_alternative<BorrowedStorage>(row_major_tensor.storage())) {
        const BorrowedStorage& borrowed = std::get<BorrowedStorage>(row_major_tensor.storage());
        memcpy(buf.data(), std::get<borrowed_buffer::Buffer<SrcType>>(borrowed.buffer).data(), shape.volume() * sizeof(SrcType));
    } else {
        GGML_ASSERT(false && "Unsupported buffer type");
    }
    // TODO: Measure the performance of the following code. This is much simpeer and does untiling on the device
    // But does not work for large tensors
    // row_major_tensor = tt::tt_metal::untilize(tensor);
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
    ttnn::Shape tt_underlying_shape = row_major_tensor.shape().with_tile_padding();
    const std::array<size_t, 4> stride = {tt_underlying_shape[1] * tt_underlying_shape[2] * tt_underlying_shape[3],
                                    tt_underlying_shape[2] * tt_underlying_shape[3],
                                    tt_underlying_shape[3],
                                    1};
    static_assert(GGML_MAX_DIMS == 4, "Looping depth is hardcoded to 4");
    size_t idx = 0;
    for(size_t w = 0; w < shape[0]; w++) {
        for(size_t z = 0; z < shape[1]; z++) {
            for(size_t y = 0; y < shape[2]; y++) {
                for(size_t x = 0; x < shape[3]; x++) {
                    const size_t src_idx = w * stride[0] + z * stride[1] + y * stride[2] + x * stride[3];
                    if(src_dst_same) {
                        mempcpy((SrcType*)intermid + idx, buf.data() + src_idx, sizeof(SrcType));
                    }
                    else {
                        float val = src_adaptor(buf[src_idx]);
                        ((float*)intermid)[idx] = val;
                    }
                    idx++;
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

// Sanity check macros to ensure that the tensors are in the correct format and we won't crash
#define GGML_METALIUM_OP_SANITY_CHECK(_node) \
    GGML_ASSERT((_node)->extra != NULL);
// Check if the tensor is on the device (so we wont'e be using the CPU) as well as letting us crash early
#define GGML_METALIUM_OP_SRC_SANITY_CHECK(_node, _idx) \
    do { \
        GGML_ASSERT((_node)->src[_idx] != NULL); \
        GGML_ASSERT((_node)->src[_idx]->extra != NULL); \
        auto _meta = (TensorWithMetadata*)((_node)->src[_idx]->extra); \
        GGML_ASSERT(_meta->tensor != NULL); \
        GGML_ASSERT(_meta->tensor->storage_type() == tt::tt_metal::StorageType::DEVICE || _meta->tensor->storage_type() == tt::tt_metal::StorageType::MULTI_DEVICE); \
        GGML_ASSERT(_meta->tensor->layout() == tt::tt_metal::Layout::TILE); \
    } while(0)
#define GGML_METALIUM_OP_SRC0_SANITY_CHECK(_node) GGML_METALIUM_OP_SRC_SANITY_CHECK(_node, 0)
#define GGML_METALIUM_OP_SRC1_SANITY_CHECK(_node) GGML_METALIUM_OP_SRC_SANITY_CHECK(_node, 1)

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

    tt::tt_metal::Tensor& a = *(((TensorWithMetadata*)src0->extra)->tensor);
    tt::tt_metal::Tensor& b = *(((TensorWithMetadata*)src1->extra)->tensor);
    TensorWithMetadata* cm = (TensorWithMetadata*)dst->extra;

    GGML_ASSERT(cm != NULL);
    auto aT = tt::tt_metal::transpose(a, -2, -1);
    // HACK: Workaround data corruption in TTNN
    // https://github.com/tenstorrent/tt-metal/issues/9849
    cm->tensor.reset();
    // TODO: Ask TT to support multiplication of pre-transposed tensors. Calling transpose here is inefficient
    // https://github.com/tenstorrent/tt-metal/issues/9709
    cm->tensor = std::make_shared<tt::tt_metal::Tensor>(ttnn::operations::matmul::matmul(b, aT, std::nullopt));
    cm->ggtype = dst->type; // Hope this is the correct approach
    GGML_ASSERT(cm->tensor->storage_type() == tt::tt_metal::StorageType::DEVICE || cm->tensor->storage_type() == tt::tt_metal::StorageType::MULTI_DEVICE);
    GGML_UNUSED(ctx);
}

static void ggml_backend_metalium_cpy(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_METALIUM_OP_SANITY_CHECK(dst);

    const struct ggml_tensor * src0 = dst->src[0];
    TensorWithMetadata* meta = (TensorWithMetadata*)src0->extra;
    TensorWithMetadata* dst_meta = (TensorWithMetadata*)dst->extra;

    // std::cout << "VIEW src shape: " << src0->ne[0] << " " << src0->ne[1] << " " << src0->ne[2] << " " << src0->ne[3] << std::endl;
    // std::cout << "VIEW dst shape: " << dst->ne[0] << " " << dst->ne[1] << " " << dst->ne[2] << " " << dst->ne[3] << std::endl;

    GGML_ASSERT(meta != NULL);
    // GGML_ASSERT(ggml_is_contiguous(dst));

    std::array src_size = std::to_array(src0->ne);
    std::array dst_size = std::to_array(dst->ne);
    std::array src_stride = std::to_array(src0->nb);
    std::array<size_t, GGML_MAX_DIMS> dst_stride;

    // TODO: Support quantized data
    size_t stride = 1;
    for(size_t i = 0; i < GGML_MAX_DIMS; i++) {
        dst_stride[i] = stride;
        stride *= dst_size[i];
    }

    // Fast path when both tensors are contiguous and have the same shape
    if(ggml_is_contiguous(src0) &&
        memcmp(src_size.data(), dst_size.data(), GGML_MAX_DIMS * sizeof(size_t)) == 0 &&
        memcmp(src_stride.data(), dst_stride.data(), GGML_MAX_DIMS * sizeof(size_t)) == 0) {
        const auto& ref_tensor = resolve_from_ggml_tensor(src0, dst_meta->bufctx);
        tt::tt_metal::Tensor ret = tt::tt_metal::zeros_like(ref_tensor);
        ret.deepcopy(*meta->tensor);
        GGML_ASSERT(ret.storage_type() == tt::tt_metal::StorageType::DEVICE || ret.storage_type() == tt::tt_metal::StorageType::MULTI_DEVICE);
        *dst_meta = {
            .tensor = std::make_shared<tt::tt_metal::Tensor>(std::move(ret)),
            .ggtype = dst->type,
            .bufctx = meta->bufctx
        };
        return;
    }
    // If just doing transpose, we can just do a reshape
    // Transpose = same shape, but stride of the 1st two dimensions are swapped
    if(src_size[0] == dst_size[0] && src_size[1] == dst_size[1] && src_size[2] == dst_size[2] && src_size[3] == dst_size[3] &&
        src_stride[1] == dst_stride[0] && src_stride[0] == dst_stride[1] / dst_size[1] && src_stride[2] == dst_stride[2] && src_stride[3] == dst_stride[3]) {
        // TODO: I feel this is wrong. But the results are correct. Need to investigate
        const auto& ref_tensor = resolve_from_ggml_tensor(src0, dst_meta->bufctx);
        tt::tt_metal::Tensor ret = tt::tt_metal::zeros_like(ref_tensor);
        ret.deepcopy(*meta->tensor);
        GGML_ASSERT(ret.storage_type() == tt::tt_metal::StorageType::DEVICE || ret.storage_type() == tt::tt_metal::StorageType::MULTI_DEVICE);
        *dst_meta = {
            .tensor = std::make_shared<tt::tt_metal::Tensor>(std::move(ret)),
            .ggtype = dst->type,
            .bufctx = meta->bufctx
        };
        return;
    }

    // Now we deal with non-contiguous source tensors. Metalium can only unpad (slice but without spacing) the tensor

    // Check we can do the unpad
    for(size_t i = 0; i < GGML_MAX_DIMS; i++) {
        GGML_ASSERT(src_size[i] >= dst_size[i]);
        GGML_ASSERT(src_stride[i] >= dst_stride[i]);
    }

    std::array<uint32_t, GGML_MAX_DIMS> start;
    std::array<uint32_t, GGML_MAX_DIMS> end;

    for(size_t i = 0; i < GGML_MAX_DIMS; i++) {
        start[i] = 0; // TODO: How do I calculate the start?
        end[i] = dst_size[i] + start[i] - 1; // end is inclusive (WTF?)
    }
    std::reverse(start.begin(), start.end());
    std::reverse(end.begin(), end.end());
    tt::tt_metal::Tensor res;
    if(dst->ne[0] % tt::constants::TILE_WIDTH == 0 && dst->ne[1] % tt::constants::TILE_HEIGHT == 0) {
        res = tt::tt_metal::unpad(*meta->tensor, start, end);
    }
    else {
        // THIS is EXTREMELY SLOW. But it works
        tt::tt_metal::Tensor tmp = meta->tensor->cpu().to(tt::tt_metal::Layout::ROW_MAJOR).unpad(start, end);
        res = tt::tt_metal::tilize_with_zero_padding(tmp.to(meta->bufctx->device));
    }

    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(res),
        .ggtype = dst->type,
        .bufctx = meta->bufctx
    };
}

static bool ggml_backend_metalium_activations(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst, ggml_unary_op op) {
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    GGML_UNUSED(ctx);

    const struct ggml_tensor * src0 = dst->src[0];
    TensorWithMetadata* meta = (TensorWithMetadata*)src0->extra;
    TensorWithMetadata* dst_meta = (TensorWithMetadata*)dst->extra;

    tt::tt_metal::Tensor ret;
    switch (op) {
        case GGML_UNARY_OP_ABS:
            ret = tt::tt_metal::abs(*meta->tensor);
            break;
        case GGML_UNARY_OP_SGN:
            ret = tt::tt_metal::sign(*meta->tensor);
            break;
        case GGML_UNARY_OP_NEG:
            ret = tt::tt_metal::neg(*meta->tensor);
            break;
        // Not supported in TTNN
        // case GGML_UNARY_OP_STEP:
        //     ret = tt::tt_metal::step(*meta->tensor);
        //     break;
        // Not accurate enough to pass unit tests
        case GGML_UNARY_OP_TANH:
            ret = tt::tt_metal::tanh(*meta->tensor);
            break;
        case GGML_UNARY_OP_ELU:
            ret = tt::tt_metal::elu(*meta->tensor, 1.0f);
            break;
        case GGML_UNARY_OP_RELU:
            ret = tt::tt_metal::relu(*meta->tensor);
            break;
        // Not accurate enough to pass unit tests
        case GGML_UNARY_OP_SIGMOID:
            ret = tt::tt_metal::sigmoid(*meta->tensor);
            break;
        case GGML_UNARY_OP_GELU:
            ret = tt::tt_metal::gelu(*meta->tensor, false);
            break;
        case GGML_UNARY_OP_GELU_QUICK:
            ret = tt::tt_metal::gelu(*meta->tensor);
            break;
        case GGML_UNARY_OP_SILU:
            ret = tt::tt_metal::silu(*meta->tensor);
            break;
        case GGML_UNARY_OP_HARDSWISH:
            ret = tt::tt_metal::hardswish(*meta->tensor);
            break;
        case GGML_UNARY_OP_HARDSIGMOID:
            ret = tt::tt_metal::hardsigmoid(*meta->tensor);
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

    float negative_slope;
    GGML_ASSERT(dst->op_params != NULL);
    memcpy(&negative_slope, dst->op_params, sizeof(float));

    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(tt::tt_metal::leaky_relu(*meta->tensor, negative_slope)),
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
    TensorWithMetadata* meta1 = (TensorWithMetadata*)src1->extra;
    TensorWithMetadata* dst_meta = (TensorWithMetadata*)dst->extra;

    std::shared_ptr<tt::tt_metal::Tensor> ret;
    switch(op) {
        case GGML_OP_ADD:
            ret = std::make_shared<tt::tt_metal::Tensor>(ttnn::add(*meta0->tensor, *meta1->tensor));
            break;
        case GGML_OP_MUL:
            ret = std::make_shared<tt::tt_metal::Tensor>(ttnn::multiply(*meta0->tensor, *meta1->tensor));
            break;
        case GGML_OP_SUB:
            ret = std::make_shared<tt::tt_metal::Tensor>(ttnn::subtract(*meta0->tensor, *meta1->tensor));
            break;
        case GGML_OP_DIV:
            ret = std::make_shared<tt::tt_metal::Tensor>(ttnn::divide(*meta0->tensor, *meta1->tensor));
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

static void ggml_backend_metalium_reshape(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst)
{
    GGML_UNUSED(ctx);
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);

    tt::tt_metal::Tensor& t = *((TensorWithMetadata*)dst->src[0]->extra)->tensor;
    TensorWithMetadata* dst_meta = (TensorWithMetadata*)dst->extra;

    std::vector<uint32_t> target_shape(GGML_MAX_DIMS, 1);
    for(int i = 0; i < GGML_MAX_DIMS; i++) {
        target_shape[i] = dst->ne[GGML_MAX_DIMS - i - 1];
    }

    if(dst->ne[0] % tt::constants::TILE_WIDTH != 0 || dst->ne[1] % tt::constants::TILE_HEIGHT != 0) {
        // This path is SLOW. Reshape on a tilized tensor only works when the last two dimensions are tile aligned
        tt::tt_metal::Tensor row_major_tensor = tt::tt_metal::untilize(t);
        tt::tt_metal::Tensor reshaped = row_major_tensor.reshape(target_shape);
        tt::tt_metal::Tensor ret = tt::tt_metal::tilize_with_zero_padding(reshaped);
        *dst_meta = {
            .tensor = std::make_shared<tt::tt_metal::Tensor>(std::move(ret)),
            .ggtype = dst->type,
            .bufctx = ((TensorWithMetadata*)dst->src[0]->extra)->bufctx
        };
        return;
    }


    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(t.reshape(target_shape)),
        .ggtype = dst->type,
        .bufctx = ((TensorWithMetadata*)dst->src[0]->extra)->bufctx
    };

}

static void ggml_backend_metalium_transpose(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst)
{
    GGML_UNUSED(ctx);
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);

    tt::tt_metal::Tensor& t = *((TensorWithMetadata*)dst->src[0]->extra)->tensor;
    TensorWithMetadata* dst_meta = (TensorWithMetadata*)dst->extra;

    // std::cout << "GGML wants reshape to: " << dst->ne[0] << " " << dst->ne[1] << " " << dst->ne[2] << " " << dst->ne[3] << std::endl;

    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(tt::tt_metal::transpose(t, -2, -1)),
        .ggtype = dst->type,
        .bufctx = ((TensorWithMetadata*)dst->src[0]->extra)->bufctx
    };
    // std::cout << "TT wants reshape to: " << dst_meta->tensor->shape() << std::endl;
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

    if(size != ggml_nbytes(tensor)) {
        fprintf(stderr, "Warning: Does not supprt writing to segmented tensor\n");
        return;
    }

    // std::cout << "Writing to tensor with address: " << tensor->data << std::endl;

    tt::ARCH processor_class = bufctx->device->arch();
    // only grayskull is supported for now.
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

    tt::tt_metal::Tensor t(std::move(storage), tt::tt_metal::Shape(shape)
        , tt::tt_metal::DataType::BFLOAT16, tt::tt_metal::Layout::ROW_MAJOR);

    // I think we can allow this.. right?
    // GGML_ASSERT(!bufctx->tensors.contains(offset));

    // TODO: Make sure this is the correct tilize we want to use
    t = tt::tt_metal::tilize_with_zero_padding(t.to(bufctx->device));
    tt::tt_metal::DataType final_type = ggml2tt_type(ggtype, processor_class);
    if(final_type != t.dtype()) {
        t = typecast(t, final_type);
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

    TensorWithMetadata * meta = (TensorWithMetadata *)tensor->extra;
    GGML_ASSERT(meta->tensor != NULL);
    GGML_ASSERT(meta->tensor->storage_type() == tt::tt_metal::StorageType::DEVICE || meta->tensor->storage_type() == tt::tt_metal::StorageType::MULTI_DEVICE);

    ggml_backend_metalium_buffer_context * ctx = (ggml_backend_metalium_buffer_context *)buffer->context;

    ggml_type dst_ggtype = tensor->type;
    tt::tt_metal::CommandQueue& queue = ctx->device->command_queue(0);

    // some sanity checks, Could remove them once TTNN is more stable
    auto shape = meta->tensor->shape();
    // std::cout << "GGML thinks shape: " << tensor->ne[0] << " " << tensor->ne[1] << " " << tensor->ne[2] << " " << tensor->ne[3] << std::endl;
    // std::cout << "TTNN thinks shape: " << shape << std::endl;
    GGML_ASSERT(ggml_tt_tensors_shape_equal(tensor, *meta->tensor));
    tt::tt_metal::Tensor* t = meta->tensor.get();
    tt::tt_metal::Tensor holder;
    if(t->dtype() != tt::tt_metal::DataType::BFLOAT16 || t->dtype() != tt::tt_metal::DataType::FLOAT32) {
        holder = tt::tt_metal::typecast(*t, tt::tt_metal::DataType::BFLOAT16);
        t = &holder;
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
    auto * ptr = bufctx->address_tensor_map.find((uintptr_t)tensor->data);
    if(ptr != nullptr) {
        GGML_ASSERT(ptr->ggtype == tensor->type || ptr->ggtype == (ggml_type)-1);
        for(int i = 0; ptr->tensor != nullptr && i < GGML_MAX_DIMS; i++) {
            GGML_ASSERT(ptr->tensor->shape()[i] == tensor->ne[GGML_MAX_DIMS - i - 1]);
        }
        tensor->extra = ptr;
        return;
    }

    bufctx->metadata_to_free.push_back(std::make_unique<TensorWithMetadata>(TensorWithMetadata{
        .tensor = nullptr,
        .ggtype = (ggml_type)-1,
        .bufctx = bufctx
    }));
    tensor->extra = bufctx->metadata_to_free.back().get();
    // std::cout << "Creating tensor with address: " << tensor->data << ", shape = " << tensor->ne[0] << " " << tensor->ne[1] << " " << tensor->ne[2] << " " << tensor->ne[3] << ", name " << tensor->name << std::endl;
    bufctx->address_tensor_map.insert((uintptr_t)tensor->data, (TensorWithMetadata *)tensor->extra);
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

    tt::tt_metal::Tensor ret = tt::tt_metal::zeros_like(src_tensor);
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

        .metadata_to_free = {},
        .address_tensor_map = {},
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

    if(!g_device_map.contains(device)) {
        ggml_backend_metalium_init();
        GGML_ASSERT(g_device_map.contains(device));
    }

    if(buffer_type_map.contains(device)) {
        return &buffer_type_map[device];
    }

    buffer_type_map[device] = {
        /* .iface    = */ ggml_backend_metalium_buffer_type_interface,
        /* .context  = */ new ggml_backend_metalium_buffer_type_context{
            /* .device = */ g_device_map[device],
            /* .name   = */ "Metalium " + std::to_string(device),
        },
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
            case GGML_OP_VIEW:
                ggml_backend_metalium_cpy(ctx, node);
                break;

            case GGML_OP_RESHAPE:
                ggml_backend_metalium_reshape(ctx, node);
                break;

            case GGML_OP_TRANSPOSE:
                ggml_backend_metalium_transpose(ctx, node);
                break;

            case GGML_OP_NONE:
                break;

            default:
                fprintf(stderr, "%s: unsupported op %s\n", __func__, ggml_op_desc(node));
                GGML_ASSERT(false);
        }
        TensorWithMetadata* meta = (TensorWithMetadata*)node->extra;
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
    [[maybe_unused]] const struct ggml_tensor * src1 = op->src[1];
    ggml_backend_metalium_context * ctx = (ggml_backend_metalium_context *)backend->context;

    // The metalium backend has seperated internal data types from the GGML data types. We really only care about
    // what we can convert to and from. For now we only support F32, F16, and BF16. Quantized data types will be
    // supported in the future
    auto input_supported = [&](const struct ggml_tensor * tensor, bool req_contingous = true) {
        if(tensor == NULL ||
            !is_ggml_type_supported_by_metalium(tensor->type, ctx->device->arch()) ||
            !(!req_contingous || ggml_is_contiguous(tensor))) {
            return false;
        }
        // TTNN requires the tensor to be 4-byte aligned
        // TODO: Update this when we supported FP32
        return tensor->ne[0] % 2 == 0;
    };
    auto output_supported = [&](const struct ggml_tensor * tensor, bool req_contingous = true) {
        if(tensor == NULL ||
            !is_ggml_type_supported_by_metalium(tensor->type, ctx->device->arch()) ||
            !(!req_contingous || ggml_is_contiguous(tensor))) {
            return false;
        }
        // TTNN requires the tensor to be 4-byte aligned
        // TODO: Update this when we supported FP32
        if(!ggml_is_quantized(tensor->type)) {
            return tensor->ne[0] % 2 == 0;
        }
        return tensor->ne[0] % 4 == 0;
    };

    // std::cout << "Checking if op is supported: " << ggml_op_desc(op) << std::endl;
    // std::cout << "Output supported: " << output_supported(op, false) << std::endl;
    // if(op->op == GGML_OP_CONT || op->op == GGML_OP_VIEW || op->op == GGML_OP_CPY || op->op == GGML_OP_TRANSPOSE) {
    //     std::cout << "Output tensor details:\n"
    //         << "  data: " << op->data << "\n"
    //         << "  ne: " << op->ne[0] << " " << op->ne[1] << " " << op->ne[2] << " " << op->ne[3] << "\n"
    //         << "  nb: " << op->nb[0] << " " << op->nb[1] << " " << op->nb[2] << " " << op->nb[3] << "\n"
    //         << "  type: " << ggml_type_name(op->type) << "\n"
    //         << "  view_src: " << op->view_src << "\n"
    //         << "\n\n"
    //         << "Source 0 tensor details:\n"
    //         << "  data: " << src0->data << "\n"
    //         << "  ne: " << src0->ne[0] << " " << src0->ne[1] << " " << src0->ne[2] << " " << src0->ne[3] << "\n"
    //         << "  nb: " << src0->nb[0] << " " << src0->nb[1] << " " << src0->nb[2] << " " << src0->nb[3] << "\n"
    //         << "  type: " << ggml_type_name(src0->type) << "\n"
    //         << "  view_src: " << src0->view_src << "\n"
    //         << "\n";
    // }

    GGML_ASSERT(op != NULL);
    if(!output_supported(op, false)) {
        return false;
    }

    if(op->op == GGML_OP_CONT || op->op == GGML_OP_VIEW || op->op == GGML_OP_CPY || op->op == GGML_OP_TRANSPOSE) {
        return input_supported(src0, false);
    }

    if(!ggml_is_contiguous(op)) {
        return false;
    }

    switch (op->op) {
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_TANH: // Not accurate enough on Grayskull to pass unit tests
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_HARDSIGMOID:
                    return true;
                default:
                    return false;
            }
        case GGML_OP_LEAKY_RELU:
        case GGML_OP_NONE:
            return true;
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_DIV:
        case GGML_OP_MUL: // DITTO
            // DIV does not support broadcasting on TTNN
            return input_supported(src0) && input_supported(src1) &&
                (memcmp(src0->ne, src1->ne, sizeof(src0->ne)) == 0 || (numpy_broadcast_rule(src0, src1) && op->op != GGML_OP_DIV));
        case GGML_OP_MUL_MAT:
            return true;
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

    auto it = g_backend_map.find(device_id);
    if (it != g_backend_map.end()) {
        return it->second;
    }

    ggml_backend_metalium_context * ctx = new ggml_backend_metalium_context {
        /* device            = */ &ttnn::device::open_device(device_id),
        /* device_id         = */ device_id,
        /* name              = */ "Metalium " + std::to_string(device_id),
    };
    AutoFormat::SetDefaultDevice(ctx->device);


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