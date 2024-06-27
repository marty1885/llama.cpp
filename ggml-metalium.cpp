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
#include "tt_dnn/op_library/composite/composite_ops.hpp"
#include "tt_dnn/op_library/tilize/tilize_op.hpp"
#include "tt_dnn/op_library/untilize/untilize_op.hpp"
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <tt_eager/tensor/tensor.hpp>
#include <ttnn/core.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <tt_eager/tt_dnn/op_library/transpose/transpose_op.hpp>
#include <ttnn/device.hpp>
#include <tt_dnn/op_library/fully_connected/fully_connected_op.hpp>
#include <tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp>
#include <tt_eager/tensor/tensor.hpp>


#include <memory>
#include <type_traits>

struct ggml_backend_metalium_context {
    ttnn::device::Device* device = nullptr;
    int device_id = 0;
    std::string name;
};

struct TensorWithMetadata
{
    std::shared_ptr<tt::tt_metal::Tensor> tensor;
    ggml_type ggtype = (ggml_type)-1;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////
// Backend internal state tracking because GGML API does not allow
///////////////////////////////////////////////////////////////////////////////////////////////////////

// maps device id to device
static std::map<int, ttnn::Device*> g_device_map;

///////////////////////////////////////////////////////////////////////////////////////////////////////
// Actual backend code
///////////////////////////////////////////////////////////////////////////////////////////////////////

static tt::tt_metal::DataType ggml2tt_type(ggml_type ggtype, tt::ARCH arch) {
    if(arch == tt::ARCH::GRAYSKULL) {
        switch(ggtype) {
            case GGML_TYPE_F32:
            case GGML_TYPE_F16:
            case GGML_TYPE_BF16:
                return tt::tt_metal::DataType::BFLOAT16;
            default:
                GGML_ASSERT(false && "Unsupported data type");
        }
    }
    else {
        GGML_ASSERT(false && "Unsupported Tenstorrent card architecture");
    }
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
        memcpy(vec.data(), src, size * sizeof(Src));
    }
    else {
        for(size_t i = 0; i < size; i++) {
            dst_adaptor(vec[i], src_adaptor(src[i]));
        }
    }
    auto owned = tt::tt_metal::owned_buffer::create(std::move(vec));
    return OwnedStorage(std::move(owned));

}

template <typename SrcType, typename DstType>
void tensor2ggml(const tt::tt_metal::Tensor& tensor, DstType* dst, tt::tt_metal::CommandQueue& queue) {
    // Converts TT tensors to GGML types
    // TODO: Support reading quantized data
    static_assert(std::is_same_v<SrcType, bfloat16>, "Unsupported data type conversion");
    ttnn::Shape shape = tensor.shape();

    tt::tt_metal::Tensor row_major_tensor = tt::tt_metal::untilize(tensor);
    GGML_ASSERT(row_major_tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR);

    std::vector<bfloat16> buf(shape.volume()); // .volume() returns the underlying volume of the tensor not the logical one (TT enforces 32x32 tiles)
    tt::tt_metal::memcpy(queue, buf.data(), row_major_tensor);
    tt::tt_metal::Finish(queue);

    ttnn::Shape tt_underlying_shape = row_major_tensor.shape().with_tile_padding();
    std::array<size_t, 4> stride = {1, tt_underlying_shape[3], tt_underlying_shape[3] * tt_underlying_shape[2], tt_underlying_shape[3] * tt_underlying_shape[2] * tt_underlying_shape[1]};

    auto dst_adaptor = [](DstType& dst, float val) {
        if constexpr(std::is_same_v<DstType, float>) {
            dst = val;
        }
        else if constexpr(std::is_same_v<DstType, ggml_fp16_t>) {
            dst = ggml_fp32_to_fp16(val);
        }
        else if constexpr(std::is_same_v<DstType, ggml_bf16_t>) {
            dst = ggml_fp32_to_bf16(val);
        }
        else {
            abort();
        }
        return;
    };

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
    static_assert(GGML_MAX_DIMS == 4, "Looping depth is hardcoded to 4");
    size_t idx = 0;
    for(size_t w = 0; w < shape[0]; w++) {
        for(size_t z = 0; z < shape[1]; z++) {
            for(size_t x = 0; x < shape[3]; x++) { // FIXME: Don't know why but the shape is reversed...????
                for(size_t y = 0; y < shape[2]; y++) {
                    const size_t src_idx = x * stride[0] + y * stride[1] + z * stride[2] + w * stride[3];
                    dst_adaptor(dst[idx++], src_adaptor(buf[src_idx]));
                }
            }
        }
    }
}

static void ggml_backend_metalium_mul_mat(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst) {
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

    TensorWithMetadata* am = (TensorWithMetadata*)src0->extra;
    TensorWithMetadata* bm = (TensorWithMetadata*)src1->extra;
    TensorWithMetadata* cm = (TensorWithMetadata*)dst->extra;

    GGML_ASSERT(am != NULL);
    GGML_ASSERT(bm != NULL);
    GGML_ASSERT(cm != NULL);
    GGML_ASSERT(am->tensor != NULL);
    GGML_ASSERT(bm->tensor != NULL);

    tt::tt_metal::Tensor& a = *am->tensor;
    tt::tt_metal::Tensor& b = *bm->tensor;

    GGML_ASSERT(a.storage_type() == tt::tt_metal::StorageType::DEVICE || a.storage_type() == tt::tt_metal::StorageType::MULTI_DEVICE);
    GGML_ASSERT(b.storage_type() == tt::tt_metal::StorageType::DEVICE || b.storage_type() == tt::tt_metal::StorageType::MULTI_DEVICE);

    auto aT = tt::tt_metal::transpose(a, -2, -1);
#if !defined(NDEBUG) || 1
    // TODO: Remove this in the future. TTNN has buggy transpose implementation
    std::cout << "a.shape: " << a.shape() << " aT.shape: " << aT.shape() << std::endl;
    GGML_ASSERT(aT.shape()[0] == a.shape()[0]);
    GGML_ASSERT(aT.shape()[1] == a.shape()[1]);
    GGML_ASSERT(aT.shape()[3] == a.shape()[2]);
    GGML_ASSERT(aT.shape()[2] == a.shape()[3]);
#endif

    std::cout << "a.shape: " << a.shape() << " b.shape: " << b.shape() << std::endl;

    // TODO: Ask TT to support multiplication of pre-transposed tensors. Calling transpose here is inefficient
    // https://github.com/tenstorrent/tt-metal/issues/9709
    cm->tensor = std::make_shared<tt::tt_metal::Tensor>(tt::tt_metal::fully_connected(b, aT));
    GGML_UNUSED(ctx);
}

static void ggml_backend_metalium_cpy(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    const struct ggml_tensor * src0 = dst->src[0];
    TensorWithMetadata* meta = (TensorWithMetadata*)src0->extra;
    GGML_ASSERT(meta != NULL);
    GGML_ASSERT(meta->tensor != NULL);

    TensorWithMetadata* dst_meta = (TensorWithMetadata*)dst->extra;
    GGML_ASSERT(dst_meta != NULL);

    tt::tt_metal::Tensor ret = tt::tt_metal::zeros_like(*meta->tensor);
    ret.deepcopy(*meta->tensor);
    GGML_ASSERT(ret.storage_type() == tt::tt_metal::StorageType::DEVICE || ret.storage_type() == tt::tt_metal::StorageType::MULTI_DEVICE);
    dst_meta->tensor = std::make_shared<tt::tt_metal::Tensor>(std::move(ret));
    dst_meta->ggtype = dst->type;
}

static bool ggml_backend_metalium_activations(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst, ggml_unary_op op) {
    const struct ggml_tensor * src0 = dst->src[0];
    TensorWithMetadata* meta = (TensorWithMetadata*)src0->extra;
    GGML_ASSERT(meta != NULL);
    GGML_ASSERT(meta->tensor != NULL);
    GGML_UNUSED(ctx);

    TensorWithMetadata* dst_meta = (TensorWithMetadata*)dst->extra;
    GGML_ASSERT(dst_meta != NULL);

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
        // case GGML_UNARY_OP_TANH:
        //     ret = tt::tt_metal::tanh(*meta->tensor);
        //     break;
        // ELU needs an additional parameter. Find where in GGML this is stored
        // case GGML_UNARY_OP_ELU:
        //     ret = tt::tt_metal::elu(*meta->tensor);
        //     break;
        case GGML_UNARY_OP_RELU:
            ret = tt::tt_metal::relu(*meta->tensor);
            break;
        // Not accurate enough to pass unit tests
        // case GGML_UNARY_OP_SIGMOID:
        //     ret = tt::tt_metal::sigmoid(*meta->tensor);
        //     break;
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
    GGML_ASSERT(ret.storage_type() == tt::tt_metal::StorageType::DEVICE || ret.storage_type() == tt::tt_metal::StorageType::MULTI_DEVICE);
    dst_meta->tensor = std::make_shared<tt::tt_metal::Tensor>(std::move(ret));
    dst_meta->ggtype = dst->type;
    return true;
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

struct ggml_backend_metalium_buffer_context {

    size_t ggml_buffer_size_bytes = 0;
    std::string name;
    ttnn::device::Device* device = nullptr;

    // Tracking our own allocations because Metalium limitations and GGML assuming them
    std::vector<std::unique_ptr<TensorWithMetadata>> metadata_to_free;
};

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
    // Must be setting the entire tensor at once
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));
    GGML_ASSERT(tensor->extra != NULL);
    GGML_ASSERT((size_t)ggml_nelements(tensor) == size / ggml_type_size(tensor->type) && "Must write the entire tensor at once");

    ggml_backend_metalium_buffer_context * bufctx = (ggml_backend_metalium_buffer_context *)buffer->context;
    ggml_type ggtype = tensor->type;
    TensorWithMetadata * meta = (TensorWithMetadata *)tensor->extra;

    tt::ARCH processor_class = bufctx->device->arch();
    // only grayskull is supported for now.
    GGML_ASSERT(processor_class == tt::ARCH::GRAYSKULL);

    // TODO: See if we can use BorrowedStorage to avoid copying the data
    OwnedStorage storage;
    switch (ggtype) {
        case GGML_TYPE_F32:
            // For now we cast F32 to BF16. Need a scalable way to handle this as WORMHOLD_B0 have native support for F32
            // TODO: Might want to consider disabling F32 support for Grayskull in the future
            storage = data2owned_storage<float, bfloat16>((const float*)data, size / sizeof(float));
            break;
        case GGML_TYPE_BF16:
            storage = data2owned_storage<ggml_bf16_t, bfloat16>((const ggml_bf16_t*)data, size / sizeof(bfloat16));
            break;
        case GGML_TYPE_F16:
            // TT hardware claims to support FP16 but the API does not expose it. For now we use BF16 as it is close enough
            storage = data2owned_storage<ggml_fp16_t, bfloat16>((const ggml_fp16_t*)data, size / sizeof(ggml_fp16_t));
            break;
        default:
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
    GGML_ASSERT(t.dtype() == ggml2tt_type(ggtype, processor_class));
    *meta = TensorWithMetadata {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(std::move(t)),
        .ggtype = ggtype,
    };
}

static void ggml_backend_metalium_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                const ggml_tensor *tensor,
                                                void *data, size_t offset,
                                                size_t size)
{
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
    GGML_ASSERT(shape[0] == tensor->ne[3] && "Shape mismatch between GGML and TTNN tensor on dimension 0");
    GGML_ASSERT(shape[1] == tensor->ne[2] && "Shape mismatch between GGML and TTNN tensor on dimension 1");
    GGML_ASSERT(shape[2] == tensor->ne[1] && "Shape mismatch between GGML and TTNN tensor on dimension 2");
    GGML_ASSERT(shape[3] == tensor->ne[0] && "Shape mismatch between GGML and TTNN tensor on dimension 3");

    // TODO: Proper handling of data types
    if(dst_ggtype == GGML_TYPE_F32) {
        if(meta->tensor->dtype() == tt::tt_metal::DataType::BFLOAT16) {
            tensor2ggml<bfloat16, float>(*meta->tensor, (float*)data, queue);
        }
        else {
            GGML_ASSERT(false && "Unsupported data type held in TT kernel");
        }
    }
    else if(dst_ggtype == GGML_TYPE_BF16) {
        if(meta->tensor->dtype() == tt::tt_metal::DataType::BFLOAT16) {
            tensor2ggml<bfloat16, ggml_bf16_t>(*meta->tensor, (ggml_bf16_t*)data, queue);
        }
        else {
            GGML_ASSERT(false && "Unsupported data type held in TT kernel");
        }
    }
    else if(dst_ggtype == GGML_TYPE_F16) {
        if(meta->tensor->dtype() == tt::tt_metal::DataType::BFLOAT16) {
            tensor2ggml<bfloat16, ggml_fp16_t>(*meta->tensor, (ggml_fp16_t*)data, queue);
        }
        else {
            GGML_ASSERT(false && "Unsupported data type held in TT kernel");
        }
    }
    else {
        GGML_ASSERT(false && "Unsupported destination data type");
    }
}

static void * ggml_backend_metalium_buffer_get_base(ggml_backend_buffer_t buffer) {
    // Not using this. Metalium's allication model is not compatible with GGML's allocator
    GGML_UNUSED(buffer);
    return (void*)0xdeadbeef;
}

GGML_CALL static void
ggml_backend_sycl_buffer_init_tensor(ggml_backend_buffer_t buffer,
                                     ggml_tensor *tensor)
{
    ggml_backend_metalium_buffer_context * bufctx = (ggml_backend_metalium_buffer_context *)buffer->context;
    bufctx->metadata_to_free.push_back(std::make_unique<TensorWithMetadata>());
    tensor->extra = bufctx->metadata_to_free.back().get();
    GGML_UNUSED(buffer);
}

static struct ggml_backend_buffer_i ggml_backend_metalium_buffer_interface = {
    /* .get_name        = */ ggml_backend_metalium_buffer_get_name,
    /* .free_buffer     = */ ggml_backend_metalium_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_metalium_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_sycl_buffer_init_tensor,
    /* .set_tensor      = */ ggml_backend_metalium_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_metalium_buffer_get_tensor,
    /* .cpy_tensor      = */ nullptr, //ggml_backend_metalium_buffer_cpy_tensor,
    /* .clear           = */ nullptr, //ggml_backend_metalium_buffer_clear,
    /* .reset           = */ nullptr,
};


GGML_CALL static ggml_backend_buffer_t
ggml_backend_metalium_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                           size_t size) {
    ggml_backend_metalium_buffer_type_context * buft_ctx = (ggml_backend_metalium_buffer_type_context *)buft->context;
    ggml_backend_metalium_buffer_context* ctx = new ggml_backend_metalium_buffer_context;

    // real allocation is deferred until the first tensor is set because we don't know the underlying tensor type yet
    // TODO: Use a constructor
    ctx->ggml_buffer_size_bytes = size;
    ctx->name = ctx->name;
    ctx->device = buft_ctx->device;
    // FIXME: GGML unit tests fails if I don't add some additional memory to the buffer beyond the requested size
    return ggml_backend_buffer_init(buft, ggml_backend_metalium_buffer_interface, ctx, size + 4096 * 1024);
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

    GGML_ASSERT(g_device_map.contains(device));

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
    GGML_UNUSED(backend);
}

GGML_CALL static enum ggml_status ggml_backend_metalium_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    ggml_backend_metalium_context * ctx = (ggml_backend_metalium_context *)backend->context;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];


        switch (node->op) {
            case GGML_OP_UNARY: {
                ggml_unary_op unary_op = ggml_get_unary_op(node);
                bool ok = false;
                switch (unary_op) {
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_NEG:
                //case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_RELU:
                //case GGML_UNARY_OP_SIGMOID:
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
            case GGML_OP_MUL_MAT:
                ggml_backend_metalium_mul_mat(ctx, node);
                break;
            
            case GGML_OP_CPY:
                ggml_backend_metalium_cpy(ctx, node);
                break;
            
            case GGML_OP_NONE:
                break;

            default:
                fprintf(stderr, "%s: unsupported op %s\n", __func__, ggml_op_desc(node));
                GGML_ASSERT(false);
        }
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
    auto input_supported = [&](const struct ggml_tensor * tensor) {
        if (!(tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_F16 || tensor->type == GGML_TYPE_BF16)) {
            return false;
        }
        // TTNN requires the tensor to be 4-byte aligned
        return tensor->ne[0] * tttype_size(ggml2tt_type(tensor->type, ctx->device->arch())) % 4 == 0;
    };
    auto output_supported = [&](const struct ggml_tensor * tensor) {
        if (!(tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_F16 || tensor->type == GGML_TYPE_BF16)) {
            return false;
        }
        // TTNN requires the tensor to be 4-byte aligned
        return tensor->ne[0] * tttype_size(ggml2tt_type(tensor->type, ctx->device->arch())) % 4 == 0;
    };
    
    GGML_ASSERT(op != NULL);
    if(!output_supported(op)) {
        return false;
    }

    switch (op->op) {
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_NEG:
                //case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_RELU:
                //case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_HARDSIGMOID:
                    return true;
                default:
                    return false;
            }
        case GGML_OP_NONE:
            return  true;
        // FIXME: This crash for most shapes due to a bug in TTNN transpose implementation. Unmask this
        // when the bug is fixed
        // case GGML_OP_MUL_MAT:
        //     return op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32;
        case GGML_OP_CPY:
            return input_supported(src0) && output_supported(op);
        default:
            return false;
    }
}

GGML_CALL static bool ggml_backend_metalium_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_host(buft);

    GGML_UNUSED(backend);
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
    return backend;
}

GGML_CALL bool ggml_backend_is_metalium(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_metalium_guid());
}


GGML_CALL ggml_backend_t ggml_backend_reg_metalium_init(const char * params, void * user_data)
{
    GGML_UNUSED(params);
    GGML_UNUSED(user_data);
    return ggml_backend_metalium_init();
}