#include "ggml-backend-impl.h"
#include "ggml.h"
#include "ggml-metalium.h"

#include "host_api.hpp"
#include "tensor/host_buffer/functions.hpp"
#include "tensor/types.hpp"
#include "tt_dnn/op_library/auto_format.hpp"
#include <cstddef>
#include <cstdint>
#include <tt_eager/tensor/tensor.hpp>
#include <ttnn/core.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/device.hpp>
#include <tt_dnn/op_library/fully_connected/fully_connected_op.hpp>
#include <tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp>
#include <tt_eager/tensor/tensor.hpp>


#include <memory>

struct ggml_backend_metalium_context {
    ttnn::device::Device* device = nullptr;
    int device_id = 0;
    std::string name;
};


///////////////////////////////////////////////////////////////////////////////////////////////////////
// Backend internal state tracking because GGML API does not allow
///////////////////////////////////////////////////////////////////////////////////////////////////////

// maps device id to device
static std::map<int, ttnn::Device*> g_device_map;

///////////////////////////////////////////////////////////////////////////////////////////////////////
// Actual backend code
///////////////////////////////////////////////////////////////////////////////////////////////////////

static void ggml_backend_metalium_mul_mat(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    abort();
}

static void ggml_backend_metalium_out_prod(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    abort();
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
    return 4096; // assume the wosre, BFP16 on tile boundary
    GGML_UNUSED(buft);
}

// NOTE: I might need to add a metalium tensor wrapper to work around TT tensors have hardware-tagged data types
//       and GGML tensors does not specify the data type during tensor creation.
static size_t ggml_backend_metalium_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_backend_metalium_buffer_type_context * ctx = (ggml_backend_metalium_buffer_type_context *)buft->context;
    return ctx->device->num_dram_channels() * ctx->device->dram_size_per_channel();
}

GGML_CALL static size_t ggml_backend_metalium_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    // TODO: Make sure this is correct
    if(ggml_is_quantized(tensor->type)) {
        return ggml_nbytes(tensor);
    }
    intmax_t nelements = 1;
    for(int i = 0; i < 4; i++) {
        nelements *= i < 2 ? tensor->ne[i] / 32 + (tensor->ne[i] % 32 != 0) : tensor->ne[i];
    }
    return nelements * ggml_type_size(tensor->type);
    GGML_UNUSED(buft);
}

struct ggml_backend_metalium_buffer_context {

    size_t ggml_buffer_size_bytes = 0;
    std::string name;
    
    // These initializations are deferred due to GGML API limitations
    tt::tt_metal::Tensor tensor;
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
    ggml_backend_metalium_buffer_context * ctx = (ggml_backend_metalium_buffer_context *)buffer->context;
    ggml_type ggtype = tensor->type;
    GGML_ASSERT(ggtype == GGML_TYPE_F16);
    abort(); // not implemented yet

    GGML_ASSERT(offset == 0);
}

static struct ggml_backend_buffer_i ggml_backend_metalium_buffer_interface = {
    /* .get_name        = */ ggml_backend_metalium_buffer_get_name,
    /* .free_buffer     = */ ggml_backend_metalium_buffer_free_buffer,
    /* .get_base        = */ nullptr, //ggml_backend_metalium_buffer_get_base,
    /* .init_tensor     = */ nullptr, //ggml_backend_metalium_buffer_init_tensor,
    /* .set_tensor      = */ ggml_backend_metalium_buffer_set_tensor,
    /* .get_tensor      = */ nullptr, //ggml_backend_metalium_buffer_get_tensor,
    /* .cpy_tensor      = */ nullptr, //ggml_backend_metalium_buffer_cpy_tensor,
    /* .clear           = */ nullptr, //ggml_backend_metalium_buffer_clear,
    /* .reset           = */ nullptr,
};


GGML_CALL static ggml_backend_buffer_t
ggml_backend_metalium_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                           size_t size) {
    // ggml_backend_metalium_buffer_type_context * buft_ctx = (ggml_backend_metalium_buffer_type_context *)buft->context;
    ggml_backend_metalium_buffer_context* ctx = new ggml_backend_metalium_buffer_context;

    // real allocation is deferred until the first tensor is set because we don't know the underlying tensor type yet
    // TODO: Use a constructor
    ctx->ggml_buffer_size_bytes = size;
    ctx->name = ctx->name;
    return ggml_backend_buffer_init(buft, ggml_backend_metalium_buffer_interface, ctx, size);
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
    abort(); // nothing supported yet
    ggml_backend_metalium_context * ctx = (ggml_backend_metalium_context *)backend->context;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        switch (node->op) {
            case GGML_OP_MUL_MAT:
                ggml_backend_metalium_mul_mat(ctx, node);
                break;

            case GGML_OP_OUT_PROD:
                ggml_backend_metalium_out_prod(ctx, node);
                break;

            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
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
    const struct ggml_tensor * src1 = op->src[1];
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);
    return false;

    /*return (op->op == GGML_OP_MUL_MAT  && ggml_backend_blas_use_blas(op)) ||
           (op->op == GGML_OP_OUT_PROD && op->src[0]->type == GGML_TYPE_F32 &&
                                          op->src[1]->type == GGML_TYPE_F32 &&
                                          ggml_is_matrix(src0) &&
                                          ggml_is_matrix(src1) &&
                                          ggml_is_contiguous(src0) &&
                                          (ggml_is_contiguous(src1) || ggml_is_transposed(src1)));*/

    GGML_UNUSED(backend);
}

GGML_CALL static bool ggml_backend_metalium_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_host(buft);

    GGML_UNUSED(backend);
}

static struct ggml_backend_i metalium_backend_i = {
    /* .get_name                = */ ggml_backend_metalium_name,
    /* .free                    = */ ggml_backend_metalium_free,
    /* .get_default_buffer_type = */ ggml_backend_metalium_get_default_buffer_type,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
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
    g_device_map[0] = ctx->device;

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