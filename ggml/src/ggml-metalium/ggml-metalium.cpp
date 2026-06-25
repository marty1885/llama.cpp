#include "fmt/base.h"
#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"
#include "ggml-metalium.h"

#include "ggml-metalium-internal.hpp"
#include "compiler.hpp"

#include "hostdevcommon/common_values.hpp"
#include "tt-metalium/bfloat16.hpp"
#include "tt-metalium/host_buffer.hpp"
#include "tt-metalium/memory_pin.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/moreh/moreh_group_norm/moreh_group_norm.hpp"
#include "ttnn/operations/trace.hpp"
#include "ttnn/tensor/layout/layout.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "types/arch.hpp"
#include "umd/device/types/arch.hpp"
#include "umd/device/types/cluster_descriptor_types.hpp"
#include <tt-logger/tt-logger.hpp>
#include <string.h>
#include <sys/types.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <chrono>
#include <map>
#include <optional>
#include <unordered_set>
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
#include <ttnn/operations/creation/creation.hpp>
#include <ttnn/operations/eltwise/unary/unary_composite.hpp>
#include <ttnn/operations/data_movement/transpose/transpose.hpp>
#include <ttnn/operations/data_movement/permute/permute.hpp>
#include <ttnn/operations/data_movement/repeat/repeat.hpp>
#include <ttnn/operations/data_movement/concat/concat.hpp>
#include <ttnn/operations/data_movement/copy/copy.hpp>
#include <ttnn/operations/data_movement/clone/clone.hpp>
#include <ttnn/operations/copy/typecast/typecast.hpp>
#include <ttnn/operations/normalization/softmax/softmax.hpp>
#include <tt-metalium/experimental/kernel_cache.hpp>
#include <ttnn/operations/data_movement/reshape_view/reshape.hpp>
#include <ttnn/operations/reduction/generic/generic_reductions.hpp>
#include <ttnn/cpp/ttnn/operations/data_movement/gather/tosa/gather_tosa.hpp>
#include <ttnn/cpp/ttnn/operations/data_movement/scatter/tosa_scatter.hpp>
#include <ttnn/cpp/ttnn/operations/transformer/sdpa_decode/sdpa_decode.hpp>
#include <ttnn/cpp/ttnn/operations/transformer/sdpa/sdpa.hpp>
#include <ttnn/cpp/ttnn/operations/data_movement/slice/slice.hpp>
#include <ttnn/operations/experimental/slice_write/slice_write.hpp>


#include <memory>
#include <type_traits>
#include <vector>

#include "rope.hpp"
#include "soft_max.hpp"
#include "wkv7.hpp"
#include "slice_write_folded.hpp"

extern void metalium_register_all_kernel();

struct ggml_backend_metalium_device_context {
    std::shared_ptr<ttnn::MeshDevice> device = nullptr;
    int device_id = -1;
    std::string name;
    std::string description;
    std::unique_ptr<MetaliumGraphCompiler> compiler;
};

struct ggml_backend_metalium_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

struct ggml_backend_metalium_buffer_context {

    size_t ggml_buffer_size_bytes = 0;
    std::string name;
    std::shared_ptr<ttnn::MeshDevice> device = nullptr;
    size_t base_offset = 0;

    // Tracking our own allocations because Metalium limitations and GGML assuming them
    std::vector<std::unique_ptr<ggml_tensor_extra_metalium>> metadata_to_free;
};

static bool ggml_tt_tensors_shape_equal(const ggml_tensor* ggtensor, const tt::tt_metal::Tensor& ttensor)
{
    const ttnn::Shape& shape = ttensor.logical_shape();
    for(size_t i = 0; i < std::min<size_t>(GGML_MAX_DIMS, shape.size()); i++) {
        if(ggtensor->ne[GGML_MAX_DIMS - i - 1] != shape[i]) {
            return false;
        }
    }

    if(shape.size() > GGML_MAX_DIMS) {
        for(size_t i = GGML_MAX_DIMS; i < shape.size(); i++) {
            if(shape[i] != 1) {
                return false;
            }
        }
    }
    else if(shape.size() < GGML_MAX_DIMS) {
        for(size_t i = shape.size(); i < GGML_MAX_DIMS; i++) {
            if(ggtensor->ne[GGML_MAX_DIMS - i - 1] != 1) {
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

ttnn::DeviceComputeKernelConfig make_compute_kernel_config(ttnn::IDevice* device)
{
    ttnn::DeviceComputeKernelConfig cfg;
    if (device->arch() == tt::ARCH::WORMHOLE_B0 || device->arch() == tt::ARCH::BLACKHOLE) {
        cfg = ttnn::WormholeComputeKernelConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .math_approx_mode = false,
            .fp32_dest_acc_en = false,
            .packer_l1_acc = false
        };
    }
    else {
        fmt::println(stderr,"Unsupported device arch {} in make_compute_kernel_config", device->arch());
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
    bool disable_program_cache = false;     // Disables the program cache
    bool experimental_ops = false;          // Enable experimental ops that is known to cause trouble
    bool disable_graph_compiler = false;    // Skip the graph compiler entirely; fall back to native per-op dispatch
    bool print_local_timing = false;        // Per-op host+device timing (Finish after each op). Separate from Tracy.
    bool print_trace_mem = false;           // DRAM attribution under tracing: pinned intermediates vs IO vs trace cmd buffers
};

static const ggml_backend_metalium_debug_flags g_debug_flags = []() {
    auto parse_env = [](const char* env) -> bool {
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
        .print_rejected_ops = parse_env("GGML_METALIUM_PRINT_REJECTED_OPS"),
        .print_view = parse_env("GGML_METALIUM_PRINT_VIEW"),
        .disable_program_cache = parse_env("GGML_METALIUM_DISABLE_PROGRAM_CACHE"),
        .experimental_ops = parse_env("GGML_METALIUM_EXPERIMENTAL_OPS"),
        .disable_graph_compiler = parse_env("GGML_METALIUM_DISABLE_GRAPH_COMPILER"),
        .print_local_timing = parse_env("GGML_METALIUM_PRINT_LOCAL_TIMING"),
        .print_trace_mem = parse_env("GGML_METALIUM_TRACE_MEM")
    };
}();

///////////////////////////////////////////////////////////////////////////////////////////////////////
// Backend internal state tracking because GGML API does not allow
///////////////////////////////////////////////////////////////////////////////////////////////////////

// Maintain all base addresses are unique
// TODO: Do we still need this since we already removed the virtual address mapping hack?
static size_t g_metalium_base_offset = 0;

// Unlike g_debug_flags it is MUTABLE so a test can set or unset it.
// on via ggml_backend_metalium_set_tracing() BEFORE the device opens (the device must reserve a
// trace-capable region at open time). Default OFF because this feature is unstable and in development
static bool g_metalium_trace_enabled = []() {
    const char* v = std::getenv("GGML_METALIUM_TRACE");
    if(v == nullptr) return false;
    std::string s(v);
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return s != "0" && s != "false" && s != "no" && s != "off";
}();

extern "C" {
void ggml_backend_metalium_set_tracing(bool enable) {
    g_metalium_trace_enabled = enable;
}
bool ggml_backend_metalium_tracing_enabled(void) {
    return g_metalium_trace_enabled;
}
}

// [elease all captured traces. Defined further down where the state struct is complete;
// forward-declared here because ggml_backend_metalium_free (above) calls it.
static void metalium_trace_release_all(ttnn::MeshDevice* mesh);

///////////////////////////////////////////////////////////////////////////////////////////////////////
// Actual backend code
///////////////////////////////////////////////////////////////////////////////////////////////////////

static tt::tt_metal::DataType ggml2tt_type_internal(ggml_type ggtype, tt::ARCH arch) {
    // This table is consulted to map GGML types to TT types dueing tensor creation
    if(arch == tt::ARCH::WORMHOLE_B0) {
        // NOTE: size is deduced from the entries below (NOT GGML_TYPE_COUNT) on purpose. If GGML
        // adds new types, they fall past the end of this table and the guard below maps them to
        // INVALID - the backend keeps working and simply rejects the unknown type instead of
        // silently treating it as BFLOAT16 (DataType == 0) or failing to build.
        static constexpr tt::tt_metal::DataType table[] = {
            /*GGML_TYPE_F32        = */ tt::tt_metal::DataType::BFLOAT16,
            /*GGML_TYPE_F16        = */ tt::tt_metal::DataType::BFLOAT16,
            /*GGML_TYPE_Q4_0       = */ tt::tt_metal::DataType::BFLOAT8_B,
            /*GGML_TYPE_Q4_1       = */ tt::tt_metal::DataType::BFLOAT8_B,
            tt::tt_metal::DataType::INVALID,
            tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_Q5_0       = */ tt::tt_metal::DataType::BFLOAT8_B,
            /*GGML_TYPE_Q5_1       = */ tt::tt_metal::DataType::BFLOAT8_B,
            /*GGML_TYPE_Q8_0       = */ tt::tt_metal::DataType::BFLOAT8_B,
            /*GGML_TYPE_Q8_1       = */ tt::tt_metal::DataType::BFLOAT8_B,
            /*GGML_TYPE_Q2_K       = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_Q3_K       = */ tt::tt_metal::DataType::BFLOAT4_B,
            /*GGML_TYPE_Q4_K       = */ tt::tt_metal::DataType::BFLOAT4_B,
            /*GGML_TYPE_Q5_K       = */ tt::tt_metal::DataType::BFLOAT8_B,
            /*GGML_TYPE_Q6_K       = */ tt::tt_metal::DataType::BFLOAT8_B,
            /*GGML_TYPE_Q8_K       = */ tt::tt_metal::DataType::BFLOAT8_B,
            /*GGML_TYPE_IQ2_XXS    = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_IQ2_XS     = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_IQ3_XXS    = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_IQ1_S      = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_IQ4_NL     = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_IQ3_S      = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_IQ2_S      = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_IQ4_XS     = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_I8         = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_I16        = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_I32        = */ tt::tt_metal::DataType::UINT32, // Yeah not ideal. but don't have support for tilizing int32 on device
            /*GGML_TYPE_I64        = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_F64        = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_IQ1_M      = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_BF16       = */ tt::tt_metal::DataType::BFLOAT16,
            /*GGML_TYPE_Q4_0_4_4   = */ tt::tt_metal::DataType::INVALID, // Support removed from GGML
            /*GGML_TYPE_Q4_0_4_8   = */ tt::tt_metal::DataType::INVALID, // Support removed from GGML
            /*GGML_TYPE_Q4_0_8_8   = */ tt::tt_metal::DataType::INVALID, // Support removed from GGML
            /*GGML_TYPE_TQ1_0      = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_TQ2_0      = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_IQ4_NL_4_4 = */ tt::tt_metal::DataType::INVALID, // Support removed from GGML
            /*GGML_TYPE_IQ4_NL_4_8 = */ tt::tt_metal::DataType::INVALID, // Support removed from GGML
            /*GGML_TYPE_IQ4_NL_8_8 = */ tt::tt_metal::DataType::INVALID, // Support removed from GGML
            /*GGML_TYPE_MXFP4      = */ tt::tt_metal::DataType::BFLOAT4_B,
            /*GGML_TYPE_NVFP4      = */ tt::tt_metal::DataType::INVALID,
            /*GGML_TYPE_Q1_0       = */ tt::tt_metal::DataType::INVALID,
        };
        // The table must never claim to map more types than GGML defines (catches a stale table
        // after a type is removed). It is allowed to be shorter than GGML_TYPE_COUNT - any type
        // not covered is handled by the OOB guard below.
        static_assert(std::size(table) <= GGML_TYPE_COUNT, "Type conversion table is out of sync with ggml_type");
        // Stop-gap for newly invented GGML types: anything beyond what the table covers is
        // treated as unsupported rather than read out of bounds (or mis-mapped to BFLOAT16).
        if((size_t)ggtype >= std::size(table)) {
            return tt::tt_metal::DataType::INVALID;
        }
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
        fmt::println(stderr, "Unsupported data type: {}", ggml_type_name(ggtype));
        GGML_ASSERT(false && "Unsupported data type");
    }
    return type;

}

static bool is_ggml_type_supported_by_metalium(ggml_type ggtype, tt::ARCH arch) {
    return ggml2tt_type_internal(ggtype, arch) != tt::tt_metal::DataType::INVALID;
}

template <typename SrcType, typename DstType>
static tt::tt_metal::HostBuffer host_data_to_tt_host_buffer(const SrcType* src, size_t size) {
    // Converts GGML floating point (FP32, FP16, BF16) to TT floating point (FP32, BF16)
    using Src = std::remove_cv_t<std::remove_reference_t<SrcType>>;
    using Dst = std::remove_cv_t<std::remove_reference_t<DstType>>;
    // Convert from  GGML types to TT types
    static_assert(std::is_same_v<Src, float> || std::is_same_v<Src, ggml_bf16_t> || std::is_same_v<Src, ggml_fp16_t> || std::is_same_v<Src, int>);
    static_assert(std::is_same_v<Dst, float> || std::is_same_v<Dst, bfloat16> || std::is_same_v<Dst, uint32_t>);

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
        else if constexpr(std::is_same_v<Src, int>) {
            return static_cast<float>(src);
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
        else if constexpr(std::is_same_v<Dst, int>) {
            dst = static_cast<int>(val);
        }
        else if constexpr(std::is_same_v<Dst, uint32_t>) {
            dst = static_cast<uint32_t>(val);
        }
        else {
            GGML_UNREACHABLE();
        }
    };

    // Optimization: avoid unnecessary initialization and copying like vec<float>(size) as it tanks performance
    Dst* vec = new Dst[size];
    // special case if both GGML and TT types have the same underlying type (e.g. both FP32 or BF16)
    if constexpr(std::is_same_v<Src, Dst>
        || (std::is_same_v<Src, ggml_bf16_t> && std::is_same_v<Dst, bfloat16>)
        || (std::is_same_v<Src, int> && std::is_same_v<Dst, uint32_t>)) { // we really don't care about signedness here
        static_assert(sizeof(Src) == sizeof(Dst), "Src and Dst must have the same size");
        // Make GCC shut up about writing into a class like it's flat memory
        memcpy((void*)vec, src, size * sizeof(Src));
    }
    // special case for BFP16 (much faster then TTNN's implementation)
    else if constexpr(std::is_same_v<Src, float> && std::is_same_v<Dst, bfloat16>) {
        const auto* trait = ggml_get_type_traits_cpu(GGML_TYPE_BF16);
        assert(trait != nullptr);
        trait->from_float(src, vec, size);
    }
    else {
        for(size_t i = 0; i < size; i++) {
            dst_adaptor(vec[i], src_adaptor(src[i]));
        }
    }

    int* refcount = new int(0);
    tt::tt_metal::MemoryPin pin(
        [refcount]() mutable {
            (*refcount)++;
        },
        [refcount, vec]() mutable {
            assert(refcount != nullptr);
            (*refcount)--;
            if(*refcount == 0) {
                delete refcount;
                delete [] vec;
                refcount = nullptr;
            }
        }
    );
    auto storage = tt::tt_metal::HostBuffer(ttsl::Span<DstType>(vec, size), std::move(pin));

    return storage;
}

template <typename DstType>
static tt::tt_metal::HostBuffer quantized_ggml_data_to_tt_host_buffer(const void* src, const ggml_tensor* tensor) {
    const ggml_type_traits* trait = ggml_get_type_traits(tensor->type);
    const size_t size = ggml_nelements(tensor);
    GGML_ASSERT(trait->to_float != NULL);

    if constexpr(std::is_same_v<DstType, float>) {
        std::shared_ptr<float[]> vec(new float[size]);
        trait->to_float(src, vec.get(), size);
        int* refcount = new int(0);
        float* vec_ptr = vec.get();
        tt::tt_metal::MemoryPin pin(
            [refcount]() mutable {
                (*refcount)++;
            },
            [refcount, vec=std::move(vec)]() mutable {
                assert(refcount != nullptr);
                (*refcount)--;
                if(*refcount == 0) {
                    delete refcount;
                    vec.reset();
                }
            }
        );
        return tt::tt_metal::HostBuffer(ttsl::Span<float>(vec_ptr, size), std::move(pin));
    }
    else if constexpr(std::is_same_v<DstType, bfloat16>) {
        std::shared_ptr<bfloat16[]> vec(new bfloat16[size]);
        size_t block_size_in_bytes = ggml_type_size(tensor->type);
        size_t block_size_in_elements = ggml_blck_size(tensor->type);
        std::vector<float> tmp(block_size_in_elements);

        const auto* bfp16trait = ggml_get_type_traits(GGML_TYPE_BF16);

        size_t idx = 0;
        const auto* data = (const std::byte*) src;
        for(size_t i=0;i<ggml_nbytes(tensor);i+=block_size_in_bytes) {
            trait->to_float(data+i, tmp.data(), block_size_in_elements);
            bfp16trait->from_float_ref(tmp.data(), vec.get() + idx, block_size_in_elements);
            idx += block_size_in_elements;
        }

        int* refcount = new int(0);
        bfloat16* vec_ptr = vec.get();
        tt::tt_metal::MemoryPin pin(
            [refcount]() mutable {
                (*refcount)++;
            },
            [refcount, vec=std::move(vec)]() mutable {
                assert(refcount != nullptr);
                (*refcount)--;
                if(*refcount == 0) {
                    delete refcount;
                    vec.reset();
                }
            }
        );
        return tt::tt_metal::HostBuffer(ttsl::Span<bfloat16>(vec_ptr, size), std::move(pin));
    }
    else {
        std::shared_ptr<float[]> vec(new float[size]);
        trait->to_float(src, vec.get(), size);
        return host_data_to_tt_host_buffer<float, DstType>(vec.get(), size);
    }
}

// Copies the content of the TT tensor into memory pointed by `dst` with data of type `dst_ggtype`
// This function will do it's best to convert whatever it is in the TT tensor into types accaptable
// by GGML
// This function works by deciding if the tensor is already in the desired format, and if not
// convert to FP32 then convert into the desired format
template <typename SrcType>
static void copy_tt_tensor_to_host_pointer(const tt::tt_metal::Tensor& tensor, void* dst, ggml_type dst_ggtype) {
    ttnn::Shape shape = tensor.logical_shape();
    ttnn::Shape padded_shape = tensor.padded_shape();

    // we only support reading from these types that is held in TT tensor
    static_assert(std::is_same_v<SrcType, float> || std::is_same_v<SrcType, bfloat16> || std::is_same_v<SrcType, uint32_t>);

    tt::tt_metal::Tensor row_major_tensor = tensor;
    if(tensor.layout() == ttnn::TILE_LAYOUT) {
        // Convert tile -> row-major on device, then copy to host. The host-side
        // Thanks TT for finally fixing this
        row_major_tensor = ttnn::untilize(tensor).cpu();
    }
    else {
       row_major_tensor = tensor.cpu();
    }
    GGML_ASSERT(row_major_tensor.storage_type() == tt::tt_metal::StorageType::HOST);
    GGML_ASSERT(row_major_tensor.layout() == ttnn::ROW_MAJOR_LAYOUT);


    // Grab the data held in the TT tensor
    const tt::tt_metal::HostStorage& storage = row_major_tensor.host_storage();
    const auto buffer = storage.buffer().get_shard({0, 0}).value();
    auto view = buffer.view_as<SrcType>();
    const SrcType* buf = &view[0];
    size_t buf_size = view.size();
    GGML_ASSERT(buf != nullptr);

    // Determine our conversion strategy
    void* intermid = nullptr;                // pointer to a buffer that can hold the intermediate data (if needed)
    bool need_quantized_conversion = false;  // flag indicating whether we need to qunatize the value extracted from TT later for GGML use
    bool src_dst_same = false;               // If TT and GGML both have the same type - we can just memcpy

    std::vector<std::byte> intermid_buf;     // In case we need it, some place to put data

    // If both side is FP32
    if(dst_ggtype == GGML_TYPE_F32 && !std::is_same_v<SrcType, float>) {
        intermid = dst;
        need_quantized_conversion = false;
        src_dst_same = false;
    }
    // If both side are the same type fundimentally
    // NOTE: Just putting the integer types here to remind me TT tensors can have integer types
    else if ((std::is_same_v<SrcType, float> && dst_ggtype == GGML_TYPE_F32) ||
             (std::is_same_v<SrcType, bfloat16> && dst_ggtype == GGML_TYPE_BF16) ||
             (std::is_same_v<SrcType, int32_t> && dst_ggtype == GGML_TYPE_I32) ||
             (std::is_same_v<SrcType, uint32_t> && dst_ggtype == GGML_TYPE_I32) ||
             (std::is_same_v<SrcType, int16_t> && dst_ggtype == GGML_TYPE_I16) ||
             (std::is_same_v<SrcType, int8_t> && dst_ggtype == GGML_TYPE_I8)) {
        intermid = dst;
        need_quantized_conversion = false;
        src_dst_same = true;
    }
    // If both side are different - allocate the intermediate buffer and we need to convert
    else {
        intermid_buf.resize(shape.volume() * sizeof(float));
        intermid = intermid_buf.data();
        need_quantized_conversion = true;
        src_dst_same = false;
    }

    auto src_adaptor = [](const SrcType& src) -> float {
        if constexpr(std::is_same_v<SrcType, bfloat16>) {
            return static_cast<float>(src);
        }
        if constexpr (std::is_same_v<SrcType, float>) {
            return src;
        }
        if constexpr (std::is_same_v<SrcType, uint32_t>) {
            return src;
        }
        GGML_UNREACHABLE();
    };

    // Tilize to ROW_MAJOR doesn't mean the tensor is contiguous. It produces tensors that has 0 padded up to the nearest
    // 32 elements on last two (for GGML first two) dimentions.
    // Compute the stride for each dimension
    std::array<size_t, 4> stride = {1, 1, 1, 1};
    size_t cumulative_stride = 1;
    for(int i = padded_shape.size() - 1; i >= 0; i--) {
        stride[i] = cumulative_stride;
        cumulative_stride *= padded_shape[i];
    }

    // Convert TT shape to GGML shape
    std::array<size_t, 4> nshape {1, 1, 1, 1};
    for(size_t i = 0; i < shape.size(); i++) {
        nshape[4 - shape.size() + i] = shape[i];
    }

    static_assert(GGML_MAX_DIMS == 4, "Looping depth is hardcoded to 4");
    // Sanity check: src_dst_same shuld indicate there is no need for quantized conversion
    GGML_ASSERT(((src_dst_same && !need_quantized_conversion) || !src_dst_same) && "src and dst should be the same type if src_dst_same is true");
    // NOTE: The following optimizations are not full and has some slow paths taken unoptimally. But good enough for now

    // Optimization: large block copy
    // If  row major in TT is continous - memcpy it directly or (since we are converting from float) abuse the pointer
    if(nshape[3] % 32 == 0 && ((nshape[0] == 1 && nshape[1] == 1) || nshape[2] % 32 == 0)) {
        const size_t buf_size = std::accumulate(nshape.begin(), nshape.end(), 1, std::multiplies<size_t>());
        // Both sides are same type - memcpy and call it a day
        if(src_dst_same && !need_quantized_conversion) {
            memcpy(dst, buf, sizeof(SrcType) * buf_size);
            return;
        }
        // need conversion but TT side is already FP32 - pointer abuse
        if(std::is_same_v<SrcType, float> && need_quantized_conversion) {
            intermid = const_cast<void*>(static_cast<const void*>(buf));
        }
        // else we manually convert
        else {
            for(size_t i = 0; i < buf_size; i++) {
                ((float*)intermid)[i] = src_adaptor(buf[i]);
            }
        }
    }
    // If the 2nd dimension is not divisible by 32, we can still copy block by block
    else if(nshape[0] % 32 == 0 && nshape[1] % 32 != 0) {
        const size_t src_block_size = nshape[2] * nshape[3];
        const size_t src_block_stride = stride[1];
        if(src_dst_same) {
            for(size_t i=0;i<nshape[0]*nshape[1];i++) {
                memcpy((SrcType*)intermid + i * src_block_size, buf + i * src_block_stride, sizeof(SrcType) * src_block_size);
            }
        }
        else {
            for(size_t i=0;i<nshape[0]*nshape[1];i++) {
                for(size_t j=0;j<src_block_size;j++) {
                    ((SrcType*)intermid)[i * src_block_size + j] = src_adaptor(buf[i * src_block_stride + j]);
                }
            }
        }
    }
    // row-by-row copy
    // Only avoid small copies via memcpy if not copying into FP32 - we rely on raw copies for other types as the
    // fallback loop asserts FP32
    else if(src_dst_same && !need_quantized_conversion && (shape[3] >= 4 || !std::is_same_v<SrcType, float>)) {
        const size_t dst_stride = nshape[3];
        for(size_t i = 0; i < nshape[0] * nshape[1]; i++) {
            for(size_t j = 0; j < nshape[2]; j++) {
                // optimization: copy a row of memory at a time
                const size_t src_idx = i * stride[1] + j * stride[2];
                memcpy((SrcType*)intermid + j * dst_stride + i * nshape[2] * dst_stride, buf + src_idx, sizeof(SrcType) * nshape[3]);
            }
        }
    }
    // Slow path: src and dst are different types or the data is not contiguous in memory
    else {
        size_t idx = 0;
        for(size_t w = 0; w < nshape[0]; w++) {
            for(size_t z = 0; z < nshape[1]; z++) {
                for(size_t y = 0; y < nshape[2]; y++) {
                    for(size_t x = 0; x < nshape[3]; x++) {
                        const size_t src_idx = w * stride[0] + z * stride[1] + y * stride[2] + x * stride[3];
                        GGML_ASSERT(src_idx < buf_size);
                        if(!src_dst_same) {
                            ((float*)intermid)[idx] = src_adaptor(buf[src_idx]);
                        }
                        else {
                            // memcpy((SrcType*)intermid + idx, buf + src_idx, sizeof(SrcType));
                            const SrcType* src_ptr = buf + src_idx;
                            SrcType* dst_ptr = (SrcType*)intermid + idx;
                            *dst_ptr = *src_ptr;
                        }
                        idx++;
                    }
                }
            }
        }
    }

    if (need_quantized_conversion) {
        GGML_ASSERT((ggml_is_quantized(dst_ggtype) || dst_ggtype == GGML_TYPE_F16 || dst_ggtype == GGML_TYPE_I32)
            && "This block should only reach for quantized data types or FP16");
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

static bool is_integer_type(ggml_type type)
{
    std::array<ggml_type, 4> integer_types = {GGML_TYPE_I32, GGML_TYPE_I16, GGML_TYPE_I8, GGML_TYPE_I64};
    return std::find(integer_types.begin(), integer_types.end(), type) != integer_types.end();
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

    // std::cerr << "Reshaping tensor " << tensor.logical_shape() << " to " << target_shape << std::endl;
    return ttnn::reshape(tensor, ttnn::Shape(target_shape));
}

// Attempt to write value into the existing tensor buffer so tracing can work (needing a stable address
// ) this API likely needs rethinking because creating a new tensor here MIGHT break tracing.
static void ggml_metalium_store_tensor(ggml_tensor_extra_metalium* meta, tt::tt_metal::Tensor value)
{
    const auto& cur = meta->tensor;
    if(cur != nullptr
        && cur->storage_type()  == tt::tt_metal::StorageType::DEVICE
        && value.storage_type() == tt::tt_metal::StorageType::DEVICE
        && cur->dtype()         == value.dtype()
        && cur->layout()        == value.layout()
        && cur->logical_shape() == value.logical_shape()) {
        ttnn::copy(value, *cur);   // write into the existing buffer -> address preserved
    } else {
        meta->tensor = std::make_shared<tt::tt_metal::Tensor>(std::move(value));
    }
}

// In-place ops (e.g. the recurrent state cache update) write into a view of a
// pre-allocated tensor. The lazy backend only stores whole-tensor handles, so persist
// the result into the underlying tensor when the view covers it entirely, otherwise
// later graphs would read stale data. Partial writes cannot be expressed and are
// rejected at supports_op time.
static void metalium_persist_inplace_view(const ggml_tensor* node, const std::shared_ptr<tt::tt_metal::Tensor>& value)
{
    ggml_tensor* root = node->view_src;
    if(root == NULL || root->extra == NULL) {
        return;
    }
    if(!ggml_is_contiguous(node) || node->view_offs != 0 || ggml_nelements(node) != ggml_nelements(root)) {
        return;
    }
    ggml_tensor_extra_metalium* root_meta = (ggml_tensor_extra_metalium*)root->extra;
    ggml_metalium_store_tensor(root_meta, reshape_tt_tensor_into_ggml(*value, root));
}

static std::shared_ptr<tt::tt_metal::Tensor> realize_ggml_view_impl(const ggml_tensor* tensor);
static tt::tt_metal::Tensor ggml_metalium_row_unfold(const tt::tt_metal::Tensor& folded);
std::shared_ptr<tt::tt_metal::Tensor> realize_ggml_view(const ggml_tensor* tensor)
{
    ggml_tensor_extra_metalium* meta = static_cast<ggml_tensor_extra_metalium*>(tensor->extra);
    // A consumer that needs the canonical layout of a row-folded tensor (e.g. ggml_concat consuming
    // the token-shift cache) gets it unfolded on demand.
    // XXX: The compiler SHOULD guarantee that the row-folded tensor doesn't have downstream
    // consumers that would be affected by the unfolding, but aparantly there are
    if(meta != nullptr && meta->is_row_folded()) {
        return std::make_shared<tt::tt_metal::Tensor>(ggml_metalium_row_unfold(*meta->row_folded));
    }
    auto res = realize_ggml_view_impl(tensor);
    if(!ggml_tt_tensors_shape_equal(tensor, *res)) {
        std::cout << "FATAL ERROR: Shape mismatch between TTNN and GGML after view op " << ggml_op_name(tensor->op) << "\n"
            << "  Result: " << res->logical_shape() << "\n"
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
        auto parent = realize_ggml_view(src0);
        auto res = ttnn::transpose(*parent, -2, -1);
        return std::make_shared<tt::tt_metal::Tensor>(res);
    }
    if(op == GGML_OP_VIEW) {

        std::shared_ptr<tt::tt_metal::Tensor> parent = realize_ggml_view(tensor->view_src);
        std::array dst_size = std::to_array(tensor->ne);
        std::array dst_stride = std::to_array(tensor->nb);
        std::array src_size = std::to_array(src0->ne);
        std::array src_stride = std::to_array(src0->nb);
        size_t offset = tensor->view_offs;
        // ggml_backend_metalium_buffer_context* bufctx = ((ggml_tensor_extra_metalium*)tensor->extra)->bufctx;

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
        std::array<uint32_t, GGML_MAX_DIMS> step;

        // FIXME: Does not work when we are viewing into a permuted tensor. Sucks
        size_t remaining_offset = offset;
        for(size_t i = GGML_MAX_DIMS - 1; i < GGML_MAX_DIMS; i--) {
            start[i] = remaining_offset / src_stride[i];
            step[i] = src_stride[i] != 0 ? dst_stride[i] / src_stride[i] : 1;
            end[i] = start[i] + dst_size[i] * step[i];
            remaining_offset = remaining_offset % src_stride[i];
        }
        std::reverse(start.begin(), start.end());
        std::reverse(end.begin(), end.end());
        std::reverse(step.begin(), step.end());
        tt::tt_metal::Tensor res;

        if(g_debug_flags.print_view) {
            // Debug prints to help debug complicated view operations
            std::cout << "\nrealize_ggml_view() OP: " << ggml_op_desc(tensor) << "\n";
            std::cout << "  dst name: " << tensor->name << "\n";
            std::cout << "  dst shape: " << tensor->ne[0] << " " << tensor->ne[1] << " " << tensor->ne[2] << " " << tensor->ne[3] << "\n";
            std::cout << "  dst stride: " << tensor->nb[0] << " " << tensor->nb[1] << " " << tensor->nb[2] << " " << tensor->nb[3] << "\n";
            std::cout << "  dst extra: " << tensor->extra << "\n";
            if(tensor->extra != nullptr) {
                ggml_tensor_extra_metalium* meta = (ggml_tensor_extra_metalium*)tensor->extra;
                std::cout << "  dst tensor: " << meta->tensor << "\n";
                if(meta->tensor != nullptr) {
                    std::cout << "  dst tensor shape: " << meta->tensor->logical_shape() << "\n";
                }
            }
            std::cout << "  dst data: " << tensor->data << "\n";
            std::cout << "  dst view_src: " << tensor->view_src << "\n";
            std::cout << "  dst view_src shape: " << tensor->view_src->ne[0] << " " << tensor->view_src->ne[1] << " " << tensor->view_src->ne[2] << " " << tensor->view_src->ne[3] << "\n";
            std::cout << "  dst view_src stride: " << tensor->view_src->nb[0] << " " << tensor->view_src->nb[1] << " " << tensor->view_src->nb[2] << " " << tensor->view_src->nb[3] << "\n";
            std::cout << "  dst src0: " << src0 << "\n";
            std::cout << "  dst src1: " << tensor->src[1] << "\n";
            std::cout << "  src0 shape: " << src0->ne[0] << " " << src0->ne[1] << " " << src0->ne[2] << " " << src0->ne[3] << "\n";
            std::cout << "  src0 stride: " << src0->nb[0] << " " << src0->nb[1] << " " << src0->nb[2] << " " << src0->nb[3] << "\n";
            std::cout << "  src0 OP: " << ggml_op_desc(src0) << "\n";
            std::cout << "  TT parent shape: " << parent->logical_shape() << "\n";
            std::cout << "  TT slice start: " << start[0] << " " << start[1] << " " << start[2] << " " << start[3] << "\n";
            std::cout << "  TT slice end: " << end[0] << " " << end[1] << " " << end[2] << " " << end[3] << "\n";
            std::cout << std::flush;
        }

        // Actually a reshape written as a view
        if(offset == 0 && ggml_nelements(src0) == ggml_nelements(tensor)) {
            res = reshape_tt_tensor_into_ggml(*parent, tensor);
        }
        // Trying to convert a flat 1D tensor to N-D tensor (with an offset, else's it's the above case)
        else if(ggml_n_dims(src0) == 1 && ggml_n_dims(tensor) > 1) {
            // grab the source tensor, slice out the relevant part, and reshape it
            uint32_t offset_elements = offset / ggml_type_size(src0->type);
            uint32_t dst_volume = (uint32_t)ggml_nelements(tensor);
            std::array<uint32_t, GGML_MAX_DIMS> start{0, 0, 0, offset_elements};
            std::array<uint32_t, GGML_MAX_DIMS> end({1, 1, 1, dst_volume + offset_elements});
            std::array<uint32_t, GGML_MAX_DIMS> step = {1, 1, 1, 1};
            tt::tt_metal::Tensor tmp = ttnn::slice(*parent, start, end, step);
            res = reshape_tt_tensor_into_ggml(tmp, tensor);
        }
        // 1-D contiguous sub-view (with offset) of a contiguous parent: it is a flat
        // sub-range of the parent's data. The generic slicer can derive out-of-range
        // indices when src0's shape differs from the realized parent (e.g. the RWKV
        // token-shift / wkv-state split of a flat tensor), so slice the flat range directly.
        else if(ggml_n_dims(tensor) == 1 && ggml_is_contiguous(tensor) && ggml_is_contiguous(src0)) {
            uint32_t offset_elements = offset / ggml_type_size(src0->type);
            uint32_t dst_volume = (uint32_t)ggml_nelements(tensor);
            uint32_t parent_volume = (uint32_t)ggml_nelements(src0);
            tt::tt_metal::Tensor flat = ttnn::reshape(*parent, ttnn::Shape({1, 1, 1, parent_volume}));
            std::array<uint32_t, GGML_MAX_DIMS> fstart{0, 0, 0, offset_elements};
            std::array<uint32_t, GGML_MAX_DIMS> fend{1, 1, 1, dst_volume + offset_elements};
            std::array<uint32_t, GGML_MAX_DIMS> fstep{1, 1, 1, 1};
            tt::tt_metal::Tensor tmp = ttnn::slice(flat, fstart, fend, fstep);
            res = reshape_tt_tensor_into_ggml(tmp, tensor);
        }
        // Flat (contiguous) view of a multi-dimensional parent whose generic per-dim step degenerated to 0
        else if(ggml_is_contiguous(tensor) &&
                std::any_of(step.begin(), step.end(), [](uint32_t s){ return s == 0; })) {
            const auto pshape = parent->logical_shape();
            const uint32_t Cc = pshape[-1];
            const uint32_t off_e = (uint32_t)(offset / ggml_type_size(src0->type));
            const uint32_t len_e = (uint32_t)ggml_nelements(tensor);
            GGML_ASSERT(Cc != 0 && off_e % Cc == 0 && len_e % Cc == 0 &&
                "metalium: contiguous view not aligned to parent row width (unsupported)");
            const uint32_t r0 = off_e / Cc;
            const uint32_t r1 = r0 + len_e / Cc;
            std::array<uint32_t, GGML_MAX_DIMS> start{0, 0, r0, 0};
            std::array<uint32_t, GGML_MAX_DIMS> end{pshape[0], pshape[1], r1, Cc};
            std::array<uint32_t, GGML_MAX_DIMS> step = {1, 1, 1, 1};
            tt::tt_metal::Tensor tmp = ttnn::slice(*parent, start, end, step);
            res = reshape_tt_tensor_into_ggml(tmp, tensor);
        }
        // The fast path, this is what TTNN is designed for (direct slicing)
        else {
            res = ttnn::slice(*parent, start, end, step);
        }

        return std::make_shared<tt::tt_metal::Tensor>(std::move(res));
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

        bool all_zero = true;
        for(int i=0;i<GGML_MAX_DIMS;i++) {
            if(permute[i] != 0) {
                all_zero = false;
                break;
            }
        }
        if(ndiff == 0 || all_zero) {
            return t;
        }

        ttsl::SmallVector<int64_t> permute_tt(GGML_MAX_DIMS);
        for(int i=0;i<GGML_MAX_DIMS;i++) {
            permute_tt[i] = GGML_MAX_DIMS - permute[GGML_MAX_DIMS - i - 1] - 1;
        }
        ttsl::SmallVector<int64_t> permute_tt_real(GGML_MAX_DIMS);
        for(int i=0;i<GGML_MAX_DIMS;i++) {
            permute_tt_real[permute_tt[i]] = i;
        }

        auto res = ttnn::permute(*t, permute_tt_real);
        return std::make_shared<tt::tt_metal::Tensor>(std::move(res));
    }

    ggml_tensor_extra_metalium* meta = (ggml_tensor_extra_metalium*)tensor->extra;
    GGML_ASSERT(meta != nullptr);
    if(meta != nullptr && meta->tensor != nullptr) {
        return meta->tensor;
    }

    if(is_view(tensor) && tensor->view_src != nullptr) {
        // recursivly resolve the source tensor
        return realize_ggml_view(tensor->view_src);
    }

    // HACK: Fallback path: if somehow the framework does not set the real tensor, we can make our own
    // FIXME: GCC says meta->tensor is NULL
    // auto tt_type = ggml2tt_type(tensor->type, meta->tensor->device()->arch());
    // auto shape = ttnn::Shape({uint32_t(tensor->ne[3]), uint32_t(tensor->ne[2]), uint32_t(tensor->ne[1]), uint32_t(tensor->ne[0])});
    // auto res = ttnn::tilize_with_zero_padding(ttnn::zeros(shape, tt::tt_metal::DataType::BFLOAT16).to_device(meta->tensor->device()), std::nullopt, tt_type);
    // meta->tensor = std::make_shared<tt::tt_metal::Tensor>(res);
    // return meta->tensor;
    fmt::println(stderr, "Tensor \"{}\" getting through fallback path. OP = {}, dtype={}", tensor->name, ggml_op_name(tensor->op), ggml_type_name(tensor->type));
    GGML_ASSERT(false && "Fallback path not implemented");
}

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

    GGML_UNUSED(ctx);
    ggml_tensor_extra_metalium* dst_meta = (ggml_tensor_extra_metalium*)dst->extra;

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    bool can_be_processed_by_ttnn = src0->ne[0] == src1->ne[0] && src0->ne[2] == 1 && src1->ne[2] == 1 &&
        (src0->ne[3] == src1->ne[3] || src0->ne[3] == 1);

    if(can_be_processed_by_ttnn) {
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

        auto ap = realize_ggml_view(src0);
        auto bp = realize_ggml_view(src1);
        auto &a = *ap;
        auto &b = *bp;

        *dst_meta = {
            .tensor = std::make_shared<tt::tt_metal::Tensor>(ttnn::operations::matmul::matmul(
                b, a,
                /* transpose_a            = */ false,
                /* transpose_b            = */ true,
                /* memory_config          = */ std::nullopt,
                /* dtype                  = */ std::nullopt,
                /* program_config         = */ std::nullopt,
                /* activation             = */ std::nullopt,
                /* compute_kernel_config  = */ make_compute_kernel_config(a.device()))),
        };
    }
    else {
        GGML_ABORT("unsupported Metalium MUL_MAT shape");
    }
    GGML_ASSERT(dst_meta->tensor->storage_type() == tt::tt_metal::StorageType::DEVICE);
}

// Walk view_src to the storage leaf; return its extra iff that leaf is row-folded, else nullptr.
// The recurrent cache is always reached through a reshape/view, so callers pass the op's src.
static ggml_tensor_extra_metalium* ggml_metalium_resolve_folded(const ggml_tensor* t) {
    const ggml_tensor* root = t;
    while(root->view_src != nullptr) {
        root = root->view_src;
    }
    if(root->extra == nullptr) {
        return nullptr;
    }
    auto* meta = (ggml_tensor_extra_metalium*)root->extra;
    return meta->is_row_folded() ? meta : nullptr;
}

// Unfold [1, R, dim/32, 32] (TILE) -> canonical [1, 1, R, dim] (TILE). ttnn::reshape handles the tile
// relayout internally (a manual untilize would inject hardware padding and corrupt the order). This
// is the temporary bridge to the WKV kernel, which still wants canonical state -- see memory
// rwkv-cache-rowfold-unfold-decision.
static tt::tt_metal::Tensor ggml_metalium_row_unfold(const tt::tt_metal::Tensor& folded) {
    const auto s = folded.logical_shape().to_array_4D();   // [1, R, dim/32, 32]
    const uint32_t R = s[1];
    const uint32_t dim = s[2] * s[3];
    return ttnn::reshape(folded, ttnn::Shape({1, 1, R, dim}));
}

// Fold canonical [1, 1, R, dim] (TILE) -> [1, R, dim/32, 32] (TILE).
static tt::tt_metal::Tensor ggml_metalium_row_fold(const tt::tt_metal::Tensor& canonical) {
    const auto s = canonical.logical_shape().to_array_4D();  // [1, 1, R, dim]
    const uint32_t R = s[2];
    const uint32_t dim = s[3];
    GGML_ASSERT(dim % 32 == 0 && "row-fold requires a tile-aligned inner dim");
    return ttnn::reshape(canonical, ttnn::Shape({1, R, dim / 32, 32}));
}

static bool ggml_backend_metalium_can_cpy(const struct ggml_tensor * dst)
{
    if(is_integer_type(dst->type) || is_integer_type(dst->src[0]->type)) {
        return false;
    }
    if(dst->op != GGML_OP_CPY) {
        return true;
    }
    ggml_tensor* src1 = dst->src[1];
    // A zero-element copy is a no-op (e.g. the empty "extra states" copy in build_rs)
    // and is skipped at compute time.
    if(ggml_nelements(src1) == 0) {
        return true;
    }
    if(ggml_is_permuted(src1)) {
        return false;
    }
    if(is_view(src1)) {
        // The only supported view destination is one that covers an entire
        // pre-allocated tensor (e.g. the recurrent state cache). The lazy backend
        // stores whole-tensor handles, so a partial write cannot be expressed.
        ggml_tensor* root = src1->view_src;
        return root != NULL && ggml_is_contiguous(src1) && src1->view_offs == 0 &&
               ggml_nelements(src1) == ggml_nelements(root);
    }
    return true;
}

static void ggml_backend_metalium_cpy(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    // Don't need sanity check since the copy is lazy
    // GGML_METALIUM_OP_SANITY_CHECK(dst);
    // GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    ggml_tensor_extra_metalium* dst_meta = (ggml_tensor_extra_metalium*)dst->extra;
    ggml_tensor* src0 = dst->src[0];

    // Row-folded recurrent cache write-back
    if(dst->op == GGML_OP_CPY && dst->src[1] != nullptr && dst->src[1]->extra != nullptr) {
        if(ggml_tensor_extra_metalium* folded = ggml_metalium_resolve_folded(dst->src[1])) {
            ggml_tensor* dst_view = dst->src[1];
            const ggml_tensor* root = dst_view;
            while(root->view_src != nullptr) { root = root->view_src; }
            const uint32_t dim    = (uint32_t)root->ne[0];           // n_embd_s
            const uint32_t n_rows = (uint32_t)root->ne[1];
            const size_t   row_bytes = (size_t)dim * ggml_type_size(root->type);
            const uint32_t head  = (uint32_t)(dst_view->view_offs / row_bytes);
            const uint32_t cells = (uint32_t)(ggml_nelements(dst_view) / dim);
            GGML_ASSERT(n_rows == 1 && head == 0 && cells == 1
                && "folded cache write only supports n_rows==1 for now");

            // FAST PATH: src0 is the WKV7 region-2 (= view_1d(wkv_output)). Scatter it straight
            // into the folded cache via a single sub-tile kernel -- no flatten, no fold. This
            // elides the whole region-2 view -> realize(flatten) -> row_fold chain. Restricted to
            // the proven S==H==64, G==1 verbatim-face-copy case; anything else falls through to the
            // generic fold+slice_write below. (The op test never reaches here -- canonical state.)
            if(ggml_tensor* parent = src0->view_src) {
                if(parent->extra != nullptr) {
                    const uint32_t C  = (uint32_t)parent->ne[0];           // n_embd = S*H
                    const uint32_t S  = (C != 0) ? dim / C : 0;            // n_embd_s / n_embd = head_size
                    const uint32_t Hd = (S != 0) ? C / S : 0;             // head_count
                    const uint32_t SG = (C != 0) ? (uint32_t)(ggml_nelements(src0) / C) : 0;
                    const uint32_t G  = (S != 0) ? SG / S : 0;
                    const uint32_t T  = (uint32_t)parent->ne[1] - SG;     // region-2 row offset
                    const uint32_t ng = (C != 0) ? C / 32 : 0;
                    auto src_tt = realize_ggml_view(parent);              // wkv_output (already materialized)
                    const bool eligible =
                        S == 64 && Hd == 64 && G == 1 && C % 1024 == 0 &&
                        src_tt->dtype() == tt::tt_metal::DataType::BFLOAT16 &&
                        folded->row_folded->dtype() == tt::tt_metal::DataType::BFLOAT16;
                    if(eligible) {
                        ttggml::slice_write_region2_folded(*src_tt, *folded->row_folded, T, Hd, ng);
                        // The CPY result IS the (now updated) folded cache. Carry the folded handle
                        // (tensor null) so the canonical post-op shape check is skipped and any
                        // consumer unfolds on demand via realize_ggml_view. Matches get_rows' folded
                        // result convention; nothing actually consumes this node in the RWKV graph.
                        *dst_meta = { .row_folded = folded->row_folded };
                        return;
                    }
                }
            }

            auto new_state = realize_ggml_view(src0);                // [1,1,1,dim]
            tt::tt_metal::Tensor folded_new = ggml_metalium_row_fold(*new_state);  // [1,1,dim/32,32]
            const tt::tt_metal::DataType ftype = folded->row_folded->dtype();
            if(folded_new.dtype() != ftype) {
                folded_new = ttnn::typecast(folded_new, ftype);
            }
            ttnn::SmallVector<uint32_t> begins{0, head, 0, 0};
            ttnn::SmallVector<uint32_t> ends{1, head + cells, dim / 32, 32};
            ttnn::SmallVector<uint32_t> step{1, 1, 1, 1};
            ttnn::experimental::slice_write(folded_new, *folded->row_folded, begins, ends, step);

            // CPY result aliases the (canonical) new state as an ordering handle for any consumer.
            *dst_meta = { .tensor = new_state };
            return;
        }
    }

    // TODO: Check we are not writing into a view
    auto res = realize_ggml_view(src0);
    if(!ggml_tt_tensors_shape_equal(dst, *res)) {
        res = std::make_shared<tt::tt_metal::Tensor>(reshape_tt_tensor_into_ggml(*res, dst));
    }
    auto result_type = ggml2tt_type(dst->type, res->device()->arch());
    if(res->dtype() != result_type) {
        res = std::make_shared<tt::tt_metal::Tensor>(ttnn::typecast(*res, result_type));
    }
    if(dst->op == GGML_OP_CPY) {
        auto* src1 = dst->src[1];
        GGML_ASSERT(src1 != NULL);
        GGML_ASSERT(src1->extra != NULL);
        ggml_tensor_extra_metalium* src1_meta = (ggml_tensor_extra_metalium*)src1->extra;
        metalium_persist_inplace_view(dst, res);
        *src1_meta = {
            .tensor = res,
        };
    }

    *dst_meta = {
        // TODO: Type cast to the appropriate type
        .tensor = res,
    };
}

static bool ggml_backend_metalium_activations(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst, ggml_unary_op op) {
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    GGML_UNUSED(ctx);

    const struct ggml_tensor * src0 = dst->src[0];
    ggml_tensor_extra_metalium* dst_meta = (ggml_tensor_extra_metalium*)dst->extra;

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
            ret = ttnn::hardswish(*src_tensor); // , 1.f/6.f, 0.5
            break;
        case GGML_UNARY_OP_HARDSIGMOID:
            ret = ttnn::hardsigmoid(*src_tensor); // , 1.f/6.f, 0.5
            break;
        case GGML_UNARY_OP_STEP:
            // TODO: Make sure the resulting data type matches the input
            ret = ttnn::typecast(ttnn::gtz(*src_tensor), ggml2tt_type(dst->type, src_tensor->device()->arch()));
            break;
        case GGML_UNARY_OP_EXP:
            ret = ttnn::exp(*src_tensor);
            break;
        case GGML_UNARY_OP_GELU_ERF:
            ret = ttnn::gelu(ttnn::erf(*src_tensor), false);
            break;
        default:
            return false;
    }
    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(std::move(ret)),
    };
    return true;
}
static void ggml_backend_metalium_leaky_relu(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst) {
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    GGML_UNUSED(ctx);

    const struct ggml_tensor * src0 = dst->src[0];
    ggml_tensor_extra_metalium* dst_meta = (ggml_tensor_extra_metalium*)dst->extra;
    auto src_tensor = realize_ggml_view(src0);

    float negative_slope;
    GGML_ASSERT(dst->op_params != NULL);
    memcpy(&negative_slope, dst->op_params, sizeof(float));

    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(ttnn::leaky_relu(*src_tensor, negative_slope)),
    };
}
static void ggml_backend_metalium_bin_op(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst, ggml_op op) {
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC1_SANITY_CHECK(dst);
    GGML_UNUSED(ctx);

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    ggml_tensor_extra_metalium* dst_meta = (ggml_tensor_extra_metalium*)dst->extra;

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

    ggml_tensor_extra_metalium* dst_meta = (ggml_tensor_extra_metalium*)dst->extra;
    ggml_tensor_extra_metalium* src0_meta = (ggml_tensor_extra_metalium*)dst->src[0]->extra;
    ggml_tensor_extra_metalium* src1_meta = (ggml_tensor_extra_metalium*)dst->src[1]->extra;

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
        };
    }
    else {
        std::shared_ptr<tt::tt_metal::Tensor> tensor = std::make_shared<tt::tt_metal::Tensor>(res);
        *src0_meta = {
            .tensor = tensor,
        };
        *dst_meta = {
            .tensor = tensor,
        };
    }
}
static void ggml_backend_metalium_clamp(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst)
{
    GGML_UNUSED(ctx);
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);

    ggml_tensor_extra_metalium* dst_meta = (ggml_tensor_extra_metalium*)dst->extra;

    float data[2];
    memcpy(data, dst->op_params, sizeof(data));
    auto [min, max] = std::to_array(data);

    auto t = realize_ggml_view(dst->src[0]);
    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(ttnn::clamp(*t, min, max)),
    };
}

static void ggml_backend_metalium_scale(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst)
{
    GGML_UNUSED(ctx);
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);

    ggml_tensor_extra_metalium* dst_meta = (ggml_tensor_extra_metalium*)dst->extra;

    std::array<float, 2> params;
    memcpy(params.data(), dst->op_params, sizeof(params));
    auto [scale, bias] = params;

    // Row-folded recurrent cache in-place scale (the build_rs state-clear: scale a row by 0). Scaling
    // is element-wise so it commutes with the fold -- scale the folded store's row range directly, no
    // unfold, and write it back in place. n_rows==1 (whole cache) for now.
    if(ggml_tensor_extra_metalium* folded = ggml_metalium_resolve_folded(dst->src[0])) {
        const ggml_tensor* root = dst->src[0];
        while(root->view_src != nullptr) { root = root->view_src; }
        const uint32_t dim   = (uint32_t)root->ne[0];
        const size_t   row_bytes = (size_t)dim * ggml_type_size(root->type);
        const uint32_t head  = (uint32_t)(dst->src[0]->view_offs / row_bytes);
        const uint32_t cells = (uint32_t)(ggml_nelements(dst->src[0]) / dim);
        GGML_ASSERT(root->ne[1] == 1 && head == 0 && cells == 1
            && "folded cache scale only supports n_rows==1 for now");
        ttnn::Tensor scaled = (bias == 0.f)
            ? ttnn::multiply(*folded->row_folded, scale)
            : ttnn::add(ttnn::multiply(*folded->row_folded, scale, std::nullopt, ttnn::L1_MEMORY_CONFIG), bias);
        const tt::tt_metal::DataType ftype = folded->row_folded->dtype();
        if(scaled.dtype() != ftype) { scaled = ttnn::typecast(scaled, ftype); }
        ttnn::SmallVector<uint32_t> begins{0, head, 0, 0};
        ttnn::SmallVector<uint32_t> ends{1, head + cells, dim / 32, 32};
        ttnn::SmallVector<uint32_t> step{1, 1, 1, 1};
        ttnn::experimental::slice_write(scaled, *folded->row_folded, begins, ends, step);
        // The in-place SCALE node is itself a (canonical) ggml view; if any consumer realizes it the
        // shape check needs canonical [1,1,cells,dim]. Cold path (only when rs_zero>=0), so unfold.
        *dst_meta = { .tensor = std::make_shared<tt::tt_metal::Tensor>(ggml_metalium_row_unfold(scaled)) };
        return;
    }

    auto t = realize_ggml_view(dst->src[0]);
    ttnn::Tensor res;
    if(bias == 0.f) {
        res = ttnn::multiply(*t, scale);
    }
    else {
        res = ttnn::add(ttnn::multiply(*t, scale, std::nullopt, ttnn::L1_MEMORY_CONFIG), bias);
    }
    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(std::move(res)),
    };
    metalium_persist_inplace_view(dst, dst_meta->tensor);
}

static bool ggml_backend_metalium_can_get_rows(const struct ggml_tensor * dst, tt::ARCH arch)
{
    const ggml_tensor *idxs = dst->src[1];
    // No rows to gather (e.g. the empty "extra states" gather in build_rs): no-op, skipped at compute.
    if(ggml_nelements(dst) == 0) {
        return true;
    }
    // effectivly no-op
    if(idxs->ne[0] == 1 && idxs->ne[1] == 1 && idxs->ne[2] == 1 && idxs->ne[3] == 1 && ggml_n_dims(dst->src[0]) == 1) {
        return true;
    }

    const ggml_tensor* src = dst->src[0];
    if(is_integer_type(src->type)) {
        return false;
    }

    // The non-trivial path goes through ttnn::tosa::gather, which tile-pads the source via
    // TTNN's fill_pad op. fill_pad only supports BFLOAT16/FLOAT32/UINT16/UINT32/INT32 - the
    // block-float types (BFLOAT8_B/BFLOAT4_B) are rejected, so we can't gather quantized
    // sources and must fall back to the CPU for them.
    tt::tt_metal::DataType src_tt_type = ggml2tt_type(src->type, arch);
    if(src_tt_type == tt::tt_metal::DataType::BFLOAT8_B || src_tt_type == tt::tt_metal::DataType::BFLOAT4_B) {
        return false;
    }

    // FIXME: TTNN running into issues with large tensor....?
    if(idxs->ne[0] > 256) {
        return false;
    }

    // FIXME: Doesn't seem to be working correctly when batched
    if(src->ne[2] != 1 || src->ne[3] != 1) {
        return false;
    }

    if(idxs->ne[2] == 1 && src->ne[3] == 1 && !is_view(idxs)) {
        return true;
    }

    return false;
}

static void ggml_backend_metalium_get_rows(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst)
{
    GGML_UNUSED(ctx);
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);

    ggml_tensor_extra_metalium* dst_meta = (ggml_tensor_extra_metalium*)dst->extra;

    // Row-folded recurrent cache read. For now only n_rows==1 (the whole-cache identity gather):
    // unfold the folded store back to the canonical [1,1,1,n_embd_s] the WKV kernel needs. General
    // n_rows>1 needs an on-device folded gather (TODO: scatter/gather kernel).
    if(ggml_tensor_extra_metalium* folded = ggml_metalium_resolve_folded(dst->src[0])) {
        const ggml_tensor* root = dst->src[0];
        while(root->view_src != nullptr) { root = root->view_src; }
        GGML_ASSERT(root->ne[1] == 1 && "folded cache gather only supports n_rows==1 for now");
        // Emit a folded SNAPSHOT of the cache. Aliasing the live cache buffer would let a lazy
        // consumer read a generation that a later in-place cache write has already clobbered. WKV7
        // consumes the snapshot folded; any other consumer (concat of the token-shift cache) unfolds
        // it on demand in realize_ggml_view.
        dst_meta->row_folded = std::make_shared<tt::tt_metal::Tensor>(
            ttnn::clone(*folded->row_folded, std::nullopt, std::nullopt, std::nullopt));
        return;
    }

    auto t = realize_ggml_view(dst->src[0]);
    const ggml_tensor *idxs = dst->src[1];
    if(idxs->ne[0] == 1 && idxs->ne[1] == 1 && idxs->ne[2] == 1 && idxs->ne[3] == 1 && ggml_n_dims(dst->src[0]) == 1) {
        *dst_meta = {
            .tensor = std::make_shared<tt::tt_metal::Tensor>(ttnn::clone(*t, std::nullopt, std::nullopt, std::nullopt)),
        };
    }
    else {
        ggml_tensor_extra_metalium* idx_meta = (ggml_tensor_extra_metalium*)idxs->extra;
        GGML_ASSERT(idx_meta != nullptr);
        // The operation wants 3D tensor but we have 4D, op also wants index be 2d
        auto src3d = t->reshape(t->logical_shape().to_rank(3));
        auto idx2d = idx_meta->tensor->reshape(idx_meta->tensor->logical_shape().to_rank(2));
        ttnn::Tensor gathered = ttnn::tosa::gather(src3d, ttnn::tilize_with_zero_padding(idx2d), std::nullopt);
        gathered = gathered.reshape(gathered.logical_shape().to_rank(4));
        *dst_meta = {
            .tensor = std::make_shared<ttnn::Tensor>(gathered)
        };
    }
}

static bool ggml_backend_metalium_can_set_rows(const struct ggml_tensor * dst)
{
    // GGML has a weird order
    // result->src[0] = src
    // result->src[1] = idx
    // result->src[2] = dst
    const ggml_tensor *idxs = dst->src[1];
    // effectivly no-op
    if(idxs->ne[0] == 1 && idxs->ne[1] == 1 && idxs->ne[2] == 1 && idxs->ne[3] == 1 && ggml_n_dims(dst->src[0]) == 1) {
        return true;
    }

    const ggml_tensor* src = dst->src[0];
    if(is_integer_type(src->type)) {
        return false;
    }

    if(idxs->ne[2] == 1 && src->ne[3] == 1 && !is_view(idxs)) {
        return true;
    }

    return false;
}

static void ggml_backend_metalium_set_rows(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst)
{
    GGML_UNUSED(ctx);
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC_SANITY_CHECK(dst, 2);

    ggml_tensor_extra_metalium* dst_meta = (ggml_tensor_extra_metalium*)dst->extra;
    ggml_tensor_extra_metalium* real_dst_meta = (ggml_tensor_extra_metalium*)dst->src[2]->extra;
    ggml_tensor_extra_metalium* idx_meta = (ggml_tensor_extra_metalium*)dst->src[1]->extra;

    auto real_dst = realize_ggml_view(dst->src[2]);
    auto src = realize_ggml_view(dst->src[0]);
    auto idx = idx_meta->tensor;
    const ggml_tensor *idxs = dst->src[1];

    // Setting on a 1 row tensor is guarenteed to just be a replacment
    if(idxs->ne[0] == 1 && idxs->ne[1] == 1 && idxs->ne[2] == 1 && idxs->ne[3] == 1 && ggml_n_dims(dst->src[2]) == 1) {
        *dst_meta = {
            .tensor = src,
        };
        *real_dst_meta = {
            .tensor = src,
        };
    }
    else {
        ggml_tensor_extra_metalium* idx_meta = (ggml_tensor_extra_metalium*)idxs->extra;
        GGML_ASSERT(idx_meta != nullptr);
        // The operation wants 3D tensor but we have 4D, op also wants index be 2d
        auto src3d = src->reshape(src->logical_shape().to_rank(3));
        auto idx2d = idx_meta->tensor->reshape(idx_meta->tensor->logical_shape().to_rank(2));
        auto real_dst3d = real_dst->reshape(real_dst->logical_shape().to_rank(3));
        ttnn::Tensor res = ttnn::tosa_scatter(real_dst3d, ttnn::tilize_with_zero_padding(idx2d), src3d, std::nullopt);
        fmt::println("res: {}", res.logical_shape());
        res = res.reshape(res.logical_shape().to_rank(4));
        *dst_meta = {
            .tensor = std::make_shared<ttnn::Tensor>(res),
        };
        *real_dst_meta = {
            .tensor = std::make_shared<ttnn::Tensor>(res),
        };
    }
}

static bool ggml_backend_metalium_can_norm(const struct ggml_tensor * dst, bool rms)
{
    GGML_UNUSED(rms);
    // no hard checks but this seems to work well enough, else we run out of SRAM
    if(dst->ne[0] > 4096) {
        return false;
    }
    return true;
}

static void ggml_backend_metalium_norm(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst, bool rms)
{
    GGML_UNUSED(ctx);
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);

    ggml_tensor_extra_metalium* dst_meta = (ggml_tensor_extra_metalium*)dst->extra;

    float esp = 0;
    memcpy(&esp, dst->op_params, sizeof(esp));

    // HACK: the norm implementations in TTNN does not like size 1 tensors - we know the result is going to be sign(x)
    // so let's just make that
    auto t = realize_ggml_view(dst->src[0]);
    if(t->logical_shape()[-1] == 1) {
        *dst_meta = {
            .tensor = std::make_shared<tt::tt_metal::Tensor>(ttnn::typecast(ttnn::sign(*t), t->dtype())),
        };
        return;
    }

    tt::tt_metal::Tensor res;
    if(rms) {
        res = ttnn::rms_norm(*t, esp);
    }
    else {
        res = ttnn::layer_norm(*t, esp);
    }
    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(std::move(res)),
    };
}

static void ggml_backend_metalium_l2_norm(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst)
{
    GGML_UNUSED(ctx);
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);

    ggml_tensor_extra_metalium* dst_meta = (ggml_tensor_extra_metalium*)dst->extra;

    float eps = 0;
    memcpy(&eps, dst->op_params, sizeof(eps));

    auto t = realize_ggml_view(dst->src[0]);

    // L2 norm: y = x / max(sqrt(sum(x^2)), eps), reduction along the last (ne[0]) dimension
    ttnn::WormholeComputeKernelConfig cfg{
        .math_fidelity = MathFidelity::HiFi4,
        .math_approx_mode = false,
        .fp32_dest_acc_en = true,
        .packer_l1_acc = true
    };
    auto sumsq = ttnn::sum(ttnn::square(*t), 3, /*keepdim=*/true, std::nullopt, cfg);
    auto denom = ttnn::clamp(ttnn::sqrt(sumsq), eps, std::numeric_limits<float>::max());
    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(ttnn::divide(*t, denom)),
    };
}

static void ggml_backend_metalium_add1(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst)
{
    GGML_UNUSED(ctx);
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC1_SANITY_CHECK(dst);

    ggml_tensor_extra_metalium* dst_meta = (ggml_tensor_extra_metalium*)dst->extra;

    auto t = realize_ggml_view(dst->src[0]);
    auto q = realize_ggml_view(dst->src[1]);
    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(ttnn::add(*t, *q)),
    };
}

static void ggml_backend_metalium_sqrt(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst)
{
    GGML_UNUSED(ctx);
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);

    ggml_tensor_extra_metalium* dst_meta = (ggml_tensor_extra_metalium*)dst->extra;

    float esp = 0;
    memcpy(&esp, dst->op_params, sizeof(esp));

    auto t = realize_ggml_view(dst->src[0]);
    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(ttnn::sqrt(*t)),
    };
}

static void ggml_backend_metalium_sqr(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst)
{
    GGML_UNUSED(ctx);
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);

    ggml_tensor_extra_metalium* dst_meta = (ggml_tensor_extra_metalium*)dst->extra;

    float esp = 0;
    memcpy(&esp, dst->op_params, sizeof(esp));

    auto t = realize_ggml_view(dst->src[0]);
    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(ttnn::square(*t)),
    };
}

static bool ggml_backend_metalium_can_concat(const struct ggml_tensor * dst)
{
    if(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_BF16 || dst->type == GGML_TYPE_F16) {
        return true;
    }
    return false;
}

static void ggml_backend_metalium_concat(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst)
{
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC1_SANITY_CHECK(dst);
    GGML_UNUSED(ctx);

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    ggml_tensor_extra_metalium* dst_meta = (ggml_tensor_extra_metalium*)dst->extra;

    // One operand can be empty along the concat axis. This happens e.g. in the RWKV
    // token-shift concat during generation: ggml_view of [n_embd, n_seq_tokens-1, ...]
    // collapses to a zero-sized dimension when n_seq_tokens == 1. The concat result is
    // then just the other operand. ttnn::concat cannot ingest a zero-sized tensor (its
    // untilize fallback divides by the tile count and hits an FPE), and the empty view
    // is never materialized on device, so handle this before realizing anything.
    if(ggml_nelements(src1) == 0) {
        *dst_meta = { .tensor = realize_ggml_view(src0) };
        return;
    }
    if(ggml_nelements(src0) == 0) {
        *dst_meta = { .tensor = realize_ggml_view(src1) };
        return;
    }

    auto src_tensor0 = realize_ggml_view(src0);
    auto src_tensor1 = realize_ggml_view(src1);

    int32_t axis = 0;
    memcpy(&axis, dst->op_params, sizeof(axis));
    axis = GGML_MAX_DIMS - axis - 1;

    std::vector<tt::tt_metal::Tensor> targets = {*src_tensor0, *src_tensor1};
    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(ttnn::concat(targets, axis)),
    };
}

static bool ggml_backend_metalium_can_softmax(const struct ggml_tensor * dst)
{
    GGML_UNUSED(dst);
    return true;
    // std::array<float, 2> params;
    // memcpy(&params, dst->op_params, sizeof(params));
    // auto [scale, max_bias] = params;
    // return max_bias == 0.f;
}

static void ggml_backend_metalium_softmax(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst)
{
    GGML_UNUSED(ctx);
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);

    std::array<float, 2> params;
    memcpy(&params, dst->op_params, sizeof(params));
    auto [scale, max_bias] = params;
    ggml_tensor_extra_metalium* dst_meta = (ggml_tensor_extra_metalium*)dst->extra;
#if 0
    auto x = *realize_ggml_view(dst->src[0]);
    if(dst->src[1] == NULL) {
        *dst_meta = {
            .tensor = std::make_shared<tt::tt_metal::Tensor>(ttggml::soft_max(x, scale))
        };
    }
    else {
        auto mask = *realize_ggml_view(dst->src[1]);
        *dst_meta = {
            .tensor = std::make_shared<tt::tt_metal::Tensor>(ttggml::soft_max(x, mask, scale))
        };
    }
#else

    const ggml_tensor *src0 = dst->src[0];
    const ggml_tensor *src1 = dst->src[1];

    auto t = realize_ggml_view(src0);
    tt::tt_metal::Tensor x = *t;
    // XXX: TTNN's own implementation does not handle broadcasting as GGML wants
    // if(src1 != nullptr) {
    //     auto mask = realize_ggml_view(src1);
    //     x = ttnn::operations::normalization::scale_mask_softmax(*t, scale, *mask);
    // }
    if(scale != 1.f) {
        x = ttnn::multiply(*t, scale);
    }

    if(src1 != nullptr) {
        auto mask = *realize_ggml_view(src1);
        if(max_bias == 0.f) {
            x = ttnn::add(x, mask);
        }
        else {
            // TODO: Replace this with a single operator that works on the mask
            // Also TODO: Make a new softmax that just does everything GGML wants
            const int n_head      = src0->ne[2];
            const int n_head_log2 = 1u << (uint32_t) floorf(log2f((float) n_head));

            const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
            const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);
            auto slopes = ttnn::arange(0, n_head, 1, tt::tt_metal::DataType::FLOAT32, *x.device(), ttnn::DRAM_MEMORY_CONFIG, ttnn::TILE_LAYOUT);
            auto base = ttnn::where(ttnn::lt(slopes, n_head_log2), m0, m1);
            auto exp = ttnn::where(ttnn::lt(slopes, n_head_log2), ttnn::add(slopes, 1), ttnn::add(ttnn::multiply(ttnn::subtract(slopes, n_head_log2), 2.f), 1));
            slopes = ttnn::pow(base, exp, tt::tt_metal::DataType::BFLOAT16);
            slopes = ttnn::transpose(slopes.reshape(slopes.logical_shape().to_rank(4)), 1, 3);
            mask = ttnn::multiply(mask, slopes);
            x = ttnn::add(x, mask);
        }
    }
    x = ttnn::softmax(x, 3);
    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(std::move(x)),
    };
#endif
}

static void ggml_backend_metalium_cos(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst)
{
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    GGML_UNUSED(ctx);

    const struct ggml_tensor * src0 = dst->src[0];
    ggml_tensor_extra_metalium* dst_meta = (ggml_tensor_extra_metalium*)dst->extra;

    auto src = realize_ggml_view(src0);
    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(ttnn::cos(*src)),
    };
}

static void ggml_backend_metalium_sin(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst)
{
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    GGML_UNUSED(ctx);

    const struct ggml_tensor * src0 = dst->src[0];
    ggml_tensor_extra_metalium* dst_meta = (ggml_tensor_extra_metalium*)dst->extra;

    auto src = realize_ggml_view(src0);
    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(ttnn::sin(*src)),
    };
}

static void ggml_backend_metalium_log(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst)
{
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    GGML_UNUSED(ctx);

    const struct ggml_tensor * src0 = dst->src[0];
    ggml_tensor_extra_metalium* dst_meta = (ggml_tensor_extra_metalium*)dst->extra;

    auto src = realize_ggml_view(src0);
    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(ttnn::log(*src)),
    };
}

static void ggml_backend_metalium_arange(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst)
{
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_UNUSED(ctx);

    ggml_tensor_extra_metalium* dst_meta = (ggml_tensor_extra_metalium*)dst->extra;
    auto* device = dst_meta->tensor->device();
    std::array<float, 3> params;
    memcpy(&params, dst->op_params, sizeof(params));
    auto [start, end, step] = params;
    auto dtype = ggml2tt_type(dst->type, device->arch());
    if(dtype == tt::tt_metal::DataType::INVALID) {
        fmt::println(stderr, "Unsupported GGML type {}", ggml_type_name(dst->type));
        GGML_ASSERT(false && "Unsupported GGML type");
    }

    auto tensor = ttnn::arange(start, end, step, dtype, *device, ttnn::DRAM_MEMORY_CONFIG, ttnn::TILE_LAYOUT);
    tensor = tensor.reshape(tensor.logical_shape().to_rank(4));
    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(std::move(tensor)),
    };
}

static void ggml_backend_metalium_group_norm(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst)
{
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_UNUSED(ctx);

    ggml_tensor_extra_metalium* dst_meta = (ggml_tensor_extra_metalium*)dst->extra;
    int n_groups;
    float eps;
    memcpy(&n_groups, dst->op_params, sizeof(n_groups));
    memcpy(&eps, dst->op_params + 1, sizeof(eps));

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
    };
}

static bool ggml_backend_metalium_can_repeat(const struct ggml_tensor * dst)
{
    // TODO: File bug report that repear op should support UINT32
    if(dst->type == GGML_TYPE_I32) {
        return false;
    }
    ggml_tensor *src0 = dst->src[0];
    for(int i = 0; i < GGML_MAX_DIMS; i++) {
        if(dst->ne[i] % src0->ne[i] != 0) {
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

    ggml_tensor_extra_metalium* dst_meta = (ggml_tensor_extra_metalium*)dst->extra;
    ggml_tensor* src0 = dst->src[0];

    auto tensor = realize_ggml_view(dst->src[0]);
    ttsl::SmallVector<uint32_t> repeats;
    repeats.resize(GGML_MAX_DIMS);
    int ndiff = 0;
    for(int i = 0; i < GGML_MAX_DIMS; i++) {
        auto repeat = dst->ne[i] / src0->ne[i];
        repeats[GGML_MAX_DIMS - i - 1] = repeat;
        ndiff += (repeat != 1);
    }
    if(ndiff == 0) {
        *dst_meta = {
            .tensor = std::make_shared<tt::tt_metal::Tensor>(*tensor),
        };
        return;
    }

    auto res = ttnn::repeat(*tensor, ttnn::Shape(repeats));
    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(res),
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
    return num_ones_in_shape(dst->src[0]) == 3 && num_ones_in_shape(dst->src[1]) == 3;
}

static void ggml_backend_metalium_outer_product(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst)
{
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC1_SANITY_CHECK(dst);
    GGML_UNUSED(ctx);

    ggml_tensor_extra_metalium* dst_meta = (ggml_tensor_extra_metalium*)dst->extra;

    auto src0 = realize_ggml_view(dst->src[0]);
    auto src1 = realize_ggml_view(dst->src[1]);

    auto res = ttnn::outer(*src1, *src0);
    // HACK: GGML and TT has different ideas about the shape of the result, sometimes
    if(!ggml_tt_tensors_shape_equal(dst, res)) {
        // Magic herustics
        if(dst->ne[3] == res.logical_shape()[2]) {
            res = ttnn::transpose(res, 0, 2);
        }
        else if(dst->ne[2] == res.logical_shape()[2]) {
            res = ttnn::transpose(res, 1, 2);
        }
        else {
            std::cerr << "GGML shape: " << dst->ne[0] << ", " << dst->ne[1] << ", " << dst->ne[2] << ", " << dst->ne[3] << "\n";
            std::cerr << "TT shape: " << res.logical_shape() << "\n";
            GGML_ASSERT(false && "Unsupported outer product shape mismatch");
        }
    }
    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(res),
    };
}
static void ggml_backend_metalium_sum(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst)
{
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    GGML_UNUSED(ctx);

    ggml_tensor_extra_metalium* dst_meta = (ggml_tensor_extra_metalium*)dst->extra;

    auto t = realize_ggml_view(dst->src[0]);
    ttnn::WormholeComputeKernelConfig cfg{
        .math_fidelity = MathFidelity::HiFi4,
        .math_approx_mode = false,
        .fp32_dest_acc_en = true,
        .packer_l1_acc = true
    };
    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(ttnn::sum(*t, std::nullopt, false, std::nullopt, cfg)),
    };
}

static bool ggml_backend_metalium_can_sum_rows(const struct ggml_tensor * dst)
{
    GGML_UNUSED(dst);
    return true;
}

static void ggml_backend_metalium_sum_rows(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst)
{
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    GGML_UNUSED(ctx);

    ggml_tensor_extra_metalium* dst_meta = (ggml_tensor_extra_metalium*)dst->extra;

    auto t = realize_ggml_view(dst->src[0]);
    ttnn::WormholeComputeKernelConfig cfg{
        .math_fidelity = MathFidelity::HiFi4,
        .math_approx_mode = false,
        .fp32_dest_acc_en = true,
        .packer_l1_acc = true
    };
    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(ttnn::sum(*t, 3, /*keepdim=*/true, std::nullopt, cfg)),
    };
}

static bool ggml_backend_metalium_can_glu(const struct ggml_tensor * dst)
{
    constexpr std::array<ggml_glu_op, 5> supported = {
        GGML_GLU_OP_REGLU,
        GGML_GLU_OP_GEGLU_ERF,
        GGML_GLU_OP_GEGLU_QUICK,
        GGML_GLU_OP_GEGLU,
        GGML_GLU_OP_SWIGLU
    };
    if(std::find_if(supported.begin(), supported.end(), [&](ggml_glu_op op) {
            return ggml_get_glu_op(dst) == op;
        }) == supported.end()) {
        return false;
    }


    bool split = dst->src[1] != NULL;
    if(split) {
        return true;
    }

    return dst->src[0]->ne[0] % 2 == 0;
}

static void ggml_backend_metalium_glu(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst)
{
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    GGML_UNUSED(ctx);

    ggml_tensor_extra_metalium* dst_meta = (ggml_tensor_extra_metalium*)dst->extra;

    ttnn::Tensor a;
    ttnn::Tensor b;
    int swap = ggml_get_op_params_i32(dst, 1);
    bool split = dst->src[1] != NULL;

    if(split) {
        a = *realize_ggml_view(dst->src[1]);
        b = *realize_ggml_view(dst->src[0]);
    }
    else {
        auto t = realize_ggml_view(dst->src[0]);
        // split along the last dimension
        int64_t w = dst->ne[0];

        using Slice = std::array<uint32_t, GGML_MAX_DIMS>;
        Slice mid = {uint32_t(w), uint32_t(dst->ne[1]), uint32_t(dst->ne[2]), uint32_t(dst->ne[3])};
        std::reverse(mid.begin(), mid.end());
        Slice mid_start = {uint32_t(w), 0, 0, 0};
        std::reverse(mid_start.begin(), mid_start.end());
        Slice end = {uint32_t(w * 2), uint32_t(dst->ne[1]), uint32_t(dst->ne[2]), uint32_t(dst->ne[3])};
        std::reverse(end.begin(), end.end());
        Slice begin = {0, 0, 0, 0};
        Slice stride = {1, 1, 1, 1};

        a = ttnn::slice(*t, mid_start, end, stride);
        b = ttnn::slice(*t, begin, mid, stride);
    }


    if(swap) {
        std::swap(a, b);
    }

    ttnn::Tensor res;
    switch(ggml_get_glu_op(dst)) {
        case GGML_GLU_OP_REGLU:
            res = ttnn::multiply(a, ttnn::relu(b, ttnn::L1_MEMORY_CONFIG));
            break;
        case GGML_GLU_OP_GEGLU_ERF: // ?
        case GGML_GLU_OP_GEGLU_QUICK:
        case GGML_GLU_OP_GEGLU:
            res = ttnn::multiply(a, ttnn::gelu(b, false, ttnn::L1_MEMORY_CONFIG));
            break;
        case GGML_GLU_OP_SWIGLU:
            res = ttnn::multiply(a, ttnn::swish(b, ttnn::L1_MEMORY_CONFIG));
            break;
        default:
            GGML_ASSERT(false && "Unsupported GLU operation");
    }

    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(std::move(res)),
    };
}

static bool ggml_backend_metalium_can_rope(const struct ggml_tensor * dst)
{
    // In-place RoPE (dst is a view aliasing src0) whose source is a non-contiguous
    // view is mishandled by the backend's view write-back, so the result is wrong.
    // Decline it and let the CPU backend handle those cases. (Maps to the
    // test-backend-ops inplace=1,v=1 variants.)
    if(dst->view_src != nullptr && !ggml_is_contiguous(dst->src[0])) {
        return false;
    }

    std::array<int32_t, 5> int_params;
    memcpy(int_params.data(), dst->op_params, sizeof(int_params));
    auto [
        n_past,
        n_dims,
        mode,
        n_ctx,
        n_ctx_orig
    ] = int_params;

    if(mode == GGML_ROPE_TYPE_NEOX) {
        return n_dims % 64 == 0;
    }
    if(mode == GGML_ROPE_TYPE_NORMAL) {
        ggml_tensor* ff = dst->src[2];
        if(dst->src[2]) {
            return !ggml_is_quantized(ff->type)  // we do sub-tile hacking
            && ff->ne[0] % 32 == 0 && n_dims % 32 == 0; // XXX: This case fails
        }
        return n_dims % 32 == 0;
    }

    return false;
}


static void ggml_backend_metalium_rope(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst)
{
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    GGML_UNUSED(ctx);

    ggml_tensor_extra_metalium* dst_meta = (ggml_tensor_extra_metalium*)dst->extra;

    std::array<int32_t, 5> int_params;
    memcpy(int_params.data(), dst->op_params, sizeof(int_params));
    auto [
        n_past,
        n_dims,
        mode,
        n_ctx,
        n_ctx_orig ] = int_params;

    std::array<float, 6> float_params;
    memcpy(float_params.data(), dst->op_params + int_params.size() , sizeof(float_params));
    auto [
        freq_base,
        freq_scale,
        ext_factor,
        attn_factor,
        beta_fast,
        beta_slow
    ] = float_params;

    auto res = [&](){
        if(dst->src[2]) {
            return ttggml::rope(
                *realize_ggml_view(dst->src[0]),
                *realize_ggml_view(dst->src[1]),
                *realize_ggml_view(dst->src[2]),
                n_dims,
                mode == GGML_ROPE_TYPE_NEOX ? ttggml::RoPEType::NeoX : ttggml::RoPEType::Normal,
                n_ctx_orig,
                freq_base,
                freq_scale,
                ext_factor,
                attn_factor,
                beta_fast,
                beta_slow);
        }
        return ttggml::rope(
            *realize_ggml_view(dst->src[0]),
            *realize_ggml_view(dst->src[1]),
            n_dims,
            mode == GGML_ROPE_TYPE_NEOX ? ttggml::RoPEType::NeoX : ttggml::RoPEType::Normal,
            n_ctx_orig,
            freq_base,
            freq_scale,
            ext_factor,
            attn_factor,
            beta_fast,
            beta_slow);
    }();
    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(std::move(res)),
    };
}

static void ggml_backend_metalium_rwkv_wkv7(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst)
{
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    for(int i = 0; i < 7; i++) {
        GGML_METALIUM_OP_SRC_SANITY_CHECK(dst, i);
    }
    GGML_UNUSED(ctx);

    ggml_tensor_extra_metalium* dst_meta = (ggml_tensor_extra_metalium*)dst->extra;

    // State (src[6]) always reaches the reader row-folded [1,G,Es/32,32]. Fast path: the state gather
    // propagated the folded handle, consume it with zero relayout. Fallback (op test, canonical state):
    // fold on entry, one relayout per WKV7 call.
    std::shared_ptr<tt::tt_metal::Tensor> state;
    if (ggml_tensor_extra_metalium* folded = ggml_metalium_resolve_folded(dst->src[6])) {
        state = folded->row_folded;
    } else {
        state = std::make_shared<tt::tt_metal::Tensor>(ggml_metalium_row_fold(*realize_ggml_view(dst->src[6])));
    }

    // r/w/k/v (src 0..3) are [n_embd,T]->[S,H,T] head reshapes in the real model. Feed the un-reshaped
    // [n_embd,T] parent instead and let the reader address each head's column-tiles itself (one
    // ttnn::reshape/input/layer eliminated). a/b (src 4,5) are l2_norm-derived and stay [S,H,T]. The
    // guard (RESHAPE of a contiguous [S*H,T] parent) means a non-reshape src -- e.g. test-backend-ops
    // leaf inputs -- stays on the reshaped path; invoke() then detects the reshaped shape and never
    // defines WKV7_INPUT_FLAT, so host and reader agree.
    auto realize_wkv7_input = [&](int i) -> std::shared_ptr<tt::tt_metal::Tensor> {
        const ggml_tensor* s = dst->src[i];
        if (i >= 0 && i <= 3
            && s->op == GGML_OP_RESHAPE && s->src[0] != nullptr) {
            const ggml_tensor* p = s->src[0];
            // parent must be a contiguous [n_embd=S*H, T] tensor (n_embd = s->ne[0]*s->ne[1]).
            if (ggml_is_contiguous(p) && p->ne[2] == 1 && p->ne[3] == 1
                && p->ne[0] == s->ne[0] * s->ne[1] && p->ne[1] == s->ne[2]) {
                return realize_ggml_view(p);
            }
        }
        return realize_ggml_view(s);
    };

    // ggml src order: r,w,k,v,a,b,state -> realize each to its ggml-native device tensor.
    auto res = ttggml::rwkv_wkv7(
        *realize_wkv7_input(0),            // r
        *realize_wkv7_input(1),            // w
        *realize_wkv7_input(2),            // k
        *realize_wkv7_input(3),            // v
        *realize_ggml_view(dst->src[4]),   // a
        *realize_ggml_view(dst->src[5]),   // b
        *state);                           // state

    *dst_meta = {
        .tensor = std::make_shared<tt::tt_metal::Tensor>(std::move(res)),
    };
}

static bool ggml_backend_metalium_can_rwkv_wkv7(const struct ggml_tensor * dst)
{
    for(int i = 0; i < 7; i++) {
        if(dst->src[i] == NULL) {
            return false;
        }
    }
    const struct ggml_tensor * k     = dst->src[2];
    const struct ggml_tensor * state = dst->src[6];
    const int64_t S = k->ne[0];           // head_size
    const int64_t H = k->ne[1];           // head_count
    const int64_t L = k->ne[2];           // n_seq_tokens
    const int64_t G = state->ne[1];       // n_seqs

    // The kernel is validated for head_size 64; needs head_count | head_size (region-2
    // scatter). Arbitrary n_seq_tokens is supported (the reader neutral-pads the partial
    // last chunk on-device) AND arbitrary n_seqs is supported: the state-seed gather pages
    // across 32-row flat-strip blocks via (sq/32)*tpr, so G is not bounded to one row-tile
    // (verified PASS at G=64/128 for chunked + decodeL + decode). chunked (L>=2) still needs
    // (G*H) even for its NB=2 group (always true for even H; only bites odd head_count).
    if(S != 64 || H <= 0 || (S % H) != 0) {
        return false;
    }
    if(G < 1) {
        return false;
    }
    if(L >= 2 && ((G * H) % 2) != 0) {
        return false;
    }
    return true;
}

static bool ggml_backend_metalium_can_flash_attn(const struct ggml_tensor * dst)
{
    if(!g_debug_flags.experimental_ops) {
        return false;
    }
    auto follow_tensor_upstream = [](const ggml_tensor* tensor) -> const ggml_tensor* {
        if(!tensor) {
            return NULL;
        }
        while(tensor->op == GGML_OP_TRANSPOSE || tensor->op == GGML_OP_PERMUTE) {
            tensor = tensor->src[0];
        }
        return tensor;
    };
    const ggml_tensor* q = follow_tensor_upstream(dst->src[0]);
    const ggml_tensor* k = follow_tensor_upstream(dst->src[1]);
    const ggml_tensor* v = follow_tensor_upstream(dst->src[2]);
    const ggml_tensor* mask = follow_tensor_upstream(dst->src[3]);

    std::array<float, 3> params;
    memcpy(params.data(), dst->op_params, sizeof(float) * 3);
    auto [scale, max_bias, logit_softcap] = params;

    if(max_bias != 0.f) {  // ALiBi: ttnn SDPA has no per-head slope bias
        return false;
    }
    if(logit_softcap != 0.f) {  // tanh logit gating (Gemma2): unsupported by ttnn SDPA
        return false;
    }

    // Examoke input
    // REJECT op FLASH_ATTN_EXT (__fattn__-6)
    //   src0 shape [64 32 32 1], dtype = f32, name = 'Qcur-6 (view) (permuted)'
    //   src1 shape [64 1536 8 1], dtype = f16, name = 'cache_k_l6 (view) (permuted)'
    //   src2 shape [64 1536 8 1], dtype = f16, name = 'cache_v_l6 (view) (permuted)'
    //   src3 shape [1536 64 1 1], dtype = f16, name = ' (copy)'
    //   FlashAttention debug details:
    //     src0 follow - query shape [64 32 32 1], dtype = f32, name = 'Qcur-6 (view)'
    //     src1 follow - key shape [64 8 1536 1], dtype = f16, name = 'cache_k_l6 (view)'
    //     src2 follow - value shape [64 8 1536 1], dtype = f16, name = 'cache_v_l6 (view)'
    //     src3 follow - mask shape [1536 64 1 1], dtype = f16, name = ' (copy)'
    //
    // GGML:
    // q:    [n_embd_k, n_batch,     n_head,    ne3 ]
    // k:    [n_embd_k, n_kv,        n_head_kv, ne3 ]
    // v:    [n_embd_v, n_kv,        n_head_kv, ne3 ] !! not transposed !!
    // mask: [n_kv,     n_batch_pad, ne32,      ne33] !! n_batch_pad = GGML_PAD(n_batch, GGML_KQ_MASK_PAD) !!
    // res:  [n_embd_v, n_head,      n_batch,   ne3 ] !! permuted !!
    //
    // TT
    // input_tensor_q (ttnn.Tensor): the input tensor [1 x b x nh x dh]
    // input_tensor_k (ttnn.Tensor): the input tensor [b x nkv x   s x dh]
    // input_tensor_v (ttnn.Tensor): the input tensor [b x nkv x   s x dh]

    int64_t ne3 = q->ne[3];
    if(ne3 != 1) {
        return false;
    }
    if(k->ne[3] != 1 || v->ne[3] != 1) {
        return false;
    }
    if(k->ne[1] < 32 || v->ne[1] < 32 || k->ne[1] % 32 != 0 || v->ne[1] % 32 != 0) {
        return false;
    }
    // ttnn SDPA forbids padding on the head_dim (last ttnn dim == ggml ne[0]).
    // A non-tile head dim (e.g. 40/72/80) gets padded up to a TILE multiple when
    // realized, which trips "Padding is not supported on the head_dim dimension".
    // Restrict to tile-aligned head dims (models here use 64/128).
    if(q->ne[0] % 32 != 0 || k->ne[0] % 32 != 0 || v->ne[0] % 32 != 0) {
        return false;
    }
    // EXPERIMENT (prefill SDPA): attention sinks (src[4]) are not forwarded; reject.
    if(dst->src[4]) {
        return false;
    }
    // EXPERIMENT: trimming the padded mask requires a TILE-aligned Q seq length,
    // otherwise the mask slice faults on device. Restrict to Sq % 32 == 0 for now.
    if(q->ne[1] % 32 != 0) {
        return false;
    }
    int64_t b = q->ne[1];
    // Either we don't need to broadcast or we broadcast for them
    if(mask && mask->ne[2] != 1 && !(mask->ne[3] == 1 || mask->ne[3] == b)) {
        return false;
    }
    // The op does not support broadcasting mask. mask->ne[1] (Sq rows) may be
    // padded up (GGML_KQ_MASK_PAD) above q->ne[1]; the dispatch trims it, so only
    // reject if it's too small or the kv width mismatches.
    if(mask && (mask->ne[0] != k->ne[1] || mask->ne[1] < q->ne[1])) {
        return false;
    }
    // ttnn SDPA GQA: Q heads must be a multiple of KV heads.
    if(q->ne[2] % k->ne[2] != 0) {
        return false;
    }

    return true;
}

static void ggml_backend_metalium_flash_attn(ggml_backend_metalium_context * ctx, struct ggml_tensor * dst)
{
    GGML_METALIUM_OP_SANITY_CHECK(dst);
    GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);
    GGML_UNUSED(ctx);
    ggml_tensor_extra_metalium* dst_meta = (ggml_tensor_extra_metalium*)dst->extra;

    auto follow_tensor_upstream = [](const ggml_tensor* tensor) -> const ggml_tensor* {
        if(!tensor) {
            return NULL;
        }
        while(tensor->op == GGML_OP_TRANSPOSE || tensor->op == GGML_OP_PERMUTE) {
            tensor = tensor->src[0];
        }
        return tensor;
    };
    const ggml_tensor* q = follow_tensor_upstream(dst->src[0]);
    const ggml_tensor* k = follow_tensor_upstream(dst->src[1]);
    const ggml_tensor* v = follow_tensor_upstream(dst->src[2]);
    const ggml_tensor* mask = follow_tensor_upstream(dst->src[3]);

    std::array<float, 3> params;
    memcpy(params.data(), dst->op_params, sizeof(float) * 3);
    auto [scale, max_bias, logit_softcap] = params;

    GGML_UNUSED(max_bias);
    GGML_UNUSED(logit_softcap);

    // ggml's FLASH_ATTN_EXT inputs map 1:1 onto ttnn's prefill
    // scaled_dot_product_attention (ggml ne[] is the reverse of ttnn's shape):
    //   q ne[Dk, Nq, n_head,    ne3] -> qt [B, n_head,    Sq, Dh]
    //   k ne[Dk, Nkv, n_head_kv, ne3] -> kt [B, n_head_kv, Sk, Dh]
    //   v ne[Dv, Nkv, n_head_kv, ne3] -> vt [B, n_head_kv, Sk, Dv]   (ggml V not transposed -> matches)
    //   mask ne[Nkv, Nq_pad, m2, m3] -> mt [Bm, NHm, Sq_pad, Sk]
    // realize_ggml_view already hands back BFLOAT16 tensors (f32/f16 -> bf16),
    // which is exactly what SDPA accepts. GQA (n_head % n_head_kv == 0) is handled
    // natively by SDPA, so K/V are passed without manual head broadcasting.
    auto qt = *realize_ggml_view(q);
    auto kt = *realize_ggml_view(k);
    auto vt = *realize_ggml_view(v);

    const uint32_t Sq = qt.logical_shape()[2];

    std::optional<ttnn::Tensor> mask_tensor;
    if(mask) {
        auto mt = *realize_ggml_view(mask);
        // ggml pads the mask query-rows to GGML_KQ_MASK_PAD; the prefill op
        // requires mask Sq == Q Sq, so trim the padded rows.
        const auto ms = mt.logical_shape();
        if((uint32_t)ms[2] != Sq) {
            mt = ttnn::slice(mt,
                ttnn::SmallVector<uint32_t>{0u, 0u, 0u, 0u},
                ttnn::SmallVector<uint32_t>{(uint32_t)ms[0], (uint32_t)ms[1], Sq, (uint32_t)ms[3]},
                ttnn::SmallVector<uint32_t>{1u, 1u, 1u, 1u});
        }
        mask_tensor = mt;
    }

    // ggml always supplies an additive mask with causality baked in -> use the
    // mask path with is_causal=false.
    // NOTE: SDPA runs in bf16, landing at NMSE ~6e-4 (q4_0 KV up to ~3.5e-3), over
    // ggml's strict 5e-4 FA bound. A WormholeComputeKernelConfig{fp32_dest_acc_en=
    // true, HiFi4} passed as compute_kernel_config would tighten it at a perf cost.
    auto res = ttnn::transformer::scaled_dot_product_attention(
        qt,
        kt,
        vt,
        mask_tensor,
        /*is_causal=*/false,
        scale);

    // SDPA returns [B, n_head, Sq, Dv]. ggml's FLASH_ATTN_EXT output is the
    // *permuted* layout ne[Dv, n_head, Nq, ne3] -> TTNN order [B, Sq, n_head, Dv],
    // so swap the n_head and Sq axes (dims 1 and 2) to match.
    res = ttnn::transpose(res, 1, 2);

    *dst_meta = {
        .tensor = std::make_shared<ttnn::Tensor>(res)
    };
}

// backend interface

static const char * ggml_backend_metalium_name(ggml_backend_t backend) {
    return "Metalium";

    GGML_UNUSED(backend);
}

static void ggml_backend_metalium_free(ggml_backend_t backend) {
    ggml_backend_metalium_context * ctx = (ggml_backend_metalium_context *)backend->context;
    // Release captured traces HERE (backend free runs while the program is alive), NOT at
    // atexit. Trace buffer deallocation calls GraphTracker::track_deallocate, which reads a
    // thread_local that is already destroyed during static teardown -> UAF segfault.
    // It's dumb
    if(g_metalium_trace_enabled && ctx->device != nullptr) {
        metalium_trace_release_all(ctx->device->get_mesh_device().get());
    }
    delete ctx;
    delete backend;
}

struct ggml_backend_metalium_buffer_type_context {
    std::shared_ptr<ttnn::MeshDevice> device = nullptr;
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
    return ctx->device->num_dram_channels() * (size_t)ctx->device->dram_size_per_channel();
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

// Row-folded host contract. A folded cache is stored as [1, n_rows, dim/32, 32] (TILE), decoupled
// from GGML's declared [dim, n_rows]. GGML still addresses it in canonical row-major bytes, but the
// recurrent save/restore path only ever touches WHOLE rows (offset/size are multiples of one row),
// so the host<->device bridge is just a row-range slice plus the free row-major reshape
// [cells, dim] <-> [cells, dim/32, 32]. No fold kernel needed: the byte order is identical.

static void ggml_backend_metalium_set_tensor_folded(ggml_backend_metalium_buffer_context * bufctx,
                                                ggml_tensor *tensor, ggml_tensor_extra_metalium * meta,
                                                const void *data, size_t offset, size_t size)
{
    const ggml_type ggtype = tensor->type;
    const uint32_t  dim    = (uint32_t)tensor->ne[0];
    const size_t    row_bytes = (size_t)dim * ggml_type_size(ggtype);
    // Proven contract: recurrent save/restore writes whole rows only (see ROW_FOLD_PLAN.md).
    GGML_ASSERT(row_bytes > 0 && offset % row_bytes == 0 && size % row_bytes == 0
        && "folded cache set_tensor must be whole-row aligned");
    GGML_ASSERT(dim % 32 == 0 && "folded cache inner dim must be tile aligned");
    const uint32_t head  = (uint32_t)(offset / row_bytes);
    const uint32_t cells = (uint32_t)(size   / row_bytes);
    if(cells == 0) {
        return;
    }

    // Host [cells, dim] bytes are bit-identical to [1, cells, dim/32, 32] row-major (free reshape).
    std::optional<tt::tt_metal::HostBuffer> storage;
    const size_t n_elems = (size_t)cells * dim;
    if(ggtype == GGML_TYPE_F32) {
        storage = host_data_to_tt_host_buffer<float, bfloat16>((const float*)data, n_elems);
    }
    else if(ggtype == GGML_TYPE_F16) {
        storage = host_data_to_tt_host_buffer<ggml_fp16_t, bfloat16>((const ggml_fp16_t*)data, n_elems);
    }
    else if(ggtype == GGML_TYPE_BF16) {
        storage = host_data_to_tt_host_buffer<ggml_bf16_t, bfloat16>((const ggml_bf16_t*)data, n_elems);
    }
    else {
        GGML_ASSERT(false && "Unsupported folded cache data type");
    }

    tt::tt_metal::Tensor chunk(std::move(*storage), ttnn::Shape({1, cells, dim / 32, 32}),
        tt::tt_metal::DataType::BFLOAT16, tt::tt_metal::Layout::ROW_MAJOR);
    const tt::tt_metal::DataType final_type = ggml2tt_type(ggtype, bufctx->device->arch());
    chunk = ttnn::tilize_with_zero_padding(chunk.to_device(bufctx->device.get()), std::nullopt, final_type);

    // Scatter the row range [head, head+cells) into the folded store along the un-tiled outer dim.
    const uint32_t dim_t = dim / 32;
    ttnn::SmallVector<uint32_t> begins{0, head, 0, 0};
    ttnn::SmallVector<uint32_t> ends{1, head + cells, dim_t, 32};
    ttnn::SmallVector<uint32_t> step{1, 1, 1, 1};
    ttnn::experimental::slice_write(chunk, *meta->row_folded, begins, ends, step);
}

static void ggml_backend_metalium_get_tensor_folded(const ggml_tensor *tensor, ggml_tensor_extra_metalium * meta,
                                                void *data, size_t offset, size_t size)
{
    const ggml_type ggtype = tensor->type;
    const uint32_t  dim    = (uint32_t)tensor->ne[0];
    const size_t    row_bytes = (size_t)dim * ggml_type_size(ggtype);
    GGML_ASSERT(row_bytes > 0 && offset % row_bytes == 0 && size % row_bytes == 0
        && "folded cache get_tensor must be whole-row aligned");
    GGML_ASSERT(dim % 32 == 0 && "folded cache inner dim must be tile aligned");
    const uint32_t head  = (uint32_t)(offset / row_bytes);
    const uint32_t cells = (uint32_t)(size   / row_bytes);
    if(cells == 0) {
        return;
    }

    const uint32_t dim_t = dim / 32;
    ttnn::SmallVector<uint32_t> begins{0, head, 0, 0};
    ttnn::SmallVector<uint32_t> ends{1, head + cells, dim_t, 32};
    ttnn::SmallVector<uint32_t> step{1, 1, 1, 1};
    auto sliced = ttnn::slice(*meta->row_folded, begins, ends, step);
    // Logical [1, cells, dim/32, 32] row-major order is exactly canonical [cells, dim].
    copy_tt_tensor_to_host_pointer<bfloat16>(sliced, data, ggtype);
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
    // TODO: Make a scalable way to decide which GGML type casts to TT quantized types
    GGML_ASSERT(tensor->extra != NULL);

    ggml_backend_metalium_buffer_context * bufctx = (ggml_backend_metalium_buffer_context *)buffer->context;
    GGML_ASSERT(bufctx != NULL);
    ggml_type ggtype = tensor->type;
    ggml_tensor_extra_metalium * meta = (ggml_tensor_extra_metalium *)tensor->extra;

    // Row-folded store owns its own whole-row scatter path (supports nonzero offsets).
    if(meta->is_row_folded()) {
        ggml_backend_metalium_set_tensor_folded(bufctx, tensor, meta, data, offset, size);
        return;
    }
    GGML_ASSERT(offset == 0);

    // Make sure we are not writing to a view tensor
    if(size != ggml_nbytes(tensor) || (meta->tensor && ggml_tt_tensors_shape_equal(tensor, *meta->tensor) == false)
        || tensor->view_src != NULL) {
        // FIXME: Reenable this when got time
        // fprintf(stderr, "Warning: Metalium set_tensor() does not work with tensor views\n");
        return;
    }

    std::optional<tt::tt_metal::HostBuffer> storage;
    tt::tt_metal::DataType intermidiate_type = tt::tt_metal::DataType::BFLOAT16;
    bool tilize = true;
    if(ggtype == GGML_TYPE_F32) {
        // For now we cast F32 to BF16. Need a scalable way to handle this as WORMHOLD_B0 have native support for F32
        // TODO: Enable proper FP32 when all related bugs gets fixed for devices that support it
        storage = host_data_to_tt_host_buffer<float, bfloat16>((const float*)data, size / sizeof(float));
    }
    else if (ggtype == GGML_TYPE_F16) {
        // TT hardware claims to support FP16 but the API does not expose it. For now we use BF16 as it is close enough
        storage = host_data_to_tt_host_buffer<ggml_fp16_t, bfloat16>((const ggml_fp16_t*)data, size / sizeof(ggml_fp16_t));
    }
    else if (ggtype == GGML_TYPE_BF16) {
        storage = host_data_to_tt_host_buffer<ggml_bf16_t, bfloat16>((const ggml_bf16_t*)data, size / sizeof(ggml_bf16_t));
    }
    else if (ggtype == GGML_TYPE_I32) {
        storage = host_data_to_tt_host_buffer<int, uint32_t>((const int*)data, size / sizeof(int));
        intermidiate_type = tt::tt_metal::DataType::UINT32;
        tilize = false; // Integer tensors are indices - operations will want them untiled
    }
    else if (ggml_is_quantized(ggtype)) {
        // Even though in theory transfering BFP16 to device uses much less bandwidth then FP32. GGML nativly have support
        // converting quantized types into FP32. Converting to BFP16 would be an extra step making everything slower
        storage = quantized_ggml_data_to_tt_host_buffer<float>(data, tensor);
        intermidiate_type = tt::tt_metal::DataType::FLOAT32;
    }
    else {
        fmt::println(stderr, "Unsupported data type while uploading to device: {}, name '{}', op type: {}\n", ggml_type_name(ggtype), tensor->name, ggml_op_name(tensor->op));
        GGML_ASSERT(false && "Unsupported data type while uploading to device");
    }
    GGML_ASSERT(storage.has_value() && "Failed to convert data to TT storage");

    // Convert GGML shape to TT shape
    ttsl::SmallVector<uint32_t> shape(GGML_MAX_DIMS, 1);
    for(int i = 0; i < GGML_MAX_DIMS; i++) {
        // GGML stores the shape in reverse order
        shape[i] = tensor->ne[GGML_MAX_DIMS - i - 1];
    }

    std::optional<ttsl::SmallVector<int64_t>> permute;
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
        ttsl::SmallVector<int64_t> perm(GGML_MAX_DIMS, -1);
        for(int i = 0; i < GGML_MAX_DIMS; i++) {
            perm[strides[i].second] = i;
        }
        permute = perm;
    }

    tt::tt_metal::Tensor t(std::move(*storage), ttnn::Shape(shape)
        , intermidiate_type, tt::tt_metal::Layout::ROW_MAJOR);

    tt::tt_metal::DataType final_type = ggml2tt_type(ggtype, bufctx->device->arch());
    if(tilize) {
        t = ttnn::tilize_with_zero_padding(t.to_device(bufctx->device.get()), std::nullopt, final_type);
        if(permute.has_value()) {
            t = ttnn::permute(t, *permute);
        }
    }
    else {
        t = t.to_device(bufctx->device.get());
        GGML_ASSERT(t.dtype() == final_type && "Tensor dtype mismatch during tensor creation for row major tensors");
        GGML_ASSERT(!permute.has_value() && "Cannot permute tensor without tilizing");
    }

    GGML_ASSERT(t.storage_type() == tt::tt_metal::StorageType::DEVICE);
    GGML_ASSERT(t.dtype() == final_type);
    GGML_ASSERT(ggml_tt_tensors_shape_equal(tensor, t));
    GGML_ASSERT(t.layout() == (tilize ? tt::tt_metal::Layout::TILE : tt::tt_metal::Layout::ROW_MAJOR));
    ggml_metalium_store_tensor(meta, std::move(t));
}

static void ggml_backend_metalium_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                const ggml_tensor *tensor,
                                                void *data, size_t offset,
                                                size_t size)
{
    GGML_UNUSED(buffer);
    // Here's the general logic of get_tensor
    // 1. Get the TT tensor from the metadata
    // 2. If the TT tensor is quantized, cast it to BFLOAT16
    // 3. Call copy_tt_tensor_to_host_pointer to convert the TT tensor to GGML tensor
    //    - copy_tt_tensor_to_host_pointer internally handles the data type conversion
    GGML_ASSERT(tensor->extra != NULL);

    // Row-folded store owns its own whole-row gather path (supports nonzero offsets / sub-ranges).
    {
        ggml_tensor_extra_metalium * meta = (ggml_tensor_extra_metalium *)tensor->extra;
        if(meta->is_row_folded()) {
            ggml_backend_metalium_get_tensor_folded(tensor, meta, data, offset, size);
            return;
        }
    }
    GGML_ASSERT(size == ggml_nbytes(tensor));
    GGML_UNUSED(offset);

    // ggml_backend_metalium_buffer_context * ctx = (ggml_backend_metalium_buffer_context *)buffer->context;

    ggml_type dst_ggtype = tensor->type;

    // auto *meta = (ggml_tensor_extra_metalium*)tensor->extra;
    // auto shape = meta->tensor->logical_shape();
    // std::cout << "get_tensor():\n";
    // std::cout << "  GGML thinks shape: " << tensor->ne[0] << " " << tensor->ne[1] << " " << tensor->ne[2] << " " << tensor->ne[3] << std::endl;
    // std::cout << "  TTNN thinks shape: " << shape << std::endl;
    std::shared_ptr<tt::tt_metal::Tensor> t;
    if(tensor->op == GGML_OP_TRANSPOSE) {
        // std::cout << "Reading out to transpose tensor" << std::endl;
        // HACK: Yeah this one is stupid. GGML as a row-major framework uses lazy evaluation for transpose.
        //      Which means if we try to copy a transposed tensor. We should not transpose it. Else the other
        //      backend would transpose it again.
        ggml_tensor* src = tensor->src[0];
        bool do_transpose = true;
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
    if(t->dtype() != tt::tt_metal::DataType::BFLOAT16 && t->dtype() != tt::tt_metal::DataType::FLOAT32 && t->dtype() != tt::tt_metal::DataType::UINT32) {
        t = std::make_shared<tt::tt_metal::Tensor>(ttnn::typecast(*t, tt::tt_metal::DataType::BFLOAT16));
    }

    // TODO: Proper handling of data types
    GGML_ASSERT(dst_ggtype != GGML_TYPE_F64 && dst_ggtype != GGML_TYPE_I16 && dst_ggtype != GGML_TYPE_I8);
    switch(t->dtype()) {
        case tt::tt_metal::DataType::BFLOAT16:
            copy_tt_tensor_to_host_pointer<bfloat16>(*t, (float*)data, dst_ggtype);
            break;
        case tt::tt_metal::DataType::FLOAT32:
            copy_tt_tensor_to_host_pointer<float>(*t, (float*)data, dst_ggtype);
            break;
        case tt::tt_metal::DataType::UINT32:
            copy_tt_tensor_to_host_pointer<uint32_t>(*t, (int*)data, dst_ggtype);
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

static enum ggml_status
ggml_backend_metalium_buffer_init_tensor(ggml_backend_buffer_t buffer,
                                     ggml_tensor *tensor)
{
    ggml_backend_metalium_buffer_context * bufctx = (ggml_backend_metalium_buffer_context *)buffer->context;

    bufctx->metadata_to_free.push_back(std::make_unique<ggml_tensor_extra_metalium>(ggml_tensor_extra_metalium{
        .tensor = nullptr,
    }));
    ggml_tensor_extra_metalium* meta = bufctx->metadata_to_free.back().get();
    tensor->extra = meta;

    // HACK: Make KV cache work. They don't get set before first use
    // TODO: Most likely we'd want to refer this allocation to first time use of the tensor to support proper KV cache setup
    //       as the "real" shape information (GGML allocates KV cache as a very long 1D tensor) is missing here
    std::string_view name(tensor->name);
    // RWKV recurrent state hacks to improve DRAM compatness in Metalim. Folding converts tensor [1, 1, nrows, nembed]
    // into [1, nrows, nembed/32, 32] (TILE) for storage. Because Metalium runs on tiles, not doing so leads to large
    // bandwidth waste always reading the full 32x32 tile even just needing one row.
    const bool is_rwkv_cache = name.rfind("cache_r_l", 0) == 0 || name.rfind("cache_s_l", 0) == 0;
    if(is_rwkv_cache && tensor->op == GGML_OP_NONE) {
        GGML_ASSERT(ggml_n_dims(tensor) <= 2 && "row-fold only supports <=2D caches");
        const uint32_t dim    = (uint32_t)tensor->ne[0];
        const uint32_t n_rows = (uint32_t)tensor->ne[1];
        GGML_ASSERT(dim % 32 == 0 && "row-fold requires a tile-aligned inner dim");
        auto z = ttnn::zeros(ttnn::Shape({1, n_rows, dim / 32, 32}),
            ggml2tt_type(tensor->type, bufctx->device->arch()), tt::tt_metal::Layout::ROW_MAJOR);
        z = ttnn::tilize_with_zero_padding(z.to_device(bufctx->device.get()));
        meta->row_folded = std::make_shared<tt::tt_metal::Tensor>(std::move(z));
    }
    else if(std::string_view(name).find("cache") != std::string::npos && tensor->op == GGML_OP_NONE) {
        std::vector<uint32_t> shape(tensor->ne, tensor->ne + GGML_MAX_DIMS);
        std::reverse(shape.begin(), shape.end());
        auto t = ttnn::zeros(ttnn::Shape(shape), ggml2tt_type(tensor->type, bufctx->device->arch()), tt::tt_metal::Layout::ROW_MAJOR);
        t = ttnn::tilize_with_zero_padding(t.to_device(bufctx->device.get()));
        meta->tensor = std::make_shared<tt::tt_metal::Tensor>(std::move(t));
    }
    // std::cout << "Creating tensor with address: " << tensor->data << ", shape = " << tensor->ne[0] << " " << tensor->ne[1] << " " << tensor->ne[2] << " " << tensor->ne[3] << ", name " << tensor->name << std::endl;
    GGML_UNUSED(buffer);
    return GGML_STATUS_SUCCESS;
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

    ggml_tensor_extra_metalium * src_meta = (ggml_tensor_extra_metalium *)src->extra;
    ggml_tensor_extra_metalium * dst_meta = (ggml_tensor_extra_metalium *)dst->extra;

    tt::tt_metal::Tensor& src_tensor = *src_meta->tensor;

    tt::tt_metal::Tensor ret = ttnn::identity(src_tensor);
    GGML_ASSERT(ret.storage_type() == tt::tt_metal::StorageType::DEVICE);
    dst_meta->tensor = std::make_shared<tt::tt_metal::Tensor>(std::move(ret));
    return true;
}

static void ggml_backend_metalium_buffer_reset(ggml_backend_buffer_t buffer) {
    ggml_backend_metalium_buffer_context * bufctx = (ggml_backend_metalium_buffer_context *)buffer->context;
    bufctx->metadata_to_free.clear();
}

static struct ggml_backend_buffer_i ggml_backend_metalium_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_metalium_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_metalium_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_metalium_buffer_init_tensor,
    /* .memset_tensor   = */ nullptr,
    /* .set_tensor      = */ ggml_backend_metalium_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_metalium_buffer_get_tensor,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ ggml_backend_metalium_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_metalium_buffer_clear,
    /* .reset           = */ ggml_backend_metalium_buffer_reset,
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

    static std::map<int, ggml_backend_buffer_type> buffer_type_map;
    static std::set<std::unique_ptr<ggml_backend_metalium_buffer_type_context>> buffer_type_context_deleter;
    auto it = buffer_type_map.find(device_id);
    if(it != buffer_type_map.end()) {
        return &it->second;
    }

    auto bufctx = std::make_unique<ggml_backend_metalium_buffer_type_context>(
        ggml_backend_metalium_buffer_type_context{
            .device = dev_ctx->device,
            .name = "Metalium " + std::to_string(device_id),
        });
    auto* bufctx_ptr = bufctx.get();
    buffer_type_context_deleter.insert(std::move(bufctx));

    // TODO: Make sure the device_id we got is valid
    buffer_type_map[device_id] = {
        /* .iface    = */ ggml_backend_metalium_buffer_type_interface,
        /* .device   = */ regctx->devices[0],
        /* .context  = */ bufctx_ptr,
    };
    return &buffer_type_map[device_id];
}

// Persistent capture/replay state for one graph signature, held for the process lifetime.
// @note Lifecycle across passes of the same signature: pass 1 runs eagerly (warmup: JIT compile,
//   weight pre-transpose, allocs); pass 2 captures then executes once (capture only RECORDS the
//   command stream, so an execute is needed to populate outputs); pass 3+ replay.
struct metalium_trace_exec_state {
    uint64_t passes = 0;                        ///< Number of times this signature has been seen.
    bool captured = false;                      ///< Whether the trace has been captured (=> replay).
    ttnn::MeshTraceId tid = ttnn::MeshTraceId{0}; ///< Handle to the captured trace on the device.
    // Captured per-node device tensors, by node index.
    // @note The backend allocates a fresh node->extra every pass, so replay must re-bind these;
    //   see metalium_trace_dispatch::rebind_nodes / snapshot_nodes.
    std::vector<std::shared_ptr<tt::tt_metal::Tensor>> node_tensors;

    // Lazily-memoized replay plan
    bool replay_plan_built = false;
    std::vector<std::pair<ggml_tensor*, std::shared_ptr<tt::tt_metal::Tensor>>> pin_plan;  // input node + baked anchor
    std::vector<std::pair<int, std::shared_ptr<tt::tt_metal::Tensor>>> rebind_plan;        // boundary node idx + tensor
};
static std::map<uint64_t, metalium_trace_exec_state>& metalium_trace_exec_states() {
    static std::map<uint64_t, metalium_trace_exec_state> m;
    return m;
}


// Tracking for tracing support

struct GGMLTensorMeta
{
    void* data;
    ggml_type type;
    int64_t ne[GGML_MAX_DIMS];
    char name[GGML_MAX_NAME];

    GGMLTensorMeta(const ggml_tensor* t)
        : data(t->data), type(t->type) {
        memcpy(ne, t->ne, sizeof(ne));
        strncpy(name, t->name, GGML_MAX_NAME);
    }
};

struct GGMLTensorHasher
{
    size_t operator() (const GGMLTensorMeta& t) const
    {
        size_t h = 1469598103934665603ull;
        auto mix = [&](uint64_t v){ h ^= v; h *= 1099511628211ull; };
        mix((uint64_t)(uintptr_t)t.data);
        mix((uint64_t)t.type);
        for(int d = 0; d < GGML_MAX_DIMS; d++) mix((uint64_t)t.ne[d]);
        h ^= std::hash<std::string_view>()(std::string_view(t.name));
        return h;
    }
};

struct GGMLTensorEqual
{
    bool operator() (const GGMLTensorMeta& lhs, const GGMLTensorMeta& rhs) const
    {
        return lhs.data == rhs.data && lhs.type == rhs.type
            && memcmp(lhs.ne, rhs.ne, sizeof(lhs.ne)) == 0
            && std::string_view(lhs.name) == std::string_view(rhs.name);
    }
};

static std::unordered_map<GGMLTensorMeta, std::shared_ptr<tt::tt_metal::Tensor>, GGMLTensorHasher, GGMLTensorEqual> g_metalium_pinned_tensors;

static void metalium_trace_release_all(ttnn::MeshDevice* mesh) {
    auto& states = metalium_trace_exec_states();
    for(auto& kv : states) {
        if(kv.second.captured) {
            try { ttnn::operations::trace::release_trace(mesh, kv.second.tid); }
            catch(...) {}
        }
    }
    states.clear();
    g_metalium_pinned_tensors.clear();
}

static void ggml_metalium_trace_mem_report(ttnn::MeshDevice* mesh, const char* tag) {
    if(!g_debug_flags.print_trace_mem) return;

    std::unordered_set<uint64_t> seen;
    auto buf_bytes = [&](const std::shared_ptr<tt::tt_metal::Tensor>& t) -> uint64_t {
        if(!t || t->storage_type() != tt::tt_metal::StorageType::DEVICE) return 0;
        try {
            uint64_t addr = (uint64_t)t->buffer()->address();
            if(!seen.insert(addr).second) return 0; // already counted this buffer
            return (uint64_t)t->buffer()->size();
        } catch(...) { return 0; }
    };

    auto& states = metalium_trace_exec_states();
    size_t n_traces = 0, n_nodes_pinned = 0;
    uint64_t node_bytes = 0;
    for(auto& kv : states) {
        if(!kv.second.captured) continue;
        n_traces++;
        for(auto& nt : kv.second.node_tensors) {
            if(nt) { n_nodes_pinned++; node_bytes += buf_bytes(nt); }
        }
    }
    uint64_t io_bytes = 0;
    for(auto& kv : g_metalium_pinned_tensors) io_bytes += buf_bytes(kv.second);

    uint64_t dram_alloc = 0, dram_free = 0;
    try {
        auto s = mesh->allocator()->get_statistics(tt::tt_metal::BufferType::DRAM);
        dram_alloc = s.total_allocated_bytes;
        dram_free  = s.total_free_bytes;
    } catch(...) {}

    const double MB = 1024.0 * 1024.0;
    fprintf(stderr,
        "[trace-mem %-10s] DRAM alloc=%.1fMB free=%.1fMB | traces=%zu | "
        "pinned-intermediates: nodes=%zu bytes=%.1fMB | pinned-IO: bytes=%.1fMB\n",
        tag, dram_alloc / MB, dram_free / MB, n_traces,
        n_nodes_pinned, node_bytes / MB, io_bytes / MB);
}

static uint64_t metalium_trace_graph_signature(const ggml_cgraph* g) {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](uint64_t v){ h ^= v; h *= 1099511628211ull; };
    mix((uint64_t)g->n_nodes);
    // XXX: Slow but at least guarentees a unique signature per graph
    for(int i = 0; i < g->n_nodes; i++) {
        const ggml_tensor* n = g->nodes[i];
        mix((uint64_t)n->op);
        mix(GGMLTensorHasher()(GGMLTensorMeta(n)));
    }
    return h;
}

// Stable identity for a cgraph: uid if present, else the (slow) topology signature.
uint64_t metalium_graph_key(const ggml_cgraph* g) {
    return g->uid != 0 ? g->uid : metalium_trace_graph_signature(g);
}

/// TTNN capture/replay for a single graph_compute call
struct metalium_trace_dispatch {
    ggml_cgraph* graph = nullptr;
    ttnn::MeshDevice* mesh = nullptr;
    metalium_trace_exec_state* state = nullptr;
    bool capturing = false;

    // Re-attach the captured per-node device tensors onto this pass's nodes, by node index.
    // @note ggml hands every node a fresh null-bound extra each pass and only the node loop binds
    //   it; replay skips that loop, so without this re-bind a reader hits a null binding. Indexing by
    //   position is valid because an identical signature implies identical topology.
    void rebind_nodes() const {
        const auto& nt = state->node_tensors;
        if((int)nt.size() != graph->n_nodes) return;
        for(int i = 0; i < graph->n_nodes; i++) {
            ggml_tensor* n = graph->nodes[i];
            if(n->extra != nullptr && nt[i] != nullptr) {
                ((ggml_tensor_extra_metalium*)n->extra)->tensor = nt[i];
            }
        }
    }

    // Snapshot every node's device tensor at capture so replay can re-bind them.
    // @note Holding the shared_ptrs also pins the buffers whose addresses the trace baked, for the
    //   process lifetime.
    void snapshot_nodes() const {
        state->node_tensors.assign(graph->n_nodes, nullptr);
        for(int i = 0; i < graph->n_nodes; i++) {
            ggml_tensor* n = graph->nodes[i];
            if(n->extra == nullptr) continue;

            // only pin IO boundry nodes
            if((n->flags & (GGML_TENSOR_FLAG_INPUT | GGML_TENSOR_FLAG_OUTPUT)) == 0) continue;
            state->node_tensors[i] = ((ggml_tensor_extra_metalium*)n->extra)->tensor;
        }
    }

    // Copy t's freshly-fed device tensor into its baked anchor so the trace reads it at the stable
    // address. Pointer-equal (already there) or incompatible -> no-op. No map lookup: the anchor is
    // supplied by the caller (the cached plan), so this is the per-replay hot path.
    static void apply_pin(ggml_tensor* t, const std::shared_ptr<tt::tt_metal::Tensor>& anchor) {
        if(t->extra == nullptr) return;
        auto* m = (ggml_tensor_extra_metalium*)t->extra;
        if(m->tensor == nullptr || m->tensor->storage_type() != tt::tt_metal::StorageType::DEVICE) return;
        if(m->tensor.get() == anchor.get()) return; // already at the fixed address (also dedups repeats)
        if(anchor->storage_type() != tt::tt_metal::StorageType::DEVICE
            || m->tensor->dtype()  != anchor->dtype()
            || m->tensor->layout() != anchor->layout()
            || !ggml_tt_tensors_shape_equal(t, *anchor)) {
            return; // incompatible -> leave untouched
        }
        ttnn::copy(*m->tensor, *anchor); // actual copy of device memory
        m->tensor = anchor;
    }

    // Pin one op == NONE external input to its anchor. Metal Trace bakes the device handle at
    // capture, so every input the trace reads must live at a stable handle across replays. Resolves
    // (and on first sight populates) the anchor via the global map -- the SLOW pre-capture path.
    void pin_input(ggml_tensor* t, bool may_populate) const {
        if(t->op != GGML_OP_NONE || t->extra == nullptr) return;
        auto* m = (ggml_tensor_extra_metalium*)t->extra;
        if(m->tensor == nullptr || m->tensor->storage_type() != tt::tt_metal::StorageType::DEVICE) return;
        auto& pins = g_metalium_pinned_tensors;
        auto key = GGMLTensorMeta(t);
        auto it = pins.find(key);
        if(it == pins.end()) {
            if(may_populate) pins.emplace(key, m->tensor);
            return;
        }
        apply_pin(t, it->second);
    }

    // Build the cached replay plan once from the captured graph: the op==NONE device inputs that have
    // a resolved anchor (deduped by tensor) + the boundary node rebinds snapshotted at capture. This
    // is exactly the work pin_inputs(true)+rebind_nodes do, recorded so replay skips the n_nodes walk.
    void build_replay_plan() const {
        state->pin_plan.clear();
        state->rebind_plan.clear();
        std::unordered_set<const ggml_tensor*> seen;
        for(int i = 0; i < graph->n_nodes; i++) {
            ggml_tensor* n = graph->nodes[i];
            for(int j = 0; j < GGML_MAX_SRC; j++) {
                ggml_tensor* s = n->src[j];
                if(s == nullptr || s->op != GGML_OP_NONE || s->extra == nullptr) continue;
                if(!seen.insert(s).second) continue;
                auto* m = (ggml_tensor_extra_metalium*)s->extra;
                if(m->tensor == nullptr || m->tensor->storage_type() != tt::tt_metal::StorageType::DEVICE) continue;
                auto it = g_metalium_pinned_tensors.find(GGMLTensorMeta(s));
                if(it == g_metalium_pinned_tensors.end()) continue;
                state->pin_plan.emplace_back(s, it->second);
            }
        }
        for(int i = 0; i < (int)state->node_tensors.size(); i++) {
            if(state->node_tensors[i]) state->rebind_plan.emplace_back(i, state->node_tensors[i]);
        }
        state->replay_plan_built = true;
    }

    // Per-replay hot path: inject inputs and rebind boundary nodes from the cached plan, no walk.
    void replay_inject() const { for(auto& pr : state->pin_plan) apply_pin(pr.first, pr.second); }
    // FIXME: Im theory we can avoid rebind. But welp. fix later
    void replay_rebind() const {
        for(auto& pr : state->rebind_plan) {
            ggml_tensor* n = graph->nodes[pr.first];
            if(n->extra != nullptr) ((ggml_tensor_extra_metalium*)n->extra)->tensor = pr.second;
        }
    }

    // Pin every external input to its fixed anchor
    void pin_inputs(bool may_populate) const {
        for(int i = 0; i < graph->n_nodes; i++) {
            ggml_tensor* n = graph->nodes[i];
            for(int j = 0; j < GGML_MAX_SRC; j++) {
                if(n->src[j] == nullptr) break;
                pin_input(n->src[j], may_populate);
            }
        }
    }

    //  Returns true if the graph was fully served by REPLAY
    // returns false to run the node loop eagerly (and, on the capture pass, with capture recording active).
    // @note Pass 1 is the eager warmup (JIT compile, weight pre-transpose, allocs); pass 2 captures;
    //   pass 3+ replay. Every graph is eligible while tracing is on -- the test drives which graphs
    //   reach here; real-model n_tokens gating is a later rung.
    bool begin(ggml_backend_metalium_context* ctx, ggml_cgraph* cgraph) {
        if(!g_metalium_trace_enabled) return false;
        graph = cgraph;
        mesh  = ctx->device->get_mesh_device().get();
        // uid is OPTIONAL and some times not set. We use it if present, else fall back to a signature
        // of the graph's topology and node ops (slow).
        uint64_t key = cgraph->uid != 0 ? cgraph->uid : metalium_trace_graph_signature(cgraph);
        state = &metalium_trace_exec_states()[key];
        state->passes++;

        if(state->captured) {
            // Lazy init to reduce overhead setting up reply
            if(!state->replay_plan_built) build_replay_plan();
            replay_inject();                    // inject freshly-fed inputs at their baked addresses
            ttnn::operations::trace::execute_trace(mesh, state->tid, std::nullopt, /*blocking*/false);
            replay_rebind();
            return true;
        }
        // Load inputs into their anchors BEFORE opening the capture window, so the input-injection
        // copies are NOT baked into the trace (we re-do them ourselves on every replay above).
        pin_inputs(/*may_populate*/true);
        if(state->passes >= 2) {
            state->tid = ttnn::operations::trace::begin_trace_capture(mesh, std::nullopt);
            capturing = true;
        }
        return false;
    }

    // Run after the node loop to close an in-progress capture. No-op unless this pass is capturing.
    // @note Capture only RECORDS the command stream -- it does not run on device -- so we
    //   execute_trace once here to populate this pass's outputs, matching a normal eager compute.
    void finish() {
        // Tracing off (begin() returned before binding graph): nothing to settle.
        if(!g_metalium_trace_enabled) return;
        pin_inputs(/*may_populate*/false);
        if(!capturing) return;
        ttnn::operations::trace::end_trace_capture(mesh, state->tid, std::nullopt);
        state->captured = true;
        snapshot_nodes();
        ttnn::operations::trace::execute_trace(mesh, state->tid, std::nullopt, /*blocking*/false);
        ggml_metalium_trace_mem_report(mesh, "captured");
    }
};

static enum ggml_status ggml_backend_metalium_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    ggml_backend_metalium_context * ctx = (ggml_backend_metalium_context *)backend->context;

    metalium_trace_dispatch trace;
    if(trace.begin(ctx, cgraph)) {
        return GGML_STATUS_SUCCESS;
    }

    // Recognise multi-node fusions once for this graph (only the eager / trace-capture passes reach
    // here; replay returned above). Populates the inert set + fusion roots consulted in the loop.
    if(ctx->compiler != nullptr) {
        ctx->compiler->analyzeGraph(cgraph);
    }

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

        // no tensor -> no allocated TTNN tensor
        if(ggml_nelements(node) == 0) {
            continue;
        }

        // Folded into a fusion root (e.g. the SUB/REPEAT/MUL of a lerp): its work is subsumed by the
        // root, so skip it -- don't execute and don't let it be baked into a trace capture.
        if(ctx->compiler != nullptr && ctx->compiler->isInert(node)) {
            continue;
        }

        // std::cout << ggml_op_name(node->op) << " node " << node->name << " with address " << node->data << std::endl;

        std::chrono::steady_clock::time_point lt_t0;
        if(g_debug_flags.print_local_timing) {
            // Serialize so the elapsed time below reflects this op's host+device cost,
            // not pipelined overlap with later ops.
            tt::tt_metal::distributed::Finish(ctx->device->get_mesh_device()->mesh_command_queue());
            lt_t0 = std::chrono::steady_clock::now();
        }

        if (ctx->compiler != nullptr && ctx->compiler->tryLowerNode(ctx, node)) {
            // handled by the graph compiler
        } else switch (node->op) {
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
                case GGML_UNARY_OP_GELU_ERF:
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
                ggml_backend_metalium_get_rows(ctx, node);
                break;

            case GGML_OP_NORM:
                ggml_backend_metalium_norm(ctx, node, false);
                break;

            case GGML_OP_RMS_NORM:
                ggml_backend_metalium_norm(ctx, node, true);
                break;

            case GGML_OP_L2_NORM:
                ggml_backend_metalium_l2_norm(ctx, node);
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

            case GGML_OP_SUM:
                ggml_backend_metalium_sum(ctx, node);
                break;

            case GGML_OP_SUM_ROWS:
                ggml_backend_metalium_sum_rows(ctx, node);
                break;

            case GGML_OP_GLU:
                ggml_backend_metalium_glu(ctx, node);
                break;

            case GGML_OP_ROPE:
                ggml_backend_metalium_rope(ctx, node);
                break;

            case GGML_OP_FLASH_ATTN_EXT:
                ggml_backend_metalium_flash_attn(ctx, node);
                break;

            case GGML_OP_SET_ROWS:
                ggml_backend_metalium_set_rows(ctx, node);
                break;

            case GGML_OP_RWKV_WKV7:
                ggml_backend_metalium_rwkv_wkv7(ctx, node);
                break;

            case GGML_OP_NONE:
                break;

            default:
                fprintf(stderr, "%s: unsupported op %s\n", __func__, ggml_op_desc(node));
                GGML_ASSERT(false);
        }
        ggml_tensor_extra_metalium* meta = (ggml_tensor_extra_metalium*)node->extra;
        // std::cout << "Executed " << ggml_op_desc(node) << " with address " << node->data << " and shape " << meta->tensor->logical_shape() << ", GGML wants " << node->ne[0] << " " << node->ne[1] << " " << node->ne[2] << " " << node->ne[3] << std::endl;
        GGML_ASSERT(meta != NULL);
        // A row-folded result carries its device data in `row_folded` (its `tensor` is intentionally
        // null); its logical shape differs from the ggml node, so skip the canonical post-checks.
        if(!meta->is_row_folded()) {
            GGML_ASSERT(meta->tensor != NULL);
            GGML_ASSERT(meta->tensor->storage_type() == tt::tt_metal::StorageType::DEVICE);
            if(!ggml_tt_tensors_shape_equal(node, *meta->tensor)) {
                fmt::println(stderr, "Mismatched tensor shapes for node '{}' ({}): GGML wants [{}, {}, {}, {}], TTNN generates {}\n"
                    , node->name, ggml_op_name(node->op), node->ne[0], node->ne[1], node->ne[2], node->ne[3], meta->tensor->logical_shape());
                abort();
            }
        }

        if(g_debug_flags.print_local_timing) {
            // Finish so the device work this op enqueued is fully drained before we stop the clock.
            tt::tt_metal::distributed::Finish(ctx->device->get_mesh_device()->mesh_command_queue());
            double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - lt_t0).count();
            static std::map<std::string, std::pair<double, uint64_t>> acc; // op -> {total_us, calls}
            static uint64_t timed_ops = 0;
            auto& e = acc[ggml_op_desc(node)];
            e.first += us;
            e.second += 1;
            // Print a cumulative table every so often; the last one before exit is the full picture.
            timed_ops++;
            if(timed_ops % 4096 == 0) {
                double grand = 0;
                for(auto& kv : acc) grand += kv.second.first;
                std::vector<std::pair<std::string, std::pair<double, uint64_t>>> rows(acc.begin(), acc.end());
                std::sort(rows.begin(), rows.end(), [](auto& a, auto& b){ return a.second.first > b.second.first; });
                fprintf(stderr, "\n=== METALIUM LOCAL TIMING (cumulative, %lu timed ops, %.1f ms total device+host) ===\n",
                    (unsigned long)timed_ops, grand / 1000.0);
                fprintf(stderr, "%-28s %8s %12s %7s %12s\n", "op", "calls", "total_ms", "%", "us/call");
                for(auto& r : rows) {
                    fprintf(stderr, "%-28s %8lu %12.2f %6.1f%% %12.1f\n",
                        r.first.c_str(), (unsigned long)r.second.second, r.second.first / 1000.0,
                        grand > 0 ? 100.0 * r.second.first / grand : 0.0,
                        r.second.first / r.second.second);
                }
                fflush(stderr);
            }
        }
    }

    trace.finish();

    return GGML_STATUS_SUCCESS;
    GGML_UNUSED(backend);
}

static bool ggml_backend_metalium_device_supports_op_internal(ggml_backend_dev_t device, const struct ggml_tensor * op);

static bool ggml_backend_metalium_device_supports_op(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    bool ok = ggml_backend_metalium_device_supports_op_internal(device, op);
    // debug print to log rejected ops
    if(!ok && g_debug_flags.print_rejected_ops) {
        fprintf(stderr, "REJECT op %s (%s)\n", ggml_op_name(op->op), op->name);
        for(int i = 0; i < GGML_MAX_SRC; i++) {
            if(!op->src[i]) {
                break;
            }
            fprintf(stderr, "  src%d shape [%ld %ld %ld %ld], dtype = %s, name = '%s'\n", i, op->src[i]->ne[0], op->src[i]->ne[1], op->src[i]->ne[2], op->src[i]->ne[3], ggml_type_name(op->src[i]->type), op->src[i]->name);
        }

        // Follow op details
        if(op->op == GGML_OP_FLASH_ATTN_EXT) {
            fprintf(stderr, "  FlashAttention debug details:\n");
            const char* names[] = {"query", "key", "value", "mask"};
            for(int i = 0; i < 4; i++) {
                if(!op->src[i]) {
                    break;
                }
                ggml_tensor* t = op->src[i];
                while(t->op == GGML_OP_PERMUTE) {
                    t = t->src[0];
                }
                fprintf(stderr, "    src%d follow - %s shape [%ld %ld %ld %ld], dtype = %s, name = '%s'\n", i, names[i], t->ne[0], t->ne[1], t->ne[2], t->ne[3], ggml_type_name(t->type), t->name);
            }
        }
    }
    return ok;
}


static bool ggml_backend_metalium_device_supports_op_internal(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    GGML_ASSERT(op != NULL);
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    ggml_backend_metalium_device_context * ctx = (ggml_backend_metalium_device_context *)device->context;

    // Keep nodes with a special lowering on the device instead of the CPU.
    if (ctx->compiler != nullptr && ctx->compiler->hasLowering(op)) {
        return true;
    }

    // The metalium backend has seperated internal data types from the GGML data types. We really only care about
    // what we can convert to and from.
    auto tensor_supported = [&](const struct ggml_tensor * tensor) {
        if(tensor == NULL || !is_ggml_type_supported_by_metalium(tensor->type, ctx->device->arch())) {
            return false;
        }

        tt::tt_metal::DataType tt_type = ggml2tt_type(tensor->type, ctx->device->arch());
        switch(tt_type) {
            case tt::tt_metal::DataType::BFLOAT16:
            case tt::tt_metal::DataType::UINT16:
            case tt::tt_metal::DataType::FLOAT32:
            case tt::tt_metal::DataType::UINT32:
            case tt::tt_metal::DataType::INT32:
            case tt::tt_metal::DataType::BFLOAT8_B:
            case tt::tt_metal::DataType::BFLOAT4_B:
                return true;
            case tt::tt_metal::DataType::INVALID:
                GGML_ASSERT(false && "Unsupported data type");
                break;
            default:
                return false;
        }
        GGML_UNREACHABLE();
    };

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
        case GGML_OP_NORM:
            return ggml_backend_metalium_can_norm(op, false);
        case GGML_OP_RMS_NORM:
            return ggml_backend_metalium_can_norm(op, true);
        case GGML_OP_L2_NORM:
            return ggml_backend_metalium_can_norm(op, false);
        case GGML_OP_LEAKY_RELU:
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_CLAMP:
        case GGML_OP_SCALE:
        case GGML_OP_ADD1:
        case GGML_OP_SQRT:
        case GGML_OP_SQR:
        case GGML_OP_PERMUTE:
        case GGML_OP_LOG:
        case GGML_OP_VIEW:
        case GGML_OP_SUM:
            return true;
        case GGML_OP_GROUP_NORM:
            return false; // Disabled because the operator seems to be broken
        case GGML_OP_SUM_ROWS:
            return ggml_backend_metalium_can_sum_rows(op);

        case GGML_OP_CONT:
        case GGML_OP_CPY:
        case GGML_OP_DUP:
            return ggml_backend_metalium_can_cpy(op);

        case GGML_OP_SIN:
        case GGML_OP_COS:
            return true;
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            return tensor_supported(src1) && numpy_broadcast_rule(src0, src1);

        case GGML_OP_MUL_MAT:
            return tensor_supported(src1) && ggml_backend_metalium_can_mul_mat(op);
        case GGML_OP_SET:
            return tensor_supported(src1) && ggml_backend_metalium_can_set(op);
        case GGML_OP_SOFT_MAX:
            return ggml_backend_metalium_can_softmax(op);
        case GGML_OP_GET_ROWS:
            return tensor_supported(src1) && ggml_backend_metalium_can_get_rows(op, ctx->device->arch());
        case GGML_OP_CONCAT:
            return tensor_supported(src1) && ggml_backend_metalium_can_concat(op);
        case GGML_OP_REPEAT:
            return ggml_backend_metalium_can_repeat(op);
        case GGML_OP_OUT_PROD:
            return tensor_supported(src1) && ggml_backend_metalium_can_outer_product(op);
        case GGML_OP_GLU:
            return ((src1 && tensor_supported(src1)) || !src1) && ggml_backend_metalium_can_glu(op);
        case GGML_OP_ROPE:
            return tensor_supported(src1) && ggml_backend_metalium_can_rope(op);
        case GGML_OP_FLASH_ATTN_EXT:
            return tensor_supported(src1) && tensor_supported(op->src[2]) && ggml_backend_metalium_can_flash_attn(op);
        case GGML_OP_SET_ROWS:
            return tensor_supported(src1) && ggml_backend_metalium_can_set_rows(op);
        case GGML_OP_RWKV_WKV7:
            for(int i = 1; i < 7; i++) {
                if(!tensor_supported(op->src[i])) {
                    return false;
                }
            }
            return ggml_backend_metalium_can_rwkv_wkv7(op);
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
    return buft_ctx->device == ctx->device;
}

static void ggml_backend_metalium_synchronize(ggml_backend_t backend)
{
    ggml_backend_metalium_context * ctx = (ggml_backend_metalium_context *)backend->context;
    tt::tt_metal::distributed::Finish(ctx->device->get_mesh_device()->mesh_command_queue());
}

static struct ggml_backend_i metalium_backend_i = {
    /* .get_name                = */ ggml_backend_metalium_name,
    /* .free                    = */ ggml_backend_metalium_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ ggml_backend_metalium_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_metalium_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t ggml_backend_metalium_guid(void) {
    static ggml_guid guid = { 0x91, 0x69, 0xd5, 0x5f, 0x24, 0xe7, 0x44, 0x00, 0xb4, 0x2a, 0x73, 0x23, 0x48, 0xb0, 0x4e, 0xe7 };
    return &guid;
}

static ggml_backend_t ggml_backend_metalium_init(ggml_backend_metalium_device_context* dev_ctx) {
    int device_id = dev_ctx->device_id;
    ttnn::IDevice* device = dev_ctx->device.get();
    GGML_ASSERT(device_id >= 0 && (size_t)device_id < tt::tt_metal::GetNumAvailableDevices());
    GGML_ASSERT(device != nullptr);

    ggml_backend_metalium_context * ctx = new ggml_backend_metalium_context {
        /* device            = */ device,
        /* device_id         = */ device_id,
        /* name              = */ dev_ctx->name,
        /* compiler          = */ dev_ctx->compiler.get(),
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
    GGML_UNUSED(index);
    ggml_backend_metalium_reg_context * ctx = (ggml_backend_metalium_reg_context *)reg->context;
    return ctx->devices[0];
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

static void ggml_backend_metalium_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    GGML_UNUSED(dev);
    ggml_backend_metalium_device_context * ctx = (ggml_backend_metalium_device_context *)dev->context;
    size_t num_dram_channels = ctx->device->num_dram_channels();
    size_t num_devices = ctx->device->num_devices();
    auto stats = ctx->device->allocator()->get_statistics(tt::tt_metal::BufferType::DRAM);

    *total = stats.total_allocatable_size_bytes * num_dram_channels * num_devices;
    *free = stats.total_free_bytes * num_dram_channels * num_devices;
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
    ggml_backend_metalium_get_memory(dev, &free, &total);
    *props = ggml_backend_dev_props {
        .name = ctx->name.c_str(),
        .description = ctx->description.c_str(),
        .memory_free = free,
        .memory_total = total,
        .type = ggml_backend_metalium_get_type(dev),
        .device_id = NULL, // TODO: Set this to a proper ID
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

static std::vector<std::unique_ptr<ggml_backend_device>> g_backend_device_holder;
static std::vector<std::unique_ptr<ggml_backend_metalium_device_context>> g_backend_device_context_holder;

static std::vector<std::shared_ptr<ttnn::MeshDevice>> g_metalium_open_devices;
static void ggml_metalium_close_all_devices() {
    for (auto& dev : g_metalium_open_devices) {
        if (dev) {
            try {
                ttnn::close_device(*dev);
            }
            catch (...) {
                // We are at process teardown; nothing useful to do with an exception here.
            }
        }
    }
    g_metalium_open_devices.clear();
}

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_metalium_reg()
{
    static ggml_backend_reg reg;
    static std::once_flag once;
    std::call_once(once, [&]() {
        // TTNN's tilize/untilize ops read the (trivial) tile spec out of ROW_MAJOR input tensors,
        // which spams a "extract tile information out of a ROW MAJOR layout" deprecation warning
        // (tt-metal #18536) on every tensor upload. The warning is internal to TTNN and harmless
        // for us, so quiet the LogMetal channel down to errors only.
        tt::LoggerRegistry::instance().get(tt::LogMetal)->set_level(spdlog::level::err);
        metalium_register_all_kernel();
        // TODO: TTNN though not have peoper system packaging yet. Does support working in installed for (via Python packages rn)
        // Remove this limitation
        if(getenv("TT_METAL_RUNTIME_ROOT") == NULL) {
            fmt::println(stderr, "The TT_METAL_RUNTIME_ROOT environment variables must be set to use the Metalium backend");
            abort();
        }
        // Persistent kernel caching is now always-on in the TTNN SDK; the explicit
        // EnablePersistentKernelCache() toggle was removed. Honor the debug flag by
        // clearing the in-memory cache so kernels are recompiled instead.
        if(g_debug_flags.disable_program_cache) {
            fmt::println("Disabling persistent kernel cache. Things will be slower");
            tt::tt_metal::experimental::ClearKernelCache();
        }
        // TODO: Support multiple devices (TT supports mesh configuration so it's going to be tricky)
        // but for now we just work on 1 device at a time
        static std::unique_ptr<ggml_backend_metalium_reg_context> ctx = std::make_unique<ggml_backend_metalium_reg_context>();
        const size_t num_devices = 1;
        int device_id = 0;

        const char* device_id_env = getenv("GGML_METALIUM_DEVICE_ID"); // example GGML_METALIUM_DEVICE_ID=0 - use device 0
        const char* mesh_env = getenv("GGML_METALIUM_MESH_SHAPE"); // example GGML_METALIUM_MESH_SHAPE=2,4 use mesh of shape 2,4
        ttnn::MeshShape mesh_shape;
        if(device_id_env != NULL && mesh_env != NULL) {
            GGML_ABORT("Both GGML_METALIUM_DEVICE_ID and GGML_METALIUM_MESH_SHAPE are set. Only one can be used at the same time");
        }
        if(device_id_env != NULL) {
            try {
                device_id = std::stoi(device_id_env);
            }
            catch(const std::invalid_argument& e) {
                GGML_ABORT("Invalid device ID in GGML_METALIUM_DEVICE_ID");
            }
        }
        if(mesh_env != NULL) {
            std::string_view mesh_view(mesh_env);
            size_t n = mesh_view.find('x');
            if(n == std::string_view::npos) {
                GGML_ABORT("Invalid mesh shape in GGML_METALIUM_MESH_SHAPE. Expected format WxH. ex: 2x4");
            }
            int y = 0;
            int x = 0;
            try {
                y = std::stoi(std::string(mesh_view.substr(0, n)));
                x = std::stoi(std::string(mesh_view.substr(n + 1)));
            }
            catch(const std::invalid_argument& e) {
                GGML_ABORT("Invalid mesh shape in GGML_METALIUM_MESH_SHAPE");
            }

            GGML_ASSERT(x > 0 && y > 0 && "Invalid mesh shape in GGML_METALIUM_MESH_SHAPE");
            mesh_shape = ttnn::MeshShape(x, y);
        }

        ctx->devices.reserve(num_devices);
        ggml_backend_metalium_device_context * dev_ctx = new ggml_backend_metalium_device_context;
        std::shared_ptr<ttnn::MeshDevice> device;
        // Trace region size. 0 = tt-metal DYNAMIC ALLOCATION MODE: during capture it tracks DRAM
        // alloc/free high-water-marks so per-op alloc/free/reuse work normally and the trace buffer
        // is sized to the reuse-optimized peak after capture. A NONZERO value reserves a fixed
        // region up front (static mode) where capture-time buffers cannot be freed -- a graph that
        // allocates per-op (like ours) accumulates and OOMs. So when tracing is on, use 0; else keep
        // tt-metal's default. Read the runtime flag (a test may have flipped it on before open).
        const size_t trace_region_size = g_metalium_trace_enabled ? 0 : DEFAULT_TRACE_REGION_SIZE;
        if(mesh_env == NULL) {
            device = ttnn::open_mesh_device(device_id, DEFAULT_L1_SMALL_SIZE, trace_region_size);
        }
        else {
            device = ttnn::distributed::open_mesh_device(mesh_shape, DEFAULT_L1_SMALL_SIZE, trace_region_size, 2, tt::tt_metal::DispatchCoreType::ETH);
        }
        g_metalium_open_devices.push_back(device);
        std::atexit(ggml_metalium_close_all_devices); // track and kill on eexit
        if(!g_debug_flags.disable_program_cache) {
            ttnn::enable_program_cache(*device);
        }
        // Limit device support to the ones I own (GS is removed as TTNN dropped support)
        GGML_ASSERT(device->arch() == tt::ARCH::WORMHOLE_B0);
        dev_ctx->device = device;
        dev_ctx->device_id = device_id;
        dev_ctx->name = "METALIUM" + std::to_string(device_id);
        if(!g_debug_flags.disable_graph_compiler) {
            dev_ctx->compiler = std::make_unique<MetaliumGraphCompiler>();
        }
        // WHY???
        // chip_id_t MeshDevice::build_id() const { return reference_device()->id(); }
        // Reference device should be the same... Dafaq?
        tt::ChipId id = device->get_devices().size() == 1 ? device->get_devices()[0]->id() : device->id();
        GGML_ASSERT(id == device_id && "WTF? Metalium ID should match with asked device ID");
        std::string arch_str = tt::arch_to_str(device->arch());
        std::transform(arch_str.begin(), arch_str.end(), arch_str.begin(), ::toupper);
        if(device->get_devices().size() == 1) {
            auto* real_device = device->get_devices()[0];
            auto grid = real_device->compute_with_storage_grid_size();
            dev_ctx->description = fmt::format("Tenstorrent {} [grid: {}x{}, id: {}]", arch_str, grid.x, grid.y, id);
        }
        else {
            auto devshape = device->get_view().shape();
            std::string devshape_str;
            for(size_t i = 0; i < devshape.dims(); i++) {
                devshape_str += std::to_string(devshape[i]) + "x";
            }
            devshape_str.pop_back();
            dev_ctx->description = fmt::format("Tenstorrent {} {} mesh [id: {}]", arch_str, devshape_str, id);
        }

        ggml_backend_dev_t dev = new ggml_backend_device {
            .iface = ggml_backend_metalium_device_interface,
            .reg = &reg,
            .context = dev_ctx
        };
        ctx->devices.push_back(dev);
        // GGML does not have free for backend_reg and devices. Will force free on exit (thanks to RAII) but Metalium
        // already de-init at that point
        // g_backend_device_context_holder.push_back(std::unique_ptr<ggml_backend_metalium_device_context>(dev_ctx));
        // g_backend_device_holder.push_back(std::unique_ptr<ggml_backend_device>(dev));

        reg = ggml_backend_reg {
            /* .api_version = */ GGML_BACKEND_API_VERSION,
            /* .interface   = */ ggml_backend_metalium_reg_interface,
            /* .context     = */ ctx.get()
        };
    });
    return &reg;
}
