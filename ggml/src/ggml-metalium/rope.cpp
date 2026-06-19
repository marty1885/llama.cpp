#include "rope.hpp"
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/kernel_types.hpp"
#include "tt-metalium/tt_backend_api_types.hpp"
#include <ttnn/tensor/layout/layout.hpp>
#include <cmath>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "utils.hpp"

using namespace tt::tt_metal;

namespace ttggml {
using namespace ttnn;

// ---- RoPEDeviceOperation ----

rope_device::RoPEDeviceOperation::program_factory_t
rope_device::RoPEDeviceOperation::select_program_factory(
    const operation_attributes_t& /*attrs*/,
    const tensor_args_t& /*tensor_args*/)
{
    return program::RoPEProgramFactory{};
}

rope_device::RoPEDeviceOperation::spec_return_value_t
rope_device::RoPEDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args)
{
    const auto& src = tensor_args.src;
    return TensorSpec(
        src.logical_shape(),
        tt::tt_metal::TensorLayout(
            src.dtype(),
            tt::tt_metal::PageConfig(src.layout()),
            operation_attributes.output_mem_config));
}

rope_device::RoPEDeviceOperation::tensor_return_value_t
rope_device::RoPEDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args)
{
    auto spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(spec, tensor_args.src.device());
}

void rope_device::RoPEDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args)
{
    const auto& src_tensor = tensor_args.src;
    const auto& index_tensor = tensor_args.index;
    const auto& src_shape = src_tensor.logical_shape();
    const auto& index_shape = index_tensor.logical_shape();

    // Standard RoPE supplies one position per token (index length == n_token).
    // mRoPE / interleaved-mRoPE (ggml_rope_multi, used by Qwen2-VL / Qwen3-VL) packs
    // four position sections [t, h, w, e] so the index length is 4*n_token; only the
    // leading n_token (t) positions are consumed by the kernel below.
    TT_FATAL(index_shape[-1] == src_shape[-3] || index_shape[-1] == 4 * src_shape[-3],
        "Shape mismatch: src_shape = {}, index_shape = {}. Expect index length n_token "
        "(standard RoPE) or 4*n_token (mRoPE) for format [batch, n_token, vec_dim]",
        src_shape, index_shape);

    TT_FATAL(index_tensor.dtype() == tt::tt_metal::DataType::INT32 ||
        index_tensor.dtype() == tt::tt_metal::DataType::UINT32, "Index tensor must be of type (U)INT32");
    TT_FATAL(src_tensor.layout() == tt::tt_metal::Layout::TILE, "Source tensor must be of layout TILE");
    TT_FATAL(index_tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR, "Index tensor must be of layout ROW_MAJOR");
    TT_FATAL(index_tensor.storage_type() == tt::tt_metal::StorageType::DEVICE, "Index tensor must be on device");
    TT_FATAL(src_tensor.storage_type() == tt::tt_metal::StorageType::DEVICE, "Source tensor must be on device");

    if (tensor_args.freq_factor.has_value()) {
        const auto& freq_factor = tensor_args.freq_factor.value();
        const auto& freq_factor_shape = freq_factor.logical_shape();
        TT_FATAL(freq_factor_shape[-1] == operation_attributes.active_dim_size / 2,
            "Frequency factor must have the same size as active dimension");
        for (size_t i = 0; i < freq_factor_shape.size() - 1; i++) {
            TT_FATAL(freq_factor_shape[i] == 1,
                "Frequency factor shape must have shape [active_dim_size/2], got {}", freq_factor_shape);
        }
        if (operation_attributes.rope_type == RoPEType::Normal) {
            TT_FATAL(freq_factor.dtype() == tt::tt_metal::DataType::BFLOAT16,
                "Frequency factor tensor must be of type BFLOAT16 for Normal RoPE");
        }
    }

    if (operation_attributes.rope_type == RoPEType::NeoX) {
        TT_FATAL(operation_attributes.active_dim_size % 64 == 0, "For NeoX RoPE, active_dim must be a multiple of 64");
    } else {
        TT_FATAL(operation_attributes.active_dim_size % 32 == 0, "For Normal RoPE, active_dim must be a multiple of 32");
    }
    TT_FATAL(operation_attributes.active_dim_size <= src_tensor.padded_shape()[-1],
        "active_dim must be less than the last dimension of the source tensor");
    TT_FATAL(operation_attributes.freq_base >= 0, "base_freq must be non-negative");
    TT_FATAL(operation_attributes.freq_scale > 0, "freq_scale must be positive");
}

void rope_device::RoPEDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args)
{
    validate_on_program_cache_miss(operation_attributes, tensor_args);
}

std::tuple<rope_device::operation_attributes_t, rope_device::tensor_args_t>
rope_device::RoPEDeviceOperation::invoke(
    const Tensor& src,
    const Tensor& index,
    std::optional<Tensor> freq_factor,
    uint32_t active_dim_size,
    RoPEType rope_type,
    uint32_t n_ctx_orig,
    float freq_base,
    float freq_scale,
    float ext_factor,
    float attn_factor,
    float beta_fast,
    float beta_slow)
{
    return {
        operation_attributes_t{
            src.memory_config(),
            active_dim_size,
            n_ctx_orig,
            rope_type,
            freq_base,
            freq_scale,
            ext_factor,
            attn_factor,
            beta_fast,
            beta_slow
        },
        tensor_args_t{src, index, std::move(freq_factor)}
    };
}

// ---- RoPEProgramFactory ----

rope_device::program::RoPEProgramFactory::cached_program_t
rope_device::program::RoPEProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value)
{
    tt::tt_metal::Program program{};
    const auto& src_tensor = tensor_args.src;
    const auto& index_tensor = tensor_args.index;
    const auto& output_tensor = tensor_return_value;
    const auto& freq_factor = tensor_args.freq_factor;

    const uint32_t B = src_tensor.logical_shape()[-3];
    const uint32_t D = src_tensor.logical_shape()[-1];
    const uint32_t D_active = operation_attributes.active_dim_size;
    const uint32_t N = src_tensor.logical_shape()[-2];

    tt::tt_metal::IDevice* device = src_tensor.device();

    auto* src = src_tensor.buffer();
    auto* idxs = index_tensor.buffer();
    auto* dst = output_tensor.buffer();

    const uint32_t Dt = D/32 + (D % 32 != 0);
    const uint32_t Nt = N/32 + (N % 32 != 0);
    const uint32_t D_activet = D_active/32; // D_active is always a multiple of 64, so D_activet is always a multiple of 2

    auto core_grid = device->compute_with_storage_grid_size();

    uint32_t active_tiles = operation_attributes.rope_type == RoPEType::NeoX ? D_activet/2 * Nt * B : D_activet * Nt * B;
    uint32_t passive_tiles = (Dt - D_activet) * Nt * B;
    auto [num_cores_active,
        all_cores_active,
        core_group_1_active,
        core_group_2_active,
        work_per_core1_active,
        work_per_core2_active] =
        tt::tt_metal::split_work_to_cores(core_grid, active_tiles);
    auto [num_cores_passive,
        all_cores_passive,
        core_group_1_passive,
        core_group_2_passive,
        work_per_core1_passive,
        work_per_core2_passive] =
        tt::tt_metal::split_work_to_cores(core_grid, passive_tiles);

    auto all_cores = all_cores_active.merge(all_cores_passive);

    MakeCircularBuffer(program, all_cores, tt::CBIndex::c_0, 4, src_tensor.dtype()); // cb_in0
    MakeCircularBuffer(program, all_cores, tt::CBIndex::c_1, B*sizeof(int32_t), B*sizeof(int32_t), tt::DataFormat::Int32); // cb_in1
    MakeCircularBuffer(program, all_cores, tt::CBIndex::c_16, 2, output_tensor.dtype()); // cb_out
    MakeCircularBuffer(program, all_cores, tt::CBIndex::c_17, 4, src_tensor.dtype()); // cb_bypass
    if (freq_factor) {
        MakeCircularBuffer(program, all_cores, tt::CBIndex::c_2, 4, freq_factor->dtype()); // cb_in2
    }

    std::map<std::string, std::string> reader_defines;
    std::map<std::string, std::string> defines;
    defines["FREQ_BASE"] = to_string_precise(operation_attributes.freq_base);
    defines["FREQ_BASE_LOG"] = to_string_precise(std::log(operation_attributes.freq_base));
    if (operation_attributes.attn_factor != 1.f) {
        defines["ATTN_FACTOR"] = to_string_precise(operation_attributes.attn_factor);
    }
    if (operation_attributes.freq_scale != 1.f) {
        defines["FREQ_SCALE"] = to_string_precise(operation_attributes.freq_scale);
        defines["LOG_1_FREQ_SCALE"] = to_string_precise(std::log(1.0f / operation_attributes.freq_scale));
    }
    if (operation_attributes.ext_factor != 0.f) {
        defines["EXT_FACTOR"] = to_string_precise(operation_attributes.ext_factor);
        float corr_dims[2];
        ggml_rope_yarn_corr_dims(D_active, operation_attributes.n_ctx_orig, operation_attributes.freq_base,
            operation_attributes.beta_fast, operation_attributes.beta_slow, corr_dims);
        if (std::isinf(corr_dims[0]) || std::isnan(corr_dims[0])) {
            corr_dims[0] = 0;
        }
        if (std::isinf(corr_dims[1]) || std::isnan(corr_dims[1])) {
            corr_dims[1] = 0;
        }
        defines["CORR_DIMS0"] = to_string_precise(corr_dims[0]);
        defines["CORR_DIMS1"] = to_string_precise(corr_dims[1]);
    }
    if (freq_factor) {
        defines["HAS_FREQ_FACTOR"] = "1";
        reader_defines["HAS_FREQ_FACTOR"] = "1";
    }

    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(*src).append_to(reader_compile_time_args);
    TensorAccessorArgs(*idxs).append_to(reader_compile_time_args);
    if (freq_factor) {
        const auto* freq_fact = freq_factor->buffer();
        TensorAccessorArgs(*freq_fact).append_to(reader_compile_time_args);
    }
    std::string variant = operation_attributes.rope_type == RoPEType::NeoX ? "neox" : "normal";
    KernelHandle reader = CreateMetaliumKernel(program, fmt::format("rope_{}_reader", variant), all_cores, DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default,
        .compile_args = reader_compile_time_args,
        .defines = reader_defines,
        .named_compile_args = {}
    });

    std::vector<uint32_t> writer_compile_time_args;
    TensorAccessorArgs(*dst).append_to(writer_compile_time_args);
    KernelHandle writer = CreateMetaliumKernel(program, fmt::format("rope_{}_writer", variant), all_cores, DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = NOC::RISCV_1_default,
        .compile_args = writer_compile_time_args,
        .defines = {},
        .named_compile_args = {}
    });

    KernelHandle compute = CreateMetaliumKernel(program, fmt::format("rope_{}_compute", variant), all_cores, ComputeConfig{
        .fp32_dest_acc_en = true,
        .unpack_to_dest_mode = {},
        .compile_args = {},
        .defines = defines,
        .named_compile_args = {},
    });

    uint32_t active_id = 0;
    uint32_t passive_id = 0;
    auto freq_factor_addr = freq_factor ? freq_factor->buffer()->address() : 0;
    for (const auto& range : all_cores.ranges()) {
        for (const auto& core : range) {
            uint32_t active_size = 0;
            uint32_t passive_size = 0;

            if (core_group_1_active.contains(core)) {
                active_size = work_per_core1_active;
            } else if (core_group_2_active.contains(core)) {
                active_size = work_per_core2_active;
            }

            if (core_group_1_passive.contains(core)) {
                passive_size = work_per_core1_passive;
            } else if (core_group_2_passive.contains(core)) {
                passive_size = work_per_core2_passive;
            }

            SetRuntimeArgs(program, reader, core, std::vector<uint32_t>{
                src->address(), D_activet, Dt, Nt, idxs->address(), B,
                active_id, active_id+active_size, passive_id, passive_id+passive_size, N, freq_factor_addr});
            SetRuntimeArgs(program, compute, core, std::vector<uint32_t>{
                D_activet, Dt, Nt, B, active_id, active_id+active_size, N});
            SetRuntimeArgs(program, writer, core, std::vector<uint32_t>{
                dst->address(), D_activet, Dt, Nt, B, active_id, active_id+active_size, passive_id, passive_id+passive_size});

            active_id += active_size;
            passive_id += passive_size;
        }
    }

    return {std::move(program), shared_variables_t{reader, writer, all_cores}};
}

void rope_device::program::RoPEProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value)
{
    auto& program = cached_program.program;
    const auto& reader = cached_program.shared_variables.reader;
    const auto& writer = cached_program.shared_variables.writer;
    const auto& all_cores = cached_program.shared_variables.all_cores;

    auto* src_buffer = tensor_args.src.buffer();
    auto* idx_buffer = tensor_args.index.buffer();
    auto* dst_buffer = tensor_return_value.buffer();

    for (const auto& range : all_cores.ranges()) {
        for (const auto& core : range) {
            {
                auto& runtime_args = GetRuntimeArgs(program, reader, core);
                runtime_args[0] = src_buffer->address();
                runtime_args[4] = idx_buffer->address();
                if (tensor_args.freq_factor.has_value()) {
                    runtime_args[11] = tensor_args.freq_factor->buffer()->address();
                }
            }
            {
                auto& runtime_args = GetRuntimeArgs(program, writer, core);
                runtime_args[0] = dst_buffer->address();
            }
        }
    }
}

// ---- prim::rope ----

ttnn::Tensor prim::rope(
    const Tensor& src,
    const Tensor& index,
    std::optional<Tensor> freq_factor,
    uint32_t active_dim_size,
    RoPEType rope_type,
    uint32_t n_ctx_orig,
    float freq_base,
    float freq_scale,
    float ext_factor,
    float attn_factor,
    float beta_fast,
    float beta_slow)
{
    auto [attrs, args] = rope_device::RoPEDeviceOperation::invoke(
        src, index, std::move(freq_factor), active_dim_size, rope_type,
        n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    return ttnn::device_operation::detail::launch<rope_device::RoPEDeviceOperation>(attrs, args);
}

// ---- rope ----

ttnn::Tensor rope(
    const Tensor& src_tensor, const Tensor& index_tensor,
    uint32_t active_dim_size, RoPEType rope_type, uint32_t n_ctx_orig,
    float freq_base, float freq_scale, float ext_factor, float attn_factor,
    float beta_fast, float beta_slow)
{
    return ttggml::prim::rope(src_tensor, index_tensor, std::nullopt,
        active_dim_size, rope_type, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow);
}

ttnn::Tensor rope(
    const Tensor& src_tensor, const Tensor& index_tensor, const Tensor& freq_factor,
    uint32_t active_dim_size, RoPEType rope_type, uint32_t n_ctx_orig,
    float freq_base, float freq_scale, float ext_factor, float attn_factor,
    float beta_fast, float beta_slow)
{
    return ttggml::prim::rope(src_tensor, index_tensor, freq_factor,
        active_dim_size, rope_type, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow);
}

} // namespace ttggml
