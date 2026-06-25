#include "slice_write_folded.hpp"

#include <array>
#include <map>
#include <string>
#include <vector>

#include "tt-metalium/host_api.hpp"
#include "tt-metalium/kernel_types.hpp"
#include "tt-metalium/tt_backend_api_types.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <ttnn/tensor/layout/layout.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "utils.hpp"

using namespace tt::tt_metal;

namespace ttggml {
using namespace ttnn;

namespace swf_device {

SliceWriteFoldedDeviceOperation::program_factory_t
SliceWriteFoldedDeviceOperation::select_program_factory(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& /*t*/) {
    return program::SWFProgramFactory{};
}

void SliceWriteFoldedDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& a, const tensor_args_t& t) {
    TT_FATAL(t.src.storage_type()   == tt::tt_metal::StorageType::DEVICE, "swf: src must be on device");
    TT_FATAL(t.cache.storage_type() == tt::tt_metal::StorageType::DEVICE, "swf: cache must be on device");
    TT_FATAL(t.src.layout()   == tt::tt_metal::Layout::TILE, "swf: src must be TILE layout");
    TT_FATAL(t.cache.layout() == tt::tt_metal::Layout::TILE, "swf: cache must be TILE layout");
    TT_FATAL(t.src.dtype()   == tt::tt_metal::DataType::BFLOAT16, "swf: src must be bf16");
    TT_FATAL(t.cache.dtype() == tt::tt_metal::DataType::BFLOAT16, "swf: cache must be bf16");
    TT_FATAL(a.ng % 32 == 0, "swf: groups-per-head must be a tile multiple (got {})", a.ng);
}

void SliceWriteFoldedDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& a, const tensor_args_t& t) {
    validate_on_program_cache_miss(a, t);
}

SliceWriteFoldedDeviceOperation::spec_return_value_t
SliceWriteFoldedDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*a*/, const tensor_args_t& t) {
    // In-place: the op output IS the folded cache.
    return t.cache.tensor_spec();
}

SliceWriteFoldedDeviceOperation::tensor_return_value_t
SliceWriteFoldedDeviceOperation::create_output_tensors(
    const operation_attributes_t& /*a*/, const tensor_args_t& t) {
    return t.cache;   // write in place; no fresh allocation
}

std::tuple<operation_attributes_t, tensor_args_t>
SliceWriteFoldedDeviceOperation::invoke(
    const Tensor& src, const Tensor& cache, uint32_t T, uint32_t H, uint32_t ng) {
    return { operation_attributes_t{T, H, ng}, tensor_args_t{src, cache} };
}

namespace program {

SWFProgramFactory::cached_program_t
SWFProgramFactory::create(
    const operation_attributes_t& attrs,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    Program prog = CreateProgram();
    using CB = tt::CBIndex;

    const uint32_t H  = attrs.H;
    const uint32_t ng = attrs.ng;

    IDevice* device = tensor_args.src.device();
    auto grid = device->compute_with_storage_grid_size();
    // One unit of work = one head.
    auto [num_cores, core, core_group_1, core_group_2, units_per_core_1, units_per_core_2] =
        split_work_to_cores(grid, H);

    // Scratch CB 0: one head's worth of faces = ng*2*32 bytes = (ng/32)*4 tiles.
    const uint32_t ts = sizeof(bfloat16) * 32 * 32;
    const uint32_t scratch_tiles = (ng * 2 * 32 + ts - 1) / ts;
    {
        CircularBufferConfig c =
            CircularBufferConfig(scratch_tiles * ts, {{CB::c_0, tt::DataFormat::Float16_b}})
                .set_page_size(CB::c_0, ts);
        CreateCircularBuffer(prog, core, c);
    }

    std::vector<uint32_t> ct;
    TensorAccessorArgs(*tensor_args.src.buffer()).append_to(ct);
    TensorAccessorArgs(*tensor_return_value.buffer()).append_to(ct);   // cache (output)
    KernelHandle dm = CreateMetaliumKernel(prog, "slice_write_folded_dm", core, DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = ct});

    const std::vector<CoreCoord> cores = corerange_to_cores(core, num_cores);
    uint32_t done = 0;
    for (uint32_t i = 0; i < num_cores; i++) {
        const CoreCoord cc = cores[i];
        const uint32_t here = core_group_1.contains(cc) ? units_per_core_1 : units_per_core_2;
        const uint32_t hs = done, he = done + here;
        done += here;
        SetRuntimeArgs(prog, dm, cc, std::vector<uint32_t>{
            attrs.T, H, ng,
            (uint32_t)tensor_args.src.buffer()->address(),
            (uint32_t)tensor_return_value.buffer()->address(),
            hs, he});
    }

    return {std::move(prog), shared_variables_t{dm, core}};
}

void SWFProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*attrs*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program   = cached_program.program;
    const auto& dm  = cached_program.shared_variables.dm;
    const auto& all = cached_program.shared_variables.all_cores;

    auto* src_buf = tensor_args.src.buffer();
    auto* dst_buf = tensor_return_value.buffer();
    for (const auto& range : all.ranges()) {
        for (const auto& core : range) {
            auto& ra = GetRuntimeArgs(program, dm, core);
            ra[3] = (uint32_t)src_buf->address();
            ra[4] = (uint32_t)dst_buf->address();
        }
    }
}

} // namespace program

} // namespace swf_device

ttnn::Tensor slice_write_region2_folded(
    const Tensor& src, const Tensor& cache, uint32_t T, uint32_t H, uint32_t ng) {
    auto [attrs, args] = swf_device::SliceWriteFoldedDeviceOperation::invoke(src, cache, T, H, ng);
    return ttnn::device_operation::detail::launch<swf_device::SliceWriteFoldedDeviceOperation>(attrs, args);
}

} // namespace ttggml
