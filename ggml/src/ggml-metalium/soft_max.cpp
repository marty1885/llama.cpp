#include "soft_max.hpp"
#include "tt-metalium/core_coord.hpp"
#include "tt-metalium/hal_types.hpp"
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/kernel_types.hpp"
#include <ttnn/run_operation.hpp>
#include <ttnn/tensor/layout/layout.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "tt-metalium/tt_backend_api_types.hpp"
#include "tt_stl/assert.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "utils.hpp"


using namespace tt::tt_metal;

struct SoftMaxDeviceOperation {
    const tt::tt_metal::MemoryConfig output_mem_config;
    const tt::tt_metal::DataType output_dtype{};
    float scale = 1.f;

    void validate_with_output_tensors(
        const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const;
    std::vector<ttnn::TensorSpec> compute_output_specs(
        const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const;

    std::vector<Tensor> create_output_tensors(
        const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const;
    tt::tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;
};

ttnn::Tensor ttggml::SoftMaxOperation::invoke(const Tensor& a, float scale) {
    return tt::tt_metal::operation::run(
        SoftMaxDeviceOperation{
            a.memory_config(),
            a.dtype(),
            scale
        },
        {a},
        {},
        {})[0];
}

ttnn::Tensor ttggml::SoftMaxOperation::invoke(const Tensor& a, const Tensor& mask, float scale) {
    return tt::tt_metal::operation::run(
        SoftMaxDeviceOperation{
            a.memory_config(),
            a.dtype(),
            scale
        },
        {a, mask},
        {},
        {})[0];
}

std::vector<ttnn::TensorSpec> SoftMaxDeviceOperation::compute_output_specs(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const
{
    if (!output_tensors.empty() && output_tensors[0].has_value()) {
        return {output_tensors[0]->tensor_spec()};
    }

    const auto& a = input_tensors.at(0);
    return {TensorSpec(
        a.logical_shape(),
        tt::tt_metal::TensorLayout(
            output_dtype,
            tt::tt_metal::PageConfig(ttnn::TILE_LAYOUT),
            output_mem_config)
    )};
}

std::vector<ttnn::Tensor> SoftMaxDeviceOperation::create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    if (!output_tensors.empty() && output_tensors[0].has_value()) {
        return {output_tensors[0].value()};
    }
    const auto& input_tensor = input_tensors.at(0);
    auto spec = compute_output_specs(input_tensors, output_tensors)[0];
    return {create_device_tensor(spec, input_tensor.device())};
}

void SoftMaxDeviceOperation::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
        const auto& a_tensor = input_tensors.at(0);
        if(!output_tensors.empty() && output_tensors[0].has_value()) {
            const auto& o_tensor = output_tensors[0].value();
            TT_FATAL(input_tensors.at(0).logical_shape() == output_tensors[0].value().logical_shape(), "Expect shape be same");
            // XXX: We will deal with alternative data type support later
            TT_FATAL(o_tensor.dtype() == tt::tt_metal::DataType::BFLOAT16, "Output data type must be BFLOAT16");
        }

        // XXX: We will deal with alternative data type support later
        TT_FATAL(a_tensor.dtype() == tt::tt_metal::DataType::BFLOAT16, "Input data type must be BFLOAT16");

        if(input_tensors.size() > 1) {
            const auto& mask_tensor = input_tensors.at(1);
            TT_FATAL(mask_tensor.dtype() == tt::tt_metal::DataType::BFLOAT16, "Mask data type must be BFLOAT16");
            TT_FATAL(a_tensor.logical_shape()[0] % mask_tensor.logical_shape()[0] == 0 &&
                a_tensor.logical_shape()[1] % mask_tensor.logical_shape()[1] == 0 &&
                a_tensor.logical_shape()[2] == mask_tensor.logical_shape()[2] &&
                a_tensor.logical_shape()[3] == mask_tensor.logical_shape()[3], "Mask shape must be broadcastable in the first 2 dimensions and same to input in the last 2");
        }

}

tt::tt_metal::operation::ProgramWithCallbacks SoftMaxDeviceOperation::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const
{
    tt::tt_metal::Program program{};
    const auto& a_tensor = input_tensors.at(0);
    const auto& o_tensor = output_tensors.at(0);
    const auto mask_tensor = at_index(input_tensors, 1);

    const uint32_t width = a_tensor.logical_shape()[-1];
    const uint32_t height = a_tensor.logical_shape()[-2];
    const uint32_t n_head = a_tensor.logical_shape()[-3];
    const uint32_t batch = a_tensor.logical_shape()[-4];

    // tt::tt_metal::IDevice* device = a_tensor.device();
    tt::tt_metal::CoreCoord core_grid = tt::tt_metal::CoreCoord(1, 1);

    auto* a = a_tensor.buffer();
    auto* o = o_tensor.buffer();
    auto* mask = mask_tensor ? mask_tensor->buffer() : nullptr;

    const uint32_t width_tiles = width / 32 + (width % 32 != 0);
    const uint32_t height_tiles = (height / 32 + (height % 32 != 0));

    const bool need_tile_mask = width % 32 != 0 || height % 32 != 0;

    auto [num_cores,
        all_cores,
        core_group_1,
        core_group_2,
        work_per_core1,
        work_per_core2] =
        tt::tt_metal::split_work_to_cores(core_grid, height_tiles);

    MakeCircularBuffer(program, all_cores, tt::CBIndex::c_0, 2, a_tensor.dtype()); // cb_in0
    if(mask_tensor) {
        MakeCircularBuffer(program, all_cores, tt::CBIndex::c_1, 2, mask_tensor->dtype()); // cb_in1 (mask)
    }
    MakeCircularBuffer(program, all_cores, tt::CBIndex::c_16, 2, o_tensor.dtype()); // cb_out
    MakeCircularBuffer(program, all_cores, tt::CBIndex::c_24, 1, tt::tt_metal::DataType::BFLOAT4_B); // cb_const1
    MakeCircularBuffer(program, all_cores, tt::CBIndex::c_25, 1, tt::tt_metal::DataType::BFLOAT16); // cb_sum
    MakeCircularBuffer(program, all_cores, tt::CBIndex::c_26, 1, tt::tt_metal::DataType::BFLOAT16); // cb_max
    MakeCircularBuffer(program, all_cores, tt::CBIndex::c_27, 1, tt::tt_metal::DataType::BFLOAT16); // cb_tmp
    MakeCircularBuffer(program, all_cores, tt::CBIndex::c_28, 1, tt::tt_metal::DataType::BFLOAT16); // cb_global_max
    MakeCircularBuffer(program, all_cores, tt::CBIndex::c_29, 1, tt::tt_metal::DataType::BFLOAT16); // cb_global_sum
    MakeCircularBuffer(program, all_cores, tt::CBIndex::c_30, 1, tt::tt_metal::DataType::BFLOAT16); // cb_tmp2
    if(need_tile_mask) {
        MakeCircularBuffer(program, all_cores, tt::CBIndex::c_31, 4, tt::tt_metal::DataType::BFLOAT16); // cb_tile_mask
    }


    std::map<std::string, std::string> reader_defines;
    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(*a).append_to(reader_compile_time_args);
    if(mask) {
        TensorAccessorArgs(*mask).append_to(reader_compile_time_args);
        reader_defines["HAS_MASK"] = "1";
    }
    KernelHandle reader = CreateMetaliumKernel(program, "soft_max_reader", all_cores, DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default,
        .compile_args = reader_compile_time_args,
        .defines = reader_defines,
        .named_compile_args = {}
    });

    std::vector<uint32_t> writer_compile_time_args;
    TensorAccessorArgs(*o).append_to(writer_compile_time_args);
    KernelHandle writer = CreateMetaliumKernel(program, "soft_max_writer", all_cores, DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = NOC::RISCV_1_default,
        .compile_args = writer_compile_time_args,
        .defines = {},
        .named_compile_args = {}
    });

    std::map<std::string, std::string> defines;
    if(scale != 1.f) {
        defines["SCALE"] = to_string_precise(scale);
        const uint32_t* scale_int = reinterpret_cast<const uint32_t*>(&scale);
        defines["SCALE_FP32_ENCODED_AS_INT"] = std::to_string(*scale_int);
    }
    if(mask) {
        defines["HAS_MASK"] = "1";
    }
    if(need_tile_mask) {
        defines["NEED_TILE_MASK"] = "1";
    }
    KernelHandle compute = CreateMetaliumKernel(program, "soft_max_compute", all_cores, ComputeConfig{
        .fp32_dest_acc_en = true,
        .unpack_to_dest_mode = {},
        .compile_args = {},
        .defines = std::move(defines),
        .named_compile_args = {}
    });

    auto work_groups = {std::make_pair(core_group_1, work_per_core1), std::make_pair(core_group_2, work_per_core2)};
    for(const auto& [group, work_per_item] : work_groups) {
        for(const auto& range : group.ranges()) {
            for(const auto& core : range) {

                DeviceAddr mask_addr = mask ? mask->address() : 0;
                uint32_t mask_n_head = mask_tensor ? mask_tensor->logical_shape()[1] : 0;
                uint32_t mask_batch = mask_tensor ? mask_tensor->logical_shape()[0] : 0;

                SetRuntimeArgs(program, reader, core, std::vector<uint32_t>{a->address(), width_tiles, height_tiles, n_head, batch, (uint32_t)mask_addr, mask_n_head, mask_batch});
                SetRuntimeArgs(program, compute, core, std::vector<uint32_t>{width, height, batch, n_head, batch});
                SetRuntimeArgs(program, writer, core, std::vector<uint32_t>{o->address(), width_tiles, height_tiles, n_head, batch});
            }
        }
    }

    auto override_runtime_args_callback = [reader, writer, all_cores](
                                                  const void* operation,
                                                  Program& program,
                                                  const std::vector<Tensor>& input_tensors,
                                                  const std::vector<std::optional<const Tensor>>&,
                                                  const std::vector<Tensor>& output_tensors) {
            (void)operation;
            auto* a = input_tensors.at(0).buffer();
            auto* o = output_tensors.at(0).buffer();

            for(const auto& range : all_cores.ranges()) {
                for (const auto& core : range) {
                    {
                        auto& runtime_args = GetRuntimeArgs(program, reader, core);
                        runtime_args[0] = a->address();

                        if(input_tensors.size() > 1) {
                            auto* mask = input_tensors.at(1).buffer();
                            runtime_args[5] = mask->address();
                        }
                    }

                    {
                        auto& runtime_args = GetRuntimeArgs(program, writer, core);
                        runtime_args[0] = o->address();
                    }
                }
            }
        };

        return {std::move(program), override_runtime_args_callback};
}
