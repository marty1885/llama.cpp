#include "soft_max.hpp"
#include "tt-metalium/core_coord.hpp"
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

    void validate_with_output_tensors(
        const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const;
    std::vector<ttnn::TensorSpec> compute_output_specs(
        const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const;

    std::vector<Tensor> create_output_tensors(
        const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const;
    tt::tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;
};

ttnn::Tensor ttggml::SoftMaxOperation::invoke(const Tensor& a) {
    return tt::tt_metal::operation::run(
        SoftMaxDeviceOperation{
            a.memory_config(),
            a.dtype()
        },
        {a},
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
        if(!output_tensors.empty() && output_tensors[0].has_value()) {
            TT_FATAL(input_tensors.at(0).logical_shape() == output_tensors[0].value().logical_shape(), "Expect shape be same");
        }
}

tt::tt_metal::operation::ProgramWithCallbacks SoftMaxDeviceOperation::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const
{
    tt::tt_metal::Program program{};
    const auto& a_tensor = input_tensors.at(0);
    const auto& o_tensor = output_tensors.at(0);

    const uint32_t width = a_tensor.logical_shape()[-1];
    const uint32_t height = a_tensor.logical_shape()[-2];
    const uint32_t batch = a_tensor.logical_shape()[-3] * a_tensor.logical_shape()[-4];

    std::cout << "Softmax: width=" << width << ", height=" << height << ", batch=" << batch << std::endl;

    // tt::tt_metal::IDevice* device = a_tensor.device();
    CoreCoord core_grid = CoreCoord(1, 1);

    auto* a = a_tensor.buffer();
    auto* o = o_tensor.buffer();

    const uint32_t width_tiles = width / 32 + (width % 32 != 0);
    const uint32_t height_tiles = (height / 32 + (height % 32 != 0)) * batch;
    std::cout << "Softmax: width_tiles = " << width_tiles << ", height_tiles = " << height_tiles << std::endl;

    auto [num_cores,
        all_cores,
        core_group_1,
        core_group_2,
        work_per_core1,
        work_per_core2] =
        tt::tt_metal::split_work_to_cores(core_grid, height_tiles);

    // We'll deal with it later
    TT_FATAL(a_tensor.dtype() == tt::tt_metal::DataType::BFLOAT16, "Unsupported data type");
    TT_FATAL(o_tensor.dtype() == tt::tt_metal::DataType::BFLOAT16, "Unsupported data type");

    MakeCircularBuffer(program, all_cores, tt::CBIndex::c_0, 2, a_tensor.dtype()); // cb_in0
    MakeCircularBuffer(program, all_cores, tt::CBIndex::c_16, 2, o_tensor.dtype()); // cb_out
    MakeCircularBuffer(program, all_cores, tt::CBIndex::c_24, 1, tt::tt_metal::DataType::BFLOAT16); // cb_const1
    MakeCircularBuffer(program, all_cores, tt::CBIndex::c_25, 1, tt::tt_metal::DataType::BFLOAT16); // cb_sum
    MakeCircularBuffer(program, all_cores, tt::CBIndex::c_26, 1, tt::tt_metal::DataType::BFLOAT16); // cb_max
    MakeCircularBuffer(program, all_cores, tt::CBIndex::c_27, 1, tt::tt_metal::DataType::BFLOAT16); // cb_tmp
    MakeCircularBuffer(program, all_cores, tt::CBIndex::c_28, 1, tt::tt_metal::DataType::BFLOAT16); // cb_global_max
    MakeCircularBuffer(program, all_cores, tt::CBIndex::c_29, 1, tt::tt_metal::DataType::BFLOAT16); // cb_global_sum
    MakeCircularBuffer(program, all_cores, tt::CBIndex::c_30, 1, tt::tt_metal::DataType::BFLOAT16); // cb_tmp2
    MakeCircularBuffer(program, all_cores, tt::CBIndex::c_31, 4, tt::tt_metal::DataType::BFLOAT16); // cb_tile_mask


    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(*a).append_to(reader_compile_time_args);
    KernelHandle reader = CreateMetaliumKernel(program, "soft_max_reader", all_cores, DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default,
        .compile_args = reader_compile_time_args,
        .defines = {},
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

    KernelHandle compute = CreateMetaliumKernel(program, "soft_max_compute", all_cores, ComputeConfig{
        .fp32_dest_acc_en = true,
        .unpack_to_dest_mode = {},
        .compile_args = {},
        .defines = {},
        .named_compile_args = {}
    });

    auto work_groups = {std::make_pair(core_group_1, work_per_core1), std::make_pair(core_group_2, work_per_core2)};
    for(const auto& [group, work_per_item] : work_groups) {
        for(const auto& range : group.ranges()) {
            for(const auto& core : range) {

                SetRuntimeArgs(program, reader, core, std::vector<uint32_t>{a->address(), width_tiles, height_tiles});
                SetRuntimeArgs(program, compute, core, std::vector<uint32_t>{width, height, batch});
                SetRuntimeArgs(program, writer, core, std::vector<uint32_t>{o->address(), width_tiles, height_tiles});
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
