#include "mul_mat.hpp"
#include "tt-metalium/core_coord.hpp"
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/kernel_types.hpp"
#include "tt-metalium/tt_backend_api_types.hpp"
#include <ttnn/run_operation.hpp>
#include <ttnn/tensor/layout/layout.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "tt_stl/assert.hpp"
#include "ttnn/types.hpp"
#include "utils.hpp"

using namespace tt::tt_metal;

struct MulMatDeviceOperation {
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

ttnn::Tensor ttggml::MulMatOperation::invoke(const Tensor& a, const Tensor& b) {
    return tt::tt_metal::operation::run(
        MulMatDeviceOperation{
            b.memory_config(),
            b.dtype(),
        },
        {a, b},
        {},
        {})[0];
}

std::vector<ttnn::TensorSpec> MulMatDeviceOperation::compute_output_specs(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const
{
    if (!output_tensors.empty() && output_tensors[0].has_value()) {
        return {output_tensors[0]->tensor_spec()};
    }

    const auto& a = input_tensors.at(0);
    const auto& b = input_tensors.at(1);
    ttnn::Shape output_shape({
        std::max(a.logical_shape()[0], b.logical_shape()[0]),
        std::max(a.logical_shape()[1], b.logical_shape()[1]),
        a.logical_shape()[2],
        b.logical_shape()[2],
    });
    return {TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout(
            output_dtype,
            tt::tt_metal::PageConfig(ttnn::TILE_LAYOUT),
            output_mem_config)
    )};
}

std::vector<ttnn::Tensor> MulMatDeviceOperation::create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    if (!output_tensors.empty() && output_tensors[0].has_value()) {
        return {output_tensors[0].value()};
    }
    const auto& input_tensor = input_tensors.at(0);
    auto spec = compute_output_specs(input_tensors, output_tensors)[0];
    return {create_device_tensor(spec, input_tensor.device())};
}

void MulMatDeviceOperation::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    std::cout << input_tensors.size() << ", " << output_tensors.size() << std::endl;
    const auto& a = input_tensors.at(0);
    const auto& b = input_tensors.at(1);
    const auto& a_shape = a.logical_shape();
    const auto& b_shape = b.logical_shape();
    const auto& a_shape4d = a.logical_shape().to_array_4D();
    const auto& b_shape4d = b.logical_shape().to_array_4D();

    TT_FATAL(a.layout() == ttnn::TILE_LAYOUT, "Expected layout TILE_LAYOUT for tensor a");
    TT_FATAL(b.layout() == ttnn::TILE_LAYOUT, "Expected layout TILE_LAYOUT for tensor b");

    TT_FATAL(a_shape4d[3] == b_shape4d[3] &&
        a_shape4d[1] % b_shape4d[1] == 0 && a_shape4d[1] > 0 &&
        a_shape4d[0] % b_shape4d[0] == 0 && a_shape4d[0] > 0 &&
        a_shape4d[2] > 0 && b_shape4d[2] > 0,
        "Expcted format a: [B, N, M, K], b: [B*x, C*x, N, K] but get a: {}, b: {}",
        a_shape, b_shape);

    if(!output_tensors.empty()) {
        const auto& o = output_tensors.at(0);
        const auto& o_shape = o->logical_shape();
        const auto& o_shape4d = o->logical_shape().to_array_4D();
        TT_FATAL(o->layout() == ttnn::TILE_LAYOUT, "Expected layout TILE_LAYOUT for tensor o");
        TT_FATAL(o_shape4d[0] == b_shape4d[0] &&
                 o_shape4d[1] == b_shape4d[1] &&
                 o_shape4d[2] == a_shape4d[2] &&
                 o_shape4d[3] == a_shape4d[3],
                 "Expected output shape: [B*x, C*x, M, N], but got: {}. Input shapes were a: {}, b: {}",
                 o_shape, a_shape, b_shape);
    }
}

tt::tt_metal::operation::ProgramWithCallbacks MulMatDeviceOperation::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const
{
    tt::tt_metal::Program program{};
    const auto& a_tensor = input_tensors.at(0);
    const auto& b_tensor = input_tensors.at(1);
    const auto& o_tensor = output_tensors.at(0);

    const uint32_t K = a_tensor.logical_shape()[-1];
    const uint32_t N = b_tensor.logical_shape()[-2];
    const uint32_t M = o_tensor.logical_shape()[-2];


    tt::tt_metal::IDevice* device = a_tensor.device();

    auto* a = a_tensor.buffer();
    auto* b = b_tensor.buffer();
    auto* o = o_tensor.buffer();

    const uint32_t Kt = K/32 + (K % 32 != 0);
    const uint32_t Nt = N/32 + (N % 32 != 0);
    const uint32_t Mt = M/32 + (M % 32 != 0);

    // auto core_grid = device->compute_with_storage_grid_size();
    CoreCoord core_grid(1, 1);

    auto [num_cores,
        all_cores,
        core_group_1,
        core_group_2,
        work_per_core1,
        work_per_core2] =
        tt::tt_metal::split_work_to_cores(core_grid, Mt*Nt);

    MakeCircularBuffer(program, all_cores, tt::CBIndex::c_0, 4, a_tensor.dtype()); // cb_in0
    MakeCircularBuffer(program, all_cores, tt::CBIndex::c_1, 4, b_tensor.dtype()); // cb_in1
    MakeCircularBuffer(program, all_cores, tt::CBIndex::c_16, 2, o_tensor.dtype()); // cb_out


    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(*a).append_to(reader_compile_time_args);
    TensorAccessorArgs(*b).append_to(reader_compile_time_args);
    KernelHandle reader = CreateMetaliumKernel(program, "mul_mat_reader", all_cores, DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default,
        .compile_args = reader_compile_time_args
    });

    std::vector<uint32_t> writer_compile_time_args;
    TensorAccessorArgs(*o).append_to(writer_compile_time_args);
    KernelHandle writer = CreateMetaliumKernel(program, "mul_mat_writer", all_cores, DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = NOC::RISCV_1_default,
        .compile_args = writer_compile_time_args
    });

    KernelHandle compute = CreateMetaliumKernel(program, "mul_mat_compute", all_cores, ComputeConfig{
        .fp32_dest_acc_en = true,
    });

    uint32_t id = 0;
    auto work_groups = {std::make_pair(core_group_1, work_per_core1), std::make_pair(core_group_2, work_per_core2)};
    for(const auto& [group, work_per_item] : work_groups) {
        for(const auto& range : group.ranges()) {
            for(const auto& core : range) {

                SetRuntimeArgs(program, reader, core, std::vector<uint32_t>{a->address(), b->address(), Mt, Nt, Kt, id, work_per_item});
                SetRuntimeArgs(program, compute, core, std::vector<uint32_t>{Mt, Nt, Kt, id, work_per_item});
                SetRuntimeArgs(program, writer, core, std::vector<uint32_t>{o->address(), Mt, Nt, Kt, id, work_per_item});

                id += work_per_item;
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
            auto* b = input_tensors.at(1).buffer();
            auto* o = output_tensors.at(0).buffer();

            for(const auto& range : all_cores.ranges()) {
                for (const auto& core : range) {
                    {
                        auto& runtime_args = GetRuntimeArgs(program, reader, core);
                        runtime_args[0] = a->address();
                        runtime_args[1] = b->address();
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
