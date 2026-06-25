#pragma once

#include <tuple>
#include <variant>

#include <ttnn/device_operation.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/tensor/types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>

namespace ttggml {
using namespace ttnn;

namespace swf_device {

struct operation_attributes_t {
    uint32_t T;    // region-2 row offset (= n_tokens)
    uint32_t H;    // head count
    uint32_t ng;   // groups per head = C/32
};

struct tensor_args_t {
    const Tensor& src;     // wkv_output (canonical, region-2 read from row T)
    const Tensor& cache;   // folded state cache -- written in place (also the op output)
};

using spec_return_value_t   = ttnn::TensorSpec;
using tensor_return_value_t = ttnn::Tensor;

namespace program {

struct SWFProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle dm;
        tt::tt_metal::CoreRangeSet all_cores;
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const operation_attributes_t&,
        const tensor_args_t&,
        tensor_return_value_t&);

    static void override_runtime_arguments(
        cached_program_t&,
        const operation_attributes_t&,
        const tensor_args_t&,
        tensor_return_value_t&);
};

} // namespace program

struct SliceWriteFoldedDeviceOperation {
    using operation_attributes_t = swf_device::operation_attributes_t;
    using tensor_args_t          = swf_device::tensor_args_t;
    using spec_return_value_t     = swf_device::spec_return_value_t;
    using tensor_return_value_t   = swf_device::tensor_return_value_t;
    using program_factory_t       = std::variant<program::SWFProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& src, const Tensor& cache, uint32_t T, uint32_t H, uint32_t ng);
};

} // namespace swf_device

// Scatter WKV7 canonical region-2 (final state) directly into the row-folded cache,
// in place. Returns the cache tensor (as an ordering handle). S==H==64, G==1 only.
ttnn::Tensor slice_write_region2_folded(
    const Tensor& src, const Tensor& cache, uint32_t T, uint32_t H, uint32_t ng);

} // namespace ttggml
