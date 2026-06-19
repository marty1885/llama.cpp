#pragma once

#include <optional>
#include <tuple>
#include <variant>

#include <ttnn/device_operation.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/tensor/types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>

namespace ttggml {
using namespace ttnn;

enum class RoPEType {
    Normal = 0,
    NeoX = 1,
};

namespace rope_device {

struct operation_attributes_t {
    tt::tt_metal::MemoryConfig output_mem_config;
    uint32_t active_dim_size;
    uint32_t n_ctx_orig;
    RoPEType rope_type;
    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;
};

struct tensor_args_t {
    const Tensor& src;
    const Tensor& index;
    std::optional<Tensor> freq_factor;
};

using spec_return_value_t = ttnn::TensorSpec;
using tensor_return_value_t = ttnn::Tensor;

namespace program {

struct RoPEProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader;
        tt::tt_metal::KernelHandle writer;
        tt::tt_metal::CoreRangeSet all_cores;
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

} // namespace program

struct RoPEDeviceOperation {
    using operation_attributes_t = rope_device::operation_attributes_t;
    using tensor_args_t = rope_device::tensor_args_t;
    using spec_return_value_t = rope_device::spec_return_value_t;
    using tensor_return_value_t = rope_device::tensor_return_value_t;
    using program_factory_t = std::variant<program::RoPEProgramFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t&,
        const tensor_args_t&);

    static void validate_on_program_cache_hit(
        const operation_attributes_t&,
        const tensor_args_t&);

    static void validate_on_program_cache_miss(
        const operation_attributes_t&,
        const tensor_args_t&);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t&,
        const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t&,
        const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
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
        float beta_slow);
};

} // namespace rope_device

namespace prim {
ttnn::Tensor rope(
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
    float beta_slow);
} // namespace prim

ttnn::Tensor rope(
    const Tensor& src_tensor,
    const Tensor& index_tensor,
    uint32_t active_dim_size,
    RoPEType rope_type,
    uint32_t n_ctx_orig = 512,
    float freq_base = 10000.0f,
    float freq_scale = 1.f,
    float ext_factor = 0.f,
    float attn_factor = 1.f,
    float beta_fast = 0.f,
    float beta_slow = 0.f);

ttnn::Tensor rope(
    const Tensor& src_tensor,
    const Tensor& index_tensor,
    const Tensor& freq_factor,
    uint32_t active_dim_size,
    RoPEType rope_type,
    uint32_t n_ctx_orig = 512,
    float freq_base = 10000.0f,
    float freq_scale = 1.f,
    float ext_factor = 0.f,
    float attn_factor = 1.f,
    float beta_fast = 0.f,
    float beta_slow = 0.f);

} // namespace ttggml
