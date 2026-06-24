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

namespace wkv7_device {

struct operation_attributes_t {
    tt::tt_metal::MemoryConfig output_mem_config;
    uint32_t S;              // head_size
    uint32_t H;              // head_count
    uint32_t L;              // n_seq_tokens
    uint32_t G;              // n_seqs
    bool     use_decode;     // decode-L kernel (small L) vs chunked-parallel kernel
};

struct tensor_args_t {
    const Tensor& r;
    const Tensor& w;
    const Tensor& k;
    const Tensor& v;
    const Tensor& a;
    const Tensor& b;
    const Tensor& state;
};

using spec_return_value_t = ttnn::TensorSpec;
using tensor_return_value_t = ttnn::Tensor;

namespace program {

struct WKV7ProgramFactory {
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

struct RWKVWKV7DeviceOperation {
    using operation_attributes_t = wkv7_device::operation_attributes_t;
    using tensor_args_t = wkv7_device::tensor_args_t;
    using spec_return_value_t = wkv7_device::spec_return_value_t;
    using tensor_return_value_t = wkv7_device::tensor_return_value_t;
    using program_factory_t = std::variant<program::WKV7ProgramFactory>;

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
        const Tensor& r,
        const Tensor& w,
        const Tensor& k,
        const Tensor& v,
        const Tensor& a,
        const Tensor& b,
        const Tensor& state);
};

} // namespace wkv7_device

// True when WKV7 consumes the row-folded state layout [1,G,Es/32,32] instead of the canonical
// flat-strip. Gated by GGML_METALIUM_WKV7_FOLDED_STATE; the ggml-metalium handler must fold/pass
// the state to match.
bool wkv7_folded_state();

ttnn::Tensor rwkv_wkv7(
    const Tensor& r,
    const Tensor& w,
    const Tensor& k,
    const Tensor& v,
    const Tensor& a,
    const Tensor& b,
    const Tensor& state);

} // namespace ttggml
