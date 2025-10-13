#include <ttnn/decorators.hpp>
#include <ttnn/run_operation.hpp>
#include <ttnn/tensor/layout/layout.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/tensor/types.hpp>

namespace ttggml {
using namespace ttnn;

struct RoPEOperation {
    static ttnn::Tensor invoke(const Tensor& src_tensor, const Tensor& index_tensor, uint32_t active_dim_size, float freq_base = 10000.0f);
};
constexpr auto rope = ttnn::register_operation<"ttggml::rope", ttggml::RoPEOperation>();

struct RoPEDeviceOperation {
    const tt::tt_metal::MemoryConfig output_mem_config;
    const tt::tt_metal::DataType output_dtype{};
    const uint32_t active_dim_size = 0;
    const float freq_base = 10000.0f;

    void validate_with_output_tensors(
        const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const;
    std::vector<ttnn::TensorSpec> compute_output_specs(
        const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const;

    std::vector<Tensor> create_output_tensors(
        const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const;
    tt::tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;
};
}
