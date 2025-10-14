#include <ttnn/decorators.hpp>
#include <ttnn/run_operation.hpp>
#include <ttnn/tensor/layout/layout.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/tensor/types.hpp>

namespace ttggml {
using namespace ttnn;

struct RoPEOperation {
    static ttnn::Tensor invoke(const Tensor& src_tensor, const Tensor& index_tensor, uint32_t active_dim_size, float freq_base = 10000.0f, float freq_scale = 1.f, float attn_factor = 1.f);
};
constexpr auto rope = ttnn::register_operation<"ttggml::rope", ttggml::RoPEOperation>();
}
