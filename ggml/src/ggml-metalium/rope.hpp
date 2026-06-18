#include <ttnn/tensor/tensor.hpp>
#include <ttnn/tensor/types.hpp>
#include <utility>

namespace ttggml {
using namespace ttnn;

enum class RoPEType {
    Normal = 0,
    NeoX = 1,
};

struct RoPEOperation {
    static ttnn::Tensor invoke(const Tensor& src_tensor, const Tensor& index_tensor, uint32_t active_dim_size, RoPEType rope_type, uint32_t n_ctx_orig = 512, float freq_base = 10000.0f, float freq_scale = 1.f
            , float ext_factor = 0.f, float attn_factor = 1.f, float beta_fast = 0.f, float beta_slow = 0.f);
    static ttnn::Tensor invoke(const Tensor& src_tensor, const Tensor& index_tensor, const Tensor& freq_factor, uint32_t active_dim_size, RoPEType rope_type, uint32_t n_ctx_orig = 512, float freq_base = 10000.0f, float freq_scale = 1.f
            , float ext_factor = 0.f, float attn_factor = 1.f, float beta_fast = 0.f, float beta_slow = 0.f);
};
// TTNN removed the ttnn::register_operation<> decorator framework; expose as a forwarding function.
template <typename... Args>
inline ttnn::Tensor rope(Args&&... args) {
    return RoPEOperation::invoke(std::forward<Args>(args)...);
}
}
