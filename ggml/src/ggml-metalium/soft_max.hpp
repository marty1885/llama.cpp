#include <ttnn/tensor/tensor.hpp>
#include <ttnn/tensor/types.hpp>
#include <utility>

namespace ttggml {
using namespace ttnn;

struct SoftMaxOperation {
    static ttnn::Tensor invoke(const Tensor& a, float scale = 1.f);
    static ttnn::Tensor invoke(const Tensor& a, const Tensor& mask, float scale = 1.f);
};

// TTNN removed the ttnn::register_operation<> decorator framework; expose as a forwarding function.
template <typename... Args>
inline ttnn::Tensor soft_max(Args&&... args) {
    return SoftMaxOperation::invoke(std::forward<Args>(args)...);
}
}
