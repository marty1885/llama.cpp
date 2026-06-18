#include <ttnn/tensor/tensor.hpp>
#include <ttnn/tensor/types.hpp>
#include <utility>

namespace ttggml {
using namespace ttnn;

struct MulMatOperation {
    static ttnn::Tensor invoke(const Tensor& a, const Tensor& b, bool high_percision = false);
};

/**
 * Implements GGML's MUL_MAT operation wich computes b @ aT
 */
// TTNN removed the ttnn::register_operation<> decorator framework; this composite op is
// now exposed as a plain forwarding function to MulMatOperation::invoke().
template <typename... Args>
inline ttnn::Tensor mul_mat(Args&&... args) {
    return MulMatOperation::invoke(std::forward<Args>(args)...);
}
}
