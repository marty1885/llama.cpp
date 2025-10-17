#include <ttnn/decorators.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/tensor/types.hpp>

namespace ttggml {
using namespace ttnn;

struct MulMatOperation {
    static ttnn::Tensor invoke(const Tensor& a, const Tensor& b);
};

/**
 * Implements GGML's MUL_MAT operation wich computes b @ aT
 */
constexpr auto mul_mat = ttnn::register_operation<"ttggml::mul_mat", ttggml::MulMatOperation>();
}
