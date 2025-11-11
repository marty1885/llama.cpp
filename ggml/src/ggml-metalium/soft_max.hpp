#include <ttnn/decorators.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/tensor/types.hpp>

namespace ttggml {
using namespace ttnn;

struct SoftMaxOperation {
    static ttnn::Tensor invoke(const Tensor& a);
};

constexpr auto soft_max = ttnn::register_operation<"ttggml::soft_max", ttggml::SoftMaxOperation>();
}
