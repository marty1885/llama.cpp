#include <ttnn/tensor/tensor.hpp>
#include <ttnn/tensor/types.hpp>
#include <utility>

namespace ttggml {
using namespace ttnn;

// Builds the gather-friendly embedding variant: folds a canonical block-float weight
// (TT shape [1, 1, vocab, embed], TILE) into a bf16 tensor of TT shape
// [1, vocab, embed/32, 32], where variant(v, i, j) = embed[i*32 + j] of vocab v. The
// caller typecasts the result to block-float for residency.
struct EmbeddingFoldVariant {
    static ttnn::Tensor invoke(const Tensor& canonical);
};

// Gathers rows of the folded variant by token index (ROW_MAJOR uint32 [1,1,users,n]) and
// scatters into the ggml-canonical output [1, 1, n_tokens, embed] (bf16, TILE). embed is the
// true (unpadded) embedding width.
struct EmbeddingGather {
    static ttnn::Tensor invoke(const Tensor& variant, const Tensor& index, uint32_t embed);
};
}
