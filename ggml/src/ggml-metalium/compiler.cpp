#include "compiler.hpp"
#include "ggml-metalium-internal.hpp"
#include "embedding.hpp"

#include <ttnn/operations/copy/typecast/typecast.hpp>
#include <ttnn/tensor/tensor.hpp>


class MetaliumLowering {
public:
    virtual ~MetaliumLowering() = default;

    virtual std::string_view name() const = 0;
    virtual bool matchesNode(const ggml_tensor * node) const = 0;
    virtual void lowerNode(ggml_backend_metalium_context * ctx, ggml_tensor * node) const = 0;
};

// Quantized token embedding: GET_ROWS gathering rows of a block-float (Qx) weight by
// I32 token id. The generic ttnn::tosa::gather path can't tile-pad block-float dtypes, so
// supports_op rejects this and the scheduler falls back to CPU - which drags the whole
// embedding matrix off device every forward pass. Matching here keeps the weight resident
// and dispatches our own gather/dequant. See generated/embedding-get-rows-spec.md.
class EmbeddingGetRowsLowering final : public MetaliumLowering {
public:
    std::string_view name() const override { return "embedding_get_rows"; }

    bool matchesNode(const ggml_tensor * node) const override {
        if (node->op != GGML_OP_GET_ROWS) {
            return false;
        }
        const ggml_tensor * a = node->src[0];
        const ggml_tensor * b = node->src[1];
        if (a == nullptr || b == nullptr) {
            return false;
        }
        // Block-float weight is exactly the case the generic gather path rejects.
        if (!ggml_is_quantized(a->type)) {
            return false;
        }
        if (b->type != GGML_TYPE_I32) {
            return false;
        }
        // Views of the weight (alias/offset) are a later concern.
        if (a->view_src != nullptr) {
            return false;
        }
        return true;
    }

    void lowerNode(ggml_backend_metalium_context * ctx, ggml_tensor * dst) const override {
        GGML_UNUSED(ctx);
        GGML_METALIUM_OP_SANITY_CHECK(dst);
        GGML_METALIUM_OP_SRC0_SANITY_CHECK(dst);

        auto * a_meta = (ggml_tensor_extra_metalium *)dst->src[0]->extra;

        // Fold the weight into the gather-friendly [1, vocab, embed/32, 32] layout once, then drop
        // the canonical copy: the fold is now the authoritative residency for this weight. The fold
        // op produces bf16; we keep it block-float for residency (the gather dequants in compute).
        if (a_meta->row_folded == nullptr) {
            auto canonical = realize_ggml_view(dst->src[0]);
            auto variant = ttnn::typecast(ttggml::EmbeddingFoldVariant::invoke(*canonical), tt::tt_metal::DataType::BFLOAT8_B);
            a_meta->row_folded = std::make_shared<tt::tt_metal::Tensor>(std::move(variant));
            a_meta->tensor.reset();
        }

        // Gather the indexed rows of the fold and scatter into the ggml-canonical output.
        auto index = realize_ggml_view(dst->src[1]);
        auto gathered = ttggml::EmbeddingGather::invoke(*a_meta->row_folded, *index, (uint32_t)dst->ne[0]);

        auto * dst_meta = (ggml_tensor_extra_metalium *)dst->extra;
        dst_meta->tensor = std::make_shared<tt::tt_metal::Tensor>(std::move(gathered));
    }
};

MetaliumGraphCompiler::MetaliumGraphCompiler() {
    lowerings.push_back(std::make_unique<EmbeddingGetRowsLowering>());
}

MetaliumGraphCompiler::~MetaliumGraphCompiler() = default;

const MetaliumLowering * MetaliumGraphCompiler::findLowering(const ggml_tensor * node) const {
    for (const auto & lowering : lowerings) {
        if (lowering->matchesNode(node)) {
            return lowering.get();
        }
    }
    return nullptr;
}

bool MetaliumGraphCompiler::hasLowering(const ggml_tensor * node) const {
    return findLowering(node) != nullptr;
}

bool MetaliumGraphCompiler::tryLowerNode(ggml_backend_metalium_context * ctx, ggml_tensor * node) const {
    if (const MetaliumLowering * lowering = findLowering(node)) {
        lowering->lowerNode(ctx, node);
        return true;
    }
    return false;
}
