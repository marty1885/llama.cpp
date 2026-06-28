#include "compiler.hpp"
#include "ggml-metalium-internal.hpp"
#include "embedding.hpp"
#include "ggml.h"

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifdef GGML_METALIUM_HAVE_TTPRM
#include "view_realize_op.hpp"
#endif

#include <ttnn/operations/copy/typecast/typecast.hpp>
#include <ttnn/operations/eltwise/ternary/ternary.hpp>
#include <ttnn/operations/eltwise/ternary/ternary_composite_op.hpp>
#include <ttnn/operations/matmul/matmul.hpp>
#include <ttnn/operations/normalization/layernorm/layernorm.hpp>
#include <ttnn/operations/normalization/rmsnorm/rmsnorm.hpp>
#include <ttnn/tensor/tensor.hpp>

struct FusionFacts {
    const std::unordered_map<const ggml_tensor *, int> * uses = nullptr;

    bool is_private(const ggml_tensor * node) const {
        if (node == nullptr || uses == nullptr) {
            return false;
        }
        auto it = uses->find(node);
        return it != uses->end() && it->second == 1 &&
               (node->flags & GGML_TENSOR_FLAG_OUTPUT) == 0;
    }
};

struct FusionCandidate {
    std::vector<ggml_tensor *> kill_requests;
};

class MetaliumLowering {
public:
    virtual ~MetaliumLowering() = default;

    virtual std::string_view name() const = 0;
    virtual bool matchesNode(const ggml_tensor * node) const = 0;
    virtual void lowerNode(ggml_backend_metalium_context * ctx, ggml_tensor * node) const = 0;
    virtual void resetPlan() const {}
    virtual bool isFusion() const { return false; }
    virtual std::optional<FusionCandidate> matchFusion(
            ggml_tensor * node,
            const FusionFacts & facts) const {
        GGML_UNUSED(node);
        GGML_UNUSED(facts);
        return std::nullopt;
    }
    virtual bool bindFusion(ggml_tensor * node, const FusionFacts & facts) const {
        GGML_UNUSED(node);
        GGML_UNUSED(facts);
        return false;
    }
    virtual bool lowerFusion(
            ggml_backend_metalium_context * ctx,
            ggml_tensor * node) const {
        GGML_UNUSED(ctx);
        GGML_UNUSED(node);
        return false;
    }
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

namespace {

enum class PatternKind {
    Any,
    Op,
    AnyOrder,
};

struct PatternNode;

class Pat {
public:
    Pat() = default;

    Pat of(std::initializer_list<Pat> children) const;
    Pat of(Pat a) const;
    Pat of(Pat a, Pat b) const;
    template <typename Site>
    Pat bind(ggml_tensor * Site::* member) const {
        return bind_erased([member](void * site, ggml_tensor * node) {
            static_cast<Site *>(site)->*member = node;
        });
    }
    template <typename Site>
    Pat guard(bool (*pred)(const Site &)) const {
        return guard_erased([pred](const void * site) {
            return pred(*static_cast<const Site *>(site));
        });
    }
    template <typename Site>
    Pat collect_ttprm_view(ggml_tensor * Site::* param, bool Site::* request) const {
        return bind_erased([param, request](void * site, ggml_tensor * node) {
            Site * s = static_cast<Site *>(site);
            s->*param = node;
            s->*request = true;
        });
    }
    Pat kill() const;
    Pat note(const char * text) const;
    bool valid() const;
    const PatternNode * get() const;

private:
    explicit Pat(std::shared_ptr<PatternNode> node) : node_(std::move(node)) {}

    Pat bind_erased(std::function<void(void *, ggml_tensor *)> binder) const;
    Pat guard_erased(std::function<bool(const void *)> pred) const;

    std::shared_ptr<PatternNode> node_;

    friend Pat any();
    friend Pat op(int op);
    friend Pat op(std::initializer_list<int> ops);
    friend Pat any_order(int op);
};

struct PatternNode {
    PatternKind kind = PatternKind::Any;
    int         op   = GGML_OP_NONE;
    std::vector<int> ops;
    const char * note = nullptr; // diagnostics only
    bool kill_node = false;
    std::vector<Pat> children;
    std::vector<std::function<void(void *, ggml_tensor *)>> binders;
    std::vector<std::function<bool(const void *)>> guards;
};

Pat Pat::of(std::initializer_list<Pat> children) const {
    GGML_ASSERT(node_ != nullptr);
    GGML_ASSERT(node_->children.empty());
    auto copy = std::make_shared<PatternNode>(*node_);
    copy->children.assign(children.begin(), children.end());
    return Pat(copy);
}

Pat Pat::of(Pat a) const {
    return of({ a });
}

Pat Pat::of(Pat a, Pat b) const {
    return of({ a, b });
}

Pat Pat::bind_erased(std::function<void(void *, ggml_tensor *)> binder) const {
    GGML_ASSERT(node_ != nullptr);
    auto copy = std::make_shared<PatternNode>(*node_);
    copy->binders.push_back(std::move(binder));
    return Pat(copy);
}

Pat Pat::guard_erased(std::function<bool(const void *)> pred) const {
    GGML_ASSERT(node_ != nullptr);
    auto copy = std::make_shared<PatternNode>(*node_);
    copy->guards.push_back(std::move(pred));
    return Pat(copy);
}

Pat Pat::kill() const {
    GGML_ASSERT(node_ != nullptr);
    auto copy = std::make_shared<PatternNode>(*node_);
    copy->kill_node = true;
    return Pat(copy);
}

Pat Pat::note(const char * text) const {
    GGML_ASSERT(node_ != nullptr);
    auto copy = std::make_shared<PatternNode>(*node_);
    copy->note = text;
    return Pat(copy);
}

bool Pat::valid() const {
    return node_ != nullptr;
}

const PatternNode * Pat::get() const {
    return node_.get();
}

struct GraphMatch {
    std::vector<std::pair<const PatternNode *, ggml_tensor *>> captures;
    std::vector<ggml_tensor *> kill_requests;

    ggml_tensor * operator[](const Pat & pat) const {
        const PatternNode * key = pat.get();
        for (const auto & it : captures) {
            if (it.first == key) {
                return it.second;
            }
        }
        return nullptr;
    }

    bool bind(const Pat & pat, ggml_tensor * node) {
        const PatternNode * key = pat.get();
        if (key == nullptr) {
            return true;
        }
        for (const auto & it : captures) {
            if (it.first == key) {
                return it.second == node;
            }
        }
        captures.push_back({ key, node });
        return true;
    }

    void request_kill(ggml_tensor * node) {
        kill_requests.push_back(node);
    }
};

Pat any() {
    auto n = std::make_shared<PatternNode>();
    n->kind = PatternKind::Any;
    return Pat(n);
}

Pat op(int op) {
    auto n = std::make_shared<PatternNode>();
    n->kind = PatternKind::Op;
    n->op = op;
    n->ops.push_back(op);
    return Pat(n);
}

Pat op(std::initializer_list<int> ops) {
    GGML_ASSERT(ops.size() > 0);
    auto n = std::make_shared<PatternNode>();
    n->kind = PatternKind::Op;
    n->op = *ops.begin();
    n->ops.assign(ops.begin(), ops.end());
    return Pat(n);
}

Pat any_order(int op) {
    auto n = std::make_shared<PatternNode>();
    n->kind = PatternKind::AnyOrder;
    n->op = op;
    n->ops.push_back(op);
    return Pat(n);
}

template <typename Site>
static bool match_recursive(ggml_tensor * node, const Pat & pat, GraphMatch & m, Site & site) {
    if (node == nullptr || !pat.valid()) {
        return false;
    }

    const PatternNode & p = *pat.get();
    if (p.kind != PatternKind::Any &&
        std::find(p.ops.begin(), p.ops.end(), node->op) == p.ops.end()) {
        return false;
    }

    if (!m.bind(pat, node)) {
        return false;
    }
    for (const auto & binder : p.binders) {
        binder(&site, node);
    }
    if (p.kind == PatternKind::Any) {
        for (const auto & guard : p.guards) {
            if (!guard(&site)) {
                return false;
            }
        }
        if (p.kill_node) {
            m.request_kill(node);
        }
        return true;
    }

    if (p.kind == PatternKind::Op) {
        for (size_t i = 0; i < p.children.size(); ++i) {
            if (i >= GGML_MAX_SRC || !p.children[i].valid()) {
                return false;
            }
            if (!match_recursive(node->src[i], p.children[i], m, site)) {
                return false;
            }
        }
    } else if (p.kind == PatternKind::AnyOrder) {
        if (p.children.size() != 2 || !p.children[0].valid() || !p.children[1].valid()) {
            return false;
        }
        GraphMatch try1 = m;
        Site site1 = site;
        if (match_recursive(node->src[0], p.children[0], try1, site1) &&
            match_recursive(node->src[1], p.children[1], try1, site1)) {
            m = std::move(try1);
            site = std::move(site1);
        } else {
            GraphMatch try2 = m;
            Site site2 = site;
            if (match_recursive(node->src[1], p.children[0], try2, site2) &&
                match_recursive(node->src[0], p.children[1], try2, site2)) {
                m = std::move(try2);
                site = std::move(site2);
            } else {
                return false;
            }
        }
    }
    for (const auto & guard : p.guards) {
        if (!guard(&site)) {
            return false;
        }
    }
    if (p.kill_node) {
        m.request_kill(node);
    }
    return true;
}

template <typename Site>
struct PatternMatch {
    Site site;
    std::vector<ggml_tensor *> kill_requests;
};

template <typename Site>
std::optional<PatternMatch<Site>> match_site(const Pat & pattern, ggml_tensor * root) {
    PatternMatch<Site> out;
    GraphMatch gm;
    if (!match_recursive(root, pattern, gm, out.site)) {
        return std::nullopt;
    }
    out.kill_requests = std::move(gm.kill_requests);
    return out;
}

const char * ggml_unary_act_string(const ggml_tensor * u) {
    if (u == nullptr || u->op != GGML_OP_UNARY) {
        return nullptr;
    }
    switch (ggml_get_unary_op(u)) {
        case GGML_UNARY_OP_SIGMOID: return "sigmoid";
        case GGML_UNARY_OP_RELU:    return "relu";
        case GGML_UNARY_OP_GELU:    return "gelu";
        case GGML_UNARY_OP_SILU:    return "silu";
        case GGML_UNARY_OP_TANH:    return "tanh";
        default:                    return nullptr;
    }
}

bool is_linear_bias(const ggml_tensor * bias, const ggml_tensor * matmul) {
    // ttnn::linear takes a per-output-channel bias vector broadcast over rows, not a full matrix.
    return bias != nullptr && matmul != nullptr &&
           bias->ne[0] == matmul->ne[0] && bias->ne[1] == 1 && bias->ne[2] == 1 && bias->ne[3] == 1;
}

struct HeadOperand {
    ggml_tensor * target = nullptr; // tensor to realize_ggml_view
    int64_t       row0   = 0;       // first live token row to slice (flat producers)
    bool          flat   = false;   // true: target is flat [n_embd, *], gather rows [row0,row0+nt) -> head;
                                     // false: target is already the head tensor, present its full grid
};

// n_embd = head_size*head_count. Absorb a flat->head relayout: a reshape-of-flat, or a contiguous
// full-width row sub-view (the wkv-output view_1d ln_x feeds). Anything else realizes as-is.
inline HeadOperand resolve_head_producer(ggml_tensor * node, int64_t n_embd) {
    HeadOperand op;
    if (node == nullptr) {
        return op;
    }
    if (node->op == GGML_OP_RESHAPE && node->src[0] != nullptr &&
        node->src[0]->ne[0] == n_embd && ggml_is_contiguous(node->src[0])) {
        op.target = node->src[0]; // realize the flat producer, never the head reshape
        op.flat   = true;
        return op;
    }
    if (node->view_src != nullptr && node->view_src->ne[0] == n_embd && ggml_is_contiguous(node) &&
        (node->view_offs % (n_embd * ggml_type_size(node->type))) == 0) {
        op.target = node->view_src;
        op.row0   = node->view_offs / (n_embd * ggml_type_size(node->type));
        op.flat   = true;
        return op;
    }
    op.target = node; // already head-shaped (or unabsorbable): realize directly
    return op;
}

// A flat [n_embd, nt] -> per-head [head_size, head_count, nt] reshape we can absorb as a producer-View.
inline bool is_flat_head_reshape(const ggml_tensor * node, int64_t n_embd, int64_t nt) {
    return node != nullptr && node->op == GGML_OP_RESHAPE && node->src[0] != nullptr &&
           node->src[0]->ne[0] == n_embd && node->src[0]->ne[1] == nt && ggml_is_contiguous(node->src[0]);
}

inline bool is_elemwise_op(const ggml_tensor * n) {
    return n != nullptr && (n->op == GGML_OP_MUL || n->op == GGML_OP_ADD || n->op == GGML_OP_SUB);
}

struct GridLeaf {
    ggml_tensor * target = nullptr;          // tensor to realize_ggml_view
    int64_t       row0   = 0;                // Flat: first live token row to slice
    enum How { Flat, Head, Bcast } how = Head;
    // Flat : realize a flat [n_embd, *] producer, gather rows [row0,row0+nt) -> [hc*nt, hs]
    // Head : already the [head_size, head_count, nt] head tensor -> present its full grid
    // Bcast: a per-(head,lane) [n_embd] / [hs,hc] operand (no token dim) -> [hc, hs] broadcast over nt
};

// Classify how a realize-boundary leaf presents on the [hc*nt, hs] head grid. Reuses resolve_head_producer
// for the flat<->head relayout absorb, then adds the group-broadcast case (weight/bias/r_k).
inline GridLeaf classify_grid_leaf(ggml_tensor * node, int64_t hs, int64_t hc) {
    const int64_t n_embd = hs * hc;
    GridLeaf g;
    HeadOperand op = resolve_head_producer(node, n_embd);
    if (op.flat) {
        g.target = op.target; g.row0 = op.row0; g.how = GridLeaf::Flat;
        return g;
    }
    // Group-broadcast operand: exactly n_embd elements and no token dim ([n_embd] or [hs,hc]) -> [hc,hs].
    if (node != nullptr && ggml_nelements(node) == n_embd && node->ne[2] <= 1 && node->ne[3] <= 1 &&
        (node->ne[1] == 1 || (node->ne[0] == hs && node->ne[1] == hc))) {
        g.target = node; g.how = GridLeaf::Bcast;
        return g;
    }
    if (node != nullptr && node->ne[0] == n_embd) { // flat [n_embd, nt] producer reached directly
        g.target = node; g.how = GridLeaf::Flat;
        return g;
    }
    g.target = node; g.how = GridLeaf::Head; // already head-shaped (or unabsorbable): realize as-is
    return g;
}

struct ViewChain {
    enum Kind { Leaf, Reshape, Binary, Norm } kind = Leaf;
    ggml_tensor *             node = nullptr; // the ggml node this represents
    GridLeaf                  leaf;           // Leaf: how to present the realized producer
    int                       op   = 0;       // Binary: ggml_op (MUL/ADD/SUB)
    float                     eps  = 0.0f;    // Norm
    std::vector<ViewChain>    in;             // children (Binary: 2, Reshape/Norm: 1)
    std::vector<ggml_tensor*> absorbed;       // nodes this subtree makes inert (collected bottom-up)
};

// head_shaped/flat_shaped: the two endpoints of an absorbable relayout on the head grid.
inline bool vc_head_shaped(const ggml_tensor * n, int64_t hs, int64_t hc, int64_t nt) {
    return n != nullptr && n->ne[0] == hs && n->ne[1] == hc && n->ne[2] == nt && n->ne[3] == 1;
}
inline bool vc_flat_shaped(const ggml_tensor * n, int64_t n_embd, int64_t nt) {
    return n != nullptr && n->ne[0] == n_embd && n->ne[1] == nt && n->ne[2] == 1 && n->ne[3] == 1;
}

// Recursively lower `node` into a head-grid view-chain. `is_private(n)` gates absorption: a shared or
// output node can't be privately deleted, so it becomes a Leaf (realize boundary). Always succeeds -- the
// base case realizes the node and views its result.
inline ViewChain analyze_view_chain(ggml_tensor * node, int64_t hs, int64_t hc, int64_t nt,
                                    const std::function<bool(const ggml_tensor *)> & is_private,
                                    bool is_root = true) {
    const int64_t n_embd = hs * hc;
    ViewChain vc;
    vc.node = node;
    auto make_leaf = [&]() -> ViewChain {
        vc.kind = ViewChain::Leaf;
        vc.leaf = classify_grid_leaf(node, hs, hc);
        if (!is_root && is_private(node) && vc.leaf.target != node) {
            vc.absorbed.push_back(node);
        }
        return vc;
    };
    auto recurse = [&](ggml_tensor * child) {
        return analyze_view_chain(child, hs, hc, nt, is_private, /*is_root=*/false);
    };
    auto absorb_children = [&]() {
        vc.absorbed.push_back(node);
        for (const ViewChain & c : vc.in) {
            for (ggml_tensor * a : c.absorbed) {
                vc.absorbed.push_back(a);
            }
        }
    };
    if (node == nullptr) {
        return make_leaf();
    }
    // A shared/output node is a realize boundary even if expressible -- EXCEPT the root, which is the node
    // being computed (privacy only governs whether a PARENT may absorb a child).
    if (!is_root && !is_private(node)) {
        return make_leaf();
    }
    // Absorb a pure flat<->head reshape whose child is itself head-grid-expressible (e.g. the head->flat
    // reshape sitting between a NORM and the affine). It is identity on the head grid -> pass the child
    // through. (A reshape OFF a flat leaf is the upstream case, handled at the Leaf by resolve_head_producer.)
    if (node->op == GGML_OP_RESHAPE && node->src[0] != nullptr) {
        ggml_tensor * child = node->src[0];
        const bool flat_head_pair =
            (vc_head_shaped(node, hs, hc, nt) && vc_flat_shaped(child, n_embd, nt)) ||
            (vc_flat_shaped(node, n_embd, nt) && vc_head_shaped(child, hs, hc, nt));
        if (flat_head_pair && is_private(child)) {
            ViewChain inner = recurse(child);
            if (inner.kind != ViewChain::Leaf) { // only absorb when the child is genuinely a head op
                vc.kind = ViewChain::Reshape;
                vc.in.push_back(std::move(inner));
                absorb_children();
                return vc;
            }
        }
    }
    // Absorb a per-head NORM: NORM(reshape_head(x)) reducing over head_size.
    if (node->op == GGML_OP_NORM && node->src[0] != nullptr && node->src[0]->op == GGML_OP_RESHAPE) {
        ggml_tensor * rh = node->src[0];
        ggml_tensor * x  = rh->src[0];
        if (x != nullptr && node->ne[0] == hs && node->ne[1] == hc && node->ne[2] == nt &&
            node->ne[3] == 1 && ggml_nelements(x) == n_embd * nt && is_private(rh)) {
            vc.kind = ViewChain::Norm;
            memcpy(&vc.eps, node->op_params, sizeof(float));
            vc.in.push_back(recurse(x));
            vc.absorbed.push_back(node);
            vc.absorbed.push_back(rh);
            for (ggml_tensor * a : vc.in[0].absorbed) {
                vc.absorbed.push_back(a);
            }
            return vc;
        }
    }
    // Absorb elementwise MUL/ADD/SUB on the head grid (operands resolve recursively; a Bcast operand such
    // as a per-channel weight is fine -- ttprm broadcasts it over the token groups).
    if (is_elemwise_op(node) && node->src[0] != nullptr && node->src[1] != nullptr) {
        vc.kind = ViewChain::Binary;
        vc.op   = node->op;
        vc.in.push_back(recurse(node->src[0]));
        vc.in.push_back(recurse(node->src[1]));
        absorb_children();
        return vc;
    }
    return make_leaf();
}

} // namespace

class FusionLoweringBase : public MetaliumLowering {
public:
    bool matchesNode(const ggml_tensor * node) const override {
        GGML_UNUSED(node);
        return false;
    }

    void lowerNode(ggml_backend_metalium_context * ctx, ggml_tensor * node) const override {
        GGML_UNUSED(ctx);
        GGML_UNUSED(node);
        GGML_ABORT("fusion lowering called as a single-node lowering");
    }

    bool isFusion() const override {
        return true;
    }
};

// Fusing a linear interpolation idiom into ttnn::lerp.
//   ADD(MUL([REPEAT](SUB(x_prev, cur)), weight), cur)
//
// Only the repeat form used by token-shift can use the ttprm path.
class LerpLowering final : public FusionLoweringBase {
public:
    struct Site {
        ggml_tensor * root   = nullptr;
        ggml_tensor * cur    = nullptr;
        ggml_tensor * x_prev = nullptr;
        ggml_tensor * weight = nullptr;
        ggml_tensor * sub    = nullptr;
        ggml_tensor * mul    = nullptr;
        ggml_tensor * repeat = nullptr;
        std::vector<ggml_tensor *> kill_requests;
    };

    std::string_view name() const override { return "lerp"; }
    void resetPlan() const override { sites_.clear(); }
    std::optional<FusionCandidate> matchFusion(ggml_tensor * node, const FusionFacts & facts) const override;
    bool bindFusion(ggml_tensor * node, const FusionFacts & facts) const override;
    bool lowerFusion(ggml_backend_metalium_context * ctx, ggml_tensor * node) const override;
    std::optional<Site> match(ggml_tensor * add, const FusionFacts & facts) const;
    bool apply(ggml_backend_metalium_context * ctx, const Site & site) const;
private:
    mutable std::unordered_map<const ggml_tensor *, Site> sites_;
};

// Fusing matmul plus vector bias into ttnn::linear.
//   ADD(MUL_MAT(weight, input), bias)
//
// Bias must be a per-output-channel vector.
class LinearLowering final : public FusionLoweringBase {
public:
    struct Site {
        ggml_tensor * root   = nullptr;
        ggml_tensor * weight = nullptr;
        ggml_tensor * input  = nullptr;
        ggml_tensor * bias   = nullptr;
        ggml_tensor * matmul = nullptr;
        std::vector<ggml_tensor *> kill_requests;
    };

    std::string_view name() const override { return "linear"; }
    void resetPlan() const override { sites_.clear(); }
    std::optional<FusionCandidate> matchFusion(ggml_tensor * node, const FusionFacts & facts) const override;
    bool bindFusion(ggml_tensor * node, const FusionFacts & facts) const override;
    bool lowerFusion(ggml_backend_metalium_context * ctx, ggml_tensor * node) const override;
    std::optional<Site> match(ggml_tensor * add, const FusionFacts & facts) const;
    bool apply(ggml_backend_metalium_context * ctx, const Site & site) const;
    static bool shape(const Site & site);
private:
    mutable std::unordered_map<const ggml_tensor *, Site> sites_;
};

// Fusing norm, scale, and bias into affine norm.
//   ADD(MUL(NORM(x) or RMS_NORM(x), weight), bias)
//
// Weight and bias must be per-channel vectors; bias must be a parameter.
class NormAffineLowering final : public FusionLoweringBase {
public:
    struct Site {
        ggml_tensor * root   = nullptr;
        ggml_tensor * norm   = nullptr;
        ggml_tensor * x      = nullptr;
        ggml_tensor * weight = nullptr;
        ggml_tensor * bias   = nullptr;
        bool          is_rms = false;
        ggml_tensor * mul    = nullptr;
        std::vector<ggml_tensor *> kill_requests;
    };

    std::string_view name() const override { return "norm_affine"; }
    void resetPlan() const override { sites_.clear(); }
    std::optional<FusionCandidate> matchFusion(ggml_tensor * node, const FusionFacts & facts) const override;
    bool bindFusion(ggml_tensor * node, const FusionFacts & facts) const override;
    bool lowerFusion(ggml_backend_metalium_context * ctx, ggml_tensor * node) const override;
    std::optional<Site> match(ggml_tensor * add, const FusionFacts & facts) const;
    bool apply(ggml_backend_metalium_context * ctx, const Site & site) const;
    static bool shape(const Site & site);
private:
    mutable std::unordered_map<const ggml_tensor *, Site> sites_;
};

// Same graph shape as NormAffineLowering, but prefer ttprm for plain layer norm.
//   ADD(MUL(NORM(x), weight), bias)
//
// RMS_NORM stays on the generic TTNN lowering unless a ttprm RMS affine policy is added explicitly.
class TtprmNormAffineLowering final : public FusionLoweringBase {
public:
    struct Site {
        ggml_tensor * root   = nullptr;
        ggml_tensor * norm   = nullptr;
        ggml_tensor * x      = nullptr;
        ggml_tensor * weight = nullptr;
        ggml_tensor * bias   = nullptr;
        ggml_tensor * mul    = nullptr;
        std::vector<ggml_tensor *> kill_requests;
    };

    std::string_view name() const override { return "ttprm_norm_affine"; }
    void resetPlan() const override { sites_.clear(); }
    std::optional<FusionCandidate> matchFusion(ggml_tensor * node, const FusionFacts & facts) const override;
    bool bindFusion(ggml_tensor * node, const FusionFacts & facts) const override;
    bool lowerFusion(ggml_backend_metalium_context * ctx, ggml_tensor * node) const override;
    std::optional<Site> match(ggml_tensor * add, const FusionFacts & facts) const;
    bool apply(ggml_backend_metalium_context * ctx, const Site & site) const;
    static bool shape(const Site & site);
private:
    mutable std::unordered_map<const ggml_tensor *, Site> sites_;
};

// Fusing an activation into its producing matmul or linear op.
//   UNARY(MUL_MAT(weight, input))
//   UNARY(ADD(MUL_MAT(weight, input), bias))
//
// Supports the unary activations accepted by the Metalium matmul path.
class ActLowering final : public FusionLoweringBase {
public:
    struct Site {
        ggml_tensor * root      = nullptr;
        ggml_tensor * weight    = nullptr;
        ggml_tensor * input     = nullptr;
        ggml_tensor * bias      = nullptr;
        ggml_tensor * matmul    = nullptr;
        std::string   act;
        bool          is_linear = false;
        std::vector<ggml_tensor *> kill_requests;
    };

    std::string_view name() const override { return "activation"; }
    void resetPlan() const override { sites_.clear(); }
    std::optional<FusionCandidate> matchFusion(ggml_tensor * node, const FusionFacts & facts) const override;
    bool bindFusion(ggml_tensor * node, const FusionFacts & facts) const override;
    bool lowerFusion(ggml_backend_metalium_context * ctx, ggml_tensor * node) const override;
    std::optional<Site> match(ggml_tensor * u, const FusionFacts & facts) const;
    bool apply(ggml_backend_metalium_context * ctx, const Site & site) const;
    static bool linear_shape(const Site & site);
private:
    mutable std::unordered_map<const ggml_tensor *, Site> sites_;
};

// Fusing multiply-add into ttnn::mac.
//   ADD(MUL(a, b), c)
//
// Lower priority than more specific ADD-rooted fusions.
class MacLowering final : public FusionLoweringBase {
public:
    struct Site {
        ggml_tensor * root = nullptr;
        ggml_tensor * a    = nullptr;
        ggml_tensor * b    = nullptr;
        ggml_tensor * c    = nullptr;
        ggml_tensor * mul  = nullptr;
        std::vector<ggml_tensor *> kill_requests;
    };

    std::string_view name() const override { return "mac"; }
    void resetPlan() const override { sites_.clear(); }
    std::optional<FusionCandidate> matchFusion(ggml_tensor * node, const FusionFacts & facts) const override;
    bool bindFusion(ggml_tensor * node, const FusionFacts & facts) const override;
    bool lowerFusion(ggml_backend_metalium_context * ctx, ggml_tensor * node) const override;
    std::optional<Site> match(ggml_tensor * add, const FusionFacts & facts) const;
    bool apply(ggml_backend_metalium_context * ctx, const Site & site) const;
private:
    mutable std::unordered_map<const ggml_tensor *, Site> sites_;
};

// Fusing per-head L2 normalization of a product into ttprm::mul_l2_norm.
//   L2_NORM(RESHAPE(MUL(a, b)))
//
// The reshape must be a flat [n_embd, nt] to [head_size, head_count, nt] view.
class L2HeadNormLowering final : public FusionLoweringBase {
public:
    struct Site {
        ggml_tensor * root       = nullptr;
        ggml_tensor * a          = nullptr;
        ggml_tensor * b          = nullptr;
        int64_t       head_size  = 0;
        int64_t       head_count = 0;
        float         eps        = 0.0f;
        ggml_tensor * mul        = nullptr;
        std::vector<ggml_tensor *> kill_requests;
    };

    std::string_view name() const override { return "l2_head_norm"; }
    void resetPlan() const override { sites_.clear(); }
    std::optional<FusionCandidate> matchFusion(ggml_tensor * node, const FusionFacts & facts) const override;
    bool bindFusion(ggml_tensor * node, const FusionFacts & facts) const override;
    bool lowerFusion(ggml_backend_metalium_context * ctx, ggml_tensor * node) const override;
    std::optional<Site> match(ggml_tensor * l2, const FusionFacts & facts) const;
    bool apply(ggml_backend_metalium_context * ctx, const Site & site) const;
    static bool shape(const Site & site);
private:
    mutable std::unordered_map<const ggml_tensor *, Site> sites_;
};

// Fusing per-head layer norm into ttprm::layer_norm.
//   NORM(RESHAPE(x))
//
// The reshape must be a flat-to-head view with more than one head.
class HeadNormLowering final : public FusionLoweringBase {
public:
    struct Site {
        ggml_tensor * root       = nullptr;
        ggml_tensor * x          = nullptr;
        int64_t       head_size  = 0;
        int64_t       head_count = 0;
        float         eps        = 0.0f;
        std::vector<ggml_tensor *> kill_requests;
    };

    std::string_view name() const override { return "head_norm"; }
    void resetPlan() const override { sites_.clear(); }
    std::optional<FusionCandidate> matchFusion(ggml_tensor * node, const FusionFacts & facts) const override;
    bool bindFusion(ggml_tensor * node, const FusionFacts & facts) const override;
    bool lowerFusion(ggml_backend_metalium_context * ctx, ggml_tensor * node) const override;
    std::optional<Site> match(ggml_tensor * nrm, const FusionFacts & facts) const;
    bool apply(ggml_backend_metalium_context * ctx, const Site & site) const;
    static bool shape(const Site & site);
private:
    mutable std::unordered_map<const ggml_tensor *, Site> sites_;
};

// Fusing per-head norm followed by per-channel affine.
//   ADD(MUL(RESHAPE(NORM(RESHAPE(x))), weight), bias)
//
// Enabled only for the ttprm affine path.
class HeadAffineLowering final : public FusionLoweringBase {
public:
    struct Site {
        ggml_tensor * root       = nullptr;
        ggml_tensor * x          = nullptr;
        ggml_tensor * norm_flat  = nullptr;
        ggml_tensor * weight     = nullptr;
        ggml_tensor * bias       = nullptr;
        int64_t       head_size  = 0;
        int64_t       head_count = 0;
        float         eps        = 0.0f;
        ggml_tensor * norm       = nullptr;
        ggml_tensor * reshape_h  = nullptr;
        ggml_tensor * mul        = nullptr;
        std::vector<ggml_tensor *> kill_requests;
    };

    std::string_view name() const override { return "head_affine"; }
    void resetPlan() const override { sites_.clear(); }
    std::optional<FusionCandidate> matchFusion(ggml_tensor * node, const FusionFacts & facts) const override;
    bool bindFusion(ggml_tensor * node, const FusionFacts & facts) const override;
    bool lowerFusion(ggml_backend_metalium_context * ctx, ggml_tensor * node) const override;
    std::optional<Site> match(ggml_tensor * add, const FusionFacts & facts) const;
    bool apply(ggml_backend_metalium_context * ctx, const Site & site) const;
    static bool shape(const Site & site);
private:
    mutable std::unordered_map<const ggml_tensor *, Site> sites_;
};

// Fusing RWKV rk gate into ttprm::rk.
//   ADD(cur, RESHAPE(MUL(v, SUM_ROWS(MUL(MUL(k, r), r_k)))))
//
// May absorb a private ttprm-expressible cur chain.
class RkGateLowering final : public FusionLoweringBase {
public:
    struct Site {
        ggml_tensor * root       = nullptr;
        ggml_tensor * r          = nullptr;
        ggml_tensor * k          = nullptr;
        ggml_tensor * v          = nullptr;
        ggml_tensor * cur        = nullptr;
        ggml_tensor * r_k        = nullptr;
        int64_t       head_size  = 0;
        int64_t       head_count = 0;
        bool          absorb_cur = false;
        ggml_tensor * reshape = nullptr;
        ggml_tensor * mul_v_rk = nullptr;
        ggml_tensor * rk = nullptr;
        ggml_tensor * mul_kr_rk = nullptr;
        ggml_tensor * mul_kr = nullptr;
        std::vector<ggml_tensor *> kill_requests;
        bool cur_ttprm_view = false;
    };

    std::string_view name() const override { return "rk_gate"; }
    void resetPlan() const override { sites_.clear(); }
    std::optional<FusionCandidate> matchFusion(ggml_tensor * node, const FusionFacts & facts) const override;
    std::optional<Site> match(ggml_tensor * add, const FusionFacts & facts) const;
    bool bindFusion(ggml_tensor * node, const FusionFacts & facts) const override;
    bool lowerFusion(ggml_backend_metalium_context * ctx, ggml_tensor * node) const override;
    bool apply(ggml_backend_metalium_context * ctx, const Site & site) const;
    static bool shape(const Site & site);
private:
    mutable std::unordered_map<const ggml_tensor *, Site> sites_;
};

// Fusing head-grid elementwise ops over absorbable operand views.
//   ADD/MUL/SUB(a, b)
//
// At least one operand must contain a private view chain expressible by ttprm.
class ElemwiseViewLowering final : public FusionLoweringBase {
public:
    struct Site {
        ggml_tensor * root       = nullptr;
        ggml_tensor * op_node    = nullptr;
        ggml_tensor * a          = nullptr;
        ggml_tensor * b          = nullptr;
        int32_t       op         = 0;
        int64_t       head_size  = 0;
        int64_t       head_count = 0;
        std::vector<ggml_tensor *> kill_requests;
    };

    std::string_view name() const override { return "elemwise_view"; }
    void resetPlan() const override { sites_.clear(); }
    std::optional<FusionCandidate> matchFusion(ggml_tensor * node, const FusionFacts & facts) const override;
    bool bindFusion(ggml_tensor * node, const FusionFacts & facts) const override;
    bool lowerFusion(ggml_backend_metalium_context * ctx, ggml_tensor * node) const override;
    std::optional<Site> match(ggml_tensor * node, const FusionFacts & facts) const;
    bool apply(ggml_backend_metalium_context * ctx, const Site & site) const;
    static bool shape(const Site & site);
private:
    mutable std::unordered_map<const ggml_tensor *, Site> sites_;
};

#define DEFINE_FUSION_PLUMBING(Lowering)                                                        \
std::optional<FusionCandidate> Lowering::matchFusion(ggml_tensor * node,                        \
                                                     const FusionFacts & facts) const {          \
    auto site = match(node, facts);                                                              \
    if (!site) {                                                                                 \
        return std::nullopt;                                                                     \
    }                                                                                            \
    FusionCandidate candidate;                                                                   \
    candidate.kill_requests = site->kill_requests;                                               \
    return candidate;                                                                            \
}                                                                                                \
bool Lowering::bindFusion(ggml_tensor * node, const FusionFacts & facts) const {                 \
    auto site = match(node, facts);                                                              \
    if (!site) {                                                                                 \
        return false;                                                                            \
    }                                                                                            \
    sites_[site->root] = std::move(*site);                                                       \
    return true;                                                                                 \
}                                                                                                \
bool Lowering::lowerFusion(ggml_backend_metalium_context * ctx, ggml_tensor * node) const {      \
    auto it = sites_.find(node);                                                                 \
    return it != sites_.end() && apply(ctx, it->second);                                         \
}

DEFINE_FUSION_PLUMBING(LerpLowering)
DEFINE_FUSION_PLUMBING(LinearLowering)
DEFINE_FUSION_PLUMBING(TtprmNormAffineLowering)
DEFINE_FUSION_PLUMBING(NormAffineLowering)
DEFINE_FUSION_PLUMBING(ActLowering)
DEFINE_FUSION_PLUMBING(MacLowering)
DEFINE_FUSION_PLUMBING(L2HeadNormLowering)
DEFINE_FUSION_PLUMBING(HeadNormLowering)
DEFINE_FUSION_PLUMBING(HeadAffineLowering)
DEFINE_FUSION_PLUMBING(RkGateLowering)
DEFINE_FUSION_PLUMBING(ElemwiseViewLowering)

#undef DEFINE_FUSION_PLUMBING

std::optional<LerpLowering::Site>
LerpLowering::match(ggml_tensor * add, const FusionFacts & facts) const {
    GGML_UNUSED(facts);
    static const Pat without_repeat = [] {
        Pat cur = any().bind(&LerpLowering::Site::cur);
        Pat sub = op(GGML_OP_SUB)
            .bind(&LerpLowering::Site::sub)
            .kill()
            .of(any().bind(&LerpLowering::Site::x_prev), cur);
        Pat mul = any_order(GGML_OP_MUL)
            .bind(&LerpLowering::Site::mul)
            .kill()
            .of(sub, any().bind(&LerpLowering::Site::weight));
        return any_order(GGML_OP_ADD)
            .bind(&LerpLowering::Site::root)
            .of(mul, cur);
    }();
    static const Pat with_repeat = [] {
        Pat cur = any().bind(&LerpLowering::Site::cur);
        Pat sub = op(GGML_OP_SUB)
            .bind(&LerpLowering::Site::sub)
            .kill()
            .of(any().bind(&LerpLowering::Site::x_prev), cur);
        Pat repeat = op(GGML_OP_REPEAT)
            .bind(&LerpLowering::Site::repeat)
            .kill()
            .of(sub);
        Pat mul = any_order(GGML_OP_MUL)
            .bind(&LerpLowering::Site::mul)
            .kill()
            .of(repeat, any().bind(&LerpLowering::Site::weight));
        return any_order(GGML_OP_ADD)
            .bind(&LerpLowering::Site::root)
            .of(mul, cur);
    }();

    const Pat * forms[] = { &with_repeat, &without_repeat };
    for (const Pat * pattern : forms) {
        if (auto r = match_site<LerpLowering::Site>(*pattern, add)) {
            LerpLowering::Site site = std::move(r->site);
            site.kill_requests = std::move(r->kill_requests);
            return site;
        }
    }
    return std::nullopt;
}

bool LinearLowering::shape(const LinearLowering::Site & site) {
    return is_linear_bias(site.bias, site.matmul);
}

std::optional<LinearLowering::Site>
LinearLowering::match(ggml_tensor * add, const FusionFacts & facts) const {
    GGML_UNUSED(facts);
    static const Pat pattern = [] {
        Pat matmul = op(GGML_OP_MUL_MAT)
            .bind(&LinearLowering::Site::matmul)
            .kill()
            .of(any().bind(&LinearLowering::Site::weight), any().bind(&LinearLowering::Site::input));
        return any_order(GGML_OP_ADD)
            .bind(&LinearLowering::Site::root)
            .guard(&LinearLowering::shape)
            .of(matmul, any().bind(&LinearLowering::Site::bias));
    }();

    auto r = match_site<LinearLowering::Site>(pattern, add);
    if (!r) {
        return std::nullopt;
    }
    LinearLowering::Site site = std::move(r->site);
    site.kill_requests = std::move(r->kill_requests);
    return site;
}

bool NormAffineLowering::shape(const NormAffineLowering::Site & site) {
    ggml_tensor * nrm = site.norm;
    ggml_tensor * x = site.x;
    ggml_tensor * weight = site.weight;
    ggml_tensor * bias = site.bias;
    if (nrm == nullptr || x == nullptr || weight == nullptr || bias == nullptr) {
        return false;
    }
    if (x->ne[0] == 1) {
        return false;
    }
    const int64_t n = nrm->ne[0];
    if (weight->ne[0] != n || weight->ne[1] != 1 || weight->ne[2] != 1 || weight->ne[3] != 1) {
        return false;
    }
    if (bias->ne[0] != n || bias->ne[1] != 1 || bias->ne[2] != 1 || bias->ne[3] != 1) {
        return false;
    }
#ifdef GGML_METALIUM_HAVE_TTPRM
    if (site.norm->op == GGML_OP_NORM) {
        return false;
    }
#endif
    return bias->op == GGML_OP_NONE;
}

std::optional<NormAffineLowering::Site>
NormAffineLowering::match(ggml_tensor * add, const FusionFacts & facts) const {
    GGML_UNUSED(facts);
    static const Pat pattern = [] {
        Pat norm = op({ GGML_OP_NORM, GGML_OP_RMS_NORM })
            .bind(&NormAffineLowering::Site::norm)
            .kill()
            .of(any().bind(&NormAffineLowering::Site::x));
        Pat mul = any_order(GGML_OP_MUL)
            .bind(&NormAffineLowering::Site::mul)
            .kill()
            .of(norm, any().bind(&NormAffineLowering::Site::weight));
        return any_order(GGML_OP_ADD)
            .bind(&NormAffineLowering::Site::root)
            .guard(&NormAffineLowering::shape)
            .of(mul, any().bind(&NormAffineLowering::Site::bias));
    }();

    auto r = match_site<NormAffineLowering::Site>(pattern, add);
    if (!r) {
        return std::nullopt;
    }
    NormAffineLowering::Site site = std::move(r->site);
    site.is_rms = site.norm->op == GGML_OP_RMS_NORM;
    site.kill_requests = std::move(r->kill_requests);
    return site;
}

bool TtprmNormAffineLowering::shape(const TtprmNormAffineLowering::Site & site) {
    ggml_tensor * nrm = site.norm;
    ggml_tensor * x = site.x;
    ggml_tensor * weight = site.weight;
    ggml_tensor * bias = site.bias;
    if (nrm == nullptr || x == nullptr || weight == nullptr || bias == nullptr) {
        return false;
    }
    if (nrm->op != GGML_OP_NORM || x->ne[0] == 1 || x->ne[0] != nrm->ne[0]) {
        return false;
    }
    const int64_t n = nrm->ne[0];
    if (weight->ne[0] != n || weight->ne[1] != 1 || weight->ne[2] != 1 || weight->ne[3] != 1) {
        return false;
    }
    if (bias->ne[0] != n || bias->ne[1] != 1 || bias->ne[2] != 1 || bias->ne[3] != 1) {
        return false;
    }
    return bias->op == GGML_OP_NONE;
}

std::optional<TtprmNormAffineLowering::Site>
TtprmNormAffineLowering::match(ggml_tensor * add, const FusionFacts & facts) const {
#ifndef GGML_METALIUM_HAVE_TTPRM
    GGML_UNUSED(add);
    GGML_UNUSED(facts);
    return std::nullopt;
#else
    GGML_UNUSED(facts);
    static const Pat pattern = [] {
        Pat norm = op(GGML_OP_NORM)
            .bind(&TtprmNormAffineLowering::Site::norm)
            .kill()
            .of(any().bind(&TtprmNormAffineLowering::Site::x));
        Pat mul = any_order(GGML_OP_MUL)
            .bind(&TtprmNormAffineLowering::Site::mul)
            .kill()
            .of(norm, any().bind(&TtprmNormAffineLowering::Site::weight));
        return any_order(GGML_OP_ADD)
            .bind(&TtprmNormAffineLowering::Site::root)
            .guard(&TtprmNormAffineLowering::shape)
            .of(mul, any().bind(&TtprmNormAffineLowering::Site::bias));
    }();

    auto r = match_site<TtprmNormAffineLowering::Site>(pattern, add);
    if (!r) {
        return std::nullopt;
    }
    TtprmNormAffineLowering::Site site = std::move(r->site);
    site.kill_requests = std::move(r->kill_requests);
    return site;
#endif
}

std::optional<MacLowering::Site>
MacLowering::match(ggml_tensor * add, const FusionFacts & facts) const {
    GGML_UNUSED(facts);
    static const Pat pattern = [] {
        Pat mul = op(GGML_OP_MUL)
            .bind(&MacLowering::Site::mul)
            .kill()
            .of(any().bind(&MacLowering::Site::a), any().bind(&MacLowering::Site::b));
        return any_order(GGML_OP_ADD)
            .bind(&MacLowering::Site::root)
            .of(mul, any().bind(&MacLowering::Site::c));
    }();

    auto r = match_site<MacLowering::Site>(pattern, add);
    if (!r) {
        return std::nullopt;
    }
    MacLowering::Site site = std::move(r->site);
    site.kill_requests = std::move(r->kill_requests);
    return site;
}

bool ActLowering::linear_shape(const ActLowering::Site & site) {
    return is_linear_bias(site.bias, site.matmul);
}

std::optional<ActLowering::Site>
ActLowering::match(ggml_tensor * u, const FusionFacts & facts) const {
    GGML_UNUSED(facts);
    const char * act = ggml_unary_act_string(u);
    if (act == nullptr) {
        return std::nullopt;
    }

    static const Pat bare = [] {
        Pat matmul = op(GGML_OP_MUL_MAT)
            .kill()
            .of(any().bind(&ActLowering::Site::weight), any().bind(&ActLowering::Site::input));
        return op(GGML_OP_UNARY).bind(&ActLowering::Site::root).of(matmul);
    }();

    static const Pat linear = [] {
        Pat matmul = op(GGML_OP_MUL_MAT)
            .bind(&ActLowering::Site::matmul)
            .kill()
            .of(any().bind(&ActLowering::Site::weight), any().bind(&ActLowering::Site::input));
        Pat add = any_order(GGML_OP_ADD)
            .kill()
            .of(matmul, any().bind(&ActLowering::Site::bias));
        return op(GGML_OP_UNARY)
            .bind(&ActLowering::Site::root)
            .guard(&ActLowering::linear_shape)
            .of(add);
    }();

    const Pat * forms[] = { &linear, &bare };
    for (const Pat * pattern : forms) {
        if (auto r = match_site<ActLowering::Site>(*pattern, u)) {
            ActLowering::Site site = std::move(r->site);
            site.kill_requests = std::move(r->kill_requests);
            site.act = act;
            site.is_linear = site.bias != nullptr;
            return site;
        }
    }
    return std::nullopt;
}

bool L2HeadNormLowering::shape(const L2HeadNormLowering::Site & site) {
    ggml_tensor * l2 = site.root;
    ggml_tensor * mul = site.mul;
    if (l2 == nullptr || mul == nullptr) {
        return false;
    }
    if (l2->ne[1] <= 1 || l2->ne[3] != 1) {
        return false;
    }
    return l2->ne[0] * l2->ne[1] == mul->ne[0] && mul->ne[1] == l2->ne[2];
}

std::optional<L2HeadNormLowering::Site>
L2HeadNormLowering::match(ggml_tensor * l2, const FusionFacts & facts) const {
#ifndef GGML_METALIUM_HAVE_TTPRM
    GGML_UNUSED(l2);
    GGML_UNUSED(facts);
    return std::nullopt;
#else
    GGML_UNUSED(facts);
    static const Pat pattern = [] {
        Pat mul = op(GGML_OP_MUL)
            .bind(&L2HeadNormLowering::Site::mul)
            .of(any().bind(&L2HeadNormLowering::Site::a), any().bind(&L2HeadNormLowering::Site::b));
        Pat reshape = op(GGML_OP_RESHAPE).of(mul);
        return op(GGML_OP_L2_NORM)
            .bind(&L2HeadNormLowering::Site::root)
            .guard(&L2HeadNormLowering::shape)
            .of(reshape);
    }();

    auto r = match_site<L2HeadNormLowering::Site>(pattern, l2);
    if (!r) {
        return std::nullopt;
    }
    L2HeadNormLowering::Site site = std::move(r->site);
    site.head_size = site.root->ne[0];
    site.head_count = site.root->ne[1];
    memcpy(&site.eps, site.root->op_params, sizeof(site.eps));
    return site;
#endif
}

bool HeadNormLowering::shape(const HeadNormLowering::Site & site) {
    ggml_tensor * nrm = site.root;
    ggml_tensor * x = site.x;
    if (nrm == nullptr || x == nullptr) {
        return false;
    }
    if (nrm->ne[1] <= 1 || nrm->ne[3] != 1) {
        return false;
    }
    return ggml_nelements(x) == nrm->ne[0] * nrm->ne[1] * nrm->ne[2];
}

std::optional<HeadNormLowering::Site>
HeadNormLowering::match(ggml_tensor * nrm, const FusionFacts & facts) const {
#ifndef GGML_METALIUM_HAVE_TTPRM
    GGML_UNUSED(nrm);
    GGML_UNUSED(facts);
    return std::nullopt;
#else
    GGML_UNUSED(facts);
    static const Pat pattern = [] {
        Pat reshape = op(GGML_OP_RESHAPE).of(any().bind(&HeadNormLowering::Site::x));
        return op(GGML_OP_NORM)
            .bind(&HeadNormLowering::Site::root)
            .guard(&HeadNormLowering::shape)
            .of(reshape);
    }();

    auto r = match_site<HeadNormLowering::Site>(pattern, nrm);
    if (!r) {
        return std::nullopt;
    }
    HeadNormLowering::Site site = std::move(r->site);
    site.head_size = site.root->ne[0];
    site.head_count = site.root->ne[1];
    memcpy(&site.eps, site.root->op_params, sizeof(site.eps));
    return site;
#endif
}

bool HeadAffineLowering::shape(const HeadAffineLowering::Site & site) {
    ggml_tensor * add = site.root;
    ggml_tensor * x = site.x;
    ggml_tensor * nrm = site.norm;
    ggml_tensor * norm_flat = site.norm_flat;
    ggml_tensor * weight = site.weight;
    ggml_tensor * bias = site.bias;
    if (add == nullptr || x == nullptr || nrm == nullptr || norm_flat == nullptr ||
        weight == nullptr || bias == nullptr) {
        return false;
    }
    const int64_t hs = nrm->ne[0];
    const int64_t hc = nrm->ne[1];
    const int64_t nt = nrm->ne[2];
    const int64_t n_embd = hs * hc;
    if (hc <= 1 || nrm->ne[3] != 1) {
        return false;
    }
    if (norm_flat->ne[0] != n_embd || norm_flat->ne[1] != nt) {
        return false;
    }
    if (weight->ne[0] != n_embd || weight->ne[1] != 1 || weight->ne[2] != 1 || weight->ne[3] != 1) {
        return false;
    }
    if (bias->ne[0] != n_embd || bias->ne[1] != 1 || bias->ne[2] != 1 || bias->ne[3] != 1 ||
        bias->op != GGML_OP_NONE) {
        return false;
    }
    return ggml_nelements(x) == n_embd * nt;
}

std::optional<HeadAffineLowering::Site>
HeadAffineLowering::match(ggml_tensor * add, const FusionFacts & facts) const {
#ifndef GGML_METALIUM_HAVE_TTPRM
    GGML_UNUSED(add);
    GGML_UNUSED(facts);
    return std::nullopt;
#else
    GGML_UNUSED(facts);
    // if (getenv("GGML_METALIUM_TTPRM_AFFINE") == nullptr) {
    //     return std::nullopt;
    // }
    static const Pat pattern = [] {
        Pat reshape_h = op(GGML_OP_RESHAPE)
            .bind(&HeadAffineLowering::Site::reshape_h)
            .kill()
            .of(any().bind(&HeadAffineLowering::Site::x));
        Pat norm = op(GGML_OP_NORM)
            .bind(&HeadAffineLowering::Site::norm)
            .kill()
            .of(reshape_h);
        Pat norm_flat = op(GGML_OP_RESHAPE)
            .bind(&HeadAffineLowering::Site::norm_flat)
            .kill()
            .of(norm);
        Pat mul = any_order(GGML_OP_MUL)
            .bind(&HeadAffineLowering::Site::mul)
            .kill()
            .of(norm_flat, any().bind(&HeadAffineLowering::Site::weight));
        return any_order(GGML_OP_ADD)
            .bind(&HeadAffineLowering::Site::root)
            .guard(&HeadAffineLowering::shape)
            .of(mul, any().bind(&HeadAffineLowering::Site::bias));
    }();

    auto r = match_site<HeadAffineLowering::Site>(pattern, add);
    if (!r) {
        return std::nullopt;
    }
    HeadAffineLowering::Site site = std::move(r->site);
    site.head_size = site.norm->ne[0];
    site.head_count = site.norm->ne[1];
    memcpy(&site.eps, site.norm->op_params, sizeof(site.eps));
    site.kill_requests = std::move(r->kill_requests);
    return site;
#endif
}

bool RkGateLowering::shape(const RkGateLowering::Site & site) {
    ggml_tensor * add = site.root;
    ggml_tensor * cur = site.cur;
    ggml_tensor * r   = site.r;
    ggml_tensor * k   = site.k;
    ggml_tensor * v   = site.v;
    ggml_tensor * rkw = site.r_k;
    ggml_tensor * rk  = site.rk;
    if (add == nullptr || cur == nullptr || r == nullptr || k == nullptr ||
        v == nullptr || rkw == nullptr || rk == nullptr) {
        return false;
    }

    const int64_t n_embd = add->ne[0];
    const int64_t nt     = add->ne[1];
    if (add->ne[2] != 1 || add->ne[3] != 1) {
        return false;
    }

    const int64_t hs = v->ne[0];
    const int64_t hc = v->ne[1];
    if (hc <= 1 || hs * hc != n_embd || v->ne[2] != nt) {
        return false;
    }
    if (k->ne[0] != hs || k->ne[1] != hc || r->ne[0] != hs || r->ne[1] != hc) {
        return false;
    }
    if (rk->ne[0] != 1 || rk->ne[1] != hc || rkw->ne[0] != hs || rkw->ne[1] != hc) {
        return false;
    }
    return true;
}

std::optional<RkGateLowering::Site>
RkGateLowering::match(ggml_tensor * add, const FusionFacts & facts) const {
#ifndef GGML_METALIUM_HAVE_TTPRM
    GGML_UNUSED(add);
    GGML_UNUSED(facts);
    return std::nullopt;
#else
    static const Pat pattern = [] {
        Pat k = any().bind(&RkGateLowering::Site::k);
        Pat r = any().bind(&RkGateLowering::Site::r);
        Pat r_k = any().bind(&RkGateLowering::Site::r_k);
        Pat v = any().bind(&RkGateLowering::Site::v);
        Pat cur = any().collect_ttprm_view(&RkGateLowering::Site::cur, &RkGateLowering::Site::cur_ttprm_view);
        Pat mul_kr = any_order(GGML_OP_MUL)
            .bind(&RkGateLowering::Site::mul_kr)
            .kill()
            .of(k, r);
        Pat mul_kr_rk = any_order(GGML_OP_MUL)
            .bind(&RkGateLowering::Site::mul_kr_rk)
            .kill()
            .of(mul_kr, r_k);
        Pat rk = op(GGML_OP_SUM_ROWS)
            .bind(&RkGateLowering::Site::rk)
            .kill()
            .of(mul_kr_rk);
        Pat mul_v_rk = any_order(GGML_OP_MUL)
            .bind(&RkGateLowering::Site::mul_v_rk)
            .kill()
            .of(v, rk);
        Pat reshape = op(GGML_OP_RESHAPE)
            .bind(&RkGateLowering::Site::reshape)
            .kill()
            .of(mul_v_rk);
        return any_order(GGML_OP_ADD)
            .bind(&RkGateLowering::Site::root)
            .guard(&RkGateLowering::shape)
            .of(cur, reshape);
    }();

    auto r = match_site<RkGateLowering::Site>(pattern, add);
    if (!r) {
        return std::nullopt;
    }

    RkGateLowering::Site site = std::move(r->site);
    site.kill_requests = std::move(r->kill_requests);
    site.head_size = site.v->ne[0];
    site.head_count = site.v->ne[1];
    const int64_t nt = site.root->ne[1];
    if (site.cur_ttprm_view && facts.is_private(site.cur) && (site.head_count % 32) == 0) {
        auto is_private = [&](const ggml_tensor * n) {
            return facts.is_private(n);
        };
        ViewChain cur_chain = analyze_view_chain(site.cur, site.head_size, site.head_count, nt, is_private);
        if (cur_chain.kind != ViewChain::Leaf) {
            site.absorb_cur = true;
            for (ggml_tensor * n : cur_chain.absorbed) {
                site.kill_requests.push_back(n);
            }
        }
    }
    return site;
#endif
}

bool ElemwiseViewLowering::shape(const ElemwiseViewLowering::Site & site) {
    ggml_tensor * op_node = site.op_node;
    ggml_tensor * a = site.a;
    ggml_tensor * b = site.b;
    if (op_node == nullptr || a == nullptr || b == nullptr) {
        return false;
    }
    if (op_node->ne[3] != 1) {
        return false;
    }
    const int64_t hs = op_node->ne[0];
    const int64_t hc = op_node->ne[1];
    const int64_t nt = op_node->ne[2];
    if (hc <= 1) {
        return false;
    }
    auto operand_ok = [&](const ggml_tensor * x) {
        return x != nullptr && x->ne[0] == hs && x->ne[1] == hc && x->ne[2] == nt && x->ne[3] == 1;
    };
    if (!operand_ok(a) || !operand_ok(b)) {
        return false;
    }
    return true;
}

std::optional<ElemwiseViewLowering::Site>
ElemwiseViewLowering::match(ggml_tensor * node, const FusionFacts & facts) const {
#ifndef GGML_METALIUM_HAVE_TTPRM
    GGML_UNUSED(node);
    GGML_UNUSED(facts);
    return std::nullopt;
#else
    if (!is_elemwise_op(node) || node->src[0] == nullptr || node->src[1] == nullptr) {
        return std::nullopt;
    }

    Site site;
    site.root       = node;
    site.op_node    = node;
    site.a          = node->src[0];
    site.b          = node->src[1];
    site.op         = (int32_t) node->op;
    site.head_size  = node->ne[0];
    site.head_count = node->ne[1];
    if (!shape(site)) {
        return std::nullopt;
    }

    const int64_t hs     = site.head_size;
    const int64_t hc     = site.head_count;
    const int64_t nt     = site.op_node->ne[2];
    const int64_t n_embd = hs * hc;

    auto is_private = [&](const ggml_tensor * n) { return facts.is_private(n); };
    ViewChain a_chain = analyze_view_chain(site.a, hs, hc, nt, is_private, /*is_root=*/false);
    ViewChain b_chain = analyze_view_chain(site.b, hs, hc, nt, is_private, /*is_root=*/false);

    auto request_kill = [&](ggml_tensor * n) {
        if (n == nullptr) {
            return;
        }
        if (std::find(site.kill_requests.begin(), site.kill_requests.end(), n) == site.kill_requests.end()) {
            site.kill_requests.push_back(n);
        }
    };
    auto request_absorbed = [&](ggml_tensor * operand, const ViewChain & chain) {
        for (ggml_tensor * n : chain.absorbed) {
            request_kill(n);
        }
        if (chain.kind == ViewChain::Leaf && chain.leaf.target != operand) {
            request_kill(operand);
        }
        if (is_flat_head_reshape(operand, n_embd, nt)) {
            request_kill(operand);
        }
    };
    request_absorbed(site.a, a_chain);
    request_absorbed(site.b, b_chain);

    if (site.kill_requests.empty()) {
        return std::nullopt;
    }
    return site;
#endif
}

void MetaliumGraphCompiler::analyzeGraph(const ggml_cgraph * cgraph) {
    inert_.clear();
    planned_roots_.clear();
    uses_.clear();
    for (const auto & lowering : lowerings) {
        lowering->resetPlan();
    }
    if (cgraph == nullptr) {
        return;
    }
    const uint64_t key = metalium_graph_key(cgraph);
    auto it = plan_cache_.find(key);
    if (it == plan_cache_.end()) {
        it = plan_cache_.emplace(key, buildPlan(cgraph)).first;
    }
    bindPlan(cgraph, it->second);
}

MetaliumGraphCompiler::FusionPlan MetaliumGraphCompiler::buildPlan(const ggml_cgraph * cgraph) const {
    FusionPlan plan;

    std::unordered_map<const ggml_tensor *, int> index_of;
    std::unordered_map<const ggml_tensor *, int> uses;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        index_of[cgraph->nodes[i]] = i;
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            if (cgraph->nodes[i]->src[s] != nullptr) {
                uses[cgraph->nodes[i]->src[s]]++;
            }
        }
    }
    FusionFacts facts{ &uses };

    std::unordered_set<const ggml_tensor *> claimed;

    auto commit = [&](const MetaliumLowering * lowering, int root_idx, const FusionCandidate & candidate) {
        plan.roots.push_back({ root_idx, lowering });
        claimed.insert(cgraph->nodes[root_idx]);
        for (ggml_tensor * n : candidate.kill_requests) {
            auto it = index_of.find(n);
            if (n != nullptr && it != index_of.end()) {
                plan.inert_idx.push_back(it->second);
                claimed.insert(n);
            }
        }
    };

    auto blocked = [&](const FusionCandidate & candidate) {
        for (const ggml_tensor * n : candidate.kill_requests) {
            if (!facts.is_private(n) || claimed.find(n) != claimed.end() || index_of.find(n) == index_of.end()) {
                return true;
            }
        }
        return false;
    };

    for (const auto & lowering : lowerings) {
        if (!lowering->isFusion()) {
            continue;
        }
        for (int i = 0; i < cgraph->n_nodes; i++) {
            ggml_tensor * node = cgraph->nodes[i];
            if (claimed.find(node) != claimed.end()) {
                continue;
            }
            std::optional<FusionCandidate> candidate = lowering->matchFusion(node, facts);
            if (!candidate || blocked(*candidate)) {
                continue;
            }
            commit(lowering.get(), i, *candidate);
        }
    }
    return plan;
}

void MetaliumGraphCompiler::bindPlan(const ggml_cgraph * cgraph, const FusionPlan & plan) {
    auto node = [&](int idx) { return cgraph->nodes[idx]; };

    uses_.clear();
    for (int i = 0; i < cgraph->n_nodes; i++) {
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            if (cgraph->nodes[i]->src[s] != nullptr) {
                uses_[cgraph->nodes[i]->src[s]]++;
            }
        }
    }
    FusionFacts facts{ &uses_ };

    for (int idx : plan.inert_idx) {
        inert_.insert(node(idx));
    }

    for (const FusionPlan::PlannedRoot & root : plan.roots) {
        if (root.lowering == nullptr || root.root_idx < 0 || root.root_idx >= cgraph->n_nodes) {
            continue;
        }
        ggml_tensor * root_node = node(root.root_idx);
        if (root.lowering->bindFusion(root_node, facts)) {
            planned_roots_[root_node] = root.lowering;
        }
    }
}

bool MetaliumGraphCompiler::isInert(const ggml_tensor * node) const {
    return inert_.find(node) != inert_.end();
}

bool LerpLowering::apply(ggml_backend_metalium_context * ctx, const Site & site) const {
    GGML_UNUSED(ctx);
    GGML_METALIUM_OP_SANITY_CHECK(site.root);

    auto cur    = realize_ggml_view(site.cur);
    auto x_prev = realize_ggml_view(site.x_prev);
    auto weight = realize_ggml_view(site.weight);

    // ttnn::lerp requires start/end/weight share a dtype; activations and the weight may differ.
    const auto dt = cur->dtype();
    auto match_dtype = [&](std::shared_ptr<tt::tt_metal::Tensor> & t) {
        if (t->dtype() != dt) {
            t = std::make_shared<tt::tt_metal::Tensor>(ttnn::typecast(*t, dt));
        }
    };
    match_dtype(x_prev);
    match_dtype(weight);

#ifdef GGML_METALIUM_HAVE_TTPRM
    const int64_t G = site.root->ne[3];        // gate-group count: 6 or 5 for time-mix, 1 otherwise
    if (dt == tt::tt_metal::DataType::BFLOAT16 && G > 1 && site.root->ne[2] == 1) {
        ttprm::View a = ttprm::view_of(*cur);     // [nt, n_embd]
        ttprm::View b = ttprm::view_of(*x_prev);  // [nt, n_embd]
        ttprm::View w = ttprm::view_of(*weight);  // [G,  n_embd]  (the per-group lerp weight)
        const bool is_grouped_tokenshift =
            w.rows() == G && a.rows() == site.root->ne[1] && a.cols() == site.root->ne[0] &&
            b.rows() == site.root->ne[1] && b.cols() == site.root->ne[0];
        if (is_grouped_tokenshift) {
            auto res = ttprm::lerp(a, b, w, ttprm::view_of_shape(a.rows() * G, a.cols()),
                                   /*canonical_out=*/true);
            if (res) {
                auto * meta = (ggml_tensor_extra_metalium *)site.root->extra;
                meta->tensor = std::make_shared<tt::tt_metal::Tensor>(res.value());
                return true;
            }
        }
    }
#endif

    auto out = ttnn::lerp(*cur, *x_prev, *weight);
    auto * meta = (ggml_tensor_extra_metalium *)site.root->extra;
    meta->tensor = std::make_shared<tt::tt_metal::Tensor>(std::move(out));
    return true;
}

bool LinearLowering::apply(ggml_backend_metalium_context * ctx, const Site & site) const {
    GGML_UNUSED(ctx);
    GGML_METALIUM_OP_SANITY_CHECK(site.root);

    auto a    = realize_ggml_view(site.weight); // matmul src0
    auto b    = realize_ggml_view(site.input);  // matmul src1
    auto bias = realize_ggml_view(site.bias);

    // linear adds bias in the output dtype (input_a == b); make the bias agree.
    if (bias->dtype() != b->dtype()) {
        bias = std::make_shared<tt::tt_metal::Tensor>(ttnn::typecast(*bias, b->dtype()));
    }

    auto out = ttnn::operations::matmul::linear(
        *b, *a,
        /* bias                   = */ *bias,
        /* transpose_a            = */ false,
        /* transpose_b            = */ true,
        /* memory_config          = */ std::nullopt,
        /* dtype                  = */ std::nullopt,
        /* program_config         = */ std::nullopt,
        /* activation             = */ std::nullopt,
        /* compute_kernel_config  = */ make_compute_kernel_config(b->device()));

    auto * meta = (ggml_tensor_extra_metalium *)site.root->extra;
    meta->tensor = std::make_shared<tt::tt_metal::Tensor>(std::move(out));
    return true;
}

bool NormAffineLowering::apply(ggml_backend_metalium_context * ctx, const Site & site) const {
    GGML_UNUSED(ctx);
    GGML_METALIUM_OP_SANITY_CHECK(site.root);

    auto x      = realize_ggml_view(site.x);
    auto weight = realize_ggml_view(site.weight);
    auto bias   = realize_ggml_view(site.bias);

    float eps = 0.0f;
    memcpy(&eps, site.norm->op_params, sizeof(eps));

    // gamma/beta must agree with the activation dtype.
    const auto dt = x->dtype();
    auto match_dtype = [&](std::shared_ptr<tt::tt_metal::Tensor> & t) {
        if (t->dtype() != dt) {
            t = std::make_shared<tt::tt_metal::Tensor>(ttnn::typecast(*t, dt));
        }
    };
    match_dtype(weight);
    match_dtype(bias);

    auto out = site.is_rms ? ttnn::rms_norm(*x, eps, *weight, *bias)
                           : ttnn::layer_norm(*x, eps, *weight, *bias);
    auto * meta = (ggml_tensor_extra_metalium *)site.root->extra;
    meta->tensor = std::make_shared<tt::tt_metal::Tensor>(std::move(out));
    return true;
}

bool TtprmNormAffineLowering::apply(ggml_backend_metalium_context * ctx, const Site & site) const {
    GGML_UNUSED(ctx);
#ifdef GGML_METALIUM_HAVE_TTPRM
    GGML_METALIUM_OP_SANITY_CHECK(site.root);

    auto x      = realize_ggml_view(site.x);
    auto weight = realize_ggml_view(site.weight);
    auto bias   = realize_ggml_view(site.bias);

    float eps = 0.0f;
    memcpy(&eps, site.norm->op_params, sizeof(eps));

    const auto BF16 = tt::tt_metal::DataType::BFLOAT16;
    if (x->dtype() != BF16) {
        x = std::make_shared<tt::tt_metal::Tensor>(ttnn::typecast(*x, BF16));
    }
    if (weight->dtype() != BF16) {
        weight = std::make_shared<tt::tt_metal::Tensor>(ttnn::typecast(*weight, BF16));
    }
    if (bias->dtype() != BF16) {
        bias = std::make_shared<tt::tt_metal::Tensor>(ttnn::typecast(*bias, BF16));
    }

    auto live = [](const tt::tt_metal::Tensor & t) {
        ttprm::View v = ttprm::view_of(t);
        return v.slice({ { 0, v.rows(), 1 }, { 0, v.cols(), 1 } });
    };

    ttprm::View xv = live(*x);
    ttprm::View wv = ttprm::view_of(*weight).squeeze_dim(0).squeeze_dim(0);
    ttprm::View bv = ttprm::view_of(*bias).squeeze_dim(0).squeeze_dim(0);

    auto make_output = [&]() {
        return tt::tt_metal::create_device_tensor(
            ttnn::TensorSpec(
                ttnn::Shape({
                    (uint32_t) site.root->ne[3],
                    (uint32_t) site.root->ne[2],
                    (uint32_t) site.root->ne[1],
                    (uint32_t) site.root->ne[0],
                }),
                tt::tt_metal::TensorLayout(
                    BF16,
                    tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
                    tt::tt_metal::MemoryConfig{})),
            x->device());
    };

    auto out = make_output();
    auto res = ttprm::layer_norm(xv, wv, bv, eps, live(out));
    if (!res) {
        GGML_ABORT("ttprm_norm_affine rejected layer norm: %s", res.error().c_str());
    }

    auto * meta = (ggml_tensor_extra_metalium *)site.root->extra;
    meta->tensor = std::make_shared<tt::tt_metal::Tensor>(res.value());
    return true;
#else
    GGML_UNUSED(site);
    return false;
#endif
}

bool MacLowering::apply(ggml_backend_metalium_context * ctx, const Site & site) const {
    GGML_UNUSED(ctx);
    GGML_METALIUM_OP_SANITY_CHECK(site.root);

    auto a = realize_ggml_view(site.a);
    auto b = realize_ggml_view(site.b);
    auto c = realize_ggml_view(site.c);

    // mac == multiply(a,b) + c via the same broadcasting binaries the standalone MUL/ADD use, so no
    // dtype coercion is needed (matches the unfused path).
    auto out = ttnn::mac(*a, *b, *c, std::nullopt);
    auto * meta = (ggml_tensor_extra_metalium *)site.root->extra;
    meta->tensor = std::make_shared<tt::tt_metal::Tensor>(std::move(out));
    return true;
}

#ifdef GGML_METALIUM_HAVE_TTPRM
// Apply-time builder for a view-chain: realizes leaves and emits the ttprm ops, returning the head-grid
// View plus the tensors that must outlive it (bases + intermediate results). ok=false on any ttprm REJECT
// -> the caller falls back to a native recompute. This is the second half of the generic facility.
struct BuiltView {
    std::optional<ttprm::View>                         view;
    std::shared_ptr<tt::tt_metal::Tensor>              result; // materialized top-of-chain (for storing)
    std::vector<std::shared_ptr<tt::tt_metal::Tensor>> keep;   // bases + intermediate ttprm results
    bool                                               ok = true;
};

static BuiltView build_view_chain(const ViewChain & vc, int64_t hs, int64_t hc, int64_t nt) {
    const int64_t n_embd = hs * hc;
    const auto    BF16   = tt::tt_metal::DataType::BFLOAT16;
    auto to_bf16 = [&](std::shared_ptr<tt::tt_metal::Tensor> & t) {
        if (t->dtype() != BF16) {
            t = std::make_shared<tt::tt_metal::Tensor>(ttnn::typecast(*t, BF16));
        }
    };
    auto whole = [](const ttprm::View & v) { // peel the live rows off the padded tile grid before reshape
        return v.slice({ { 0, v.rows(), 1 }, { 0, v.cols(), 1 } });
    };
    BuiltView out;
    switch (vc.kind) {
        case ViewChain::Leaf: {
            auto t = realize_ggml_view(vc.leaf.target);
            to_bf16(t);
            ttprm::View v = ttprm::view_of(*t);
            switch (vc.leaf.how) {
                case GridLeaf::Flat:
                    out.view = v.slice({ { vc.leaf.row0, vc.leaf.row0 + nt, 1 }, { 0, n_embd, 1 } })
                                   .reshape({ hc * nt, hs });
                    break;
                case GridLeaf::Bcast:
                    out.view = whole(v).reshape({ hc, hs });
                    break;
                case GridLeaf::Head:
                default:
                    out.view = whole(v).reshape({ hc * nt, hs });
                    break;
            }
            out.result = t;
            out.keep.push_back(t);
            return out;
        }
        case ViewChain::Reshape:
            return build_view_chain(vc.in[0], hs, hc, nt); // flat<->head reshape is identity on the grid
        case ViewChain::Norm: {
            BuiltView c = build_view_chain(vc.in[0], hs, hc, nt);
            if (!c.ok) { out.ok = false; return out; }
            auto r = ttprm::layer_norm(*c.view, /*gamma=*/nullptr, /*beta=*/nullptr, vc.eps);
            if (!r) { out.ok = false; return out; }
            auto rt = std::make_shared<tt::tt_metal::Tensor>(r.value());
            out.keep = std::move(c.keep);
            out.keep.push_back(rt);
            out.result = rt;
            out.view = whole(ttprm::view_of(*rt));
            return out;
        }
        case ViewChain::Binary: {
            BuiltView a = build_view_chain(vc.in[0], hs, hc, nt);
            BuiltView b = build_view_chain(vc.in[1], hs, hc, nt);
            if (!a.ok || !b.ok) { out.ok = false; return out; }
            auto r = vc.op == GGML_OP_ADD ? ttprm::add(*a.view, *b.view)
                   : vc.op == GGML_OP_SUB ? ttprm::sub(*a.view, *b.view)
                   :                        ttprm::mul(*a.view, *b.view);
            if (!r) { out.ok = false; return out; }
            auto rt = std::make_shared<tt::tt_metal::Tensor>(r.value());
            out.keep = std::move(a.keep);
            for (auto & k : b.keep) {
                out.keep.push_back(k);
            }
            out.keep.push_back(rt);
            out.result = rt;
            out.view = whole(ttprm::view_of(*rt));
            return out;
        }
    }
    out.ok = false;
    return out;
}
#endif

bool L2HeadNormLowering::apply(ggml_backend_metalium_context * ctx, const Site & site) const {
    GGML_UNUSED(ctx);
#ifdef GGML_METALIUM_HAVE_TTPRM
    GGML_METALIUM_OP_SANITY_CHECK(site.root);

    auto a = realize_ggml_view(site.a); // k     [nt, n_embd]
    auto b = realize_ggml_view(site.b); // k_k   [1,  n_embd]  (broadcast over the tokens)

    // ttprm::mul_l2_norm requires bf16 operands; bail to native otherwise (the lerp path gates the same).
    if (a->dtype() != tt::tt_metal::DataType::BFLOAT16) {
        return false;
    }
    if (b->dtype() != a->dtype()) {
        b = std::make_shared<tt::tt_metal::Tensor>(ttnn::typecast(*b, a->dtype()));
    }

    const int64_t hs = site.head_size;
    const int64_t hc = site.head_count;
    const int64_t nt = site.root->ne[2];

    ttprm::View av = ttprm::view_of(*a);
    ttprm::View bv = ttprm::view_of(*b);
    ttprm::View a_head = av.slice({ { 0, av.rows(), 1 }, { 0, av.cols(), 1 } }).reshape({ hc * nt, hs });
    ttprm::View b_head = bv.slice({ { 0, bv.rows(), 1 }, { 0, bv.cols(), 1 } }).reshape({ hc, hs });

    auto res = ttprm::mul_l2_norm(a_head, b_head, site.eps);
    if (!res) {
        return false;
    }
    auto * meta = (ggml_tensor_extra_metalium *)site.root->extra;
    meta->tensor = std::make_shared<tt::tt_metal::Tensor>(reshape_tt_tensor_into_ggml(res.value(), site.root));
    return true;
#else
    GGML_UNUSED(site);
    return false;
#endif
}

bool HeadNormLowering::apply(ggml_backend_metalium_context * ctx, const Site & site) const {
    GGML_UNUSED(ctx);
#ifdef GGML_METALIUM_HAVE_TTPRM
    GGML_METALIUM_OP_SANITY_CHECK(site.root);

    const int64_t hs     = site.head_size;
    const int64_t hc     = site.head_count;
    const int64_t nt     = site.root->ne[2];
    const int64_t n_embd = hs * hc;

    ggml_tensor * xnode  = site.x;
    ggml_tensor * target = xnode;
    int64_t       row0   = 0;
    if (xnode->view_src != nullptr && xnode->view_src->ne[0] == n_embd && ggml_is_contiguous(xnode) &&
        (xnode->view_offs % (n_embd * ggml_type_size(xnode->type))) == 0) {
        target = xnode->view_src;
        row0   = xnode->view_offs / (n_embd * ggml_type_size(xnode->type));
    }

    auto x = realize_ggml_view(target);
    if (x->dtype() != tt::tt_metal::DataType::BFLOAT16) {
        return false;
    }

    ttprm::View xv = ttprm::view_of(*x);
    ttprm::View base = (target == xnode)
        ? xv.slice({ { 0, xv.rows(), 1 }, { 0, xv.cols(), 1 } })
        : xv.slice({ { row0, row0 + nt, 1 }, { 0, n_embd, 1 } });
    ttprm::View x_head = base.reshape({ hc * nt, hs });

    auto res = ttprm::layer_norm(x_head, /*gamma=*/nullptr, /*beta=*/nullptr, site.eps);
    if (!res) {
        return false;
    }
    auto * meta = (ggml_tensor_extra_metalium *)site.root->extra;
    meta->tensor = std::make_shared<tt::tt_metal::Tensor>(reshape_tt_tensor_into_ggml(res.value(), site.root));
    return true;
#else
    GGML_UNUSED(site);
    return false;
#endif
}

bool HeadAffineLowering::apply(ggml_backend_metalium_context * ctx, const Site & site) const {
    GGML_UNUSED(ctx);
#ifdef GGML_METALIUM_HAVE_TTPRM
    GGML_METALIUM_OP_SANITY_CHECK(site.root);

    const int64_t hs     = site.head_size;
    const int64_t hc     = site.head_count;
    const int64_t nt     = site.root->ne[1];
    const auto    BF16   = tt::tt_metal::DataType::BFLOAT16;
    auto * meta = (ggml_tensor_extra_metalium *)site.root->extra;

    std::unordered_set<const ggml_tensor *> killed(site.kill_requests.begin(), site.kill_requests.end());
    auto is_priv = [&](const ggml_tensor * n) {
        return killed.contains(n);
    };
    ViewChain chain = analyze_view_chain(site.root, hs, hc, nt, is_priv);
    BuiltView bv    = build_view_chain(chain, hs, hc, nt);
    if (bv.ok && bv.result) {
        meta->tensor =
            std::make_shared<tt::tt_metal::Tensor>(reshape_tt_tensor_into_ggml(*bv.result, site.root));
        return true;
    }
    // The absorbed nodes are inert, so fallback recomputes the native result here.
    auto to_bf16 = [&](std::shared_ptr<tt::tt_metal::Tensor> & t) {
        if (t->dtype() != BF16) {
            t = std::make_shared<tt::tt_metal::Tensor>(ttnn::typecast(*t, BF16));
        }
    };
    auto nf = realize_ggml_view(site.norm_flat); to_bf16(nf); // [n_embd, nt]
    auto wt = realize_ggml_view(site.weight);    to_bf16(wt);
    auto bt = realize_ggml_view(site.bias);      to_bf16(bt);
    auto scaled = ttnn::multiply(*nf, *wt);
    meta->tensor = std::make_shared<tt::tt_metal::Tensor>(ttnn::add(scaled, *bt));
    return true;
#else
    GGML_UNUSED(site);
    return false;
#endif
}

bool RkGateLowering::apply(ggml_backend_metalium_context * ctx, const Site & site) const {
    GGML_UNUSED(ctx);
#ifdef GGML_METALIUM_HAVE_TTPRM
    GGML_METALIUM_OP_SANITY_CHECK(site.root);

    const int64_t hs     = site.head_size;
    const int64_t hc     = site.head_count;
    const int64_t nt     = site.root->ne[1];
    const int64_t n_embd = hs * hc;
    const auto    BF16   = tt::tt_metal::DataType::BFLOAT16;

    auto to_bf16 = [&](std::shared_ptr<tt::tt_metal::Tensor> & t) {
        if (t->dtype() != BF16) {
            t = std::make_shared<tt::tt_metal::Tensor>(ttnn::typecast(*t, BF16));
        }
    };
    auto head_realize = [&](ggml_tensor * node) {
        ggml_tensor * target = node;
        if (node->op == GGML_OP_RESHAPE && node->src[0] != nullptr &&
            node->src[0]->ne[0] == n_embd && ggml_is_contiguous(node->src[0])) {
            target = node->src[0];
        }
        auto t = realize_ggml_view(target);
        to_bf16(t);
        return t;
    };

    auto rt = head_realize(site.r);
    auto kt = head_realize(site.k);
    auto vt = head_realize(site.v);
    auto wt = realize_ggml_view(site.r_k); to_bf16(wt);

    auto head_view = [&](const tt::tt_metal::Tensor & t) {
        ttprm::View tv = ttprm::view_of(t);
        return tv.slice({ { 0, tv.rows(), 1 }, { 0, tv.cols(), 1 } }).reshape({ hc * nt, hs });
    };
    auto make_output = [&]() {
        return tt::tt_metal::create_device_tensor(
            ttnn::TensorSpec(
                ttnn::Shape({
                    (uint32_t) site.root->ne[3],
                    (uint32_t) site.root->ne[2],
                    (uint32_t) site.root->ne[1],
                    (uint32_t) site.root->ne[0],
                }),
                tt::tt_metal::TensorLayout(
                    BF16,
                    tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
                    tt::tt_metal::MemoryConfig{})),
            rt->device());
    };
    ttprm::View rv = head_view(*rt);
    ttprm::View kv = head_view(*kt);
    ttprm::View vv = head_view(*vt);
    ttprm::View wv0 = ttprm::view_of(*wt);
    ttprm::View wv  = wv0.slice({ { 0, wv0.rows(), 1 }, { 0, wv0.cols(), 1 } }).reshape({ hc, hs });

    BuiltView                             cur_bv;   // keeps the head-grid chain's tensors alive for `cv`
    std::shared_ptr<tt::tt_metal::Tensor> ct;       // legacy flat realize (non-absorb path)
    std::optional<ttprm::View>            cv;
    if (site.absorb_cur) {
        GGML_ASSERT(site.cur_ttprm_view);
        std::unordered_set<const ggml_tensor *> killed(site.kill_requests.begin(), site.kill_requests.end());
        auto is_priv = [&](const ggml_tensor * n) {
            return killed.contains(n);
        };
        cur_bv = build_view_chain(analyze_view_chain(site.cur, hs, hc, nt, is_priv), hs, hc, nt);
        if (cur_bv.ok && cur_bv.view) {
            cv = *cur_bv.view;
        }
    } else {
        ct = realize_ggml_view(site.cur); to_bf16(ct);
        cv = head_view(*ct);
    }

    auto * meta = (ggml_tensor_extra_metalium *)site.root->extra;
    if (cv) {
        auto out = make_output();
        auto ov = head_view(out);
        auto res = ttprm::rk(rv, kv, wv, vv, *cv, ov);
        if (res) {
            meta->tensor = std::make_shared<tt::tt_metal::Tensor>(res.value());
            return true;
        }
    }
    // The absorbed nodes are inert, so fallback recomputes the native result here.
    auto rh = realize_ggml_view(site.r);
    auto kh = realize_ggml_view(site.k);
    auto vh = realize_ggml_view(site.v);
    ttnn::WormholeComputeKernelConfig cfg{
        .math_fidelity = tt::tt_metal::MathFidelity::HiFi3, .math_approx_mode = false,
        .fp32_dest_acc_en = true, .packer_l1_acc = true
    };
    auto kr     = ttnn::multiply(*kh, *rh);
    auto kr_rk  = ttnn::multiply(kr, *wt);                       // r_k broadcast over the tokens
    auto rk_red = ttnn::sum(kr_rk, 3, /*keepdim=*/true, std::nullopt, cfg); // sum_rows over head_size
    auto v_rk   = ttnn::multiply(*vh, rk_red);                  // rk broadcast over the head lanes
    auto v_rk_flat = reshape_tt_tensor_into_ggml(v_rk, site.root); // [head_size, head_count, nt] -> [n_embd, nt]
    // cur for the fallback add: legacy path realized it flat (ct); the absorb path already built the
    // head-grid cur (cur_bv.result) so flatten THAT (the cur chain is inert and can't be re-realized). The
    // absorb gate (hc % 32 == 0) guarantees cur_bv built whenever absorb_cur, so one of these always holds.
    std::shared_ptr<tt::tt_metal::Tensor> cur_flat;
    if (ct) {
        cur_flat = ct;
    } else if (cur_bv.result) {
        cur_flat = std::make_shared<tt::tt_metal::Tensor>(reshape_tt_tensor_into_ggml(*cur_bv.result, site.root));
    } else {
        cur_flat = realize_ggml_view(site.cur); // cur was left live (not absorbed)
    }
    to_bf16(cur_flat);
    meta->tensor = std::make_shared<tt::tt_metal::Tensor>(ttnn::add(*cur_flat, v_rk_flat));
    return true;
#else
    GGML_UNUSED(site);
    return false;
#endif
}

bool ElemwiseViewLowering::apply(ggml_backend_metalium_context * ctx, const Site & site) const {
    GGML_UNUSED(ctx);
#ifdef GGML_METALIUM_HAVE_TTPRM
    GGML_METALIUM_OP_SANITY_CHECK(site.root);

    const int64_t hs     = site.head_size;
    const int64_t hc     = site.head_count;
    const int64_t nt     = site.op_node->ne[2];
    const auto    BF16   = tt::tt_metal::DataType::BFLOAT16;

    auto to_bf16 = [&](std::shared_ptr<tt::tt_metal::Tensor> & t) {
        if (t->dtype() != BF16) {
            t = std::make_shared<tt::tt_metal::Tensor>(ttnn::typecast(*t, BF16));
        }
    };

    std::unordered_set<const ggml_tensor *> killed(site.kill_requests.begin(), site.kill_requests.end());
    auto is_private = [&](const ggml_tensor * n) { return killed.find(n) != killed.end(); };
    BuiltView a = build_view_chain(analyze_view_chain(site.a, hs, hc, nt, is_private, /*is_root=*/false),
                                   hs, hc, nt);
    BuiltView b = build_view_chain(analyze_view_chain(site.b, hs, hc, nt, is_private, /*is_root=*/false),
                                   hs, hc, nt);

    auto * meta = (ggml_tensor_extra_metalium *)site.root->extra;
    if (a.ok && b.ok && a.view && b.view) {
        auto store = [&](auto && res) {
            if (!res) {
                return false;
            }
            meta->tensor = std::make_shared<tt::tt_metal::Tensor>(
                    reshape_tt_tensor_into_ggml(res.value(), site.root));
            return true;
        };
        switch (site.op) {
            case GGML_OP_ADD:
                if (store(ttprm::add(*a.view, *b.view))) {
                    return true;
                }
                break;
            case GGML_OP_SUB:
                if (store(ttprm::sub(*a.view, *b.view))) {
                    return true;
                }
                break;
            default:
                if (store(ttprm::mul(*a.view, *b.view))) {
                    return true;
                }
                break;
        }
    }
    // The absorbed nodes are inert, so fallback recomputes the native result here.
    auto ah = realize_ggml_view(site.a); to_bf16(ah);
    auto bh = realize_ggml_view(site.b); to_bf16(bh);
    auto out = [&]() {
        switch (site.op) {
            case GGML_OP_ADD: return ttnn::add(*ah, *bh);
            case GGML_OP_SUB: return ttnn::subtract(*ah, *bh);
            default:          return ttnn::multiply(*ah, *bh);
        }
    }();
    meta->tensor = std::make_shared<tt::tt_metal::Tensor>(reshape_tt_tensor_into_ggml(out, site.root));
    return true;
#else
    GGML_UNUSED(site);
    return false;
#endif
}

bool ActLowering::apply(ggml_backend_metalium_context * ctx, const Site & site) const {
    GGML_UNUSED(ctx);
    GGML_METALIUM_OP_SANITY_CHECK(site.root);

    // Same operand mapping as the backend's MUL_MAT lowering: (b=input, a=weight, transpose_b=true).
    auto a = realize_ggml_view(site.weight);
    auto b = realize_ggml_view(site.input);
    ttnn::Activation act{ site.act };

    tt::tt_metal::Tensor out;
    if (site.is_linear) {
        auto bias = realize_ggml_view(site.bias);
        if (bias->dtype() != b->dtype()) {
            bias = std::make_shared<tt::tt_metal::Tensor>(ttnn::typecast(*bias, b->dtype()));
        }
        out = ttnn::operations::matmul::linear(
            *b, *a, *bias, /*transpose_a*/ false, /*transpose_b*/ true,
            std::nullopt, std::nullopt, std::nullopt, /*activation*/ act,
            make_compute_kernel_config(b->device()));
    } else {
        out = ttnn::operations::matmul::matmul(
            *b, *a, /*transpose_a*/ false, /*transpose_b*/ true,
            std::nullopt, std::nullopt, std::nullopt, /*activation*/ act,
            make_compute_kernel_config(b->device()));
    }

    auto * meta = (ggml_tensor_extra_metalium *)site.root->extra;
    meta->tensor = std::make_shared<tt::tt_metal::Tensor>(std::move(out));
    return true;
}

MetaliumGraphCompiler::MetaliumGraphCompiler() {
    lowerings.push_back(std::make_unique<ActLowering>());
    lowerings.push_back(std::make_unique<RkGateLowering>());
    lowerings.push_back(std::make_unique<HeadAffineLowering>());
    lowerings.push_back(std::make_unique<L2HeadNormLowering>());
    lowerings.push_back(std::make_unique<HeadNormLowering>());
    lowerings.push_back(std::make_unique<LerpLowering>());
    lowerings.push_back(std::make_unique<LinearLowering>());
    lowerings.push_back(std::make_unique<TtprmNormAffineLowering>());
    lowerings.push_back(std::make_unique<NormAffineLowering>());
    lowerings.push_back(std::make_unique<MacLowering>());
    lowerings.push_back(std::make_unique<ElemwiseViewLowering>());
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
    auto planned = planned_roots_.find(node);
    if (planned != planned_roots_.end()) {
        return planned->second->lowerFusion(ctx, node);
    }

    if (const MetaliumLowering * lowering = findLowering(node)) {
        lowering->lowerNode(ctx, node);
        return true;
    }
    return false;
}
