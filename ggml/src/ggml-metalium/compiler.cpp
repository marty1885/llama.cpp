#include "compiler.hpp"
#include "ggml-metalium-internal.hpp"
#include "embedding.hpp"

#include <optional>

#include <ttnn/operations/copy/typecast/typecast.hpp>
#include <ttnn/operations/eltwise/ternary/ternary.hpp>
#include <ttnn/operations/eltwise/ternary/ternary_composite_op.hpp>
#include <ttnn/operations/matmul/matmul.hpp>
#include <ttnn/operations/normalization/layernorm/layernorm.hpp>
#include <ttnn/operations/normalization/rmsnorm/rmsnorm.hpp>
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

// RWKV's token-shift:
//     sx  = SUB(x_prev, cur)
//     sx  = REPEAT(sx, x6)          // time-mix only; channel-mix has no repeat
//     xxx = ADD( MUL(sx, w), cur )
namespace {

struct LerpMatch {
    ggml_tensor * root   = nullptr;
    ggml_tensor * cur    = nullptr;
    ggml_tensor * x_prev = nullptr;
    ggml_tensor * weight = nullptr;
    ggml_tensor * sub    = nullptr;
    ggml_tensor * mul    = nullptr;
    ggml_tensor * repeat = nullptr; // optional
};

// Pure structural walk back from an ADD. No use-count / privacy checks here -- callers add those
std::optional<LerpMatch> match_lerp_idiom(ggml_tensor * add) {
    if (add == nullptr || add->op != GGML_OP_ADD) {
        return std::nullopt;
    }
    // ADD(mul, cur) - try both operand orders (ggml emits mul first, but also try the other way)
    for (int i = 0; i < 2; i++) {
        ggml_tensor * mul = add->src[i];
        ggml_tensor * cur = add->src[1 - i];
        if (mul == nullptr || cur == nullptr || mul->op != GGML_OP_MUL) {
            continue;
        }
        // MUL(sx_chain, weight) -- again either order.
        for (int j = 0; j < 2; j++) {
            ggml_tensor * sx_chain = mul->src[j];
            ggml_tensor * weight   = mul->src[1 - j];
            if (sx_chain == nullptr || weight == nullptr) {
                continue;
            }
            ggml_tensor * repeat = nullptr;
            ggml_tensor * sub    = sx_chain;
            if (sub->op == GGML_OP_REPEAT) {
                repeat = sub;
                sub    = sub->src[0];
            }
            if (sub == nullptr || sub->op != GGML_OP_SUB) {
                continue;
            }
            // the cur subtracted in SUB must be the cur added in ADD
            if (sub->src[1] != cur) {
                continue;
            }
            LerpMatch m;
            m.root   = add;
            m.cur    = cur;
            m.x_prev = sub->src[0];
            m.weight = weight;
            m.sub    = sub;
            m.mul    = mul;
            m.repeat = repeat;
            return m;
        }
    }
    return std::nullopt;
}

struct LinearMatch {
    ggml_tensor * root   = nullptr;
    ggml_tensor * matmul = nullptr;
    ggml_tensor * weight = nullptr; // matmul src0
    ggml_tensor * input  = nullptr; // matmul src1
    ggml_tensor * bias   = nullptr;
};

std::optional<LinearMatch> match_bias_matmul(ggml_tensor * add) {
    if (add == nullptr || add->op != GGML_OP_ADD) {
        return std::nullopt;
    }
    for (int i = 0; i < 2; i++) {
        ggml_tensor * mm   = add->src[i];
        ggml_tensor * bias = add->src[1 - i];
        if (mm == nullptr || bias == nullptr || mm->op != GGML_OP_MUL_MAT) {
            continue;
        }
        if (mm->src[0] == nullptr || mm->src[1] == nullptr) {
            continue;
        }
        // ttnn::linear takes a per-output-channel bias vector broadcast over rows, not a full
        // matrix. Require bias = [N, 1, 1, 1] with N == the matmul's output width.
        if (bias->ne[0] != mm->ne[0] || bias->ne[1] != 1 || bias->ne[2] != 1 || bias->ne[3] != 1) {
            continue;
        }
        LinearMatch m;
        m.root   = add;
        m.matmul = mm;
        m.weight = mm->src[0];
        m.input  = mm->src[1];
        m.bias   = bias;
        return m;
    }
    return std::nullopt;
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
        default:                    return nullptr;
    }
}

struct ActMatch {
    ggml_tensor * root      = nullptr; // the unary
    ggml_tensor * weight    = nullptr;
    ggml_tensor * input     = nullptr;
    ggml_tensor * bias      = nullptr;
    ggml_tensor * producer  = nullptr; // MUL_MAT (bare) or the ADD (linear) -> inert
    ggml_tensor * matmul    = nullptr; // linear case inner matmul -> inert
    const char *  act       = nullptr;
    bool          is_linear = false;
};

std::optional<ActMatch> match_activation(ggml_tensor * u) {
    const char * act = ggml_unary_act_string(u);
    if (act == nullptr) {
        return std::nullopt;
    }
    ggml_tensor * p = u->src[0];
    if (p == nullptr) {
        return std::nullopt;
    }
    if (p->op == GGML_OP_MUL_MAT && p->src[0] != nullptr && p->src[1] != nullptr) {
        ActMatch m;
        m.root = u; m.act = act; m.is_linear = false;
        m.weight = p->src[0]; m.input = p->src[1];
        m.producer = p;
        return m;
    }
    if (auto lm = match_bias_matmul(p)) {
        ActMatch m;
        m.root = u; m.act = act; m.is_linear = true;
        m.weight = lm->weight; m.input = lm->input; m.bias = lm->bias;
        m.producer = p;            // the ADD
        m.matmul   = lm->matmul;   // inner matmul
        return m;
    }
    return std::nullopt;
}

struct NormAffineMatch {
    ggml_tensor * root   = nullptr;
    ggml_tensor * mul    = nullptr;
    ggml_tensor * norm   = nullptr;
    ggml_tensor * x      = nullptr;
    ggml_tensor * weight = nullptr;
    ggml_tensor * bias   = nullptr;
    bool          is_rms = false;
};

std::optional<NormAffineMatch> match_norm_affine(ggml_tensor * add) {
    if (add == nullptr || add->op != GGML_OP_ADD) {
        return std::nullopt;
    }
    for (int i = 0; i < 2; i++) {
        ggml_tensor * mul  = add->src[i];
        ggml_tensor * bias = add->src[1 - i];
        if (mul == nullptr || bias == nullptr || mul->op != GGML_OP_MUL) {
            continue;
        }
        for (int j = 0; j < 2; j++) {
            ggml_tensor * nrm    = mul->src[j];
            ggml_tensor * weight = mul->src[1 - j];
            if (nrm == nullptr || weight == nullptr) {
                continue;
            }
            if (nrm->op != GGML_OP_NORM && nrm->op != GGML_OP_RMS_NORM) {
                continue;
            }
            if (nrm->src[0] == nullptr) {
                continue;
            }
            // Width-1 rows hit the backend's sign() special-case in the plain norm path; leave those.
            if (nrm->src[0]->ne[0] == 1) {
                continue;
            }
            // ttnn rms_norm/layer_norm take a per-channel gamma and beta vector broadcast over rows.
            // gamma and beta must be [N,1,1,1], and beta must be a loaded parameter rather than an
            // activation. Otherwise a residual add such as Gemma2's l_out = norm(x)*w + sa_out matches
            // here and the residual gets misbroadcast as beta. Such cases fall back to the native path.
            const int64_t n = nrm->ne[0];
            if (weight->ne[0] != n || weight->ne[1] != 1 || weight->ne[2] != 1 || weight->ne[3] != 1) {
                continue;
            }
            if (bias->ne[0] != n || bias->ne[1] != 1 || bias->ne[2] != 1 || bias->ne[3] != 1) {
                continue;
            }
            if (bias->op != GGML_OP_NONE) {
                continue;
            }
            NormAffineMatch m;
            m.root   = add;
            m.mul    = mul;
            m.norm   = nrm;
            m.x      = nrm->src[0];
            m.weight = weight;
            m.bias   = bias;
            m.is_rms = nrm->op == GGML_OP_RMS_NORM;
            return m;
        }
    }
    return std::nullopt;
}

struct MacMatch {
    ggml_tensor * root = nullptr;
    ggml_tensor * mul  = nullptr;
    ggml_tensor * a    = nullptr;
    ggml_tensor * b    = nullptr;
    ggml_tensor * c    = nullptr;
};

std::optional<MacMatch> match_mac(ggml_tensor * add) {
    if (add == nullptr || add->op != GGML_OP_ADD) {
        return std::nullopt;
    }
    for (int i = 0; i < 2; i++) {
        ggml_tensor * mul = add->src[i];
        ggml_tensor * c   = add->src[1 - i];
        if (mul == nullptr || c == nullptr || mul->op != GGML_OP_MUL) {
            continue;
        }
        if (mul->src[0] == nullptr || mul->src[1] == nullptr) {
            continue;
        }
        MacMatch m;
        m.root = add;
        m.mul  = mul;
        m.a    = mul->src[0];
        m.b    = mul->src[1];
        m.c    = c;
        return m;
    }
    return std::nullopt;
}

} // namespace

MetaliumGraphCompiler::MetaliumGraphCompiler() {
    lowerings.push_back(std::make_unique<EmbeddingGetRowsLowering>());
}

void MetaliumGraphCompiler::analyzeGraph(const ggml_cgraph * cgraph) {
    inert_.clear();
    act_roots_.clear();
    lerp_roots_.clear();
    linear_roots_.clear();
    norm_roots_.clear();
    mac_roots_.clear();
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

    // Use counts over the whole graph: an intermediate is private to an idiom iff its only consumer is
    // the next node in that idiom (and it is not a graph output).
    std::unordered_map<const ggml_tensor *, int> uses;
    std::unordered_map<const ggml_tensor *, int> index_of;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        index_of[cgraph->nodes[i]] = i;
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            if (cgraph->nodes[i]->src[s] != nullptr) {
                uses[cgraph->nodes[i]->src[s]]++;
            }
        }
    }
    auto is_private = [&](const ggml_tensor * n) {
        return n != nullptr && uses[n] == 1 && (n->flags & GGML_TENSOR_FLAG_OUTPUT) == 0;
    };
    // Record a fusion: root index + the intermediate nodes it kills (nulls ignored).
    auto commit = [&](std::vector<int> & roots, int root_idx, std::initializer_list<const ggml_tensor *> dead) {
        roots.push_back(root_idx);
        for (const ggml_tensor * n : dead) {
            if (n != nullptr) {
                plan.inert_idx.push_back(index_of[n]);
            }
        }
    };

    // Pre-pass: fold activations into their producing GEMM (highest priority). It reaches two levels
    // down (unary <- add <- matmul), so it must claim that chain before the matmul+bias pass can
    // independently root the same ADD.
    std::unordered_set<const ggml_tensor *> claimed;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        auto m = match_activation(cgraph->nodes[i]);
        if (!m || !is_private(m->producer) || (m->is_linear && !is_private(m->matmul))) {
            continue;
        }
        commit(plan.act_root_idx, i, { m->producer, m->is_linear ? m->matmul : nullptr });
        claimed.insert(m->root);
        claimed.insert(m->producer);
        claimed.insert(m->matmul);
    }

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        if (claimed.count(node) != 0) {
            continue; // already consumed by the activation pre-pass
        }
        // Each idiom only fuses when the intermediates it would delete are private; on a structural
        // match it always `continue`s so a more general matcher below can't re-claim the same node.
        if (auto m = match_lerp_idiom(node)) {
            if (is_private(m->sub) && is_private(m->mul) && (m->repeat == nullptr || is_private(m->repeat))) {
                commit(plan.lerp_root_idx, i, { m->sub, m->mul, m->repeat });
            }
        } else if (auto m = match_bias_matmul(node)) {
            if (is_private(m->matmul)) {
                commit(plan.linear_root_idx, i, { m->matmul });
            }
        } else if (auto m = match_norm_affine(node)) {
            if (is_private(m->norm) && is_private(m->mul)) {
                commit(plan.norm_root_idx, i, { m->norm, m->mul });
            }
        } else if (auto m = match_mac(node)) { // generic FMA -- last, claims only what the rest didn't
            if (is_private(m->mul)) {
                commit(plan.mac_root_idx, i, { m->mul });
            }
        }
    }
    return plan;
}

void MetaliumGraphCompiler::bindPlan(const ggml_cgraph * cgraph, const FusionPlan & plan) {
    auto node = [&](int idx) { return cgraph->nodes[idx]; };
    for (int idx : plan.inert_idx) {
        inert_.insert(node(idx));
    }
    for (int idx : plan.act_root_idx) {
        if (auto m = match_activation(node(idx))) {
            act_roots_[m->root] = ActSite{ m->root, m->weight, m->input, m->bias, std::string(m->act), m->is_linear };
        }
    }
    for (int idx : plan.lerp_root_idx) {
        if (auto m = match_lerp_idiom(node(idx))) {
            lerp_roots_[m->root] = LerpSite{ m->root, m->cur, m->x_prev, m->weight };
        }
    }
    for (int idx : plan.linear_root_idx) {
        if (auto m = match_bias_matmul(node(idx))) {
            linear_roots_[m->root] = LinearSite{ m->root, m->weight, m->input, m->bias };
        }
    }
    for (int idx : plan.norm_root_idx) {
        if (auto m = match_norm_affine(node(idx))) {
            norm_roots_[m->root] = NormAffineSite{ m->root, m->norm, m->x, m->weight, m->bias, m->is_rms };
        }
    }
    for (int idx : plan.mac_root_idx) {
        if (auto m = match_mac(node(idx))) {
            mac_roots_[m->root] = MacSite{ m->root, m->a, m->b, m->c };
        }
    }
}

bool MetaliumGraphCompiler::isInert(const ggml_tensor * node) const {
    return inert_.find(node) != inert_.end();
}

void MetaliumGraphCompiler::applyLerp(ggml_backend_metalium_context * ctx, const LerpSite & site) const {
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

    auto out = ttnn::lerp(*cur, *x_prev, *weight);
    auto * meta = (ggml_tensor_extra_metalium *)site.root->extra;
    meta->tensor = std::make_shared<tt::tt_metal::Tensor>(std::move(out));
}

void MetaliumGraphCompiler::applyLinear(ggml_backend_metalium_context * ctx, const LinearSite & site) const {
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
}

void MetaliumGraphCompiler::applyNormAffine(ggml_backend_metalium_context * ctx, const NormAffineSite & site) const {
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
}

void MetaliumGraphCompiler::applyMac(ggml_backend_metalium_context * ctx, const MacSite & site) const {
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
}

void MetaliumGraphCompiler::applyAct(ggml_backend_metalium_context * ctx, const ActSite & site) const {
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
    // Fusion roots planned by analyzeGraph take priority over the per-node lowerings.
    auto a = act_roots_.find(node);
    if (a != act_roots_.end()) {
        applyAct(ctx, a->second);
        return true;
    }
    auto site = lerp_roots_.find(node);
    if (site != lerp_roots_.end()) {
        applyLerp(ctx, site->second);
        return true;
    }
    auto lin = linear_roots_.find(node);
    if (lin != linear_roots_.end()) {
        applyLinear(ctx, lin->second);
        return true;
    }
    auto nrm = norm_roots_.find(node);
    if (nrm != norm_roots_.end()) {
        applyNormAffine(ctx, nrm->second);
        return true;
    }
    auto mc = mac_roots_.find(node);
    if (mc != mac_roots_.end()) {
        applyMac(ctx, mc->second);
        return true;
    }
    if (const MetaliumLowering * lowering = findLowering(node)) {
        lowering->lowerNode(ctx, node);
        return true;
    }
    return false;
}
