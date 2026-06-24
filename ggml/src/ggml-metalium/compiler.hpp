#pragma once

// The Metalium graph compiler recognises graph nodes so we can dispatch to custom implementations
// or mess with the graph that GGML won't allow us to per the programming contract. This also **should**
// as in I haven't wrote it, use to decide which tensors should live on SRAM and how sharding is applied
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct ggml_tensor;
struct ggml_cgraph;
struct ggml_backend_metalium_context;

class MetaliumLowering;

class MetaliumGraphCompiler {
public:
    MetaliumGraphCompiler();
    ~MetaliumGraphCompiler();

    // Recognise multi-node fusions on this graph and record a plan (which node is a fusion root,
    // which nodes become inert) ONCE per graph, cached on metalium_graph_key. Cheap on cache hit:
    // it only re-binds the plan's node indices to this pass's node pointers. Call before the
    // graph_compute node loop (the loop only runs on eager / trace-capture passes, not replay).
    void analyzeGraph(const ggml_cgraph * cgraph);

    // True for nodes the active plan folded into a fusion root: the compute loop must SKIP them
    // (their work is subsumed by the root and they must not be executed or baked into a trace).
    bool isInert(const ggml_tensor * node) const;

    // @returns true if we have a custom lowering for this node, false otherwise. Note that this is not
    bool hasLowering(const ggml_tensor * node) const;

    // @param ctx The context to use for lowering.
    // @param node The node to try to lower.
    // @return true if we successfully lowered the node, false otherwise
    // Note that this will not attempt to lower the node if we don't have a lowering for it, so you
    // should call hasLowering first if you want to be sure.
    bool tryLowerNode(ggml_backend_metalium_context * ctx, ggml_tensor * node) const;

private:
    const MetaliumLowering * findLowering(const ggml_tensor * node) const;

    std::vector<std::unique_ptr<MetaliumLowering>> lowerings;

    // A lerp fusion site, re-derived (cheaply) from the live graph each analyzeGraph call.
    struct LerpSite {
        ggml_tensor * root   = nullptr; // the ADD producing xxx -- emits ttnn::lerp
        ggml_tensor * cur    = nullptr; // lerp start  (the ADD's non-mul operand)
        ggml_tensor * x_prev = nullptr; // lerp end    (the SUB's minuend)
        ggml_tensor * weight = nullptr; // lerp weight (the MUL's non-sx operand)
    };
    // A matmul+bias fusion site: ADD(MUL_MAT(weight, input), bias) -> ttnn::linear.
    struct LinearSite {
        ggml_tensor * root   = nullptr; // the ADD producing the biased output -- emits ttnn::linear
        ggml_tensor * weight = nullptr; // MUL_MAT src0
        ggml_tensor * input  = nullptr; // MUL_MAT src1
        ggml_tensor * bias   = nullptr; // the ADD's non-matmul operand
    };
    // A norm+affine site: ADD(MUL(NORM(x), weight), bias) -> ttnn::layer_norm/rms_norm(x, w, b).
    struct NormAffineSite {
        ggml_tensor * root   = nullptr; // the ADD -- emits the affine-fused norm
        ggml_tensor * norm   = nullptr; // the NORM/RMS_NORM node (eps read from its op_params)
        ggml_tensor * x      = nullptr; // the normalized input (NORM src0)
        ggml_tensor * weight = nullptr; // gamma
        ggml_tensor * bias   = nullptr; // beta
        bool          is_rms = false;
    };
    // An activation folded into its producing GEMM: UNARY(act) <- MUL_MAT  -> matmul(activation),
    // or UNARY(act) <- ADD(MUL_MAT, bias) -> linear(activation). Unlike the composite fusions this
    // is a TRUE single-kernel fold -- the activation runs in the matmul's pack stage, so the unary
    // kernel disappears. Highest priority (claims its producer chain before linear can root on it).
    struct ActSite {
        ggml_tensor * root      = nullptr; // the UNARY -- emits the activated GEMM
        ggml_tensor * weight    = nullptr; // MUL_MAT src0
        ggml_tensor * input     = nullptr; // MUL_MAT src1
        ggml_tensor * bias      = nullptr; // linear only (null => bare matmul)
        std::string   act;                 // "sigmoid" / "relu" / "gelu" / "silu"
        bool          is_linear = false;
    };
    // A generic FMA site: ADD(MUL(a, b), c) -> ttnn::mac(a, b, c) == a*b + c. Lowest priority --
    // matched only after lerp/linear/norm so it can't swallow those more specific idioms.
    struct MacSite {
        ggml_tensor * root = nullptr; // the ADD -- emits ttnn::mac
        ggml_tensor * a    = nullptr; // MUL operand 0
        ggml_tensor * b    = nullptr; // MUL operand 1
        ggml_tensor * c    = nullptr; // the ADD's non-mul operand (addend)
    };
    // Structural decision cached per graph key: positions in cgraph->nodes (stable per key).
    struct FusionPlan {
        std::vector<int> act_root_idx;    // UNARY roots folded into matmul/linear activation (first)
        std::vector<int> lerp_root_idx;   // ADD roots folded to ttnn::lerp
        std::vector<int> linear_root_idx; // ADD roots folded to ttnn::linear (matmul+bias)
        std::vector<int> norm_root_idx;   // ADD roots folded to affine-fused norm
        std::vector<int> mac_root_idx;    // ADD roots folded to ttnn::mac (generic FMA, last)
        std::vector<int> inert_idx;       // intermediate nodes made dead by a fusion
    };

    // Scan the graph once and record which nodes fuse (roots) and which die (inert), by index.
    FusionPlan buildPlan(const ggml_cgraph * cgraph) const;
    // Rebind a cached plan's node indices to this pass's live node pointers.
    void bindPlan(const ggml_cgraph * cgraph, const FusionPlan & plan);

    void applyLerp(ggml_backend_metalium_context * ctx, const LerpSite & site) const;
    void applyLinear(ggml_backend_metalium_context * ctx, const LinearSite & site) const;
    void applyNormAffine(ggml_backend_metalium_context * ctx, const NormAffineSite & site) const;
    void applyMac(ggml_backend_metalium_context * ctx, const MacSite & site) const;
    void applyAct(ggml_backend_metalium_context * ctx, const ActSite & site) const;

    std::unordered_map<uint64_t, FusionPlan> plan_cache_;        // built once per graph key
    // Transient, rebuilt each analyzeGraph for the current graph's live node pointers.
    std::unordered_set<const ggml_tensor *>                 inert_;
    std::unordered_map<const ggml_tensor *, ActSite>        act_roots_;
    std::unordered_map<const ggml_tensor *, LerpSite>       lerp_roots_;
    std::unordered_map<const ggml_tensor *, LinearSite>     linear_roots_;
    std::unordered_map<const ggml_tensor *, NormAffineSite> norm_roots_;
    std::unordered_map<const ggml_tensor *, MacSite>        mac_roots_;
};
