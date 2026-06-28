#pragma once

// The Metalium graph compiler recognises graph nodes so we can dispatch to custom implementations
// or mess with the graph that GGML won't allow us to per the programming contract. This also **should**
// as in I haven't wrote it, use to decide which tensors should live on SRAM and how sharding is applied
#include <cstdint>
#include <memory>
#include <optional>
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

    // Structural decision cached per graph key: positions in cgraph->nodes (stable per key).
    struct FusionPlan {
        struct PlannedRoot {
            int root_idx = -1;
            const MetaliumLowering * lowering = nullptr;
        };
        std::vector<PlannedRoot> roots;   // fusion roots, in lowering registration order
        std::vector<int> inert_idx;       // intermediate nodes made dead by a fusion
    };

    // Scan the graph once and record which nodes fuse (roots) and which die (inert), by index.
    FusionPlan buildPlan(const ggml_cgraph * cgraph) const;
    // Rebind a cached plan's node indices to this pass's live node pointers.
    void bindPlan(const ggml_cgraph * cgraph, const FusionPlan & plan);

    std::unordered_map<uint64_t, FusionPlan> plan_cache_;        // built once per graph key
    // Transient, rebuilt each analyzeGraph for the current graph's live node pointers.
    std::unordered_set<const ggml_tensor *>                 inert_;
    std::unordered_map<const ggml_tensor *, int>            uses_; // graph use-counts (view-chain privacy)
    std::unordered_map<const ggml_tensor *, const MetaliumLowering *> planned_roots_;
};
