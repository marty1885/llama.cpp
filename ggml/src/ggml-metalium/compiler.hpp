#pragma once

// The Metalium graph compiler recognises graph nodes so we can dispatch to custom implementations
// or mess with the graph that GGML won't allow us to per the programming contract. This also **should**
// as in I haven't wrote it, use to decide which tensors should live on SRAM and how sharding is applied
#include <memory>
#include <vector>

struct ggml_tensor;
struct ggml_backend_metalium_context;

class MetaliumLowering;

class MetaliumGraphCompiler {
public:
    MetaliumGraphCompiler();
    ~MetaliumGraphCompiler();

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
};
