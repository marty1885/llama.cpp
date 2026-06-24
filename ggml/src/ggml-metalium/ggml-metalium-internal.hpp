#pragma once

// Backend internals shared between ggml-metalium.cpp and the graph compiler.
// These are the few types and helpers the lowering dispatch needs to bridge ggml
// tensors and TTNN tensors; everything else stays private to ggml-metalium.cpp.

#include <memory>

#include <ttnn/tensor/tensor.hpp>
#include <ttnn/device.hpp>
#include <ttnn/operations/core/compute_kernel/compute_kernel_config.hpp>

#include "ggml.h"

class MetaliumGraphCompiler;

// Per ggml-tensor device state, hung off ggml_tensor::extra.
struct ggml_tensor_extra_metalium {
    std::shared_ptr<tt::tt_metal::Tensor> tensor;

    // Row-folded physical storage for a tensor whose GGML-declared shape would tile-pad badly
    // (1-D / short-penultimate caches and the token embedding). When non-null, this is the
    // authoritative device copy stored as [n_rows, dim/32, 32] and `tensor` is dropped
    std::shared_ptr<tt::tt_metal::Tensor> row_folded;

    bool is_row_folded() const { return row_folded != nullptr; }
};

struct ggml_backend_metalium_context {
    ttnn::IDevice* device = nullptr;
    int device_id = 0;
    std::string name;
    MetaliumGraphCompiler * compiler = nullptr; // non-owning, owned by the device context
};

// Materialises the TTNN tensor backing a (possibly lazily-viewed) ggml tensor.
std::shared_ptr<tt::tt_metal::Tensor> realize_ggml_view(const ggml_tensor* tensor);

// Stable identity for a cgraph: its uid when set, else a topology+shape signature. The graph
// compiler caches its per-graph fusion plan on this key (the same key the trace layer uses).
uint64_t metalium_graph_key(const ggml_cgraph* cgraph);

// The matmul math-fidelity config the backend uses for all GEMMs (HiFi4 on Wormhole). Fusions that
// emit their own matmul/linear must reuse this so accuracy matches the unfused path.
ttnn::DeviceComputeKernelConfig make_compute_kernel_config(ttnn::IDevice* device);

inline void ggml_metalium_op_src_sanity_check(const ggml_tensor * node, int idx) {
    GGML_ASSERT(node->src[idx] != NULL);
    GGML_ASSERT(node->src[idx]->extra != NULL);
    auto* meta = (ggml_tensor_extra_metalium*)(node->src[idx]->extra);
    if(meta->tensor != NULL) {
        GGML_ASSERT(meta->tensor->storage_type() == tt::tt_metal::StorageType::DEVICE);
        GGML_ASSERT(meta->tensor->layout() == tt::tt_metal::Layout::TILE);
    }
}

#define GGML_METALIUM_OP_SANITY_CHECK(_node) \
    GGML_ASSERT((_node)->extra != NULL);
#define GGML_METALIUM_OP_SRC_SANITY_CHECK(_node, _idx) ggml_metalium_op_src_sanity_check(_node, _idx);
#define GGML_METALIUM_OP_SRC0_SANITY_CHECK(_node) GGML_METALIUM_OP_SRC_SANITY_CHECK(_node, 0)
#define GGML_METALIUM_OP_SRC1_SANITY_CHECK(_node) GGML_METALIUM_OP_SRC_SANITY_CHECK(_node, 1)
