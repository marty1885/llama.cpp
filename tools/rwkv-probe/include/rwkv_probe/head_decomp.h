// rwkv_probe/head_decomp.h — per-head SVD decomposition of state deltas.
//
// Provides a first-class HeadDecomp type that encapsulates the rank-1
// direction (sigma, u, v, r1_ratio) extracted from a per-head state delta.
// This pattern was independently re-derived in 8+ experiments.
#pragma once

#include "state.h"

#include <vector>

namespace rwkv_probe {

struct HeadDecomp {
    int    layer = 0;
    int    head  = 0;
    double sigma    = 0.0;  // leading singular value
    double r1_ratio = 0.0;  // sigma^2 / sum(sigma_i^2) — rank-1 dominance
    std::vector<double> u;  // left singular vector  (value direction), size = hs
    std::vector<double> v;  // right singular vector (address direction), size = hs
};

// Decompose a single head's delta (sb - sa) via SVD.
// Returns the rank-1 direction. If the delta is near-zero, sigma == 0.
HeadDecomp decompose_head(const StateBuf & sa, const StateBuf & sb,
                          int layer, int head);

// Decompose all heads across the given layers.
// Returns n_layers * n_head entries, ordered by (layer, head).
std::vector<HeadDecomp> decompose(const StateBuf & sa, const StateBuf & sb,
                                  const std::vector<int> & layers);

}  // namespace rwkv_probe
