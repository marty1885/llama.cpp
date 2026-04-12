// rwkv_probe/edit.h — state edit operations.
//
// Concentrates the six edit strategies that were scattered across experiments:
//   1. Full delta:        dst += alpha * (sb - sa)
//   2. Rank-k delta:      SVD of delta, reconstruct top-k components
//   3. Sigma-filtered:    skip heads where sigma_1 < threshold
//   4. Weighted delta:    per-head calibration weights
//   5. Donor-free:        projection-based edit using calibrated directions
//   6. Slerp:             spherical interpolation between head states
//
// All operations work per-head on s_head(layer, head) matrices and derive
// geometry (n_head, head_size) from the StateBuf itself.
#pragma once

#include "state.h"
#include "head_decomp.h"

#include <vector>

namespace rwkv_probe {

// ── full delta ───────────────────────────────────────────────────────────────
// dst_head += alpha * (sb_head - sa_head) for each head at each layer.
void apply_full_delta(StateBuf & dst,
                      const StateBuf & sa, const StateBuf & sb,
                      const std::vector<int> & layers,
                      double alpha = 1.0);

// ── rank-k delta with optional sigma filter ──────────────────────────────────
// SVD of per-head delta, reconstruct with top-k singular values.
// rank == 0 means full delta (no SVD, equivalent to apply_full_delta).
// sigma_min > 0 skips heads where sigma_1 < threshold.
void apply_rank_delta(StateBuf & dst,
                      const StateBuf & sa, const StateBuf & sb,
                      const std::vector<int> & layers,
                      int rank, double alpha = 1.0, double sigma_min = 0.0);

// ── weighted delta ───────────────────────────────────────────────────────────
// dst_head += weights[layer_idx][head] * alpha * (sb_head - sa_head).
// weights is indexed as [layer_index_in_layers_vector][head].
void apply_weighted_delta(StateBuf & dst,
                          const StateBuf & sa, const StateBuf & sb,
                          const std::vector<int> & layers,
                          const std::vector<std::vector<double>> & weights,
                          double alpha = 1.0);

// ── donor-free projection edit ───────────────────────────────────────────────
// For each head: project current state onto v1, replace with target direction.
//   S_new = S - alpha*(S*v1)*v1^T + alpha*(sigma_to * u_to)*v_to^T
//
// dirs_from and dirs_to must have the same length and correspond 1:1.
void apply_donor_free(StateBuf & dst,
                      const std::vector<HeadDecomp> & dirs_from,
                      const std::vector<HeadDecomp> & dirs_to,
                      double sigma_min = 0.0, double alpha = 1.0);

}  // namespace rwkv_probe
