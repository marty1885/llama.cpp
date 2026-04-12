// rwkv_probe/numeric.h — pure CPU helpers for analysis on float vectors.
//
// Lifts the math from lens_print_table (entropy, KL, top-k) and load_direction
// (read floats, normalize) and the steering loop (saxpy) in the original
// rwkv-probe.cpp.
#pragma once

#include "util.h"

#include <string>
#include <vector>

namespace rwkv_probe { namespace numeric {

// Shannon entropy in nats.
double entropy(span<const float> probs);

// KL divergence KL(p || q) in nats. Skips terms where p_i == 0 or q_i == 0.
double kl_divergence(span<const float> p, span<const float> q);

// indices of the top-k elements of `probs`, sorted descending. The returned
// vector has size min(k, probs.size()).
struct TopK {
    int   index;
    float prob;
};
std::vector<TopK> top_k(span<const float> probs, int k);

// y[i] += alpha * x[i]. y and x must have the same size.
void saxpy(span<float> y, span<const float> x, float alpha);

// in-place L2 normalization. No-op if the vector is all zeros.
void normalize_inplace(span<float> v);

// Read floats from a text file (whitespace-separated). Optionally normalize.
// expected_size: if >= 0, throws when the loaded count doesn't match.
std::vector<float> load_floats(const std::string & path,
                               int expected_size = -1,
                               bool normalize = true);

// ── SVD via LAPACK dgesvd ────────────────────────────────────────────────────
// Full SVD of an n×n row-major matrix.  Returns singular values (descending)
// and optionally U and Vᵀ columns.

struct SVDResult {
    std::vector<double> sigma;  // singular values, descending
    std::vector<double> U;      // n×n column-major (U[:,k] is k-th left vector)
    std::vector<double> Vt;     // n×n column-major (Vt[k,:] is k-th right vector)
    int n = 0;

    // convenience: k-th left singular vector (length n)
    const double * u_col(int k) const { return U.data() + (std::size_t)k * n; }
    // convenience: k-th right singular vector (length n, row of Vᵀ = col of V)
    // Note: Vt is column-major, so row k is stride-n apart. Use v() helper.
    std::vector<double> v(int k) const;
};

// Full SVD of n×n row-major double matrix.
SVDResult svd(const double * mat, int n);

// Convenience: from float data.
SVDResult svd(const float * mat, int n);

}}  // namespace rwkv_probe::numeric
