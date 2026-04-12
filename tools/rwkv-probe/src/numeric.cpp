// rwkv_probe/numeric.cpp — pure CPU helpers.
#include "rwkv_probe/numeric.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

namespace rwkv_probe { namespace numeric {

double entropy(span<const float> probs) {
    double H = 0.0;
    for (std::size_t i = 0; i < probs.size(); ++i) {
        const float v = probs[i];
        if (v > 0) {
            H -= static_cast<double>(v) * std::log(static_cast<double>(v));
        }
    }
    return H;
}

double kl_divergence(span<const float> p, span<const float> q) {
    if (p.size() != q.size()) {
        die("kl_divergence: size mismatch (" + std::to_string(p.size())
            + " vs " + std::to_string(q.size()) + ")");
    }
    double KL = 0.0;
    for (std::size_t i = 0; i < p.size(); ++i) {
        const float pi = p[i];
        const float qi = q[i];
        if (pi > 0 && qi > 0) {
            KL += static_cast<double>(pi)
                * (std::log(static_cast<double>(pi)) - std::log(static_cast<double>(qi)));
        }
    }
    return KL;
}

std::vector<TopK> top_k(span<const float> probs, int k) {
    const int n = static_cast<int>(probs.size());
    if (k > n) k = n;
    if (k <= 0) return {};

    std::vector<int> idx(n);
    for (int i = 0; i < n; ++i) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int a, int b) { return probs[a] > probs[b]; });

    std::vector<TopK> out;
    out.reserve(static_cast<std::size_t>(k));
    for (int i = 0; i < k; ++i) {
        out.push_back({idx[i], probs[idx[i]]});
    }
    return out;
}

void saxpy(span<float> y, span<const float> x, float alpha) {
    if (y.size() != x.size()) {
        die("saxpy: size mismatch (" + std::to_string(y.size())
            + " vs " + std::to_string(x.size()) + ")");
    }
    for (std::size_t i = 0; i < y.size(); ++i) {
        y[i] += alpha * x[i];
    }
}

void normalize_inplace(span<float> v) {
    double norm = 0.0;
    for (std::size_t i = 0; i < v.size(); ++i) {
        norm += static_cast<double>(v[i]) * v[i];
    }
    norm = std::sqrt(norm);
    if (norm > 0) {
        for (std::size_t i = 0; i < v.size(); ++i) {
            v[i] = static_cast<float>(v[i] / norm);
        }
    }
}

std::vector<float> load_floats(const std::string & path,
                               int expected_size,
                               bool normalize) {
    std::vector<float> out;
    std::ifstream ifs(path);
    if (!ifs) {
        die("load_floats: cannot open " + path);
    }
    float v;
    while (ifs >> v) {
        out.push_back(v);
    }
    if (expected_size >= 0 && static_cast<int>(out.size()) != expected_size) {
        die("load_floats: " + path + " has " + std::to_string(out.size())
            + " values, expected " + std::to_string(expected_size));
    }
    if (normalize) {
        normalize_inplace(span<float>(out.data(), out.size()));
    }
    return out;
}

// ── SVD via LAPACK dgesvd ────────────────────────────────────────────────────

// LAPACK Fortran interface (column-major, pass-by-pointer)
extern "C" void dgesvd_(const char * jobu, const char * jobvt,
                         const int * m, const int * n,
                         double * a, const int * lda,
                         double * s,
                         double * u, const int * ldu,
                         double * vt, const int * ldvt,
                         double * work, const int * lwork,
                         int * info);

std::vector<double> SVDResult::v(int k) const {
    // Row k of Vt (column-major): Vt[k + j*n] for j = 0..n-1
    std::vector<double> out(n);
    for (int j = 0; j < n; ++j) {
        out[j] = Vt[k + (std::size_t)j * n];
    }
    return out;
}

SVDResult svd(const double * mat, int n) {
    SVDResult r;
    r.n = n;
    r.sigma.resize(n);
    r.U.resize((std::size_t)n * n);
    r.Vt.resize((std::size_t)n * n);

    // dgesvd expects column-major input, so transpose (row-major → col-major)
    std::vector<double> A((std::size_t)n * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A[i + (std::size_t)j * n] = mat[i * n + j];
        }
    }

    // workspace query
    int lwork = -1;
    double work_query = 0;
    int info = 0;
    dgesvd_("A", "A", &n, &n, A.data(), &n,
            r.sigma.data(), r.U.data(), &n, r.Vt.data(), &n,
            &work_query, &lwork, &info);

    lwork = (int)work_query;
    std::vector<double> work(lwork);
    dgesvd_("A", "A", &n, &n, A.data(), &n,
            r.sigma.data(), r.U.data(), &n, r.Vt.data(), &n,
            work.data(), &lwork, &info);

    if (info != 0) {
        die("dgesvd failed with info=" + std::to_string(info));
    }
    return r;
}

SVDResult svd(const float * mat, int n) {
    std::vector<double> dmat((std::size_t)n * n);
    for (int i = 0; i < n * n; ++i) dmat[i] = mat[i];
    return svd(dmat.data(), n);
}

}}  // namespace rwkv_probe::numeric
