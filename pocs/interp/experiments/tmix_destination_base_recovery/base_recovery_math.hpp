#pragma once

#include <cblas.h>
#include <openblas/lapacke.h>
#include <TMatrixD.h>

#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

namespace tmix_base_recovery {

constexpr double k_min_norm = 1e-12;

inline double dot(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("vector width mismatch");
    }
    double result = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        result += static_cast<double>(a[i]) * b[i];
    }
    return result;
}

inline double norm(const std::vector<float> & value) {
    return std::sqrt(dot(value, value));
}

inline void center(std::vector<float> & value) {
    double mean = 0;
    for (float x : value) {
        mean += x;
    }
    mean /= value.size();
    for (float & x : value) {
        x -= static_cast<float>(mean);
    }
}

inline std::vector<float> unit(std::vector<float> value) {
    const double length = norm(value);
    if (!(length > k_min_norm) || !std::isfinite(length)) {
        throw std::runtime_error("zero or non-finite vector");
    }
    for (float & x : value) {
        x = static_cast<float>(x / length);
    }
    return value;
}

inline double rel_error(const std::vector<float> & a, const std::vector<float> & b) {
    std::vector<float> difference(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        difference[i] = a[i] - b[i];
    }
    return norm(difference) / std::max(norm(a), k_min_norm);
}

inline double projection(const TMatrixD & basis, const std::vector<float> & value) {
    double energy = 0;
    for (int column = 0; column < basis.GetNcols(); ++column) {
        double coordinate = 0;
        for (int row = 0; row < basis.GetNrows(); ++row) {
            coordinate += basis(row, column) * value[row];
        }
        energy += coordinate * coordinate;
    }
    return energy;
}

inline std::vector<float> residualize(const TMatrixD & basis, std::vector<float> value) {
    for (int column = 0; column < basis.GetNcols(); ++column) {
        double coordinate = 0;
        for (int row = 0; row < basis.GetNrows(); ++row) {
            coordinate += basis(row, column) * value[row];
        }
        for (int row = 0; row < basis.GetNrows(); ++row) {
            value[row] -= static_cast<float>(coordinate * basis(row, column));
        }
    }
    return value;
}

inline double orthogonality_error(const TMatrixD & basis) {
    double maximum = 0;
    for (int i = 0; i < basis.GetNcols(); ++i) {
        for (int j = 0; j < basis.GetNcols(); ++j) {
            double dot_product = 0;
            for (int row = 0; row < basis.GetNrows(); ++row) {
                dot_product += basis(row, i) * basis(row, j);
            }
            maximum = std::max(maximum, std::abs(dot_product - (i == j)));
        }
    }
    return maximum;
}

inline TMatrixD fit_basis(const std::vector<const std::vector<float> *> & rows, int rank) {
    if (static_cast<int>(rows.size()) < rank) {
        throw std::runtime_error("insufficient rows for rank");
    }
    const int          samples = rows.size(), width = rows.front()->size();
    std::vector<float> matrix(static_cast<size_t>(samples) * width), gram(static_cast<size_t>(samples) * samples),
        eigen(rank), eigenvectors(static_cast<size_t>(samples) * rank);
    std::vector<lapack_int> support(2 * rank);
    for (int c = 0; c < width; ++c) {
        for (int r = 0; r < samples; ++r) {
            matrix[r + static_cast<size_t>(samples) * c] = (*rows[r])[c];
        }
    }
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, samples, samples, width, 1, matrix.data(), samples,
                matrix.data(), samples, 0, gram.data(), samples);
    lapack_int selected = 0;
    if (LAPACKE_ssyevr(LAPACK_COL_MAJOR, 'V', 'I', 'U', samples, gram.data(), samples, 0, 0, samples - rank + 1,
                       samples, 0, &selected, eigen.data(), eigenvectors.data(), samples, support.data()) != 0 ||
        selected != rank) {
        throw std::runtime_error("top-rank Gram eigendecomposition failed");
    }
    TMatrixD result(width, rank);
    for (int c = 0; c < rank; ++c) {
        const int    component = rank - 1 - c;
        const double singular  = std::sqrt(std::max(0.0F, eigen[component]));
        if (!(singular > 1e-15)) {
            throw std::runtime_error("rank-deficient basis");
        }
        for (int row = 0; row < width; ++row) {
            double value = 0;
            for (int n = 0; n < samples; ++n) {
                value += (*rows[n])[row] * eigenvectors[n + static_cast<size_t>(samples) * component];
            }
            result(row, c) = value / singular;
        }
        for (int prior = 0; prior < c; ++prior) {
            double p = 0;
            for (int row = 0; row < width; ++row) {
                p += result(row, prior) * result(row, c);
            }
            for (int row = 0; row < width; ++row) {
                result(row, c) -= p * result(row, prior);
            }
        }
        double length = 0;
        for (int row = 0; row < width; ++row) {
            length += result(row, c) * result(row, c);
        }
        length = std::sqrt(length);
        if (!(length > k_min_norm)) {
            throw std::runtime_error("rank-deficient reconstructed basis");
        }
        for (int row = 0; row < width; ++row) {
            result(row, c) /= length;
        }
    }
    return result;
}

// Deterministic randomized thin SVD used only for repeated control fits.  The
// caller validates it against the exact native/source path before use.
inline TMatrixD fit_basis_randomized(const std::vector<const std::vector<float> *> & rows, int rank, uint64_t seed) {
    const int          m = rows.size(), n = rows.front()->size(), l = rank + 16;
    std::vector<float> a(static_cast<size_t>(m) * n), omega(static_cast<size_t>(n) * l), y(static_cast<size_t>(m) * l),
        z(static_cast<size_t>(n) * l), b(static_cast<size_t>(l) * n), gram(static_cast<size_t>(l) * l), eigen(l);
    std::mt19937_64                 rng(seed);
    std::normal_distribution<float> normal;
    for (int c = 0; c < n; ++c) {
        for (int r = 0; r < m; ++r) {
            a[r + static_cast<size_t>(m) * c] = (*rows[r])[c];
        }
    }
    for (float & value : omega) {
        value = normal(rng);
    }
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, l, n, 1, a.data(), m, omega.data(), n, 0, y.data(), m);
    for (int iteration = 0; iteration < 2; ++iteration) {
        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, n, l, m, 1, a.data(), m, y.data(), m, 0, z.data(), n);
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, l, n, 1, a.data(), m, z.data(), n, 0, y.data(), m);
    }
    for (int c = 0; c < l; ++c) {
        for (int p = 0; p < c; ++p) {
            float d = cblas_sdot(m, y.data() + static_cast<size_t>(m) * p, 1, y.data() + static_cast<size_t>(m) * c, 1);
            cblas_saxpy(m, -d, y.data() + static_cast<size_t>(m) * p, 1, y.data() + static_cast<size_t>(m) * c, 1);
        }
        const float length = cblas_snrm2(m, y.data() + static_cast<size_t>(m) * c, 1);
        if (!(length > 1e-8F)) {
            throw std::runtime_error("randomized basis rank deficiency");
        }
        cblas_sscal(m, 1 / length, y.data() + static_cast<size_t>(m) * c, 1);
    }
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, l, n, m, 1, y.data(), m, a.data(), m, 0, b.data(), l);
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, l, l, n, 1, b.data(), l, b.data(), l, 0, gram.data(), l);
    if (LAPACKE_ssyevd(LAPACK_COL_MAJOR, 'V', 'U', l, gram.data(), l, eigen.data()) != 0) {
        throw std::runtime_error("randomized small Gram eigendecomposition failed");
    }
    TMatrixD result(n, rank);
    for (int c = 0; c < rank; ++c) {
        const int    component = l - 1 - c;
        const double singular  = std::sqrt(std::max(0.F, eigen[component]));
        if (!(singular > 1e-12)) {
            throw std::runtime_error("randomized basis rank deficiency");
        }
        for (int row = 0; row < n; ++row) {
            double value = 0;
            for (int q = 0; q < l; ++q) {
                value += b[q + static_cast<size_t>(l) * row] * gram[q + static_cast<size_t>(l) * component];
            }
            result(row, c) = value / singular;
        }
    }
    return result;
}
}  // namespace tmix_base_recovery
