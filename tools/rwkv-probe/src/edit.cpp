// rwkv_probe/edit.cpp — state edit operations.
#include "rwkv_probe/edit.h"
#include "rwkv_probe/numeric.h"

#include <algorithm>
#include <cmath>

namespace rwkv_probe {

void apply_full_delta(StateBuf & dst,
                      const StateBuf & sa, const StateBuf & sb,
                      const std::vector<int> & layers,
                      double alpha) {
    int n_head = dst.geom().n_head;
    int hs     = dst.geom().head_size;
    for (int layer : layers) {
        for (int h = 0; h < n_head; ++h) {
            auto da = sa.s_head(layer, h);
            auto db = sb.s_head(layer, h);
            auto dd = dst.s_head(layer, h);
            for (int i = 0; i < hs * hs; ++i) {
                dd[i] += (float)(alpha * ((double)db[i] - (double)da[i]));
            }
        }
    }
}

void apply_rank_delta(StateBuf & dst,
                      const StateBuf & sa, const StateBuf & sb,
                      const std::vector<int> & layers,
                      int rank, double alpha, double sigma_min) {
    int n_head = dst.geom().n_head;
    int hs     = dst.geom().head_size;

    // rank == 0 means full delta
    if (rank == 0 && sigma_min <= 0.0) {
        apply_full_delta(dst, sa, sb, layers, alpha);
        return;
    }

    for (int layer : layers) {
        for (int h = 0; h < n_head; ++h) {
            auto da = sa.s_head(layer, h);
            auto db = sb.s_head(layer, h);
            auto dd = dst.s_head(layer, h);

            std::vector<double> delta(hs * hs);
            for (int r = 0; r < hs; ++r) {
                for (int c = 0; c < hs; ++c) {
                    delta[r * hs + c] = (double)db[r * hs + c] - (double)da[r * hs + c];
                }
            }

            auto svdr = numeric::svd(delta.data(), hs);

            if (sigma_min > 0.0 && svdr.sigma[0] < sigma_min) continue;

            if (rank > 0) {
                int k = std::min(rank, (int)svdr.sigma.size());
                for (int r = 0; r < hs; ++r) {
                    for (int c = 0; c < hs; ++c) {
                        double val = 0.0;
                        for (int ki = 0; ki < k; ++ki) {
                            // U is col-major: U[r + ki*hs]
                            // Vt row ki, col c: Vt[ki + c*hs]
                            val += svdr.sigma[ki]
                                 * svdr.U[r + (std::size_t)ki * hs]
                                 * svdr.Vt[ki + (std::size_t)c * hs];
                        }
                        dd[r * hs + c] += (float)(alpha * val);
                    }
                }
            } else {
                // rank == 0, sigma_min > 0: full delta but sigma-filtered
                for (int i = 0; i < hs * hs; ++i) {
                    dd[i] += (float)(alpha * ((double)db[i] - (double)da[i]));
                }
            }
        }
    }
}

void apply_weighted_delta(StateBuf & dst,
                          const StateBuf & sa, const StateBuf & sb,
                          const std::vector<int> & layers,
                          const std::vector<std::vector<double>> & weights,
                          double alpha) {
    int n_head = dst.geom().n_head;
    int hs     = dst.geom().head_size;

    for (std::size_t li = 0; li < layers.size(); ++li) {
        int layer = layers[li];
        for (int h = 0; h < n_head; ++h) {
            double w = weights[li][h];
            if (w == 0.0) continue;

            auto da = sa.s_head(layer, h);
            auto db = sb.s_head(layer, h);
            auto dd = dst.s_head(layer, h);
            for (int i = 0; i < hs * hs; ++i) {
                dd[i] += (float)(w * alpha * ((double)db[i] - (double)da[i]));
            }
        }
    }
}

void apply_donor_free(StateBuf & dst,
                      const std::vector<HeadDecomp> & dirs_from,
                      const std::vector<HeadDecomp> & dirs_to,
                      double sigma_min, double alpha) {
    int hs = dst.geom().head_size;

    for (std::size_t di = 0; di < dirs_from.size(); ++di) {
        const auto & df = dirs_from[di];
        const auto & dt = dirs_to[di];

        if (df.sigma < sigma_min || dt.sigma < sigma_min) continue;

        auto s = dst.s_head(df.layer, df.head);

        // compute projection: proj[r] = sum_c S[r,c] * v1[c]
        std::vector<double> proj(hs, 0);
        for (int r = 0; r < hs; ++r) {
            for (int c = 0; c < hs; ++c) {
                proj[r] += (double)s[r * hs + c] * df.v[c];
            }
        }

        // S_new = S - alpha*proj*v1^T + alpha*(sigma_to * u_to)*v_to^T
        for (int r = 0; r < hs; ++r) {
            for (int c = 0; c < hs; ++c) {
                double remove = alpha * proj[r] * df.v[c];
                double add    = alpha * dt.sigma * dt.u[r] * dt.v[c];
                s[r * hs + c] += (float)(-remove + add);
            }
        }
    }
}

}  // namespace rwkv_probe
