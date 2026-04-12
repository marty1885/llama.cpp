// rwkv_probe/head_decomp.cpp — per-head SVD decomposition.
#include "rwkv_probe/head_decomp.h"
#include "rwkv_probe/numeric.h"

namespace rwkv_probe {

HeadDecomp decompose_head(const StateBuf & sa, const StateBuf & sb,
                          int layer, int head) {
    int hs = sa.geom().head_size;
    auto a = sa.s_head(layer, head);
    auto b = sb.s_head(layer, head);

    std::vector<double> delta(hs * hs);
    double l2_sq = 0;
    for (int r = 0; r < hs; ++r) {
        for (int c = 0; c < hs; ++c) {
            double d = (double)b[r * hs + c] - (double)a[r * hs + c];
            delta[r * hs + c] = d;
            l2_sq += d * d;
        }
    }

    HeadDecomp hd;
    hd.layer = layer;
    hd.head  = head;
    hd.u.resize(hs, 0);
    hd.v.resize(hs, 0);

    if (l2_sq < 1e-20) return hd;

    auto svd = numeric::svd(delta.data(), hs);

    hd.sigma = svd.sigma[0];
    for (int i = 0; i < hs; ++i) {
        hd.u[i] = svd.u_col(0)[i];
    }
    hd.v = svd.v(0);

    double sum_sq = 0;
    for (int i = 0; i < hs; ++i) sum_sq += svd.sigma[i] * svd.sigma[i];
    hd.r1_ratio = (sum_sq > 0) ? svd.sigma[0] * svd.sigma[0] / sum_sq : 0.0;

    return hd;
}

std::vector<HeadDecomp> decompose(const StateBuf & sa, const StateBuf & sb,
                                  const std::vector<int> & layers) {
    int n_head = sa.geom().n_head;
    std::vector<HeadDecomp> out;
    out.reserve(layers.size() * n_head);
    for (int layer : layers) {
        for (int h = 0; h < n_head; ++h) {
            out.push_back(decompose_head(sa, sb, layer, h));
        }
    }
    return out;
}

}  // namespace rwkv_probe
