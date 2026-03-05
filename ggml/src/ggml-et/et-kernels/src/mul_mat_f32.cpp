#include <etsoc/common/utils.h>
#include <stdint.h>
#include "ggml_tensor.h"
#include "platform.h"

#include "et_tensor.hpp"
#include "tensor.h"

constexpr et::scp_region<0,  16, et::fp32> scp_a;
constexpr et::scp_region<16, 16, et::fp32> scp_b;
constexpr unsigned TILE = 16;
constexpr unsigned NUM_HARTS = 1024;

extern "C"
int entry_point(struct ggml_et_binary_params* params, void* env) {
    uint64_t hart_id = get_hart_id();
    if (hart_id & 1) return 0;
    uint64_t gid = ((hart_id >> 6) << 5) + ((hart_id >> 1) & 0x1F);

    const int64_t K = params->src0.ne[0];
    const int64_t M = params->src0.ne[1];
    const int64_t N = params->src1.ne[1];
    const int64_t ne2_0 = params->src0.ne[2], ne3_0 = params->src0.ne[3];
    const int64_t ne2_1 = params->src1.ne[2], ne3_1 = params->src1.ne[3];

    // Byte strides for batch dimensions, converted to float offsets
    const int64_t bs2_0 = params->src0.nb[2] / sizeof(float);
    const int64_t bs3_0 = params->src0.nb[3] / sizeof(float);
    const int64_t bs2_1 = params->src1.nb[2] / sizeof(float);
    const int64_t bs3_1 = params->src1.nb[3] / sizeof(float);
    const int64_t bs2_d = params->dst.nb[2]  / sizeof(float);
    const int64_t bs3_d = params->dst.nb[3]  / sizeof(float);

    // Row strides in bytes — for tensor load/store hardware
    const uint64_t stride_s0 = (uint64_t)params->src0.nb[1];
    const uint64_t stride_s1 = (uint64_t)params->src1.nb[1];
    const uint64_t stride_d  = (uint64_t)params->dst.nb[1];

    // Row strides in floats — for pointer arithmetic
    const int64_t rs0 = (int64_t)(stride_s0 / sizeof(float));
    const int64_t rs1 = (int64_t)(stride_s1 / sizeof(float));
    const int64_t rd  = (int64_t)(stride_d  / sizeof(float));

    const float* src0_base = (const float*)params->src0.data;
    const float* src1_base = (const float*)params->src1.data;
    float*       dst_base  = (float*)params->dst.data;

    et::setup_l1scp();
    et::clear_tensor_error();

    const int64_t m_tiles = M / TILE;
    const int64_t n_tiles = (N + TILE - 1) / TILE;
    const int64_t tpb     = m_tiles * n_tiles;
    const int64_t batches = ne2_1 * ne3_1;
    const int64_t total   = batches * tpb;
    const int64_t r2 = ne2_1 / ne2_0, r3 = ne3_1 / ne3_0;

    et::matmul_result<et::fp32> c;

    for (int64_t tile = gid; tile < total; tile += NUM_HARTS) {
        const int64_t bi = tile / tpb, ti = tile % tpb;
        const int64_t ni = ti / m_tiles, mi = ti % m_tiles;
        const int64_t i3 = bi / ne2_1, i2 = bi % ne2_1;

        const float* s0 = src0_base + (i3/r3)*bs3_0 + (i2/r2)*bs2_0;
        const float* s1 = src1_base + i3*bs3_1 + i2*bs2_1;
        float*       d  = dst_base  + i3*bs3_d + i2*bs2_d;

        const int64_t mb = mi * TILE, nb = ni * TILE;
        const unsigned n_cur = (nb + TILE <= N) ? TILE : (unsigned)(N - nb);

        for (int64_t kb = 0; kb < K; kb += TILE) {
            bool first = kb == 0;
            const unsigned k_cur = (kb + TILE <= K) ? TILE : (unsigned)(K - kb);

            // There are 2 load ports - we use 0 for matrix A and 1 for matrix B
            // these data are typed so loading the wrong format is impossible
            auto la = scp_a.load<0>(&s1[nb * rs1 + kb], n_cur, stride_s1);
            auto lb = scp_b.load_transpose<1>(&s0[mb * rs0 + kb], k_cur, stride_s0);
            // Loads are async so we wait for completion
            la.wait();
            lb.wait();

            // Perform matrix multiplication
            c = et::matmul(scp_a, scp_b, n_cur, k_cur, first);
            // Wait for completion
            c.wait();
        }

        // Write back to memory
        c.store(&d[nb * rd + mb], n_cur, stride_d).wait();
    }

    if(!et::get_tensor_error()) {
        return 1;
    }

    et::fence();
    return 0;
}
