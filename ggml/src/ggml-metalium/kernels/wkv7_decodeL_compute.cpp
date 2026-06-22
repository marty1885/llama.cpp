#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/matmul.h"
#include "api/compute/transpose_wh.h"
#include "api/compute/bcast.h"
#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"

// On-device SFPU generation of the resident const tile.
// Enum tags for the seven mask/const tile variants written by wkv7_gen_const:
//   GM_TRI  (0): lower-triangular mask, [r,c]=1 iff r >= c  (used for TriIncl/omega prefix sum)
//   GM_MSL  (1): strict-lower-triangular mask clipped to cl rows, [r,c]=1 iff r<cl && c<r    (maskSL)
//   GM_MLI  (2): inclusive-lower-triangular mask clipped to cl rows, [r,c]=1 iff r<cl && c<=r (maskLI)
//   GM_IDN  (3): identity, [r,c]=1 iff r==c
//   GM_SEL  (4): single-entry selector: [cl-1,0]=1, all else 0  (extracts token cl-1, i.e. last real token)
//   GM_NCL  (5): "not-column" mask, [r,c]=1 iff c>=cl  (PARTIAL: pads stale token-cols -> neutral w)
//   GM_RWM  (6): row mask, [r,c]=1 iff r<cl            (PARTIAL: zeros stale extract rows before transpose)
namespace { enum { GM_IDN = 3, GM_SEL = 4, GM_NCL = 5, GM_RWM = 6 }; }

// wkv7_gen_const -- SFPU tile generator: fills DST tile 0 with the requested const pattern.
// Runs entirely on the SFPU (no DRAM read), so const tiles are generated once on-core and
// pushed into their CBs resident.  Called from within a tile_regs_acquire/commit bracket.
// The 32-iteration loop visits all 32 SFPU "sub-rows" (each covers 16 elements): the
// bit-field arithmetic recovers the logical [r,c] coordinates from vConstTileId so the
// switch can evaluate a row/col predicate and write 0.0 or 1.0 to dst_reg[k].
// @param which  One of the GM_* enum values selecting the desired mask pattern.
// @param cl     Chunk length (number of real tokens in this chunk, <= 32).
#ifdef TRISC_MATH
namespace {
using namespace sfpi;
inline void wkv7_gen_const(int which, int cl) {
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(0);
    math::set_addr_mod_base();
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
    for (int k = 0; k < 32; k++) {
        const int col_base = ((k >> 3) & 1) * 16 + (k & 1);
        const int row_base = (k >> 4) * 16 + ((k >> 1) & 3) * 4;
        vInt tid = vConstTileId;
        vInt c = (vConstTileId & 15) + col_base;
        vInt r = (tid >> 4) + row_base;
        vFloat val = 0.0f;
        switch (which) {
            case GM_IDN: v_if (r == c) { val = 1.0f; } v_endif; break;
            case GM_SEL: v_if (r == cl - 1) { v_if (c == 0) { val = 1.0f; } v_endif; } v_endif; break;
            case GM_NCL: v_if (c >= cl) { val = 1.0f; } v_endif; break;
            case GM_RWM: v_if (r < cl) { val = 1.0f; } v_endif; break;
        }
        dst_reg[k] = val;
    }
    math::clear_dst_reg_addr();
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU);
    TTI_SETC16(2, 0);
}
}
#endif

#pragma GCC optimize("Os")

namespace {
// Circular-buffer slot assignments for each operand.
constexpr uint32_t cb_a = 0, cb_w = 1, cb_k = 2, cb_v = 3, cb_r = 4, cb_b = 5;
constexpr uint32_t cb_state = 6;   // on-chip carry: current S ([S,S], NS tiles)
constexpr uint32_t cb_Sa = 8, cb_Snew = 9, cb_o = 10, cb_msk = 11;
constexpr uint32_t cb_out = 16;
constexpr uint32_t c_natstage = 21;
constexpr uint32_t cb_ident = 28, ROWM = 3;  // ROWM: slot index of the row-mask tile in cb_ident
// TR[inp]:   true -> transpose tile during extract (a, r, b are stored row-transposed by reader)
// MASK[inp]: true -> multiply by RWM row-mask to zero rows 1..31 (a, k, v, b have multi-row tiles)
constexpr bool TR[6]   = {true,  false, false, true,  true,  false};
constexpr bool MASK[6] = {true,  false, true,  true,  false, true};
}

void kernel_main() {
    uint32_t St      = get_arg_val<uint32_t>(0);  // tiles per head dimension (S/32); S=64 -> St=2
    uint32_t IC      = get_arg_val<uint32_t>(1);  // inner-channel count (unused in this path)
    uint32_t L       = get_arg_val<uint32_t>(2);  // token count (1..32)
    uint32_t Ht      = get_arg_val<uint32_t>(3);  // number of heads (unused in this path)
    uint32_t nc      = get_arg_val<uint32_t>(4);  // writer nc (unused in compute; used by writer)
    uint32_t inst_lo = get_arg_val<uint32_t>(5);  // first instance index for this core
    uint32_t inst_hi = get_arg_val<uint32_t>(6);  // one-past-last instance index for this core
    (void)IC; (void)Ht; (void)nc;
    const uint32_t NS = St * St;  // number of [32x32] tiles in one [S,S] head state

    compute_kernel_hw_startup(cb_state, cb_a, cb_out);

    // Generate the four const tiles into cb_ident
    auto gc = [&](int which, uint32_t slot) {
        tile_regs_acquire();
        MATH(wkv7_gen_const(which, 1));
        tile_regs_commit(); tile_regs_wait();
        pack_tile(0, cb_ident, slot);
        tile_regs_release();
    };
    cb_reserve_back(cb_ident, 4);
    gc(GM_IDN, 0); gc(GM_SEL, 1); gc(GM_NCL, 2); gc(GM_RWM, 3);
    cb_push_back(cb_ident, 4);
    cb_wait_front(cb_ident, 4);   // resident consts are now available in cb_ident[0..3]

    for (uint32_t inst = inst_lo; inst < inst_hi; inst++) {
        // Reads the NS tiles that form the initial [S,S] head state S_0 from the
        // staging CB c_natstage
        cb_reserve_back(cb_state, NS);
        cb_wait_front(c_natstage, NS);
        copy_tile_init(c_natstage);
        tile_regs_acquire();
        for (uint32_t t = 0; t < NS; t++) copy_tile(c_natstage, t, t);
        tile_regs_commit(); tile_regs_wait();
        for (uint32_t t = 0; t < NS; t++) pack_tile(t, cb_state, t);
        tile_regs_release();
        cb_pop_front(c_natstage, NS);
        cb_push_back(cb_state, NS); cb_wait_front(cb_state, NS);

        for (uint32_t tok = 0; tok < L; tok++) {
            const bool last = (tok + 1 == L);
            // Extract 6 per-token inputs from c_natstage
            // Each input arrives as Sttiles in c_natstage with the token vector in
            // row 0 and zeros (or garbage) in rows 1..31.  Depending on MASK[inp] and TR[inp]:
            //   MASK only:       element-wise multiply by RWM (cb_ident[ROWM]) to zero rows 1..31.
            //   TR only:         transpose the tile (column vector -> row vector).
            //   MASK then TR:    mask first (via cb_msk scratch), then transpose the masked tile.
            //   neither:         straight copy_tile.
            // After extraction each input CB (cb_a..cb_b) holds St tiles ready for matmul.
            // NOTE: when both MASK and TR are needed, a two-pass approach through cb_msk is used
            // because the hardware does not support combined mul+transpose in a single op.
            for (uint32_t inp = 0; inp < 6; inp++) cb_reserve_back(cb_a + inp, St);
            for (uint32_t inp = 0; inp < 6; inp++) {
                cb_wait_front(c_natstage, St);
                if (MASK[inp] && TR[inp]) {
                    // Two-pass: mask into cb_msk, then transpose cb_msk into cb_a+inp.
                    cb_reserve_back(cb_msk, St);
                    mul_tiles_init(c_natstage, cb_ident);
                    tile_regs_acquire();
                    for (uint32_t st = 0; st < St; st++) mul_tiles(c_natstage, cb_ident, st, ROWM, st);
                    tile_regs_commit(); tile_regs_wait();
                    for (uint32_t st = 0; st < St; st++) pack_tile(st, cb_msk, st);
                    tile_regs_release();
                    cb_push_back(cb_msk, St); cb_wait_front(cb_msk, St);
                    transpose_wh_init(cb_msk, cb_a + inp);
                    tile_regs_acquire();
                    for (uint32_t st = 0; st < St; st++) transpose_wh_tile(cb_msk, st, st);
                    tile_regs_commit(); tile_regs_wait();
                    for (uint32_t st = 0; st < St; st++) pack_tile(st, cb_a + inp, st);
                    tile_regs_release();
                    cb_pop_front(cb_msk, St);
                } else {
                    // Single-pass: mask, transpose, or copy, depending on the input's flags.
                    if (MASK[inp])    mul_tiles_init(c_natstage, cb_ident);
                    else if (TR[inp]) transpose_wh_init(c_natstage, cb_a + inp);
                    else              copy_tile_init(c_natstage);
                    tile_regs_acquire();
                    for (uint32_t st = 0; st < St; st++) {
                        if (MASK[inp])    mul_tiles(c_natstage, cb_ident, st, ROWM, st);
                        else if (TR[inp]) transpose_wh_tile(c_natstage, st, st);
                        else              copy_tile(c_natstage, st, st);
                    }
                    tile_regs_commit(); tile_regs_wait();
                    for (uint32_t st = 0; st < St; st++) pack_tile(st, cb_a + inp, st);
                    tile_regs_release();
                }
                cb_pop_front(c_natstage, St);
            }
            for (uint32_t inp = 0; inp < 6; inp++) cb_push_back(cb_a + inp, St);
            for (uint32_t inp = 0; inp < 6; inp++) cb_wait_front(cb_a + inp, St);

            // Sa = S @ a
            cb_reserve_back(cb_Sa, St);
            mm_init(cb_state, cb_a, cb_Sa);
            tile_regs_acquire();
            for (uint32_t it = 0; it < St; it++)
                for (uint32_t jt = 0; jt < St; jt++) matmul_tiles(cb_state, cb_a, it * St + jt, jt, it);
            tile_regs_commit(); tile_regs_wait();
            for (uint32_t it = 0; it < St; it++) pack_tile(it, cb_Sa, it);
            tile_regs_release();
            cb_push_back(cb_Sa, St); cb_wait_front(cb_Sa, St);

            // S' = S*diag(w) + Sa(x)b + v(x)k
            //
            // term 1: S * diag(w)
            // term 2: Sa (x) b            (x) = outer product (via matmul)
            // term 3: v (x) k
            cb_reserve_back(cb_Snew, NS);
            tile_regs_acquire();
            mul_bcast_rows_init_short(cb_state, cb_w);
            for (uint32_t o = 0; o < NS; o++) { uint32_t jt = o % St; mul_tiles_bcast_rows(cb_state, cb_w, o, jt, o); }
            mm_init_short(cb_Sa, cb_b, 0);
            for (uint32_t o = 0; o < NS; o++) { uint32_t it = o / St, jt = o % St; matmul_tiles(cb_Sa, cb_b, it, jt, o); }
            mm_init_short(cb_v, cb_k, 0);
            for (uint32_t o = 0; o < NS; o++) { uint32_t it = o / St, jt = o % St; matmul_tiles(cb_v, cb_k, it, jt, o); }
            tile_regs_commit(); tile_regs_wait();
            for (uint32_t o = 0; o < NS; o++) pack_tile(o, cb_Snew, o);
            tile_regs_release();
            cb_push_back(cb_Snew, NS); cb_wait_front(cb_Snew, NS);

            // o = S' @ r
            cb_reserve_back(cb_o, St);
            mm_init(cb_Snew, cb_r, cb_o);
            tile_regs_acquire();
            for (uint32_t it = 0; it < St; it++)
                for (uint32_t jt = 0; jt < St; jt++) matmul_tiles(cb_Snew, cb_r, it * St + jt, jt, it);
            tile_regs_commit(); tile_regs_wait();
            for (uint32_t it = 0; it < St; it++) pack_tile(it, cb_o, it);
            tile_regs_release();
            cb_push_back(cb_o, St); cb_wait_front(cb_o, St);

            // ---- Emit: region-1 output + region-2 state into cb_out ----
            // Each token pushes St + NS tiles to cb_out in a single reservation:
            //   slots [0, St):      region-1 -- o transposed (column vector -> row form for
            //                       the ggml writer); St tiles packed from cb_o.
            //   slots [St, St+NS):  region-2 -- S' verbatim (NS tiles from cb_Snew).
            // The ggml writer (nc=L, tpc=1) reads St+NS tiles per token and interprets the
            // last token's region-2 as the updated head state to write back to DRAM.
            cb_reserve_back(cb_out, St + NS);
            transpose_wh_init(cb_o, cb_out);
            tile_regs_acquire();
            for (uint32_t it = 0; it < St; it++) transpose_wh_tile(cb_o, it, it);
            tile_regs_commit(); tile_regs_wait();
            for (uint32_t it = 0; it < St; it++) pack_tile(it, cb_out, it);
            tile_regs_release();
            copy_tile_init(cb_Snew);
            tile_regs_acquire();
            for (uint32_t t = 0; t < NS; t++) copy_tile(cb_Snew, t, t);
            tile_regs_commit(); tile_regs_wait();
            for (uint32_t t = 0; t < NS; t++) pack_tile(t, cb_out, St + t);
            tile_regs_release();
            cb_push_back(cb_out, St + NS);

            // On-chip carry: S <- S' (only if not last token)
            cb_pop_front(cb_state, NS);
            if (!last) {
                cb_reserve_back(cb_state, NS);
                copy_tile_init(cb_Snew);
                tile_regs_acquire();
                for (uint32_t t = 0; t < NS; t++) copy_tile(cb_Snew, t, t);
                tile_regs_commit(); tile_regs_wait();
                for (uint32_t t = 0; t < NS; t++) pack_tile(t, cb_state, t);
                tile_regs_release();
                cb_push_back(cb_state, NS); cb_wait_front(cb_state, NS);
            }
            cb_pop_front(cb_Snew, NS); cb_pop_front(cb_o, St); cb_pop_front(cb_Sa, St);
            for (uint32_t inp = 0; inp < 6; inp++) cb_pop_front(cb_a + inp, St);
        }
    }
}
