#include "api/compute/pack.h"
#include "api/compute/reg_api.h"

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/clamp.h"
#include "api/compute/matmul.h"
#include "api/compute/transpose_wh.h"
#include "api/compute/bcast.h"
#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"   // log_tile

// On-device SFPU generation of the resident const tile.
// Enum tags for the seven mask/const tile variants written by wkv7_gen_const:
//   GM_TRI  (0): lower-triangular mask, [r,c]=1 iff r >= c  (used for TriIncl/omega prefix sum)
//   GM_MSL  (1): strict-lower-triangular mask clipped to cl rows, [r,c]=1 iff r<cl && c<r    (maskSL)
//   GM_MLI  (2): inclusive-lower-triangular mask clipped to cl rows, [r,c]=1 iff r<cl && c<=r (maskLI)
//   GM_IDN  (3): identity, [r,c]=1 iff r==c
//   GM_SEL  (4): single-entry selector: [cl-1,0]=1, all else 0  (extracts token cl-1, i.e. last real token)
//   GM_NCL  (5): "not-column" mask, [r,c]=1 iff c>=cl  (PARTIAL: pads stale token-cols -> neutral w)
//   GM_RWM  (6): row mask, [r,c]=1 iff r<cl            (PARTIAL: zeros stale extract rows before transpose)
namespace { enum { GM_TRI = 0, GM_MSL, GM_MLI, GM_IDN, GM_SEL, GM_NCL, GM_RWM }; }
#ifdef TRISC_MATH
namespace {
using namespace sfpi;
// wkv7_gen_const -- SFPU tile generator: fills DST tile 0 with the requested const pattern.
// Runs entirely on the SFPU (no DRAM read), so const tiles are generated once on-core and
// pushed into their CBs resident.  Called from within a tile_regs_acquire/commit bracket.
// The 32-iteration loop visits all 32 SFPU "sub-rows" (each covers 16 elements): the
// bit-field arithmetic recovers the logical [r,c] coordinates from vConstTileId so the
// switch can evaluate a row/col predicate and write 0.0 or 1.0 to dst_reg[k].
// @param which  One of the GM_* enum values selecting the desired mask pattern.
// @param cl     Chunk length (number of real tokens in this chunk, <= 32).
inline void wkv7_gen_const(int which, int cl) {
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(0);
    math::set_addr_mod_base();
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
    for (int k = 0; k < 32; k++) {
        const int col_base = ((k >> 3) & 1) * 16 + (k & 1);
        const int row_base = (k >> 4) * 16 + ((k >> 1) & 3) * 4;
        vInt tid = vConstTileId;                       // materialize CReg for >>
        vInt c = (vConstTileId & 15) + col_base;
        vInt r = (tid >> 4) + row_base;
        vFloat val = 0.0f;
        switch (which) {
            case GM_TRI: v_if (r <= c) { val = 1.0f; } v_endif; break;
            case GM_MSL: v_if (r < cl) { v_if (c < r)     { val = 1.0f; } v_endif; } v_endif; break;
            case GM_MLI: v_if (r < cl) { v_if (c < r + 1) { val = 1.0f; } v_endif; } v_endif; break;
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


// This kernel is too large that we need to ban inlining
#pragma GCC optimize("Os")

namespace {
// CB (circular buffer) index assignments.  The kernel shares some CB slots between
// phases whose lifetimes do not overlap; aliases are documented where they are introduced.
//
// Inputs streamed per group by the reader (one entry per NB-group):
constexpr uint32_t cb_a = 0, cb_w = 1, cb_k = 2, cb_v = 3, cb_r = 4, cb_b = 5;
// Initial recurrent state S0 for the group (NB*NS tiles, [head,S,S]):
constexpr uint32_t cb_state = 6, cb_tri = 7;
// Omega chain and its derived quantities -- some slots are REUSED by the triangular
// inverse (see the "triangular inverse" section comment for the alias map):
constexpr uint32_t cb_logw = 8, cb_cumlog = 9, cb_omega = 10;
constexpr uint32_t cb_romega = 11, cb_rw = 12, cb_oprev = 13;
// Normalized per-token coefficient tiles (live across multiple stages):
constexpr uint32_t cb_ahat = 14, cb_btil = 15, cb_out = 16;
constexpr uint32_t cb_ktil = 17, cb_rhat = 18;

// Resident mask tiles (never popped after their initial push):
constexpr uint32_t cb_maskSL = 19, cb_maskLI = 20, cb_ahatT = 8, cb_rhatT = 13;
// Reader -> extract staging CBs (dedicated, never aliases):
constexpr uint32_t c_natstage = 21, c_selstage = 22;
// Gram-matrix intermediates and raw scratch:
constexpr uint32_t cb_B = 23, cb_Ka = 24, cb_Mb = 25, cb_Mk = 26, cb_gramraw = 27;
// Resident const tiles packed into cb_ident (single CB, multiple tile slots):
//   tile 0 = identity [L,L]
//   tile 1 = sel_last
//   tile 2 = notcol           [*,t]=1 iff t>=cl, pads w on a partial last chunk (cl<32)
//   tile 3 = rowmask          [token,*]=1 iff token<cl, zeros stale rows on a partial chunk
constexpr uint32_t cb_ident = 28;   // identity [L,L] (resident)
// triangular inverse reuses the now-dead omega-chain CBs by index:
//   N=logw, M ping-pong=cumlog/omega, P ping-pong=romega/rw, M-square scratch=oprev.
// sel_last/notcol/rowmask merged into the ident CB as tiles 1/2/3 (frees indices).
constexpr uint32_t cb_sel_last = cb_ident, SELLAST = 1;
// Partial-chunk masks (active when cl<32, neutral when cl==32): notcol[*,t]=(t>=cl)
// as ident tile 2 (w += notcol so padding token-cols 1 -> log/recip finite).
// rowmask[token,dim]=(token<cl) as ident tile 3 (zero stale extract rows >= cl before transpose).
constexpr uint32_t cb_notcol = cb_ident, NOTCOL = 2;
constexpr uint32_t cb_rowmask = cb_ident, ROWM = 3;
// omega_last: last-column (token cl-1) of omega, shape [S,1] per instance (NT tiles total).
// Used in the final-state computation as the per-element decay factor at the chunk boundary.
constexpr uint32_t cb_omega_last = 30;
// SA [S,L] (NT tiles): S0*Ahat + V*Ka^T, accumulated during the SA stage and
// parked here from the end of stage-SA until it is consumed by Out and final-state.
constexpr uint32_t cb_SA = 31;
// On-chip Sfinal carry buffer between consecutive chunks (used when L>32 spans NC>1
// chunks).  Holds NB*NS tiles; the producer (final-state emit) and consumer (extract
// seed) in successive iterations of the chunk (cc) loop exchange through this CB without DRAM.
constexpr uint32_t cb_carry = 29;
}

// SA is computed into scratch cb_rw and consumed by the Out stage; for the final
// state it must live past Out, so a clean copy is parked in cb_SA right after SA.

// Helpers to reduce code written
namespace ttggml {

// gen_tiles: core tile-production helper.
// Reserves `pad` (or `n` if pad==0) slots in cb_dst, runs `init()` once to configure
// math engines, then processes tiles in groups of up to 4 (the DST register file holds
// 8 bf16 / 16 fp32 tiles, but we cap at 4 to leave headroom).  Each group:
//   tile_regs_acquire -> body(dstIdx, tileIdx) for each tile -> commit/wait -> pack -> release.
// After the loop, pushes `pad` slots and waits for `held+pad` to be visible to the reader.
// The `pad > n` idiom keeps a mixed-width CB's write pointer aligned to the WIDEST access
// width used later (e.g. NT-wide reads in SA/Out), so a subsequent wide cb_wait_front can
// never straddle fifo_limit into the physical neighbor.  Only the first `n` tile slots are
// populated; the extra pad slots advance the pointer but are never read (their stale
// contents are inert).
// NOTE: DST capacity is 8 bf16 tiles; this helper holds at most 4 simultaneously per acquire.
// @param cb_dst  Destination CB to write produced tiles into.
// @param n       Number of real tiles to produce (body is called n times).
// @param init    Callable (no args) that runs once before the tile loop to initialize math engines.
// @param body    Callable (uint32_t dstReg, uint32_t tileIdx) that computes one tile into DST slot dstReg.
// @param held    Number of extra already-present tiles to include in the final cb_wait_front count.
// @param pad     Total slots to reserve/push/pop (>= n); if 0, defaults to n.  Keeps the CB write
//                pointer aligned to the widest subsequent read width.
template <typename Init, typename Body>
inline void gen_tiles(uint32_t cb_dst, uint32_t n, Init init, Body body,
                      uint32_t held = 0, uint32_t pad = 0) {
    uint32_t pn = pad < n ? n : pad;
    cb_reserve_back(cb_dst, pn);
    init();
    for (uint32_t b = 0; b < n; b += 4) {
        uint32_t c = (n - b) < 4 ? (n - b) : 4;
        tile_regs_acquire();
        for (uint32_t i = 0; i < c; i++) body(i, b + i);
        tile_regs_commit(); tile_regs_wait();
        for (uint32_t i = 0; i < c; i++) pack_tile(i, cb_dst, b + i);
        tile_regs_release();
    }
    cb_push_back(cb_dst, pn);
    cb_wait_front(cb_dst, held + pn);
}

// wrappers
//
// copy -- copies n tiles from CB `in` into CB `dst` (tile-for-tile, no arithmetic).
// @param dst  Destination CB to write tiles into.
// @param in   Source CB to read tiles from (tile indices 0..n-1).
// @param n    Number of tiles to copy.
// @param held Number of extra already-present tiles to include in the final cb_wait_front count.
// @param pad  Number of total tiles to reserve/push/pop (>= n) to keep the CB pointer aligned for later wide reads.
inline void copy(uint32_t dst, uint32_t in, uint32_t n, uint32_t held = 0, uint32_t pad = 0) {
    gen_tiles(dst, n, [&]{ copy_tile_init(in); },
        [&](uint32_t d, uint32_t t){ copy_tile(in, t, d); }, held, pad);
}
// add -- elementwise add: dst[t] = x[t] + y[t]  for t in 0..n-1.
// @param dst  Destination CB to write result tiles into.
// @param x    First source CB (tile indices 0..n-1).
// @param y    Second source CB (tile indices 0..n-1).
// @param n    Number of tiles to process.
// @param held Number of extra already-present tiles to include in the final cb_wait_front count.
inline void add(uint32_t dst, uint32_t x, uint32_t y, uint32_t n, uint32_t held = 0) {
    gen_tiles(dst, n, [&]{ add_tiles_init(x, y); },
        [&](uint32_t d, uint32_t t){ add_tiles(x, y, t, t, d); }, held);
}
// mul -- elementwise multiply: dst[t] = x[t] * y[t]  for t in 0..n-1.
// @param dst  Destination CB to write result tiles into.
// @param x    First source CB (tile indices 0..n-1).
// @param y    Second source CB (tile indices 0..n-1).
// @param n    Number of tiles to process.
// @param held Number of extra already-present tiles to include in the final cb_wait_front count.
inline void mul(uint32_t dst, uint32_t x, uint32_t y, uint32_t n, uint32_t held = 0) {
    gen_tiles(dst, n, [&]{ mul_tiles_init(x, y); },
        [&](uint32_t d, uint32_t t){ mul_tiles(x, y, t, t, d); }, held);
}
// transpose -- transposes n tiles from CB `in` into CB `dst`:  dst[t] = in[t]^T.
// @param dst  Destination CB to write transposed tiles into.
// @param in   Source CB to read tiles from (tile indices 0..n-1).
// @param n    Number of tiles to transpose.
// @param held Number of extra already-present tiles to include in the final cb_wait_front count.
// @param pad  Number of total tiles to reserve/push/pop (>= n) to keep the CB pointer aligned for later wide reads.
inline void transpose(uint32_t dst, uint32_t in, uint32_t n, uint32_t held = 0, uint32_t pad = 0) {
    gen_tiles(dst, n, [&]{ transpose_wh_init(in, dst); },
        [&](uint32_t d, uint32_t t){ transpose_wh_tile(in, t, d); }, held, pad);
}

// add_bc0 -- tile-level "broadcast tile 0 of a0" add: dst[t] = a0[0] + y[t]  for t in 0..n-1.
// a0's tile 0 is used as the fixed left operand for every t.  This is NOT the hardware
// column/row broadcast instruction (which fans a 32x1 or 1x32 strip); it is a software
// loop that pins the source tile index of the left operand to 0 while advancing the right
// operand index t.  Used in the triangular inverse to compute I+N by keeping the identity
// tile (cb_ident[0]) pinned and adding it to each of the n N-matrix tiles.
// @param dst  Destination CB to write result tiles into.
// @param a0   CB whose tile 0 is the fixed (broadcast) left operand for every addition.
// @param y    CB providing the varying right operand (tile indices 0..n-1).
// @param n    Number of tiles to process.
// @param held Number of extra already-present tiles to include in the final cb_wait_front count.
// @param pad  Number of total tiles to reserve/push/pop (>= n) to keep the CB pointer aligned for later wide reads.
inline void add_bc0(uint32_t dst, uint32_t a0, uint32_t y, uint32_t n, uint32_t held = 0, uint32_t pad = 0) {
    gen_tiles(dst, n, [&]{ add_tiles_init(a0, y); },
        [&](uint32_t d, uint32_t t){ add_tiles(a0, y, 0, t, d); }, held, pad);
}
// mul_bcast -- hardware column-broadcast multiply: dst[base+t] = x[base+t] * col[:,g_col]
// for t in 0..n-1, where col is a [S,1] column tile (one tile per instance) stored in CB
// `col` at tile index `base+t` (i.e. the same per-instance offset as x).
// Uses mul_tiles_bcast_cols (a hardware broadcast instruction, NOT a software tile-0 pin)
// to fan the single column entry across all 32 columns of x.
// Used to compute bsharp = btil * omega_last and ksharp = ktil * omega_last in the
// per-instance final-state loop.
// @param dst   Destination CB to write result tiles into.
// @param x     Source CB providing the [S,L] per-instance tiles (tile indices base..base+n-1).
// @param col   CB providing the [S,1] column tile to broadcast (one tile per instance, at index base+t).
// @param n     Number of tiles to process (St tiles per instance).
// @param base  Tile-index offset into x and col corresponding to this instance's start (= g*St).
// @param held  Number of extra already-present tiles to include in the final cb_wait_front count.
inline void mul_bcast(uint32_t dst, uint32_t x, uint32_t col, uint32_t n,
                      uint32_t base, uint32_t held = 0) {
    gen_tiles(dst, n, [&]{ mul_bcast_cols_init_short(x, col); },
        [&](uint32_t d, uint32_t t){ mul_tiles_bcast_cols(x, col, base + t, base + t, d); }, held);
}

// sfpu -- two-phase SFPU unary: copy n tiles from CB `in` into DST, then apply an
// elementwise SFPU op in place, then pack to CB `dst`.
// The two-phase structure (copy-all then op-all within each acquire group) is required
// because copy_tile and the SFPU op need different math engine configurations and cannot
// interleave per tile without an expensive reconfig between every pair.  Instead, all
// copies in the group complete under copy_tile_init, then op_init reconfigures once for
// the batch of SFPU ops.
// For an FPU result that is already in DST (e.g. the matmul+exp fusion), use gen_tiles
// with both operations in the body lambda instead of this helper.
// @param dst      Destination CB to write SFPU-processed tiles into.
// @param in       Source CB to read input tiles from (tile indices 0..n-1).
// @param n        Number of tiles to process.
// @param op_init  Callable (no args) that configures the SFPU op once per acquire group (e.g. exp_tile_init).
// @param op       Callable (uint32_t dstReg) that applies the SFPU op to one DST tile in place (e.g. exp_tile).
// @param held     Number of extra already-present tiles to include in the final cb_wait_front count.
template <typename InitOp, typename Op>
inline void sfpu(uint32_t dst, uint32_t in, uint32_t n, InitOp op_init, Op op,
                 uint32_t held = 0) {
    cb_reserve_back(dst, n);
    for (uint32_t b = 0; b < n; b += 4) {
        uint32_t c = (n - b) < 4 ? (n - b) : 4;
        tile_regs_acquire();
        copy_tile_init(in);
        for (uint32_t i = 0; i < c; i++) copy_tile(in, b + i, i);
        op_init();
        for (uint32_t i = 0; i < c; i++) op(i);
        tile_regs_commit(); tile_regs_wait();
        for (uint32_t i = 0; i < c; i++) pack_tile(i, dst, b + i);
        tile_regs_release();
    }
    cb_push_back(dst, n);
    cb_wait_front(dst, held + n);
}

}  // namespace ttggml

void kernel_main() {
    // Kernel arguments:
    //   St  -- number of 32x32 tiles per head dimension (S/32; S=64 -> St=2).
    //   IC  -- total instance count = G*H (G sequences * H heads per sequence).
    //   L   -- chunk length in tokens (<= 32; equals the full sequence length when L<=32).
    //   Ht  -- head-tile stride (unused at runtime; consumed only by the reader, kept for ABI).
    uint32_t St = get_arg_val<uint32_t>(0);
    uint32_t IC = get_arg_val<uint32_t>(1);
    uint32_t L  = get_arg_val<uint32_t>(2);
    uint32_t Ht = get_arg_val<uint32_t>(3);
    // NS = St*St = number of [32x32] tiles in one head's [S,S] state matrix (4 when S=64).
    const uint32_t NS = St * St;
    // nin: number of streamed [S,L] input tiles per head (= L*St, one row-strip per head).
    const uint32_t nin = L * St;   // streamed input tiles/head = one head's row-tile (Ht-agnostic)
    // The outer instance loop is preceded by an inner chunk loop over NC consecutive
    // chunks of up to 32 tokens each (a single chunk when L<=32).  inst_lo/inst_hi are
    // SPMD slice indices so multiple cores can divide the IC instances between them.
    uint32_t NC = get_arg_val<uint32_t>(4);   // number of 32-token chunks (group-carry)
    uint32_t inst_lo = get_arg_val<uint32_t>(5);   // SPMD: this core's instance slice
    uint32_t inst_hi = get_arg_val<uint32_t>(6);   // (defaults 0..IC for single core)
    (void)Ht;

    // ---- Matmul helpers (index arithmetic visible inline) ----------------------
    //
    // gram -- computes a masked [L,L] gram matrix for one NB-group:
    //   gramraw[g] = at[g*St .. g*St+St-1] @ y[g*St .. g*St+St-1]    ([L,L], contraction over St)
    //   dst[g]     = gramraw[g] * mask[0]                            (element-wise, mask is resident tile 0)
    // The intermediate gramraw is pushed to cb_gramraw then immediately popped after the mask multiply.
    // 'at' is a transposed coefficient matrix (e.g. cb_ahatT or cb_rhatT), 'y' is the matching
    // non-transposed matrix (e.g. cb_btil or cb_ktil).  The mask selects either the strictly-lower
    // (maskSL, causal: output depends only on strictly earlier tokens) or inclusive-lower (maskLI)
    // triangular region.  Result shape per instance: one [L,L] tile.
    // @param at    CB holding the transposed [L,S] input tiles (nb*St tiles, instance g at g*St).
    // @param y     CB holding the [S,L] input tiles to contract against (nb*St tiles, instance g at g*St).
    // @param mask  CB holding the resident [L,L] triangular mask (tile 0 is always used).
    // @param dst   CB to write the nb masked gram-matrix tiles into.
    // @param nbi   Number of instances in this NB-group (= nb, typically 2).
    auto gram = [&](uint32_t at, uint32_t y, uint32_t mask, uint32_t dst, uint32_t nbi) {
        ttggml::gen_tiles(cb_gramraw, nbi, [&]{ mm_init(at, y, cb_gramraw); },
            [&](uint32_t d, uint32_t o){ for (uint32_t kt = 0; kt < St; kt++)
                matmul_tiles(at, y, o*St + kt, o*St + kt, d); });
        ttggml::gen_tiles(dst, nbi, [&]{ mul_tiles_init(cb_gramraw, mask); },
            [&](uint32_t d, uint32_t o){ mul_tiles(cb_gramraw, mask, o, 0, d); });
        cb_pop_front(cb_gramraw, nbi);
    };
    // mmstate -- multiply a flat [S,S] state block by an [S,L] matrix X, for all nb instances.
    // Computes: dst[g*St + it] = sum_{jt=0}^{St-1} s[g*NS + it*St + jt] @ x[g*St + jt]
    // i.e. for each instance g, the [S,L] result row-tile 'it' is the contraction of the
    // it-th row-strip of S (St tiles) with the jt column-strips of X, accumulating into DST.
    // Total output: nbi*St tiles.  Tile layout in 's': instance g occupies tiles [g*NS .. g*NS+NS-1]
    // in row-major order (it*St + jt within the block).
    // @param s    CB holding the flat state tiles, shape [NB*NS] (each instance g at g*NS).
    // @param x    CB holding the [S,L] matrix to multiply (each instance g at g*St).
    // @param dst  CB to write the nbi*St result tiles into.
    // @param nbi  Number of instances in this NB-group.
    auto mmstate = [&](uint32_t s, uint32_t x, uint32_t dst, uint32_t nbi) {
        ttggml::gen_tiles(dst, nbi*St, [&]{ mm_init(s, x, dst); },
            [&](uint32_t d, uint32_t o){ uint32_t g = o/St, it = o%St;
                for (uint32_t jt = 0; jt < St; jt++)
                    matmul_tiles(s, x, g*NS + it*St + jt, g*St + jt, d); });
    };
    // mmcolt -- multiply an [S,L] matrix X by a per-instance [L,L] column matrix C, for all nb instances.
    // Computes: dst[g*St + it] = x[g*St + it] @ C[g]
    // i.e. each row-tile of X is right-multiplied by the single [L,L] tile for instance g.
    // Total output: nbi*St tiles.
    // tr=1 activates the B-operand transpose flag in mm_init (the unpacker transposes C before use),
    // computing X @ C^T without a separate transpose_wh pass.  The flag is part of mm_init's unpacker
    // configuration, NOT a hardware matmul instruction variant -- it simply tells the unpacker to
    // deliver C in transposed order into SrcB.
    // NOTE: tr=1 (B-operand transpose in unpack) avoids a full transpose_wh pass on C, but the flag
    // lives in mm_init; calling code must not mix tr=0 and tr=1 configs in the same acquire group.
    // @param x    CB holding the [S,L] input tiles (each instance g at g*St).
    // @param col  CB holding the [L,L] per-instance column tiles (one tile per instance, at index g).
    // @param dst  CB to write the nbi*St result tiles into.
    // @param nbi  Number of instances in this NB-group.
    // @param tr   B-operand transpose flag: 0 = use C as-is; 1 = transpose C in-unpack (X @ C^T).
    auto mmcolt = [&](uint32_t x, uint32_t col, uint32_t dst, uint32_t nbi, uint32_t tr = 0) {
        ttggml::gen_tiles(dst, nbi*St, [&]{ mm_init(x, col, dst, tr); },
            [&](uint32_t d, uint32_t o){ uint32_t g = o/St, it = o%St;
                matmul_tiles(x, col, g*St + it, g, d); });
    };

    unary_op_init_common(cb_w, cb_out);

    // SFPU-generate the resident const tiles
    // Generate all mask/const tiles
    auto gc = [&](int which, uint32_t cb, uint32_t slot) {
        tile_regs_acquire();
        MATH(wkv7_gen_const(which, (int)L));
        tile_regs_commit(); tile_regs_wait();
        pack_tile(0, cb, slot);
        tile_regs_release();
    };
    // cb_tri   tile 0: lower-triangular TriIncl [r>=c] for the omega prefix-sum matmul.
    cb_reserve_back(cb_tri, 1);    gc(GM_TRI, cb_tri, 0);    cb_push_back(cb_tri, 1);
    // cb_maskSL tile 0: strict-lower-triangular [r<cl, c<r]  for causal gram matrices (B, Ka).
    cb_reserve_back(cb_maskSL, 1); gc(GM_MSL, cb_maskSL, 0); cb_push_back(cb_maskSL, 1);
    // cb_maskLI tile 0: inclusive-lower-triangular [r<cl, c<=r] for output gram matrices (Mb, Mk).
    cb_reserve_back(cb_maskLI, 1); gc(GM_MLI, cb_maskLI, 0); cb_push_back(cb_maskLI, 1);
    // ident CB tiles 0-3 = ident, sel_last, notcol, rowmask (notcol/rowmask pad the
    // partial last chunk where cl<32; neutral when cl==32).
    cb_reserve_back(cb_ident, 4);
    gc(GM_IDN, cb_ident, 0); gc(GM_SEL, cb_ident, 1); gc(GM_NCL, cb_ident, 2); gc(GM_RWM, cb_ident, 3);
    cb_push_back(cb_ident, 4);

    cb_wait_front(cb_tri, 1);             // resident global constants
    cb_wait_front(cb_maskSL, 1);
    cb_wait_front(cb_maskLI, 1);
    cb_wait_front(cb_ident, 4);   // 0=ident, 1=sel_last, 2=notcol, 3=rowmask

    // main loop
    // Process NB independent (head, sequence) instances per group. Batching NB=2
    // instances for reduced overhead: each gen_tiles/sfpu/et el call processes NB
    // tiles.
    // FIXME: NB=1 triggers a latent single-instance bug in the chunked recurrence path.
    // DO NOT reduce NB without checking.
    // IC = G*H total instances; only the SPMD slice [inst_lo, inst_hi) is processed by
    // this core.  The reader prefetches the next group's c_natstage data while the
    // compute for the current group is in flight.
    constexpr uint32_t NB = 2;
    const uint32_t inst_first = inst_lo, inst_last = inst_hi;
    for (uint32_t inst0 = inst_first; inst0 < inst_last; inst0 += NB) {
        uint32_t nb = (IC - inst0) < NB ? (IC - inst0) : NB;
        // NT = total tiles for this group across all nb instances (nb*St row-tiles per head dimension).
        uint32_t NT = nb * St;

        // Chunk loop over NC chunks of up to 32 tokens each (NC==1 when L<=32):
        //   first=true (cc==0):     seed cb_state from the reader's initial S0.
        //   first=false (cc>0):     copy the on-chip cb_carry into cb_state.
        //   last=false (cc<NC-1):   stash S_final into cb_carry instead of emitting to DRAM.
        //   last=true (cc==NC-1):   emit S_final normally (region-2 write to DRAM output).
        // When L<=32, NC==1 so the loop runs once with first==last==true.
        for (uint32_t cc = 0; cc < NC; cc++) {
            const bool first = (cc == 0), last = (cc + 1 == NC);

        for (uint32_t inp = 0; inp < 6; inp++) cb_reserve_back(cb_a + inp, NT);
        if (first) cb_reserve_back(cb_state, nb * NS);
        // Stage inputs instance-major: the reader delivers each instance's 6 inputs as
        // St [token,dim] tiles, then (chunk 0) that instance's NS state tiles.
        for (uint32_t g = 0; g < nb; g++) {
            for (uint32_t inp = 0; inp < 6; inp++) {
                cb_wait_front(c_natstage, St);
                for (uint32_t st = 0; st < St; st++) {
                    uint32_t xsrc = c_natstage, xidx = st;
                    // Zero stale padding rows (token >= cl) via element-wise multiply
                    // with cb_rowmask (resident tile ROWM) before transposing, so the [dim,token]
                    // tiles carry zeros for out-of-range token slots.  cb_logw used as scratch.
                    cb_reserve_back(cb_logw, 1);
                    mul_tiles_init(c_natstage, cb_rowmask);
                    tile_regs_acquire();
                    mul_tiles(c_natstage, cb_rowmask, st, ROWM, 0);
                    tile_regs_commit(); tile_regs_wait();
                    pack_tile(0, cb_logw, 0);
                    tile_regs_release();
                    cb_push_back(cb_logw, 1); cb_wait_front(cb_logw, 1);
                    xsrc = cb_logw; xidx = 0;
                    transpose_wh_init(xsrc, cb_a + inp);
                    tile_regs_acquire();
                    transpose_wh_tile(xsrc, xidx, 0);
                    tile_regs_commit(); tile_regs_wait();
                    pack_tile(0, cb_a + inp, g * St + st);
                    tile_regs_release();
                    cb_pop_front(cb_logw, 1);
                }
                cb_pop_front(c_natstage, St);
            }
            if (first) {
                cb_wait_front(c_natstage, NS);
                copy_tile_init(c_natstage);
                tile_regs_acquire();
                for (uint32_t t = 0; t < NS; t++) copy_tile(c_natstage, t, t);
                tile_regs_commit(); tile_regs_wait();
                for (uint32_t t = 0; t < NS; t++) pack_tile(t, cb_state, g * NS + t);
                tile_regs_release();
                cb_pop_front(c_natstage, NS);
            }
        }
        for (uint32_t inp = 0; inp < 6; inp++) cb_push_back(cb_a + inp, NT);
        if (first) cb_push_back(cb_state, nb * NS);
        else {
            // Non-first chunk (cc>0): the initial state for this chunk is the Sfinal that
            // was stashed into cb_carry at the end of the previous chunk iteration.
            // Copy the NB*NS carry tiles into cb_state for use in this chunk's recurrence.
            cb_wait_front(cb_carry, nb * NS);
            ttggml::copy(cb_state, cb_carry, nb * NS);
            cb_pop_front(cb_carry, nb * NS);
        }

        cb_wait_front(cb_state, nb * NS);
        cb_wait_front(cb_a, NT); cb_wait_front(cb_w, NT); cb_wait_front(cb_k, NT);
        cb_wait_front(cb_v, NT); cb_wait_front(cb_r, NT); cb_wait_front(cb_b, NT);

        // compute per-token decay factors
        // omega[t] = exp( sum_{s=0}^{t} log(w[s]) ) = exp( logw @ TriIncl )
        //   where TriIncl[r,c] = 1 iff r>=c (lower-triangular inclusive, = cb_tri).
        // This gives the cumulative product of w along the token axis as a prefix-sum
        // in log space, exponentiated back.
        //
        // Step 1: wclamp = clamp(w [+ notcol], WMIN=0.1, WMAX=2.0) -> cb_cumlog (scratch).
        //   Under PARTIAL, notcol (tile NOTCOL, [*,t]=1 iff t>=cl) is added so that padding
        //   token-columns (t>=cl, w=0) become 1 before clamping, keeping log/recip finite.
        //   WMIN=0.1 ensures omega=prod(w) >= 0.1^32 ~ 1e-32 (safely representable in bf16)
        //   even for adversarial inputs (test-backend-ops fills w ~ Uniform[-1,1]; w<=0
        //   would give log(w)=NaN and w~0 would give 1/omega=inf).
        //   Real RWKV-7 decays have w in (0,1), so the clamp is effectively a no-op for them.
        //   cb_cumlog is later freed and reused as the M ping-pong buffer of the triangular
        //   inverse, which does NOT begin until after cb_cumlog is popped below.
        //   WMAX=2.0 is inert for real/test inputs where w <= 1.
        constexpr uint32_t WMIN = 0x3dcccccdu;  // 0.1f
        constexpr uint32_t WMAX = 0x40000000u;  // 2.0f  (real/test w <= 1, so upper is inert)
        // wfix: for padding token-cols t>=cl, add notcol (shifting 0->1) before clamping
        // (notcol is all-zero when cl==32, so this reduces to a plain clamp).
        ttggml::gen_tiles(cb_cumlog, NT, [&]{ add_tiles_init(cb_w, cb_notcol); clamp_tile_init(); },
            [&](uint32_t d, uint32_t t){ add_tiles(cb_w, cb_notcol, t, NOTCOL, d); clamp_tile(d, WMIN, WMAX); });
        // Step 2: logw = log(wclamp) -> cb_logw.  Both logw and 1/wclamp derive from
        // the same clamped copy (cb_cumlog) so they are numerically consistent.
        ttggml::sfpu(cb_logw, cb_cumlog, NT, []{ log_tile_init(); }, [](uint32_t d){ log_tile(d); });
        // Step 3: omega = exp(logw @ TriIncl): matmul straight into DST, then exp in place.
        // The matmul+exp fusion avoids materializing an intermediate cb_cumlog2; no cb_cumlog
        // tile is consumed here (it stays alive for the 1/wclamp step below).
        ttggml::gen_tiles(cb_omega, NT, [&]{ mm_init(cb_logw, cb_tri, cb_omega); exp_tile_init(); },
            [&](uint32_t d, uint32_t t){ matmul_tiles(cb_logw, cb_tri, t, 0, d); exp_tile(d); });

        // Step 4: omega_last[:,0] = omega @ sel_last: extract the last real token's column
        // (token index cl-1) from omega as a [S,1] column tile per instance.
        // sel_last is the resident [L,L] one-hot tile at cb_ident slot SELLAST: element
        // [cl-1, 0]=1, all others 0.  The matmul picks out column cl-1 of omega and compresses
        // it to a single-column tile used later as the per-element state decay factor.
        ttggml::gen_tiles(cb_omega_last, NT, [&]{ mm_init(cb_omega, cb_sel_last, cb_omega_last); },
            [&](uint32_t d, uint32_t t){ matmul_tiles(cb_omega, cb_sel_last, t, SELLAST, d); });

        // normalize
        // Derive the five scaled input streams used by the gram and state stages.
        // All are elementwise (tile-by-tile); the SFPU two-phase helper is used throughout.
        //   romega[t] = 1 / omega[t]                  (needed for ahat, btil, ktil below)
        //   rw[t]     = 1 / wclamp[t]                 (needed for oprev = omega/w)
        //   oprev[t]  = omega[t] / wclamp[t]          (= omega_{t-1}, decay to previous step)
        //   ahat[t]   = a[t] * oprev[t]               (a scaled by one-step-earlier omega)
        //   btil[t]   = b[t] * romega[t]              (b normalized by cumulative omega)
        //   ktil[t]   = k[t] * romega[t]              (k normalized by cumulative omega)
        //   rhat[t]   = r[t] * omega[t]               (r scaled by cumulative omega)
        // cb_cumlog is popped after rw to free the slot for the triangular inverse.
        ttggml::sfpu(cb_romega, cb_omega, NT, []{ recip_tile_init(); }, [](uint32_t d){ recip_tile(d); });
        // 1/w' = recip(wclamp) (same clamped copy as logw -> consistent, finite).
        ttggml::sfpu(cb_rw, cb_cumlog, NT, []{ recip_tile_init(); }, [](uint32_t d){ recip_tile(d); });
        cb_pop_front(cb_cumlog, NT);   // free scratch for the triangular inverse
        ttggml::mul(cb_oprev, cb_omega, cb_rw, NT);
        ttggml::mul(cb_ahat, cb_a, cb_oprev, NT);
        ttggml::mul(cb_btil, cb_b, cb_romega, NT);
        ttggml::mul(cb_ktil, cb_k, cb_romega, NT);
        ttggml::mul(cb_rhat, cb_r, cb_omega, NT);

        // Release intermediate omega-chain temporaries now that all five outputs exist.
        // cb_cumlog was already popped above; logw and omega were fused into later products
        // and are no longer needed.
        cb_pop_front(cb_logw, NT); cb_pop_front(cb_omega, NT);   // cumlog fused away
        cb_pop_front(cb_romega, NT); cb_pop_front(cb_rw, NT); cb_pop_front(cb_oprev, NT);

        // Gram matrices for the 4 causal [L,L] products
        // Transpose ahat and rhat once; then form the four masked gram matrices:
        //   B  = maskSL * (ahatT @ btil)  [L,L] per instance  (strict-lower, causal a*b)
        //   Ka = maskSL * (ahatT @ ktil)  [L,L] per instance  (strict-lower, causal a*k)
        //   Mb = maskLI * (rhatT @ btil)  [L,L] per instance  (inclusive-lower, output r*b)
        //   Mk = maskLI * (rhatT @ ktil)  [L,L] per instance  (inclusive-lower, output r*k)
        // Each gram() call contracts over the S dimension (St tiles) for all nb instances.
        ttggml::transpose(cb_ahatT, cb_ahat, NT);
        ttggml::transpose(cb_rhatT, cb_rhat, NT);
        gram(cb_ahatT, cb_btil, cb_maskSL, cb_B, nb);
        gram(cb_ahatT, cb_ktil, cb_maskSL, cb_Ka, nb);
        gram(cb_rhatT, cb_btil, cb_maskLI, cb_Mb, nb);
        gram(cb_rhatT, cb_ktil, cb_maskLI, cb_Mk, nb);
        cb_pop_front(cb_ahatT, NT); cb_pop_front(cb_rhatT, NT);

        // ---- Release spent inputs; note survivor lifetimes for later stages -----
        // Freed now (no longer needed): a, w, k, r, b.
        // Survivors and their next consumer:
        //   B    -> triangular inverse
        //   Ka   -> SA stage
        //   Mb   -> Out stage
        //   Mk   -> Out stage
        //   ahat -> SA stage
        //   rhat -> Out stage
        //   btil -> final-state loop
        //   ktil -> final-state loop
        //   v    -> SA stage, Out stage, final-state loop
        //   state -> SA stage, Out stage, final-state loop
        cb_pop_front(cb_a, NT); cb_pop_front(cb_w, NT); cb_pop_front(cb_k, NT);
        cb_pop_front(cb_r, NT); cb_pop_front(cb_b, NT);

        // Triangular inverse: T = (I+B^T)^{-1} via doubling
        // Computes the inverse of the unit-lower-triangular matrix (I + B^T) using the
        // five-step Cayley-Hamilton / doubling identity:
        //   T = (I+N)(I+N^2)(I+N^4)(I+N^8)(I+N^16),  N = B^T  (nilpotent of order L<=32)
        // Only nb tiles are live at once (one per instance), but ALL omega-chain CBs
        // (cb_logw, cb_cumlog, cb_omega, cb_romega, cb_rw, cb_oprev) are reused as
        // M / P ping-pong and scratch buffers:
        //   N  lives in cb_logw     (transposed B, NT-padded)
        //   M ping-pong: cb_cumlog / cb_omega
        //   P ping-pong: cb_romega  / cb_rw
        //   M-square scratch: cb_oprev (copy of M, since matmul can't alias L/R operands)
        //
        // IMPORTANT: NT padding on every op:
        // The omega-chain CBs are later read at NT-tile width (SA and Out stages).  If the
        // triangular inverse pushed only nb<NT tiles, the CB write pointer would drift by nb
        // each step, causing a subsequent NT-wide cb_wait_front to straddle fifo_limit into
        // the physically adjacent CB -> silent data corruption.  Every gen_tiles call here
        // uses pad=NT to keep the pointer NT-aligned.  Only nb real tiles are produced and
        // read; the pad slots advance the pointer but are never consumed.  cb_B is pushed at
        // nb (not NT) so its pop remains nb.
        cb_wait_front(cb_B, nb);
        // N = B^T  -> cb_logw  (read nb from cb_B, push NT-padded to cb_logw)
        ttggml::transpose(cb_logw, cb_B, nb, 0, NT);         // N = B^T  -> cb_logw
        cb_pop_front(cb_B, nb);
        // P = I + N  -> cb_romega  (add_bc0 pins cb_ident tile 0 as the fixed left operand)
        ttggml::add_bc0(cb_romega, cb_ident, cb_logw, nb, 0, NT);  // P = I + N -> cb_romega
        // M = N  -> cb_cumlog  (initial M before squaring)
        ttggml::copy(cb_cumlog, cb_logw, nb, 0, NT);         // M = N     -> cb_cumlog

        // Doubling loop: 4 iterations squares M (N, N^2, N^4, N^8) and accumulates P.
        // After iteration it: M = N^{2^{it+1}}, P = (I+N)(I+N^2)...(I+N^{2^{it+1}}).
        uint32_t Mc = cb_cumlog, Mn = cb_omega, Pc = cb_romega, Pn = cb_rw;
        for (uint32_t it = 0; it < 4; it++) {
            // Mcopy = Mc -> cb_oprev  (matmul cannot read the same CB for both left and right operand)
            ttggml::copy(cb_oprev, Mc, nb, 0, NT);
            // M = Mc @ Mcopy -> Mn  (M squared)
            ttggml::gen_tiles(Mn, nb, [&]{ mm_init(Mc, cb_oprev, Mn); },
                [&](uint32_t d, uint32_t g){ matmul_tiles(Mc, cb_oprev, g, g, d); }, 0, NT);
            cb_pop_front(cb_oprev, NT);
            // Compute P = Pc @ (I+Mn) = Pc + Pc@Mn without materializing I+Mn.
            // Seed DST with Pc (copy_tile), then accumulate Pc@Mn on top (matmul_tiles adds
            // to the existing DST partial).  Both seed and matmul use srcA=Pc, so no
            // reconfig_data_format_srca is required between them -- mm_init_short only
            // reconfigures srcB (Mn) and math; the copy partial survives in DST.
            // Two-phase within each acquire (seed-all then mm-all) avoids per-tile init thrash
            // at the cost of two passes over the c2 tiles.  NT-padded for pointer alignment.
            cb_reserve_back(Pn, NT);
            for (uint32_t b2 = 0; b2 < nb; b2 += 4) {
                uint32_t c2 = (nb - b2) < 4 ? (nb - b2) : 4;
                tile_regs_acquire();
                copy_tile_init(Pc);
                for (uint32_t i = 0; i < c2; i++) copy_tile(Pc, b2 + i, i);
                mm_init_short(Pc, Mn, 0);
                for (uint32_t i = 0; i < c2; i++) matmul_tiles(Pc, Mn, b2 + i, b2 + i, i);
                tile_regs_commit(); tile_regs_wait();
                for (uint32_t i = 0; i < c2; i++) pack_tile(i, Pn, b2 + i);
                tile_regs_release();
            }
            cb_push_back(Pn, NT); cb_wait_front(Pn, NT);
            cb_pop_front(Mc, NT); cb_pop_front(Pc, NT);
            uint32_t t;
            t = Mc; Mc = Mn; Mn = t;
            t = Pc; Pc = Pn; Pn = t;
        }
        // N (cb_logw) and the final M (Mc, unused) are consumed; T lives in Pc (= cb_romega
        // after 4 swap iterations), NT-padded, with nb real tiles.
        cb_pop_front(cb_logw, NT);   // N done
        cb_pop_front(Mc, NT);        // final M unused
        // T now lives in Pc (== cb_romega after 4 swaps), NT-padded (nb real tiles).

        // SA = (S0@Ahat + V@Ka^T) @ T   [S,L] (T in cb_romega)
        // Computes the chunk-parallel "state contribution" matrix SA in two steps:
        //   RHS = S0@Ahat + V@Ka^T         (accumulated in one DST pass -> cb_oprev, NT tiles)
        //   SA  = RHS @ T                  (mmcolt with T, result -> cb_rw, NT tiles)
        //
        // RHS accumulation: for each tile o=(g*St+it), the FPU first computes S0@Ahat
        // (using mm_init_short(state, ahat, 0) -- no B-transpose) then accumulates V@Ka^T
        // on top (mm_init_short(v, Ka, 1) -- B-transpose: Ka unpacked as Ka^T).
        // Both terms share the same DST partial; no intermediate CB for either S0A or VKa
        // is needed and no explicit add is performed.
        // After the RHS loop, SA = RHS @ T is computed via mmcolt (one [L,L] tile T per
        // instance, no B-transpose).  A clean copy of SA is immediately parked into cb_SA
        // (flat-strip, NT tiles) for the final-state loop which needs it after Out is done.
        cb_reserve_back(cb_oprev, NT);
        for (uint32_t b = 0; b < NT; b += 4) {
            uint32_t c4 = (NT - b) < 4 ? (NT - b) : 4;
            tile_regs_acquire();
            mm_init_short(cb_state, cb_ahat, 0);
            for (uint32_t i = 0; i < c4; i++) { uint32_t o = b+i, g = o/St, it = o%St;
                for (uint32_t jt = 0; jt < St; jt++) matmul_tiles(cb_state, cb_ahat, g*NS+it*St+jt, g*St+jt, i); }
            mm_init_short(cb_v, cb_Ka, 1);                    // V @ Ka^T (B-transpose), accumulate
            for (uint32_t i = 0; i < c4; i++) { uint32_t o = b+i, g = o/St, it = o%St;
                matmul_tiles(cb_v, cb_Ka, g*St+it, g, i); }
            tile_regs_commit(); tile_regs_wait();
            for (uint32_t i = 0; i < c4; i++) pack_tile(i, cb_oprev, b+i);
            tile_regs_release();
        }
        cb_push_back(cb_oprev, NT); cb_wait_front(cb_oprev, NT);
        mmcolt(cb_oprev, cb_romega, cb_rw, nb);         // SA = RHS @ T -> cb_rw (NT)
        cb_pop_front(cb_oprev, NT);
        cb_pop_front(cb_romega, NT);                    // T consumed (pushed NT-padded by the inverse)
        cb_pop_front(cb_Ka, nb); cb_pop_front(cb_ahat, NT);
        ttggml::copy(cb_SA, cb_rw, NT);                 // park SA for final-state

        // OUT = S0@Rhat + SA@Mb^T + V@Mk^T   [S,L]
        // Computes the full per-token output matrix (NT tiles) by accumulating three additive
        // terms into a single DST pass -> cb_romega:
        //   S0R  = S0  @ Rhat    (mm_init_short(state, rhat, 0)  -- no B-transpose)
        //   SAMb = SA  @ Mb^T    (mm_init_short(rw,    Mb,   1)  -- B-transpose: Mb as Mb^T)
        //   VMk  = V   @ Mk^T   (mm_init_short(v,     Mk,   1)  -- B-transpose: Mk as Mk^T)
        // SA is in cb_rw (the copy parked above for Out; cb_SA holds the surviving copy).
        // No intermediate CBs for S0R / SAMb / VMk and no explicit add operations are needed.
        // After the loop, Out lives in cb_romega (NT tiles, [S,L] layout per instance).
        cb_reserve_back(cb_romega, NT);
        for (uint32_t b = 0; b < NT; b += 4) {
            uint32_t c4 = (NT - b) < 4 ? (NT - b) : 4;
            tile_regs_acquire();
            mm_init_short(cb_state, cb_rhat, 0);             // S0R = S0 @ Rhat
            for (uint32_t i = 0; i < c4; i++) { uint32_t o = b+i, g = o/St, it = o%St;
                for (uint32_t jt = 0; jt < St; jt++) matmul_tiles(cb_state, cb_rhat, g*NS+it*St+jt, g*St+jt, i); }
            mm_init_short(cb_rw, cb_Mb, 1);                  // + SA @ Mb^T
            for (uint32_t i = 0; i < c4; i++) { uint32_t o = b+i, g = o/St, it = o%St;
                matmul_tiles(cb_rw, cb_Mb, g*St+it, g, i); }
            mm_init_short(cb_v, cb_Mk, 1);                   // + V @ Mk^T
            for (uint32_t i = 0; i < c4; i++) { uint32_t o = b+i, g = o/St, it = o%St;
                matmul_tiles(cb_v, cb_Mk, g*St+it, g, i); }
            tile_regs_commit(); tile_regs_wait();
            for (uint32_t i = 0; i < c4; i++) pack_tile(i, cb_romega, b+i);
            tile_regs_release();
        }
        cb_push_back(cb_romega, NT); cb_wait_front(cb_romega, NT);
        cb_pop_front(cb_rw, NT);                        // SA done
        cb_pop_front(cb_Mb, nb); cb_pop_front(cb_Mk, nb); cb_pop_front(cb_rhat, NT);

        // per instance final state and interleaved output emit
        // For each instance g in 0..nb-1, computes S_final and emits cb_out:
        //
        //   S_final = (SA@bsharp^T + V@ksharp^T)  +  S0 * omega_last
        //
        // where bsharp = btil * omega_last, ksharp = ktil * omega_last
        // (each an [S,L] flat-strip multiplied element-wise by the [S,1] omega_last column,
        // using hardware column-broadcast mul_bcast).
        //
        // The first term (SAb + VKsh) is accumulated into NS DST tiles (one acquire for
        // the whole [S,S] block per instance) using the B-transpose matmul trick:
        //   SAb  = SA  @ bsharp^T   (mm_init_short(SA, logw,  1) -- bsharp in cb_logw)
        //   VKsh = V   @ ksharp^T   (mm_init_short(v,  ahat,  1) -- ksharp in cb_ahat)
        // Both terms share the same DST; the result goes to cb_omega (NS tiles).
        //
        // The second term (term1 = S0 * omega_last per element) is computed via a single
        // hardware row-broadcast mul: omega_last is transposed once per chunk from the
        // column [S,1] to a row [1,S] (stored in cb_cumlog), then each instance does
        // mul_bcast_rows of S0 against that row -- no per-instance transpose_wh passes.
        //
        // Output emit to cb_out is UNIFORM (St+NS) slots every instance, even for non-last
        // CARRY chunks (host ignores intermediate S_final; on-chip carry is used instead).
        // Uniformity prevents the cb_out FIFO pointer from diverging between chunk iterations.
        //
        // S_final is packed twice when CARRY && !last: once to cb_out (placeholder for the
        // host/writer) and once to cb_carry for the next chunk's state seed.
        //
        // NOTE: NS=4 tiles exactly fills the fp32 DST (16 slots / 4 = 4 fp32 tiles or 8 bf16);
        // the SAb+VKsh accumulation above works within this limit.
        if (!last) cb_reserve_back(cb_carry, nb * NS);   // stash Sfinal for the next chunk
        // Pre-transpose omega_last from [S,1] to [1,S] once per chunk (into cb_cumlog,
        // which is free in this section).  The per-instance loop then uses a single
        // row-broadcast mul to compute S0[i,j]*omega_last[j] without any per-instance
        // transpose_wh passes on S0.
        ttggml::transpose(cb_cumlog, cb_omega_last, NT);
        for (uint32_t g = 0; g < nb; g++) {
            // bsharp = btil[g*St..] * omega_last[g*St..] -> cb_logw (St tiles, hardware col-broadcast)
            // ksharp = ktil[g*St..] * omega_last[g*St..] -> cb_ahat (St tiles, hardware col-broadcast)
            ttggml::mul_bcast(cb_logw, cb_btil, cb_omega_last, St, g*St);   // bsharp -> cb_logw (St)
            ttggml::mul_bcast(cb_ahat, cb_ktil, cb_omega_last, St, g*St);   // ksharp -> cb_ahat (St)
            // Accumulate SAb + VKsh into NS DST tiles -> cb_omega.
            // Tile index o=it*St+jt maps to state row-tile it and column-tile jt.
            cb_reserve_back(cb_omega, NS);
            tile_regs_acquire();
            mm_init_short(cb_SA, cb_logw, 1);                  // SAb = SA @ bsharp^T
            for (uint32_t o = 0; o < NS; o++) { uint32_t it = o/St, jt = o%St;
                matmul_tiles(cb_SA, cb_logw, g*St+it, jt, o); }
            mm_init_short(cb_v, cb_ahat, 1);                   // + VKsh = V @ ksharp^T
            for (uint32_t o = 0; o < NS; o++) { uint32_t it = o/St, jt = o%St;
                matmul_tiles(cb_v, cb_ahat, g*St+it, jt, o); }
            tile_regs_commit(); tile_regs_wait();
            for (uint32_t o = 0; o < NS; o++) pack_tile(o, cb_omega, o);
            tile_regs_release();
            cb_push_back(cb_omega, NS); cb_wait_front(cb_omega, NS);
            cb_pop_front(cb_logw, St); cb_pop_front(cb_ahat, St);
            // term1[i,j] = S0[i,j] * omega_last_row[j] via hardware row-broadcast.
            // omega_last_row (shape [1,S], one tile per instance) is in cb_cumlog at g*St+jt.
            // Output tile o=it*St+jt matches the S_final layout (row-tile it, col-tile jt).
            ttggml::gen_tiles(cb_rhat, NS, [&]{ mul_bcast_rows_init_short(cb_state, cb_cumlog); },
                [&](uint32_t d, uint32_t o){ uint32_t it = o/St, jt = o%St;
                    mul_tiles_bcast_rows(cb_state, cb_cumlog, g*NS+it*St+jt, g*St+jt, d); });
            // emit to cb_out
            // cb_out receives (St + NS) tiles per instance per chunk, always:
            //   tiles 0..St-1  : region-1 (per-token output), one [S,L] row-tile per tile-row
            //   tiles St..St+NS-1: region-2 (S_final state), NS [S,S] tiles
            // Uniform push size prevents cb_out pointer drift across CARRY chunk iterations.
            cb_reserve_back(cb_out, St + NS);
            // transpose Out from [dim,token] to [token,dim] before writing.
            // The writer drops tiles straight into ggml output rows (col=channel h*S+i, row=token).
            transpose_wh_init(cb_romega, cb_out);
            tile_regs_acquire();
            for (uint32_t it = 0; it < St; it++) transpose_wh_tile(cb_romega, g*St+it, it);
            tile_regs_commit(); tile_regs_wait();
            for (uint32_t it = 0; it < St; it++) pack_tile(it, cb_out, it);
            tile_regs_release();
            // CARRY: S_final = (SAb+VKsh) + term1  (elementwise add of cb_omega + cb_rhat).
            // Both are NS tiles in [S,S] layout.  The result is packed into cb_out (slots St..St+NS-1)
            // for the host writer (region-2).  Under CARRY && !last, the same DST is also packed into
            // cb_carry at offset g*NS so the next chunk can seed its state without DRAM.
            add_tiles_init(cb_omega, cb_rhat);
            tile_regs_acquire();
            for (uint32_t t = 0; t < NS; t++) add_tiles(cb_omega, cb_rhat, t, t, t);
            tile_regs_commit(); tile_regs_wait();
            for (uint32_t t = 0; t < NS; t++) pack_tile(t, cb_out, St + t);
            if (!last) { for (uint32_t t = 0; t < NS; t++) pack_tile(t, cb_carry, g * NS + t); }
            tile_regs_release();
            cb_push_back(cb_out, St + NS);
            cb_pop_front(cb_omega, NS); cb_pop_front(cb_rhat, NS);
        }
        if (!last) cb_push_back(cb_carry, nb * NS);   // carry now visible to the next chunk
        cb_pop_front(cb_cumlog, NT);   // omega_last_row consumed
        cb_pop_front(cb_romega, NT);
        cb_pop_front(cb_SA, NT); cb_pop_front(cb_omega_last, NT);
        cb_pop_front(cb_btil, NT); cb_pop_front(cb_ktil, NT);
        cb_pop_front(cb_v, NT); cb_pop_front(cb_state, nb * NS);
        }
    }
}
