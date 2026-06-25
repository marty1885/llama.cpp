#include <cstdint>

// WKV7 decodeL reader kernel
namespace { constexpr uint32_t c_natstage = 21, c_selstage = 22;
            constexpr uint32_t c_tri = 7, c_maskSL = 19, c_maskLI = 20, c_ident = 28; }

void kernel_main() {
    // 0  H          : number of heads per group
    // 1  L          : token count for this decode call (1..32)
    // 2  St         : tiles per head dimension = S/32 = 2
    // 3  Ht         : tile-rows per head in the input layout (H/32, rounded up)
    // 4  G          : number of groups (read but unused in this kernel -- (void)'d below)
    // 5..10 in_addr : per-input DRAM base addresses for the 6 input tensors
    //                 [a, w, k, v, r, b] in that order; all share input accessor 0
    // 11 st_addr    : DRAM base address of the flat-strip state buffer [Gpad, S*S*H]
    // 12 selh_addr  : (read but unused here) head-selection address; kept to preserve
    //                 arg index alignment with other kernels in this build
    // 13 sels_addr  : (read but unused here) sequence-selection address; same reason
    // 14 cst_addr   : (read but unused here) constants buffer address; same reason --
    //                 NOTE: this reader deliberately does NOT construct a 3rd TensorAccessor
    //                 for cst (see accessor section below), so cst_addr is just suppressed
    // 15 nc         : (read but unused here) tile-count constant; (void)'d below
    // 16 inst_start : first instance index this core handles (inclusive)
    // 17 inst_end   : last  instance index this core handles (exclusive)
    uint32_t H  = get_arg_val<uint32_t>(0);
    uint32_t L  = get_arg_val<uint32_t>(1);     // token count (1..32)
    uint32_t St = get_arg_val<uint32_t>(2);
    uint32_t Ht = get_arg_val<uint32_t>(3);
    uint32_t G  = get_arg_val<uint32_t>(4);
    uint32_t in_addr[6]; for (uint32_t i = 0; i < 6; i++) in_addr[i] = get_arg_val<uint32_t>(5 + i);
    uint32_t st_addr   = get_arg_val<uint32_t>(11);
    uint32_t selh_addr = get_arg_val<uint32_t>(12);
    uint32_t sels_addr = get_arg_val<uint32_t>(13);
    uint32_t cst_addr  = get_arg_val<uint32_t>(14);
    uint32_t nc         = get_arg_val<uint32_t>(15);
    uint32_t inst_start = get_arg_val<uint32_t>(16);
    uint32_t inst_end   = get_arg_val<uint32_t>(17);
    // selh_addr, sels_addr, G, nc are read above to keep arg-index alignment but are
    // not used by this reader.
    (void)selh_addr; (void)sels_addr; (void)G; (void)nc;

    const uint32_t tb  = get_tile_size(c_natstage);
    // ntok: total tiles of input data per instance (L tokens * Ht tile-rows * St col-tiles)
    const uint32_t ntok = L * Ht * St;
    // NS: number of [S,S] state tiles per head = St * St = 4
    const uint32_t NS  = St * St;
    // CONVENTION: ggml wkv7.cpp appends exactly two accessor arg-blocks into the
    // compile-time args: input[0] first, then state.  All 6 input tensors share
    // the single input accessor (they differ only in their runtime DRAM base
    // address in_addr[inp]).
    //
    // NOTE: DO NOT construct a 3rd accessor here:
    //   This build does NOT append a "sel" or "cst" TensorAccessorArgs block.
    //   If a TensorAccessorArgs<IN_NA+ST_NA> were constructed it would read
    //   compile-time args past the end of what was provided, causing the same
    //   out-of-range trap that broke an older constants accessor in this kernel.
    //   cst_addr (arg 14) is therefore suppressed via (void) below.
    constexpr uint32_t IN_NA = TensorAccessorArgs<0>::num_compile_time_args();
    constexpr auto in_args   = TensorAccessorArgs<0>();
    // State accessor starts immediately after the input accessor's CT arg block.
    constexpr auto st_args   = TensorAccessorArgs<IN_NA>();
    const auto st_acc   = TensorAccessor(st_args,  st_addr,   tb);
    // Suppress unused vars; constants (c_tri etc.) are SFPU-generated in compute,
    // not read from DRAM in this reader.
    (void)cst_addr; (void)c_tri; (void)c_maskSL; (void)c_maskLI; (void)c_ident;

    for (uint32_t inst = inst_start; inst < inst_end; inst++) {
        // sq: sequence index (which row of the batch), h: head index within group,
        // ht: tile-row index of head h (h/32), lh: intra-tile head row (h%32).
        uint32_t sq = inst / H, h = inst % H, ht = h / 32, lh = h % 32;

        // INPUT FACE OFFSETS for per-token reads (lh = intra-tile head row).
        // Each 32-element input row for head h spans two column-faces (32B each).
        //   soff0: byte offset to face (lh/16, 0) -- left  column-face
        //   soff1: byte offset to face (lh/16, 1) -- right column-face = soff0 + 512
        // NOTE: these are named soff0/soff1 (input offsets). Do NOT confuse with
        //       ssoff0/ssoff1 below, which serve the same role but for the state seed.
        const uint32_t soff0 = (lh / 16) * 1024 + (lh % 16) * 32;
        const uint32_t soff1 = soff0 + 512;

        // PHASE 1 -- S0 STATE SEED: sub-page gather from the flat-strip buffer
        // The state buffer has layout [Gpad, S*S*H] stored as a FLAT STRIP: one
        // long sequence of tiles, NOT arranged as per-head tile-blocks (STDIRECT).
        //
        // Head h's [S,S] state for sequence sq lives in source tile rows whose
        // intra-tile row is (sq%32).  The full [S,S] head state spans NS=4 source
        // tiles arranged as St rows x St cols of 32x32-element tile blocks.
        //
        // Flat-strip tile index math:
        //   tpr      = NS * 32 * H   -- tiles per full strip row (= S*S*H/32),
        //                               i.e., the stride to advance one intra-tile
        //                               row across all heads
        //   srow     = sq % 32       -- which intra-tile row within a 32-row block
        //                               carries this sequence's data
        //   src_page = (sq/32)*tpr + h*NS*32 + i*St + jt
        //              ^-- strip-block  ^-- head offset  ^-- tile (it-row i, col jt)
        //
        // STATE FACE OFFSETS (ssoff0/ssoff1): same formula as soff0/soff1 but for
        // srow instead of lh.  These extract row srow from each source tile.
        //   ssoff0 = byte offset to face (srow/16, 0)
        //   ssoff1 = ssoff0 + 512  (face (srow/16, 1), 512B later in the tile)
        //
        // DEST LAYOUT: NS=4 tiles written to c_natstage starting at wp.
        //   Tile (it, jt) at tilebase = wp + (it*St + jt)*tb.
        //   Within that dest tile, row r gets its 32 elements from two 32B reads
        //   into doff0 and doff1 = doff0+512 (dest face offsets for row r).
        //   Source page index for dest row r: i = it*32 + r.
        cb_reserve_back(c_natstage, NS);
        { uint32_t wp = get_write_ptr(c_natstage);
#ifdef WKV7_STATE_FOLDED
          // Row-folded state [1,G,Es/32,32]: head h's [S,S] is dense (no 32x flat-strip pad). The fold
          // is a logical reshape; physical bytes stay face-tiled. For dest tile (it,jt) row r, the source
          // folded logical row is h*(NS*32) + it*(St*32) + 2r + jt (the fold's 2-row interleave: state
          // row i -> folded rows 2i, 2i+1); read its full 32 cols (both col-faces). TPS = NS*H = tiles
          // per sequence in the folded buffer.
          const uint32_t TPS = NS * H;
          for (uint32_t it = 0; it < St; it++) for (uint32_t jt = 0; jt < St; jt++) {
              uint32_t tilebase = wp + (it * St + jt) * tb;
              for (uint32_t r = 0; r < 32; r++) {
                  uint32_t flr      = h * (NS * 32) + it * (St * 32) + 2 * r + jt;  // folded logical row
                  uint32_t src_page = sq * TPS + flr / 32;
                  uint32_t sr       = flr % 32;
                  uint32_t ssoff0   = (sr / 16) * 1024 + (sr % 16) * 32;
                  uint32_t ssoff1   = ssoff0 + 512;
                  uint32_t doff0    = (r / 16) * 1024 + (r % 16) * 32;
                  uint32_t doff1    = doff0 + 512;
                  noc_async_read(st_acc.get_noc_addr(src_page, ssoff0), tilebase + doff0, 32);
                  noc_async_read(st_acc.get_noc_addr(src_page, ssoff1), tilebase + doff1, 32);
              }
          }
#else
          // Canonical flat-strip state [Gpad, S*S*H]: head h's [S,S] scattered across source tiles,
          // only intra-tile row srow=sq%32 valid (32x pad waste).
          const uint32_t tpr   = NS * 32 * H;            // tiles per flat-strip row (= S*S*H/32)
          const uint32_t srow  = sq % 32;                // source intra-tile row
          const uint32_t ssoff0 = (srow / 16) * 1024 + (srow % 16) * 32;  // src face (srow/16,0)
          const uint32_t ssoff1 = ssoff0 + 512;                           // src face (srow/16,1)
          for (uint32_t it = 0; it < St; it++) for (uint32_t jt = 0; jt < St; jt++) {
              uint32_t tilebase = wp + (it * St + jt) * tb;
              for (uint32_t r = 0; r < 32; r++) {
                  uint32_t i = it * 32 + r;
                  uint32_t src_page = (sq / 32) * tpr + h * (NS * 32) + i * St + jt;
                  uint32_t doff0 = (r / 16) * 1024 + (r % 16) * 32;
                  uint32_t doff1 = doff0 + 512;
                  noc_async_read(st_acc.get_noc_addr(src_page, ssoff0), tilebase + doff0, 32);
                  noc_async_read(st_acc.get_noc_addr(src_page, ssoff1), tilebase + doff1, 32);
              }
          }
#endif
          noc_async_read_barrier(); }
        cb_push_back(c_natstage, NS);

        // PHASE 2: PER-TOKEN INPUTS: stream [a, w, k, v, r, b] for t=0..L-1
        // For each token t, read all 6 inputs in order (inp=0..5).  Each input
        // contributes St=2 tiles placed at dest row 0 in c_natstage, so the
        // compute kernel sees every token's data in the same tile-row position and
        // can reuse its single-token extract path unchanged across all L tokens.
        //
        // src_page = (sq * L + t) * Ht * St + ht * St + st
        //             ^-- sequence/token stride  ^-- head tile offset  ^-- col tile
        // soff0/soff1: same input face offsets computed above from lh (head row).
        //   dest face 0 is written at tilebase + 0   (row 0, left  face)
        //   dest face 1 is written at tilebase + 512 (row 0, right face)
        // NOTE: the dest offsets here are literal 0 and 512 (row 0), which is
        //       exactly what doff0/doff1 would equal for r=0: (0/16)*1024+(0%16)*32=0.
        for (uint32_t t = 0; t < L; t++) {
            // base: starting tile page for token t, head h's tile-row within the input tensor
            uint32_t base = (sq * L + t) * Ht * St + ht * St;
#ifdef WKV7_INPUT_FLAT
            // Flat inputs r/w/k/v (inp 1..4) are read from the original embedding-style
            // layout instead of the standard per-token/per-head layout used by a and b.
            //
            // For global token index tg = sq * L + t:
            //   - ftr = tg % 32 selects the source row within the tile
            //   - tg / 32 selects the source row-tile block
            //
            // For head h, this head's S-wide slice occupies column tiles
            // [h * St, h * St + St), and each row-tile block contains H * St tiles
            // total (= n_embd / 32).
            //
            // Inputs a and b (0 and 5) are not flat and continue to use the normal
            // [S, H, T] path below.
            const uint32_t tg    = sq * L + t;
            const uint32_t ftr   = tg % 32;
            const uint32_t ffso0 = (ftr / 16) * 1024 + (ftr % 16) * 32;
            const uint32_t ffso1 = ffso0 + 512;
            const uint32_t fbase = (tg / 32) * (H * St) + h * St;
#endif
            for (uint32_t inp = 0; inp < 6; inp++) {
                // Construct input accessor freshly per input: each of the 6 inputs has a
                // different DRAM base address (in_addr[inp]) but shares the same CT args.
                const auto in_acc = TensorAccessor(in_args, in_addr[inp], tb);
                cb_reserve_back(c_natstage, St);
                uint32_t wp = get_write_ptr(c_natstage);
                for (uint32_t st = 0; st < St; st++) {
                    uint32_t tilebase = wp + st * tb;
#ifdef WKV7_INPUT_FLAT
                    if (inp >= 1 && inp <= 4) {
                        // Flat input: token-row source, head's col-tile page; dest row 0.
                        uint32_t src_page = fbase + st;
                        noc_async_read(in_acc.get_noc_addr(src_page, ffso0), tilebase + 0, 32);
                        noc_async_read(in_acc.get_noc_addr(src_page, ffso1), tilebase + 512, 32);
                        continue;
                    }
#endif
                    // src_page: tile page for col-tile st of this input/token/head
                    uint32_t src_page = base + st;
                    // Read 32 bytes from left column-face (face (lh/16,0)) into dest row 0
                    noc_async_read(in_acc.get_noc_addr(src_page, soff0), tilebase + 0, 32);
                    // Read 32 bytes from right column-face (face (lh/16,1)) into dest row 0
                    noc_async_read(in_acc.get_noc_addr(src_page, soff1), tilebase + 512, 32);
                }
                noc_async_read_barrier();
                cb_push_back(c_natstage, St);
            }
        }
    }
    // c_selstage is reserved for a selection-stage CB used by other kernel variants;
    // suppressed here to avoid an unused-variable warning.
    (void)c_selstage;
}
