#include <cstdint>

namespace { constexpr uint32_t c_natstage = 21, c_selstage = 22;
            constexpr uint32_t c_tri = 7, c_maskSL = 19, c_maskLI = 20, c_ident = 28; }

void kernel_main() {
    // Runtime args (get_arg_val indices 0..17)
    //
    //  [0]  H          - number of heads per sequence
    //  [1]  L          - chunk length cl (always 32)
    //  [2]  St         - tile columns per head dim (S/32 = 2)
    //  [3]  Ht         - tile columns per sequence dim (H/32)
    //  [4]  G          - number of sequences in this batch
    //  [5..10] in_addr[0..5] - DRAM base page addresses for the 6 input tensors
    //                           (r, w, k, v, a, b); shared accessor, address swapped per tensor
    //  [11] st_addr    - DRAM base page address for the flat-strip state buffer
    //  [12] Lreal      - real tokens per sequence; may be < nc*cl when the last
    //                    chunk is partial; reader pads the tail with neutral tokens on-device
    //  [13] sels_addr  - sequence-selector address (unused in SUBPAGE path; silenced)
    //  [14] cst_addr   - consts address (unused; consts are SFPU-generated in compute)
    //  [15] nc         - number of chunks per sequence (= ceil(Lreal / cl))
    //  [16] inst_start - first instance index assigned to this core
    //  [17] inst_end   - one-past-last instance index assigned to this core
    uint32_t H  = get_arg_val<uint32_t>(0);
    uint32_t L  = get_arg_val<uint32_t>(1);     // = cl (on-device chunk size, always 32)
    uint32_t St = get_arg_val<uint32_t>(2);
    uint32_t Ht = get_arg_val<uint32_t>(3);
    uint32_t G  = get_arg_val<uint32_t>(4);
    uint32_t in_addr[6]; for (uint32_t i = 0; i < 6; i++) in_addr[i] = get_arg_val<uint32_t>(5 + i);
    uint32_t st_addr   = get_arg_val<uint32_t>(11);
    uint32_t Lreal     = get_arg_val<uint32_t>(12);   // real tokens/seq (>= cl; partial-chunk tail padded neutral)
    uint32_t sels_addr = get_arg_val<uint32_t>(13);
    uint32_t cst_addr  = get_arg_val<uint32_t>(14);
    uint32_t nc         = get_arg_val<uint32_t>(15);
    uint32_t inst_start = get_arg_val<uint32_t>(16);
    uint32_t inst_end   = get_arg_val<uint32_t>(17);

    const uint32_t tb  = get_tile_size(c_natstage);    // 2048 bytes per 32x32 bf16 tile
    const uint32_t NS  = St * St;                      // tiles per [S,S] head state = 4
    const uint32_t nst = St * St * 32;                 // elements per head state row-block

    constexpr uint32_t IN_NA = TensorAccessorArgs<0>::num_compile_time_args();
    constexpr auto in_args   = TensorAccessorArgs<0>();
    constexpr auto st_args   = TensorAccessorArgs<IN_NA>();
    const auto st_acc   = TensorAccessor(st_args,  st_addr,   tb);
    // ggml tensor_args = {inputs, state} only: head/seq selectors unused under SUBPAGE,
    // consts SFPU-generated in compute. So just two accessors: in (idx 0) + state.
    (void)sels_addr; (void)cst_addr;
    (void)c_tri; (void)c_maskSL; (void)c_maskLI; (void)c_ident;

    // Const tiles (tri/maskSL/maskLI/ident) are no longer streamed here — the compute
    // kernel generates them on-device via SFPU (allocation-free / Metal-trace friendly).

    // dest face offsets within a [token x dim] tile for token t (row t, both col-faces):
    //   doff0 = col-face 0 (dims 0..15), doff1 = col-face 1 (dims 16..31)
    // GROUP-CARRY: outer NB-instance groups, inner chunks (matches compute consumer).
    constexpr uint32_t NBr = 2;
    for (uint32_t g0 = inst_start; g0 < inst_end; g0 += NBr) {
        uint32_t nb = (inst_end - g0) < NBr ? (inst_end - g0) : NBr;
        for (uint32_t c = 0; c < nc; c++) {
            for (uint32_t gi = 0; gi < nb; gi++) {
                uint32_t inst = g0 + gi, sq = inst / H, h = inst % H;
                // lh = intra-tile row for head h; ht = which 32-head tile row h lives in.
                uint32_t ht = h / 32, lh = h % 32;
                // Source col-face byte offsets within each input source tile for row lh.
                // soff0: col-face 0 (dims 0..15 of this St block); soff1: col-face 1 (+512 B).
                // Two 32-byte reads cover all 32 dims of head h's row in one source tile.
                const uint32_t soff0 = (lh / 16) * 1024 + (lh % 16) * 32;   // src face (lh/16, 0)
                const uint32_t soff1 = soff0 + 512;                         // src face (lh/16, 1)

                uint32_t cl_real = Lreal - c * L;            // L == cl (chunk size); c*cl < Lreal always
                if (cl_real > L) cl_real = L;

                // Read input
                // we are abusing the tile internal layour to select the head's 32-byte row slice via byte offsets
                // to reduce compute overhead
                // source face offset for head h (computed once per instance, reused across
                // all tokens and tensors)
                //
                //   lh   = h % 32  - intra-tile row index for head h inside a 32-head source tile
                //   ht   = h / 32  - which 32-head tile row this head lives in
                //
                //   Using the tile byte-offset formula: byte([r,c]) = ((r/16)*2 + c/16)*512 + (r%16)*32 + (c%16)*2
                //   For row lh, col-face 0 (cols 0..15):
                //     soff0 = (lh/16)*1024 + (lh%16)*32   [face row lh/16, face col 0, intra-face row lh%16, col 0]
                //   For row lh, col-face 1 (cols 16..31):
                //     soff1 = soff0 + 512                  [same face row, next face column = +512 bytes]
                //   Each read is 32 bytes = 16 bf16 elements covering one col-face of head h's row.
                for (uint32_t inp = 0; inp < 6; inp++) {
                    const auto in_acc = TensorAccessor(in_args, in_addr[inp], tb);
                    cb_reserve_back(c_natstage, St);
                    uint32_t wp = get_write_ptr(c_natstage);
                    const uint16_t neutral = (inp == 1) ? 0x3F80 : 0x0000;   // w -> 1.0, else 0.0
#ifdef WKV7_INPUT_FLAT
                    // Under WKV7_INPUT_FLAT, inputs 1..4 use the flat embedding layout [T, n_embd]
                    // (equivalently [n_embd, T] in logical terms) instead of the reshaped [S, H, T]
                    // path used below. Inputs 0 and 5 (a, b) stay on the reshaped l2-norm-derived path.
                    //
                    // For head h, its S-dimensional slice occupies column tiles [h*St, h*St + St).
                    // Token tg selects the source row: tr = tg % 32 within the tile, and tg / 32 selects
                    // the row-tile page. Each row-tile page has H*St column tiles total (= n_embd / 32).
                    // Destination packing is unchanged: token t is written into row t of the output tile.
                    if (inp >= 1 && inp <= 4) {
                        const uint32_t tg0 = sq * Lreal + c * L;   // global token base for this chunk
                        const uint32_t cpr = H * St;               // col-tiles per row-tile (= n_embd/32)
                        for (uint32_t st = 0; st < St; st++) {
                            uint32_t tilebase = wp + st * tb;
                            for (uint32_t t = 0; t < L; t++) {
                                uint32_t doff0 = (t / 16) * 1024 + (t % 16) * 32;
                                uint32_t doff1 = doff0 + 512;
                                if (t < cl_real) {
                                    uint32_t tg   = tg0 + t;
                                    uint32_t tr   = tg % 32;
                                    uint32_t fso0 = (tr / 16) * 1024 + (tr % 16) * 32;
                                    uint32_t fso1 = fso0 + 512;
                                    uint32_t src_page = (tg / 32) * cpr + h * St + st;
                                    noc_async_read(in_acc.get_noc_addr(src_page, fso0), tilebase + doff0, 32);
                                    noc_async_read(in_acc.get_noc_addr(src_page, fso1), tilebase + doff1, 32);
                                } else {
                                    volatile tt_l1_ptr uint16_t* p0 = (volatile tt_l1_ptr uint16_t*)(tilebase + doff0);
                                    volatile tt_l1_ptr uint16_t* p1 = (volatile tt_l1_ptr uint16_t*)(tilebase + doff1);
                                    for (uint32_t e = 0; e < 16; e++) { p0[e] = neutral; p1[e] = neutral; }
                                }
                            }
                        }
                        noc_async_read_barrier();
                        cb_push_back(c_natstage, St);
                        continue;
                    }
#endif
                    uint32_t base = (sq * Lreal + c * L) * (Ht * St) + ht * St;
                    for (uint32_t st = 0; st < St; st++) {
                        uint32_t tilebase = wp + st * tb;
                        for (uint32_t t = 0; t < L; t++) {
                            uint32_t doff0 = (t / 16) * 1024 + (t % 16) * 32;
                            uint32_t doff1 = doff0 + 512;
                            if (t < cl_real) {
                                uint32_t src_page = base + t * (Ht * St) + st;
                                noc_async_read(in_acc.get_noc_addr(src_page, soff0), tilebase + doff0, 32);
                                noc_async_read(in_acc.get_noc_addr(src_page, soff1), tilebase + doff1, 32);
                            } else {
                                volatile tt_l1_ptr uint16_t* p0 = (volatile tt_l1_ptr uint16_t*)(tilebase + doff0);
                                volatile tt_l1_ptr uint16_t* p1 = (volatile tt_l1_ptr uint16_t*)(tilebase + doff1);
                                for (uint32_t e = 0; e < 16; e++) { p0[e] = neutral; p1[e] = neutral; }
                            }
                        }
                    }
                    noc_async_read_barrier();
                    cb_push_back(c_natstage, St);
                }

                if (c == 0) {
                    cb_reserve_back(c_natstage, NS);
                    uint32_t wp = get_write_ptr(c_natstage);
#ifdef WKV7_STATE_FOLDED
                    // Row-folded state [1,G,Es/32,32]: head h's [S,S] is dense (no 32x flat-strip pad).
                    // The fold is a logical reshape; physical bytes stay face-tiled. For dest tile (it,jt)
                    // row r, the source folded logical row is h*(NS*32) + it*(St*32) + 2r + jt (the fold's
                    // 2-row interleave: state row i -> folded rows 2i, 2i+1). Read its full 32 cols (both
                    // col-faces). TPS = NS*H = tiles per sequence in the folded buffer.
                    const uint32_t TPS = NS * H;
                    for (uint32_t it = 0; it < St; it++) for (uint32_t jt = 0; jt < St; jt++) {
                        uint32_t tilebase = wp + (it * St + jt) * tb;
                        for (uint32_t r = 0; r < 32; r++) {
                            uint32_t flr      = h * (NS * 32) + it * (St * 32) + 2 * r + jt;
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
                    // Canonical flat-strip state [Gpad, S*S*H]: head h's [S,S] scattered across
                    // h*(NS*32)+i*St+jt tiles, only row srow=sq%32 valid per tile (32x pad waste).
                    const uint32_t tpr  = NS * 32 * H;          // tiles per flat-strip row (= S*S*H/32)
                    const uint32_t srow = sq % 32;              // source intra-tile row
                    const uint32_t soff0 = (srow / 16) * 1024 + (srow % 16) * 32;  // src face (srow/16,0)
                    const uint32_t soff1 = soff0 + 512;                            // src face (srow/16,1)
                    for (uint32_t it = 0; it < St; it++) for (uint32_t jt = 0; jt < St; jt++) {
                        uint32_t tilebase = wp + (it * St + jt) * tb;
                        for (uint32_t r = 0; r < 32; r++) {
                            uint32_t i = it * 32 + r;
                            uint32_t src_page = (sq / 32) * tpr + h * (NS * 32) + i * St + jt;
                            uint32_t doff0 = (r / 16) * 1024 + (r % 16) * 32;
                            uint32_t doff1 = doff0 + 512;
                            noc_async_read(st_acc.get_noc_addr(src_page, soff0), tilebase + doff0, 32);
                            noc_async_read(st_acc.get_noc_addr(src_page, soff1), tilebase + doff1, 32);
                        }
                    }
#endif
                    noc_async_read_barrier();
                    cb_push_back(c_natstage, NS);
                }
            }
        }
    }
    (void)NS; (void)nst;
}
