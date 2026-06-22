#include <cstdint>

namespace { constexpr uint32_t cb_out = 16; }

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t inst_lo  = get_arg_val<uint32_t>(1);
    uint32_t inst_hi  = get_arg_val<uint32_t>(2);
    uint32_t IC       = get_arg_val<uint32_t>(3);
    uint32_t nc       = get_arg_val<uint32_t>(4);
    uint32_t H        = get_arg_val<uint32_t>(5);
    uint32_t St       = get_arg_val<uint32_t>(6);
    uint32_t Ct       = get_arg_val<uint32_t>(7);
    uint32_t T        = get_arg_val<uint32_t>(8);   // region-1 token rows (= L*G)
    uint32_t Lr       = get_arg_val<uint32_t>(9);   // real tokens/seq (L)
    uint32_t tpc      = get_arg_val<uint32_t>(10);  // tokens per emitted chunk (chunked=32, decode=1)
    uint32_t NB       = get_arg_val<uint32_t>(11);  // compute group size (chunked=2, decode P=1)

    // Derived constants.
    const uint32_t NS = St * St; // tiles in one [S, S] recurrent-state block
    const uint32_t S  = St * 32; // head size in elements (64 when St=2)
    const uint32_t tb = get_tile_size(cb_out); // bytes per 32x32 bf16 tile
    constexpr auto dst_args = TensorAccessorArgs<0>();
    const auto dst = TensorAccessor(dst_args, dst_addr, tb);

    // walk assigned instances in NB-wide groups, matching the compute drain cadence.
    for (uint32_t inst0 = inst_lo; inst0 < inst_hi; inst0 += NB) {
        // Clamp the last group to the remaining instance count (handles non-multiple ranges).
        uint32_t nb = (inst_hi - inst0) < NB ? (inst_hi - inst0) : NB;
        //iteration per compute "chunk" (32 tokens for chunked, 1 for decodeL).
        for (uint32_t cc = 0; cc < nc; cc++) {
            const bool last = (cc + 1 == nc); // true on the final chunk — triggers region-2 write
            // Instance loop within the group: each instance is one (sequence sq, head h) pair.
            for (uint32_t g = 0; g < nb; g++) {
                uint32_t inst = inst0 + g, sq = inst / H, h = inst % H;
                cb_wait_front(cb_out, St);
                uint32_t rp = get_read_ptr(cb_out);
                if (Lr % 32 == 0 && tpc == 32) {
                    // fast path: chunked kernel, and Lr is a multiple of 32.
                    // Each (sequence sq, chunk cc) exactly fills one tile row in region-1.
                    // Page of tile-column st:  (sq*nc + cc)*Ct + h*St + st.
                    // A single noc_async_write_page copies the entire 32x32 tile at once.
                    for (uint32_t st = 0; st < St; st++)
                        noc_async_write_page((sq * nc + cc) * Ct + h * St + st, dst, rp + st * tb);
                }
                else {
                    // slow path:  per-row scatter used when:
                    //   (a) decodeL: tpc=1, so each "chunk" is a single token, OR
                    //   (b) chunked but Lr % 32 != 0 (last chunk may be shorter).
                    //
                    // cl_real: actual number of valid tokens in this chunk (clamped for the
                    // last chunk when Lr is not a multiple of tpc).
                    uint32_t cl_real = (Lr - cc * tpc) < tpc ? (Lr - cc * tpc) : tpc;
                    for (uint32_t st = 0; st < St; st++) {
                        uint32_t src = rp + st * tb; // byte pointer to source tile st in cb_out
                        for (uint32_t lr = 0; lr < cl_real; lr++) {
                            // Global token index in the sequence, mapped to its ggml row.
                            uint32_t tg   = sq * Lr + cc * tpc + lr; // absolute token index
                            uint32_t page = (tg / 32) * Ct + h * St + st; // destination tile page
                            uint32_t rrow = tg % 32; // intra-tile row in destination

                            // Source face/row within the source tile (lr is the intra-chunk row).
                            uint32_t sfr  = lr / 16, srow = lr % 16;   // src face-row, sub-row
                            // Destination face/row within the destination tile page.
                            uint32_t dfr  = rrow / 16, drow = rrow % 16; // dst face-row, sub-row

                            // Write the two column-faces (fc=0: left 16 cols, fc=1: right 16 cols)
                            // as 32-byte bursts.  Face byte offset: (face_row*2 + fc)*512 + sub_row*32.
                            // NOTE: this writes exactly one element-row of 32 bf16 values (64 bytes
                            // split into two 32-byte halves) per fc iteration.
                            for (uint32_t fc = 0; fc < 2; fc++)
                                noc_async_write(src + (sfr * 2 + fc) * 512 + srow * 32,
                                                dst.get_noc_addr(page, (dfr * 2 + fc) * 512 + drow * 32), 32);
                        }
                    }
                }
                noc_async_write_barrier(); // flush all region-1 NoC writes before advancing cb_out
                cb_pop_front(cb_out, St);
                // write NS final-state tiles to the recurrent-state rows.
                // Compute always pushes NS tiles after the region-1 tiles regardless of
                // whether this is the last chunk. We always wait and pop to keep the
                // circular buffer in sync, but only issue NoC writes on the last chunk.
                cb_wait_front(cb_out, NS);
                uint32_t rp2 = get_read_ptr(cb_out);
                if (last) {
                    // Scatter all NS = St*St state tiles.  Tile index t encodes [it, jt]:
                    //   it = t / St  (row-tile index within the [S,S] state block)
                    //   jt = t % St  (col-tile index)
                    for (uint32_t t = 0; t < NS; t++) {
                        uint32_t it  = t / St, jt = t % St;
                        uint32_t src = rp2 + t * tb; // byte pointer to source state tile t
                        // Row-by-row scatter: each source row il (0..31) maps to a distinct
                        // ggml output row because the state is stored with heads interleaved.
                        for (uint32_t il = 0; il < 32; il++) {
                            uint32_t glob_i = it * 32 + il; // row index within head h's [S] state

                            // Map (sq, h, glob_i) to the ggml output row R in region-2.
                            // Region-2 starts at row T; each sequence occupies S rows.
                            // Within those S rows, head h's rows are interleaved: row
                            // (h*S + glob_i) is placed at offset (h*S + glob_i) / H.
                            // NOTE: this assumes H divides S evenly.
                            uint32_t R = T + S * sq + (h * S + glob_i) / H;

                            // Column base for tile [it, jt]: the j-dimension (second state axis)
                            // runs along the C axis; col_base selects the starting column element.
                            uint32_t col_base = ((h * S + glob_i) % H) * S + jt * 32;

                            // Tile page and intra-tile destination row.
                            uint32_t page = (R / 32) * Ct + col_base / 32;
                            uint32_t rrow = R % 32;

                            // Source face/row from the state tile.
                            uint32_t sfr = il / 16, srow = il % 16;
                            // Destination face/row within the destination page.
                            uint32_t dfr = rrow / 16, drow = rrow % 16;

                            // Write both column-faces of this element-row (same 32-byte burst
                            // pattern as region-1 scatter above).
                            for (uint32_t fc = 0; fc < 2; fc++) {
                                uint32_t soff = (sfr * 2 + fc) * 512 + srow * 32;
                                uint32_t doff = (dfr * 2 + fc) * 512 + drow * 32;
                                noc_async_write(src + soff, dst.get_noc_addr(page, doff), 32);
                            }
                        }
                    }
                    noc_async_write_barrier(); // flush region-2 writes before popping
                }
                cb_pop_front(cb_out, NS); // always pop to unblock compute, even if last=false
            }
        }
    }
}
