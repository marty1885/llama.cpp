#include <cstdint>

// Fold each canonical bf16 tile (cb_out) into the variant [1, vocab, embed/32, 32].
// Canonical tile ti = (vr, ec) holds 32 vocab rows x embed[ec*32 : ec*32+32]. Row k of
// that tile is vocab (vr*32 + k)'s embed-group ec, which lands at variant tile
// (vc*Tev + ec/32), intra-tile row (ec%32). Each element-row is written as two 32-byte
// face bursts (see tiles layout: byte off = (face_row*2 + fc)*512 + sub_row*32).
void kernel_main() {
    uint32_t v_addr  = get_arg_val<uint32_t>(0);
    uint32_t embed_t = get_arg_val<uint32_t>(1);
    uint32_t Tev     = get_arg_val<uint32_t>(2);
    uint32_t id      = get_arg_val<uint32_t>(3);
    uint32_t size    = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    const uint32_t tb = get_tile_size(cb_out);
    constexpr auto v_args = TensorAccessorArgs<0>();
    const auto v = TensorAccessor(v_args, v_addr, tb);

    for (uint32_t ti = id; ti < id + size; ti++) {
        uint32_t vr = ti / embed_t;
        uint32_t ec = ti % embed_t;
        uint32_t tr      = ec / 32;       // variant tile-row within the vocab band
        uint32_t dst_row = ec % 32;       // intra-tile destination row
        uint32_t dfr = dst_row / 16, drow = dst_row % 16;

        cb_wait_front(cb_out, 1);
        uint32_t rp = get_read_ptr(cb_out);
        for (uint32_t k = 0; k < 32; k++) {
            uint32_t vc   = vr * 32 + k;
            uint32_t page = vc * Tev + tr;
            uint32_t sfr  = k / 16, srow = k % 16;
            for (uint32_t fc = 0; fc < 2; fc++) {
                noc_async_write(rp + (sfr * 2 + fc) * 512 + srow * 32,
                                v.get_noc_addr(page, (dfr * 2 + fc) * 512 + drow * 32), 32);
            }
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
