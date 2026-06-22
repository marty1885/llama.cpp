#include <cstdint>

// Scatter each token's bf16 band into the ggml-canonical output [1, 1, n_tokens, embed].
// cb_out holds the token's Tev band tiles; embed-group ec (the 32 values embed[ec*32:+32])
// lives at band tile (ec/32), row (ec%32), and must land at output tile (r/32, ec), row
// (r%32). Each element-row is two 32-byte face bursts:
//   face byte off = (face_row*2 + fc)*512 + sub_row*32   (32x32 bf16 tile, 16x16 faces).
void kernel_main() {
    uint32_t o_addr  = get_arg_val<uint32_t>(0);
    uint32_t embed_t = get_arg_val<uint32_t>(1);  // embed col-tiles = ceil(embed/32)
    uint32_t Tev     = get_arg_val<uint32_t>(2);  // band tiles per token = ceil(embed/1024)
    uint32_t id      = get_arg_val<uint32_t>(3);  // first token for this core
    uint32_t size    = get_arg_val<uint32_t>(4);  // tokens for this core

    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    const uint32_t tb = get_tile_size(cb_out);
    constexpr auto o_args = TensorAccessorArgs<0>();
    const auto o = TensorAccessor(o_args, o_addr, tb);

    for (uint32_t r = id; r < id + size; r++) {
        uint32_t tile_row = r / 32;
        uint32_t dst_row  = r % 32;
        uint32_t dfr = dst_row / 16;
        uint32_t drow = dst_row % 16;

        cb_wait_front(cb_out, Tev);
        uint32_t rp = get_read_ptr(cb_out);
        for (uint32_t ec = 0; ec < embed_t; ec++) {
            uint32_t src_tile = ec / 32;
            uint32_t src_row = ec % 32;
            uint32_t sfr = src_row / 16;
            uint32_t srow = src_row % 16;
            uint32_t src  = rp + src_tile * tb;
            uint32_t page = tile_row * embed_t + ec;
            for (uint32_t fc = 0; fc < 2; fc++) {
                noc_async_write(src + (sfr * 2 + fc) * 512 + srow * 32,
                                o.get_noc_addr(page, (dfr * 2 + fc) * 512 + drow * 32), 32);
            }
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, Tev);
    }
}
