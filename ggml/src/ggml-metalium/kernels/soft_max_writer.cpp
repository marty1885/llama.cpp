#include <stdint.h>
#include <cstdint>

void kernel_main() {
    uint32_t b_addr = get_arg_val<uint32_t>(0);
    uint32_t width_tiles = get_arg_val<uint32_t>(1);
    uint32_t height_tiles = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out0 = tt::CBIndex::c_16;

    const uint32_t out0_tile_size_bytes = get_tile_size(cb_out0);
    constexpr auto b_args = TensorAccessorArgs<0>();
    const auto b = TensorAccessor(b_args, b_addr, out0_tile_size_bytes);

    for(uint32_t y = 0; y < height_tiles; ++y) {
        for(uint32_t x = 0; x < width_tiles; ++x) {
            const uint32_t out_idx = y * width_tiles + x;
            cb_wait_front(cb_out0, 1);
            uint32_t cb_out_addr = get_read_ptr(cb_out0);
            noc_async_write_tile(out_idx, b, cb_out_addr);
            noc_async_write_barrier();
            cb_pop_front(cb_out0, 1);
        }
    }

}
