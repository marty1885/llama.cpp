#include <stdint.h>
#include <cstdint>

void kernel_main() {
    uint32_t a_addr = get_arg_val<uint32_t>(0);
    uint32_t width_tiles = get_arg_val<uint32_t>(1);
    uint32_t height_tiles = get_arg_val<uint32_t>(2);
    uint32_t n_head = get_arg_val<uint32_t>(3);
    uint32_t batch = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;

    const uint32_t in0_tile_size_bytes = get_tile_size(cb_in0);
    constexpr auto a_args = TensorAccessorArgs<0>();
    const auto a = TensorAccessor(a_args, a_addr, in0_tile_size_bytes);

    for(uint32_t y = 0; y < height_tiles*n_head*batch; ++y) {
        for(uint32_t x = 0; x < width_tiles; ++x) {
            const uint32_t a_idx = y * width_tiles + x;
            cb_reserve_back(cb_in0, 1);
            uint32_t cb_src_addr = get_write_ptr(cb_in0);
            noc_async_read_tile(a_idx, a, cb_src_addr);
            noc_async_read_barrier();
            cb_push_back(cb_in0, 1);
        }

        for(uint32_t x = 0; x < width_tiles; ++x) {
            const uint32_t a_idx = y * width_tiles + x;
            cb_reserve_back(cb_in0, 1);
            uint32_t cb_src_addr = get_write_ptr(cb_in0);
            noc_async_read_tile(a_idx, a, cb_src_addr);
            noc_async_read_barrier();
            cb_push_back(cb_in0, 1);
        }
    }
}
