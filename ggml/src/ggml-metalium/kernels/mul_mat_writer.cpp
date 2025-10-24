#include <cstdint>

void kernel_main() {
    uint32_t c_addr = get_arg_val<uint32_t>(0);
    uint32_t Mt =  get_arg_val<uint32_t>(1);
    uint32_t Nt =  get_arg_val<uint32_t>(2);
    uint32_t Kt =  get_arg_val<uint32_t>(3);
    uint32_t B = get_arg_val<uint32_t>(4);
    uint32_t C = get_arg_val<uint32_t>(5);
    uint32_t x = get_arg_val<uint32_t>(6);
    uint32_t y = get_arg_val<uint32_t>(7);
    uint32_t id = get_arg_val<uint32_t>(8);
    uint32_t size = get_arg_val<uint32_t>(10);

    constexpr uint32_t cb_out0 = tt::CBIndex::c_16;

    const uint32_t out0_tile_size_bytes = get_tile_size(cb_out0);
    constexpr auto c_args = TensorAccessorArgs<0>();
    const auto c = TensorAccessor(c_args, c_addr, out0_tile_size_bytes);
    for(uint32_t _b = 0; _b < B*x; _b++) {
        for(uint32_t _c = 0; _c < C*y; _c++) {
            uint32_t b_offset_plane = _b * C*y + _c;

            for(uint32_t m = 0; m < Mt; ++m) {
                for(uint32_t n = 0; n < Nt; ++n) {
                    uint32_t out_idx = n * Mt + m + b_offset_plane * Mt * Nt;
                    cb_wait_front(cb_out0, 1);
                    uint32_t out0_ptr = get_read_ptr(cb_out0);
                    noc_async_write_tile(out_idx, c, out0_ptr);
                    noc_async_write_barrier();
                    cb_pop_front(cb_out0, 1);
                }
            }

        }
    }
}
