#include <stdint.h>
#include <cstdint>

void kernel_main() {
    uint32_t a_addr = get_arg_val<uint32_t>(0);
    uint32_t b_addr = get_arg_val<uint32_t>(1);
    uint32_t Mt = get_arg_val<uint32_t>(2);
    uint32_t Nt = get_arg_val<uint32_t>(3);
    uint32_t Kt = get_arg_val<uint32_t>(4);
    uint32_t B = get_arg_val<uint32_t>(5);
    uint32_t C = get_arg_val<uint32_t>(6);
    uint32_t x = get_arg_val<uint32_t>(7);
    uint32_t y = get_arg_val<uint32_t>(8);
    uint32_t id = get_arg_val<uint32_t>(9);
    uint32_t size = get_arg_val<uint32_t>(10);


    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_in1 = tt::CBIndex::c_1;

    const uint32_t in0_tile_size_bytes = get_tile_size(cb_in0);
    constexpr auto a_args = TensorAccessorArgs<0>();
    const auto a = TensorAccessor(a_args, a_addr, in0_tile_size_bytes);

    const uint32_t in1_tile_size_bytes = get_tile_size(cb_in1);
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto b = TensorAccessor(b_args, b_addr, in1_tile_size_bytes);

    for(uint32_t _b = 0; _b < B*x; _b++) {
        uint32_t ab = _b / x;
        uint32_t bb = _b;
        for(uint32_t c = 0; c < C*y; c++) {
            uint32_t ac = c / y;
            uint32_t bc = c;
            uint32_t a_offset_plane = ab * C + ac;
            uint32_t b_offset_plane = bb * C*y + bc;


            for(uint32_t m = 0; m < Mt; ++m) {
                for(uint32_t n = 0; n < Nt; ++n) {
                    for(uint32_t k = 0; k < Kt; ++k) {
                        uint32_t a_idx = m * Kt + k + a_offset_plane * Kt * Mt;
                        uint32_t b_idx = n * Kt + k + b_offset_plane * Kt * Nt;

                        cb_reserve_back(cb_in0, 1);
                        cb_reserve_back(cb_in1, 1);
                        uint32_t cb_in0_addr = get_write_ptr(cb_in0);
                        uint32_t cb_in1_addr = get_write_ptr(cb_in1);
                        noc_async_read_tile(a_idx, a, cb_in0_addr);
                        noc_async_read_tile(b_idx, b, cb_in1_addr);
                        noc_async_read_barrier();
                        cb_push_back(cb_in0, 1);
                        cb_push_back(cb_in1, 1);
                    }
                }
            }

        }
    }
}
