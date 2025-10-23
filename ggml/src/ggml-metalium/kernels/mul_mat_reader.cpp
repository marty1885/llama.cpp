#include <stdint.h>

void kernel_main() {
    uint32_t a_addr = get_arg_val<uint32_t>(0);
    uint32_t b_addr = get_arg_val<uint32_t>(1);


    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_in1 = tt::CBIndex::c_1;

    const uint32_t in0_tile_size_bytes = get_tile_size(cb_in0);
    constexpr auto a_args = TensorAccessorArgs<0>();
    const auto a = TensorAccessor(a_args, a_addr, in0_tile_size_bytes);

    const uint32_t in1_tile_size_bytes = get_tile_size(cb_in1);
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto b = TensorAccessor(b_args, b_addr, in1_tile_size_bytes);

    cb_reserve_back(cb_in0, 1);
    cb_reserve_back(cb_in1, 1);
    uint32_t cb_in0_addr = get_write_ptr(cb_in0);
    uint32_t cb_in1_addr = get_write_ptr(cb_in1);
    noc_async_read_tile(0, a, cb_in0_addr);
    noc_async_read_tile(0, b, cb_in1_addr);
    noc_async_read_barrier();
    cb_push_back(cb_in0, 1);
    cb_push_back(cb_in1, 1);
}
