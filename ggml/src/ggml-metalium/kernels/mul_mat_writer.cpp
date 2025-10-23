#include <cstdint>

void kernel_main() {
    uint32_t c_addr = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_out0 = tt::CBIndex::c_16;

    const uint32_t out0_tile_size_bytes = get_tile_size(cb_out0);
    constexpr auto c_args = TensorAccessorArgs<0>();
    const auto c = TensorAccessor(c_args, c_addr, out0_tile_size_bytes);

    cb_wait_front(cb_out0, 1);
    uint32_t out0_ptr = get_read_ptr(cb_out0);
    noc_async_write_tile(0, c, out0_ptr);
    noc_async_write_barrier();
    cb_pop_front(cb_out0, 1);
}
