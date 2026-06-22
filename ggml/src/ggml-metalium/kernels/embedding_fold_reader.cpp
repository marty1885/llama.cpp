#include <cstdint>

void kernel_main() {
    uint32_t a_addr = get_arg_val<uint32_t>(0);
    uint32_t id     = get_arg_val<uint32_t>(1);
    uint32_t size   = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    const uint32_t tile_bytes = get_tile_size(cb_in0);
    constexpr auto a_args = TensorAccessorArgs<0>();
    const auto a = TensorAccessor(a_args, a_addr, tile_bytes);

    for (uint32_t ti = id; ti < id + size; ti++) {
        cb_reserve_back(cb_in0, 1);
        uint32_t w = get_write_ptr(cb_in0);
        noc_async_read_tile(ti, a, w);
        noc_async_read_barrier();
        cb_push_back(cb_in0, 1);
    }
}
