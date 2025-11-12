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

    #ifdef HAS_MASK
    uint32_t mask_addr = get_arg_val<uint32_t>(5);
    uint32_t mask_n_head = get_arg_val<uint32_t>(6);
    uint32_t mask_batch = get_arg_val<uint32_t>(7);
    constexpr uint32_t cb_in1 = tt::CBIndex::c_1;
    const uint32_t mask_tile_size_bytes = get_tile_size(cb_in1);
    constexpr auto mask_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto mask = TensorAccessor(mask_args, mask_addr, mask_tile_size_bytes);
    const uint32_t mask_strides[3] = {width_tiles, height_tiles * width_tiles, mask_n_head * height_tiles * width_tiles};
    #endif

    const uint32_t a_strides[3] = {width_tiles, height_tiles * width_tiles, n_head * height_tiles * width_tiles};
    for(uint32_t b = 0; b < batch; ++b) {
        for(uint32_t h = 0; h < n_head; ++h) {
            for(uint32_t y = 0; y < height_tiles; ++y) {
                for(uint32_t x = 0; x < width_tiles; ++x) {
                    const uint32_t a_idx = x + y * a_strides[0] + h * a_strides[1] + b * a_strides[2];
                    cb_reserve_back(cb_in0, 1);
                    uint32_t cb_src_addr = get_write_ptr(cb_in0);
                    noc_async_read_tile(a_idx, a, cb_src_addr);
                    #ifdef HAS_MASK
                    uint32_t mask_b = b % mask_batch;
                    uint32_t mask_h = h % mask_n_head;
                    const uint32_t mask_idx = x + y * mask_strides[0] + mask_h * mask_strides[1] + mask_b * mask_strides[2];
                    uint32_t cb_in1_addr = get_write_ptr(cb_in1);
                    noc_async_read_tile(mask_idx, mask, cb_in1_addr);
                    #endif
                    noc_async_read_barrier();
                    cb_push_back(cb_in0, 1);
                    #ifdef HAS_MASK
                    cb_push_back(cb_in1, 1);
                    #endif
                }

                for(uint32_t x = 0; x < width_tiles; ++x) {
                    const uint32_t a_idx = x + y * a_strides[0] + h * a_strides[1] + b * a_strides[2];
                    cb_reserve_back(cb_in0, 1);
                    uint32_t cb_src_addr = get_write_ptr(cb_in0);
                    noc_async_read_tile(a_idx, a, cb_src_addr);
                    #ifdef HAS_MASK
                    uint32_t mask_b = b % mask_batch;
                    uint32_t mask_h = h % mask_n_head;
                    const uint32_t mask_idx = x + y * mask_strides[0] + mask_h * mask_strides[1] + mask_b * mask_strides[2];
                    uint32_t cb_in1_addr = get_write_ptr(cb_in1);
                    noc_async_read_tile(mask_idx, mask, cb_in1_addr);
                    #endif
                    noc_async_read_barrier();
                    cb_push_back(cb_in0, 1);
                    #ifdef HAS_MASK
                    cb_push_back(cb_in1, 1);
                    #endif
                }
            }
        }
    }
}
