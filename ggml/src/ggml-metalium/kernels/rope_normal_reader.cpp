#include <cstdint>

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t n_tiles_width_active = get_arg_val<uint32_t>(1);
    uint32_t n_tiles_width = get_arg_val<uint32_t>(2);
    uint32_t n_tiles_height = get_arg_val<uint32_t>(3);
    uint32_t idx_addr = get_arg_val<uint32_t>(4);
    uint32_t batch_size = get_arg_val<uint32_t>(5);
    uint32_t active_begin = get_arg_val<uint32_t>(6);
    uint32_t active_end = get_arg_val<uint32_t>(7);
    uint32_t passive_begin = get_arg_val<uint32_t>(8);
    uint32_t passive_end = get_arg_val<uint32_t>(9);
    uint32_t height_elements = get_arg_val<uint32_t>(10);
    #ifdef HAS_FREQ_FACTOR
    uint32_t freq_factor_addr = get_arg_val<uint32_t>(11);
    #endif


    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_in1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_bypass = tt::CBIndex::c_17;
    const uint32_t tile_size_bytes = get_tile_size(cb_in0);
    constexpr auto src_args = TensorAccessorArgs<0>();
    const auto src = TensorAccessor(src_args, src_addr, tile_size_bytes);

    constexpr auto idx_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    const auto idx = TensorAccessor(idx_args, idx_addr, batch_size*sizeof(int32_t));

    cb_reserve_back(cb_in1, 1);
    uint32_t cb_idx_addr = get_write_ptr(cb_in1);
    uint64_t read_addr = idx.get_noc_addr(0, 0);
    noc_async_read(read_addr, cb_idx_addr, batch_size*sizeof(int));
    noc_async_read_barrier();
    cb_push_back(cb_in1, 1);

    #ifdef HAS_FREQ_FACTOR
    constexpr uint32_t cb_in2 = tt::CBIndex::c_2;
    const uint32_t ff_tile_size_bytes = get_tile_size(cb_in2);
    constexpr auto ff_args = TensorAccessorArgs<idx_args.next_compile_time_args_offset()>();
    const auto ff = TensorAccessor(ff_args, freq_factor_addr, ff_tile_size_bytes);
    uint32_t last_ff_idx = -1;
    #endif

    for(uint32_t active_id=active_begin; active_id<active_end; active_id++) {
        uint32_t h = active_id / n_tiles_width_active;
        uint32_t w = active_id % n_tiles_width_active;

        cb_reserve_back(cb_in0, 1);
        #ifdef HAS_FREQ_FACTOR
            uint32_t ff_idx = w/2;
            bool read_ff = last_ff_idx != ff_idx;
            uint32_t cb_ff_addr = 0;
            if(read_ff) {
                cb_reserve_back(cb_in2, 1);
                cb_ff_addr = get_write_ptr(cb_in2);
                noc_async_read_tile(ff_idx, ff, cb_ff_addr);
                last_ff_idx = ff_idx;
            }
        #endif
        uint32_t cb_src_addr = get_write_ptr(cb_in0);
        uint32_t tile_idx =  h * n_tiles_width + w;
        noc_async_read_tile(tile_idx, src, cb_src_addr);
        noc_async_read_barrier();
        #ifdef HAS_FREQ_FACTOR
        if(read_ff) {
            unsigned short* ff_ptr = (unsigned short*)cb_ff_addr;
            for(int i=0;i<2;i++) {
                unsigned short buf[16];
                unsigned short* ptr = ff_ptr + i * 256;
                for(int j=0;j<16;j++) {
                    buf[j] = j%2 == 0 ? ptr[j/2] : ptr[j/2+8];
                }
                memcpy(ptr, buf, sizeof(buf));
            }
            cb_push_back(cb_in2, 1);
        }
        #endif
        cb_push_back(cb_in0, 1);
    }

    uint32_t n_tiles_width_passive = n_tiles_width - n_tiles_width_active;
    for(uint32_t passive_id = passive_begin; passive_id < passive_end; passive_id++) {
        uint32_t h = passive_id / n_tiles_width_passive;
        uint32_t w = passive_id % n_tiles_width_passive + n_tiles_width_active;
        uint32_t tile_idx = h * n_tiles_width + w;
        cb_reserve_back(cb_bypass, 1);
        uint32_t cb_bypass_addr = get_write_ptr(cb_bypass);
        noc_async_read_tile(tile_idx, src, cb_bypass_addr);
        noc_async_read_barrier();
        cb_push_back(cb_bypass, 1);
    }

}
