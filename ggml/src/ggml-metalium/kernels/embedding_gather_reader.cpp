#include <cstdint>

// Gather the variant band for each token, indexed by the real index tensor.
// Index is ROW_MAJOR uint32, TT shape [1, 1, users, n]: page u is one user's stick of n
// ids. Token r flattens as r = u*n + i. We cache the current user's page in an L1 scratch
// CB (c_1) and reuse it across that user's tokens. Vocab v's band = variant tiles
// [v*Tev, (v+1)*Tev).
void kernel_main() {
    uint32_t v_addr   = get_arg_val<uint32_t>(0);
    uint32_t idx_addr = get_arg_val<uint32_t>(1);
    uint32_t Tev      = get_arg_val<uint32_t>(2);
    uint32_t n        = get_arg_val<uint32_t>(3);  // ids per user (last dim)
    uint32_t id       = get_arg_val<uint32_t>(4);  // first token for this core
    uint32_t size     = get_arg_val<uint32_t>(5);  // tokens for this core

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_idx = tt::CBIndex::c_1;

    const uint32_t vtile = get_tile_size(cb_in0);
    constexpr auto v_args = TensorAccessorArgs<0>();
    const auto v = TensorAccessor(v_args, v_addr, vtile);

    const uint32_t idx_page_bytes = n * 4;
    constexpr auto idx_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    const auto idx = TensorAccessor(idx_args, idx_addr, idx_page_bytes);

    // c_1 is a reusable L1 scratch for one index page (reserve once, overwrite in place).
    cb_reserve_back(cb_idx, 1);
    uint32_t idx_l1 = get_write_ptr(cb_idx);
    volatile tt_l1_ptr uint32_t* ids = (volatile tt_l1_ptr uint32_t*)idx_l1;

    uint32_t loaded_user = 0xffffffff;
    for (uint32_t r = id; r < id + size; r++) {
        uint32_t u = r / n;
        uint32_t i = r % n;
        if (u != loaded_user) {
            noc_async_read_page(u, idx, idx_l1);
            noc_async_read_barrier();
            loaded_user = u;
        }
        uint32_t vocab = ids[i];
        for (uint32_t t = 0; t < Tev; t++) {
            cb_reserve_back(cb_in0, 1);
            uint32_t w = get_write_ptr(cb_in0);
            noc_async_read_tile(vocab * Tev + t, v, w);
            noc_async_read_barrier();
            cb_push_back(cb_in0, 1);
        }
    }
}
