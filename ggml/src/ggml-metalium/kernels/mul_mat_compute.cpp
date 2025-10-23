#include <cstdint>
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"

using std::uint32_t;

namespace NAMESPACE {
void MAIN {

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_in1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_out0 = tt::CBIndex::c_16;

    mm_init(cb_in0, cb_in1, cb_out0, true);
    tile_regs_acquire();
    cb_wait_front(cb_in0, 1);
    cb_wait_front(cb_in1, 1);
    matmul_tiles(cb_in1, cb_in0, 0, 0, 0, true);
    cb_pop_front(cb_in0, 1);
    cb_pop_front(cb_in1, 1);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_out0, 1);
    pack_tile(0, cb_out0);
    cb_push_back(cb_out0, 1);
    tile_regs_release();
}
}  // namespace NAMESPACE
