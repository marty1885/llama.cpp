#include <cstdint>
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"

using std::uint32_t;

namespace NAMESPACE {
void MAIN {
    uint32_t Mt =  get_arg_val<uint32_t>(0);
    uint32_t Nt =  get_arg_val<uint32_t>(1);
    uint32_t Kt =  get_arg_val<uint32_t>(2);
    uint32_t B = get_arg_val<uint32_t>(3);
    uint32_t C = get_arg_val<uint32_t>(4);
    uint32_t x = get_arg_val<uint32_t>(5);
    uint32_t y = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_in1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_out0 = tt::CBIndex::c_16;

    mm_init(cb_in1, cb_in0, cb_out0, true);
    for(uint32_t b = 0; b < B*x; b++) {
        for(uint32_t c = 0; c < C*y; c++) {
            for(uint32_t m = 0; m < Mt; ++m) {
                for(uint32_t n = 0; n < Nt; ++n) {
                    tile_regs_acquire();
                    for(uint32_t k = 0; k < Kt; ++k) {
                        cb_wait_front(cb_in0, 1);
                        cb_wait_front(cb_in1, 1);
                        matmul_tiles(cb_in1, cb_in0, 0, 0, 0, true);
                        cb_pop_front(cb_in0, 1);
                        cb_pop_front(cb_in1, 1);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    cb_reserve_back(cb_out0, 1);
                    pack_tile(0, cb_out0);
                    cb_push_back(cb_out0, 1);
                    tile_regs_release();
                }
            }
        }
    }

}
}  // namespace NAMESPACE
