#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"

// Casting, shared by both folding and gather kernel
void kernel_main() {
    uint32_t size = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    compute_kernel_hw_startup(cb_in0, cb_out);
    copy_tile_init(cb_in0);

    for (uint32_t i = 0; i < size; i++) {
        cb_wait_front(cb_in0, 1);
        tile_regs_acquire();
        copy_tile(cb_in0, 0, 0);
        tile_regs_commit();

        tile_regs_wait();
        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);
        tile_regs_release();

        cb_pop_front(cb_in0, 1);
    }
}
