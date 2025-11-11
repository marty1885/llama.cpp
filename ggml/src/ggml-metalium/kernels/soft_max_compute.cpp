#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/fill.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/eltwise_unary/recip.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"

#include <debug/dprint_tensix.h>


using std::uint32_t;

#ifdef TRISC_MATH
using namespace sfpi;
using namespace ckernel::sfpu;

void update_online_softmax_values_internal(const uint32_t dst_index_in0, const uint32_t dst_index_in1, const uint32_t dst_index_out) {
    constexpr uint32_t n_vector_in_tile = 32;

    // Calculate base indices for each tile in the Dst register array.
    // Each tile occupies 32 consecutive Dst registers (n_vector_in_tile) in WH and BH
    // For example: tile 0 uses dst_reg[0-31], tile 1 uses dst_reg[32-63], etc.
    const uint32_t in_base_idx = dst_index_in0 * n_vector_in_tile;
    const uint32_t sum_base_idx = dst_index_in1 * n_vector_in_tile;
    const uint32_t max_base_idx = dst_index_out * n_vector_in_tile;

    // Process one face of the tile (8 SIMD operations covering 256 elements).
    // Each iteration processes 32 elements, so 8 iterations = 256 elements = one 16x16 face.
    for (size_t i = 0; i < 8; i++) {
        vFloat x = dst_reg[in_base_idx];
        vFloat sum = dst_reg[sum_base_idx];
        vFloat max = dst_reg[max_base_idx];

        vFloat new_max = max;
        v_if(x > max) {
            new_max = x;
        } v_endif;

        sum = sum * _sfpu_exp_21f_<true>(max - new_max) + _sfpu_exp_21f_<true>(x - new_max);

        dst_reg[sum_base_idx] = sum;
        dst_reg[max_base_idx] = new_max;
        dst_reg++;
    }
}


#endif

static void update_online_softmax_values() {
    // parameters not used - we alwasys put input on tile 0, sum on 1 and max on 2
    MATH(_llk_math_eltwise_binary_sfpu_params_<false>(update_online_softmax_values_internal, 0, 1, 2));
}


namespace NAMESPACE {
void MAIN {
    uint32_t width_tiles = get_arg_val<uint32_t>(0);
    uint32_t height_tiles = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_out0 = tt::CBIndex::c_16;
    constexpr uint32_t cb_const1 = tt::CBIndex::c_24;
    constexpr uint32_t cb_sum = tt::CBIndex::c_25;
    constexpr uint32_t cb_max = tt::CBIndex::c_26;
    constexpr uint32_t cb_tmp = tt::CBIndex::c_27;
    constexpr uint32_t cb_global_max = tt::CBIndex::c_28;
    constexpr uint32_t cb_global_sum = tt::CBIndex::c_29;
    constexpr uint32_t cb_tmp2 = tt::CBIndex::c_30;
    init_sfpu(cb_in0, cb_out0);

    // setup
    {
        tile_regs_acquire();
        cb_reserve_back(cb_const1, 1);
        fill_tile(0, 1.f); // const1
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_const1);
        tile_regs_release();
        cb_push_back(cb_const1, 1);
    }

    for(uint32_t y = 0; y < height_tiles; ++y) {
        tile_regs_acquire();
        fill_tile(1, 0.f);   // sum
        fill_tile(2, -10.f); // max: should be small enough
        for(uint32_t x = 0; x < width_tiles; ++x) {
            cb_wait_front(cb_in0, 1);
            copy_tile_init(cb_in0);
            copy_tile(cb_in0, 0, 0); // Tile 0 -> input

            update_online_softmax_values();
        }

        tile_regs_commit();
        tile_regs_wait();
        cb_pop_front(cb_in0, 1);
        pack_tile(1, cb_sum);
        pack_tile(2, cb_max);
        tile_regs_release();
        cb_push_back(cb_sum, 1);
        cb_push_back(cb_max, 1);

        // reduce across the rows in tile to have the real max and sum
        {
            // What we need to do:
            // m_global = reduce_max(m_vec)
            // s_global = reduce_sum(s_vec * exp(m_vec - m_global))

            // Reduce partial max into row wide max
            tile_regs_acquire();
            cb_wait_front(cb_max, 1);
            cb_reserve_back(cb_tmp, 1);
            reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_max, cb_const1, cb_tmp);
            reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_max, cb_const1, 0, 0, 0);
            reduce_uninit();
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_tmp);
            tile_regs_release();
            cb_push_back(cb_tmp, 1);

            tile_regs_acquire();
            cb_wait_front(cb_tmp, 1);
            cb_wait_front(cb_sum, 1);
            cb_reserve_back(cb_global_max, 1);
            cb_reserve_back(cb_tmp2, 1);
            unary_bcast_init<BroadcastType::COL>(cb_tmp, cb_global_max);
            unary_bcast<BroadcastType::COL>(cb_tmp, 0, 0);
            copy_tile_init(cb_sum);
            copy_tile(cb_sum, 0, 1);
            copy_tile_init(cb_max);
            copy_tile(cb_max, 0, 2);
            sub_binary_tile(2, 0, 3);
            exp_tile(3);
            mul_binary_tile(1, 3, 3);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_global_max);
            pack_tile(3, cb_tmp2);
            tile_regs_release();
            cb_push_back(cb_global_max, 1);
            cb_push_back(cb_tmp2, 1);
            cb_pop_front(cb_tmp, 1);

            tile_regs_acquire();
            cb_wait_front(cb_tmp2, 1);
            cb_reserve_back(cb_tmp, 1);
            reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_tmp2, cb_const1, cb_tmp);
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_tmp2, cb_const1, 0, 0, 0);
            reduce_uninit();
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_tmp);
            tile_regs_release();
            cb_push_back(cb_tmp, 1);

            tile_regs_acquire();
            cb_wait_front(cb_tmp, 1);
            cb_reserve_back(cb_global_sum, 1);
            unary_bcast_init<BroadcastType::COL>(cb_tmp, cb_global_sum);
            unary_bcast<BroadcastType::COL>(cb_tmp, 0, 0);
            recip_tile_init();
            recip_tile(0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_global_sum);
            tile_regs_release();
            cb_push_back(cb_global_sum, 1);
            cb_pop_front(cb_tmp, 0);
        }

        for(uint32_t x = 0; x < width_tiles; ++x) {
            tile_regs_acquire();
            cb_wait_front(cb_in0, 1);
            copy_tile_init(cb_in0);
            copy_tile(cb_in0, 0, 0); // Tile 0 -> input
            cb_wait_front(cb_global_sum, 1);
            copy_tile_init(cb_global_sum);
            copy_tile(cb_global_sum, 0, 1); // Tile 1 -> Sum (gobal inverse)
            cb_wait_front(cb_global_max, 1);
            copy_tile_init(cb_global_max);
            copy_tile(cb_global_max, 0, 2); // Tile 2 -> max (global)
            cb_reserve_back(cb_out0, 1); // Output tile


            // dprint_tensix_dest_reg(/*tile_id*/0);
            // dprint_tensix_dest_reg(/*tile_id*/1);
            // dprint_tensix_dest_reg(/*tile_id*/2);


            sub_binary_tile(0, 2, 3);
            dprint_tensix_dest_reg(/*tile_id*/3);

            exp_tile(3);
            dprint_tensix_dest_reg(/*tile_id*/3);
            mul_binary_tile(1, 3, 0);

            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out0);
            cb_pop_front(cb_in0, 1);
            tile_regs_release();
            cb_push_back(cb_out0, 1);
        }
        cb_pop_front(cb_global_max, 1);
        cb_pop_front(cb_global_sum, 1);
        cb_pop_front(cb_sum, 1);
        cb_pop_front(cb_max, 1);
    }

}
}  // namespace NAMESPACE
