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
#include "compute_kernel_api/eltwise_binary.h"

#include <debug/dprint_tensix.h>

using std::uint32_t;

constexpr int TILE_SIZE = 32;
constexpr float NEG_FP16_MAX = -65504.0f;

// ============================================================================
// SFPU KERNELS
// ============================================================================

#ifdef TRISC_MATH
using namespace sfpi;
using namespace ckernel::sfpu;

inline void make_mask_face(const int w, const int h, const int dst_tile_id) {
    const int write_offset = dst_tile_id * 32;
    if(w <= 0 || h <= 0) {
        #pragma unroll 0
        for(int i=0; i<8; i++) {
            dst_reg[write_offset] = vFloat(0.f);
            dst_reg++;
        }
        return;
    }
    if(w >= 16 && h >= 16) {
        #pragma unroll 0
        for(int i=0; i<8; i++) {
            dst_reg[write_offset] = vFloat(1.f);
            dst_reg++;
        }
        return;
    }

    for(int i = 0; i < 4; i++) {
        vInt y = vConstTileId;
        v_if(y < 16) { y = 0; }
        v_elseif(y < 32) { y = 1; }
        v_elseif(y < 48) { y = 2; }
        v_else { y = 3; }
        v_endif;
        y += i*4;

        for (int half = 0; half < 2; half++) {
            vInt x = (vConstTileId & 15) + half;
            vFloat res = 0.f;
            v_if(y < h && x < w) {
                res = vFloat(1.f);
            }
            v_endif;
            dst_reg[write_offset] = res;
            dst_reg++;
        }
    }
}

inline void make_mask_internal(const uint32_t w, const uint32_t h, const int dst_tile_id) {
    math::set_dst_write_addr<DstTileLayout::Default, DstTileShape::Tile32x32>(0);
    math::set_addr_mod_base();
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);

    for (int face = 0; face < 4; face++) {
        int x = (face % 2) * 16;
        int y = (face / 2) * 16;
        make_mask_face(w - x, h - y, dst_tile_id);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
    }

    math::clear_dst_reg_addr();
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU);
    math::clear_addr_mod_base();
}

void update_online_softmax_values_internal(const uint32_t dst_index_in0, const uint32_t dst_index_in1, const uint32_t dst_index_out) {
    constexpr uint32_t n_vector_in_tile = 32;
    const uint32_t in_base_idx = dst_index_in0 * n_vector_in_tile;
    const uint32_t sum_base_idx = dst_index_in1 * n_vector_in_tile;
    const uint32_t max_base_idx = dst_index_out * n_vector_in_tile;
    const uint32_t tile_mask_idx = 3 * n_vector_in_tile;

    for (size_t i = 0; i < 8; i++) {
        vFloat x = dst_reg[in_base_idx];
        vFloat sum = dst_reg[sum_base_idx];
        vFloat max = dst_reg[max_base_idx];
        vFloat tile_mask = dst_reg[tile_mask_idx];

        #ifdef SCALE
            x = x * SCALE;
        #endif

        vFloat new_max = max;
        v_if(x > max && tile_mask == 1.f) {
            new_max = x;
        } v_endif;

        v_if(tile_mask == 1.f) {
            sum = sum * _sfpu_exp_21f_<true>(max - new_max) + _sfpu_exp_21f_<true>(x - new_max);
        }
        v_endif;

        dst_reg[sum_base_idx] = sum;
        dst_reg[max_base_idx] = new_max;
        dst_reg++;
    }
}

void compute_result_for_online_softmax_internal(const uint32_t dst_index_in0, const uint32_t dst_index_in1, const uint32_t dst_index_out) {
    constexpr uint32_t n_vector_in_tile = 32;
    const uint32_t in_base_idx = dst_index_out * n_vector_in_tile;
    const uint32_t sum_base_idx = dst_index_in0 * n_vector_in_tile;
    const uint32_t max_base_idx = dst_index_in1 * n_vector_in_tile;
    const uint32_t tile_mask_idx = 3 * n_vector_in_tile;

    for (size_t i = 0; i < 8; i++) {
        vFloat x = dst_reg[in_base_idx];
        vFloat inv_sum = dst_reg[sum_base_idx];
        vFloat x_max = dst_reg[max_base_idx];
        vFloat tile_mask = dst_reg[tile_mask_idx];

        #ifdef SCALE
            x = x * SCALE;
        #endif

        vFloat res = 0;
        v_if(tile_mask == 1.f) {
            res = _sfpu_exp_21f_<true>(x - x_max) * inv_sum;
        }
        v_endif;

        dst_reg[in_base_idx] = res;
        dst_reg++;
    }
}

#endif

// ============================================================================
// OPERATION WRAPPERS
// ============================================================================

static void update_online_softmax_values() {
    MATH(_llk_math_eltwise_binary_sfpu_params_<false>(update_online_softmax_values_internal, 0, 1, 2));
}

static void compute_result_for_online_softmax() {
    MATH(_llk_math_eltwise_binary_sfpu_params_<false>(compute_result_for_online_softmax_internal, 1, 2, 0));
}

static void make_mask(const int w, const int h, const int dst_tile_id) {
    MATH(make_mask_internal(w, h, dst_tile_id));
}

// ============================================================================
// MAIN KERNEL
// ============================================================================

namespace NAMESPACE {
void MAIN {
    uint32_t width = get_arg_val<uint32_t>(0);
    uint32_t height = get_arg_val<uint32_t>(1);
    uint32_t n_head = get_arg_val<uint32_t>(3);
    uint32_t batch = get_arg_val<uint32_t>(4);

    const uint32_t width_tiles = (width + TILE_SIZE - 1) / TILE_SIZE;
    const uint32_t height_tiles = (height + TILE_SIZE - 1) / TILE_SIZE;

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_out0 = tt::CBIndex::c_16;
    constexpr uint32_t cb_const1 = tt::CBIndex::c_24;
    constexpr uint32_t cb_sum = tt::CBIndex::c_25;
    constexpr uint32_t cb_max = tt::CBIndex::c_26;
    constexpr uint32_t cb_tmp = tt::CBIndex::c_27;
    constexpr uint32_t cb_global_max = tt::CBIndex::c_28;
    constexpr uint32_t cb_global_sum = tt::CBIndex::c_29;
    constexpr uint32_t cb_tmp2 = tt::CBIndex::c_30;
    constexpr uint32_t cb_tile_mask = tt::CBIndex::c_31;

    auto select_tile_mask = [=](uint32_t y, uint32_t x) {
        if(x == width_tiles-1 && y == height_tiles-1) return 3;
        if(x == width_tiles-1) return 2;
        if(y == height_tiles-1) return 1;
        return 0;
    };

    auto select_reduce_mask = [=](uint32_t y) {
        if(width < TILE_SIZE && height < TILE_SIZE) return 3;
        if(y == height_tiles-1) return 1;
        return 0;
    };

    // ========================================================================
    // ONE-TIME SETUP
    // ========================================================================

    init_sfpu(cb_in0, cb_out0);
    binary_op_init_common(cb_in0, cb_const1, cb_out0);

    // Create constant tile (1.0)
    {
        tile_regs_acquire();
        cb_reserve_back(cb_const1, 1);
        fill_tile(0, 1.f);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_const1);
        tile_regs_release();
        cb_push_back(cb_const1, 1);
    }

    // Create mask tiles for different edge cases
    {
        tile_regs_acquire();
        cb_reserve_back(cb_tile_mask, 4);
        const uint32_t remaining_width = width % TILE_SIZE == 0 ? TILE_SIZE : width % TILE_SIZE;
        const uint32_t remaining_height = height % TILE_SIZE == 0 ? TILE_SIZE : height % TILE_SIZE;
        make_mask(32, 32, 0);                             // Full tile
        make_mask(32, remaining_height, 1);               // Bottom edge
        make_mask(remaining_width, 32, 2);                // Right edge
        make_mask(remaining_width, remaining_height, 3);  // Bottom-right corner
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_tile_mask, 0);
        pack_tile(1, cb_tile_mask, 1);
        pack_tile(2, cb_tile_mask, 2);
        pack_tile(3, cb_tile_mask, 3);
        tile_regs_release();
        cb_push_back(cb_tile_mask, 4);
    }

    // ========================================================================
    // MAIN COMPUTATION LOOP
    // ========================================================================

    for(uint32_t b = 0; b < n_head*batch; ++b) {
        for(uint32_t y = 0; y < height_tiles; ++y) {

            // ================================================================
            // PHASE 1: Compute per-tile max/sum statistics across row
            // ================================================================

            tile_regs_acquire();
            fill_tile(1, 0.f);   // sum accumulator
            fill_tile(2, NEG_FP16_MAX); // max accumulator (small initial value)

            for(uint32_t x = 0; x < width_tiles; ++x) {
                cb_wait_front(cb_in0, 1);
                copy_tile_init(cb_in0);
                copy_tile(cb_in0, 0, 0); // input -> tile 0

                cb_wait_front(cb_tile_mask, 4);
                copy_tile_init(cb_tile_mask);
                copy_tile(cb_tile_mask, select_tile_mask(y, x), 3); // mask -> tile 3

                update_online_softmax_values(); // updates tiles 1,2 with running max/sum
                cb_pop_front(cb_in0, 1);
            }

            tile_regs_commit();
            tile_regs_wait();
            pack_tile(1, cb_sum);    // partial sums
            pack_tile(2, cb_max);    // partial maxes
            tile_regs_release();
            cb_push_back(cb_sum, 1);
            cb_push_back(cb_max, 1);

            // ================================================================
            // PHASE 2: Reduce to global row statistics
            // TODO: Implement all-reduce across cores
            // ================================================================

            // Step 2a: Reduce max across row to get global max
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

            // Step 2b: Compute numerically stable sum: s_vec * exp(m_vec - m_global)
            tile_regs_acquire();
            cb_wait_front(cb_tmp, 1);     // global max
            cb_wait_front(cb_sum, 1);     // partial sums
            cb_reserve_back(cb_global_max, 1);
            cb_reserve_back(cb_tmp2, 1);

            unary_bcast_init<BroadcastType::COL>(cb_tmp, cb_global_max);
            unary_bcast<BroadcastType::COL>(cb_tmp, 0, 0);  // broadcast global max
            copy_tile_init(cb_sum);
            copy_tile(cb_sum, 0, 1);      // partial sums -> tile 1
            copy_tile_init(cb_max);
            copy_tile(cb_max, 0, 2);      // partial maxes -> tile 2

            sub_binary_tile(2, 0, 3);     // m_vec - m_global -> tile 3
            exp_tile(3);                  // exp(m_vec - m_global) -> tile 3
            mul_binary_tile(1, 3, 3);     // s_vec * exp(m_vec - m_global) -> tile 3
            copy_tile_init(cb_tile_mask);
            copy_tile(cb_tile_mask, select_reduce_mask(y), 2);  // apply mask
            mul_binary_tile(2, 3, 3);

            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_global_max);  // save global max
            pack_tile(3, cb_tmp2);        // save adjusted sum
            tile_regs_release();
            cb_push_back(cb_global_max, 1);
            cb_push_back(cb_tmp2, 1);
            cb_pop_front(cb_tmp, 1);
            cb_pop_front(cb_sum, 1);
            cb_pop_front(cb_max, 1);

            // Step 2c: Reduce sum and compute reciprocal
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
            recip_tile(0);                // 1/sum -> tile 0
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_global_sum);
            tile_regs_release();
            cb_push_back(cb_global_sum, 1);

            cb_pop_front(cb_tmp, 1);
            cb_pop_front(cb_tmp2, 1);

            // ================================================================
            // PHASE 3: Compute final softmax values
            // ================================================================

            cb_wait_front(cb_global_sum, 1);
            cb_wait_front(cb_global_max, 1);

            for(uint32_t x = 0; x < width_tiles; ++x) {
                tile_regs_acquire();
                cb_wait_front(cb_in0, 1);
                copy_tile_init(cb_in0);
                copy_tile(cb_in0, 0, 0);              // input -> tile 0
                copy_tile_init(cb_global_sum);
                copy_tile(cb_global_sum, 0, 1);       // 1/sum -> tile 1
                copy_tile_init(cb_global_max);
                copy_tile(cb_global_max, 0, 2);       // global max -> tile 2
                copy_tile_init(cb_tile_mask);
                copy_tile(cb_tile_mask, select_tile_mask(y, x), 3); // mask -> tile 3
                cb_reserve_back(cb_out0, 1);

                compute_result_for_online_softmax(); // exp(x-max) * (1/sum)

                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_out0);
                cb_pop_front(cb_in0, 1);
                tile_regs_release();
                cb_push_back(cb_out0, 1);
            }
            cb_pop_front(cb_global_max, 1);
            cb_pop_front(cb_global_sum, 1);
        }
    }
}
}
