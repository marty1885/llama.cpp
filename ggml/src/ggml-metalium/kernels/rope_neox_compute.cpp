#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/eltwise_unary/recip.h"
#include "compute_kernel_api/eltwise_unary/trigonometry.h"
#include <string.h>

#include <tools/profiler/kernel_profiler.hpp>

#ifdef TRISC_MATH
using namespace sfpi;

// Implemented algorithm exp_f24 from https://ieeexplore.ieee.org/document/9810030
inline vFloat vector_exp(sfpi::vFloat val) {
    sfpi::vFloat y = 0.0f;
    // Intermediary values can overflow if input value is below -88.0f, which leads to output increasing again instead
    // of staying at 0. This overflow happens when `log2(e) * val < 127.0f`, which correspond to `val < 88.0f`
    v_if(val > -88.0f) {
        // The paper relies on the following formula (c.f. Section 2 and 3 of paper):
        // z = (bias + x * factor * N_m; where:
        // factor = 0x00b8aa3b (computed through log(e))
        // bias = 0x3f800000
        sfpi::vInt z = sfpu::_float_to_int32_(val * sfpi::vFloat(0x00b8aa3b) + sfpi::vFloat(0x3f800000));
        sfpi::vInt zii = exexp(sfpi::reinterpret<sfpi::vFloat>(z));         // Extract exponent
        sfpi::vInt zif = sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z));  // Extract mantissa

        // Polynomial coefficients for approximation of exp on [1; 2]
        vFloat POLY_D1;
        vInt POLY_D2;
        vInt POLY_D3;

        v_if(zif > 0x00600000) {
            // Fourth segment (highest values of the mantissa)
            POLY_D1 = 0.52496276e-7f;
            POLY_D2 = 0x81354a;
            POLY_D3 = 0x10a440;
        }
        v_elseif(zif > 0x00400000) {
            // Third segment
            POLY_D1 = 0.4414393e-7f;
            POLY_D2 = 0xcdf4b4;
            POLY_D3 = 0x3e4d6;
        }
        v_elseif(zif > 0x00200000) {
            // Second segment
            POLY_D1 =0.37120473e-7f;
            POLY_D2 = 0x1113a74;
            POLY_D3 = 0x9f16;
        }
        v_else {
            // First segment
            POLY_D1 = 0.31214472e-7f;
            POLY_D2 = 0x151d842;
            // Note: The original C code has a float constant here
            // We treat it as an integer for performance
            POLY_D3 = 328;
        }
        v_endif;

        sfpi::vFloat d1 = sfpi::vFloat(POLY_D1);
        sfpi::vFloat d2 = sfpi::int32_to_float(sfpi::vInt(POLY_D2) + zif, 0);
        sfpi::vFloat d3 = sfpi::int32_to_float(sfpi::vInt(POLY_D3) + zif, 0);
        d2 = d1 * d2;
        zif = sfpu::_float_to_int32_(d2 * d3);

        // Restore exponent
        zii = sfpi::reinterpret<sfpi::vInt>(
            sfpi::setexp(sfpi::reinterpret<sfpi::vFloat>(zif), 127U + zii));  // restore exponent

        y = sfpi::reinterpret<sfpi::vFloat>(zii);
    }
    v_endif;
    return y;
}

inline vFloat vector_sin_phase(vFloat x)
{
    vFloat v = x;
    vInt whole_v = float_to_int16(v, 0);
    v -= int32_to_float(whole_v, 0);

    v = ckernel::sfpu::sfpu_sinpi<false>(v);
    v_if(whole_v & 1) { v = -v; }
    v_endif;
    return v;
}

// Each vInt is really treated as if it's a 8x4(width=8, height=4) block in SFPU and interacting with Dst
// This function takes in a pointer `ptr` and load each integer into one of the rows of a vInt
// int*      vInt
// a         aaaaaaaa
// b         bbbbbbbb
// c    ->   cccccccc
// d         dddddddd
inline vInt load_into_row(int* ptr)
{
    vInt row_mask = vConstTileId & (~15);
    vInt v = ptr[0];
    v_if(row_mask == 16) {
        v = ptr[1];
    }
    v_elseif(row_mask == 32) {
        v = ptr[2];
    }
    v_elseif(row_mask == 48) {
        v = ptr[3];
    }
    v_endif;
    return v;
}

inline void rope_face(int pos, float inv_d, int vec_offset, int face_idx)
{
    DeviceZoneScopedN("ROPE-FACE");
    // RoPE - we need to calculate the final rotation sin(angle) and cos(angle)
    // Where andgle = pos * freq
    // and freq = pow(100000, 2.0f * i / DIM_SIZE)
    //
    // To improve SFPU accuracy (and better prformance), we can rewrite the
    // compute as the following using a few identities:
    // evaulate sin_phase(angle_phase) and cos_phase(angle_phase)
    // angle_phase = pos * pow(10000, 2.0f * i / DIM_SIZE) / PI
    //             = pos * exp(-(2.0f * i / DIM_SIZE) * log(10000)) / PI
    //             = pos * exp(-(2.0f * i / DIM_SIZE) * log(10000) + log(1/PI))
    // where we compute
    //     exponent = 2.0f * i / DIM_SIZE
    // and
    //     log(10000) = 9.21034037, log(1/PI) = -1.14472988585
    // thus
    // angle_phase = pos * exp(-exponent * 9.21034037f - 1.14472988585f)
    // and
    //      we preload 9.21034037f and 1.14472988585f into vConstFloatPrgm{0,1}
    //      to avoid loading values into LReg in runtime
    // NOTE: SFPU does not have a / operator. Scalars can be done on RISC-V (softfp)
    //      which is slow. So the value is computed once and loaded into
    //      vConstFloatPrg2. Reused across the kernel.
    // TODO: DIM_SIZE should be treated as a constant and this 1.f/DIM_SIZE can be
    //      evaulated at compile time.
    int face_row = face_idx / 2;
    int face_col = face_idx % 2;
    int dst_offset = face_idx*8;
    for (int h = 0; h < 2; h++) {
        vFloat freq = dst_reg[64+8+face_col*2+h];
        for (int i = 0; i < 4; i++) {
            vFloat vpos = int32_to_float(pos);

            // Standard RoPE math
            vFloat angle_phase = vpos * freq;
            vFloat sin_value = vector_sin_phase(angle_phase);
            vFloat cos_value = vector_sin_phase(0.5f - angle_phase);

            size_t idx = i*2+h;
            vFloat x = dst_reg[dst_offset+idx];
            vFloat y = dst_reg[dst_offset+idx+32];
            dst_reg[dst_offset+idx] = x * cos_value - y * sin_value;
            dst_reg[dst_offset+idx+32] = x * sin_value + y * cos_value;
        }
    }
}

inline void rope_tile_init(float inv_d)
{
    vConstFloatPrgm0 = float(FREQ_BASE_LOG);
    vConstFloatPrgm1 = 1.14472988585f;
    vConstFloatPrgm2 = inv_d;
}

inline void rope_tile(int pos, float inv_d, int vec_offset)
{
    (void)inv_d; // Unused
    DeviceZoneScopedN("ROPE-TILE");
    math::set_dst_write_addr<DstTileLayout::Default, DstTileShape::Tile32x32>(0);
    math::set_addr_mod_base();
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);

    for(int i=0;i<4;i++) {
        int internal_offset = ((i / 2 == 0) ? 0 : 16);
        int pos_in_vector = vec_offset + internal_offset;
        vFloat block_lane_id = int32_to_float((vConstTileId & 15) + (pos_in_vector + i % 2)); // No mod operator on SFPI, use bit hack
        vFloat exponent = block_lane_id * vConstFloatPrgm2;

        vFloat term_to_exp = -exponent * vConstFloatPrgm0 - vConstFloatPrgm1;
        vFloat freq = vector_exp(term_to_exp);
        dst_reg[64+8+i] = freq;
    }

    for (int face = 0; face < 4; face++) {
        int internal_offset = ((face % 2 == 0) ? 0 : 16);
        int idx_offset = face > 1 ? 16 : 0;
        rope_face(pos, inv_d, vec_offset + internal_offset, face);
    }

    math::clear_dst_reg_addr();
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU);
    math::clear_addr_mod_base();
}

// inline void rope_tile_precompute_pos(int* pos)
// {
//     DeviceZoneScopedN("ROPE-TILE-PRECOMP-POS");
//     math::set_dst_write_addr<DstTileLayout::Default, DstTileShape::Tile32x32>(0);
//     math::set_addr_mod_base();
//     TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);

//     for (int i=0;i<8;i++) {
//         vFloat vpos = int32_to_float(load_into_row(pos+i*4));
//         dst_reg[64+i] = vpos;
//     }

//     math::clear_dst_reg_addr();
//     TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU);
//     math::clear_addr_mod_base();
// }
#endif

namespace NAMESPACE {
void MAIN {

    uint32_t n_tiles_width_active = get_arg_val<uint32_t>(0);
    uint32_t n_tiles_width = get_arg_val<uint32_t>(1);
    uint32_t n_tiles_height = get_arg_val<uint32_t>(2);
    uint32_t batch_size = get_arg_val<uint32_t>(3);
    uint32_t active_begin = get_arg_val<uint32_t>(4);
    uint32_t active_end = get_arg_val<uint32_t>(5);
    uint32_t height_elements = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_in1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_out0 = tt::CBIndex::c_16;

    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_16);
    float inv_d = 1.f/(n_tiles_width_active * (32 / 2));
    MATH(rope_tile_init(inv_d));

    uint32_t last_h = (uint32_t)-1;
    int* idxs_ptr = nullptr;
    cb_wait_front(cb_in1, 1);
    cb_get_tile(cb_in1, 0, &idxs_ptr);
    idxs_ptr += 4; // Need to shift because read ptr is off by 1 << 4 bytes in BBE


    for(uint32_t active_id=active_begin; active_id<active_end; active_id++) {
        uint32_t b = active_id / (n_tiles_width_active/2) / n_tiles_height;
        uint32_t w = active_id % (n_tiles_width_active/2);
        cb_wait_front(cb_in0, 2);
        tile_regs_acquire();

        copy_tile_init(cb_in0);
        copy_tile(cb_in0, 0, 0);
        copy_tile(cb_in0, 1, 1);
        MATH(rope_tile(idxs_ptr[b], inv_d, w*32));
        tile_regs_commit();
        tile_regs_wait();

        cb_reserve_back(cb_out0, 2);
        pack_reconfig_data_format(cb_out0);
        pack_tile(0, cb_out0, 0);
        pack_tile(1, cb_out0, 1);
        tile_regs_release();
        cb_push_back(cb_out0, 2);
        cb_pop_front(cb_in0, 2);
    }

    cb_pop_front(cb_in1, 1);

}
}
