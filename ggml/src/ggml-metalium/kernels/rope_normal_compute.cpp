#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/eltwise_unary/recip.h"
#include "compute_kernel_api/eltwise_unary/identity.h"
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

inline void rope_face(int pos, int D, int vec_offset, int face)
{
    vFloat freq = dst_reg[64+face%2];
    vFloat mscale = dst_reg[64+face%2+2];
    int dst_offset = face * 8;
    for (int i = 0; i < 4; i++) {
        // Standard RoPE math
        vFloat angle = int32_to_float(pos) * freq;
        vFloat sin_angle = vector_sin_phase(angle) * mscale;
        vFloat cos_angle = vector_sin_phase(0.5f - angle) * mscale;

        // Thanks that dst interleaves lanes by default
        vFloat x = dst_reg[dst_offset+i*2];
        vFloat y = dst_reg[dst_offset+i*2+1];

        dst_reg[dst_offset+i*2] = x * cos_angle - y * sin_angle;
        dst_reg[dst_offset+i*2+1] = x * sin_angle + y * cos_angle;
    }
}

inline void rope_tile(int pos, int D, int vec_offset)
{

    math::set_dst_write_addr<DstTileLayout::Default, DstTileShape::Tile32x32>(0);
    math::set_addr_mod_base();
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);

    for(int i=0;i<2;i++) {
        int internal_offset = i * 16;
        int pos_in_vector = vec_offset + internal_offset;
        vFloat block_lane_id = int32_to_float((vConstTileId & 15) + pos_in_vector); // No mod operator on SFPI, use bit hack
        vFloat exponent = block_lane_id * vConstFloatPrgm2;

        vFloat term_to_exp = -exponent * vConstFloatPrgm0 - vConstFloatPrgm1;
        vFloat freq = vector_exp(term_to_exp);

        vFloat freq_scaled = freq;
        vFloat mscale = 1.f;
        #ifdef FREQ_SCALE
            freq_scaled = freq * FREQ_SCALE;
        #endif
        #ifdef ATTN_FACTOR
            mscale = ATTN_FACTOR;
        #endif
        vFloat theta = freq_scaled;
        // enable YaRN if needed
        #ifdef EXT_FACTOR
            vFloat ramp_mix = rope_yarn_ramp(block_lane_id) * EXT_FACTOR;
            theta = freq_scaled * (1 - ramp_mix) + freq * ramp_mix;
            #ifdef LOG_1_FREQ_SCALE
                mscale *= 1.0f + 0.1f * LOG_1_FREQ_SCALE;
            #endif // else mscahe *= 1 (the other half collasps to 0) - does nothing
        #endif
        dst_reg[64+i] = theta;
        dst_reg[64+i+2] = mscale;
    }

    for (int face = 0; face < 4; face++) {
        rope_face(pos, D, vec_offset + ((face % 2 == 0) ? 0 : 16), face);
    }

    math::clear_dst_reg_addr();
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU);
    math::clear_addr_mod_base();
}

inline void rope_tile_init(float inv_d)
{
    vConstFloatPrgm0 = float(FREQ_BASE_LOG);
    vConstFloatPrgm1 = 1.14472988585f;
    vConstFloatPrgm2 = inv_d;
}
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
    float inv_d = 1.f/(n_tiles_width_active * 32);
    MATH(rope_tile_init(inv_d));

    int* idxs_ptr = nullptr;
    cb_wait_front(cb_in1, 1);
    cb_get_tile(cb_in1, 0, &idxs_ptr);
    idxs_ptr += 4; // Need to shift because read ptr is off by 1 << 4 bytes in BBE


    copy_tile_init(cb_in0);
    pack_reconfig_data_format(cb_out0);
    for(uint32_t active_id=active_begin; active_id<active_end; active_id++) {
        uint32_t b = active_id / n_tiles_width_active / n_tiles_height;
        uint32_t w = active_id % n_tiles_width_active;
        cb_wait_front(cb_in0, 1);
        tile_regs_acquire();

        copy_tile(cb_in0, 0, 0);
        MATH(rope_tile(idxs_ptr[b], inv_d, w*32));
        tile_regs_commit();
        tile_regs_wait();

        cb_reserve_back(cb_out0, 2);
        pack_tile(0, cb_out0, 0);
        tile_regs_release();
        cb_push_back(cb_out0, 1);
        cb_pop_front(cb_in0, 1);
    }

    cb_pop_front(cb_in1, 1);

}
}
