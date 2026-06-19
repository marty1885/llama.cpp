#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/identity.h"
#include "api/compute/eltwise_unary/trigonometry.h"
#include <string.h>

#include <tools/profiler/kernel_profiler.hpp>

#ifdef TRISC_MATH
using namespace sfpi;

sfpi_inline vFloat sfpu_sinpi(vFloat x) {
    vFloat xx = x * x;

    return x *
           ((((0x1.406628p-4f * xx - 0x9.93f86p-4f) * xx + 0x2.8cd64p+0f) * xx - 0x5.2aef6p+0f) * xx + 0x3.243f6cp+0f);
}

inline vFloat vector_sin_phase(vFloat x)
{
    vFloat v = x;
    vInt whole_v = float_to_int16(v, RoundMode::NearestEven);
    v -= int32_to_float(whole_v, RoundMode::NearestEven);

    v = sfpu_sinpi(v);
    v_if(whole_v & 1) { v = -v; }
    v_endif;
    return v;
}

#ifdef EXT_FACTOR
sfpi_inline vFloat rope_yarn_ramp(vFloat vec_pos) {
    vFloat y = (vec_pos - CORR_DIMS0) * (1.f / std::max(0.001f, float(CORR_DIMS1 - CORR_DIMS0)));
    v_if(y < 0.f) {
        y = 0;
    }
    v_elseif(y > 1.f) {
        y = 1;
    }
    v_endif;
    return 1.f - y;
}
#endif

template <int max_iter = 3>
sfpi_inline sfpi::vFloat _reciprocal_compat_(const sfpi::vFloat in)
{
    // Force sign to 1 (make number negative)
    sfpi::vFloat val = sfpi::setsgn(in, 1);

    val = setexp(val, 126); // Set exponent to 126 to make the number in 0.5-1
    // Use 1.44 as first guess at x, ideal value would be 1.33.
    // Grayskull has hardwired 1.44 and uses it to avoid a load.
    // We use it here for consistency.
    sfpi::vFloat vConstLn2Recip = 1.442695f;
    sfpi::vFloat two            = 2.0f;
    sfpi::vFloat result         = vConstLn2Recip * (val * vConstLn2Recip + two);

    for (int s_iter = 0; s_iter < (max_iter - 1); s_iter++)
    {
        result = result * (val * result + two);
    }

    sfpi::vInt orig_exp = exexp(in);
    sfpi::vInt new_exp  = exexp(result);

    // "Subtract" exponents, and re-bias.
    // Execute: -1 - exp, then exp += 127
    new_exp -= orig_exp;
    new_exp += 126;

    v_if (new_exp < 0)
    {
        // If rebiased exponent is negative, we need to saturate at 0.
        // This means the initial number was too big so reciprocal result should be 0
        result  = 0.0F;
        new_exp = 0;
    }
    v_endif;

    // Set newly denormalized exponent to result exponent field
    return setexp(result, new_exp);
}

inline void rope_face(int pos, int D, int vec_offset, int face)
{
    vFloat sin_angle = dst_reg[64+face%2];
    vFloat cos_angle = dst_reg[64+face%2+2];
    int dst_offset = face * 8;
    for (int i = 0; i < 4; i++) {
        // Thanks that dst interleaves lanes by default
        vFloat x = dst_reg[dst_offset+i*2];
        vFloat y = dst_reg[dst_offset+i*2+1];

        dst_reg[dst_offset+i*2] = x * cos_angle - y * sin_angle;
        dst_reg[dst_offset+i*2+1] = x * sin_angle + y * cos_angle;
    }
}

inline void rope_tile(int pos, int D, int vec_offset)
{
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(0);
    math::set_addr_mod_base(); // dst_reg[] addressing below uses addr mods 4..7; dropped in the new-SDK port
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);

    #ifdef HAS_FREQ_FACTOR
    // Seperate computation of inverse of freq_factor as otherwise SFPI fails to compile due to
    // failing to allocate registers
    for(int i=0;i<2;i++) {
        int ff_idx = 96+((vec_offset/32)%2)*8+i;
        vFloat ff = vFloat(dst_reg[ff_idx]);
        vFloat d0 = ff;
        vFloat d1 = ff;
        vFloat d2 = ff;
        vFloat d3 = ff;
        sfpi::subvec_transp(d0, d1, d2, d3);
        vFloat r = _reciprocal_compat_<8>(d0);
        v_if(ff < 0) {
            r = -r;
        }
        v_endif;
        dst_reg[ff_idx] = r;
    }
    #endif

    for(int i=0;i<2;i++) {
        int internal_offset = i * 16;
        int pos_in_vector = vec_offset + internal_offset;
        vFloat block_lane_id = int32_to_float((vConstTileId & 15) + pos_in_vector); // No mod operator on SFPI, use bit hack
        vFloat exponent = block_lane_id * vConstFloatPrgm2;

        vFloat term_to_exp = -exponent * vConstFloatPrgm0 - vConstFloatPrgm1;
        vFloat freq = sfpu::_sfpu_exp_fp32_accurate_(term_to_exp);
        #ifdef HAS_FREQ_FACTOR
            int ff_idx = 96+((vec_offset/32)%2)*8+i;
            freq = freq * vFloat(dst_reg[ff_idx]);
        #endif

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

        vFloat vpos = int32_to_float(pos);
        vFloat angle_phase = vpos * theta;
        vFloat sin_value = vector_sin_phase(angle_phase) * mscale;
        vFloat cos_value = vector_sin_phase(0.5f - angle_phase) * mscale;
        dst_reg[64+i] = sin_value;
        dst_reg[64+i+2] = cos_value;
    }

    for (int face = 0; face < 4; face++) {
        rope_face(pos, D, vec_offset + ((face % 2 == 0) ? 0 : 16), face);
    }

    math::clear_dst_reg_addr();
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU);
    // math::clear_addr_mod_base();
    TTI_SETC16(2, 0); // semantically equivalent to above
}

inline void rope_tile_init(float inv_d)
{
    vConstFloatPrgm0 = float(FREQ_BASE_LOG);
    vConstFloatPrgm1 = 1.14472988585f;
    vConstFloatPrgm2 = inv_d;
}
#endif

void kernel_main() {

    uint32_t n_tiles_width_active = get_arg_val<uint32_t>(0);
    uint32_t n_tiles_width = get_arg_val<uint32_t>(1);
    uint32_t n_tiles_height = get_arg_val<uint32_t>(2);
    uint32_t batch_size = get_arg_val<uint32_t>(3);
    uint32_t active_begin = get_arg_val<uint32_t>(4);
    uint32_t active_end = get_arg_val<uint32_t>(5);
    uint32_t height_elements = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_in1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_in2 = tt::CBIndex::c_2;
    constexpr uint32_t cb_out0 = tt::CBIndex::c_16;

    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_16);
    float inv_d = 1.f/(n_tiles_width_active * 32);
    MATH(rope_tile_init(inv_d));

    volatile int* idxs_ptr = nullptr;
    cb_wait_front(cb_in1, 1);
    idxs_ptr = reinterpret_cast<int *>(get_tile_address(cb_in1, 0));
    // idxs_ptr += 4; // Need to shift because read ptr is off by 1 << 4 bytes in BBE

    #ifdef HAS_FREQ_FACTOR
    uint32_t last_ff_idx = -1;
    #endif


    pack_reconfig_data_format(cb_out0);
    for(uint32_t active_id=active_begin; active_id<active_end; active_id++) {
        DeviceZoneScopedN("RoPE Normal");
        uint32_t b = active_id / n_tiles_width_active / n_tiles_height;
        uint32_t w = active_id % n_tiles_width_active;
        cb_wait_front(cb_in0, 1);
        #ifdef HAS_FREQ_FACTOR
        uint32_t ff_idx = w/2;
        bool process_ff = last_ff_idx != ff_idx;
        if(last_ff_idx != ff_idx) {
            if(last_ff_idx != uint32_t(-1)) {
                cb_pop_front(cb_in2, 1);
            }
            cb_wait_front(cb_in2, 1);
        }
        last_ff_idx = ff_idx;
        #endif
        tile_regs_acquire();

        #ifdef HAS_FREQ_FACTOR
        copy_tile_init(cb_in2);
        copy_tile(cb_in2, 0, 3);
        #endif

        copy_tile_init(cb_in0);
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
    #ifdef HAS_FREQ_FACTOR
    if(last_ff_idx != uint32_t(-1)) {
        cb_pop_front(cb_in2, 1);
    }
    #endif

}
