// ================================================================
// et_tensor.hpp — Type-safe zero-cost C++17 abstraction for
//                 ET-SoC-1 tensor instructions
//
// Copyright (c) 2025 Ainekko, Co.
// SPDX-License-Identifier: Apache-2.0
//
// Usage:
//
//   constexpr et::scp_region<0,  16, et::fp32> scp_a;
//   constexpr et::scp_region<16, 16, et::fp32> scp_b;
//
//   scp_a.load<0>(addr, rows, stride).wait();
//   scp_b.load_transpose<1>(addr, k, stride).wait();
//
//   et::matmul_result<et::fp32> c;
//   c = et::matmul(scp_a, scp_b, rows, k, first_pass);
//   c.wait();
//   c.store(addr, rows, stride).wait();
//
// ================================================================
#pragma once

#include <cstdint>
#include <type_traits>

namespace et {

// ================================================================
// Element types
// ================================================================

struct fp32  {};
struct fp16  {};
struct int8  {};
struct int16 {};
struct int32 {};

// ================================================================
// Element traits
// ================================================================

template<typename T> struct elem_traits;

template<> struct elem_traits<fp32> {
    using c_type = float;
    static constexpr unsigned size           = 4;
    static constexpr unsigned per_line       = 16;
    static constexpr unsigned max_k          = 16;
    static constexpr unsigned transpose_code = 7;      // TensorLoadTranspose32
    static constexpr unsigned fma_opcode     = 0;      // TensorType = 000
    static constexpr bool     is_float       = true;
};

template<> struct elem_traits<fp16> {
    using c_type = uint16_t;  // no native half; raw bits
    static constexpr unsigned size           = 2;
    static constexpr unsigned per_line       = 32;
    static constexpr unsigned max_k          = 32;
    static constexpr unsigned transpose_code = 3;      // TensorLoadTranspose16
    static constexpr unsigned fma_opcode     = 2;      // TensorType = 010
    static constexpr bool     is_float       = true;
};

template<> struct elem_traits<int8> {
    using c_type = int8_t;
    static constexpr unsigned size           = 1;
    static constexpr unsigned per_line       = 64;
    static constexpr unsigned max_k          = 64;
    static constexpr unsigned transpose_code = 2;      // TensorLoadTranspose8
    static constexpr unsigned fma_opcode     = 3;      // TensorType = 011
    static constexpr bool     is_float       = false;
};

template<> struct elem_traits<int16> {
    using c_type = int16_t;
    static constexpr unsigned size           = 2;
    static constexpr unsigned per_line       = 32;
    static constexpr unsigned max_k          = 32;
    static constexpr unsigned transpose_code = 3;      // TensorLoadTranspose16
    static constexpr unsigned fma_opcode     = 4;      // TensorType = 100
    static constexpr bool     is_float       = false;
};

template<> struct elem_traits<int32> {
    using c_type = int32_t;
    static constexpr unsigned size           = 4;
    static constexpr unsigned per_line       = 16;
    static constexpr unsigned max_k          = 16;
    static constexpr unsigned transpose_code = 7;
    static constexpr unsigned fma_opcode     = 0;
    static constexpr bool     is_float       = false;
};

// Input type -> accumulator type
// fp32  x fp32  -> fp32     (TensorFMA32)
// fp16  x fp16  -> fp32     (TensorFMA16)
// int8  x int8  -> int32    (TensorIMA8)
// int16 x int16 -> int32    (TensorIMA16)
template<typename T> struct acc_type_of;
template<> struct acc_type_of<fp32>  { using type = fp32;  };
template<> struct acc_type_of<fp16>  { using type = fp32;  };
template<> struct acc_type_of<int8>  { using type = int32; };
template<> struct acc_type_of<int16> { using type = int32; };

template<typename T>
using acc_type_of_t = typename acc_type_of<T>::type;

// Encoding helpers

constexpr unsigned bcols_encode(unsigned out_cols) { return (out_cols / 4) - 1; }
constexpr unsigned size_encode(unsigned out_cols)  { return (out_cols / 4) - 1; }
constexpr unsigned regs_per_row(unsigned bcols)    { return (bcols >= 2) ? 2 : 1; }

namespace low_level {
// Low-level hardware control
//
// Raw CSR encoding + inline asm


enum class wait_id : unsigned {
    load_0     = 0,
    load_1     = 1,
    load_l2_0  = 2,
    load_l2_1  = 3,
    prefetch_0 = 4,
    prefetch_1 = 5,
    cacheops   = 6,
    fma        = 7,
    store      = 8,
    reduce     = 9,
    quant      = 10,
};

inline __attribute__((always_inline))
void tensor_wait(wait_id id) {
    uint64_t v = static_cast<unsigned>(id);
    __asm__ __volatile__("csrw 0x830, %[id]\n" : : [id] "r"(v) : "memory");
}

inline __attribute__((always_inline))
void tensor_load(
    bool tmask, bool coop,
    unsigned dst_start, unsigned transform, unsigned tenb,
    uint64_t addr, unsigned offset, unsigned num_lines_m1,
    uint64_t stride, unsigned id)
{
    uint64_t csr =
        (((uint64_t)tmask & 1)        << 63) |
        (((uint64_t)coop & 1)         << 62) |
        (((uint64_t)transform & 7)    << 59) |
        (((uint64_t)dst_start & 0x3F) << 53) |
        (((uint64_t)tenb & 1)         << 52) |
        ((addr & 0xFFFFFFFFFFC0ULL))         |
        (((uint64_t)offset & 3)       << 4)  |
        (((uint64_t)num_lines_m1 & 0xF));
    uint64_t x31 =
        (stride & 0xFFFFFFFFFFC0ULL) | ((uint64_t)id & 1);
    __asm__ __volatile__(
        "mv x31, %[x31v]\n"
        "csrw 0x83f, %[csrv]\n"
        : : [x31v] "r"(x31), [csrv] "r"(csr) : "x31", "memory");
}

inline __attribute__((always_inline))
void tensor_fma(
    bool tmask, unsigned bcols, unsigned arows_m1, unsigned acols_m1,
    unsigned aoffset, bool tenc_loc, bool tena_unsigned, bool tenb_unsigned,
    bool tenb_loc, unsigned bstart, unsigned astart, unsigned opcode, bool mul)
{
    uint64_t csr =
        (((uint64_t)tmask & 1)         << 63) |
        (((uint64_t)bcols & 3)         << 55) |
        (((uint64_t)arows_m1 & 0xF)   << 51) |
        (((uint64_t)acols_m1 & 0xF)   << 47) |
        (((uint64_t)aoffset & 0xF)    << 43) |
        (((uint64_t)tenc_loc & 1)      << 23) |
        (((uint64_t)tena_unsigned & 1) << 22) |
        (((uint64_t)tenb_unsigned & 1) << 21) |
        (((uint64_t)tenb_loc & 1)      << 20) |
        (((uint64_t)bstart & 0xFF)    << 12) |
        (((uint64_t)astart & 0xFF)    << 4)  |
        (((uint64_t)opcode & 7)       << 1)  |
        (((uint64_t)mul & 1));
    __asm__ __volatile__("csrw 0x801, %[csr]\n" : : [csr] "r"(csr) :);
}

inline __attribute__((always_inline))
void tensor_store_regs(
    unsigned step, unsigned freg, unsigned size,
    unsigned rows_m1, uint64_t addr, unsigned coop, uint64_t stride)
{
    uint64_t csr =
        (((uint64_t)step & 3)      << 62) |
        (((uint64_t)freg & 0x1F)   << 57) |
        (((uint64_t)size & 3)      << 55) |
        (((uint64_t)rows_m1 & 0xF) << 51) |
        (((uint64_t)coop & 3)      << 49) |
        ((addr & 0xFFFFFFFFFFF0ULL));
    uint64_t x31 = stride & 0xFFFFFFFFFF0ULL;
    __asm__ __volatile__(
        "mv x31, %[x31v]\n"
        "csrw 0x87f, %[csrv]\n"
        : : [x31v] "r"(x31), [csrv] "r"(csr) : "x31", "memory");
}

inline __attribute__((always_inline))
void tensor_store_scp(
    unsigned entry_stride, unsigned scp_start,
    unsigned rows_m1, uint64_t addr, uint64_t stride)
{
    uint64_t csr =
        (((uint64_t)entry_stride & 3) << 62) |
        (((uint64_t)scp_start & 0x3F) << 56) |
        ((addr & 0xFFFFFFFFFFC0ULL))          |
        (((uint64_t)rows_m1 & 0xF)    << 51) |
        ((uint64_t)1                   << 48);
    uint64_t x31 = stride & 0xFFFFFFFFFFC0ULL;
    __asm__ __volatile__(
        "mv x31, %[x31v]\n"
        "csrw 0x87f, %[csrv]\n"
        : : [x31v] "r"(x31), [csrv] "r"(csr) : "x31", "memory");
}

inline __attribute__((always_inline))
uint64_t tensor_error() {
    uint64_t err;
    __asm__ __volatile__("csrr %0, 0x808" : "=r"(err));
    return err;
}

inline __attribute__((always_inline))
void clear_tensor_error() { __asm__ __volatile__("csrwi 0x808, 0" : :); }

inline __attribute__((always_inline))
void ucache_control(bool scp_en, unsigned rep_rate = 0, unsigned max_out = 0) {
    uint64_t csr =
        (((uint64_t)max_out & 0x1F) << 6) |
        (((uint64_t)rep_rate & 7)   << 2) |
        (((uint64_t)scp_en)         << 1);
    __asm__ __volatile__("csrw 0x810, %[csr]\n" : : [csr] "r"(csr) : "x31");
}

inline __attribute__((always_inline))
void fence() { __asm__ __volatile__("fence\n"); }

} // namespace low_level

template<low_level::wait_id ID>
struct [[nodiscard]] wait_token {
    inline __attribute__((always_inline))
    void wait() const { low_level::tensor_wait(ID); }

    wait_token() = default;
    wait_token(const wait_token&) = delete;
    wait_token& operator=(const wait_token&) = delete;
    wait_token(wait_token&&) = default;
    wait_token& operator=(wait_token&&) = default;
};

using load_token_0 = wait_token<low_level::wait_id::load_0>;
using load_token_1 = wait_token<low_level::wait_id::load_1>;
using store_token  = wait_token<low_level::wait_id::store>;

// matmul_result: C matrix living in f-registers
//
// Default-constructible (hoist before loops), Move-only (one set
// of f-registers, one owner). Typed store to prevent writes to
// incompatiable types

template<
    typename AccType,
    unsigned OutCols = 16,
    unsigned FReg    = 0,
    unsigned Step    = 0>
struct matmul_result {
    static_assert(OutCols == 4 || OutCols == 8 || OutCols == 12 || OutCols == 16,
        "Output columns must be 4, 8, 12, or 16");
    static_assert(FReg < 32, "Starting f-register must be < 32");
    static_assert(Step <= 3, "Register step must be 0-3");

    using acc_type = AccType;
    using c_type   = typename elem_traits<AccType>::c_type;

    static constexpr unsigned out_cols = OutCols;
    static constexpr unsigned freg     = FReg;
    static constexpr unsigned step     = Step;
    static constexpr unsigned size_enc = size_encode(OutCols);
    static constexpr unsigned max_rows = 32 / regs_per_row(bcols_encode(OutCols));

    inline __attribute__((always_inline))
    void wait() const { low_level::tensor_wait(low_level::wait_id::fma); }

    [[nodiscard]] inline __attribute__((always_inline))
    store_token store(c_type* addr, unsigned rows, uint64_t stride) const {
        low_level::tensor_store_regs(Step, FReg, size_enc, rows - 1, (uint64_t)addr, 0, stride);
        return {};
    }

    [[nodiscard]] inline __attribute__((always_inline))
    store_token store_coop(c_type* addr, unsigned rows, uint64_t stride,
                           unsigned partners) const {
        low_level::tensor_store_regs(Step, FReg, size_enc, rows - 1, (uint64_t)addr, partners, stride);
        return {};
    }

    matmul_result() = default;
    matmul_result(const matmul_result&) = delete;
    matmul_result& operator=(const matmul_result&) = delete;
    matmul_result(matmul_result&&) = default;
    matmul_result& operator=(matmul_result&&) = default;
};

// SCP region: compile-time typed scratchpad metadata
namespace detail {

template<typename R1, typename R2>
struct regions_overlap {
    static constexpr bool value =
        (R1::start < R2::start + R2::max_lines) &&
        (R2::start < R1::start + R1::max_lines);
};

} // namespace detail

template<unsigned Start, unsigned MaxLines, typename ElemType>
struct scp_region {
    static_assert(Start < 48, "SCP start must be < 48");
    static_assert(MaxLines > 0 && MaxLines <= 48, "SCP region: 1-48 lines");
    static_assert(Start + MaxLines <= 48, "SCP region exceeds 48-line limit");

    using elem_type = ElemType;
    using traits    = elem_traits<ElemType>;
    using ptr_t     = const typename traits::c_type*;

    static constexpr unsigned start     = Start;
    static constexpr unsigned max_lines = MaxLines;

    template<unsigned ID>
    [[nodiscard]] inline __attribute__((always_inline))
    auto load(ptr_t addr, unsigned rows, uint64_t stride) const {
        static_assert(ID <= 1, "Load pipeline must be 0 or 1");
        constexpr auto wid = (ID == 0) ? low_level::wait_id::load_0
                                       : low_level::wait_id::load_1;
        low_level::tensor_load(false, false, Start, 0, 0,
                               (uint64_t)addr, 0, rows - 1, stride, ID);
        return wait_token<wid>{};
    }

    template<unsigned ID>
    [[nodiscard]] inline __attribute__((always_inline))
    auto load_transpose(ptr_t addr, unsigned k_elems, uint64_t stride) const {
        static_assert(ID <= 1, "Load pipeline must be 0 or 1");
        constexpr auto wid = (ID == 0) ? low_level::wait_id::load_0
                                       : low_level::wait_id::load_1;
        low_level::tensor_load(false, false, Start, traits::transpose_code, 0,
                               (uint64_t)addr, 0, k_elems - 1, stride, ID);
        return wait_token<wid>{};
    }

    template<unsigned ID>
    [[nodiscard]] inline __attribute__((always_inline))
    auto load_masked(ptr_t addr, unsigned rows, uint64_t stride) const {
        static_assert(ID <= 1, "Load pipeline must be 0 or 1");
        constexpr auto wid = (ID == 0) ? low_level::wait_id::load_0
                                       : low_level::wait_id::load_1;
        low_level::tensor_load(true, false, Start, 0, 0,
                               (uint64_t)addr, 0, rows - 1, stride, ID);
        return wait_token<wid>{};
    }

    [[nodiscard]] inline __attribute__((always_inline))
    store_token store(typename traits::c_type* addr, unsigned rows,
                      uint64_t stride, unsigned entry_stride = 0) const {
        low_level::tensor_store_scp(entry_stride, Start, rows - 1,
                                    (uint64_t)addr, stride);
        return {};
    }
};

// Dispatches to TensorFMA (fp32, fp16) or TensorIMA (int8, int16)
// based on element type. Type mismatch between A and B is a
// compile error.
//
// For integer types, signedness is configurable via template
// parameters AUnsigned / BUnsigned (default: signed).
// These are ignored for floating-point types.

template<
    unsigned OutCols   = 16,
    unsigned FReg      = 0,
    unsigned Step      = 0,
    bool     AUnsigned = false,
    bool     BUnsigned = false,
    unsigned AStart, unsigned ALines, typename ElemA,
    unsigned BStart, unsigned BLines, typename ElemB>
[[nodiscard]] inline __attribute__((always_inline))
auto matmul(
    scp_region<AStart, ALines, ElemA>,
    scp_region<BStart, BLines, ElemB>,
    unsigned a_rows,
    unsigned a_cols,
    bool     first_pass,
    unsigned a_offset = 0)
    -> matmul_result<acc_type_of_t<ElemA>, OutCols, FReg, Step>
{
    static_assert(std::is_same_v<ElemA, ElemB>,
        "A and B element types must match");
    static_assert(!detail::regions_overlap<
        scp_region<AStart, ALines, ElemA>,
        scp_region<BStart, BLines, ElemB>>::value,
        "A and B SCP regions must not overlap");

    constexpr bool is_fp = elem_traits<ElemA>::is_float;

    // Signedness bits are ignored by hardware for FP types,
    // but static_assert if user explicitly passes them to catch mistakes.
    static_assert(is_fp || true,  // always pass — integer path uses them
        "");
    static_assert(!is_fp || (!AUnsigned && !BUnsigned),
        "AUnsigned/BUnsigned are meaningless for floating-point types; "
        "remove them or use default (false)");

    low_level::tensor_fma(
        false, bcols_encode(OutCols),
        a_rows - 1, a_cols - 1, a_offset,
        false,
        is_fp ? false : AUnsigned,
        is_fp ? false : BUnsigned,
        false,
        BStart, AStart,
        elem_traits<ElemA>::fma_opcode,
        first_pass);

    return {};
}

// SCP setup and error handling
inline void setup_l1scp() {
    low_level::fence();
    low_level::ucache_control(true);
    low_level::tensor_wait(low_level::wait_id::cacheops);
}

inline void teardown_l1scp() {
    low_level::fence();
    low_level::ucache_control(false);
    low_level::tensor_wait(low_level::wait_id::cacheops);
}

struct tensor_error {
    uint64_t raw;
    explicit operator bool() const { return raw != 0; }
    bool load_transform() const { return (raw >> 1) & 1; }
    bool fcc_overflow()   const { return (raw >> 3) & 1; }
    bool scp_disabled()   const { return (raw >> 4) & 1; }
    bool lock_sw()        const { return (raw >> 5) & 1; }
    bool fma_pairing()    const { return (raw >> 6) & 1; }
    bool mem_fault()      const { return (raw >> 7) & 1; }
    bool store_coop()     const { return (raw >> 8) & 1; }
    bool reduce_err()     const { return (raw >> 9) & 1; }
};

inline tensor_error get_tensor_error()   { return {low_level::tensor_error()}; }
inline void         clear_tensor_error() { low_level::clear_tensor_error(); }
inline void         fence()              { low_level::fence(); }

} // namespace et
