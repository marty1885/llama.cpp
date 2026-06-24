#include "wkv7.hpp"

#include <algorithm>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include "tt-metalium/host_api.hpp"
#include "tt-metalium/kernel_types.hpp"
#include "tt-metalium/tt_backend_api_types.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <ttnn/tensor/layout/layout.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "utils.hpp"

using namespace tt::tt_metal;

namespace ttggml {
using namespace ttnn;

// Dispatch (there are 2 kernels)
//   decodeL  -> sequential per-token. Faster for decode
//   chunked  -> Higher throughput large decode

// Switch the WKV7 state input/reader between the canonical flat-strip layout (default) and the
// row-folded layout [1,G,Es/32,32]. Must agree across invoke() (G derivation), create() (reader
// define) and the ggml-metalium handler (which folds/passes the state accordingly).
bool wkv7_folded_state() {
    static const bool v = std::getenv("GGML_METALIUM_WKV7_FOLDED_STATE") != nullptr;
    return v;
}

namespace {
constexpr uint32_t TW = 32, TH = 32;

// One bf16 32x32 tile CB of n tiles (matches the runner's MakeCB exactly).
inline void MakeCB(Program& p, const CoreRangeSet& cr, tt::CBIndex cb, uint32_t n) {
    const uint32_t ts = sizeof(bfloat16) * TW * TH;
    CircularBufferConfig c = CircularBufferConfig(n * ts, {{cb, tt::DataFormat::Float16_b}}).set_page_size(cb, ts);
    CreateCircularBuffer(p, cr, c);
}
} // namespace

wkv7_device::RWKVWKV7DeviceOperation::program_factory_t
wkv7_device::RWKVWKV7DeviceOperation::select_program_factory(
    const operation_attributes_t& /*attrs*/,
    const tensor_args_t& /*tensor_args*/)
{
    return program::WKV7ProgramFactory{};
}

wkv7_device::RWKVWKV7DeviceOperation::spec_return_value_t
wkv7_device::RWKVWKV7DeviceOperation::compute_output_specs(
    const operation_attributes_t& a,
    const tensor_args_t& /*tensor_args*/)
{
    const uint32_t C    = a.S * a.H;
    const uint32_t T    = a.L * a.G;
    const uint32_t rows = T + a.S * a.G;            // region-1 tokens + region-2 final state
    // ggml output ne=[C, rows] -> TTNN 4-D [1,1,rows,C] (ggml_tt_tensors_shape_equal
    // matches ne[3-i] against shape[i]; only the inner two dims are tiled).
    return TensorSpec(
        ttnn::Shape({1, 1, rows, C}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            a.output_mem_config));
}

wkv7_device::RWKVWKV7DeviceOperation::tensor_return_value_t
wkv7_device::RWKVWKV7DeviceOperation::create_output_tensors(
    const operation_attributes_t& a,
    const tensor_args_t& tensor_args)
{
    auto spec = compute_output_specs(a, tensor_args);
    return create_device_tensor(spec, tensor_args.r.device());
}

void wkv7_device::RWKVWKV7DeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& a,
    const tensor_args_t& t)
{
    TT_FATAL(t.r.storage_type() == tt::tt_metal::StorageType::DEVICE, "WKV7 inputs must be on device");
    TT_FATAL(t.r.layout() == tt::tt_metal::Layout::TILE, "WKV7 inputs must be TILE layout");
    TT_FATAL(a.S == 64, "WKV7 metalium kernel currently requires head_size == 64, got {}", a.S);
    TT_FATAL(a.S % a.H == 0, "WKV7 requires head_count | head_size (got H={}, S={})", a.H, a.S);
    if (a.L >= 2 && !a.use_decode) {
        TT_FATAL((a.G * a.H) % 2 == 0, "WKV7 chunked kernel requires (n_seqs*head_count) even");
    }
}

void wkv7_device::RWKVWKV7DeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& a,
    const tensor_args_t& t)
{
    validate_on_program_cache_miss(a, t);
}

std::tuple<wkv7_device::operation_attributes_t, wkv7_device::tensor_args_t>
wkv7_device::RWKVWKV7DeviceOperation::invoke(
    const Tensor& r,
    const Tensor& w,
    const Tensor& k,
    const Tensor& v,
    const Tensor& a,
    const Tensor& b,
    const Tensor& state)
{
    // Inputs are TTNN [T, H, S] (ggml ne=[S,H,T]); state TTNN [G, S*S*H] (ggml ne=[S*S*H,G]).
    const auto& rs = r.logical_shape();
    const uint32_t S = rs[-1];
    const uint32_t H = rs[-2];
    const uint32_t T = rs[-3];
    // Canonical state TTNN [1,1,G,Es] -> G at [-2]; folded state TTNN [1,G,Es/32,32] -> G at [-3].
    const uint32_t G = wkv7_folded_state() ? state.logical_shape()[-3] : state.logical_shape()[-2];
    const uint32_t L = T / G;
    return {
        operation_attributes_t{
            r.memory_config(),
            S, H, L, G,
            // Dispatch (decodeL vs chunked), from the measured data
            (L == 1)
                || (G == 1 && L <= 16)
                || (G <= 8 && L <= 7)
                || (L <= 4),
        },
        tensor_args_t{r, w, k, v, a, b, state}
    };
}

// ---- WKV7ProgramFactory ----

wkv7_device::program::WKV7ProgramFactory::cached_program_t
wkv7_device::program::WKV7ProgramFactory::create(
    const operation_attributes_t& attrs,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value)
{
    Program prog = CreateProgram();
    using CB = tt::CBIndex;

    const int64_t S = attrs.S, H = attrs.H, L = attrs.L, G = attrs.G, C = S * H;
    const uint32_t St = S / TW, Ht = H / TH, NS = St * St;
    const bool decode = attrs.use_decode;

    // cl/nc: chunked & decode chunk by 32 (partial for L<32, on-chip carry for L>32).
    const int64_t cl = (L < 32) ? L : 32;
    const uint32_t nc = (uint32_t)((L + cl - 1) / cl);
    const uint32_t nin = (uint32_t)(cl * St);
    const uint32_t IC = (uint32_t)(G * H);
    const uint32_t Ct = (uint32_t)(C / 32);
    const int64_t T = L * G;

    IDevice* device = tensor_args.r.device();
    auto grid = device->compute_with_storage_grid_size();

    // Inputs in kernel order [a,w,k,v,r,b] (ggml src [4,1,2,3,0,5]); state separate.
    const std::array<const Tensor*, 6> in_t = {
        &tensor_args.a, &tensor_args.w, &tensor_args.k, &tensor_args.v, &tensor_args.r, &tensor_args.b};
    auto* st_buf  = tensor_args.state.buffer();
    auto* out_buf = tensor_return_value.buffer();

    constexpr uint32_t NBg = 2;             // chunked group size (reader/compute hardcode NB=2)
    const uint32_t P = 4;                   // decode instance-packing
    // SPMD split via the tt-metal primitive. One unit of work = one decode instance (decodeL)
    // or one NBg-instance group (chunked, so each core gets whole NB=2 groups). split_work_to_cores
    // gives core_group_1 the larger share (units_per_core_1) and core_group_2 the remainder
    // (units_per_core_1 - 1); the two groups together are exactly `core` (= all used cores).
    const uint32_t num_units = decode ? IC : (IC / NBg);
    auto [num_cores, core, core_group_1, core_group_2, units_per_core_1, units_per_core_2] =
        split_work_to_cores(grid, num_units);

    if (!decode) {
        for (int i = 0; i < 6; i++) MakeCB(prog, core, (CB)(CB::c_0 + i), St * 8);
        MakeCB(prog, core, CB::c_6, NS * 4);
        MakeCB(prog, core, CB::c_7, 1);
        MakeCB(prog, core, CB::c_8, St * 8); MakeCB(prog, core, CB::c_9, St * 8); MakeCB(prog, core, CB::c_10, St * 8);
        MakeCB(prog, core, CB::c_11, St * 8); MakeCB(prog, core, CB::c_12, St * 8); MakeCB(prog, core, CB::c_13, St * 8);
        MakeCB(prog, core, CB::c_14, St * 8); MakeCB(prog, core, CB::c_15, St * 8); MakeCB(prog, core, CB::c_16, (St + NS) * 2);
        MakeCB(prog, core, CB::c_17, St * 8); MakeCB(prog, core, CB::c_18, St * 8); MakeCB(prog, core, CB::c_19, 1); MakeCB(prog, core, CB::c_20, 1);
        MakeCB(prog, core, CB::c_21, std::max(nin, 32u));
        MakeCB(prog, core, CB::c_22, std::max((uint32_t)cl, 32u));
        MakeCB(prog, core, CB::c_23, 4); MakeCB(prog, core, CB::c_24, 4); MakeCB(prog, core, CB::c_25, 4); MakeCB(prog, core, CB::c_26, 4); MakeCB(prog, core, CB::c_27, 4);
        MakeCB(prog, core, CB::c_28, 4); MakeCB(prog, core, CB::c_29, NBg * NS); MakeCB(prog, core, CB::c_30, St * 4); MakeCB(prog, core, CB::c_31, St * 4);
    } else {
        for (int i = 0; i < 6; i++) MakeCB(prog, core, (CB)(CB::c_0 + i), P * St);
        MakeCB(prog, core, CB::c_6, P * NS);
        MakeCB(prog, core, CB::c_7, 1);
        MakeCB(prog, core, CB::c_8, P * St); MakeCB(prog, core, CB::c_9, P * NS); MakeCB(prog, core, CB::c_10, P * St);
        MakeCB(prog, core, CB::c_11, St); MakeCB(prog, core, CB::c_12, 2); MakeCB(prog, core, CB::c_13, 2);
        MakeCB(prog, core, CB::c_14, 2); MakeCB(prog, core, CB::c_15, 2); MakeCB(prog, core, CB::c_16, (St + NS) * 2);
        MakeCB(prog, core, CB::c_17, 2); MakeCB(prog, core, CB::c_18, 2); MakeCB(prog, core, CB::c_19, 1); MakeCB(prog, core, CB::c_20, 1);
        MakeCB(prog, core, CB::c_21, 32u);
        MakeCB(prog, core, CB::c_22, 2u);
        MakeCB(prog, core, CB::c_23, 2); MakeCB(prog, core, CB::c_24, 2); MakeCB(prog, core, CB::c_25, 2); MakeCB(prog, core, CB::c_26, 2); MakeCB(prog, core, CB::c_27, 2);
        MakeCB(prog, core, CB::c_28, 4); MakeCB(prog, core, CB::c_29, NBg * NS); MakeCB(prog, core, CB::c_30, St * 2); MakeCB(prog, core, CB::c_31, St * 2);
    }

    std::vector<uint32_t> rct, wct;
    TensorAccessorArgs(*in_t[0]->buffer()).append_to(rct);
    TensorAccessorArgs(*st_buf).append_to(rct);
    TensorAccessorArgs(*out_buf).append_to(wct);

    std::map<std::string, std::string> reader_defines;
    if (wkv7_folded_state()) reader_defines["WKV7_STATE_FOLDED"] = "1";
    KernelHandle reader = CreateMetaliumKernel(prog, decode ? "wkv7_decodeL_reader" : "wkv7_reader", core, DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = rct, .defines = reader_defines});
    KernelHandle writer = CreateMetaliumKernel(prog, "wkv7_writer", core, DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = wct});

    KernelHandle compute;
    if (!decode) {
        ComputeConfig cc{.math_fidelity = MathFidelity::HiFi4};
        cc.fp32_dest_acc_en = true;
        compute = CreateMetaliumKernel(prog, "wkv7_chunked_compute", core, cc);
    } else {
        ComputeConfig cc{.math_fidelity = MathFidelity::HiFi4};
        compute = CreateMetaliumKernel(prog, "wkv7_decodeL_compute", core, cc);
    }

    std::vector<uint32_t> ra{(uint32_t)H, (uint32_t)cl, St, Ht, (uint32_t)G};
    for (int i = 0; i < 6; i++) ra.push_back((uint32_t)in_t[i]->buffer()->address());
    ra.push_back((uint32_t)st_buf->address());
    ra.push_back((uint32_t)L);   // Lreal: real tokens/seq (reader neutral-pads the partial last chunk)
    ra.push_back(0u);   // sels_addr  (unused)
    ra.push_back(0u);   // cst_addr   (unused: consts SFPU-generated)
    ra.push_back(nc);

    const uint32_t writer_tpc = decode ? 1u : 32u;
    const uint32_t writer_NB  = decode ? 1u : NBg;
    const uint32_t writer_nc  = decode ? (uint32_t)L : nc;

    // Walk the used cores in order, handing each a contiguous run of units; convert the unit
    // run to an instance range (x NBg for chunked groups, x1 for decode). Which physical core
    // gets which slice is irrelevant -- the kernels are identical and only the inst range
    // differs; we just need a clean partition of [0, IC).
    const uint32_t unit_to_inst = decode ? 1u : NBg;
    const std::vector<CoreCoord> cores = corerange_to_cores(core, num_cores);
    uint32_t units_done = 0;
    for (uint32_t kk = 0; kk < num_cores; kk++) {
        const CoreCoord cc2 = cores[kk];
        const uint32_t units_here = core_group_1.contains(cc2) ? units_per_core_1 : units_per_core_2;
        const uint32_t inst_start = units_done * unit_to_inst;
        const uint32_t inst_end   = (units_done + units_here) * unit_to_inst;
        units_done += units_here;
        auto rak = ra; rak.push_back(inst_start); rak.push_back(inst_end);
        SetRuntimeArgs(prog, reader, cc2, rak);
        if (!decode)
            SetRuntimeArgs(prog, compute, cc2, std::vector<uint32_t>{St, IC, (uint32_t)cl, Ht, nc, inst_start, inst_end});
        else
            SetRuntimeArgs(prog, compute, cc2, std::vector<uint32_t>{St, IC, (uint32_t)cl, Ht, nc, inst_start, inst_end, P});
        SetRuntimeArgs(prog, writer, cc2, std::vector<uint32_t>{
            (uint32_t)out_buf->address(), inst_start, inst_end, IC, writer_nc, (uint32_t)H, St, Ct, (uint32_t)T, (uint32_t)L, writer_tpc, writer_NB});
    }

    return {std::move(prog), shared_variables_t{reader, writer, core}};
}

void wkv7_device::program::WKV7ProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*attrs*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value)
{
    auto& program = cached_program.program;
    const auto& reader = cached_program.shared_variables.reader;
    const auto& writer = cached_program.shared_variables.writer;
    const auto& all_cores = cached_program.shared_variables.all_cores;

    const std::array<const Tensor*, 6> in_t = {
        &tensor_args.a, &tensor_args.w, &tensor_args.k, &tensor_args.v, &tensor_args.r, &tensor_args.b};
    auto* st_buf  = tensor_args.state.buffer();
    auto* out_buf = tensor_return_value.buffer();

    for (const auto& range : all_cores.ranges()) {
        for (const auto& core : range) {
            {
                auto& ra = GetRuntimeArgs(program, reader, core);
                for (int i = 0; i < 6; i++) ra[5 + i] = (uint32_t)in_t[i]->buffer()->address();
                ra[11] = (uint32_t)st_buf->address();
            }
            {
                auto& wa = GetRuntimeArgs(program, writer, core);
                wa[0] = (uint32_t)out_buf->address();
            }
        }
    }
}

ttnn::Tensor rwkv_wkv7(
    const Tensor& r,
    const Tensor& w,
    const Tensor& k,
    const Tensor& v,
    const Tensor& a,
    const Tensor& b,
    const Tensor& state)
{
    auto [attrs, args] = wkv7_device::RWKVWKV7DeviceOperation::invoke(r, w, k, v, a, b, state);
    return ttnn::device_operation::detail::launch<wkv7_device::RWKVWKV7DeviceOperation>(attrs, args);
}

} // namespace ttggml
