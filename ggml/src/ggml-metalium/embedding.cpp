#include "embedding.hpp"
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/kernel_types.hpp"
#include <ttnn/device_operation.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <ttnn/operations/creation/creation.hpp>
#include "ttnn/types.hpp"
#include "utils.hpp"

#include <variant>

using namespace tt::tt_metal;

namespace {

// Folds canonical [1,1,vocab,embed] block-float into bf16 [1,vocab,embed/32,32]. The
// dataflow is one canonical tile per work item: reader streams the tile, compute casts it
// to bf16, writer scatters its 32 vocab rows to 32 distinct variant tiles.
struct EmbeddingFoldDeviceOperation {
    struct operation_attributes_t {};
    struct tensor_args_t {
        const Tensor& a;
    };
    using spec_return_value_t = ttnn::TensorSpec;
    using tensor_return_value_t = Tensor;

    struct ProgramFactory {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle reader;
            tt::tt_metal::KernelHandle writer;
            tt::tt_metal::CoreRangeSet all_cores;
        };
        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
        static void override_runtime_arguments(
            cached_program_t&, const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
    };

    using program_factory_t = std::variant<ProgramFactory>;
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&) {
        return ProgramFactory{};
    }
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t& ta) {
        const auto s = ta.a.logical_shape().to_array_4D();
        TT_FATAL(ta.a.layout() == ttnn::TILE_LAYOUT, "embedding fold: canonical must be TILE");
        TT_FATAL(s[0] == 1 && s[1] == 1, "embedding fold: expected [1,1,vocab,embed], got {}", ta.a.logical_shape());
        TT_FATAL(s[2] % 32 == 0 && s[3] % 32 == 0, "embedding fold: vocab and embed must be tile aligned (got {})", ta.a.logical_shape());
    }
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t& ta) {
        const auto s = ta.a.logical_shape().to_array_4D();
        const uint32_t vocab = s[2];
        const uint32_t embed = s[3];
        ttnn::Shape out_shape({1, vocab, embed / 32, 32});
        return TensorSpec(out_shape, tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(ttnn::TILE_LAYOUT),
            ta.a.memory_config()));
    }
    static tensor_return_value_t create_output_tensors(const operation_attributes_t& attr, const tensor_args_t& ta) {
        // Zero-init so the padding rows so we don't need to do that manually in the kernel.
        auto spec = compute_output_specs(attr, ta);
        auto z = ttnn::zeros(spec.logical_shape(), tt::tt_metal::DataType::BFLOAT16, ttnn::TILE_LAYOUT);
        return z.to_device(ta.a.device());
    }
};

EmbeddingFoldDeviceOperation::ProgramFactory::cached_program_t
EmbeddingFoldDeviceOperation::ProgramFactory::create(
    const operation_attributes_t&, const tensor_args_t& ta, tensor_return_value_t& out) {
    tt::tt_metal::Program program{};
    const auto& a_tensor = ta.a;

    const auto s = a_tensor.logical_shape().to_array_4D();
    const uint32_t vocab = s[2], embed = s[3];
    const uint32_t vocab_t = vocab / 32;
    const uint32_t embed_t = embed / 32;
    const uint32_t Tev = (embed_t + 31) / 32;  // variant tiles per vocab = ceil(embed/1024)

    IDevice* device = a_tensor.device();
    auto* a = a_tensor.buffer();
    auto* o = out.buffer();

    auto core_grid = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, cg1, cg2, work1, work2] =
        tt::tt_metal::split_work_to_cores(core_grid, vocab_t * embed_t);

    MakeCircularBuffer(program, all_cores, tt::CBIndex::c_0, 4, a_tensor.dtype());     // canonical block-float
    MakeCircularBuffer(program, all_cores, tt::CBIndex::c_16, 4, out.dtype());          // bf16 cast

    std::vector<uint32_t> reader_ct;
    TensorAccessorArgs(*a).append_to(reader_ct);
    KernelHandle reader = CreateMetaliumKernel(program, "embedding_fold_reader", all_cores, DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = reader_ct});

    std::vector<uint32_t> writer_ct;
    TensorAccessorArgs(*o).append_to(writer_ct);
    KernelHandle writer = CreateMetaliumKernel(program, "embedding_fold_writer", all_cores, DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = writer_ct});

    KernelHandle compute = CreateMetaliumKernel(program, "embedding_fold_compute", all_cores, ComputeConfig{});

    uint32_t id = 0;
    for (const auto& [group, work] : {std::make_pair(cg1, work1), std::make_pair(cg2, work2)}) {
        for (const auto& range : group.ranges()) {
            for (const auto& core : range) {
                SetRuntimeArgs(program, reader,  core, std::vector<uint32_t>{a->address(), id, work});
                SetRuntimeArgs(program, compute, core, std::vector<uint32_t>{work});
                SetRuntimeArgs(program, writer,  core, std::vector<uint32_t>{o->address(), embed_t, Tev, id, work});
                id += work;
            }
        }
    }
    return {std::move(program), {reader, writer, all_cores}};
}

void EmbeddingFoldDeviceOperation::ProgramFactory::override_runtime_arguments(
    cached_program_t& cp, const operation_attributes_t&, const tensor_args_t& ta, tensor_return_value_t& out) {
    auto& program = cp.program;
    auto* a = ta.a.buffer();
    auto* o = out.buffer();
    for (const auto& range : cp.shared_variables.all_cores.ranges()) {
        for (const auto& core : range) {
            GetRuntimeArgs(program, cp.shared_variables.reader, core)[0] = a->address();
            GetRuntimeArgs(program, cp.shared_variables.writer, core)[0] = o->address();
        }
    }
}

// Gathers variant bands by token index. Output [1, n_tokens, embed/32, 32].
struct EmbeddingGatherDeviceOperation {
    struct operation_attributes_t { uint32_t embed; };
    struct tensor_args_t { const Tensor& variant; const Tensor& index; };
    using spec_return_value_t = ttnn::TensorSpec;
    using tensor_return_value_t = Tensor;

    struct ProgramFactory {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle reader;
            tt::tt_metal::KernelHandle writer;
            tt::tt_metal::CoreRangeSet all_cores;
        };
        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;
        static cached_program_t create(
            const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
        static void override_runtime_arguments(
            cached_program_t&, const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
    };
    using program_factory_t = std::variant<ProgramFactory>;
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&) {
        return ProgramFactory{};
    }
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t& ta) {
        TT_FATAL(ta.variant.layout() == ttnn::TILE_LAYOUT, "embedding gather: variant must be TILE");
        TT_FATAL(ta.index.layout() == ttnn::ROW_MAJOR_LAYOUT, "embedding gather: index must be ROW_MAJOR");
        TT_FATAL(ta.index.dtype() == tt::tt_metal::DataType::UINT32, "embedding gather: index must be UINT32");
    }
    static spec_return_value_t compute_output_specs(const operation_attributes_t& attr, const tensor_args_t& ta) {
        const auto idx = ta.index.logical_shape().to_array_4D();
        const uint32_t n_tokens = idx[2] * idx[3];  // users * n
        ttnn::Shape out_shape({1, 1, n_tokens, attr.embed});  // ggml-canonical [1,1,n_tokens,embed]
        return TensorSpec(out_shape, tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(ttnn::TILE_LAYOUT),
            ta.variant.memory_config()));
    }
    static tensor_return_value_t create_output_tensors(const operation_attributes_t& attr, const tensor_args_t& ta) {
        return create_device_tensor(compute_output_specs(attr, ta), ta.variant.device());
    }
};

EmbeddingGatherDeviceOperation::ProgramFactory::cached_program_t
EmbeddingGatherDeviceOperation::ProgramFactory::create(
    const operation_attributes_t&, const tensor_args_t& ta, tensor_return_value_t& out) {
    tt::tt_metal::Program program{};
    const auto s = ta.variant.logical_shape().to_array_4D();
    const auto idx = ta.index.logical_shape().to_array_4D();
    const uint32_t embed_t = s[2];
    const uint32_t Tev = (embed_t + 31) / 32;
    const uint32_t n = idx[3];                 // ids per user (index last dim)
    const uint32_t n_tokens = idx[2] * idx[3]; // users * n

    IDevice* device = ta.variant.device();
    auto* vbuf = ta.variant.buffer();
    auto* idxbuf = ta.index.buffer();
    auto* obuf = out.buffer();

    auto core_grid = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, cg1, cg2, work1, work2] =
        tt::tt_metal::split_work_to_cores(core_grid, n_tokens);

    MakeCircularBuffer(program, all_cores, tt::CBIndex::c_0, 4, ta.variant.dtype());
    MakeCircularBuffer(program, all_cores, tt::CBIndex::c_16, 4, out.dtype());
    // c_1: L1 scratch for one index page (n uint32). Reused across a user's tokens.
    MakeCircularBuffer(program, all_cores, tt::CBIndex::c_1, n * 4, n * 4, tt::DataFormat::UInt32);

    std::vector<uint32_t> reader_ct;
    TensorAccessorArgs(*vbuf).append_to(reader_ct);
    TensorAccessorArgs(*idxbuf).append_to(reader_ct);
    KernelHandle reader = CreateMetaliumKernel(program, "embedding_gather_reader", all_cores, DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = reader_ct});

    std::vector<uint32_t> writer_ct;
    TensorAccessorArgs(*obuf).append_to(writer_ct);
    KernelHandle writer = CreateMetaliumKernel(program, "embedding_gather_writer", all_cores, DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = writer_ct});

    KernelHandle compute = CreateMetaliumKernel(program, "embedding_fold_compute", all_cores, ComputeConfig{});

    uint32_t id = 0;
    for (const auto& [group, work] : {std::make_pair(cg1, work1), std::make_pair(cg2, work2)}) {
        for (const auto& range : group.ranges()) {
            for (const auto& core : range) {
                SetRuntimeArgs(program, reader,  core, std::vector<uint32_t>{vbuf->address(), idxbuf->address(), Tev, n, id, work});
                SetRuntimeArgs(program, compute, core, std::vector<uint32_t>{work * Tev});
                SetRuntimeArgs(program, writer,  core, std::vector<uint32_t>{obuf->address(), embed_t, Tev, id, work});
                id += work;
            }
        }
    }
    return {std::move(program), {reader, writer, all_cores}};
}

void EmbeddingGatherDeviceOperation::ProgramFactory::override_runtime_arguments(
    cached_program_t& cp, const operation_attributes_t&, const tensor_args_t& ta, tensor_return_value_t& out) {
    auto& program = cp.program;
    for (const auto& range : cp.shared_variables.all_cores.ranges()) {
        for (const auto& core : range) {
            auto& ra = GetRuntimeArgs(program, cp.shared_variables.reader, core);
            ra[0] = ta.variant.buffer()->address();
            ra[1] = ta.index.buffer()->address();
            GetRuntimeArgs(program, cp.shared_variables.writer, core)[0] = out.buffer()->address();
        }
    }
}

} // namespace

ttnn::Tensor ttggml::EmbeddingFoldVariant::invoke(const Tensor& canonical) {
    return ttnn::device_operation::launch<EmbeddingFoldDeviceOperation>(
        EmbeddingFoldDeviceOperation::operation_attributes_t{},
        EmbeddingFoldDeviceOperation::tensor_args_t{canonical});
}

ttnn::Tensor ttggml::EmbeddingGather::invoke(const Tensor& variant, const Tensor& index, uint32_t embed) {
    return ttnn::device_operation::launch<EmbeddingGatherDeviceOperation>(
        EmbeddingGatherDeviceOperation::operation_attributes_t{embed},
        EmbeddingGatherDeviceOperation::tensor_args_t{variant, index});
}
