#include "rwkv_jacobian_lens.hpp"

#include "llama-model.h"

#include <TFile.h>
#include <TTree.h>

#include <algorithm>
#include <cmath>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void quiet_llama_logs(ggml_log_level, const char *, void *) {}

struct branch_result {
    llama_interp::activation_set source;
    std::vector<llama_interp::activation_set> targets;
};

float norm(const std::vector<float> & values) {
    double sum = 0.0;
    for (float value : values) sum += (double) value * value;
    return (float) std::sqrt(sum);
}

float agreement_cosine(const std::vector<float> & left, const std::vector<float> & right) {
    if (left.size() != right.size()) throw std::runtime_error("capture vector size mismatch");
    double dot = 0.0;
    for (size_t i = 0; i < left.size(); ++i) dot += (double) left[i] * right[i];
    const float left_norm = norm(left), right_norm = norm(right);
    if (left_norm == 0.0f && right_norm == 0.0f) return 1.0f;
    if (left_norm == 0.0f || right_norm == 0.0f) return 0.0f;
    return (float) (dot / ((double) left_norm * right_norm));
}

float safe_ratio(float numerator, float denominator) {
    if (denominator != 0.0f) return numerator / denominator;
    return numerator == 0.0f ? 0.0f : std::numeric_limits<float>::max();
}

uint64_t splitmix64(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

const std::vector<float> & captured(const llama_interp::activation_set & captures, const std::string & name) {
    return rwkv_experiment::require_capture(captures, name).data_f32;
}

llama_interp::task<> run_branch(
        llama_interp::runtime & runtime,
        const llama_interp::rwkv_state & before_source,
        llama_token source_token,
        const std::vector<llama_token> & suffix,
        const std::string & source_tap,
        const std::string & target_tap,
        const std::vector<int32_t> & offsets,
        const std::vector<ggml_fp16_t> * perturbation,
        bool capture_source,
        branch_result & output) {
    auto source = runtime.prefill_tokens(before_source, { source_token });
    if (capture_source) source.capture_f32(rwkv_jacobian_lens::exact_regex(source_tap), output.source);
    if (std::binary_search(offsets.begin(), offsets.end(), 0)) {
        source.capture_f32(rwkv_jacobian_lens::exact_regex(target_tap), output.targets[0]);
    }
    if (perturbation) source.perturb(rwkv_jacobian_lens::exact_regex(source_tap), llama_interp::runtime::add_head(-1, *perturbation));
    llama_interp::rwkv_state state = co_await source;
    for (size_t offset = 1; offset <= suffix.size(); ++offset) {
        auto step = runtime.prefill_tokens(state, { suffix[offset - 1] });
        if (std::binary_search(offsets.begin(), offsets.end(), (int32_t) offset)) {
            step.capture_f32(rwkv_jacobian_lens::exact_regex(target_tap), output.targets[offset]);
        }
        state = co_await step;
    }
}

std::vector<ggml_fp16_t> perturbation(const std::vector<float> & direction, float epsilon) {
    std::vector<ggml_fp16_t> result(direction.size());
    for (size_t i = 0; i < direction.size(); ++i) result[i] = ggml_fp32_to_fp16(epsilon * direction[i]);
    return result;
}

llama_interp::rwkv_state prefix_state(
        llama_interp::runtime & runtime,
        const llama_interp::rwkv_state & initial,
        const std::vector<llama_token> & prefix) {
    if (prefix.empty()) return initial;
    llama_interp::rwkv_state result;
    auto call = [&]() -> llama_interp::task<> { result = co_await runtime.prefill_tokens(initial, prefix); }();
    runtime.run();
    call.rethrow_if_failed();
    return result;
}

void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s -m MODEL --corpus FILE --output FILE [--max-windows N] [--window-bytes N]\\n"
        "       [--source-position N] [--relative-epsilon E] [--offset N ...] [-ngl N]\\n",
        argv0);
}

} // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    llama_log_set(quiet_llama_logs, nullptr);

    std::string model_path, corpus_path, output_path;
    uint64_t max_windows = 1000, window_bytes = 512;
    int32_t source_position = 32;
    float relative_epsilon = 0.01f;
    std::vector<int32_t> offsets;
    int n_gpu_layers = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (std::strcmp(argv[i], "--corpus") == 0 && i + 1 < argc) corpus_path = argv[++i];
        else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) output_path = argv[++i];
        else if (std::strcmp(argv[i], "--max-windows") == 0 && i + 1 < argc) max_windows = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "--window-bytes") == 0 && i + 1 < argc) window_bytes = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "--source-position") == 0 && i + 1 < argc) source_position = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--relative-epsilon") == 0 && i + 1 < argc) relative_epsilon = std::strtof(argv[++i], nullptr);
        else if (std::strcmp(argv[i], "--offset") == 0 && i + 1 < argc) offsets.push_back(std::atoi(argv[++i]));
        else if (std::strcmp(argv[i], "-ngl") == 0 && i + 1 < argc) n_gpu_layers = std::atoi(argv[++i]);
        else { usage(argv[0]); return 1; }
    }
    if (model_path.empty() || corpus_path.empty() || output_path.empty() || max_windows == 0 || window_bytes == 0 ||
        source_position < 0 || relative_epsilon <= 0.0f) { usage(argv[0]); return 1; }
    if (offsets.empty()) offsets = { 0, 1, 2, 4 };
    std::sort(offsets.begin(), offsets.end());
    offsets.erase(std::unique(offsets.begin(), offsets.end()), offsets.end());
    if (offsets.front() < 0) throw std::runtime_error("offsets must be non-negative");
    const int32_t max_offset = offsets.back();

    std::ifstream corpus(corpus_path, std::ios::binary);
    if (!corpus) throw std::runtime_error("failed to open corpus: " + corpus_path);
    ggml_backend_load_all();
    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model || model->arch != LLM_ARCH_RWKV7) throw std::runtime_error("an RWKV-7 model is required");
    const int32_t n_layer = model->hparams.n_layer();
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = std::max<int>(128, (int) window_bytes + 8);
    context_params.n_batch = context_params.n_ctx;
    context_params.n_ubatch = context_params.n_ctx;
    llama_context * ctx = llama_init_from_model(model, context_params);
    if (!ctx) throw std::runtime_error("failed to create context");
    llama_interp::runtime runtime(ctx, 1);
    const llama_interp::rwkv_state initial = runtime.make_state();

    TFile output(output_path.c_str(), "RECREATE");
    if (output.IsZombie()) throw std::runtime_error("failed to create ROOT output: " + output_path);
    TTree metadata("metadata", "RWKV Jacobian linearity corpus metadata");
    std::string metadata_model = model_path, metadata_corpus = corpus_path;
    uint64_t metadata_window_bytes = window_bytes, metadata_max_windows = max_windows;
    int32_t metadata_source_position = source_position, metadata_layers = n_layer;
    float metadata_relative_epsilon = relative_epsilon;
    metadata.Branch("model", &metadata_model); metadata.Branch("corpus", &metadata_corpus);
    metadata.Branch("window_bytes", &metadata_window_bytes); metadata.Branch("max_windows", &metadata_max_windows);
    metadata.Branch("source_position", &metadata_source_position); metadata.Branch("offsets", &offsets);
    metadata.Branch("relative_epsilon", &metadata_relative_epsilon); metadata.Branch("n_layer", &metadata_layers);
    metadata.Fill();

    TTree rows("linearity", "one scalar Jacobian-linearity result per window, source layer, and offset");
    uint64_t window = 0, byte_offset = 0;
    int32_t row_layer = 0, row_offset = 0, row_source_token = 0, row_token_count = 0;
    float row_source_norm = 0, row_epsilon = 0, row_min_scale_cosine = 0, row_norm_spread = 0;
    float row_max_center_ratio = 0, row_repeat_cosine = 0;
    bool row_passed = false;
    rows.Branch("window", &window); rows.Branch("byte_offset", &byte_offset); rows.Branch("layer", &row_layer);
    rows.Branch("offset", &row_offset); rows.Branch("source_token", &row_source_token); rows.Branch("token_count", &row_token_count);
    rows.Branch("source_l2_norm", &row_source_norm); rows.Branch("epsilon", &row_epsilon);
    rows.Branch("min_pairwise_scale_cosine", &row_min_scale_cosine); rows.Branch("scale_response_norm_relative_spread", &row_norm_spread);
    rows.Branch("max_center_error_ratio", &row_max_center_ratio); rows.Branch("repeat_1x_cosine", &row_repeat_cosine); rows.Branch("passed", &row_passed);
    rows.SetAutoFlush(1);

    const std::string target_tap = "rwkv.layer." + std::to_string(n_layer - 1) + ".resid.out";
    std::string text(window_bytes, '\0');
    while (window < max_windows && corpus.read(text.data(), (std::streamsize) window_bytes)) {
        const std::vector<llama_token> tokens = common_tokenize(vocab, text, false, true);
        const uint64_t this_byte_offset = byte_offset;
        byte_offset += window_bytes;
        if (source_position + max_offset >= (int32_t) tokens.size()) continue;
        const llama_interp::rwkv_state before_source = prefix_state(runtime, initial,
            std::vector<llama_token>(tokens.begin(), tokens.begin() + source_position));
        const llama_token source_token = tokens[source_position];
        const std::vector<llama_token> suffix(tokens.begin() + source_position + 1,
                                              tokens.begin() + source_position + 1 + max_offset);
        for (int32_t layer = 0; layer < n_layer; ++layer) {
            const std::string source_tap = "rwkv.layer." + std::to_string(layer) + ".resid.in";
            branch_result native;
            native.targets.resize(max_offset + 1);
            auto native_run = run_branch(runtime, before_source, source_token, suffix, source_tap, target_tap, offsets, nullptr, true, native);
            runtime.run(); native_run.rethrow_if_failed();
            const std::vector<float> & source = captured(native.source, source_tap);
            row_source_norm = norm(source);
            if (row_source_norm == 0.0f) throw std::runtime_error("source activation has zero L2 norm");
            std::mt19937_64 generator(splitmix64(this_byte_offset ^ (uint64_t) layer));
            std::uniform_int_distribution<int> sign(0, 1);
            std::vector<float> direction(source.size());
            const float unit = 1.0f / std::sqrt((float) direction.size());
            for (float & value : direction) value = sign(generator) ? unit : -unit;
            const float scales[] = { 0.5f, 1.0f, 2.0f };
            std::vector<branch_result> plus(3), minus(3);
            for (size_t scale = 0; scale < 3; ++scale) {
                plus[scale].targets.resize(max_offset + 1); minus[scale].targets.resize(max_offset + 1);
                const float epsilon = scales[scale] * relative_epsilon * row_source_norm;
                const auto p = perturbation(direction, epsilon), m = perturbation(direction, -epsilon);
                auto plus_run = run_branch(runtime, before_source, source_token, suffix, source_tap, target_tap, offsets, &p, false, plus[scale]);
                runtime.run(); plus_run.rethrow_if_failed();
                auto minus_run = run_branch(runtime, before_source, source_token, suffix, source_tap, target_tap, offsets, &m, false, minus[scale]);
                runtime.run(); minus_run.rethrow_if_failed();
            }
            branch_result repeat_plus, repeat_minus;
            repeat_plus.targets.resize(max_offset + 1); repeat_minus.targets.resize(max_offset + 1);
            const float base_epsilon = relative_epsilon * row_source_norm;
            const auto repeat_p = perturbation(direction, base_epsilon), repeat_m = perturbation(direction, -base_epsilon);
            auto repeat_plus_run = run_branch(runtime, before_source, source_token, suffix, source_tap, target_tap, offsets, &repeat_p, false, repeat_plus);
            runtime.run(); repeat_plus_run.rethrow_if_failed();
            auto repeat_minus_run = run_branch(runtime, before_source, source_token, suffix, source_tap, target_tap, offsets, &repeat_m, false, repeat_minus);
            runtime.run(); repeat_minus_run.rethrow_if_failed();
            for (int32_t offset : offsets) {
                std::vector<float> responses[3];
                float response_norms[3], center_ratios[3];
                const std::vector<float> & target = captured(native.targets[offset], target_tap);
                for (size_t scale = 0; scale < 3; ++scale) {
                    const auto & p = captured(plus[scale].targets[offset], target_tap);
                    const auto & m = captured(minus[scale].targets[offset], target_tap);
                    responses[scale].resize(target.size());
                    double center_sum = 0.0, odd_sum = 0.0;
                    const float epsilon = scales[scale] * base_epsilon;
                    for (size_t i = 0; i < target.size(); ++i) {
                        responses[scale][i] = (p[i] - m[i]) / (2.0f * epsilon);
                        const float center = 0.5f * (p[i] + m[i]) - target[i];
                        const float odd = 0.5f * (p[i] - m[i]);
                        center_sum += (double) center * center; odd_sum += (double) odd * odd;
                    }
                    response_norms[scale] = norm(responses[scale]);
                    center_ratios[scale] = safe_ratio((float) std::sqrt(center_sum), (float) std::sqrt(odd_sum));
                }
                std::vector<float> repeat_response(target.size());
                const auto & rp = captured(repeat_plus.targets[offset], target_tap);
                const auto & rm = captured(repeat_minus.targets[offset], target_tap);
                for (size_t i = 0; i < target.size(); ++i) repeat_response[i] = (rp[i] - rm[i]) / (2.0f * base_epsilon);
                row_min_scale_cosine = std::min({ agreement_cosine(responses[0], responses[1]), agreement_cosine(responses[0], responses[2]), agreement_cosine(responses[1], responses[2]) });
                const auto [minimum, maximum] = std::minmax_element(response_norms, response_norms + 3);
                row_norm_spread = *maximum == 0.0f ? 0.0f : (*maximum - *minimum) / *maximum;
                row_max_center_ratio = std::max({ center_ratios[0], center_ratios[1], center_ratios[2] });
                row_repeat_cosine = agreement_cosine(responses[1], repeat_response);
                row_passed = row_min_scale_cosine >= 0.995f && row_norm_spread <= 0.05f && row_max_center_ratio <= 0.05f && row_repeat_cosine >= 0.999f;
                row_layer = layer; row_offset = offset; row_source_token = source_token; row_token_count = (int32_t) tokens.size(); row_epsilon = base_epsilon;
                rows.Fill();
            }
            // A full all-layer window is expensive; persist completed layers independently.
            rows.AutoSave("SaveSelf");
            std::printf("window=%llu layer=%d/%d rows=%lld\n", (unsigned long long) window, layer + 1, n_layer, rows.GetEntries());
            std::fflush(stdout);
        }
        ++window;
        rows.AutoSave("SaveSelf");
        std::printf("windows=%llu rows=%lld\n", (unsigned long long) window, rows.GetEntries());
        std::fflush(stdout);
    }
    metadata.Write(); rows.Write(); output.Close();
    llama_free(ctx); llama_model_free(model); llama_backend_free();
    return window == max_windows ? 0 : 2;
}
