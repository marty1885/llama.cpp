#include "rwkv_activation_store.h"
#include "rwkv_experiment.hpp"

#include "llama-model.h"

#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleWriter.hxx>
#include <TFile.h>
#include <openssl/sha.h>

#include <array>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <cmath>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int k_layers[] = { 15, 30, 45, 60 };

struct behavior_row {
    uint64_t prompt_id;
    int32_t expected_value_token_id;
    double expected_value_log_probability;
    int32_t expected_value_rank;
    int32_t native_top1_token_id;
    bool native_top1_correct;
};

struct sample_row {
    uint64_t prompt_id;
    int32_t carrier_id;
    int32_t carrier_token_id;
    bool carrier_train;
    int32_t value_id;
    int32_t value_token_id;
    int32_t layer;
    int32_t read_position;
    int32_t read_token_id;
    double residual_identity_error;
};

void quiet_llama_logs(ggml_log_level, const char *, void *) {}

void usage(const char * argv0) {
    std::fprintf(stderr, "usage: %s --self-test | -m MODEL --prompts FILE --output ROOT [-ngl N] [--max-prompts N] [--development-run | --classified --manifest MANIFEST.json --registration REGISTRATION.json]\n", argv0);
}

std::string read_text(const std::string & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot read " + path);
    return { std::istreambuf_iterator<char>(input), {} };
}

std::string sha256_file(const std::string & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot read " + path);
    SHA256_CTX context;
    SHA256_Init(&context);
    std::array<char, 1024 * 1024> chunk{};
    while (input.read(chunk.data(), chunk.size()) || input.gcount() != 0) SHA256_Update(&context, chunk.data(), size_t(input.gcount()));
    std::array<unsigned char, SHA256_DIGEST_LENGTH> digest{};
    SHA256_Final(digest.data(), &context);
    static constexpr char hex[] = "0123456789abcdef";
    std::string result(64, '0');
    for (size_t i = 0; i < digest.size(); ++i) { result[2 * i] = hex[digest[i] >> 4]; result[2 * i + 1] = hex[digest[i] & 15]; }
    return result;
}

std::string json_string(const std::string & text, const char * key) {
    std::smatch match;
    if (!std::regex_search(text, match, std::regex(std::string("\\\"") + key + "\\\"\\s*:\\s*\\\"([^\\\"]+)\\\""))) {
        throw std::runtime_error(std::string("JSON lacks ") + key);
    }
    return match[1];
}

int32_t json_integer(const std::string & text, const char * key) {
    std::smatch match;
    if (!std::regex_search(text, match, std::regex(std::string("\\\"") + key + "\\\"\\s*:\\s*([0-9]+)"))) {
        throw std::runtime_error(std::string("JSON lacks ") + key);
    }
    return int32_t(std::stol(match[1]));
}

void verify_registration(const std::string & registration_path, const std::string & manifest_path,
                         const std::string & prompts_path, const std::string & model_path, int n_gpu_layers) {
    const std::string registration = read_text(registration_path);
    if (json_string(registration, "status") != "frozen" || json_string(registration, "backend") != "Vulkan" ||
        json_integer(registration, "n_gpu_layers") != 99 || n_gpu_layers != 99 ||
        json_string(registration, "manifest_sha256") != sha256_file(manifest_path) ||
        json_string(registration, "corpus_sha256") != sha256_file(prompts_path) ||
        json_string(registration, "model_sha256") != sha256_file(model_path)) {
        throw std::runtime_error("frozen registration does not match collector inputs");
    }
}

behavior_row summarize_behavior(uint64_t prompt_id, const std::vector<llama_token> & tokens, size_t value_position,
                               const float * logits, int32_t vocabulary_size) {
    if (!logits || tokens.size() <= value_position) throw std::runtime_error("missing native logits or controlled value token");
    const int32_t expected = tokens[value_position];
    if (expected < 0 || expected >= vocabulary_size) throw std::runtime_error("controlled value token outside vocabulary");
    float maximum = -INFINITY;
    int32_t top1 = 0;
    for (int32_t token = 0; token < vocabulary_size; ++token) if (logits[token] > maximum) { maximum = logits[token]; top1 = token; }
    double sum = 0.0;
    int32_t rank = 1;
    for (int32_t token = 0; token < vocabulary_size; ++token) {
        sum += std::exp(double(logits[token]) - maximum);
        if (logits[token] > logits[expected]) ++rank;
    }
    return { prompt_id, expected, double(logits[expected]) - maximum - std::log(sum), rank, top1, top1 == expected };
}

} // namespace

int main(int argc, char ** argv) try {
    llama_log_set(quiet_llama_logs, nullptr);
    std::string model_path, prompts_path, output_path, manifest_path, registration_path;
    int n_gpu_layers = 0;
    uint64_t max_prompts = 768;
    bool self_test = false, development = false, classified = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--self-test")) self_test = true;
        else if (!std::strcmp(argv[i], "-m") && i + 1 < argc) model_path = argv[++i];
        else if (!std::strcmp(argv[i], "--prompts") && i + 1 < argc) prompts_path = argv[++i];
        else if (!std::strcmp(argv[i], "--output") && i + 1 < argc) output_path = argv[++i];
        else if (!std::strcmp(argv[i], "--manifest") && i + 1 < argc) manifest_path = argv[++i];
        else if (!std::strcmp(argv[i], "--registration") && i + 1 < argc) registration_path = argv[++i];
        else if (!std::strcmp(argv[i], "--development-run")) development = true;
        else if (!std::strcmp(argv[i], "--classified")) classified = true;
        else if (!std::strcmp(argv[i], "-ngl") && i + 1 < argc) n_gpu_layers = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--max-prompts") && i + 1 < argc) max_prompts = std::strtoull(argv[++i], nullptr, 10);
        else { usage(argv[0]); return 1; }
    }
    if (self_test) {
        std::vector<llama_token> tokens(10, 0);
        tokens[9] = 3;
        const std::vector<float> logits{ -2.0f, 0.0f, 1.0f, 4.0f };
        const behavior_row result = summarize_behavior(7, tokens, 9, logits.data(), int32_t(logits.size()));
        if (result.prompt_id != 7 || result.expected_value_token_id != 3 || result.expected_value_rank != 1 ||
            result.native_top1_token_id != 3 || !result.native_top1_correct || !(result.expected_value_log_probability < 0.0)) {
            throw std::runtime_error("behavior self-test failed");
        }
        std::puts("self-tests=passed");
        return 0;
    }
    if (development == classified || model_path.empty() || prompts_path.empty() || output_path.empty() || std::filesystem::exists(output_path) ||
        (classified && (manifest_path.empty() || registration_path.empty() || max_prompts != 768))) {
        usage(argv[0]); return 1;
    }
    if (classified) verify_registration(registration_path, manifest_path, prompts_path, model_path, n_gpu_layers);
    std::array<int32_t, 16> value_tokens{};
    std::array<int32_t, 48> carrier_tokens{};
    int32_t read_position = 0, read_token_id = 0, carrier_position = 0, value_position = 0;
    if (classified) {
        const std::string manifest = read_text(manifest_path);
        if (json_string(manifest, "status") != "frozen" || json_integer(manifest, "prompt_count") != 768) {
            throw std::runtime_error("manifest is not a frozen controlled-value factorial");
        }
        read_position = json_integer(manifest, "read_position");
        read_token_id = json_integer(manifest, "read_token_id");
        carrier_position = json_integer(manifest, "carrier_position");
        value_position = json_integer(manifest, "value_position");
        std::vector<int32_t> tokens;
        const std::regex token_pattern("\\\"token_id\\\"\\s*:\\s*([0-9]+)");
        for (std::sregex_iterator it(manifest.begin(), manifest.end(), token_pattern), end; it != end; ++it) {
            tokens.push_back(int32_t(std::stol((*it)[1])));
        }
        if (tokens.size() != 64) throw std::runtime_error("manifest token schema is incomplete");
        std::copy_n(tokens.begin(), value_tokens.size(), value_tokens.begin());
        std::copy_n(tokens.begin() + value_tokens.size(), carrier_tokens.size(), carrier_tokens.begin());
    }
    std::ifstream prompts(prompts_path);
    if (!prompts) throw std::runtime_error("cannot open prompt corpus");
    std::vector<std::string> source_names, taps;
    for (int layer : k_layers) for (const char * tap : { "resid.in", "time.out", "resid.time" }) {
        source_names.push_back("rwkv.layer." + std::to_string(layer) + "." + tap);
        taps.push_back(source_names.back());
    }
    ggml_backend_load_all(); llama_backend_init();
    llama_model_params params = llama_model_default_params(); params.n_gpu_layers = n_gpu_layers;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(model_path.c_str(), params), llama_model_free);
    if (!model || model->arch != LLM_ARCH_RWKV7) throw std::runtime_error("an RWKV-7 model is required");
    llama_context_params cp = llama_context_default_params(); cp.n_ctx = 64; cp.n_batch = 64; cp.n_ubatch = 64;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
    if (!ctx) throw std::runtime_error("context creation failed");
    llama_interp::runtime runtime(ctx.get(), 1);
    const llama_vocab * vocab = llama_model_get_vocab(model.get());
    std::unique_ptr<rwkv_activation_store::activation_dataset_writer> writer;
    std::vector<behavior_row> behavior;
    std::vector<sample_row> samples;
    behavior.reserve(max_prompts);
    samples.reserve(max_prompts * std::size(k_layers));
    uint64_t line = 0;
    for (std::string prompt; line < max_prompts && std::getline(prompts, prompt); ++line) {
        const auto tokens = common_tokenize(vocab, prompt, false, false);
        if (tokens.empty()) throw std::runtime_error("empty prompt tokenization");
        const int32_t carrier_id = int32_t(line / 16), value_id = int32_t(line % 16);
        if (classified && (tokens.size() <= size_t(read_position) || tokens.back() != read_token_id ||
                           tokens[carrier_position] != carrier_tokens[carrier_id] || tokens[value_position] != value_tokens[value_id])) {
            throw std::runtime_error("prompt token positions do not match frozen manifest");
        }
        llama_interp::activation_set captures;
        auto run = [&]() -> llama_interp::task<> {
            auto call = runtime.prefill_tokens(runtime.make_state(), tokens);
            for (const std::string & tap : taps) call.capture_f32("^" + tap + "$", captures);
            call.discard_state();
            (void) co_await call;
        }();
        runtime.run(); run.rethrow_if_failed();
        behavior.push_back(summarize_behavior(line, tokens, classified ? size_t(value_position) : 9,
                                              // runtime prefill keeps one final-logit row per experiment, not per prompt token.
                                              llama_get_logits_ith(ctx.get(), 0), llama_vocab_n_tokens(vocab)));
        std::vector<std::vector<float>> final_vectors;
        std::vector<std::span<const float>> vectors;
        final_vectors.reserve(taps.size());
        vectors.reserve(taps.size());
        for (const std::string & tap : taps) {
            const std::vector<float> & values = rwkv_experiment::require_capture(captures, tap).data_f32;
            if (values.empty() || values.size() % tokens.size() != 0) throw std::runtime_error("malformed batched capture: " + tap);
            const size_t width = values.size() / tokens.size();
            final_vectors.emplace_back(values.end() - width, values.end());
            vectors.push_back(final_vectors.back());
        }
        if (!writer) writer = std::make_unique<rwkv_activation_store::activation_dataset_writer>(
            rwkv_activation_store::activation_dataset_writer::create(output_path, source_names, vectors.front().size()));
        writer->append({ line, line, (int32_t) tokens.size() - 1, tokens.back(), (int32_t) tokens.size() }, vectors);
        for (size_t layer_index = 0; layer_index < std::size(k_layers); ++layer_index) {
            double identity_error = 0.0;
            const auto & x = final_vectors[layer_index * 3];
            const auto & w = final_vectors[layer_index * 3 + 1];
            const auto & y = final_vectors[layer_index * 3 + 2];
            for (size_t column = 0; column < x.size(); ++column) identity_error = std::max(identity_error, std::abs(double(x[column]) + w[column] - y[column]));
            samples.push_back({ line, carrier_id, classified ? carrier_tokens[carrier_id] : -1, carrier_id < 32,
                                value_id, classified ? value_tokens[value_id] : -1, k_layers[layer_index],
                                int32_t(tokens.size()) - 1, tokens.back(), identity_error });
        }
        if ((line + 1) % 16 == 0) {
            writer->commit();
            std::printf("captured=%llu/768 committed_cluster=yes\n", (unsigned long long) line + 1);
            std::fflush(stdout);
        }
    }
    if (!writer || (max_prompts == 768 && line != 768)) throw std::runtime_error("expected exactly 768 prompts");
    writer->commit();
    std::printf("captured=%llu/%llu committed_cluster=yes complete=yes\n", (unsigned long long) line,
                (unsigned long long) max_prompts);
    std::fflush(stdout);
    writer.reset();
    {
        TFile file(output_path.c_str(), "UPDATE");
        if (file.IsZombie()) throw std::runtime_error("cannot reopen capture ROOT file for behavior");
        auto model = ROOT::RNTupleModel::Create();
        auto prompt_id = model->MakeField<uint64_t>("prompt_id");
        auto expected_token = model->MakeField<int32_t>("expected_value_token_id");
        auto log_probability = model->MakeField<double>("expected_value_log_probability");
        auto rank = model->MakeField<int32_t>("expected_value_rank");
        auto top1 = model->MakeField<int32_t>("native_top1_token_id");
        auto correct = model->MakeField<bool>("native_top1_correct");
        auto behavior_writer = ROOT::RNTupleWriter::Append(std::move(model), "behavior", file);
        for (const behavior_row & row : behavior) {
            *prompt_id = row.prompt_id; *expected_token = row.expected_value_token_id;
            *log_probability = row.expected_value_log_probability; *rank = row.expected_value_rank;
            *top1 = row.native_top1_token_id; *correct = row.native_top1_correct;
            behavior_writer->Fill();
        }
        behavior_writer->CommitCluster();
    }
    {
        TFile file(output_path.c_str(), "UPDATE");
        auto model = ROOT::RNTupleModel::Create();
        auto prompt_id = model->MakeField<uint64_t>("prompt_id");
        auto carrier_id = model->MakeField<int32_t>("carrier_id");
        auto carrier_token_id = model->MakeField<int32_t>("carrier_token_id");
        auto carrier_split = model->MakeField<std::string>("carrier_split");
        auto value_id = model->MakeField<int32_t>("value_id");
        auto value_token_id = model->MakeField<int32_t>("value_token_id");
        auto layer = model->MakeField<int32_t>("layer");
        auto position = model->MakeField<int32_t>("read_position");
        auto token = model->MakeField<int32_t>("read_token_id");
        auto identity = model->MakeField<double>("residual_identity_error");
        auto sample_writer = ROOT::RNTupleWriter::Append(std::move(model), "samples", file);
        for (const sample_row & row : samples) {
            *prompt_id = row.prompt_id; *carrier_id = row.carrier_id; *carrier_token_id = row.carrier_token_id;
            *carrier_split = row.carrier_train ? "train" : "test"; *value_id = row.value_id; *value_token_id = row.value_token_id;
            *layer = row.layer; *position = row.read_position; *token = row.read_token_id; *identity = row.residual_identity_error;
            sample_writer->Fill();
        }
        sample_writer->CommitCluster();
    }
    {
        TFile file(output_path.c_str(), "UPDATE");
        auto model = ROOT::RNTupleModel::Create();
        auto key = model->MakeField<std::string>("key");
        auto value = model->MakeField<std::string>("value");
        auto metadata_writer = ROOT::RNTupleWriter::Append(std::move(model), "metadata", file);
        const std::array<std::pair<std::string, std::string>, 6> metadata{{
            { "status", classified ? "frozen_capture" : "development_only" },
            { "model_sha256", sha256_file(model_path) }, { "corpus_sha256", sha256_file(prompts_path) },
            { "manifest_sha256", classified ? sha256_file(manifest_path) : "" },
            { "registration_sha256", classified ? sha256_file(registration_path) : "" },
            { "backend", classified ? "Vulkan" : "unspecified" }
        }};
        for (const auto & [name, text] : metadata) { *key = name; *value = text; metadata_writer->Fill(); }
        metadata_writer->CommitCluster();
    }
    ctx.reset();
    model.reset();
    llama_backend_free();
    return 0;
} catch (const std::exception & error) {
    std::fprintf(stderr, "error: %s\n", error.what()); llama_backend_free(); return 1;
}
