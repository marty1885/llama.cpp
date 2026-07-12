#include "interp.hpp"
#include "rwkv_activation_store.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct scored_token {
    int id;
    std::string piece;
    float score;
};

std::string activation_name(int layer, const char * activation) {
    return "rwkv.layer." + std::to_string(layer) + "." + activation;
}

std::vector<std::string> all_time_mix_names(int n_layer) {
    const std::vector<const char *> activations = {
        "time.r", "time.w", "time.k", "time.v", "time.a", "time.g", "time.wkv", "time.rkv",
    };
    std::vector<std::string> names;
    names.reserve(n_layer * activations.size());
    for (int layer = 0; layer < n_layer; ++layer) {
        for (const char * activation : activations) {
            names.push_back(activation_name(layer, activation));
        }
    }
    return names;
}

void write_json_string(std::ostream & output, const std::string & value) {
    static const char hex[] = "0123456789abcdef";
    output << '"';
    for (unsigned char c : value) {
        switch (c) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\b': output << "\\b"; break;
            case '\f': output << "\\f"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (c < 0x20) output << "\\u00" << hex[c >> 4] << hex[c & 0x0f];
                else output << c;
        }
    }
    output << '"';
}

const std::vector<float> & require_capture(
        const llama_interp::activation_set & captures,
        const std::string & name) {
    const std::vector<float> * result = nullptr;
    for (const auto & capture : captures) {
        if (capture.name == name && !capture.data_f32.empty()) {
            result = &capture.data_f32;
        }
    }
    if (!result) throw std::runtime_error("missing FP32 capture: " + name);
    return *result;
}

std::vector<ggml_fp16_t> centered_delta(
        const std::vector<float> & activation,
        const std::vector<float> & center,
        float epsilon) {
    if (activation.size() != center.size()) throw std::runtime_error("centroid dimension differs from activation");
    double squared_norm = 0.0;
    for (size_t i = 0; i < activation.size(); ++i) {
        const double d = activation[i] - center[i];
        squared_norm += d * d;
    }
    const double norm = std::sqrt(squared_norm);
    if (!std::isfinite(norm) || norm == 0.0) throw std::runtime_error("centered direction has invalid norm");
    std::vector<ggml_fp16_t> result;
    result.reserve(activation.size());
    for (size_t i = 0; i < activation.size(); ++i) {
        result.push_back(ggml_fp32_to_fp16((float) (epsilon * (activation[i] - center[i]) / norm)));
    }
    return result;
}

std::vector<ggml_fp16_t> negate(const std::vector<ggml_fp16_t> & values) {
    std::vector<ggml_fp16_t> result;
    result.reserve(values.size());
    for (ggml_fp16_t value : values) result.push_back(ggml_fp32_to_fp16(-ggml_fp16_to_fp32(value)));
    return result;
}

llama_interp::task<> prefill_state(
        llama_interp::runtime & rt,
        const llama_interp::rwkv_state & initial,
        const std::vector<llama_token> & tokens,
        llama_interp::rwkv_state & out) {
    out = co_await rt.prefill_tokens(initial, tokens);
}

llama_interp::task<> capture_clean_step(
        llama_interp::runtime & rt,
        const llama_interp::rwkv_state & before,
        llama_token input,
        const std::vector<std::string> & names,
        llama_interp::rwkv_state & after,
        llama_interp::activation_set & captures) {
    auto path = rt.prefill_tokens(before, { input });
    for (const auto & name : names) path.capture_f32("^" + name + "$", captures);
    after = co_await path;
}

llama_interp::task<> perturb_row(
        llama_interp::runtime & rt,
        const llama_interp::rwkv_state & before,
        llama_token input,
        const std::string & source,
        std::vector<ggml_fp16_t> delta,
        const std::string * final,
        llama_interp::activation_set * captures) {
    auto path = rt.prefill_tokens(before, { input });
    path.perturb("^" + source + "$", llama_interp::runtime::add_head(-1, std::move(delta)));
    if (final && captures) path.capture_f32("^" + *final + "$", *captures);
    path.discard_state();
    co_await path;
}

std::vector<scored_token> top_tokens(
        const llama_vocab * vocab,
        const float * plus,
        const float * minus,
        int n_vocab,
        double epsilon,
        int n_top,
        bool positive) {
    std::vector<int> ids(n_vocab);
    std::iota(ids.begin(), ids.end(), 0);
    std::partial_sort(ids.begin(), ids.begin() + n_top, ids.end(), [plus, minus, epsilon, positive](int a, int b) {
        const float da = (float) ((plus[a] - minus[a]) / (2.0 * epsilon));
        const float db = (float) ((plus[b] - minus[b]) / (2.0 * epsilon));
        return positive ? da > db : da < db;
    });
    std::vector<scored_token> result;
    result.reserve(n_top);
    for (int i = 0; i < n_top; ++i) {
        const int token = ids[i];
        result.push_back({ token, common_token_to_piece(vocab, token, true), (float) ((plus[token] - minus[token]) / (2.0 * epsilon)) });
    }
    return result;
}

void write_tokens(std::ostream & output, const std::vector<scored_token> & tokens) {
    output << "[";
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i) output << ", ";
        output << "{\"rank\": " << i + 1 << ", \"id\": " << tokens[i].id << ", \"piece\": ";
        write_json_string(output, tokens[i].piece);
        output << ", \"logit\": " << tokens[i].score << "}";
    }
    output << "]";
}

void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s -m MODEL --center-root CENTROIDS.root --json TRACE.json [-p PROMPT] [-n N] [--epsilon E] [--top N] [--max-sources N] [--perturb-batch N] [-ngl N]\n"
        "\n"
        "Exports centered symmetric perturbation token derivatives for all 488 raw RWKV time-mix sources.\n",
        argv0);
}

} // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    std::string model_path;
    std::string center_root_path;
    std::string json_path;
    std::string prompt = "A computer's central processing unit is the";
    int n_predict = 10;
    int n_gpu_layers = 0;
    int n_top = 10;
    int max_sources = 0;
    int perturb_batch = 16;
    float epsilon = 0.2f;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (std::strcmp(argv[i], "--center-root") == 0 && i + 1 < argc) center_root_path = argv[++i];
        else if (std::strcmp(argv[i], "--json") == 0 && i + 1 < argc) json_path = argv[++i];
        else if (std::strcmp(argv[i], "-p") == 0 && i + 1 < argc) prompt = argv[++i];
        else if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) n_predict = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-ngl") == 0 && i + 1 < argc) n_gpu_layers = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--top") == 0 && i + 1 < argc) n_top = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--max-sources") == 0 && i + 1 < argc) max_sources = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--epsilon") == 0 && i + 1 < argc) epsilon = std::strtof(argv[++i], nullptr);
        else if (std::strcmp(argv[i], "--perturb-batch") == 0 && i + 1 < argc) perturb_batch = std::atoi(argv[++i]);
        else {
            usage(argv[0]);
            return 1;
        }
    }
    if (model_path.empty() || center_root_path.empty() || json_path.empty() || n_predict <= 0 || n_top <= 0 || epsilon <= 0.0f || perturb_batch != 2) {
        usage(argv[0]);
        return 1;
    }
    const std::filesystem::path output_path(json_path);
    if (!output_path.parent_path().empty()) std::filesystem::create_directories(output_path.parent_path());

    ggml_backend_load_all();
    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) throw std::runtime_error("failed to load model");
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const std::vector<llama_token> prompt_tokens = common_tokenize(vocab, prompt, false, true);
    if (prompt_tokens.empty()) throw std::runtime_error("prompt tokenized to zero tokens");

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = 512;
    context_params.n_batch = perturb_batch;
    context_params.n_ubatch = perturb_batch;
    context_params.n_seq_max = perturb_batch;
    llama_context * ctx = llama_init_from_model(model, context_params);
    if (!ctx) throw std::runtime_error("failed to create context");
    llama_interp::runtime rt(ctx, perturb_batch);
    const auto initial = rt.make_state();
    if (initial.n_embd_r / 2 != rwkv_activation_store::k_default_dimension) throw std::runtime_error("unexpected RWKV activation dimension");
    std::vector<std::string> sources = all_time_mix_names((int) initial.n_layer);
    if (max_sources > 0 && (size_t) max_sources < sources.size()) sources.resize(max_sources);
    const std::string final = activation_name((int) initial.n_layer - 1, "resid.out");
    std::vector<std::vector<float>> centers(sources.size());
    const auto center_reader = rwkv_activation_store::centroid_store_reader::open(center_root_path);
    for (size_t source = 0; source < sources.size(); ++source) {
        const auto found = center_reader.read_source((uint32_t) source, 1);
        if (found.size() != 1 || found[0].values.size() != (size_t) initial.n_embd_r / 2) {
            throw std::runtime_error("centroid file does not match source " + sources[source]);
        }
        centers[source] = found[0].values;
    }

    llama_interp::rwkv_state state = initial;
    if (prompt_tokens.size() > 1) {
        auto task = prefill_state(rt, initial, std::vector<llama_token>(prompt_tokens.begin(), prompt_tokens.end() - 1), state);
        rt.run();
        task.rethrow_if_failed();
    }
    llama_token input = prompt_tokens.back();
    const int n_vocab = llama_vocab_n_tokens(vocab);
    n_top = std::min(n_top, n_vocab);
    std::ofstream output(json_path);
    if (!output) throw std::runtime_error("failed to open JSON output: " + json_path);
    output << std::setprecision(9);
    output << "{\n  \"schema_version\": 1,\n  \"trace_kind\": \"rwkv_perturbation\",\n  \"model_path\": ";
    write_json_string(output, model_path);
    output << ",\n  \"n_gpu_layers\": " << n_gpu_layers << ",\n  \"top_k\": " << n_top << ",\n  \"sources\": [";
    for (size_t source = 0; source < sources.size(); ++source) {
        if (source) output << ", ";
        write_json_string(output, sources[source] + ".positive");
        output << ", ";
        write_json_string(output, sources[source] + ".negative");
    }
    output << "],\n  \"runs\": [{\n    \"prompt\": ";
    write_json_string(output, prompt);
    output << ",\n    \"prompt_token_count\": " << prompt_tokens.size() << ",\n    \"stopped\": \"length\",\n    \"steps\": [";

    const size_t sources_per_batch = (size_t) perturb_batch / 2;
    for (int step = 0; step < n_predict; ++step) {
        llama_interp::activation_set clean_caps;
        llama_interp::rwkv_state next_state;
        auto clean_task = capture_clean_step(rt, state, input, sources, next_state, clean_caps);
        rt.run();
        clean_task.rethrow_if_failed();
        if (step) output << ",";
        output << "\n      {\"index\": " << step << ", \"input\": {\"id\": " << input << ", \"piece\": ";
        write_json_string(output, common_token_to_piece(vocab, input, true));
        output << "}, \"next\": {\"id\": " << next_state.next_token << ", \"piece\": ";
        write_json_string(output, common_token_to_piece(vocab, next_state.next_token, true));
        output << "}, \"final_round_trip_error\": 0, \"readouts\": [";
        bool first_readout = true;
        for (size_t begin = 0; begin < sources.size(); begin += sources_per_batch) {
            const size_t count = std::min(sources_per_batch, sources.size() - begin);
            llama_interp::activation_set final_caps;
            std::vector<llama_interp::task<>> tasks;
            tasks.reserve(count * 2);
            for (size_t offset = 0; offset < count; ++offset) {
                const size_t source = begin + offset;
                const auto delta = centered_delta(require_capture(clean_caps, sources[source]), centers[source], epsilon);
                tasks.emplace_back(perturb_row(rt, state, input, sources[source], delta, offset == 0 ? &final : nullptr, offset == 0 ? &final_caps : nullptr));
                tasks.emplace_back(perturb_row(rt, state, input, sources[source], negate(delta), nullptr, nullptr));
            }
            rt.run();
            for (const auto & task : tasks) task.rethrow_if_failed();
            const auto & rows = require_capture(final_caps, final);
            const size_t dimension = (size_t) initial.n_embd_r / 2;
            if (rows.size() != count * 2 * dimension) throw std::runtime_error("batched final capture has unexpected size");
            std::vector<float> pair_logits(2 * n_vocab);
            for (size_t offset = 0; offset < count; ++offset) {
                const size_t source = begin + offset;
                const float * residual_pair = rows.data() + (2 * offset) * dimension;
                if (!llama_interp_rwkv_final_readout(ctx, residual_pair, 2, pair_logits.data())) {
                    throw std::runtime_error("output-only RWKV final readout failed");
                }
                const auto positive = top_tokens(vocab, pair_logits.data(), pair_logits.data() + n_vocab, n_vocab, epsilon, n_top, true);
                const auto negative = top_tokens(vocab, pair_logits.data(), pair_logits.data() + n_vocab, n_vocab, epsilon, n_top, false);
                if (!first_readout) output << ", ";
                first_readout = false;
                output << "{\"source\": ";
                write_json_string(output, sources[source] + ".positive");
                output << ", \"top_tokens\": ";
                write_tokens(output, positive);
                output << "}, {\"source\": ";
                write_json_string(output, sources[source] + ".negative");
                output << ", \"top_tokens\": ";
                write_tokens(output, negative);
                output << "}";
            }
        }
        output << "]}";
        std::printf("step=%d/%d input=%s next=%s sources=%zu\n", step + 1, n_predict,
            common_token_to_piece(vocab, input, true).c_str(), common_token_to_piece(vocab, next_state.next_token, true).c_str(), sources.size());
        state = std::move(next_state);
        input = state.next_token;
    }
    output << "\n    ]\n  }]\n}\n";
    if (!output) throw std::runtime_error("failed to write JSON output");
    std::printf("json=%s\n", json_path.c_str());
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
