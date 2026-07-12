#include "interp.hpp"
#include "rwkv_activation_store.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

static std::string activation_name(int layer, const char * name) {
    return "rwkv.layer." + std::to_string(layer) + "." + name;
}

static std::vector<int> parse_layers(const char * value) {
    std::vector<int> layers;
    std::stringstream input(value);
    std::string item;
    while (std::getline(input, item, ',')) {
        if (item.empty()) {
            throw std::runtime_error("empty layer in --layers");
        }
        layers.push_back(std::stoi(item));
    }
    return layers;
}

static std::string parse_source(const char * value, int n_layer) {
    const std::string spec(value);
    const size_t separator = spec.find(':');
    if (separator == std::string::npos || separator == 0 || separator + 1 == spec.size()) {
        throw std::runtime_error("source must be LAYER:ACTIVATION");
    }
    const int layer = std::stoi(spec.substr(0, separator));
    if (layer < 0 || layer >= n_layer) {
        throw std::runtime_error("source layer is out of range: " + spec);
    }
    return activation_name(layer, spec.substr(separator + 1).c_str());
}

static const std::vector<float> & require_capture(
        const llama_interp::activation_set & captures,
        const std::string & name) {
    const std::vector<float> * found = nullptr;
    for (const auto & capture : captures) {
        if (capture.name == name && !capture.data_f32.empty()) {
            // A multi-token prefill produces one capture per token. The final capture is
            // the activation that produced the prompt's next-token logits.
            found = &capture.data_f32;
        }
    }
    if (found) {
        return *found;
    }
    throw std::runtime_error("missing FP32 activation capture: " + name);
}

static std::vector<float> copy_logits(const llama_context * ctx, int n_vocab) {
    const float * logits = llama_get_logits_ith(const_cast<llama_context *>(ctx), 0);
    if (!logits) {
        throw std::runtime_error("missing logits");
    }
    return std::vector<float>(logits, logits + n_vocab);
}

static double l2_norm(const std::vector<float> & values) {
    double sum = 0.0;
    for (float value : values) {
        sum += (double) value * value;
    }
    return std::sqrt(sum);
}

static std::vector<int> top_tokens(const std::vector<float> & logits, int n_top) {
    std::vector<int> ids(logits.size());
    std::iota(ids.begin(), ids.end(), 0);
    n_top = std::min(n_top, (int) ids.size());
    std::partial_sort(ids.begin(), ids.begin() + n_top, ids.end(), [&logits](int a, int b) {
        return logits[a] > logits[b];
    });
    ids.resize(n_top);
    return ids;
}

static double max_abs_difference(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("logit dimensions differ");
    }
    double maximum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        maximum = std::max(maximum, std::abs((double) a[i] - b[i]));
    }
    return maximum;
}

static void print_top_tokens(const llama_vocab * vocab, const std::vector<float> & logits, int n_top) {
    int rank = 1;
    for (int token : top_tokens(logits, n_top)) {
        std::printf("%2d  %9.4f  %6d  %s\n", rank++, logits[token], token,
            common_token_to_piece(vocab, token, true).c_str());
    }
}

struct trace_token {
    int id;
    std::string piece;
    float logit;
};

struct trace_readout {
    std::string name;
    std::vector<trace_token> tokens;
};

struct trace_step {
    int index;
    int input_id;
    std::string input_piece;
    int next_id;
    std::string next_piece;
    double final_round_trip_error;
    std::vector<trace_readout> readouts;
};

struct trace_run {
    std::string prompt;
    size_t prompt_token_count;
    std::vector<trace_step> steps;
    std::string stopped;
};

static std::vector<trace_token> trace_top_tokens(const llama_vocab * vocab, const std::vector<float> & logits, int n_top) {
    std::vector<trace_token> result;
    result.reserve(n_top);
    for (int token : top_tokens(logits, n_top)) {
        result.push_back({ token, common_token_to_piece(vocab, token, true), logits[token] });
    }
    return result;
}

static bool is_utf8_continuation(unsigned char value) {
    return (value & 0xc0) == 0x80;
}

static size_t valid_utf8_sequence_size(const std::string & value, size_t index) {
    const unsigned char first = (unsigned char) value[index];
    const size_t remaining = value.size() - index;
    if (first >= 0xc2 && first <= 0xdf && remaining >= 2 && is_utf8_continuation((unsigned char) value[index + 1])) {
        return 2;
    }
    if (first >= 0xe0 && first <= 0xef && remaining >= 3 &&
            is_utf8_continuation((unsigned char) value[index + 1]) &&
            is_utf8_continuation((unsigned char) value[index + 2])) {
        const unsigned char second = (unsigned char) value[index + 1];
        if ((first != 0xe0 || second >= 0xa0) && (first != 0xed || second <= 0x9f)) {
            return 3;
        }
    }
    if (first >= 0xf0 && first <= 0xf4 && remaining >= 4 &&
            is_utf8_continuation((unsigned char) value[index + 1]) &&
            is_utf8_continuation((unsigned char) value[index + 2]) &&
            is_utf8_continuation((unsigned char) value[index + 3])) {
        const unsigned char second = (unsigned char) value[index + 1];
        if ((first != 0xf0 || second >= 0x90) && (first != 0xf4 || second <= 0x8f)) {
            return 4;
        }
    }
    return 0;
}

static void write_json_string(std::ostream & output, const std::string & value) {
    static const char hex[] = "0123456789abcdef";
    output << '"';
    for (size_t i = 0; i < value.size(); ++i) {
        const unsigned char c = (unsigned char) value[i];
        switch (c) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\b': output << "\\b"; break;
            case '\f': output << "\\f"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (c < 0x20) {
                    output << "\\u00" << hex[c >> 4] << hex[c & 0x0f];
                } else if (c < 0x80) {
                    output << c;
                } else {
                    const size_t sequence_size = valid_utf8_sequence_size(value, i);
                    if (sequence_size) {
                        output.write(value.data() + i, sequence_size);
                        i += sequence_size - 1;
                    } else {
                        output << "\\u00" << hex[c >> 4] << hex[c & 0x0f];
                    }
                }
        }
    }
    output << '"';
}

static void write_trace_json(
        const std::string & path,
        const std::string & model_path,
        int n_gpu_layers,
        int n_top,
        const std::vector<std::string> & sources,
        const std::vector<trace_run> & runs) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("failed to open JSON output: " + path);
    }
    output << std::setprecision(9);
    output << "{\n  \"schema_version\": 1,\n  \"model_path\": ";
    write_json_string(output, model_path);
    output << ",\n  \"n_gpu_layers\": " << n_gpu_layers << ",\n  \"top_k\": " << n_top << ",\n  \"sources\": [";
    for (size_t i = 0; i < sources.size(); ++i) {
        if (i) output << ", ";
        write_json_string(output, sources[i]);
    }
    output << "],\n  \"runs\": [\n";
    for (size_t run_index = 0; run_index < runs.size(); ++run_index) {
        const auto & run = runs[run_index];
        if (run_index) output << ",\n";
        output << "    {\n      \"prompt\": ";
        write_json_string(output, run.prompt);
        output << ",\n      \"prompt_token_count\": " << run.prompt_token_count
               << ",\n      \"stopped\": ";
        write_json_string(output, run.stopped);
        output << ",\n      \"steps\": [\n";
        for (size_t step_index = 0; step_index < run.steps.size(); ++step_index) {
            const auto & step = run.steps[step_index];
            if (step_index) output << ",\n";
            output << "        {\"index\": " << step.index
                   << ", \"input\": {\"id\": " << step.input_id << ", \"piece\": ";
            write_json_string(output, step.input_piece);
            output << "}, \"next\": {\"id\": " << step.next_id << ", \"piece\": ";
            write_json_string(output, step.next_piece);
            output << "}, \"final_round_trip_error\": " << step.final_round_trip_error << ", \"readouts\": [";
            for (size_t readout_index = 0; readout_index < step.readouts.size(); ++readout_index) {
                const auto & readout = step.readouts[readout_index];
                if (readout_index) output << ", ";
                output << "{\"source\": ";
                write_json_string(output, readout.name);
                output << ", \"top_tokens\": [";
                for (size_t token_index = 0; token_index < readout.tokens.size(); ++token_index) {
                    const auto & token = readout.tokens[token_index];
                    if (token_index) output << ", ";
                    output << "{\"rank\": " << token_index + 1 << ", \"id\": " << token.id << ", \"piece\": ";
                    write_json_string(output, token.piece);
                    output << ", \"logit\": " << token.logit << "}";
                }
                output << "]}";
            }
            output << "]}";
        }
        output << "\n      ]\n    }";
    }
    output << "\n  ]\n}\n";
    if (!output) {
        throw std::runtime_error("failed to write JSON output: " + path);
    }
}

static llama_interp::task<> capture_activations(
        llama_interp::runtime & rt,
        const llama_interp::rwkv_state & initial,
        const std::vector<llama_token> & tokens,
        const std::vector<std::string> & names,
        llama_interp::activation_set & captures) {
    auto path = rt.prefill_tokens(initial, tokens);
    for (const auto & name : names) {
        path.capture_f32("^" + name + "$", captures);
    }
    path.discard_state();
    co_await path;
}

static llama_interp::task<> prefill_state(
        llama_interp::runtime & rt,
        const llama_interp::rwkv_state & initial,
        const std::vector<llama_token> & tokens,
        llama_interp::rwkv_state & out) {
    out = co_await rt.prefill_tokens(initial, tokens);
}

static llama_interp::task<> capture_step(
        llama_interp::runtime & rt,
        const llama_interp::rwkv_state & initial,
        llama_token token,
        const std::vector<std::string> & names,
        llama_interp::rwkv_state & out,
        llama_interp::activation_set & captures) {
    auto path = rt.prefill_tokens(initial, { token });
    for (const auto & name : names) {
        path.capture_f32("^" + name + "$", captures);
    }
    out = co_await path;
}

static std::vector<std::vector<float>> unembed_batch(
        llama_context * ctx,
        const std::vector<const std::vector<float> *> & activations) {
    if (activations.empty()) {
        return {};
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    const size_t dimension = activations.front()->size();
    std::vector<float> packed;
    packed.reserve(activations.size() * dimension);
    for (const auto * activation : activations) {
        if (activation->size() != dimension) {
            throw std::runtime_error("cannot batch direct unembeddings with different dimensions");
        }
        packed.insert(packed.end(), activation->begin(), activation->end());
    }
    std::vector<float> logits(activations.size() * n_vocab);
    if (!llama_interp_rwkv_final_readout(ctx, packed.data(), (uint32_t) activations.size(), logits.data())) {
        throw std::runtime_error("output-only RWKV direct readout failed");
    }
    std::vector<std::vector<float>> results;
    results.reserve(activations.size());
    for (size_t i = 0; i < activations.size(); ++i) {
        results.emplace_back(logits.begin() + i * n_vocab, logits.begin() + (i + 1) * n_vocab);
    }
    return results;
}

static void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s -m MODEL [-p PROMPT] [--prompts FILE] [--contrast PROMPT] [--contrast-activation-delta] [--center-root FILE] [-n N] [--until-eog] [--json FILE] [-ngl N] [--layer N] [--layers N,N,...] [--source LAYER:ACTIVATION] [--all-channel-out|--all-time-out|--all-time-mix] [--top N]\n"
        "\n"
        "Captures time.v, time.wkv, time.out, and resid.out at one RWKV layer on the\n"
        "prompt's final token, then reads each through the production output norm and head.\n"
        "--center-root subtracts K=1 calibration centroids from all raw time-mix sources.\n"
        "--json writes generation traces for the static rwkv_unembed_viewer.html page.\n",
        argv0);
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::string model_path;
    std::string prompt = "The capital of France is";
    std::string prompts_path;
    std::string contrast_prompt;
    std::string json_path;
    std::string center_root_path;
    int n_gpu_layers = 0;
    std::vector<int> layers;
    std::vector<std::string> source_specs;
    int n_top = 10;
    int n_predict = 0;
    bool until_eog = false;
    bool all_channel_out = false;
    bool all_time_out = false;
    bool all_time_mix = false;
    bool contrast_activation_delta = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (std::strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (std::strcmp(argv[i], "--prompts") == 0 && i + 1 < argc) {
            prompts_path = argv[++i];
        } else if (std::strcmp(argv[i], "--contrast") == 0 && i + 1 < argc) {
            contrast_prompt = argv[++i];
        } else if (std::strcmp(argv[i], "--contrast-activation-delta") == 0) {
            contrast_activation_delta = true;
        } else if (std::strcmp(argv[i], "--json") == 0 && i + 1 < argc) {
            json_path = argv[++i];
        } else if (std::strcmp(argv[i], "--center-root") == 0 && i + 1 < argc) {
            center_root_path = argv[++i];
        } else if (std::strcmp(argv[i], "-ngl") == 0 && i + 1 < argc) {
            n_gpu_layers = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--layer") == 0 && i + 1 < argc) {
            layers = { std::atoi(argv[++i]) };
        } else if (std::strcmp(argv[i], "--layers") == 0 && i + 1 < argc) {
            layers = parse_layers(argv[++i]);
        } else if (std::strcmp(argv[i], "--source") == 0 && i + 1 < argc) {
            source_specs.emplace_back(argv[++i]);
        } else if (std::strcmp(argv[i], "--all-channel-out") == 0) {
            all_channel_out = true;
        } else if (std::strcmp(argv[i], "--all-time-out") == 0) {
            all_time_out = true;
        } else if (std::strcmp(argv[i], "--all-time-mix") == 0) {
            all_time_mix = true;
        } else if ((std::strcmp(argv[i], "-n") == 0 || std::strcmp(argv[i], "--predict") == 0) && i + 1 < argc) {
            n_predict = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--until-eog") == 0) {
            until_eog = true;
        } else if (std::strcmp(argv[i], "--top") == 0 && i + 1 < argc) {
            n_top = std::atoi(argv[++i]);
        } else {
            usage(argv[0]);
            return 1;
        }
    }
    if (model_path.empty() || n_top <= 0 || n_predict < 0) {
        usage(argv[0]);
        return 1;
    }
    if (until_eog && n_predict == 0) {
        n_predict = 128;
    }
    if ((!prompts_path.empty() && !contrast_prompt.empty()) || (n_predict > 0 && !contrast_prompt.empty())) {
        throw std::runtime_error("--contrast cannot be combined with --prompts or --predict");
    }
    if (contrast_activation_delta && contrast_prompt.empty()) {
        throw std::runtime_error("--contrast-activation-delta requires --contrast");
    }
    if (!json_path.empty() && n_predict == 0) {
        throw std::runtime_error("--json requires -n/--predict or --until-eog");
    }
    if ((int) all_channel_out + (int) all_time_out + (int) all_time_mix > 1) {
        throw std::runtime_error("only one --all-... source selection can be used at once");
    }
    if (!center_root_path.empty() && (!all_time_mix || !contrast_prompt.empty() || n_predict > 0 || !json_path.empty())) {
        throw std::runtime_error("--center-root requires static --all-time-mix readout without contrast or generation");
    }

    std::vector<std::string> prompts;
    if (prompts_path.empty()) {
        prompts.push_back(prompt);
    } else {
        std::ifstream input(prompts_path);
        if (!input) {
            throw std::runtime_error("failed to open prompts file: " + prompts_path);
        }
        while (std::getline(input, prompt)) {
            if (!prompt.empty()) {
                prompts.push_back(prompt);
            }
        }
        if (prompts.empty()) {
            throw std::runtime_error("prompts file contains no nonempty lines");
        }
    }

    ggml_backend_load_all();
    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        throw std::runtime_error("failed to load model");
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<std::vector<llama_token>> token_sets;
    size_t max_prompt_tokens = 0;
    for (const auto & item : prompts) {
        auto tokens = common_tokenize(vocab, item, false, true);
        if (tokens.empty()) {
            throw std::runtime_error("prompt tokenized to zero tokens");
        }
        max_prompt_tokens = std::max(max_prompt_tokens, tokens.size());
        token_sets.push_back(std::move(tokens));
    }
    const std::vector<llama_token> contrast_tokens = contrast_prompt.empty() ? std::vector<llama_token>() :
        common_tokenize(vocab, contrast_prompt, false, true);
    if (!contrast_prompt.empty() && contrast_tokens.empty()) {
        throw std::runtime_error("contrast prompt tokenized to zero tokens");
    }

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = std::max({ 128, (int) max_prompt_tokens + 8, (int) contrast_tokens.size() + 8,
        all_time_mix ? 512 : 0 });
    context_params.n_batch = context_params.n_ctx;
    context_params.n_ubatch = context_params.n_ctx;
    llama_context * ctx = llama_init_from_model(model, context_params);
    if (!ctx) {
        throw std::runtime_error("failed to create context");
    }
    llama_interp::runtime rt(ctx, 1);
    const auto initial = rt.make_state();
    if (layers.empty() && source_specs.empty() && !all_channel_out && !all_time_out && !all_time_mix) {
        layers = { (int) initial.n_layer / 2 };
    }
    for (int layer : layers) {
        if (layer < 0 || layer >= (int) initial.n_layer) {
            throw std::runtime_error("invalid RWKV layer");
        }
    }

    const std::vector<const char *> candidates = {
        "resid.in",
        "att.norm",
        "time.r",
        "time.w",
        "time.k",
        "time.v",
        "time.a",
        "time.g",
        "time.wkv",
        "time.rkv",
        "time.out",
        "resid.time",
        "ffn.norm",
        "channel.out",
        "resid.out",
    };
    std::vector<std::string> names;
    if (all_channel_out) {
        for (int layer = 0; layer < (int) initial.n_layer; ++layer) {
            names.push_back(activation_name(layer, "channel.out"));
        }
    } else if (all_time_out) {
        for (int layer = 0; layer < (int) initial.n_layer; ++layer) {
            names.push_back(activation_name(layer, "time.out"));
        }
    } else if (all_time_mix) {
        const std::vector<const char *> time_mix = {
            "time.r", "time.w", "time.k", "time.v", "time.a", "time.g", "time.wkv", "time.rkv",
        };
        for (int layer = 0; layer < (int) initial.n_layer; ++layer) {
            for (const char * activation : time_mix) {
                names.push_back(activation_name(layer, activation));
            }
        }
    }
    if (source_specs.empty() && !all_channel_out && !all_time_out && !all_time_mix) {
        for (int layer : layers) {
            for (const char * candidate : candidates) {
                names.push_back(activation_name(layer, candidate));
            }
        }
    } else if (!source_specs.empty()) {
        for (const auto & source : source_specs) {
            names.push_back(parse_source(source.c_str(), (int) initial.n_layer));
        }
    }
    std::sort(names.begin(), names.end());
    names.erase(std::unique(names.begin(), names.end()), names.end());
    const std::vector<std::string> readout_names = names;
    const std::string final_residual = activation_name((int) initial.n_layer - 1, "resid.out");
    if (std::find(names.begin(), names.end(), final_residual) == names.end()) {
        names.push_back(final_residual);
    }
    std::unordered_map<std::string, uint32_t> center_source_indices;
    std::unique_ptr<rwkv_activation_store::centroid_store_reader> center_reader;
    if (!center_root_path.empty()) {
        const std::vector<const char *> time_mix = {
            "time.r", "time.w", "time.k", "time.v", "time.a", "time.g", "time.wkv", "time.rkv",
        };
        uint32_t source_index = 0;
        for (int layer = 0; layer < (int) initial.n_layer; ++layer) {
            for (const char * activation : time_mix) {
                center_source_indices.emplace(activation_name(layer, activation), source_index++);
            }
        }
        center_source_indices.emplace(final_residual, source_index);
        center_reader = std::make_unique<rwkv_activation_store::centroid_store_reader>(
            rwkv_activation_store::centroid_store_reader::open(center_root_path));
    }
    const int n_vocab = llama_vocab_n_tokens(vocab);
    if (n_predict > 0) {
        std::vector<trace_run> trace_runs;
        for (size_t prompt_index = 0; prompt_index < prompts.size(); ++prompt_index) {
            const auto & prompt_tokens = token_sets[prompt_index];
            llama_interp::rwkv_state state = initial;
            if (prompt_tokens.size() > 1) {
                auto prefix = prefill_state(rt, initial, std::vector<llama_token>(prompt_tokens.begin(), prompt_tokens.end() - 1), state);
                rt.run();
                prefix.rethrow_if_failed();
            }
            llama_token input = prompt_tokens.back();
            std::printf("=== Prompt %zu/%zu ===\nprompt=%s\nsources=", prompt_index + 1, prompts.size(), prompts[prompt_index].c_str());
            for (size_t i = 0; i < readout_names.size(); ++i) {
                std::printf("%s%s", i == 0 ? "" : ",", readout_names[i].c_str());
            }
            std::printf("\n--- Greedy thought trace ---\n");
            trace_run trace { prompts[prompt_index], prompt_tokens.size(), {}, "length" };
            for (int step = 0; step < n_predict; ++step) {
                llama_interp::rwkv_state next_state;
                llama_interp::activation_set captures;
                auto capture = capture_step(rt, state, input, names, next_state, captures);
                rt.run();
                capture.rethrow_if_failed();
                const std::vector<float> native_logits = copy_logits(ctx, n_vocab);
                const int next = top_tokens(native_logits, 1)[0];
                trace_step trace_entry = {
                    step,
                    input,
                    common_token_to_piece(vocab, input, true),
                    next,
                    common_token_to_piece(vocab, next, true),
                    0.0,
                    {},
                };
                std::printf("step=%d input=%s next=%s\n", step,
                    common_token_to_piece(vocab, input, true).c_str(), common_token_to_piece(vocab, next, true).c_str());
                std::vector<const std::vector<float> *> activations;
                activations.reserve(names.size());
                for (const auto & name : names) {
                    activations.push_back(&require_capture(captures, name));
                }
                const auto direct_logits = unembed_batch(
                    ctx, activations);
                for (size_t readout_index = 0; readout_index < names.size(); ++readout_index) {
                    const auto & name = names[readout_index];
                    const auto & activation = require_capture(captures, name);
                    const auto & logits = direct_logits[readout_index];
                    if (name == final_residual) {
                        trace_entry.final_round_trip_error = max_abs_difference(native_logits, logits);
                        std::printf("  final_round_trip_error=%.6e\n", trace_entry.final_round_trip_error);
                    } else {
                        std::printf("  %s\n", name.c_str());
                        print_top_tokens(vocab, logits, n_top);
                        trace_entry.readouts.push_back({ name, trace_top_tokens(vocab, logits, n_top) });
                    }
                }
                trace.steps.push_back(std::move(trace_entry));
                state = std::move(next_state);
                input = next;
                if (until_eog && llama_vocab_is_eog(vocab, next)) {
                    trace.stopped = "eog";
                    std::printf("stopped=eog\n");
                    break;
                }
            }
            trace_runs.push_back(std::move(trace));
        }
        if (!json_path.empty()) {
            write_trace_json(json_path, model_path, n_gpu_layers, n_top, readout_names, trace_runs);
            std::printf("json=%s\n", json_path.c_str());
        }
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 0;
    }
    auto evaluate_prompt = [&](const std::string & current_prompt, const std::vector<llama_token> & tokens, size_t index) {
        llama_interp::activation_set captures;
        auto capture = capture_activations(rt, initial, tokens, names, captures);
        rt.run();
        capture.rethrow_if_failed();
        const std::vector<float> native_logits = copy_logits(ctx, n_vocab);
        llama_interp::activation_set contrast_captures;
        if (!contrast_tokens.empty()) {
            auto contrast_capture = capture_activations(rt, initial, contrast_tokens, names, contrast_captures);
            rt.run();
            contrast_capture.rethrow_if_failed();
        }

        std::printf("=== Prompt %zu/%zu ===\nprompt=%s\n", index + 1, prompts.size(), current_prompt.c_str());
        if (source_specs.empty()) {
            std::printf("layers=");
            for (size_t i = 0; i < layers.size(); ++i) {
                std::printf("%s%d", i == 0 ? "" : ",", layers[i]);
            }
            std::printf("\n");
        } else {
            std::printf("sources=");
            for (size_t i = 0; i < source_specs.size(); ++i) {
                std::printf("%s%s", i == 0 ? "" : ",", source_specs[i].c_str());
            }
            std::printf("\n");
        }
        std::printf("prompt_tokens=%zu\n--- Native next-token logits ---\n", tokens.size());
        print_top_tokens(vocab, native_logits, n_top);

        std::vector<const std::vector<float> *> prompt_activations;
        std::vector<const std::vector<float> *> contrast_activations;
        std::vector<std::vector<float>> activation_deltas;
        prompt_activations.reserve(names.size());
        contrast_activations.reserve(names.size());
        activation_deltas.reserve(names.size());
        for (const auto & name : names) {
            const auto & activation = require_capture(captures, name);
            if (activation.size() != (size_t) initial.n_embd_r / 2) {
                throw std::runtime_error("unexpected activation dimension for " + name);
            }
            prompt_activations.push_back(&activation);
            if (!contrast_tokens.empty()) {
                const auto & contrast_activation = require_capture(contrast_captures, name);
                if (contrast_activation.size() != activation.size()) {
                    throw std::runtime_error("contrast activation dimension differs for " + name);
                }
                contrast_activations.push_back(&contrast_activation);
                if (contrast_activation_delta) {
                    std::vector<float> delta(activation.size());
                    for (size_t i = 0; i < activation.size(); ++i) {
                        delta[i] = activation[i] - contrast_activation[i];
                    }
                    activation_deltas.push_back(std::move(delta));
                }
            }
        }
        const auto prompt_logits = unembed_batch(ctx, prompt_activations);
        const auto contrast_logits = contrast_activations.empty() ? std::vector<std::vector<float>>() :
            unembed_batch(ctx, contrast_activations);
        std::vector<const std::vector<float> *> delta_activations;
        delta_activations.reserve(activation_deltas.size());
        for (const auto & delta : activation_deltas) {
            delta_activations.push_back(&delta);
        }
        const auto activation_delta_logits = delta_activations.empty() ? std::vector<std::vector<float>>() :
            unembed_batch(ctx, delta_activations);
        for (size_t readout_index = 0; readout_index < names.size(); ++readout_index) {
            const auto & name = names[readout_index];
            const auto & activation = *prompt_activations[readout_index];
            const auto & logits = prompt_logits[readout_index];
            std::printf("--- Direct unembedding: %s (l2=%.6g) ---\n", name.c_str(), l2_norm(activation));
            print_top_tokens(vocab, logits, n_top);
            if (!contrast_tokens.empty()) {
                std::vector<float> delta(n_vocab);
                for (int i = 0; i < n_vocab; ++i) {
                    delta[i] = logits[i] - contrast_logits[readout_index][i];
                }
                std::vector<float> negative_delta(n_vocab);
                for (int i = 0; i < n_vocab; ++i) {
                    negative_delta[i] = -delta[i];
                }
                std::printf("--- Contrast %s: prompt minus contrast ---\n", name.c_str());
                std::printf("positive: %s\n", current_prompt.c_str());
                print_top_tokens(vocab, delta, n_top);
                std::printf("positive: %s\n", contrast_prompt.c_str());
                print_top_tokens(vocab, negative_delta, n_top);
            }
            if (contrast_activation_delta) {
                std::printf("--- Activation-delta direct unembedding %s: prompt minus contrast ---\n", name.c_str());
                print_top_tokens(vocab, activation_delta_logits[readout_index], n_top);
            }
            if (name == final_residual) {
                std::printf("residual round-trip max_logit_abs_error=%.6e\n", max_abs_difference(native_logits, logits));
            }
        }
        if (center_reader) {
            std::vector<std::vector<float>> centered_activations;
            std::vector<const std::vector<float> *> centered_activation_ptrs;
            std::vector<std::string> centered_names;
            centered_activations.reserve(names.size() - 1);
            centered_activation_ptrs.reserve(names.size() - 1);
            centered_names.reserve(names.size() - 1);
            for (size_t readout_index = 0; readout_index < names.size(); ++readout_index) {
                const auto & name = names[readout_index];
                if (name == final_residual) {
                    continue; // The coordinate-valid final residual remains the uncentered control.
                }
                const auto source = center_source_indices.find(name);
                if (source == center_source_indices.end()) {
                    throw std::runtime_error("centering artifact has no source index for " + name);
                }
                const auto centers = center_reader->read_source(source->second, 1);
                const auto & activation = *prompt_activations[readout_index];
                if (centers.size() != 1 || centers[0].values.size() != activation.size()) {
                    throw std::runtime_error("centering artifact does not match source " + name);
                }
                std::vector<float> centered(activation.size());
                for (size_t i = 0; i < centered.size(); ++i) {
                    centered[i] = activation[i] - centers[0].values[i];
                }
                centered_activations.push_back(std::move(centered));
                centered_activation_ptrs.push_back(&centered_activations.back());
                centered_names.push_back(name);
            }
            const auto centered_logits = unembed_batch(ctx, centered_activation_ptrs);
            for (size_t readout_index = 0; readout_index < centered_names.size(); ++readout_index) {
                std::printf("--- Centered direct unembedding: %s (l2=%.6g) ---\n",
                    centered_names[readout_index].c_str(), l2_norm(*centered_activation_ptrs[readout_index]));
                print_top_tokens(vocab, centered_logits[readout_index], n_top);
            }
        }
    };
    for (size_t i = 0; i < prompts.size(); ++i) {
        evaluate_prompt(prompts[i], token_sets[i], i);
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
