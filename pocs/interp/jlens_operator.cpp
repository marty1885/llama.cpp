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
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct operator_results {
    llama_interp::activation_set clean_caps;
    llama_interp::activation_set repeat_clean_caps;
    llama_interp::activation_set plus_caps;
    llama_interp::activation_set minus_caps;
    llama_interp::activation_set repeat_plus_caps;
    llama_interp::decode_result clean;
    llama_interp::decode_result repeat_clean;
    llama_interp::decode_result plus;
    llama_interp::decode_result minus;
    llama_interp::decode_result repeat_plus;
    std::vector<ggml_fp16_t> perturbation;
};

struct scored_token {
    int id;
    std::string piece;
    float score;
};

struct logit_derivative {
    std::vector<scored_token> positive;
    std::vector<scored_token> negative;
};

static void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s -m MODEL [-p PROMPT] [-ngl N] [--layer N] [--source-activation residual|time-r|time-w|time-k|time-v|time-a|time-g|time-wkv|time-rkv] [--center-root FILE|--direction-prompt TEXT] [--epsilon E] [--compare-epsilon E] [--top N] [--json FILE]\n"
        "\n"
        "Runs one symmetric finite-difference check from a layer activation to the "
        "same-token time output and final residual.\n",
        argv0);
}

static std::string residual_name(int layer) {
    return "rwkv.layer." + std::to_string(layer) + ".resid.out";
}

static std::string time_out_name(int layer) {
    return "rwkv.layer." + std::to_string(layer) + ".time.out";
}

static int time_source_index(const std::string & activation) {
    static const char * const time_activations[] = {
        "time-r", "time-w", "time-k", "time-v", "time-a", "time-g", "time-wkv", "time-rkv",
    };
    for (int index = 0; index < 8; ++index) {
        const char * candidate = time_activations[index];
        if (activation == candidate) {
            return index;
        }
    }
    return -1;
}

static std::string source_name(int layer, const std::string & activation) {
    if (activation == "residual") {
        return residual_name(layer);
    }
    if (time_source_index(activation) >= 0) {
        return "rwkv.layer." + std::to_string(layer) + ".time." + (activation.substr(5));
    }
    throw std::runtime_error("unknown source activation: " + activation);
}

static const llama_interp_activation & require_capture(
        const llama_interp::activation_set & caps,
        const std::string & name) {
    const llama_interp_activation * found = nullptr;
    for (const auto & cap : caps) {
        if (cap.name == name) {
            if (found) {
                throw std::runtime_error("duplicate activation capture: " + name);
            }
            found = &cap;
        }
    }
    if (!found) {
        throw std::runtime_error("missing activation capture: " + name);
    }
    return *found;
}

static std::vector<ggml_fp16_t> rademacher_perturbation(size_t n, float epsilon) {
    std::vector<ggml_fp16_t> out(n);
    const float scale = epsilon / std::sqrt((float) n);
    for (size_t i = 0; i < n; ++i) {
        out[i] = ggml_fp32_to_fp16((i & 1) ? scale : -scale);
    }
    return out;
}

static std::vector<ggml_fp16_t> reference_perturbation(
        const std::vector<float> & activation,
        const std::vector<float> & reference,
        float epsilon) {
    if (activation.size() != reference.size()) {
        throw std::runtime_error("reference direction dimension differs from source activation");
    }
    double squared_norm = 0.0;
    for (size_t i = 0; i < activation.size(); ++i) {
        const double delta = activation[i] - reference[i];
        squared_norm += delta * delta;
    }
    const double norm = std::sqrt(squared_norm);
    if (!std::isfinite(norm) || norm == 0.0) {
        throw std::runtime_error("source-reference direction has invalid norm");
    }
    std::vector<ggml_fp16_t> out;
    out.reserve(activation.size());
    for (size_t i = 0; i < activation.size(); ++i) {
        out.push_back(ggml_fp32_to_fp16((float) (epsilon * (activation[i] - reference[i]) / norm)));
    }
    return out;
}

static std::vector<ggml_fp16_t> negate(const std::vector<ggml_fp16_t> & values) {
    std::vector<ggml_fp16_t> out;
    out.reserve(values.size());
    for (ggml_fp16_t value : values) {
        out.push_back(ggml_fp32_to_fp16(-ggml_fp16_to_fp32(value)));
    }
    return out;
}

static double l2_norm(const std::vector<ggml_fp16_t> & values) {
    double sum = 0.0;
    for (ggml_fp16_t value : values) {
        const double x = ggml_fp16_to_fp32(value);
        sum += x*x;
    }
    return std::sqrt(sum);
}

static double difference_l2(
        const std::vector<float> & a,
        const std::vector<float> & b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("activation dimensions differ");
    }
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double x = a[i] - b[i];
        sum += x*x;
    }
    return std::sqrt(sum);
}

static double symmetry_error(
        const std::vector<float> & clean,
        const std::vector<float> & plus,
        const std::vector<float> & minus) {
    if (clean.size() != plus.size() || clean.size() != minus.size()) {
        throw std::runtime_error("activation dimensions differ");
    }

    double numerator = 0.0;
    double denominator = 0.0;
    for (size_t i = 0; i < clean.size(); ++i) {
        const double y0 = clean[i];
        const double yp = plus[i];
        const double ym = minus[i];
        const double even = (yp - y0) + (ym - y0);
        const double odd = (yp - y0) - (ym - y0);
        numerator += even*even;
        denominator += odd*odd;
    }
    return std::sqrt(numerator) / std::max(std::sqrt(denominator), 1e-30);
}

static double perturbation_error_l2(
        const std::vector<float> & base,
        const std::vector<float> & perturbed,
        const std::vector<ggml_fp16_t> & expected) {
    if (base.size() != perturbed.size() || base.size() != expected.size()) {
        throw std::runtime_error("activation dimensions differ");
    }
    double sum = 0.0;
    for (size_t i = 0; i < base.size(); ++i) {
        const double error = perturbed[i] - base[i] -
                             ggml_fp16_to_fp32(expected[i]);
        sum += error*error;
    }
    return std::sqrt(sum);
}

struct derivative_comparison {
    double cosine = 0.0;
    double relative_l2_difference = 0.0;
};

static derivative_comparison compare_central_derivatives(
        const std::vector<float> & plus_a,
        const std::vector<float> & minus_a,
        double epsilon_a,
        const std::vector<float> & plus_b,
        const std::vector<float> & minus_b,
        double epsilon_b) {
    if (plus_a.size() != minus_a.size() || plus_a.size() != plus_b.size() || plus_a.size() != minus_b.size()) {
        throw std::runtime_error("activation dimensions differ");
    }

    double dot = 0.0;
    double norm_a = 0.0;
    double norm_b = 0.0;
    double difference = 0.0;
    for (size_t i = 0; i < plus_a.size(); ++i) {
        const double a = (plus_a[i] - minus_a[i]) / (2.0 * epsilon_a);
        const double b = (plus_b[i] - minus_b[i]) / (2.0 * epsilon_b);
        dot += a*b;
        norm_a += a*a;
        norm_b += b*b;
        const double d = a - b;
        difference += d*d;
    }

    derivative_comparison out;
    out.cosine = dot / std::max(std::sqrt(norm_a * norm_b), 1e-30);
    out.relative_l2_difference = std::sqrt(difference) / std::max(std::sqrt(norm_b), 1e-30);
    return out;
}

static llama_interp::task<> prefill_prompt(
        llama_interp::runtime & rt,
        const llama_interp::rwkv_state & initial_state,
        const std::string & prompt,
        llama_interp::rwkv_state & out) {
    out = co_await rt.prefill(initial_state, prompt);
}

static llama_interp::task<> capture_prompt_source(
        llama_interp::runtime & rt,
        const llama_interp::rwkv_state & initial_state,
        const std::string & prompt,
        const std::string & source,
        llama_interp::activation_set & captures) {
    auto path = rt.prefill(initial_state, prompt);
    path.capture_f32("^" + source + "$", captures);
    path.discard_state();
    co_await path;
}

static const std::vector<float> & last_capture_f32(
        const llama_interp::activation_set & captures,
        const std::string & name) {
    const std::vector<float> * result = nullptr;
    for (const auto & capture : captures) {
        if (capture.name == name && !capture.data_f32.empty()) {
            result = &capture.data_f32;
        }
    }
    if (!result) {
        throw std::runtime_error("missing FP32 activation capture: " + name);
    }
    return *result;
}

static llama_interp::task<> run_symmetric_check(
        llama_interp::runtime & rt,
        const llama_interp::rwkv_state & prompt_state,
        uint32_t n_layer,
        int source_layer,
        const std::string & source_activation,
        const std::vector<float> * reference,
        float epsilon,
        operator_results & out) {
    using namespace llama_interp;

    const std::string source = source_name(source_layer, source_activation);
    const std::string source_capture = source_activation == "residual" ? source + ".pre" : source;
    const std::string time_out = time_out_name(source_layer);
    const std::string final = residual_name((int) n_layer - 1);

    auto clean = rt.decode(prompt_state, 1);
    clean.capture_f32("^" + source_capture + "$", out.clean_caps);
    clean.capture_f32("^" + time_out + "$", out.clean_caps);
    clean.capture_f32("^" + final + "$", out.clean_caps);
    out.clean = co_await clean;

    const auto & clean_source = require_capture(out.clean_caps, source_capture);
    out.perturbation = reference ? reference_perturbation(clean_source.data_f32, *reference, epsilon) :
        rademacher_perturbation(clean_source.data_f32.size(), epsilon);

    auto repeat_clean = rt.decode(prompt_state, 1);
    repeat_clean.capture_f32("^" + time_out + "$", out.repeat_clean_caps);
    repeat_clean.capture_f32("^" + final + "$", out.repeat_clean_caps);
    out.repeat_clean = co_await repeat_clean;

    auto plus = rt.decode(prompt_state, 1);
    plus.perturb("^" + source + "$", runtime::add_head(-1, out.perturbation));
    if (source != final) {
        plus.capture_f32("^" + source + "$", out.plus_caps);
    }
    plus.capture_f32("^" + time_out + "$", out.plus_caps);
    plus.capture_f32("^" + final + "$", out.plus_caps);
    out.plus = co_await plus;

    auto minus = rt.decode(prompt_state, 1);
    minus.perturb("^" + source + "$", runtime::add_head(-1, negate(out.perturbation)));
    minus.capture_f32("^" + time_out + "$", out.minus_caps);
    minus.capture_f32("^" + final + "$", out.minus_caps);
    out.minus = co_await minus;

    auto repeat_plus = rt.decode(prompt_state, 1);
    repeat_plus.perturb("^" + source + "$", runtime::add_head(-1, out.perturbation));
    repeat_plus.capture_f32("^" + time_out + "$", out.repeat_plus_caps);
    repeat_plus.capture_f32("^" + final + "$", out.repeat_plus_caps);
    out.repeat_plus = co_await repeat_plus;
}

static void report_target(
        const char * label,
        const std::string & target,
        const operator_results & results,
        const operator_results * comparison) {
    const auto & clean = require_capture(results.clean_caps, target);
    const auto & repeat_clean = require_capture(results.repeat_clean_caps, target);
    const auto & plus = require_capture(results.plus_caps, target);
    const auto & minus = require_capture(results.minus_caps, target);
    const auto & repeat_plus = require_capture(results.repeat_plus_caps, target);
    const double epsilon = l2_norm(results.perturbation);

    std::printf("target=%s (%s) elements=%zu backend=%s\n", label, target.c_str(), clean.data_f32.size(), clean.backend.c_str());
    std::printf("%s plus delta l2=%.6e minus delta l2=%.6e central derivative l2=%.6e\n",
        label, difference_l2(plus.data_f32, clean.data_f32), difference_l2(minus.data_f32, clean.data_f32),
        difference_l2(plus.data_f32, minus.data_f32) / (2.0 * epsilon));
    std::printf("%s relative symmetry error=%.6e repeat clean exact=%s repeat plus exact=%s\n",
        label, symmetry_error(clean.data_f32, plus.data_f32, minus.data_f32),
        clean.data_f32 == repeat_clean.data_f32 ? "yes" : "no",
        plus.data_f32 == repeat_plus.data_f32 ? "yes" : "no");

    if (comparison) {
        const auto & comparison_clean = require_capture(comparison->clean_caps, target);
        const auto & comparison_plus = require_capture(comparison->plus_caps, target);
        const auto & comparison_minus = require_capture(comparison->minus_caps, target);
        const double comparison_epsilon = l2_norm(comparison->perturbation);
        const derivative_comparison derivative = compare_central_derivatives(
            plus.data_f32, minus.data_f32, epsilon,
            comparison_plus.data_f32, comparison_minus.data_f32, comparison_epsilon);
        std::printf("%s shared-state clean exact=%s central derivative cosine=%.6e relative_l2_difference=%.6e\n",
            label, clean.data_f32 == comparison_clean.data_f32 ? "yes" : "no",
            derivative.cosine, derivative.relative_l2_difference);
    }
}

static logit_derivative final_logit_derivative(
        llama_context * ctx,
        const operator_results & results,
        const std::string & final,
        int n_top) {
    const auto & plus = require_capture(results.plus_caps, final).data_f32;
    const auto & minus = require_capture(results.minus_caps, final).data_f32;
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    const double epsilon = l2_norm(results.perturbation);
    std::vector<float> rows;
    rows.reserve(plus.size() + minus.size());
    rows.insert(rows.end(), plus.begin(), plus.end());
    rows.insert(rows.end(), minus.begin(), minus.end());
    std::vector<float> logits(2 * n_vocab);
    if (!llama_interp_rwkv_final_readout(ctx, rows.data(), 2, logits.data())) {
        throw std::runtime_error("output-only RWKV final readout failed");
    }

    std::vector<float> derivative(n_vocab);
    for (int token = 0; token < n_vocab; ++token) {
        derivative[token] = (float) ((logits[token] - logits[n_vocab + token]) / (2.0 * epsilon));
    }
    std::vector<int> ids(n_vocab);
    std::iota(ids.begin(), ids.end(), 0);
    n_top = std::min(n_top, n_vocab);
    logit_derivative result;
    result.positive.reserve(n_top);
    std::partial_sort(ids.begin(), ids.begin() + n_top, ids.end(), [&derivative](int a, int b) {
        return derivative[a] > derivative[b];
    });
    for (int i = 0; i < n_top; ++i) {
        result.positive.push_back({ ids[i], common_token_to_piece(llama_model_get_vocab(llama_get_model(ctx)), ids[i], true), derivative[ids[i]] });
    }
    result.negative.reserve(n_top);
    std::partial_sort(ids.begin(), ids.begin() + n_top, ids.end(), [&derivative](int a, int b) {
        return derivative[a] < derivative[b];
    });
    for (int i = 0; i < n_top; ++i) {
        result.negative.push_back({ ids[i], common_token_to_piece(llama_model_get_vocab(llama_get_model(ctx)), ids[i], true), derivative[ids[i]] });
    }
    return result;
}

static void report_final_logit_derivative(const logit_derivative & derivative) {
    std::printf("final logit derivative top positive:\n");
    for (size_t i = 0; i < derivative.positive.size(); ++i) {
        const auto & token = derivative.positive[i];
        std::printf("%2zu  %9.4f  %6d  %s\n", i + 1, token.score, token.id, token.piece.c_str());
    }
    std::printf("final logit derivative top negative:\n");
    for (size_t i = 0; i < derivative.negative.size(); ++i) {
        const auto & token = derivative.negative[i];
        std::printf("%2zu  %9.4f  %6d  %s\n", i + 1, token.score, token.id, token.piece.c_str());
    }
}

static void write_json_string(std::ostream & output, const std::string & value) {
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
                if (c < 0x20) {
                    output << "\\u00" << hex[c >> 4] << hex[c & 0x0f];
                } else {
                    output << c;
                }
        }
    }
    output << '"';
}

static void write_perturbation_trace(
        const std::string & path,
        const std::string & model_path,
        int n_gpu_layers,
        const std::string & prompt,
        const std::string & source,
        const std::string & direction,
        const llama_interp::decode_result & clean,
        const logit_derivative & derivative) {
    if (clean.tokens.size() != 1) {
        throw std::runtime_error("perturbation trace requires one decoded token");
    }
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("failed to open JSON output: " + path);
    }
    const std::string positive_source = source + ".positive";
    const std::string negative_source = source + ".negative";
    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(clean.ctx));
    const std::string next_piece = common_token_to_piece(vocab, clean.tokens[0], true);
    output << std::setprecision(9);
    output << "{\n  \"schema_version\": 1,\n  \"trace_kind\": \"rwkv_perturbation\",\n  \"model_path\": ";
    write_json_string(output, model_path);
    output << ",\n  \"n_gpu_layers\": " << n_gpu_layers << ",\n  \"top_k\": " << derivative.positive.size()
           << ",\n  \"sources\": [";
    write_json_string(output, positive_source);
    output << ", ";
    write_json_string(output, negative_source);
    output << "],\n  \"runs\": [{\n    \"prompt\": ";
    write_json_string(output, prompt);
    output << ",\n    \"prompt_token_count\": 0,\n    \"stopped\": \"perturbation\",\n    \"steps\": [{\n      \"index\": 0,\n      \"input\": {\"id\": -1, \"piece\": ";
    write_json_string(output, source + " / " + direction);
    output << "},\n      \"next\": {\"id\": " << clean.tokens[0] << ", \"piece\": ";
    write_json_string(output, next_piece);
    output << "},\n      \"final_round_trip_error\": 0,\n      \"readouts\": [";
    const std::vector<std::pair<std::string, const std::vector<scored_token> *>> readouts = {
        { positive_source, &derivative.positive }, { negative_source, &derivative.negative },
    };
    for (size_t readout_index = 0; readout_index < readouts.size(); ++readout_index) {
        if (readout_index) output << ", ";
        output << "{\"source\": ";
        write_json_string(output, readouts[readout_index].first);
        output << ", \"top_tokens\": [";
        const auto & tokens = *readouts[readout_index].second;
        for (size_t token_index = 0; token_index < tokens.size(); ++token_index) {
            if (token_index) output << ", ";
            const auto & token = tokens[token_index];
            output << "{\"rank\": " << token_index + 1 << ", \"id\": " << token.id << ", \"piece\": ";
            write_json_string(output, token.piece);
            output << ", \"logit\": " << token.score << "}";
        }
        output << "]}";
    }
    output << "]\n    }]\n  }]\n}\n";
    if (!output) {
        throw std::runtime_error("failed to write JSON output: " + path);
    }
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::string model_path;
    std::string prompt = "The Eiffel Tower is located in";
    int n_gpu_layers = 0;
    int source_layer = -1;
    std::string source_activation = "residual";
    std::string center_root_path;
    std::string direction_prompt;
    std::string json_path;
    float epsilon = 0.1f;
    float compare_epsilon = 0.0f;
    int n_top = 5;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (std::strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (std::strcmp(argv[i], "-ngl") == 0 && i + 1 < argc) {
            n_gpu_layers = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--layer") == 0 && i + 1 < argc) {
            source_layer = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--source-activation") == 0 && i + 1 < argc) {
            source_activation = argv[++i];
        } else if (std::strcmp(argv[i], "--center-root") == 0 && i + 1 < argc) {
            center_root_path = argv[++i];
        } else if (std::strcmp(argv[i], "--direction-prompt") == 0 && i + 1 < argc) {
            direction_prompt = argv[++i];
        } else if (std::strcmp(argv[i], "--json") == 0 && i + 1 < argc) {
            json_path = argv[++i];
        } else if (std::strcmp(argv[i], "--epsilon") == 0 && i + 1 < argc) {
            epsilon = std::strtof(argv[++i], nullptr);
        } else if (std::strcmp(argv[i], "--compare-epsilon") == 0 && i + 1 < argc) {
            compare_epsilon = std::strtof(argv[++i], nullptr);
        } else if (std::strcmp(argv[i], "--top") == 0 && i + 1 < argc) {
            n_top = std::atoi(argv[++i]);
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    if (model_path.empty() || epsilon <= 0.0f || compare_epsilon < 0.0f || n_top <= 0) {
        usage(argv[0]);
        return 1;
    }
    if (!center_root_path.empty() && !direction_prompt.empty()) {
        std::fprintf(stderr, "--center-root and --direction-prompt are mutually exclusive\n");
        return 1;
    }

    ggml_backend_load_all();
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        std::fprintf(stderr, "failed to load model: %s\n", model_path.c_str());
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), (int) prompt.size(), nullptr, 0, false, true);
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = std::max(128, n_prompt + 8);
    cparams.n_batch = n_prompt;
    cparams.n_ubatch = n_prompt;
    cparams.n_seq_max = 1;
    cparams.no_perf = false;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::fprintf(stderr, "failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    llama_interp::runtime rt(ctx, 1);
    const llama_interp::rwkv_state initial_state = rt.make_state();
    if (source_layer < 0) {
        source_layer = (int) initial_state.n_layer / 2;
    }
    if (source_layer < 0 || source_layer >= (int) initial_state.n_layer) {
        std::fprintf(stderr, "invalid source layer %d for %u-layer model\n", source_layer, initial_state.n_layer);
        return 1;
    }
    try {
        source_name(source_layer, source_activation);
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    if (!center_root_path.empty() && time_source_index(source_activation) < 0) {
        std::fprintf(stderr, "--center-root requires a raw time-mix source activation\n");
        return 1;
    }

    std::vector<float> reference;
    if (!center_root_path.empty()) {
        const uint32_t source_index = (uint32_t) (source_layer * 8 + time_source_index(source_activation));
        const auto reader = rwkv_activation_store::centroid_store_reader::open(center_root_path);
        const auto centers = reader.read_source(source_index, 1);
        if (centers.size() != 1) {
            throw std::runtime_error("centroid file does not contain one K=1 center for source");
        }
        reference = centers[0].values;
    } else if (!direction_prompt.empty()) {
        llama_interp::activation_set direction_caps;
        auto capture = capture_prompt_source(rt, initial_state, direction_prompt, source_name(source_layer, source_activation), direction_caps);
        rt.run();
        capture.rethrow_if_failed();
        reference = last_capture_f32(direction_caps, source_name(source_layer, source_activation));
    }

    llama_interp::rwkv_state prompt_state;
    auto prefill_task = prefill_prompt(rt, initial_state, prompt, prompt_state);
    rt.run();
    prefill_task.rethrow_if_failed();

    operator_results results;
    const std::vector<float> * reference_ptr = center_root_path.empty() && direction_prompt.empty() ? nullptr : &reference;
    auto task = run_symmetric_check(rt, prompt_state, initial_state.n_layer, source_layer, source_activation, reference_ptr, epsilon, results);
    rt.run();
    task.rethrow_if_failed();

    const std::string source = source_name(source_layer, source_activation);
    const std::string source_capture = source_activation == "residual" ? source + ".pre" : source;
    const std::string time_out = time_out_name(source_layer);
    const std::string final = residual_name((int) initial_state.n_layer - 1);
    const auto & clean_source = require_capture(results.clean_caps, source_capture);
    const auto & plus_source = require_capture(results.plus_caps, source);
    const double actual_epsilon = l2_norm(results.perturbation);

    std::printf("source=%s\n", source.c_str());
    const std::string direction = center_root_path.empty() ? (direction_prompt.empty() ? "rademacher" : "source-minus-prompt") : "source-minus-k1-center";
    std::printf("direction=%s\n", direction.c_str());
    std::printf("epsilon requested=%g actual_l2=%g perturb_elements=%zu\n",
        epsilon, actual_epsilon, results.perturbation.size());
    std::printf("source addition l2=%.6e error_l2=%.6e\n",
        difference_l2(plus_source.data_f32, clean_source.data_f32),
        perturbation_error_l2(clean_source.data_f32, plus_source.data_f32, results.perturbation));
    std::printf("clean token=%s\n", results.clean.to_string().c_str());

    operator_results comparison_results;
    if (compare_epsilon > 0.0f) {
        auto comparison_task = run_symmetric_check(
            rt, prompt_state, initial_state.n_layer, source_layer, source_activation, reference_ptr, compare_epsilon, comparison_results);
        rt.run();
        comparison_task.rethrow_if_failed();
        std::printf("compare epsilon requested=%g actual_l2=%g\n", compare_epsilon, l2_norm(comparison_results.perturbation));
    }

    const operator_results * comparison = compare_epsilon > 0.0f ? &comparison_results : nullptr;
    report_target("time.out", time_out, results, comparison);
    report_target("final resid.out", final, results, comparison);
    const logit_derivative derivative = final_logit_derivative(ctx, results, final, n_top);
    report_final_logit_derivative(derivative);
    if (!json_path.empty()) {
        write_perturbation_trace(json_path, model_path, n_gpu_layers, prompt, source, direction, results.clean, derivative);
        std::printf("json=%s\n", json_path.c_str());
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
