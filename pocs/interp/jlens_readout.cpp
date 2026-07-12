#include "interp.hpp"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

struct evaluation_result {
    llama_interp::activation_set captures;
};

struct sample_spec {
    size_t corpus_index;
    size_t source_index;
    size_t target_index;
};

struct average_result {
    std::vector<float> action;
    std::vector<double> response_l2;
    std::vector<double> symmetry_error;
    std::vector<double> repeat_plus_error;
};

struct operator_artifact {
    int source_layer = -1;
    int target_layer = -1;
    std::string source_activation = "residual";
    size_t input_dimension = 0;
    size_t output_dimension = 0;
    size_t rank = 0;
    std::string directions_path;
    std::string responses_path;
    std::vector<float> directions;
    std::vector<float> responses;
};

static void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s -m MODEL --output PREFIX [--operator PREFIX] [-p PROMPT] [-ngl N] [--layer N] [--epsilon E] [--compare-epsilon E] [--samples N] [--corpus FILE] [--max-corpus-tokens N] [--min-future N] [--future-window N] [--repeat-plus]\n"
        "\n"
        "Estimates one J-lens readout by averaging finite differences along a held-out activation,\n"
        "or applies a saved low-rank operator to the prompt's final-token residual.\n",
        argv0);
}

static std::string residual_name(int layer) {
    return "rwkv.layer." + std::to_string(layer) + ".resid.out";
}

static std::string source_name(int layer, const std::string & activation) {
    if (activation == "residual") {
        return residual_name(layer);
    }
    if (activation == "time-wkv") {
        return "rwkv.layer." + std::to_string(layer) + ".time.wkv";
    }
    throw std::runtime_error("unknown source activation in operator: " + activation);
}

static std::string manifest_path(const std::string & prefix) {
    return prefix.ends_with(".txt") ? prefix : prefix + ".txt";
}

static void read_f32(const std::string & path, std::vector<float> & values, size_t count) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open operator data: " + path);
    }
    values.resize(count);
    input.read((char *) values.data(), (std::streamsize) (count * sizeof(float)));
    if (input.gcount() != (std::streamsize) (count * sizeof(float)) || input.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error("unexpected operator data size: " + path);
    }
}

static operator_artifact load_operator(const std::string & prefix) {
    operator_artifact out;
    std::ifstream manifest(manifest_path(prefix));
    if (!manifest) {
        throw std::runtime_error("failed to open operator manifest: " + manifest_path(prefix));
    }

    std::string line;
    while (std::getline(manifest, line)) {
        const size_t separator = line.find('=');
        if (separator == std::string::npos) {
            continue;
        }
        const std::string key = line.substr(0, separator);
        const std::string value = line.substr(separator + 1);
        if (key == "source_layer") {
            out.source_layer = std::stoi(value);
        } else if (key == "target_layer") {
            out.target_layer = std::stoi(value);
        } else if (key == "source_activation") {
            out.source_activation = value;
        } else if (key == "input_dimension") {
            out.input_dimension = std::stoull(value);
        } else if (key == "output_dimension") {
            out.output_dimension = std::stoull(value);
        } else if (key == "rank") {
            out.rank = std::stoull(value);
        } else if (key == "directions_f32") {
            out.directions_path = value;
        } else if (key == "responses_f32") {
            out.responses_path = value;
        }
    }
    if (out.source_layer < 0 || out.target_layer < 0 || out.input_dimension == 0 || out.output_dimension == 0 || out.rank == 0 ||
        out.directions_path.empty() || out.responses_path.empty()) {
        throw std::runtime_error("operator manifest is missing required fields: " + manifest_path(prefix));
    }
    read_f32(out.directions_path, out.directions, out.rank * out.input_dimension);
    read_f32(out.responses_path, out.responses, out.rank * out.output_dimension);
    return out;
}

static std::vector<float> apply_operator(const operator_artifact & op, const std::vector<float> & input) {
    if (input.size() != op.input_dimension) {
        throw std::runtime_error("operator input dimension differs from captured residual");
    }
    std::vector<float> out(op.output_dimension, 0.0f);
    const double scale = (double) op.input_dimension / op.rank;
    for (size_t k = 0; k < op.rank; ++k) {
        double dot = 0.0;
        for (size_t i = 0; i < op.input_dimension; ++i) {
            dot += (double) op.directions[k * op.input_dimension + i] * input[i];
        }
        for (size_t i = 0; i < op.output_dimension; ++i) {
            out[i] += (float) (scale * dot * op.responses[k * op.output_dimension + i]);
        }
    }
    return out;
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

static const std::vector<float> & require_capture_f32(
        const llama_interp::activation_set & caps,
        const std::string & name) {
    const auto & capture = require_capture(caps, name);
    if (capture.data_f32.empty()) {
        throw std::runtime_error("missing FP32 activation capture: " + name);
    }
    return capture.data_f32;
}

static double l2_norm(const std::vector<ggml_fp16_t> & values) {
    double sum = 0.0;
    for (ggml_fp16_t value : values) {
        const double x = ggml_fp16_to_fp32(value);
        sum += x*x;
    }
    return std::sqrt(sum);
}

static double l2_norm(const std::vector<float> & values) {
    double sum = 0.0;
    for (float value : values) {
        sum += (double) value * value;
    }
    return std::sqrt(sum);
}

static double relative_l2_difference(
        const std::vector<float> & a,
        const std::vector<ggml_fp16_t> & b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("activation dimensions differ");
    }

    double difference = 0.0;
    double reference = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double x = a[i] - ggml_fp16_to_fp32(b[i]);
        difference += x*x;
        const double y = ggml_fp16_to_fp32(b[i]);
        reference += y*y;
    }
    return std::sqrt(difference) / std::max(std::sqrt(reference), 1e-30);
}

static std::vector<ggml_fp16_t> normalized_perturbation(
        const std::vector<ggml_fp16_t> & activation,
        float epsilon) {
    const double norm = l2_norm(activation);
    if (norm == 0.0) {
        throw std::runtime_error("cannot perturb a zero activation");
    }

    std::vector<ggml_fp16_t> out;
    out.reserve(activation.size());
    for (ggml_fp16_t value : activation) {
        out.push_back(ggml_fp32_to_fp16((float) (epsilon * ggml_fp16_to_fp32(value) / norm)));
    }
    return out;
}

static std::vector<ggml_fp16_t> normalized_perturbation(
        const std::vector<float> & activation,
        float epsilon) {
    const double norm = l2_norm(activation);
    if (norm == 0.0) {
        throw std::runtime_error("cannot perturb a zero activation");
    }

    std::vector<ggml_fp16_t> out;
    out.reserve(activation.size());
    for (float value : activation) {
        out.push_back(ggml_fp32_to_fp16((float) (epsilon * value / norm)));
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

static uint64_t splitmix64(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

static sample_spec choose_sample(
        const std::vector<llama_token> & tokens,
        int min_future,
        int future_window,
        size_t sample_index) {
    const uint64_t seed = splitmix64(sample_index);
    if (tokens.size() <= (size_t) min_future) {
        throw std::runtime_error("corpus sample is too short for the requested future offset");
    }
    const size_t source_index = (size_t) (splitmix64(seed) % (tokens.size() - (size_t) min_future));
    const size_t min_target = source_index + (size_t) min_future;
    const size_t max_target = std::min(tokens.size() - 1, source_index + (size_t) future_window);
    const size_t target_index = min_target + (size_t) (splitmix64(splitmix64(seed)) % (max_target - min_target + 1));
    return { 0, source_index, target_index };
}

static sample_spec choose_sample(
        const std::vector<std::vector<llama_token>> & corpus,
        int min_future,
        int future_window,
        size_t sample_index) {
    const size_t corpus_index = (size_t) (splitmix64(sample_index) % corpus.size());
    sample_spec spec = choose_sample(corpus[corpus_index], min_future, future_window, sample_index);
    spec.corpus_index = corpus_index;
    return spec;
}

static llama_interp::task<> capture_evaluation(
        llama_interp::runtime & rt,
        const llama_interp::rwkv_state & initial_state,
        const std::vector<llama_token> & tokens,
        const std::string & source,
        int target_layer,
        evaluation_result & out) {
    using namespace llama_interp;

    const std::string target = residual_name(target_layer);
    auto prefill = rt.prefill_tokens(initial_state, tokens);
    prefill.capture("^" + source + "$", out.captures);
    if (source != target) {
        prefill.capture("^" + target + "$", out.captures);
    }
    prefill.discard_state();
    co_await prefill;
}

static llama_interp::task<> prefill_tokens(
        llama_interp::runtime & rt,
        const llama_interp::rwkv_state & initial_state,
        const std::vector<llama_token> & tokens,
        llama_interp::rwkv_state & out) {
    out = co_await rt.prefill_tokens(initial_state, tokens);
}

static llama_interp::task<> evaluate_head(
        llama_interp::runtime & rt,
        const llama_interp::rwkv_state & initial_state,
        llama_token token,
        int target_layer,
        const std::vector<ggml_fp16_t> & final_residual) {
    using namespace llama_interp;

    llama_interp_perturb_spec replace;
    replace.op = LLAMA_INTERP_PERTURB_REPLACE;
    replace.head = -1;
    replace.data = final_residual;

    // A replacement at an intermediate target is propagated through the remaining RWKV blocks.
    // Starting from a fresh state avoids coupling the readout to the evaluation prompt's next token.
    auto carrier = rt.prefill_tokens(initial_state, { token });
    carrier.perturb("^" + residual_name(target_layer) + "$", std::move(replace));
    carrier.discard_state();
    co_await carrier;
}

static llama_interp::task<> measure_sample(
        llama_interp::runtime & rt,
        const llama_interp::rwkv_state & before_source,
        const std::vector<llama_token> & tokens,
        const sample_spec & spec,
        int source_layer,
        const std::vector<ggml_fp16_t> & perturbation,
        double query_norm,
        std::vector<float> & action_sum,
        double & response_l2,
        double & symmetry_error,
        bool repeat_plus,
        double & repeat_plus_error) {
    using namespace llama_interp;

    const std::string source = residual_name(source_layer);
    const std::string final = residual_name((int) before_source.n_layer - 1);
    const std::vector<llama_token> span(
        tokens.begin() + spec.source_index,
        tokens.begin() + spec.target_index + 1);

    activation_set clean_caps;
    {
        auto clean_path = rt.prefill_tokens(before_source, span);
        clean_path.capture("^" + final + "$", clean_caps);
        clean_path.discard_state();
        co_await clean_path;
    }

    activation_set plus_caps;
    {
        auto plus_path = rt.prefill_tokens(before_source, span);
        plus_path.perturb("^" + source + "$", runtime::add_head(-1, perturbation));
        plus_path.capture("^" + final + "$", plus_caps);
        plus_path.discard_state();
        co_await plus_path;
    }

    activation_set repeat_plus_caps;
    if (repeat_plus) {
        auto repeat_plus_path = rt.prefill_tokens(before_source, span);
        repeat_plus_path.perturb("^" + source + "$", runtime::add_head(-1, perturbation));
        repeat_plus_path.capture("^" + final + "$", repeat_plus_caps);
        repeat_plus_path.discard_state();
        co_await repeat_plus_path;
    }

    activation_set minus_caps;
    {
        auto minus_path = rt.prefill_tokens(before_source, span);
        minus_path.perturb("^" + source + "$", runtime::add_head(-1, negate(perturbation)));
        minus_path.capture("^" + final + "$", minus_caps);
        minus_path.discard_state();
        co_await minus_path;
    }

    const auto & clean = require_capture_f32(clean_caps, final);
    const auto & plus = require_capture_f32(plus_caps, final);
    const auto & minus = require_capture_f32(minus_caps, final);
    if (clean.size() != action_sum.size() || plus.size() != action_sum.size() || minus.size() != action_sum.size()) {
        throw std::runtime_error("inconsistent target activation dimensions");
    }

    const double actual_epsilon = l2_norm(perturbation);
    double response_squared_norm = 0.0;
    double even_squared_norm = 0.0;
    double odd_squared_norm = 0.0;
    for (size_t i = 0; i < action_sum.size(); ++i) {
        const double y0 = clean[i];
        const double yp = plus[i];
        const double ym = minus[i];
        const float response = (float) ((yp - ym) / (2.0 * actual_epsilon) * query_norm);
        action_sum[i] += response;
        response_squared_norm += (double) response * response;
        const double even = (yp - y0) + (ym - y0);
        const double odd = (yp - y0) - (ym - y0);
        even_squared_norm += even * even;
        odd_squared_norm += odd * odd;
    }
    response_l2 = std::sqrt(response_squared_norm);
    symmetry_error = std::sqrt(even_squared_norm) / std::max(std::sqrt(odd_squared_norm), 1e-30);
    repeat_plus_error = 0.0;
    if (repeat_plus) {
        const auto & repeated = require_capture_f32(repeat_plus_caps, final);
        if (repeated.size() != plus.size()) {
            throw std::runtime_error("inconsistent repeated-plus activation dimensions");
        }
        double squared_difference = 0.0;
        for (size_t i = 0; i < plus.size(); ++i) {
            const double delta = plus[i] - repeated[i];
            squared_difference += delta * delta;
        }
        repeat_plus_error = std::sqrt(squared_difference);
    }
}

static void average_actions(
        llama_interp::runtime & rt,
        const llama_interp::rwkv_state & initial_state,
        const std::vector<std::vector<llama_token>> & corpus,
        const std::vector<sample_spec> & sample_specs,
        int source_layer,
        const std::vector<float> & query,
        size_t dimension,
        float epsilon,
        bool repeat_plus,
        average_result & primary,
        float compare_epsilon,
        average_result * comparison) {
    const double query_norm = l2_norm(query);
    const std::vector<ggml_fp16_t> perturbation = normalized_perturbation(query, epsilon);
    const std::vector<ggml_fp16_t> comparison_perturbation = comparison ? normalized_perturbation(query, compare_epsilon) : std::vector<ggml_fp16_t>();

    auto initialize = [&sample_specs, dimension](average_result & out) {
        out.action.assign(dimension, 0.0f);
        out.response_l2.reserve(sample_specs.size());
        out.symmetry_error.reserve(sample_specs.size());
        out.repeat_plus_error.reserve(sample_specs.size());
    };
    initialize(primary);
    if (comparison) {
        initialize(*comparison);
    }

    auto add_measurement = [&rt, source_layer, query_norm, repeat_plus](
            const llama_interp::rwkv_state & before_source,
            const std::vector<llama_token> & tokens,
            const sample_spec & spec,
            const std::vector<ggml_fp16_t> & perturbation,
            average_result & out) {
        double response_l2 = 0.0;
        double symmetry_error = 0.0;
        double repeat_plus_error = 0.0;
        {
            auto sample_task = measure_sample(
                rt, before_source, tokens, spec, source_layer, perturbation, query_norm,
                out.action, response_l2, symmetry_error, repeat_plus, repeat_plus_error);
            rt.run();
            sample_task.rethrow_if_failed();
        }
        out.response_l2.push_back(response_l2);
        out.symmetry_error.push_back(symmetry_error);
        out.repeat_plus_error.push_back(repeat_plus_error);
    };

    for (const auto & spec : sample_specs) {
        llama_interp::rwkv_state before_source = initial_state;
        if (spec.source_index > 0) {
            auto prefix_task = prefill_tokens(
                rt, initial_state,
                std::vector<llama_token>(corpus[spec.corpus_index].begin(), corpus[spec.corpus_index].begin() + spec.source_index),
                before_source);
            rt.run();
            prefix_task.rethrow_if_failed();
        }

        add_measurement(before_source, corpus[spec.corpus_index], spec, perturbation, primary);
        if (comparison) {
            add_measurement(before_source, corpus[spec.corpus_index], spec, comparison_perturbation, *comparison);
        }
    }
    for (float & value : primary.action) {
        value /= (float) sample_specs.size();
    }
    if (comparison) {
        for (float & value : comparison->action) {
            value /= (float) sample_specs.size();
        }
    }
}

static int average_actions_streaming(
        llama_interp::runtime & rt,
        const llama_interp::rwkv_state & initial_state,
        const llama_vocab * vocab,
        const std::string & corpus_path,
        int max_corpus_tokens,
        int requested_samples,
        int min_future,
        int future_window,
        int source_layer,
        const std::vector<float> & query,
        size_t dimension,
        float epsilon,
        bool repeat_plus,
        average_result & primary,
        float compare_epsilon,
        average_result * comparison,
        std::vector<sample_spec> & sample_specs) {
    std::ifstream input(corpus_path);
    if (!input) {
        throw std::runtime_error("failed to open corpus: " + corpus_path);
    }

    const double query_norm = l2_norm(query);
    const std::vector<ggml_fp16_t> perturbation = normalized_perturbation(query, epsilon);
    const std::vector<ggml_fp16_t> comparison_perturbation = comparison ? normalized_perturbation(query, compare_epsilon) : std::vector<ggml_fp16_t>();
    primary.action.assign(dimension, 0.0f);
    primary.response_l2.reserve(requested_samples);
    primary.symmetry_error.reserve(requested_samples);
    primary.repeat_plus_error.reserve(requested_samples);
    if (comparison) {
        comparison->action.assign(dimension, 0.0f);
        comparison->response_l2.reserve(requested_samples);
        comparison->symmetry_error.reserve(requested_samples);
        comparison->repeat_plus_error.reserve(requested_samples);
    }

    auto add_measurement = [&rt, source_layer, query_norm, repeat_plus](
            const llama_interp::rwkv_state & before_source,
            const std::vector<llama_token> & tokens,
            const sample_spec & spec,
            const std::vector<ggml_fp16_t> & delta,
            average_result & out) {
        double response_l2 = 0.0;
        double symmetry_error = 0.0;
        double repeat_plus_error = 0.0;
        {
            auto sample_task = measure_sample(
                rt, before_source, tokens, spec, source_layer, delta, query_norm,
                out.action, response_l2, symmetry_error, repeat_plus, repeat_plus_error);
            rt.run();
            sample_task.rethrow_if_failed();
        }
        out.response_l2.push_back(response_l2);
        out.symmetry_error.push_back(symmetry_error);
        out.repeat_plus_error.push_back(repeat_plus_error);
    };

    std::string line;
    size_t line_number = 0;
    while ((int) sample_specs.size() < requested_samples && std::getline(input, line)) {
        ++line_number;
        auto tokens = common_tokenize(vocab, line, false, true);
        if ((int) tokens.size() > max_corpus_tokens) {
            tokens.resize(max_corpus_tokens);
        }
        if (tokens.size() <= (size_t) min_future) {
            continue;
        }

        sample_spec spec = choose_sample(tokens, min_future, future_window, sample_specs.size());
        spec.corpus_index = line_number;
        llama_interp::rwkv_state before_source = initial_state;
        if (spec.source_index > 0) {
            auto prefix_task = prefill_tokens(
                rt, initial_state,
                std::vector<llama_token>(tokens.begin(), tokens.begin() + spec.source_index),
                before_source);
            rt.run();
            prefix_task.rethrow_if_failed();
        }

        add_measurement(before_source, tokens, spec, perturbation, primary);
        if (comparison) {
            add_measurement(before_source, tokens, spec, comparison_perturbation, *comparison);
        }
        sample_specs.push_back(spec);
    }

    if (sample_specs.empty()) {
        throw std::runtime_error("corpus produced no valid samples");
    }
    for (float & value : primary.action) {
        value /= (float) sample_specs.size();
    }
    if (comparison) {
        for (float & value : comparison->action) {
            value /= (float) sample_specs.size();
        }
    }
    return (int) sample_specs.size();
}

static std::vector<float> copy_logits(const llama_context * ctx, int n_vocab) {
    const float * logits = llama_get_logits_ith(const_cast<llama_context *>(ctx), 0);
    if (!logits) {
        throw std::runtime_error("missing logits");
    }
    return std::vector<float>(logits, logits + n_vocab);
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

static double cosine_similarity(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("action dimensions differ");
    }

    double dot = 0.0;
    double norm_a = 0.0;
    double norm_b = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += (double) a[i] * b[i];
        norm_a += (double) a[i] * a[i];
        norm_b += (double) b[i] * b[i];
    }
    return dot / std::max(std::sqrt(norm_a * norm_b), 1e-30);
}

static double relative_l2_difference(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("action dimensions differ");
    }

    double difference = 0.0;
    double reference = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double delta = a[i] - b[i];
        difference += delta * delta;
        reference += (double) b[i] * b[i];
    }
    return std::sqrt(difference) / std::max(std::sqrt(reference), 1e-30);
}

static std::vector<int> top_tokens(const std::vector<float> & logits, int n_top) {
    std::vector<int> ids(logits.size());
    std::iota(ids.begin(), ids.end(), 0);
    std::partial_sort(ids.begin(), ids.begin() + n_top, ids.end(), [&logits](int a, int b) {
        return logits[a] > logits[b];
    });
    ids.resize(n_top);
    return ids;
}

static int top_overlap(const std::vector<float> & a, const std::vector<float> & b, int n_top) {
    const auto top_a = top_tokens(a, n_top);
    const auto top_b = top_tokens(b, n_top);
    int overlap = 0;
    for (int token : top_a) {
        overlap += std::find(top_b.begin(), top_b.end(), token) != top_b.end();
    }
    return overlap;
}

static void print_top_tokens(const llama_vocab * vocab, const std::vector<float> & logits, int n_top) {
    const auto ids = top_tokens(logits, n_top);

    for (int i = 0; i < n_top; ++i) {
        const int token = ids[i];
        std::printf("%2d  %.6f  %d  %s\n", i + 1, logits[token], token,
            common_token_to_piece(vocab, token, true).c_str());
    }
}

static void print_response_summary(
        const std::vector<sample_spec> & sample_specs,
        const std::vector<double> & response_l2,
        const std::vector<double> & symmetry_error) {
    double current_sum = 0.0;
    double future_sum = 0.0;
    double current_symmetry_sum = 0.0;
    double future_symmetry_sum = 0.0;
    int n_current = 0;
    int n_future = 0;
    for (size_t i = 0; i < sample_specs.size(); ++i) {
        if (sample_specs[i].target_index == sample_specs[i].source_index) {
            current_sum += response_l2[i];
            current_symmetry_sum += symmetry_error[i];
            ++n_current;
        } else {
            future_sum += response_l2[i];
            future_symmetry_sum += symmetry_error[i];
            ++n_future;
        }
    }
    if (n_current > 0) {
        std::printf("current-target response_l2 mean=%g symmetry mean=%g n=%d\n",
            current_sum / n_current, current_symmetry_sum / n_current, n_current);
    }
    if (n_future > 0) {
        std::printf("future-target response_l2 mean=%g symmetry mean=%g n=%d\n",
            future_sum / n_future, future_symmetry_sum / n_future, n_future);
    }
}

static std::vector<ggml_fp16_t> as_fp16(const std::vector<float> & values) {
    std::vector<ggml_fp16_t> out;
    out.reserve(values.size());
    for (float value : values) {
        out.push_back(ggml_fp32_to_fp16(value));
    }
    return out;
}

static void write_f16(const std::string & path, const std::vector<float> & values) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open output: " + path);
    }
    for (float value : values) {
        const ggml_fp16_t fp16 = ggml_fp32_to_fp16(value);
        out.write((const char *) &fp16, sizeof(fp16));
    }
    if (!out) {
        throw std::runtime_error("failed to write output: " + path);
    }
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::string model_path;
    std::string output_prefix;
    std::string corpus_path;
    std::string operator_prefix;
    std::string prompt = "The Eiffel Tower is located in";
    int n_gpu_layers = 0;
    int source_layer = -1;
    float epsilon = 0.2f;
    float compare_epsilon = 0.0f;
    int samples = 4;
    int min_future = 0;
    int future_window = 2;
    int max_corpus_tokens = 128;
    bool repeat_plus = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_prefix = argv[++i];
        } else if (std::strcmp(argv[i], "--operator") == 0 && i + 1 < argc) {
            operator_prefix = argv[++i];
        } else if (std::strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (std::strcmp(argv[i], "-ngl") == 0 && i + 1 < argc) {
            n_gpu_layers = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--layer") == 0 && i + 1 < argc) {
            source_layer = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--epsilon") == 0 && i + 1 < argc) {
            epsilon = std::strtof(argv[++i], nullptr);
        } else if (std::strcmp(argv[i], "--compare-epsilon") == 0 && i + 1 < argc) {
            compare_epsilon = std::strtof(argv[++i], nullptr);
        } else if (std::strcmp(argv[i], "--samples") == 0 && i + 1 < argc) {
            samples = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--corpus") == 0 && i + 1 < argc) {
            corpus_path = argv[++i];
        } else if (std::strcmp(argv[i], "--max-corpus-tokens") == 0 && i + 1 < argc) {
            max_corpus_tokens = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--min-future") == 0 && i + 1 < argc) {
            min_future = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--future-window") == 0 && i + 1 < argc) {
            future_window = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--repeat-plus") == 0) {
            repeat_plus = true;
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    if (model_path.empty() || output_prefix.empty() || epsilon <= 0.0f || compare_epsilon < 0.0f || samples <= 0 ||
        min_future < 0 || future_window < min_future || max_corpus_tokens <= min_future) {
        usage(argv[0]);
        return 1;
    }

    const std::vector<std::string> corpus_text = {
        "The Pacific Ocean is the largest ocean on Earth.",
        "A triangle has three sides and three angles.",
        "Plants use sunlight to convert water and carbon dioxide into energy.",
        "The Moon orbits the Earth approximately once every month.",
        "A library organizes books so that readers can find information.",
        "Copper conducts electricity and is used in many electrical wires.",
        "Rain forms when water vapor cools and condenses into droplets.",
        "The decimal system uses ten digits to represent numbers.",
    };

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
    const std::vector<llama_token> evaluation_tokens = common_tokenize(vocab, prompt, false, true);
    std::vector<std::vector<llama_token>> corpus;
    size_t max_tokens = evaluation_tokens.size();
    for (const auto & text : corpus_text) {
        auto tokens = common_tokenize(vocab, text, false, true);
        if (tokens.empty()) {
            throw std::runtime_error("corpus text tokenized to zero tokens");
        }
        max_tokens = std::max(max_tokens, tokens.size());
        corpus.push_back(std::move(tokens));
    }
    if (!corpus_path.empty()) {
        max_tokens = std::max(max_tokens, (size_t) max_corpus_tokens);
    }
    if (evaluation_tokens.empty()) {
        throw std::runtime_error("prompt tokenized to zero tokens");
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = std::max(128, (int) max_tokens + 8);
    cparams.n_batch = (int) max_tokens;
    cparams.n_ubatch = (int) max_tokens;
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
    operator_artifact saved_operator;
    if (!operator_prefix.empty()) {
        saved_operator = load_operator(operator_prefix);
        if (source_layer >= 0 && source_layer != saved_operator.source_layer) {
            throw std::runtime_error("--layer does not match the saved operator source layer");
        }
        source_layer = saved_operator.source_layer;
    } else if (source_layer < 0) {
        source_layer = (int) initial_state.n_layer / 2;
    }
    if (source_layer < 0 || source_layer >= (int) initial_state.n_layer) {
        std::fprintf(stderr, "invalid source layer %d for %u-layer model\n", source_layer, initial_state.n_layer);
        return 1;
    }

    evaluation_result evaluation;
    {
        const int target_layer = operator_prefix.empty() ? (int) initial_state.n_layer - 1 : saved_operator.target_layer;
        const std::string operator_source = operator_prefix.empty() ? residual_name(source_layer) : source_name(source_layer, saved_operator.source_activation);
        auto evaluation_task = capture_evaluation(rt, initial_state, evaluation_tokens, operator_source, target_layer, evaluation);
        rt.run();
        evaluation_task.rethrow_if_failed();
    }
    const std::vector<float> native_logits = copy_logits(ctx, llama_vocab_n_tokens(vocab));

    const std::string source = operator_prefix.empty() ? residual_name(source_layer) : source_name(source_layer, saved_operator.source_activation);
    const int target_layer = operator_prefix.empty() ? (int) initial_state.n_layer - 1 : saved_operator.target_layer;
    const std::string final = residual_name(target_layer);
    const auto & query = require_capture_f32(evaluation.captures, source);
    const auto & clean_final = require_capture_f32(evaluation.captures, final);
    const double query_norm = l2_norm(query);

    if (!operator_prefix.empty()) {
        if (saved_operator.output_dimension != clean_final.size()) {
            throw std::runtime_error("operator output dimension differs from the final residual");
        }
        const std::vector<float> action = apply_operator(saved_operator, query);
        {
            auto lens_task = evaluate_head(rt, initial_state, evaluation_tokens.back(), saved_operator.target_layer, as_fp16(action));
            rt.run();
            lens_task.rethrow_if_failed();
        }
        const std::vector<float> lens_logits = copy_logits(ctx, llama_vocab_n_tokens(vocab));
        const std::string action_path = output_prefix + ".action.f16";
        const std::string readout_manifest_path = output_prefix + ".txt";
        write_f16(action_path, action);
        std::ofstream manifest(readout_manifest_path);
        if (!manifest) {
            throw std::runtime_error("failed to open output: " + readout_manifest_path);
        }
        manifest << "format=rwkv_jlens_applied_readout_v1\n";
        manifest << "operator_manifest=" << manifest_path(operator_prefix) << '\n';
        manifest << "source_layer=" << source_layer << '\n';
        manifest << "source_activation=" << saved_operator.source_activation << '\n';
        manifest << "target_layer=" << saved_operator.target_layer << '\n';
        manifest << "dimension=" << action.size() << '\n';
        manifest << "rank=" << saved_operator.rank << '\n';
        manifest << "prompt=" << prompt << '\n';
        manifest << "action_f16=" << action_path << '\n';

        std::printf("applied operator source=%s target=%s rank=%zu prompt_tokens=%zu\n",
            source.c_str(), final.c_str(), saved_operator.rank, evaluation_tokens.size());
        std::printf("source_l2=%g action_l2=%g\n", query_norm, l2_norm(action));
        std::printf("--- Applied J-lens top tokens ---\n");
        print_top_tokens(vocab, lens_logits, 10);
        std::printf("wrote %s\n", readout_manifest_path.c_str());

        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 0;
    }

    std::vector<sample_spec> sample_specs;
    sample_specs.reserve(samples);
    average_result primary;
    average_result comparison;
    int completed_samples = samples;
    if (corpus_path.empty()) {
        for (int isample = 0; isample < samples; ++isample) {
            const sample_spec spec = choose_sample(corpus, min_future, future_window, (size_t) isample);
            sample_specs.push_back(spec);
        }
        average_actions(
            rt, initial_state, corpus, sample_specs, source_layer, query, clean_final.size(), epsilon, repeat_plus,
            primary, compare_epsilon, compare_epsilon > 0.0f ? &comparison : nullptr);
    } else {
        completed_samples = average_actions_streaming(
            rt, initial_state, vocab, corpus_path, max_corpus_tokens, samples, min_future, future_window,
            source_layer, query, clean_final.size(), epsilon, repeat_plus,
            primary, compare_epsilon, compare_epsilon > 0.0f ? &comparison : nullptr, sample_specs);
    }
    for (size_t i = 0; i < sample_specs.size(); ++i) {
        const auto & spec = sample_specs[i];
        std::printf("sample %zu/%d corpus=%zu source=%zu target=%zu response_l2=%g symmetry=%g repeat_plus_l2=%g\n",
            i + 1, completed_samples, spec.corpus_index, spec.source_index, spec.target_index,
            primary.response_l2[i], primary.symmetry_error[i], primary.repeat_plus_error[i]);
    }

    // Complete all stateful Jacobian measurements before carrier passes used solely for readout.
    {
        auto round_trip_task = evaluate_head(rt, initial_state, evaluation_tokens.back(), target_layer, as_fp16(clean_final));
        rt.run();
        round_trip_task.rethrow_if_failed();
    }
    const std::vector<float> round_trip_logits = copy_logits(ctx, llama_vocab_n_tokens(vocab));

    {
        auto lens_task = evaluate_head(rt, initial_state, evaluation_tokens.back(), target_layer, as_fp16(primary.action));
        rt.run();
        lens_task.rethrow_if_failed();
    }
    const std::vector<float> lens_logits = copy_logits(ctx, llama_vocab_n_tokens(vocab));

    std::vector<float> comparison_logits;
    if (compare_epsilon > 0.0f) {
        {
            auto comparison_head = evaluate_head(rt, initial_state, evaluation_tokens.back(), target_layer, as_fp16(comparison.action));
            rt.run();
            comparison_head.rethrow_if_failed();
        }
        comparison_logits = copy_logits(ctx, llama_vocab_n_tokens(vocab));
    }

    const std::string action_path = output_prefix + ".action.f16";
    const std::string manifest_path = output_prefix + ".txt";
    write_f16(action_path, primary.action);
    std::ofstream manifest(manifest_path);
    if (!manifest) {
        throw std::runtime_error("failed to open output: " + manifest_path);
    }
    manifest << "source_layer=" << source_layer << '\n';
    manifest << "target_layer=" << initial_state.n_layer - 1 << '\n';
    manifest << "dimension=" << primary.action.size() << '\n';
    manifest << "samples=" << completed_samples << '\n';
    manifest << "min_future=" << min_future << '\n';
    manifest << "future_window=" << future_window << '\n';
    manifest << "epsilon_requested=" << epsilon << '\n';
    if (compare_epsilon > 0.0f) {
        manifest << "epsilon_compared=" << compare_epsilon << '\n';
    }
    manifest << "query_norm=" << query_norm << '\n';
    manifest << "action_f16=" << action_path << '\n';
    manifest << "corpus=" << (corpus_path.empty() ? "builtin_eight_sentences" : corpus_path) << '\n';
    if (!corpus_path.empty()) {
        manifest << "max_corpus_tokens=" << max_corpus_tokens << '\n';
    }
    for (size_t i = 0; i < sample_specs.size(); ++i) {
        const auto & spec = sample_specs[i];
        manifest << "sample_" << i << "=corpus:" << spec.corpus_index
                 << ",source:" << spec.source_index << ",target:" << spec.target_index << '\n';
    }

    std::printf("source=%s target=%s samples=%d min_future=%d future_window=%d\n",
        source.c_str(), final.c_str(), completed_samples, min_future, future_window);
    std::printf("query_l2=%g perturbation_l2=%g action_l2=%g\n",
        query_norm, l2_norm(normalized_perturbation(query, epsilon)), l2_norm(primary.action));
    print_response_summary(sample_specs, primary.response_l2, primary.symmetry_error);
    std::printf("carrier round-trip max_logit_abs_error=%.6e\n", max_abs_difference(native_logits, round_trip_logits));
    if (source_layer == (int) initial_state.n_layer - 1 && future_window == 0) {
        std::printf("final-layer identity relative_action_l2_error=%.6e\n",
            relative_l2_difference(primary.action, query));
    }
    if (compare_epsilon > 0.0f) {
        std::printf("compare epsilon requested=%g action_cosine=%.6e relative_l2_difference=%.6e top10_overlap=%d/10\n",
            compare_epsilon, cosine_similarity(primary.action, comparison.action),
            relative_l2_difference(primary.action, comparison.action), top_overlap(lens_logits, comparison_logits, 10));
        std::printf("comparison epsilon response summary:\n");
        print_response_summary(sample_specs, comparison.response_l2, comparison.symmetry_error);
    }
    std::printf("--- J-lens pilot top tokens ---\n");
    print_top_tokens(vocab, lens_logits, 10);
    std::printf("wrote %s\n", manifest_path.c_str());

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
