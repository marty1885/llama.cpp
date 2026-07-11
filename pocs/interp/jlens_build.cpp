#include "interp.hpp"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

struct sample_spec {
    size_t source_index;
    size_t target_index;
};

struct sample_record {
    const char * partition;
    size_t line;
    size_t source;
    size_t target;
    int direction;
};

struct response_measurement {
    std::vector<float> response;
    double response_l2 = 0.0;
    double symmetry_error = 0.0;
    double repeat_plus_error = 0.0;
};

struct metrics {
    double cosine_sum = 0.0;
    double relative_l2_sum = 0.0;
    double symmetry_sum = 0.0;
    double repeat_plus_sum = 0.0;
    double epsilon_cosine_sum = 0.0;
    double epsilon_relative_l2_sum = 0.0;
    int count = 0;
};

struct operator_data {
    size_t input_dimension = 0;
    size_t output_dimension = 0;
    std::vector<std::vector<float>> directions;
    std::vector<std::vector<float>> responses;
};

static void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s -m MODEL --corpus FILE --output PREFIX [-ngl N] [--layer N] [--rank N] [--samples-per-direction N] [--validation-samples N] [--validation-modulo N] [--split-seed N] [--max-corpus-tokens N] [--min-future N] [--future-window N] [--epsilon E] [--compare-epsilon E] [--repeat-plus]\n"
        "\n"
        "Builds a low-rank, strict-future Jacobian lens operator and validates it on held-out corpus lines.\n",
        argv0);
}

static std::string residual_name(int layer) {
    return "rwkv.layer." + std::to_string(layer) + ".resid.out";
}

static const std::vector<float> & require_capture_f32(
        const llama_interp::activation_set & captures,
        const std::string & name) {
    const llama_interp_activation * found = nullptr;
    for (const auto & capture : captures) {
        if (capture.name != name) {
            continue;
        }
        if (found) {
            throw std::runtime_error("duplicate activation capture: " + name);
        }
        found = &capture;
    }
    if (!found || found->data_f32.empty()) {
        throw std::runtime_error("missing FP32 activation capture: " + name);
    }
    return found->data_f32;
}

static double l2_norm(const std::vector<float> & values) {
    double sum = 0.0;
    for (float value : values) {
        sum += (double) value * value;
    }
    return std::sqrt(sum);
}

static double l2_norm(const std::vector<ggml_fp16_t> & values) {
    double sum = 0.0;
    for (ggml_fp16_t value : values) {
        const double x = ggml_fp16_to_fp32(value);
        sum += x * x;
    }
    return std::sqrt(sum);
}

static double cosine_similarity(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("vector dimensions differ");
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

static double relative_l2_difference(const std::vector<float> & estimate, const std::vector<float> & reference) {
    if (estimate.size() != reference.size()) {
        throw std::runtime_error("vector dimensions differ");
    }
    double difference = 0.0;
    double norm = 0.0;
    for (size_t i = 0; i < estimate.size(); ++i) {
        const double delta = (double) estimate[i] - reference[i];
        difference += delta * delta;
        norm += (double) reference[i] * reference[i];
    }
    return std::sqrt(difference) / std::max(std::sqrt(norm), 1e-30);
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
        uint64_t seed) {
    if (tokens.size() <= (size_t) min_future) {
        throw std::runtime_error("corpus sample is too short for the requested future offset");
    }
    const size_t source = (size_t) (splitmix64(seed) % (tokens.size() - (size_t) min_future));
    const size_t min_target = source + (size_t) min_future;
    const size_t max_target = std::min(tokens.size() - 1, source + (size_t) future_window);
    const size_t target = min_target + (size_t) (splitmix64(seed + 1) % (max_target - min_target + 1));
    return { source, target };
}

static std::vector<ggml_fp16_t> rademacher_perturbation(size_t dimension, float epsilon, uint64_t seed) {
    if (dimension == 0) {
        throw std::runtime_error("cannot construct a zero-dimensional perturbation");
    }
    const float scale = epsilon / std::sqrt((float) dimension);
    std::vector<ggml_fp16_t> out(dimension);
    for (size_t i = 0; i < dimension; ++i) {
        out[i] = ggml_fp32_to_fp16((splitmix64(seed + i) & 1) ? scale : -scale);
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

static std::vector<float> unit_vector(const std::vector<ggml_fp16_t> & values, double & norm) {
    norm = l2_norm(values);
    if (!std::isfinite(norm) || norm == 0.0) {
        throw std::runtime_error("perturbation has invalid norm");
    }
    std::vector<float> out(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        out[i] = (float) (ggml_fp16_to_fp32(values[i]) / norm);
    }
    return out;
}

static std::vector<ggml_fp16_t> normalized_query_perturbation(
        const std::vector<float> & query,
        float epsilon,
        std::vector<float> & effective_query,
        double & query_norm) {
    query_norm = l2_norm(query);
    if (!std::isfinite(query_norm) || query_norm == 0.0) {
        throw std::runtime_error("query has invalid norm");
    }
    std::vector<ggml_fp16_t> perturbation;
    perturbation.reserve(query.size());
    for (float value : query) {
        perturbation.push_back(ggml_fp32_to_fp16((float) (epsilon * value / query_norm)));
    }
    double perturbation_norm = 0.0;
    const std::vector<float> direction = unit_vector(perturbation, perturbation_norm);
    effective_query.resize(query.size());
    for (size_t i = 0; i < query.size(); ++i) {
        effective_query[i] = (float) (query_norm * direction[i]);
    }
    return perturbation;
}

static llama_interp::task<> prefill_prefix(
        llama_interp::runtime & rt,
        const llama_interp::rwkv_state & initial_state,
        const std::vector<llama_token> & tokens,
        size_t source_index,
        llama_interp::rwkv_state & out) {
    if (source_index == 0) {
        out = initial_state;
        co_return;
    }
    out = co_await rt.prefill_tokens(initial_state, std::vector<llama_token>(tokens.begin(), tokens.begin() + source_index));
}

static llama_interp::task<> capture_source(
        llama_interp::runtime & rt,
        const llama_interp::rwkv_state & before_source,
        llama_token token,
        const std::string & source,
        llama_interp::activation_set & captures) {
    auto path = rt.prefill_tokens(before_source, { token });
    path.capture_f32("^" + source + "$", captures);
    path.discard_state();
    co_await path;
}

static llama_interp::task<> capture_target(
        llama_interp::runtime & rt,
        const llama_interp::rwkv_state & before_source,
        const std::vector<llama_token> & span,
        const std::string & source,
        const std::string & target,
        const std::vector<ggml_fp16_t> * perturbation,
        llama_interp::activation_set & captures) {
    auto path = rt.prefill_tokens(before_source, span);
    if (perturbation) {
        path.perturb("^" + source + "$", llama_interp::runtime::add_head(-1, *perturbation));
    }
    path.capture_f32("^" + target + "$", captures);
    path.discard_state();
    co_await path;
}

static response_measurement measure_response(
        llama_interp::runtime & rt,
        const llama_interp::rwkv_state & before_source,
        const std::vector<llama_token> & tokens,
        const sample_spec & spec,
        const std::string & source,
        const std::string & target,
        const std::vector<ggml_fp16_t> & perturbation,
        bool repeat_plus) {
    const std::vector<llama_token> span(tokens.begin() + spec.source_index, tokens.begin() + spec.target_index + 1);
    llama_interp::activation_set clean_caps;
    llama_interp::activation_set plus_caps;
    llama_interp::activation_set minus_caps;
    llama_interp::activation_set repeat_plus_caps;

    {
        auto task = capture_target(rt, before_source, span, source, target, nullptr, clean_caps);
        rt.run();
        task.rethrow_if_failed();
    }
    {
        auto task = capture_target(rt, before_source, span, source, target, &perturbation, plus_caps);
        rt.run();
        task.rethrow_if_failed();
    }
    if (repeat_plus) {
        auto task = capture_target(rt, before_source, span, source, target, &perturbation, repeat_plus_caps);
        rt.run();
        task.rethrow_if_failed();
    }
    {
        const std::vector<ggml_fp16_t> negative = negate(perturbation);
        auto task = capture_target(rt, before_source, span, source, target, &negative, minus_caps);
        rt.run();
        task.rethrow_if_failed();
    }

    const auto & clean = require_capture_f32(clean_caps, target);
    const auto & plus = require_capture_f32(plus_caps, target);
    const auto & minus = require_capture_f32(minus_caps, target);
    if (clean.size() != plus.size() || clean.size() != minus.size()) {
        throw std::runtime_error("target capture dimensions differ");
    }
    const double epsilon = l2_norm(perturbation);
    if (!std::isfinite(epsilon) || epsilon == 0.0) {
        throw std::runtime_error("perturbation has invalid norm");
    }

    response_measurement out;
    out.response.resize(clean.size());
    double response_squared_norm = 0.0;
    double even_squared_norm = 0.0;
    double odd_squared_norm = 0.0;
    for (size_t i = 0; i < clean.size(); ++i) {
        const double plus_delta = plus[i] - clean[i];
        const double minus_delta = minus[i] - clean[i];
        out.response[i] = (float) ((plus[i] - minus[i]) / (2.0 * epsilon));
        response_squared_norm += (double) out.response[i] * out.response[i];
        const double even = plus_delta + minus_delta;
        const double odd = plus_delta - minus_delta;
        even_squared_norm += even * even;
        odd_squared_norm += odd * odd;
    }
    out.response_l2 = std::sqrt(response_squared_norm);
    out.symmetry_error = std::sqrt(even_squared_norm) / std::max(std::sqrt(odd_squared_norm), 1e-30);
    if (repeat_plus) {
        const auto & repeated = require_capture_f32(repeat_plus_caps, target);
        if (repeated.size() != plus.size()) {
            throw std::runtime_error("repeated target capture dimensions differ");
        }
        double squared_difference = 0.0;
        for (size_t i = 0; i < plus.size(); ++i) {
            const double delta = plus[i] - repeated[i];
            squared_difference += delta * delta;
        }
        out.repeat_plus_error = std::sqrt(squared_difference);
    }
    return out;
}

template <typename F>
static int stream_partition(
        const llama_vocab * vocab,
        const std::string & corpus_path,
        int max_corpus_tokens,
        int min_future,
        int future_window,
        int validation_modulo,
        uint64_t split_seed,
        bool validation,
        int requested,
        F && consume) {
    std::ifstream input(corpus_path);
    if (!input) {
        throw std::runtime_error("failed to open corpus: " + corpus_path);
    }
    std::string line;
    size_t line_number = 0;
    int accepted = 0;
    while (accepted < requested && std::getline(input, line)) {
        ++line_number;
        const bool is_validation = splitmix64(split_seed ^ line_number) % (uint64_t) validation_modulo == 0;
        if (is_validation != validation) {
            continue;
        }
        auto tokens = common_tokenize(vocab, line, false, true);
        if ((int) tokens.size() > max_corpus_tokens) {
            tokens.resize(max_corpus_tokens);
        }
        if (tokens.size() <= (size_t) min_future) {
            continue;
        }
        const sample_spec spec = choose_sample(tokens, min_future, future_window, splitmix64(split_seed + line_number + accepted));
        consume(tokens, spec, line_number, accepted);
        ++accepted;
    }
    return accepted;
}

static std::vector<float> apply_operator(const operator_data & op, const std::vector<float> & input) {
    if (input.size() != op.input_dimension || op.directions.empty() || op.responses.empty()) {
        throw std::runtime_error("invalid operator application");
    }
    std::vector<float> out(op.output_dimension, 0.0f);
    const double scale = (double) op.input_dimension / op.directions.size();
    for (size_t k = 0; k < op.directions.size(); ++k) {
        double dot = 0.0;
        for (size_t i = 0; i < input.size(); ++i) {
            dot += (double) op.directions[k][i] * input[i];
        }
        for (size_t i = 0; i < out.size(); ++i) {
            out[i] += (float) (scale * dot * op.responses[k][i]);
        }
    }
    return out;
}

static void write_f32_rows(const std::string & path, const std::vector<std::vector<float>> & rows) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open output: " + path);
    }
    size_t width = 0;
    for (const auto & row : rows) {
        if (width == 0) {
            width = row.size();
        } else if (row.size() != width) {
            throw std::runtime_error("inconsistent row widths");
        }
        out.write((const char *) row.data(), (std::streamsize) (row.size() * sizeof(float)));
    }
    if (!out) {
        throw std::runtime_error("failed to write output: " + path);
    }
}

static void write_sample_records(const std::string & path, const std::vector<sample_record> & records) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open output: " + path);
    }
    out << "partition\tline\tsource\ttarget\tdirection\n";
    for (const auto & record : records) {
        out << record.partition << '\t' << record.line << '\t' << record.source << '\t'
            << record.target << '\t' << record.direction << '\n';
    }
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::string model_path;
    std::string corpus_path;
    std::string output_prefix;
    int n_gpu_layers = 0;
    int source_layer = -1;
    int rank = 4;
    int samples_per_direction = 2;
    int validation_samples = 2;
    int validation_modulo = 5;
    uint64_t split_seed = 1;
    int max_corpus_tokens = 64;
    int min_future = 1;
    int future_window = 1;
    float epsilon = 0.2f;
    float compare_epsilon = 0.1f;
    bool repeat_plus = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (std::strcmp(argv[i], "--corpus") == 0 && i + 1 < argc) {
            corpus_path = argv[++i];
        } else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_prefix = argv[++i];
        } else if (std::strcmp(argv[i], "-ngl") == 0 && i + 1 < argc) {
            n_gpu_layers = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--layer") == 0 && i + 1 < argc) {
            source_layer = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--rank") == 0 && i + 1 < argc) {
            rank = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--samples-per-direction") == 0 && i + 1 < argc) {
            samples_per_direction = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--validation-samples") == 0 && i + 1 < argc) {
            validation_samples = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--validation-modulo") == 0 && i + 1 < argc) {
            validation_modulo = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--split-seed") == 0 && i + 1 < argc) {
            split_seed = std::strtoull(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "--max-corpus-tokens") == 0 && i + 1 < argc) {
            max_corpus_tokens = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--min-future") == 0 && i + 1 < argc) {
            min_future = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--future-window") == 0 && i + 1 < argc) {
            future_window = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--epsilon") == 0 && i + 1 < argc) {
            epsilon = std::strtof(argv[++i], nullptr);
        } else if (std::strcmp(argv[i], "--compare-epsilon") == 0 && i + 1 < argc) {
            compare_epsilon = std::strtof(argv[++i], nullptr);
        } else if (std::strcmp(argv[i], "--repeat-plus") == 0) {
            repeat_plus = true;
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    if (model_path.empty() || corpus_path.empty() || output_prefix.empty() || rank <= 0 || samples_per_direction <= 0 ||
        validation_samples <= 0 || validation_modulo < 2 || max_corpus_tokens <= min_future || min_future < 0 ||
        future_window < min_future || epsilon <= 0.0f || compare_epsilon < 0.0f) {
        usage(argv[0]);
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
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = std::max(128, max_corpus_tokens + 8);
    cparams.n_batch = max_corpus_tokens;
    cparams.n_ubatch = max_corpus_tokens;
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
        source_layer = (int) initial_state.n_layer - 2;
    }
    if (source_layer < 0 || source_layer >= (int) initial_state.n_layer - 1) {
        std::fprintf(stderr, "source layer must precede the final layer\n");
        return 1;
    }
    const std::string source = residual_name(source_layer);
    const std::string target = residual_name((int) initial_state.n_layer - 1);

    operator_data op;
    std::vector<std::vector<ggml_fp16_t>> perturbations;
    std::vector<std::vector<float>> comparison_responses;
    if (compare_epsilon > 0.0f) {
        comparison_responses.resize(rank);
    }
    std::vector<sample_record> records;
    metrics build_metrics;

    const int requested_build = rank * samples_per_direction;
    const int completed_build = stream_partition(
        vocab, corpus_path, max_corpus_tokens, min_future, future_window, validation_modulo, split_seed, false, requested_build,
        [&](const std::vector<llama_token> & tokens, const sample_spec & spec, size_t line, int sample_index) {
            llama_interp::rwkv_state before_source;
            {
                auto task = prefill_prefix(rt, initial_state, tokens, spec.source_index, before_source);
                rt.run();
                task.rethrow_if_failed();
            }
            if (op.directions.empty()) {
                llama_interp::activation_set source_caps;
                {
                    auto task = capture_source(rt, before_source, tokens[spec.source_index], source, source_caps);
                    rt.run();
                    task.rethrow_if_failed();
                }
                op.input_dimension = require_capture_f32(source_caps, source).size();
                op.directions.resize(rank);
                perturbations.resize(rank);
                for (int k = 0; k < rank; ++k) {
                    perturbations[k] = rademacher_perturbation(op.input_dimension, epsilon, splitmix64((uint64_t) k));
                    double norm = 0.0;
                    op.directions[k] = unit_vector(perturbations[k], norm);
                }
            }
            const int direction = sample_index % rank;
            const response_measurement primary = measure_response(
                rt, before_source, tokens, spec, source, target, perturbations[direction], repeat_plus);
            if (op.responses.empty()) {
                op.output_dimension = primary.response.size();
                op.responses.assign(rank, std::vector<float>(op.output_dimension, 0.0f));
                if (compare_epsilon > 0.0f) {
                    comparison_responses.assign(rank, std::vector<float>(op.output_dimension, 0.0f));
                }
            }
            if (primary.response.size() != op.output_dimension) {
                throw std::runtime_error("inconsistent target dimension");
            }
            for (size_t i = 0; i < op.output_dimension; ++i) {
                op.responses[direction][i] += primary.response[i];
            }
            build_metrics.symmetry_sum += primary.symmetry_error;
            build_metrics.repeat_plus_sum += primary.repeat_plus_error;
            if (compare_epsilon > 0.0f) {
                const std::vector<ggml_fp16_t> comparison = rademacher_perturbation(
                    op.input_dimension, compare_epsilon, splitmix64((uint64_t) direction));
                const response_measurement secondary = measure_response(
                    rt, before_source, tokens, spec, source, target, comparison, repeat_plus);
                for (size_t i = 0; i < op.output_dimension; ++i) {
                    comparison_responses[direction][i] += secondary.response[i];
                }
            }
            records.push_back({ "build", line, spec.source_index, spec.target_index, direction });
        });
    if (completed_build != requested_build) {
        throw std::runtime_error("corpus did not supply enough build samples after partitioning");
    }
    for (auto & row : op.responses) {
        for (float & value : row) {
            value /= samples_per_direction;
        }
    }
    if (compare_epsilon > 0.0f) {
        for (int k = 0; k < rank; ++k) {
            for (size_t i = 0; i < op.output_dimension; ++i) {
                comparison_responses[k][i] /= samples_per_direction;
            }
            build_metrics.epsilon_cosine_sum += cosine_similarity(op.responses[k], comparison_responses[k]);
            build_metrics.epsilon_relative_l2_sum += relative_l2_difference(op.responses[k], comparison_responses[k]);
        }
    }

    const int completed_validation = stream_partition(
        vocab, corpus_path, max_corpus_tokens, min_future, future_window, validation_modulo, split_seed, true, validation_samples,
        [&](const std::vector<llama_token> & tokens, const sample_spec & spec, size_t line, int) {
            llama_interp::rwkv_state before_source;
            {
                auto task = prefill_prefix(rt, initial_state, tokens, spec.source_index, before_source);
                rt.run();
                task.rethrow_if_failed();
            }
            llama_interp::activation_set source_caps;
            {
                auto task = capture_source(rt, before_source, tokens[spec.source_index], source, source_caps);
                rt.run();
                task.rethrow_if_failed();
            }
            const auto & query = require_capture_f32(source_caps, source);
            if (query.size() != op.input_dimension) {
                throw std::runtime_error("source capture dimension differs from operator input");
            }
            std::vector<float> effective_query;
            double query_norm = 0.0;
            const std::vector<ggml_fp16_t> primary_perturbation = normalized_query_perturbation(
                query, epsilon, effective_query, query_norm);
            const response_measurement primary = measure_response(
                rt, before_source, tokens, spec, source, target, primary_perturbation, repeat_plus);
            std::vector<float> actual = primary.response;
            for (float & value : actual) {
                value = (float) (value * query_norm);
            }
            const std::vector<float> predicted = apply_operator(op, effective_query);
            build_metrics.cosine_sum += cosine_similarity(predicted, actual);
            build_metrics.relative_l2_sum += relative_l2_difference(predicted, actual);
            build_metrics.symmetry_sum += primary.symmetry_error;
            build_metrics.repeat_plus_sum += primary.repeat_plus_error;
            ++build_metrics.count;
            if (compare_epsilon > 0.0f) {
                std::vector<float> ignored_effective_query;
                double ignored_query_norm = 0.0;
                const std::vector<ggml_fp16_t> secondary_perturbation = normalized_query_perturbation(
                    query, compare_epsilon, ignored_effective_query, ignored_query_norm);
                const response_measurement secondary = measure_response(
                    rt, before_source, tokens, spec, source, target, secondary_perturbation, repeat_plus);
                std::vector<float> comparison = secondary.response;
                for (float & value : comparison) {
                    value = (float) (value * ignored_query_norm);
                }
                build_metrics.epsilon_cosine_sum += cosine_similarity(actual, comparison);
                build_metrics.epsilon_relative_l2_sum += relative_l2_difference(actual, comparison);
            }
            records.push_back({ "validation", line, spec.source_index, spec.target_index, -1 });
        });
    if (completed_validation != validation_samples) {
        throw std::runtime_error("corpus did not supply enough validation samples after partitioning");
    }

    const std::string directions_path = output_prefix + ".directions.f32";
    const std::string responses_path = output_prefix + ".responses.f32";
    const std::string samples_path = output_prefix + ".samples.tsv";
    const std::string manifest_path = output_prefix + ".txt";
    write_f32_rows(directions_path, op.directions);
    write_f32_rows(responses_path, op.responses);
    write_sample_records(samples_path, records);
    std::ofstream manifest(manifest_path);
    if (!manifest) {
        throw std::runtime_error("failed to open output: " + manifest_path);
    }
    manifest << "format=rwkv_jlens_low_rank_v1\n";
    manifest << "source_layer=" << source_layer << '\n';
    manifest << "target_layer=" << initial_state.n_layer - 1 << '\n';
    manifest << "input_dimension=" << op.input_dimension << '\n';
    manifest << "output_dimension=" << op.output_dimension << '\n';
    manifest << "rank=" << rank << '\n';
    manifest << "samples_per_direction=" << samples_per_direction << '\n';
    manifest << "build_samples=" << completed_build << '\n';
    manifest << "validation_samples=" << completed_validation << '\n';
    manifest << "corpus=" << corpus_path << '\n';
    manifest << "max_corpus_tokens=" << max_corpus_tokens << '\n';
    manifest << "validation_modulo=" << validation_modulo << '\n';
    manifest << "split_seed=" << split_seed << '\n';
    manifest << "min_future=" << min_future << '\n';
    manifest << "future_window=" << future_window << '\n';
    manifest << "epsilon_requested=" << epsilon << '\n';
    manifest << "compare_epsilon_requested=" << compare_epsilon << '\n';
    manifest << "n_gpu_layers=" << n_gpu_layers << '\n';
    manifest << "directions_f32=" << directions_path << '\n';
    manifest << "responses_f32=" << responses_path << '\n';
    manifest << "samples_tsv=" << samples_path << '\n';
    manifest << "estimator=Jhat(x)=(input_dimension/rank)*sum_k(response[k]*dot(direction[k],x))\n";
    manifest << "validation_mean_cosine=" << build_metrics.cosine_sum / build_metrics.count << '\n';
    manifest << "validation_mean_relative_l2=" << build_metrics.relative_l2_sum / build_metrics.count << '\n';
    manifest << "mean_symmetry=" << build_metrics.symmetry_sum / (completed_build + completed_validation) << '\n';
    manifest << "mean_repeat_plus_l2=" << build_metrics.repeat_plus_sum / (completed_build + completed_validation) << '\n';
    if (compare_epsilon > 0.0f) {
        manifest << "mean_epsilon_cosine=" << build_metrics.epsilon_cosine_sum / (rank + completed_validation) << '\n';
        manifest << "mean_epsilon_relative_l2=" << build_metrics.epsilon_relative_l2_sum / (rank + completed_validation) << '\n';
    }

    std::printf("built source=%s target=%s rank=%d build_samples=%d validation_samples=%d\n",
        source.c_str(), target.c_str(), rank, completed_build, completed_validation);
    std::printf("validation cosine=%g relative_l2=%g symmetry=%g repeat_plus_l2=%g\n",
        build_metrics.cosine_sum / build_metrics.count,
        build_metrics.relative_l2_sum / build_metrics.count,
        build_metrics.symmetry_sum / (completed_build + completed_validation),
        build_metrics.repeat_plus_sum / (completed_build + completed_validation));
    if (compare_epsilon > 0.0f) {
        std::printf("epsilon comparison cosine=%g relative_l2=%g\n",
            build_metrics.epsilon_cosine_sum / (rank + completed_validation),
            build_metrics.epsilon_relative_l2_sum / (rank + completed_validation));
    }
    std::printf("wrote %s\n", manifest_path.c_str());

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
