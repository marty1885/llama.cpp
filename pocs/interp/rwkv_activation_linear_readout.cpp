#include "rwkv_activation_store.h"

#include "common.h"
#include "llama-context.h"
#include "llama-model.h"
#include "models/models.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct dataset_values {
    std::vector<rwkv_activation_store::sample_metadata> metadata;
    std::vector<float> values;
};

uint64_t splitmix64(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

std::vector<std::string> all_sources() {
    const std::vector<const char *> taps = {
        "resid.in", "att.norm",
        "time.r", "time.w", "time.k", "time.v", "time.a", "time.a_pre", "time.g", "time.k0", "time.kk", "time.wkv", "time.rkv", "time.out",
        "resid.time", "ffn.norm", "channel.out", "resid.out",
    };
    std::vector<std::string> result;
    for (int layer = 0; layer < 61; ++layer) {
        for (const char * tap : taps) result.push_back("rwkv.layer." + std::to_string(layer) + "." + tap);
    }
    return result;
}

size_t parse_source_index(const std::vector<std::string> & sources, const std::string & spec) {
    const size_t separator = spec.find(':');
    if (separator == std::string::npos) throw std::runtime_error("source must be LAYER:TAP");
    const std::string name = "rwkv.layer." + std::to_string(std::atoi(spec.substr(0, separator).c_str())) + "." + spec.substr(separator + 1);
    const auto found = std::find(sources.begin(), sources.end(), name);
    if (found == sources.end()) throw std::runtime_error("unknown source: " + spec);
    return found - sources.begin();
}

dataset_values load_source(
        const rwkv_activation_store::activation_dataset_reader & dataset,
        size_t source_index,
        size_t batch_rows) {
    dataset_values result;
    result.metadata.reserve(dataset.entries());
    result.values.reserve(dataset.entries() * dataset.dimension());
    dataset.for_each_source_batch(source_index, batch_rows, [&](rwkv_activation_store::source_batch && batch) {
        result.metadata.insert(result.metadata.end(), batch.metadata.begin(), batch.metadata.end());
        result.values.insert(result.values.end(), batch.values.begin(), batch.values.end());
    });
    return result;
}

double cosine(const float * left, const float * right, size_t dimension) {
    double dot = 0.0;
    double left_norm = 0.0;
    double right_norm = 0.0;
    for (size_t i = 0; i < dimension; ++i) {
        dot += (double) left[i] * right[i];
        left_norm += (double) left[i] * left[i];
        right_norm += (double) right[i] * right[i];
    }
    return dot / std::sqrt(std::max(left_norm * right_norm, 1e-30));
}

// Solves (L L^T) X = B after factoring the positive-definite Gram matrix in place.
void solve_ridge(std::vector<double> gram, std::vector<double> & cross, size_t rows, size_t dimension, double lambda) {
    for (size_t i = 0; i < rows; ++i) {
        gram[i * rows + i] += lambda;
        for (size_t j = 0; j < i; ++j) {
            double value = gram[i * rows + j];
            for (size_t k = 0; k < j; ++k) value -= gram[i * rows + k] * gram[j * rows + k];
            gram[i * rows + j] = value / gram[j * rows + j];
        }
        double diagonal = gram[i * rows + i];
        for (size_t k = 0; k < i; ++k) diagonal -= gram[i * rows + k] * gram[i * rows + k];
        if (diagonal <= 0.0) throw std::runtime_error("ridge system is not positive definite");
        gram[i * rows + i] = std::sqrt(diagonal);
    }
    for (size_t output = 0; output < dimension; ++output) {
        for (size_t row = 0; row < rows; ++row) {
            double value = cross[row * dimension + output];
            for (size_t column = 0; column < row; ++column) value -= gram[row * rows + column] * cross[column * dimension + output];
            cross[row * dimension + output] = value / gram[row * rows + row];
        }
        for (size_t row = rows; row-- > 0;) {
            double value = cross[row * dimension + output];
            for (size_t column = row + 1; column < rows; ++column) value -= gram[column * rows + row] * cross[column * dimension + output];
            cross[row * dimension + output] = value / gram[row * rows + row];
        }
    }
}

struct direct_graph : llm_build_rwkv7_base {
    direct_graph(const llama_model & model, const llm_graph_params & params) : llm_build_rwkv7_base(model, params) {}
};

void add_output(direct_graph & graph, ggml_tensor * tensor) {
    ggml_set_output(tensor);
    ggml_build_forward_expand(graph.gf, tensor);
}

llm_graph_params graph_params(llama_context * ctx, llm_graph_result * result) {
    const auto & model = ctx->get_model();
    llama_ubatch ubatch = {};
    ubatch.b_equal_seqs = true;
    ubatch.n_tokens = 1;
    ubatch.n_seq_tokens = 1;
    ubatch.n_seqs = 1;
    ubatch.n_seqs_unq = 1;
    ubatch.n_pos = 1;

    llm_graph_params params = {};
    params.arch = model.arch;
    params.hparams = model.hparams;
    params.cparams = ctx->get_cparams();
    params.ubatch = ubatch;
    params.gtype = LLM_GRAPH_TYPE_DEFAULT;
    params.sched = ctx->get_sched();
    static const llama_adapter_loras no_loras;
    params.loras = &no_loras;
    params.n_outputs = 1;
    params.res = result;
    return params;
}

std::vector<float> project_logits(llama_context * ctx, const std::vector<float> & values, size_t columns) {
    const auto & model = ctx->get_model();
    const size_t dimension = model.hparams.n_embd;
    if (values.size() != dimension * columns) throw std::runtime_error("invalid residual projection input");

    ctx->synchronize();
    ggml_backend_sched_reset(ctx->get_sched());
    llm_graph_result result(ctx->graph_max_nodes(1));
    direct_graph graph(model, graph_params(ctx, &result));
    ggml_tensor * vectors = ggml_new_tensor_2d(graph.ctx0, GGML_TYPE_F32, dimension, columns);
    ggml_set_input(vectors);
    ggml_tensor * normed = graph.build_norm(vectors, model.output_norm, model.output_norm_b, LLM_NORM, -1);
    ggml_tensor * logits = ggml_mul_mat(graph.ctx0, model.output, normed);
    if (model.output_s) logits = ggml_mul(graph.ctx0, logits, model.output_s);
    add_output(graph, logits);
    if (!ggml_backend_sched_alloc_graph(ctx->get_sched(), graph.gf)) throw std::runtime_error("failed to allocate output projection graph");
    ggml_backend_tensor_set(vectors, values.data(), 0, values.size() * sizeof(float));
    if (ctx->graph_compute(graph.gf, false) != GGML_STATUS_SUCCESS) throw std::runtime_error("output projection graph failed");
    ctx->synchronize();

    const size_t vocabulary = llama_vocab_n_tokens(&model.vocab);
    std::vector<float> result_logits(vocabulary * columns);
    ggml_backend_tensor_get(logits, result_logits.data(), 0, result_logits.size() * sizeof(float));
    return result_logits;
}

int argmax(const float * values, size_t count) {
    return (int) std::distance(values, std::max_element(values, values + count));
}

int rank_of(const float * values, size_t count, int token) {
    int rank = 1;
    for (size_t i = 0; i < count; ++i) rank += values[i] > values[token];
    return rank;
}

std::vector<int> top_tokens(const float * values, size_t count, size_t top) {
    std::vector<int> tokens(count);
    for (size_t i = 0; i < count; ++i) tokens[i] = (int) i;
    top = std::min(top, count);
    std::partial_sort(tokens.begin(), tokens.begin() + top, tokens.end(), [values](int left, int right) {
        return values[left] > values[right];
    });
    tokens.resize(top);
    return tokens;
}

std::string tsv_piece(std::string value) {
    for (char & character : value) {
        if (character == '\t' || character == '\n' || character == '\r') character = ' ';
    }
    return value;
}

void write_linear_map(
        const std::string & path,
        const std::vector<float> & source_mean,
        const std::vector<float> & target_mean,
        const dataset_values & source,
        const std::vector<size_t> & train_rows,
        const std::vector<double> & coefficients,
        size_t dimension) {
    const std::filesystem::path map_path(path);
    if (!map_path.parent_path().empty()) std::filesystem::create_directories(map_path.parent_path());
    std::ofstream output(path, std::ios::binary);
    if (!output) throw std::runtime_error("failed to write linear readout map");
    const char magic[8] = { 'R', 'W', 'K', 'V', 'L', 'R', 'M', '1' };
    const uint64_t stored_dimension = dimension;
    const uint64_t rows = train_rows.size();
    output.write(magic, sizeof(magic));
    output.write(reinterpret_cast<const char *>(&stored_dimension), sizeof(stored_dimension));
    output.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
    output.write(reinterpret_cast<const char *>(source_mean.data()), source_mean.size() * sizeof(float));
    output.write(reinterpret_cast<const char *>(target_mean.data()), target_mean.size() * sizeof(float));
    std::vector<float> centered(train_rows.size() * dimension);
    std::vector<float> output_coefficients(train_rows.size() * dimension);
    for (size_t train_position = 0; train_position < train_rows.size(); ++train_position) {
        const float * values = source.values.data() + train_rows[train_position] * dimension;
        for (size_t i = 0; i < dimension; ++i) {
            centered[train_position * dimension + i] = values[i] - source_mean[i];
            output_coefficients[train_position * dimension + i] = (float) coefficients[train_position * dimension + i];
        }
    }
    output.write(reinterpret_cast<const char *>(centered.data()), centered.size() * sizeof(float));
    output.write(reinterpret_cast<const char *>(output_coefficients.data()), output_coefficients.size() * sizeof(float));
    if (!output) throw std::runtime_error("failed to finish linear readout map");
}

void write_json_string(std::ostream & output, const std::string & value) {
    static const char hex[] = "0123456789abcdef";
    output << '"';
    for (unsigned char character : value) {
        switch (character) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (character < 0x20) output << "\\u00" << hex[character >> 4] << hex[character & 0x0f];
                else output << character;
        }
    }
    output << '"';
}

void write_viewer_readout(
        std::ostream & output,
        const std::string & source,
        const float * logits,
        size_t vocabulary,
        int reference_top,
        double residual_cosine,
        const llama_vocab * vocab,
        size_t top) {
    output << "{\"source\":";
    write_json_string(output, source);
    output << ",\"top_tokens\":[";
    const auto tokens = top_tokens(logits, vocabulary, top);
    for (size_t index = 0; index < tokens.size(); ++index) {
        if (index) output << ',';
        const int token = tokens[index];
        output << "{\"id\":" << token << ",\"piece\":";
        write_json_string(output, common_token_to_piece(vocab, token, true));
        output << ",\"logit\":" << logits[token] << '}';
    }
    output << "],\"reported_tokens\":[],\"baseline_cosine\":" << residual_cosine
        << ",\"target_rank\":" << rank_of(logits, vocabulary, reference_top)
        << ",\"target_logit\":" << logits[reference_top]
        << ",\"random_target_rank\":-1,\"random_target_logit\":0}";
}

void write_viewer_trace(
        const std::string & path,
        const std::string & model_path,
        int n_gpu_layers,
        const std::string & source_name,
        const std::string & target_name,
        const dataset_values & source,
        const std::vector<size_t> & test_rows,
        const std::vector<float> & predictions,
        const dataset_values & target,
        const std::vector<float> & logits,
        size_t dimension,
        size_t vocabulary,
        const llama_vocab * vocab,
        size_t top) {
    std::ofstream output(path);
    if (!output) throw std::runtime_error("failed to write viewer trace");
    output << std::setprecision(8);
    output << "{\n  \"schema_version\": 1,\n  \"model_path\": ";
    write_json_string(output, model_path);
    output << ",\n  \"n_gpu_layers\": " << n_gpu_layers << ",\n  \"top_k\": " << top
        << ",\n  \"trace_kind\": \"rwkv_linear_readout\",\n  \"sources\": [";
    write_json_string(output, target_name);
    output << ", ";
    write_json_string(output, source_name);
    output << "],\n  \"independent_samples\": true,\n  \"runs\": [{\n    \"prompt\": \"Each item is an independent held-out generated activation. The first card is the captured final residual; the second is a full-space ridge decode from time.k. Full prompt trajectories stay entirely in the train or test split.\",\n    \"steps\": [\n";
    for (size_t test_position = 0; test_position < test_rows.size(); ++test_position) {
        if (test_position) output << ",\n";
        const auto & metadata = source.metadata[test_rows[test_position]];
        const float * reference = logits.data() + (4 * test_position) * vocabulary;
        const float * decoded = logits.data() + (4 * test_position + 1) * vocabulary;
        const int reference_top = argmax(reference, vocabulary);
        output << "      {\"index\":" << test_position << ",\"input\":{\"id\":" << metadata.token_id << ",\"piece\":";
        write_json_string(output, common_token_to_piece(vocab, metadata.token_id, true));
        output << "},\"next\":{\"id\":" << reference_top << ",\"piece\":";
        write_json_string(output, common_token_to_piece(vocab, reference_top, true));
        output << "},\"final_round_trip_error\":0,\"readouts\":[";
        write_viewer_readout(output, target_name, reference, vocabulary, reference_top, 1.0, vocab, top);
        output << ',';
        write_viewer_readout(output, source_name, decoded, vocabulary, reference_top,
            cosine(predictions.data() + test_position * dimension, target.values.data() + test_rows[test_position] * dimension, dimension), vocab, top);
        output << "]}";
    }
    output << "\n    ],\n    \"stopped\": \"held_out_samples\"\n  }]\n}\n";
}

void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s -m MODEL --input ACTIVATIONS.root --source LAYER:TAP --output SUMMARY.tsv [--target LAYER:TAP] [--compact|--compact-layer-pairs] [--lambda REL] [--holdout-bucket N] [--batch-rows N] [--top N] [--write-map FILE --fit-only] [-ngl N]\n"
        "\n"
        "Fits the full-space linear ridge readout Y = A X + b in its dual form, applies it\n"
        "to held-out corpus lines, and projects predicted residuals through the native output head.\n"
        "--compact reads a two-column dataset containing only --source then --target.\n"
        "--compact-layer-pairs reads time.k/resid.out pairs in layer order.\n"
        "--fit-only writes the fitted map and skips held-out projection.\n",
        argv0);
}

} // namespace

int main(int argc, char ** argv) {
    std::string model_path;
    std::string input_path;
    std::string source_spec;
    std::string target_spec = "60:resid.out";
    std::string output_path;
    int n_gpu_layers = 0;
    double lambda_relative = 0.01;
    uint64_t holdout_bucket = 0;
    size_t batch_rows = 64;
    size_t top = 10;
    std::string write_map_path;
    bool compact = false;
    bool compact_layer_pairs = false;
    bool fit_only = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (std::strcmp(argv[i], "--input") == 0 && i + 1 < argc) input_path = argv[++i];
        else if (std::strcmp(argv[i], "--source") == 0 && i + 1 < argc) source_spec = argv[++i];
        else if (std::strcmp(argv[i], "--target") == 0 && i + 1 < argc) target_spec = argv[++i];
        else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) output_path = argv[++i];
        else if (std::strcmp(argv[i], "--lambda") == 0 && i + 1 < argc) lambda_relative = std::strtod(argv[++i], nullptr);
        else if (std::strcmp(argv[i], "--holdout-bucket") == 0 && i + 1 < argc) holdout_bucket = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "--batch-rows") == 0 && i + 1 < argc) batch_rows = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "--top") == 0 && i + 1 < argc) top = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "--write-map") == 0 && i + 1 < argc) write_map_path = argv[++i];
        else if (std::strcmp(argv[i], "--compact") == 0) compact = true;
        else if (std::strcmp(argv[i], "--compact-layer-pairs") == 0) compact_layer_pairs = true;
        else if (std::strcmp(argv[i], "--fit-only") == 0) fit_only = true;
        else if (std::strcmp(argv[i], "-ngl") == 0 && i + 1 < argc) n_gpu_layers = std::atoi(argv[++i]);
        else { usage(argv[0]); return 1; }
    }
    if (model_path.empty() || input_path.empty() || source_spec.empty() || output_path.empty() || lambda_relative <= 0.0 || holdout_bucket >= 5 || batch_rows == 0 || top == 0 || (compact && compact_layer_pairs) || (fit_only && write_map_path.empty())) {
        usage(argv[0]);
        return 1;
    }

    const std::vector<std::string> full_sources = all_sources();
    const std::string source_name = full_sources[parse_source_index(full_sources, source_spec)];
    const std::string target_name = full_sources[parse_source_index(full_sources, target_spec)];
    std::vector<std::string> sources = full_sources;
    size_t source_index = parse_source_index(sources, source_spec);
    size_t target_index = parse_source_index(sources, target_spec);
    if (compact) {
        sources = { source_name, target_name };
        source_index = 0;
        target_index = 1;
    } else if (compact_layer_pairs) {
        sources.clear();
        for (int layer = 0; layer < 61; ++layer) {
            sources.push_back("rwkv.layer." + std::to_string(layer) + ".time.k");
            sources.push_back("rwkv.layer." + std::to_string(layer) + ".resid.out");
        }
        const auto source = std::find(sources.begin(), sources.end(), source_name);
        const auto target = std::find(sources.begin(), sources.end(), target_name);
        if (source == sources.end() || target == sources.end()) {
            throw std::runtime_error("--compact-layer-pairs only supports time.k and resid.out sources");
        }
        source_index = source - sources.begin();
        target_index = target - sources.begin();
    }
    const auto dataset = rwkv_activation_store::activation_dataset_reader::open(input_path, sources);
    const size_t dimension = dataset.dimension();
    dataset_values source = load_source(dataset, source_index, batch_rows);
    dataset_values target = load_source(dataset, target_index, batch_rows);
    if (source.metadata.size() != target.metadata.size() || source.values.size() != target.values.size()) {
        throw std::runtime_error("source and target datasets do not match");
    }

    std::vector<size_t> train_rows;
    std::vector<size_t> test_rows;
    for (size_t row = 0; row < source.metadata.size(); ++row) {
        if (source.metadata[row].sample_id != target.metadata[row].sample_id) throw std::runtime_error("source and target row order differs");
        (splitmix64(source.metadata[row].corpus_line) % 5 == holdout_bucket ? test_rows : train_rows).push_back(row);
    }
    if (train_rows.size() < 2 || test_rows.empty()) throw std::runtime_error("insufficient grouped train/test rows");

    std::vector<float> source_mean(dimension);
    std::vector<float> target_mean(dimension);
    for (size_t row : train_rows) {
        const float * x = source.values.data() + row * dimension;
        const float * y = target.values.data() + row * dimension;
        for (size_t i = 0; i < dimension; ++i) {
            source_mean[i] += x[i];
            target_mean[i] += y[i];
        }
    }
    for (size_t i = 0; i < dimension; ++i) {
        source_mean[i] /= train_rows.size();
        target_mean[i] /= train_rows.size();
    }

    const size_t train_count = train_rows.size();
    std::vector<double> gram(train_count * train_count);
    for (size_t left = 0; left < train_count; ++left) {
        const float * x_left = source.values.data() + train_rows[left] * dimension;
        for (size_t right = 0; right <= left; ++right) {
            const float * x_right = source.values.data() + train_rows[right] * dimension;
            double value = 0.0;
            for (size_t i = 0; i < dimension; ++i) value += ((double) x_left[i] - source_mean[i]) * ((double) x_right[i] - source_mean[i]);
            gram[left * train_count + right] = value;
            gram[right * train_count + left] = value;
        }
    }
    double trace = 0.0;
    for (size_t i = 0; i < train_count; ++i) trace += gram[i * train_count + i];
    const double lambda = lambda_relative * trace / train_count;

    std::vector<double> coefficients(train_count * dimension);
    std::vector<double> null_coefficients(train_count * dimension);
    for (size_t train_position = 0; train_position < train_count; ++train_position) {
        const float * y = target.values.data() + train_rows[train_position] * dimension;
        const float * null_y = target.values.data() + train_rows[(train_position + 1) % train_count] * dimension;
        for (size_t i = 0; i < dimension; ++i) {
            coefficients[train_position * dimension + i] = y[i] - target_mean[i];
            null_coefficients[train_position * dimension + i] = null_y[i] - target_mean[i];
        }
    }
    solve_ridge(gram, coefficients, train_count, dimension, lambda);
    if (!write_map_path.empty()) write_linear_map(write_map_path, source_mean, target_mean, source, train_rows, coefficients, dimension);
    if (fit_only) {
        std::printf("wrote=%s train_rows=%zu\n", write_map_path.c_str(), train_count);
        return 0;
    }
    solve_ridge(gram, null_coefficients, train_count, dimension, lambda);

    const size_t test_count = test_rows.size();
    std::vector<float> predictions(test_count * dimension);
    std::vector<float> null_predictions(test_count * dimension);
    std::vector<double> weights(train_count);
    double mean_cosine = 0.0;
    double prediction_cosine = 0.0;
    double null_cosine = 0.0;
    double prediction_squared_error = 0.0;
    double mean_squared_error = 0.0;
    for (size_t test_position = 0; test_position < test_count; ++test_position) {
        const size_t row = test_rows[test_position];
        const float * x = source.values.data() + row * dimension;
        const float * y = target.values.data() + row * dimension;
        for (size_t train_position = 0; train_position < train_count; ++train_position) {
            const float * train_x = source.values.data() + train_rows[train_position] * dimension;
            double value = 0.0;
            for (size_t i = 0; i < dimension; ++i) value += ((double) x[i] - source_mean[i]) * ((double) train_x[i] - source_mean[i]);
            weights[train_position] = value;
        }
        float * prediction = predictions.data() + test_position * dimension;
        float * null_prediction = null_predictions.data() + test_position * dimension;
        for (size_t i = 0; i < dimension; ++i) {
            double value = target_mean[i];
            double null_value = target_mean[i];
            for (size_t train_position = 0; train_position < train_count; ++train_position) {
                value += weights[train_position] * coefficients[train_position * dimension + i];
                null_value += weights[train_position] * null_coefficients[train_position * dimension + i];
            }
            prediction[i] = (float) value;
            null_prediction[i] = (float) null_value;
            prediction_squared_error += (value - y[i]) * (value - y[i]);
            mean_squared_error += ((double) target_mean[i] - y[i]) * (target_mean[i] - y[i]);
        }
        mean_cosine += cosine(target_mean.data(), y, dimension);
        prediction_cosine += cosine(prediction, y, dimension);
        null_cosine += cosine(null_prediction, y, dimension);
    }

    ggml_backend_load_all();
    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) throw std::runtime_error("failed to load model");
    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = 128;
    context_params.n_batch = 128;
    context_params.n_ubatch = 128;
    context_params.n_seq_max = 1;
    llama_context * ctx = llama_init_from_model(model, context_params);
    if (!ctx) throw std::runtime_error("failed to create context");
    if (ctx->get_model().hparams.n_embd != dimension) throw std::runtime_error("model embedding dimension does not match activation dataset");

    std::vector<float> residual_columns(4 * test_count * dimension);
    for (size_t test_position = 0; test_position < test_count; ++test_position) {
        const float * y = target.values.data() + test_rows[test_position] * dimension;
        std::copy(y, y + dimension, residual_columns.begin() + (4 * test_position + 0) * dimension);
        std::copy(predictions.begin() + test_position * dimension, predictions.begin() + (test_position + 1) * dimension,
            residual_columns.begin() + (4 * test_position + 1) * dimension);
        std::copy(target_mean.begin(), target_mean.end(), residual_columns.begin() + (4 * test_position + 2) * dimension);
        std::copy(null_predictions.begin() + test_position * dimension, null_predictions.begin() + (test_position + 1) * dimension,
            residual_columns.begin() + (4 * test_position + 3) * dimension);
    }
    const std::vector<float> logits = project_logits(ctx, residual_columns, 4 * test_count);
    const size_t vocabulary = llama_vocab_n_tokens(llama_model_get_vocab(model));

    size_t decoded_top1_agreement = 0;
    size_t mean_top1_agreement = 0;
    size_t null_top1_agreement = 0;
    double decoded_logit_cosine = 0.0;
    double mean_logit_cosine = 0.0;
    double null_logit_cosine = 0.0;
    double decoded_reference_top_rank = 0.0;
    double mean_reference_top_rank = 0.0;
    double null_reference_top_rank = 0.0;

    const std::filesystem::path output(output_path);
    if (!output.parent_path().empty()) std::filesystem::create_directories(output.parent_path());
    std::ofstream samples(output_path + ".samples.tsv");
    if (!samples) throw std::runtime_error("failed to write sample readouts");
    samples << "sample_id\tcorpus_line\ttoken_position\tmodel_top\tdecoded_top\tmean_top\tnull_top\tdecoded_model_top_rank\tmean_model_top_rank\tnull_model_top_rank\tdecoded_logit_cosine\n";
    std::ofstream token_readouts(output_path + ".logits.tsv");
    if (!token_readouts) throw std::runtime_error("failed to write output-token readouts");
    token_readouts << "sample_id\tcorpus_line\ttoken_position\tselected_by\treference_rank\tdecoded_rank\tmean_rank\tnull_rank\ttoken_id\tpiece\treference_logit\tdecoded_logit\tmean_logit\tnull_logit\n";
    for (size_t test_position = 0; test_position < test_count; ++test_position) {
        const float * reference = logits.data() + (4 * test_position + 0) * vocabulary;
        const float * decoded = logits.data() + (4 * test_position + 1) * vocabulary;
        const float * mean = logits.data() + (4 * test_position + 2) * vocabulary;
        const float * null = logits.data() + (4 * test_position + 3) * vocabulary;
        const int reference_top = argmax(reference, vocabulary);
        const int decoded_top = argmax(decoded, vocabulary);
        const int mean_top = argmax(mean, vocabulary);
        const int null_top = argmax(null, vocabulary);
        const int decoded_rank = rank_of(decoded, vocabulary, reference_top);
        const int mean_rank = rank_of(mean, vocabulary, reference_top);
        const int null_rank = rank_of(null, vocabulary, reference_top);
        decoded_top1_agreement += decoded_top == reference_top;
        mean_top1_agreement += mean_top == reference_top;
        null_top1_agreement += null_top == reference_top;
        decoded_logit_cosine += cosine(decoded, reference, vocabulary);
        mean_logit_cosine += cosine(mean, reference, vocabulary);
        null_logit_cosine += cosine(null, reference, vocabulary);
        decoded_reference_top_rank += decoded_rank;
        mean_reference_top_rank += mean_rank;
        null_reference_top_rank += null_rank;
        const auto & metadata = source.metadata[test_rows[test_position]];
        samples << metadata.sample_id << '\t' << metadata.corpus_line << '\t' << metadata.token_position << '\t'
            << reference_top << '\t' << decoded_top << '\t' << mean_top << '\t' << null_top << '\t'
            << decoded_rank << '\t' << mean_rank << '\t' << null_rank << '\t' << cosine(decoded, reference, vocabulary) << '\n';

        std::vector<int> reported = top_tokens(reference, vocabulary, top);
        const auto decoded_tokens = top_tokens(decoded, vocabulary, top);
        for (int token : decoded_tokens) {
            if (std::find(reported.begin(), reported.end(), token) == reported.end()) reported.push_back(token);
        }
        for (int token : reported) {
            const bool reference_selected = std::find(reported.begin(), reported.begin() + std::min(top, reported.size()), token) != reported.begin() + std::min(top, reported.size());
            const bool decoded_selected = std::find(decoded_tokens.begin(), decoded_tokens.end(), token) != decoded_tokens.end();
            const char * selected_by = reference_selected && decoded_selected ? "both" : reference_selected ? "reference" : "decoded";
            token_readouts << metadata.sample_id << '\t' << metadata.corpus_line << '\t' << metadata.token_position << '\t' << selected_by << '\t'
                << rank_of(reference, vocabulary, token) << '\t' << rank_of(decoded, vocabulary, token) << '\t'
                << rank_of(mean, vocabulary, token) << '\t' << rank_of(null, vocabulary, token) << '\t'
                << token << '\t' << tsv_piece(common_token_to_piece(llama_model_get_vocab(model), token, true)) << '\t'
                << reference[token] << '\t' << decoded[token] << '\t' << mean[token] << '\t' << null[token] << '\n';
        }
    }

    std::ofstream summary(output_path);
    if (!summary) throw std::runtime_error("failed to write summary");
    summary << "source\ttarget\tholdout_bucket\ttrain_rows\ttest_rows\tlambda\tmean_residual_cosine\tdecoded_residual_cosine\tnull_residual_cosine\tdecoded_relative_mse\tdecoded_logit_cosine\tmean_logit_cosine\tnull_logit_cosine\tdecoded_top1_agreement\tmean_top1_agreement\tnull_top1_agreement\tdecoded_reference_top_rank\tmean_reference_top_rank\tnull_reference_top_rank\n";
    summary << sources[source_index] << '\t' << sources[target_index] << '\t' << holdout_bucket << '\t' << train_count << '\t' << test_count << '\t' << lambda << '\t'
        << mean_cosine / test_count << '\t' << prediction_cosine / test_count << '\t' << null_cosine / test_count << '\t'
        << prediction_squared_error / std::max(mean_squared_error, 1e-30) << '\t'
        << decoded_logit_cosine / test_count << '\t' << mean_logit_cosine / test_count << '\t' << null_logit_cosine / test_count << '\t'
        << decoded_top1_agreement << '/' << test_count << '\t' << mean_top1_agreement << '/' << test_count << '\t' << null_top1_agreement << '/' << test_count << '\t'
        << decoded_reference_top_rank / test_count << '\t' << mean_reference_top_rank / test_count << '\t' << null_reference_top_rank / test_count << '\n';

    const std::string viewer_path = output_path + ".viewer.json";
    write_viewer_trace(viewer_path, model_path, n_gpu_layers, sources[source_index], sources[target_index], source, test_rows, predictions, target, logits,
        dimension, vocabulary, llama_model_get_vocab(model), top);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    std::printf("wrote=%s samples=%s logits=%s viewer=%s test_rows=%zu\n", output_path.c_str(), (output_path + ".samples.tsv").c_str(), (output_path + ".logits.tsv").c_str(), viewer_path.c_str(), test_count);
    return 0;
}
