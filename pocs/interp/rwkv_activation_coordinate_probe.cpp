#include "rwkv_activation_store.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
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
        for (const char * tap : taps) {
            result.push_back("rwkv.layer." + std::to_string(layer) + "." + tap);
        }
    }
    return result;
}

size_t parse_source_index(const std::vector<std::string> & sources, const std::string & spec) {
    const size_t separator = spec.find(':');
    if (separator == std::string::npos) {
        throw std::runtime_error("source must be LAYER:TAP");
    }
    const int layer = std::atoi(spec.substr(0, separator).c_str());
    const std::string name = "rwkv.layer." + std::to_string(layer) + "." + spec.substr(separator + 1);
    const auto found = std::find(sources.begin(), sources.end(), name);
    if (found == sources.end()) {
        throw std::runtime_error("unknown source: " + spec);
    }
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

std::vector<float> project(
        const float * values,
        const std::vector<float> & mean,
        const std::vector<int8_t> & signs,
        size_t dimension,
        size_t rank) {
    std::vector<float> result(rank);
    const float scale = 1.0f / std::sqrt((float) dimension);
    for (size_t feature = 0; feature < rank; ++feature) {
        double sum = 0.0;
        const int8_t * direction = signs.data() + feature * dimension;
        for (size_t i = 0; i < dimension; ++i) {
            sum += (values[i] - mean[i]) * direction[i];
        }
        result[feature] = (float) (sum * scale);
    }
    return result;
}

void solve_ridge(std::vector<double> gram, std::vector<double> & cross, size_t rank, size_t dimension, double lambda) {
    for (size_t i = 0; i < rank; ++i) {
        gram[i * rank + i] += lambda;
        for (size_t j = 0; j < i; ++j) {
            double value = gram[i * rank + j];
            for (size_t k = 0; k < j; ++k) value -= gram[i * rank + k] * gram[j * rank + k];
            gram[i * rank + j] = value / gram[j * rank + j];
        }
        double diagonal = gram[i * rank + i];
        for (size_t k = 0; k < i; ++k) diagonal -= gram[i * rank + k] * gram[i * rank + k];
        if (diagonal <= 0.0) throw std::runtime_error("ridge system is not positive definite");
        gram[i * rank + i] = std::sqrt(diagonal);
    }
    for (size_t output = 0; output < dimension; ++output) {
        for (size_t row = 0; row < rank; ++row) {
            double value = cross[row * dimension + output];
            for (size_t column = 0; column < row; ++column) value -= gram[row * rank + column] * cross[column * dimension + output];
            cross[row * dimension + output] = value / gram[row * rank + row];
        }
        for (size_t row = rank; row-- > 0;) {
            double value = cross[row * dimension + output];
            for (size_t column = row + 1; column < rank; ++column) value -= gram[column * rank + row] * cross[column * dimension + output];
            cross[row * dimension + output] = value / gram[row * rank + row];
        }
    }
}

void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s --input ACTIVATIONS.root --source LAYER:TAP|all|all:TAP --output SUMMARY.tsv [--target LAYER:TAP] [--rank N] [--lambda REL] [--holdout-bucket N] [--batch-rows N]\n"
        "\n"
        "Fits a low-rank linear map from one captured source to the final residual. The held-out\n"
        "split is by corpus line, and a shifted-pairing fit is reported as a leakage control.\n",
        argv0);
}

} // namespace

int main(int argc, char ** argv) {
    std::string input_path;
    std::string source_spec;
    std::string target_spec = "60:resid.out";
    std::string output_path;
    size_t rank = 32;
    double lambda_relative = 0.01;
    uint64_t holdout_bucket = 0;
    size_t batch_rows = 64;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--input") == 0 && i + 1 < argc) input_path = argv[++i];
        else if (std::strcmp(argv[i], "--source") == 0 && i + 1 < argc) source_spec = argv[++i];
        else if (std::strcmp(argv[i], "--target") == 0 && i + 1 < argc) target_spec = argv[++i];
        else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) output_path = argv[++i];
        else if (std::strcmp(argv[i], "--rank") == 0 && i + 1 < argc) rank = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "--lambda") == 0 && i + 1 < argc) lambda_relative = std::strtod(argv[++i], nullptr);
        else if (std::strcmp(argv[i], "--holdout-bucket") == 0 && i + 1 < argc) holdout_bucket = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "--batch-rows") == 0 && i + 1 < argc) batch_rows = std::strtoull(argv[++i], nullptr, 10);
        else {
            usage(argv[0]);
            return 1;
        }
    }
    if (input_path.empty() || source_spec.empty() || output_path.empty() || rank == 0 || lambda_relative <= 0.0 || holdout_bucket >= 5 || batch_rows == 0) {
        usage(argv[0]);
        return 1;
    }

    const auto sources = all_sources();
    std::vector<size_t> source_indices;
    if (source_spec == "all") {
        source_indices.resize(sources.size());
        for (size_t i = 0; i < source_indices.size(); ++i) source_indices[i] = i;
    } else if (source_spec.rfind("all:", 0) == 0) {
        const std::string suffix = "." + source_spec.substr(4);
        for (size_t i = 0; i < sources.size(); ++i) {
            if (sources[i].size() >= suffix.size() && sources[i].ends_with(suffix)) source_indices.push_back(i);
        }
        if (source_indices.empty()) throw std::runtime_error("unknown source family: " + source_spec);
    } else {
        source_indices.push_back(parse_source_index(sources, source_spec));
    }
    const size_t target_index = parse_source_index(sources, target_spec);
    const auto dataset = rwkv_activation_store::activation_dataset_reader::open(input_path, sources);
    const size_t dimension = dataset.dimension();
    dataset_values target = load_source(dataset, target_index, batch_rows);

    std::vector<size_t> train_rows;
    std::vector<size_t> test_rows;
    for (size_t row = 0; row < target.metadata.size(); ++row) {
        // Keep every position from a corpus line on one side of the split.
        (splitmix64(target.metadata[row].corpus_line) % 5 == holdout_bucket ? test_rows : train_rows).push_back(row);
    }
    if (train_rows.size() < rank + 1 || test_rows.empty()) {
        throw std::runtime_error("insufficient grouped train/test rows for requested rank");
    }

    std::vector<float> target_mean(dimension);
    for (size_t row : train_rows) {
        const float * y = target.values.data() + row * dimension;
        for (size_t i = 0; i < dimension; ++i) {
            target_mean[i] += y[i];
        }
    }
    for (size_t i = 0; i < dimension; ++i) {
        target_mean[i] /= train_rows.size();
    }

    const std::filesystem::path output(output_path);
    if (!output.parent_path().empty()) std::filesystem::create_directories(output.parent_path());
    std::ofstream summary(output_path);
    if (!summary) throw std::runtime_error("failed to write summary");
    summary << "source\ttarget\tholdout_bucket\ttrain_rows\ttest_rows\trank\tlambda\traw_cosine\tmean_cosine\tprobe_cosine\tnull_cosine\tprobe_relative_mse\n";

    for (size_t sweep_index = 0; sweep_index < source_indices.size(); ++sweep_index) {
        const size_t source_index = source_indices[sweep_index];
        dataset_values source = load_source(dataset, source_index, batch_rows);
        if (source.metadata.size() != target.metadata.size() || source.values.size() != target.values.size()) {
            throw std::runtime_error("source and target datasets do not match");
        }
        for (size_t row = 0; row < source.metadata.size(); ++row) {
            if (source.metadata[row].sample_id != target.metadata[row].sample_id) {
                throw std::runtime_error("source and target row order differs");
            }
        }

        std::vector<float> source_mean(dimension);
        for (size_t row : train_rows) {
            const float * x = source.values.data() + row * dimension;
            for (size_t i = 0; i < dimension; ++i) source_mean[i] += x[i];
        }
        for (float & value : source_mean) value /= train_rows.size();

        std::vector<int8_t> signs(rank * dimension);
        for (size_t feature = 0; feature < rank; ++feature) {
            uint64_t state = splitmix64((source_index + 1) * 0x9e3779b97f4a7c15ULL + feature);
            for (size_t i = 0; i < dimension; ++i) {
                state = splitmix64(state);
                signs[feature * dimension + i] = (state >> 63) ? 1 : -1;
            }
        }

        std::vector<double> gram(rank * rank);
        std::vector<double> cross(rank * dimension);
        std::vector<double> null_cross(rank * dimension);
        for (size_t train_position = 0; train_position < train_rows.size(); ++train_position) {
            const size_t row = train_rows[train_position];
            const size_t null_row = train_rows[(train_position + 1) % train_rows.size()];
            const auto features = project(source.values.data() + row * dimension, source_mean, signs, dimension, rank);
            const float * y = target.values.data() + row * dimension;
            const float * null_y = target.values.data() + null_row * dimension;
            for (size_t left = 0; left < rank; ++left) {
                for (size_t right = 0; right < rank; ++right) gram[left * rank + right] += features[left] * features[right];
                for (size_t i = 0; i < dimension; ++i) {
                    cross[left * dimension + i] += features[left] * (y[i] - target_mean[i]);
                    null_cross[left * dimension + i] += features[left] * (null_y[i] - target_mean[i]);
                }
            }
        }
        double trace = 0.0;
        for (size_t i = 0; i < rank; ++i) trace += gram[i * rank + i];
        const double lambda = lambda_relative * trace / rank;
        solve_ridge(gram, cross, rank, dimension, lambda);
        solve_ridge(gram, null_cross, rank, dimension, lambda);

        double raw_cosine = 0.0;
        double mean_cosine = 0.0;
        double probe_cosine = 0.0;
        double null_cosine = 0.0;
        double probe_squared_error = 0.0;
        double target_squared_error = 0.0;
        std::vector<float> prediction(dimension);
        std::vector<float> null_prediction(dimension);
        for (size_t row : test_rows) {
        const float * x = source.values.data() + row * dimension;
        const float * y = target.values.data() + row * dimension;
        const auto features = project(x, source_mean, signs, dimension, rank);
        for (size_t i = 0; i < dimension; ++i) {
            double value = target_mean[i];
            double null_value = target_mean[i];
            for (size_t feature = 0; feature < rank; ++feature) {
                value += features[feature] * cross[feature * dimension + i];
                null_value += features[feature] * null_cross[feature * dimension + i];
            }
            prediction[i] = (float) value;
            null_prediction[i] = (float) null_value;
            const double probe_delta = value - y[i];
            const double target_delta = target_mean[i] - y[i];
            probe_squared_error += probe_delta * probe_delta;
            target_squared_error += target_delta * target_delta;
        }
        raw_cosine += cosine(x, y, dimension);
        mean_cosine += cosine(target_mean.data(), y, dimension);
        probe_cosine += cosine(prediction.data(), y, dimension);
        null_cosine += cosine(null_prediction.data(), y, dimension);
        }
        summary << sources[source_index] << '\t' << sources[target_index] << '\t' << holdout_bucket << '\t' << train_rows.size() << '\t' << test_rows.size() << '\t'
            << rank << '\t' << lambda << '\t' << raw_cosine / test_rows.size() << '\t' << mean_cosine / test_rows.size() << '\t'
            << probe_cosine / test_rows.size() << '\t' << null_cosine / test_rows.size() << '\t'
            << probe_squared_error / std::max(target_squared_error, 1e-30) << '\n';
        if ((sweep_index + 1) % 64 == 0 || sweep_index + 1 == source_indices.size()) {
            std::printf("probed=%zu/%zu\n", sweep_index + 1, source_indices.size());
        }
    }
    std::printf("wrote=%s sources=%zu\n", output_path.c_str(), source_indices.size());
    return 0;
}
