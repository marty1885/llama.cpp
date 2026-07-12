#include "rwkv_activation_store.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<std::string> all_sources() {
    const std::vector<const char *> time_mix = {
        "time.r", "time.w", "time.k", "time.v", "time.a", "time.g", "time.wkv", "time.rkv",
    };
    std::vector<std::string> result;
    for (int layer = 0; layer < 61; ++layer) {
        for (const char * activation : time_mix) {
            result.push_back("rwkv.layer." + std::to_string(layer) + "." + activation);
        }
    }
    result.push_back("rwkv.layer.60.resid.out");
    return result;
}

double l2(const float * values, size_t dimension) {
    double sum = 0.0;
    for (size_t i = 0; i < dimension; ++i) sum += (double) values[i] * values[i];
    return std::sqrt(sum);
}

double median(std::vector<double> values) {
    if (values.empty()) return 0.0;
    const size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + middle, values.end());
    const double upper = values[middle];
    if (values.size() % 2) return upper;
    std::nth_element(values.begin(), values.begin() + middle - 1, values.end());
    return (upper + values[middle - 1]) / 2.0;
}

void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s --input ACTIVATIONS.root --k1 K1.root [--k8 K8.root] --output SUMMARY.tsv [--batch-rows N] [--pair-samples N]\n"
        "\n"
        "Streams one source at a time. Outliers are diagnostic radial flags only; this tool never\n"
        "removes activation rows or refits centroids.\n",
        argv0);
}

} // namespace

int main(int argc, char ** argv) {
    std::string input_path;
    std::string k1_path;
    std::string k8_path;
    std::string output_path;
    size_t batch_rows = 128;
    size_t pair_samples = 256;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--input") == 0 && i + 1 < argc) input_path = argv[++i];
        else if (std::strcmp(argv[i], "--k1") == 0 && i + 1 < argc) k1_path = argv[++i];
        else if (std::strcmp(argv[i], "--k8") == 0 && i + 1 < argc) k8_path = argv[++i];
        else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) output_path = argv[++i];
        else if (std::strcmp(argv[i], "--batch-rows") == 0 && i + 1 < argc) batch_rows = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "--pair-samples") == 0 && i + 1 < argc) pair_samples = std::strtoull(argv[++i], nullptr, 10);
        else {
            usage(argv[0]);
            return 1;
        }
    }
    if (input_path.empty() || k1_path.empty() || output_path.empty() || batch_rows == 0 || pair_samples == 0) {
        usage(argv[0]);
        return 1;
    }
    const auto sources = all_sources();
    const auto dataset = rwkv_activation_store::activation_dataset_reader::open(input_path, sources);
    const auto k1 = rwkv_activation_store::centroid_store_reader::open(k1_path);
    std::unique_ptr<rwkv_activation_store::centroid_store_reader> k8;
    if (!k8_path.empty()) {
        k8 = std::make_unique<rwkv_activation_store::centroid_store_reader>(
            rwkv_activation_store::centroid_store_reader::open(k8_path));
    }
    std::ofstream output(output_path);
    if (!output) throw std::runtime_error("failed to open summary output: " + output_path);
    output << "source\trows\traw_norm_mean\traw_norm_min\traw_norm_max\tcenter_norm\tresidual_norm_mean\tresidual_norm_min\tresidual_norm_max\tresidual_ratio_mean\tpair_cosine_mean\tpair_distance_mean\tdiagnostic_outliers_3mad\tk8_min_count\tk8_max_count\tk8_singletons\tk8_empty\n";

    const size_t dimension = dataset.dimension();
    for (size_t source = 0; source < sources.size(); ++source) {
        const auto global = k1.read_source((uint32_t) source, 1);
        const auto clustered = k8 ? k8->read_source((uint32_t) source, 8) : std::vector<rwkv_activation_store::centroid>();
        if (global.size() != 1 || global[0].values.size() != dimension || (k8 && clustered.size() != 8)) {
            throw std::runtime_error("centroid file does not match activation schema for source " + std::to_string(source));
        }
        const auto & center = global[0].values;
        const double center_norm = l2(center.data(), dimension);
        uint64_t rows = 0;
        double raw_sum = 0.0;
        double residual_sum = 0.0;
        double ratio_sum = 0.0;
        double raw_min = std::numeric_limits<double>::infinity();
        double raw_max = 0.0;
        double residual_min = std::numeric_limits<double>::infinity();
        double residual_max = 0.0;
        std::vector<double> residual_norms;
        std::vector<float> pair_rows;
        dataset.for_each_source_batch(source, batch_rows, [&](rwkv_activation_store::source_batch && batch) {
            for (size_t row = 0; row < batch.metadata.size(); ++row) {
                const float * values = batch.values.data() + row * dimension;
                double raw_squared = 0.0;
                double residual_squared = 0.0;
                for (size_t i = 0; i < dimension; ++i) {
                    raw_squared += (double) values[i] * values[i];
                    const double delta = values[i] - center[i];
                    residual_squared += delta * delta;
                }
                const double raw_norm = std::sqrt(raw_squared);
                const double residual_norm = std::sqrt(residual_squared);
                ++rows;
                raw_sum += raw_norm;
                residual_sum += residual_norm;
                ratio_sum += residual_norm / std::max(raw_norm, 1e-12);
                raw_min = std::min(raw_min, raw_norm);
                raw_max = std::max(raw_max, raw_norm);
                residual_min = std::min(residual_min, residual_norm);
                residual_max = std::max(residual_max, residual_norm);
                if (residual_norms.size() < pair_samples) residual_norms.push_back(residual_norm);
                if (pair_rows.size() / dimension < pair_samples) pair_rows.insert(pair_rows.end(), values, values + dimension);
            }
        });
        const double residual_median = median(residual_norms);
        std::vector<double> deviations;
        deviations.reserve(residual_norms.size());
        for (double value : residual_norms) deviations.push_back(std::abs(value - residual_median));
        const double mad = median(deviations);
        uint64_t diagnostic_outliers = 0;
        for (double value : residual_norms) if (value > residual_median + 3.0 * mad) ++diagnostic_outliers;

        double pair_cosine_sum = 0.0;
        double pair_distance_sum = 0.0;
        uint64_t pairs = 0;
        const size_t pair_count = pair_rows.size() / dimension;
        for (size_t a = 0; a < pair_count; ++a) {
            const float * left = pair_rows.data() + a * dimension;
            const double left_norm = l2(left, dimension);
            for (size_t b = a + 1; b < pair_count; ++b) {
                const float * right = pair_rows.data() + b * dimension;
                const double right_norm = l2(right, dimension);
                double dot = 0.0;
                double squared_distance = 0.0;
                for (size_t i = 0; i < dimension; ++i) {
                    dot += (double) left[i] * right[i];
                    const double delta = left[i] - right[i];
                    squared_distance += delta * delta;
                }
                pair_cosine_sum += dot / std::max(left_norm * right_norm, 1e-12);
                pair_distance_sum += std::sqrt(squared_distance);
                ++pairs;
            }
        }
        uint64_t min_count = 0;
        uint64_t max_count = 0;
        uint64_t singletons = 0;
        uint64_t empty = 0;
        if (k8) {
            min_count = clustered[0].count;
            for (const auto & cluster : clustered) {
                min_count = std::min(min_count, cluster.count);
                max_count = std::max(max_count, cluster.count);
                singletons += cluster.count == 1;
                empty += cluster.count == 0;
            }
        }
        output << sources[source] << '\t' << rows << '\t' << raw_sum / rows << '\t' << raw_min << '\t' << raw_max << '\t'
               << center_norm << '\t' << residual_sum / rows << '\t' << residual_min << '\t' << residual_max << '\t'
               << ratio_sum / rows << '\t' << pair_cosine_sum / std::max<uint64_t>(pairs, 1) << '\t'
               << pair_distance_sum / std::max<uint64_t>(pairs, 1) << '\t' << diagnostic_outliers << '\t'
               << min_count << '\t' << max_count << '\t' << singletons << '\t' << empty << '\n';
        if ((source + 1) % 8 == 0 || source + 1 == sources.size()) {
            std::printf("analyzed=%zu/%zu\n", source + 1, sources.size());
        }
    }
    std::printf("wrote=%s sources=%zu\n", output_path.c_str(), sources.size());
    return 0;
}
