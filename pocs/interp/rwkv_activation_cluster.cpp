#include "rwkv_activation_store.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
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

size_t nearest_center(const float * row, const std::vector<float> & centers, size_t k, size_t dimension) {
    size_t nearest = 0;
    double best = INFINITY;
    for (size_t cluster = 0; cluster < k; ++cluster) {
        const float * center = centers.data() + cluster * dimension;
        double distance = 0.0;
        for (size_t i = 0; i < dimension; ++i) {
            const double delta = (double) row[i] - center[i];
            distance += delta * delta;
        }
        if (distance < best) {
            best = distance;
            nearest = cluster;
        }
    }
    return nearest;
}

std::vector<rwkv_activation_store::centroid> fit_source(
        const rwkv_activation_store::activation_dataset_reader & dataset,
        size_t source_index,
        size_t k,
        size_t batch_rows,
        size_t iterations) {
    const size_t dimension = dataset.dimension();
    std::vector<float> centers(k * dimension);
    size_t seeds = 0;
    dataset.for_each_source_batch(source_index, batch_rows, [&](rwkv_activation_store::source_batch && batch) {
        for (size_t row = 0; row < batch.metadata.size() && seeds < k; ++row, ++seeds) {
            const float * values = batch.values.data() + row * dimension;
            std::copy(values, values + dimension, centers.begin() + seeds * dimension);
        }
    });
    if (seeds != k) {
        throw std::runtime_error("dataset has fewer rows than requested cluster count");
    }

    std::vector<uint64_t> counts(k);
    for (size_t iteration = 0; iteration < iterations; ++iteration) {
        std::fill(counts.begin(), counts.end(), 0);
        std::vector<double> sums(k * dimension, 0.0);
        dataset.for_each_source_batch(source_index, batch_rows, [&](rwkv_activation_store::source_batch && batch) {
            for (size_t row = 0; row < batch.metadata.size(); ++row) {
                const float * values = batch.values.data() + row * dimension;
                const size_t cluster = nearest_center(values, centers, k, dimension);
                ++counts[cluster];
                double * sum = sums.data() + cluster * dimension;
                for (size_t i = 0; i < dimension; ++i) {
                    sum[i] += values[i];
                }
            }
        });
        for (size_t cluster = 0; cluster < k; ++cluster) {
            if (counts[cluster] == 0) {
                continue;
            }
            float * center = centers.data() + cluster * dimension;
            const double * sum = sums.data() + cluster * dimension;
            for (size_t i = 0; i < dimension; ++i) {
                center[i] = (float) (sum[i] / counts[cluster]);
            }
        }
    }

    std::fill(counts.begin(), counts.end(), 0);
    dataset.for_each_source_batch(source_index, batch_rows, [&](rwkv_activation_store::source_batch && batch) {
        for (size_t row = 0; row < batch.metadata.size(); ++row) {
            ++counts[nearest_center(batch.values.data() + row * dimension, centers, k, dimension)];
        }
    });
    std::vector<rwkv_activation_store::centroid> result;
    result.reserve(k);
    for (size_t cluster = 0; cluster < k; ++cluster) {
        result.push_back({
            (uint32_t) source_index,
            (uint32_t) k,
            (uint32_t) cluster,
            counts[cluster],
            std::vector<float>(centers.begin() + cluster * dimension, centers.begin() + (cluster + 1) * dimension),
        });
    }
    return result;
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

rwkv_activation_store::centroid fit_trimmed_global(
        const rwkv_activation_store::activation_dataset_reader & dataset,
        size_t source_index,
        size_t batch_rows,
        double trim_mad,
        size_t trim_samples) {
    const size_t dimension = dataset.dimension();
    const auto initial = fit_source(dataset, source_index, 1, batch_rows, 1)[0];
    std::vector<double> distances;
    distances.reserve(trim_samples);
    dataset.for_each_source_batch(source_index, batch_rows, [&](rwkv_activation_store::source_batch && batch) {
        for (size_t row = 0; row < batch.metadata.size() && distances.size() < trim_samples; ++row) {
            const float * values = batch.values.data() + row * dimension;
            double squared = 0.0;
            for (size_t i = 0; i < dimension; ++i) {
                const double delta = values[i] - initial.values[i];
                squared += delta * delta;
            }
            distances.push_back(std::sqrt(squared));
        }
    });
    const double center_distance = median(distances);
    std::vector<double> deviations;
    deviations.reserve(distances.size());
    for (double distance : distances) deviations.push_back(std::abs(distance - center_distance));
    const double cutoff = center_distance + trim_mad * median(deviations);
    std::vector<double> sum(dimension, 0.0);
    uint64_t retained = 0;
    dataset.for_each_source_batch(source_index, batch_rows, [&](rwkv_activation_store::source_batch && batch) {
        for (size_t row = 0; row < batch.metadata.size(); ++row) {
            const float * values = batch.values.data() + row * dimension;
            double squared = 0.0;
            for (size_t i = 0; i < dimension; ++i) {
                const double delta = values[i] - initial.values[i];
                squared += delta * delta;
            }
            if (std::sqrt(squared) > cutoff) continue;
            ++retained;
            for (size_t i = 0; i < dimension; ++i) sum[i] += values[i];
        }
    });
    if (retained == 0) {
        throw std::runtime_error("outlier cutoff removed every activation row");
    }
    std::vector<float> values(dimension);
    for (size_t i = 0; i < dimension; ++i) values[i] = (float) (sum[i] / retained);
    return { (uint32_t) source_index, 1, 0, retained, std::move(values) };
}

void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s --input ACTIVATIONS.root --output CENTROIDS.root [--batch-rows N] [--iterations N] [--k N] [--trim-mad N] [--trim-samples N]\n"
        "\n"
        "Fits one source at a time with bounded ROOT RNTuple reads. --k defaults to 1; run\n"
        "again with --k 8 for the cluster-conditioned pilot. --trim-mad enables a K=1\n"
        "radial median-plus-MAD refit using at most --trim-samples distances.\n",
        argv0);
}

} // namespace

int main(int argc, char ** argv) {
    std::string input_path;
    std::string output_path;
    size_t batch_rows = 128;
    size_t iterations = 8;
    size_t k = 1;
    double trim_mad = 0.0;
    size_t trim_samples = 8192;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            input_path = argv[++i];
        } else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_path = argv[++i];
        } else if (std::strcmp(argv[i], "--batch-rows") == 0 && i + 1 < argc) {
            batch_rows = std::strtoull(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            iterations = std::strtoull(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "--k") == 0 && i + 1 < argc) {
            k = std::strtoull(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "--trim-mad") == 0 && i + 1 < argc) {
            trim_mad = std::strtod(argv[++i], nullptr);
        } else if (std::strcmp(argv[i], "--trim-samples") == 0 && i + 1 < argc) {
            trim_samples = std::strtoull(argv[++i], nullptr, 10);
        } else {
            usage(argv[0]);
            return 1;
        }
    }
    if (input_path.empty() || output_path.empty() || batch_rows == 0 || iterations == 0 || k == 0 || trim_mad < 0.0 || trim_samples == 0) {
        usage(argv[0]);
        return 1;
    }
    const auto sources = all_sources();
    const auto dataset = rwkv_activation_store::activation_dataset_reader::open(input_path, sources);
    if (dataset.entries() < k) {
        throw std::runtime_error("cluster count exceeds activation rows");
    }
    if (trim_mad > 0.0 && k != 1) {
        throw std::runtime_error("--trim-mad currently requires --k 1");
    }
    std::vector<rwkv_activation_store::centroid> centroids;
    centroids.reserve(sources.size() * k);
    for (size_t source = 0; source < sources.size(); ++source) {
        if (trim_mad > 0.0) {
            centroids.push_back(fit_trimmed_global(dataset, source, batch_rows, trim_mad, trim_samples));
        } else {
            auto fitted = fit_source(dataset, source, k, batch_rows, iterations);
            centroids.insert(centroids.end(), std::make_move_iterator(fitted.begin()), std::make_move_iterator(fitted.end()));
        }
        if ((source + 1) % 8 == 0 || source + 1 == sources.size()) {
            std::printf("fitted=%zu/%zu k=%zu\n", source + 1, sources.size(), k);
        }
    }
    rwkv_activation_store::write_centroids(output_path, centroids, dataset.dimension());
    std::printf("wrote=%s sources=%zu k=%zu trim_mad=%.3g\n", output_path.c_str(), sources.size(), k, trim_mad);
    return 0;
}
