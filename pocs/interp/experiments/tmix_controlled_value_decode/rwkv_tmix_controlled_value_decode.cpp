#include "rwkv_activation_store.h"

#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleReader.hxx>
#include <ROOT/RNTupleWriter.hxx>
#include <TFile.h>
#include <boost/math/distributions/students_t.hpp>
#include <cblas.h>
#include <openblas/lapacke.h>
#include <omp.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <regex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

constexpr int k_rank = 32;
constexpr uint64_t k_seed = 941731;
constexpr std::array<double, 7> k_lambda_grid{ 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0 };

std::string read_text(const std::string & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot read " + path);
    return { std::istreambuf_iterator<char>(input), {} };
}

std::string json_string(const std::string & text, const char * key) {
    std::smatch match;
    if (!std::regex_search(text, match, std::regex(std::string("\\\"") + key + "\\\"\\s*:\\s*\\\"([^\\\"]+)\\\""))) {
        throw std::runtime_error(std::string("JSON lacks ") + key);
    }
    return match[1];
}

uint64_t json_integer(const std::string & text, const char * key) {
    std::smatch match;
    if (!std::regex_search(text, match, std::regex(std::string("\\\"") + key + "\\\"\\s*:\\s*([0-9]+)"))) {
        throw std::runtime_error(std::string("JSON lacks ") + key);
    }
    return std::stoull(match[1]);
}

std::map<std::string, std::string> read_capture_metadata(const std::string & path) {
    auto reader = ROOT::RNTupleReader::Open("metadata", path);
    auto key = reader->GetView<std::string>("key");
    auto value = reader->GetView<std::string>("value");
    std::map<std::string, std::string> result;
    for (auto entry : reader->GetEntryRange()) {
        if (!result.emplace(key(entry), value(entry)).second) throw std::runtime_error("duplicate capture metadata key");
    }
    return result;
}

void verify_classified_capture(const std::string & input, const std::string & registration_path) {
    const std::string registration = read_text(registration_path);
    if (json_string(registration, "status") != "frozen" || json_integer(registration, "seed") != k_seed ||
        json_integer(registration, "representation_rank") != k_rank) throw std::runtime_error("invalid frozen registration");
    const auto metadata = read_capture_metadata(input);
    const auto require = [&](const char * key) -> const std::string & {
        const auto it = metadata.find(key);
        if (it == metadata.end()) throw std::runtime_error(std::string("capture metadata lacks ") + key);
        return it->second;
    };
    if (require("status") != "frozen_capture" || require("model_sha256") != json_string(registration, "model_sha256") ||
        require("corpus_sha256") != json_string(registration, "corpus_sha256") ||
        require("manifest_sha256") != json_string(registration, "manifest_sha256")) {
        throw std::runtime_error("capture metadata does not match frozen registration");
    }
    auto samples = ROOT::RNTupleReader::Open("samples", input);
    if (samples->GetNEntries() != 3072) throw std::runtime_error("classified capture lacks complete sample rows");
    auto carrier = samples->GetView<int32_t>("carrier_id");
    auto value = samples->GetView<int32_t>("value_id");
    auto layer = samples->GetView<int32_t>("layer");
    auto identity = samples->GetView<double>("residual_identity_error");
    std::array<bool, 4 * 48 * 16> seen{};
    for (auto entry : samples->GetEntryRange()) {
        const int layer_index = layer(entry) == 15 ? 0 : layer(entry) == 30 ? 1 : layer(entry) == 45 ? 2 : layer(entry) == 60 ? 3 : -1;
        if (layer_index < 0 || carrier(entry) < 0 || carrier(entry) >= 48 || value(entry) < 0 || value(entry) >= 16 ||
            identity(entry) > 1e-3 || !std::isfinite(identity(entry))) throw std::runtime_error("invalid classified sample row");
        bool & cell = seen[size_t(layer_index * 48 * 16 + carrier(entry) * 16 + value(entry))];
        if (cell) throw std::runtime_error("duplicate classified sample cell");
        cell = true;
    }
    if (std::find(seen.begin(), seen.end(), false) != seen.end()) throw std::runtime_error("missing classified sample cell");
}

struct behavior_summary { std::string group; int32_t id; int32_t count; double log_probability; double rank; double top1_accuracy; };

std::vector<behavior_summary> summarize_behavior(const std::string & input) {
    auto reader = ROOT::RNTupleReader::Open("behavior", input);
    if (reader->GetNEntries() != 768) throw std::runtime_error("classified capture lacks complete behavior rows");
    auto prompt = reader->GetView<uint64_t>("prompt_id");
    auto log_probability = reader->GetView<double>("expected_value_log_probability");
    auto rank = reader->GetView<int32_t>("expected_value_rank");
    auto correct = reader->GetView<bool>("native_top1_correct");
    struct accumulator { int count = 0; double log_probability = 0, rank = 0, correct = 0; };
    std::array<accumulator, 48> carriers{}; std::array<accumulator, 16> values{}; accumulator all;
    for (auto entry : reader->GetEntryRange()) {
        if (prompt(entry) >= 768 || !std::isfinite(log_probability(entry)) || rank(entry) < 1) throw std::runtime_error("invalid behavior row");
        const int carrier = int(prompt(entry) / 16), value = int(prompt(entry) % 16);
        for (accumulator * target : { &all, &carriers[carrier], &values[value] }) {
            ++target->count; target->log_probability += log_probability(entry); target->rank += rank(entry); target->correct += correct(entry);
        }
    }
    std::vector<behavior_summary> result;
    const auto append = [&](const char * group, int id, const accumulator & x) {
        if (x.count == 0) throw std::runtime_error("missing behavior group");
        result.push_back({ group, id, x.count, x.log_probability / x.count, x.rank / x.count, x.correct / x.count });
    };
    append("all", -1, all); for (int i = 0; i < 48; ++i) append("carrier", i, carriers[i]); for (int i = 0; i < 16; ++i) append("value", i, values[i]);
    return result;
}

double sign_flip_p_value(const std::vector<double> & deltas);
std::vector<double> holm_adjust(std::vector<double> p_values);
std::string mechanical_outcome(bool ambient, bool transported, bool additive, bool endpoint);

// Reopen the completed artifact and derive the aggregate decoder metrics from the
// serialized predictions, rather than trusting in-memory summaries.
bool audit_result_root(const std::string & path) {
    auto predictions = ROOT::RNTupleReader::Open("predictions", path);
    auto aggregates = ROOT::RNTupleReader::Open("aggregate_metrics", path);
    auto fits = ROOT::RNTupleReader::Open("decoder_fits", path);
    auto cv = ROOT::RNTupleReader::Open("cross_validation_scores", path);
    auto carriers = ROOT::RNTupleReader::Open("per_carrier_metrics", path);
    auto tests = ROOT::RNTupleReader::Open("statistical_tests", path);
    auto outcomes = ROOT::RNTupleReader::Open("outcome_predicates", path);
    auto behavior = ROOT::RNTupleReader::Open("behavior_diagnostics", path);
    if (predictions->GetNEntries() != 5120 || aggregates->GetNEntries() != 20 || fits->GetNEntries() != 20 ||
        cv->GetNEntries() != 140 || carriers->GetNEntries() != 1024 || tests->GetNEntries() != 24 ||
        outcomes->GetNEntries() != 4 || behavior->GetNEntries() != 65) return false;
    auto prediction_layer = predictions->GetView<int32_t>("layer");
    auto prediction_model = predictions->GetView<std::string>("model");
    auto prediction_ce = predictions->GetView<double>("cross_entropy");
    auto prediction_correct = predictions->GetView<bool>("correct");
    std::map<std::pair<int, std::string>, std::array<double, 3>> derived;
    for (auto entry : predictions->GetEntryRange()) {
        const double ce = prediction_ce(entry);
        if (!std::isfinite(ce)) return false;
        auto & x = derived[{ prediction_layer(entry), prediction_model(entry) }];
        x[0] += ce; x[1] += prediction_correct(entry) ? 1.0 : 0.0; x[2] += 1.0;
    }
    if (derived.size() != 20) return false;
    auto aggregate_layer = aggregates->GetView<int32_t>("layer");
    auto aggregate_model = aggregates->GetView<std::string>("model");
    auto aggregate_ce = aggregates->GetView<double>("carrier_balanced_cross_entropy");
    auto aggregate_accuracy = aggregates->GetView<double>("carrier_balanced_accuracy");
    for (auto entry : aggregates->GetEntryRange()) {
        const auto it = derived.find({ aggregate_layer(entry), aggregate_model(entry) });
        if (it == derived.end() || it->second[2] != 256.0 || !std::isfinite(aggregate_ce(entry)) || !std::isfinite(aggregate_accuracy(entry)) ||
            std::abs(aggregate_ce(entry) - it->second[0] / 256.0) > 1e-12 ||
            std::abs(aggregate_accuracy(entry) - it->second[1] / 256.0) > 1e-12) return false;
    }
    auto fit_converged = fits->GetView<bool>("converged");
    auto fit_gradient = fits->GetView<double>("gradient_norm");
    for (auto entry : fits->GetEntryRange()) if (!fit_converged(entry) || !std::isfinite(fit_gradient(entry))) return false;
    auto carrier_layer = carriers->GetView<int32_t>("layer");
    auto carrier_id = carriers->GetView<int32_t>("carrier_id");
    auto metric_name = carriers->GetView<std::string>("metric_name");
    auto metric_value = carriers->GetView<double>("metric_value");
    std::map<std::tuple<int, int, std::string>, double> metric;
    for (auto entry : carriers->GetEntryRange()) {
        const auto key = std::make_tuple(int(carrier_layer(entry)), int(carrier_id(entry)), metric_name(entry));
        if (!std::isfinite(metric_value(entry)) || !metric.emplace(key, metric_value(entry)).second) return false;
    }
    std::array<std::array<double, 4>, 6> raw{}, adjusted{};
    const std::array<const char *, 6> names{ "gain_additive", "gain_relative_ambient", "gain_relative_transported", "gain_endpoint", "advantage_ambient", "advantage_transported" };
    const std::array<int, 4> layers{ 15, 30, 45, 60 };
    auto test_layer = tests->GetView<int32_t>("layer");
    auto test_name = tests->GetView<std::string>("test_name");
    auto test_raw = tests->GetView<double>("raw_p_value");
    auto test_adjusted = tests->GetView<double>("holm_adjusted_p_value");
    std::map<std::pair<int, std::string>, std::pair<double, double>> serialized_tests;
    for (auto entry : tests->GetEntryRange()) serialized_tests[{ test_layer(entry), test_name(entry) }] = { test_raw(entry), test_adjusted(entry) };
    for (size_t test = 0; test < names.size(); ++test) for (size_t layer = 0; layer < layers.size(); ++layer) {
        std::vector<double> deltas;
        for (int carrier = 0; carrier < 16; ++carrier) {
            const auto it = metric.find({ layers[layer], carrier, names[test] });
            if (it == metric.end()) return false;
            deltas.push_back(it->second);
        }
        raw[test][layer] = sign_flip_p_value(deltas);
    }
    const auto additive_holm = holm_adjust({ raw[0][0], raw[0][1], raw[0][2], raw[0][3] });
    const auto relative_gain_holm = holm_adjust({ raw[1][0], raw[2][0], raw[1][1], raw[2][1], raw[1][2], raw[2][2], raw[1][3], raw[2][3] });
    const auto endpoint_holm = holm_adjust({ raw[3][0], raw[3][1], raw[3][2], raw[3][3] });
    const auto relative_advantage_holm = holm_adjust({ raw[4][0], raw[5][0], raw[4][1], raw[5][1], raw[4][2], raw[5][2], raw[4][3], raw[5][3] });
    for (size_t layer = 0; layer < 4; ++layer) {
        adjusted[0][layer] = additive_holm[layer]; adjusted[3][layer] = endpoint_holm[layer];
        adjusted[1][layer] = relative_gain_holm[2 * layer]; adjusted[2][layer] = relative_gain_holm[2 * layer + 1];
        adjusted[4][layer] = relative_advantage_holm[2 * layer]; adjusted[5][layer] = relative_advantage_holm[2 * layer + 1];
    }
    for (size_t test = 0; test < names.size(); ++test) for (size_t layer = 0; layer < layers.size(); ++layer) {
        const auto it = serialized_tests.find({ layers[layer], names[test] });
        if (it == serialized_tests.end() || std::abs(it->second.first - raw[test][layer]) > 1e-15 || std::abs(it->second.second - adjusted[test][layer]) > 1e-15) return false;
    }
    auto outcome_layer = outcomes->GetView<int32_t>("layer");
    auto outcome_name = outcomes->GetView<std::string>("outcome");
    auto outcome_ambient = outcomes->GetView<bool>("relative_ambient_passes");
    auto outcome_transported = outcomes->GetView<bool>("relative_transported_passes");
    auto outcome_additive = outcomes->GetView<bool>("additive_passes");
    auto outcome_endpoint = outcomes->GetView<bool>("endpoint_passes");
    for (auto entry : outcomes->GetEntryRange()) {
        const auto layer_it = std::find(layers.begin(), layers.end(), outcome_layer(entry));
        if (layer_it == layers.end()) return false;
        const size_t layer = size_t(layer_it - layers.begin());
        std::array<double, 6> means{};
        for (size_t test = 0; test < names.size(); ++test) for (int carrier = 0; carrier < 16; ++carrier) means[test] += metric.at({ layers[layer], carrier, names[test] }) / 16.0;
        const bool ambient = means[1] > 0 && adjusted[1][layer] <= .05 && means[4] > 0 && adjusted[4][layer] <= .05;
        const bool transported = means[2] > 0 && adjusted[2][layer] <= .05 && means[5] > 0 && adjusted[5][layer] <= .05;
        const bool additive = means[0] > 0 && adjusted[0][layer] <= .05;
        const bool endpoint = means[3] > 0 && adjusted[3][layer] <= .05;
        const std::string expected = mechanical_outcome(ambient, transported, additive, endpoint);
        if (outcome_name(entry) != expected || outcome_ambient(entry) != ambient || outcome_transported(entry) != transported ||
            outcome_additive(entry) != additive || outcome_endpoint(entry) != endpoint) return false;
    }
    return true;
}

std::vector<float> fit_rank32_basis(const std::vector<float> & matrix, size_t rows, size_t width) {
    if (rows < k_rank || matrix.size() != rows * width) throw std::runtime_error("invalid train feature matrix");
    std::vector<float> gram(rows * rows), eigen(k_rank), vectors(rows * k_rank);
    std::vector<lapack_int> support(2 * k_rank);
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, int(rows), int(rows), int(width), 1.0f, matrix.data(), int(rows),
                matrix.data(), int(rows), 0.0f, gram.data(), int(rows));
    lapack_int selected = 0;
    if (LAPACKE_ssyevr(LAPACK_COL_MAJOR, 'V', 'I', 'U', int(rows), gram.data(), int(rows), 0, 0, int(rows) - k_rank + 1,
                        int(rows), 0, &selected, eigen.data(), vectors.data(), int(rows), support.data()) != 0 || selected != k_rank) {
        throw std::runtime_error("train Gram eigendecomposition failed");
    }
    std::vector<float> basis(width * k_rank);
    for (int component = 0; component < k_rank; ++component) {
        const int eigen_index = k_rank - 1 - component;
        const float singular = std::sqrt(std::max(0.0f, eigen[eigen_index]));
        if (!(singular > 1e-10f)) throw std::runtime_error("rank-deficient train basis");
        for (size_t column = 0; column < width; ++column) {
            double value = 0.0;
            for (size_t row = 0; row < rows; ++row) value += matrix[row + rows * column] * vectors[row + rows * eigen_index];
            basis[column + width * component] = float(value / singular);
        }
    }
    return basis;
}

void reorthogonalize(std::vector<float> & basis, size_t width) {
    for (int component = 0; component < k_rank; ++component) {
        for (int prior = 0; prior < component; ++prior) {
            double projection = 0.0;
            for (size_t j = 0; j < width; ++j) projection += double(basis[j + width * component]) * basis[j + width * prior];
            for (size_t j = 0; j < width; ++j) basis[j + width * component] -= float(projection * basis[j + width * prior]);
        }
        double norm = 0.0;
        for (size_t j = 0; j < width; ++j) norm += double(basis[j + width * component]) * basis[j + width * component];
        if (!(norm > 1e-20)) throw std::runtime_error("reorthogonalized train basis is rank deficient");
        for (size_t j = 0; j < width; ++j) basis[j + width * component] /= float(std::sqrt(norm));
    }
}

std::vector<float> project_rows(const std::vector<float> & rows, size_t count, size_t width, const std::vector<float> & basis) {
    if (rows.size() != count * width || basis.size() != width * k_rank) throw std::runtime_error("projection shape mismatch");
    std::vector<float> result(count * k_rank);
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, k_rank, int(count), int(width), 1.0f, basis.data(), int(width),
                rows.data(), int(width), 0.0f, result.data(), k_rank);
    return result;
}

std::vector<float> project_column_major(const std::vector<float> & rows, size_t count, size_t width, const std::vector<float> & basis) {
    if (rows.size() != count * width || basis.size() != width * k_rank) throw std::runtime_error("projection shape mismatch");
    std::vector<float> result(count * k_rank);
    for (size_t row = 0; row < count; ++row) for (int component = 0; component < k_rank; ++component) {
        double value = 0.0;
        for (size_t column = 0; column < width; ++column) value += rows[row + count * column] * basis[column + width * component];
        result[row * k_rank + component] = float(value);
    }
    return result;
}

std::vector<float> exclude_carrier_column_major(const std::vector<float> & rows, size_t width, int excluded_carrier) {
    if (rows.size() != 512 * width || excluded_carrier < 0 || excluded_carrier >= 32) throw std::runtime_error("invalid carrier-fold input");
    std::vector<float> result(496 * width);
    for (size_t column = 0; column < width; ++column) {
        size_t output_row = 0;
        for (int carrier = 0; carrier < 32; ++carrier) {
            if (carrier == excluded_carrier) continue;
            for (int value = 0; value < 16; ++value) result[output_row++ + 496 * column] = rows[size_t(carrier * 16 + value) + 512 * column];
        }
    }
    return result;
}

struct softmax_model {
    int features = 0;
    std::vector<double> weights;
    int iterations = 0;
    double gradient_norm = INFINITY;
    bool converged = false;
};

struct cv_selection {
    double lambda = NAN;
    std::vector<double> scores;
};

struct decoder_fit_record {
    double lambda = NAN;
    int features = 0;
    int iterations = 0;
    double gradient_norm = INFINITY;
    bool converged = false;
};

struct decoder_prediction {
    int32_t carrier_id = 0;
    int32_t value_id = 0;
    int32_t predicted_value_id = 0;
    double true_probability = 0.0;
    double cross_entropy = 0.0;
    bool correct = false;
};

struct feature_pipeline {
    std::vector<float> mean;
    std::vector<float> basis;
    std::array<double, k_rank> coordinate_mean{};
    std::array<double, k_rank> coordinate_std{};
};

// Fit every unsupervised transform exclusively on the carriers available to a fold.
feature_pipeline fit_feature_pipeline(const std::vector<float> & rows, size_t width, int excluded_carrier) {
    if (rows.size() != 512 * width || excluded_carrier < -1 || excluded_carrier >= 32) throw std::runtime_error("invalid feature pipeline input");
    const size_t train_count = excluded_carrier < 0 ? 512 : 496;
    feature_pipeline result;
    result.mean.resize(width);
    std::vector<float> centered(train_count * width);
    for (size_t column = 0; column < width; ++column) {
        double sum = 0.0;
        for (int carrier = 0; carrier < 32; ++carrier) if (carrier != excluded_carrier) {
            for (int value = 0; value < 16; ++value) sum += rows[size_t(carrier * 16 + value) + 512 * column];
        }
        result.mean[column] = float(sum / train_count);
        size_t destination = 0;
        for (int carrier = 0; carrier < 32; ++carrier) if (carrier != excluded_carrier) {
            for (int value = 0; value < 16; ++value) {
                centered[destination++ + train_count * column] = rows[size_t(carrier * 16 + value) + 512 * column] - result.mean[column];
            }
        }
    }
    result.basis = fit_rank32_basis(centered, train_count, width);
    reorthogonalize(result.basis, width);
    const std::vector<float> train_coordinates = project_column_major(centered, train_count, width, result.basis);
    for (int component = 0; component < k_rank; ++component) {
        double sum = 0.0;
        for (size_t row = 0; row < train_count; ++row) sum += train_coordinates[row * k_rank + component];
        result.coordinate_mean[component] = sum / train_count;
        double squares = 0.0;
        for (size_t row = 0; row < train_count; ++row) {
            const double delta = train_coordinates[row * k_rank + component] - result.coordinate_mean[component];
            squares += delta * delta;
        }
        result.coordinate_std[component] = std::sqrt(squares / train_count);
        if (!(result.coordinate_std[component] >= 1e-12)) throw std::runtime_error("rank coordinate has near-zero train standard deviation");
    }
    return result;
}

std::vector<float> transform_feature_pipeline(const std::vector<float> & rows, size_t row_count, size_t width, const feature_pipeline & pipeline) {
    if (rows.size() != row_count * width || pipeline.mean.size() != width) throw std::runtime_error("invalid feature pipeline transform");
    std::vector<float> centered(rows);
    for (size_t column = 0; column < width; ++column) for (size_t row = 0; row < row_count; ++row) centered[row + row_count * column] -= pipeline.mean[column];
    std::vector<float> coordinates = project_column_major(centered, row_count, width, pipeline.basis);
    for (size_t row = 0; row < row_count; ++row) for (int component = 0; component < k_rank; ++component) {
        coordinates[row * k_rank + component] = float((coordinates[row * k_rank + component] - pipeline.coordinate_mean[component]) / pipeline.coordinate_std[component]);
    }
    return coordinates;
}

double softmax_objective_gradient(const std::vector<double> & weights, const std::vector<float> & features,
                                  const std::vector<int> & labels, int dimension, double lambda,
                                  std::vector<double> & gradient) {
    const int rows = int(labels.size());
    std::vector<double> inputs(size_t(rows) * dimension), coefficients(size_t(16) * dimension), logits(size_t(rows) * 16), errors(size_t(rows) * 16);
    for (size_t i = 0; i < inputs.size(); ++i) inputs[i] = features[i];
    for (int value = 0; value < 16; ++value) for (int column = 0; column < dimension; ++column) coefficients[value * dimension + column] = weights[value * (dimension + 1) + column];
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, rows, 16, dimension, 1.0, inputs.data(), dimension, coefficients.data(), dimension, 0.0, logits.data(), 16);
    double objective = 0.0;
    for (int row = 0; row < rows; ++row) {
        double maximum = -INFINITY;
        for (int value = 0; value < 16; ++value) maximum = std::max(maximum, logits[row * 16 + value] + weights[value * (dimension + 1) + dimension]);
        double normalizer = 0.0;
        for (int value = 0; value < 16; ++value) normalizer += std::exp(logits[row * 16 + value] + weights[value * (dimension + 1) + dimension] - maximum);
        objective += std::log(normalizer) - (logits[row * 16 + labels[row]] + weights[labels[row] * (dimension + 1) + dimension] - maximum);
        for (int value = 0; value < 16; ++value) errors[row * 16 + value] = std::exp(logits[row * 16 + value] + weights[value * (dimension + 1) + dimension] - maximum) / normalizer - (labels[row] == value);
    }
    std::vector<double> weight_gradient(size_t(16) * dimension);
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 16, dimension, rows, 1.0, errors.data(), 16, inputs.data(), dimension, 0.0, weight_gradient.data(), dimension);
    gradient.assign(weights.size(), 0.0);
    double penalty = 0.0;
    for (int value = 0; value < 16; ++value) for (int column = 0; column <= dimension; ++column) {
        const size_t index = size_t(value) * (dimension + 1) + column;
        gradient[index] = (column == dimension ? 0.0 : weight_gradient[value * dimension + column]) / rows + lambda * weights[index];
        if (column == dimension) for (int row = 0; row < rows; ++row) gradient[index] += errors[row * 16 + value] / rows;
        penalty += weights[index] * weights[index];
    }
    return objective / rows + 0.5 * lambda * penalty;
}

softmax_model fit_softmax(const std::vector<float> & features, const std::vector<int> & labels, int dimension, double lambda) {
    if (features.size() != labels.size() * size_t(dimension) || dimension <= 0) throw std::runtime_error("invalid softmax features");
    softmax_model result{ dimension, std::vector<double>(16 * (dimension + 1)) };
    std::vector<double> gradient, direction(result.weights.size()), candidate(result.weights.size()), candidate_gradient;
    std::vector<std::vector<double>> history_s, history_y;
    std::vector<double> history_rho;
    double objective = softmax_objective_gradient(result.weights, features, labels, dimension, lambda, gradient);
    for (int iteration = 0; iteration < 500; ++iteration) {
        double norm_squared = 0.0;
        for (double value : gradient) norm_squared += value * value;
        result.iterations = iteration + 1;
        result.gradient_norm = std::sqrt(norm_squared);
        if (result.gradient_norm <= 1e-5) { result.converged = true; break; }
        direction = gradient;
        std::vector<double> alpha(history_s.size());
        for (int i = int(history_s.size()) - 1; i >= 0; --i) {
            alpha[i] = history_rho[i] * cblas_ddot(int(direction.size()), history_s[i].data(), 1, direction.data(), 1);
            cblas_daxpy(int(direction.size()), -alpha[i], history_y[i].data(), 1, direction.data(), 1);
        }
        double scale = 1.0;
        if (!history_s.empty()) {
            const auto & s = history_s.back(); const auto & y = history_y.back();
            const double yy = cblas_ddot(int(y.size()), y.data(), 1, y.data(), 1);
            if (yy > 0.0) scale = cblas_ddot(int(s.size()), s.data(), 1, y.data(), 1) / yy;
        }
        cblas_dscal(int(direction.size()), scale, direction.data(), 1);
        for (size_t i = 0; i < history_s.size(); ++i) {
            const double beta = history_rho[i] * cblas_ddot(int(direction.size()), history_y[i].data(), 1, direction.data(), 1);
            cblas_daxpy(int(direction.size()), alpha[i] - beta, history_s[i].data(), 1, direction.data(), 1);
        }
        cblas_dscal(int(direction.size()), -1.0, direction.data(), 1);
        double directional = cblas_ddot(int(direction.size()), direction.data(), 1, gradient.data(), 1);
        if (!(directional < 0.0)) { for (size_t i = 0; i < direction.size(); ++i) direction[i] = -gradient[i]; directional = -norm_squared; }
        double step = 1.0, candidate_objective = INFINITY;
        for (int trial = 0; trial < 40; ++trial) {
            for (size_t i = 0; i < candidate.size(); ++i) candidate[i] = result.weights[i] + step * direction[i];
            candidate_objective = softmax_objective_gradient(candidate, features, labels, dimension, lambda, candidate_gradient);
            if (candidate_objective <= objective + 1e-4 * step * directional) break;
            step *= 0.5;
        }
        if (!(candidate_objective <= objective + 1e-4 * step * directional)) throw std::runtime_error("L-BFGS line search failed");
        std::vector<double> s(result.weights.size()), y(result.weights.size());
        for (size_t i = 0; i < s.size(); ++i) { s[i] = candidate[i] - result.weights[i]; y[i] = candidate_gradient[i] - gradient[i]; }
        const double curvature = cblas_ddot(int(s.size()), s.data(), 1, y.data(), 1);
        if (curvature > 1e-12) {
            if (history_s.size() == 10) { history_s.erase(history_s.begin()); history_y.erase(history_y.begin()); history_rho.erase(history_rho.begin()); }
            history_s.push_back(std::move(s)); history_y.push_back(std::move(y)); history_rho.push_back(1.0 / curvature);
        }
        result.weights = candidate; gradient = candidate_gradient; objective = candidate_objective;
    }
    return result;
}

double cross_entropy(const softmax_model & model, const std::vector<float> & features, const std::vector<int> & labels) {
    if (features.size() != labels.size() * size_t(model.features)) throw std::runtime_error("invalid evaluation features");
    double total = 0.0;
    for (size_t row = 0; row < labels.size(); ++row) {
        double maximum = -INFINITY;
        std::vector<double> logits(16);
        for (int value = 0; value < 16; ++value) {
            double score = model.weights[value * (model.features + 1) + model.features];
            for (int column = 0; column < model.features; ++column) score += model.weights[value * (model.features + 1) + column] * features[row * model.features + column];
            logits[value] = score; maximum = std::max(maximum, score);
        }
        double normalizer = 0.0;
        for (double value : logits) normalizer += std::exp(value - maximum);
        total += std::log(normalizer) - (logits[labels[row]] - maximum);
    }
    return total / labels.size();
}

std::vector<decoder_prediction> predict_rows(const softmax_model & model, const std::vector<float> & features,
                                             const std::vector<int> & labels) {
    if (features.size() != labels.size() * size_t(model.features)) throw std::runtime_error("invalid prediction features");
    std::vector<decoder_prediction> result;
    result.reserve(labels.size());
    for (size_t row = 0; row < labels.size(); ++row) {
        std::array<double, 16> logits{};
        double maximum = -INFINITY;
        int predicted = 0;
        for (int value = 0; value < 16; ++value) {
            double score = model.weights[value * (model.features + 1) + model.features];
            for (int column = 0; column < model.features; ++column) score += model.weights[value * (model.features + 1) + column] * features[row * model.features + column];
            logits[value] = score;
            if (score > maximum) { maximum = score; predicted = value; }
        }
        double normalizer = 0.0;
        for (double score : logits) normalizer += std::exp(score - maximum);
        const double probability = std::exp(logits[labels[row]] - maximum) / normalizer;
        result.push_back({ int32_t(row / 16 + 32), int32_t(row % 16), int32_t(predicted), probability, -std::log(probability), predicted == labels[row] });
    }
    return result;
}

std::vector<double> carrier_cross_entropy(const softmax_model & model, const std::vector<float> & features,
                                          const std::vector<int> & labels) {
    if (labels.size() != 256) throw std::runtime_error("held-out carrier factorial must have 256 rows");
    std::vector<double> result(16);
    for (int carrier = 0; carrier < 16; ++carrier) {
        std::vector<float> rows(features.begin() + size_t(carrier) * 16 * model.features,
                                features.begin() + size_t(carrier + 1) * 16 * model.features);
        std::vector<int> values(labels.begin() + carrier * 16, labels.begin() + (carrier + 1) * 16);
        result[carrier] = cross_entropy(model, rows, values);
    }
    return result;
}

double sign_flip_p_value(const std::vector<double> & deltas) {
    if (deltas.size() != 16) throw std::runtime_error("sign-flip test requires 16 held-out carriers");
    double observed = 0.0;
    for (double value : deltas) observed += value / 16.0;
    uint64_t exceedances = 0;
    for (uint32_t mask = 0; mask < (1u << 16); ++mask) {
        double statistic = 0.0;
        for (int carrier = 0; carrier < 16; ++carrier) statistic += ((mask >> carrier) & 1 ? 1.0 : -1.0) * deltas[carrier] / 16.0;
        if (statistic >= observed) ++exceedances;
    }
    return double(exceedances) / double(1u << 16);
}

std::vector<double> holm_adjust(std::vector<double> p_values) {
    std::vector<size_t> order(p_values.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return p_values[a] < p_values[b]; });
    std::vector<double> adjusted(p_values.size());
    double previous = 0.0;
    for (size_t rank = 0; rank < order.size(); ++rank) {
        const double value = std::min(1.0, p_values[order[rank]] * (order.size() - rank));
        adjusted[order[rank]] = std::max(previous, value);
        previous = adjusted[order[rank]];
    }
    return adjusted;
}

std::string mechanical_outcome(bool ambient, bool transported, bool additive, bool endpoint) {
    if (ambient && transported) return "relative_coordinate_advantage";
    if (ambient || transported) return "coordinate_dependent_relative_signal";
    if (additive) return "additive_coordinates_match_or_win";
    if (endpoint) return "holistic_endpoint_only";
    if (!ambient && !transported) return "no_detected_layer_local_increment";
    return "ambiguous";
}

struct paired_t_result { double statistic; double two_sided_p; };

paired_t_result paired_t_test(const std::vector<double> & deltas) {
    if (deltas.size() < 2) throw std::runtime_error("paired t-test requires at least two carriers");
    double mean = 0.0;
    for (double value : deltas) { if (!std::isfinite(value)) throw std::runtime_error("non-finite paired t-test input"); mean += value; }
    mean /= deltas.size();
    double sum_sq = 0.0;
    for (double value : deltas) sum_sq += (value - mean) * (value - mean);
    const double standard_error = std::sqrt(sum_sq / (deltas.size() - 1) / deltas.size());
    if (!(standard_error > 0.0)) throw std::runtime_error("degenerate paired t-test input");
    const double statistic = mean / standard_error;
    const boost::math::students_t_distribution<double> distribution(double(deltas.size() - 1));
    return { statistic, 2.0 * boost::math::cdf(boost::math::complement(distribution, std::abs(statistic))) };
}

cv_selection select_lambda_carrier_folds(const std::vector<float> & features, const std::vector<int> & labels, int dimension) {
    static constexpr double grid[] = { 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0 };
    cv_selection result;
    double best_score = INFINITY;
    for (double lambda : grid) {
        double score = 0.0;
        for (int held_out = 0; held_out < 32; ++held_out) {
            std::vector<float> train;
            std::vector<int> train_labels, validation_labels;
            std::vector<float> validation;
            for (int carrier = 0; carrier < 32; ++carrier) for (int value = 0; value < 16; ++value) {
                const size_t row = size_t(carrier) * 16 + value;
                if (carrier == held_out) {
                    validation.insert(validation.end(), features.begin() + row * dimension, features.begin() + (row + 1) * dimension);
                    validation_labels.push_back(labels[row]);
                } else {
                    train.insert(train.end(), features.begin() + row * dimension, features.begin() + (row + 1) * dimension);
                    train_labels.push_back(labels[row]);
                }
            }
            score += cross_entropy(fit_softmax(train, train_labels, dimension, lambda), validation, validation_labels) / 32.0;
        }
        result.scores.push_back(score);
        if (score < best_score || (score == best_score && lambda > result.lambda)) { best_score = score; result.lambda = lambda; }
    }
    return result;
}

std::vector<float> concatenate_features(const std::vector<float> & left, const std::vector<float> & right, size_t rows) {
    if (left.size() != rows * k_rank || right.size() != rows * k_rank) throw std::runtime_error("feature concatenation shape mismatch");
    std::vector<float> result(rows * 2 * k_rank);
    for (size_t row = 0; row < rows; ++row) {
        std::copy_n(left.begin() + row * k_rank, k_rank, result.begin() + row * 2 * k_rank);
        std::copy_n(right.begin() + row * k_rank, k_rank, result.begin() + row * 2 * k_rank + k_rank);
    }
    return result;
}

struct carrier_fold_features {
    std::vector<float> train;
    std::vector<float> validation;
    std::vector<int> train_labels;
    std::vector<int> validation_labels;
};

struct local_coordinate_cache {
    // One standardized 512-by-rank coordinate matrix per held-out train carrier.
    std::array<std::vector<float>, 32> coordinates;
};

local_coordinate_cache build_local_coordinate_cache(const std::vector<float> & representation, size_t width) {
    local_coordinate_cache result;
#pragma omp parallel for schedule(dynamic)
    for (int held_out = 0; held_out < 32; ++held_out) {
        const feature_pipeline pipeline = fit_feature_pipeline(representation, width, held_out);
        result.coordinates[held_out] = transform_feature_pipeline(representation, 512, width, pipeline);
    }
    return result;
}

local_coordinate_cache build_transported_local_coordinate_cache(const std::vector<float> & incoming, const std::vector<float> & ambient, size_t width) {
    if (incoming.size() != 512 * width || ambient.size() != 512 * width) throw std::runtime_error("invalid transport-CV input");
    local_coordinate_cache result;
#pragma omp parallel for schedule(dynamic)
    for (int held_out = 0; held_out < 32; ++held_out) {
        std::vector<double> reference(width);
        for (int carrier = 0; carrier < 32; ++carrier) if (carrier != held_out) {
            for (int value = 0; value < 16; ++value) {
                const size_t row = size_t(carrier * 16 + value);
                for (size_t column = 0; column < width; ++column) reference[column] += incoming[row + 512 * column] / 496.0;
            }
        }
        double norm_squared = 0.0;
        for (double value : reference) norm_squared += value * value;
        if (!(norm_squared > 1e-24)) throw std::runtime_error("degenerate fold-local transport reference");
        for (double & value : reference) value /= std::sqrt(norm_squared);
        std::vector<float> transported(512 * width);
        for (size_t row = 0; row < 512; ++row) {
            double reference_dot = 0.0, tangent_dot = 0.0;
            for (size_t column = 0; column < width; ++column) {
                reference_dot += incoming[row + 512 * column] * reference[column];
                tangent_dot += ambient[row + 512 * column] * reference[column];
            }
            if (!(1.0 + reference_dot > 1e-12)) throw std::runtime_error("near-antipodal fold-local transport denominator");
            for (size_t column = 0; column < width; ++column) {
                transported[row + 512 * column] = float(ambient[row + 512 * column] - tangent_dot * (incoming[row + 512 * column] + reference[column]) / (1.0 + reference_dot));
            }
        }
        const feature_pipeline pipeline = fit_feature_pipeline(transported, width, held_out);
        result.coordinates[held_out] = transform_feature_pipeline(transported, 512, width, pipeline);
    }
    return result;
}

cv_selection select_lambda_carrier_folds_cached(const std::vector<const local_coordinate_cache *> & representations, const char * label) {
    if (representations.empty()) throw std::runtime_error("empty cached-CV representation set");
    std::vector<carrier_fold_features> folds;
    folds.reserve(32);
    const int dimension = int(representations.size()) * k_rank;
    for (int held_out = 0; held_out < 32; ++held_out) {
        carrier_fold_features fold;
        fold.train.reserve(496 * dimension);
        fold.validation.reserve(16 * dimension);
        for (int carrier = 0; carrier < 32; ++carrier) for (int value = 0; value < 16; ++value) {
            const size_t row = size_t(carrier * 16 + value);
            std::vector<float> & destination = carrier == held_out ? fold.validation : fold.train;
            for (const auto * representation : representations) {
                const std::vector<float> & coordinates = representation->coordinates[held_out];
                destination.insert(destination.end(), coordinates.begin() + row * k_rank, coordinates.begin() + (row + 1) * k_rank);
            }
            (carrier == held_out ? fold.validation_labels : fold.train_labels).push_back(value);
        }
        folds.push_back(std::move(fold));
    }
    static constexpr double grid[] = { 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0 };
    cv_selection result;
    double best_score = INFINITY;
    for (double lambda : grid) {
        double score = 0.0;
        #pragma omp parallel for reduction(+ : score) schedule(dynamic)
        for (int fold_index = 0; fold_index < int(folds.size()); ++fold_index) {
            const carrier_fold_features & fold = folds[fold_index];
            score += cross_entropy(fit_softmax(fold.train, fold.train_labels, dimension, lambda), fold.validation, fold.validation_labels) / 32.0;
        }
        result.scores.push_back(score);
        if (score < best_score || (score == best_score && lambda > result.lambda)) { best_score = score; result.lambda = lambda; }
        std::printf("progress=cv model=%s lambda=%g mean_carrier_ce=%g\n", label, lambda, score);
        std::fflush(stdout);
    }
    return result;
}

cv_selection select_lambda_carrier_folds_local(const std::vector<const std::vector<float> *> & representations, size_t width) {
    if (representations.empty()) throw std::runtime_error("empty local-CV representation set");
    std::vector<carrier_fold_features> folds;
    folds.reserve(32);
    const int dimension = int(representations.size()) * k_rank;
    for (int held_out = 0; held_out < 32; ++held_out) {
        std::vector<std::vector<float>> coordinates;
        coordinates.reserve(representations.size());
        for (const auto * representation : representations) {
            const feature_pipeline pipeline = fit_feature_pipeline(*representation, width, held_out);
            coordinates.push_back(transform_feature_pipeline(*representation, 512, width, pipeline));
        }
        carrier_fold_features fold;
        fold.train.reserve(496 * dimension);
        fold.validation.reserve(16 * dimension);
        for (int carrier = 0; carrier < 32; ++carrier) for (int value = 0; value < 16; ++value) {
            const size_t row = size_t(carrier * 16 + value);
            std::vector<float> & destination = carrier == held_out ? fold.validation : fold.train;
            for (const auto & representation_coordinates : coordinates) {
                destination.insert(destination.end(), representation_coordinates.begin() + row * k_rank,
                                   representation_coordinates.begin() + (row + 1) * k_rank);
            }
            (carrier == held_out ? fold.validation_labels : fold.train_labels).push_back(value);
        }
        folds.push_back(std::move(fold));
    }
    static constexpr double grid[] = { 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0 };
    cv_selection result;
    double best_score = INFINITY;
    for (double lambda : grid) {
        double score = 0.0;
        for (const carrier_fold_features & fold : folds) {
            score += cross_entropy(fit_softmax(fold.train, fold.train_labels, dimension, lambda), fold.validation, fold.validation_labels) / 32.0;
        }
        result.scores.push_back(score);
        if (score < best_score || (score == best_score && lambda > result.lambda)) { best_score = score; result.lambda = lambda; }
    }
    return result;
}

} // namespace

int main(int argc, char ** argv) try {
    std::string input, output_json, output_root, registration;
    bool development = false, self_test = false;
    uint64_t seed = 0;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--input-root") && i + 1 < argc) input = argv[++i];
        else if (!std::strcmp(argv[i], "--output-json") && i + 1 < argc) output_json = argv[++i];
        else if (!std::strcmp(argv[i], "--output-root") && i + 1 < argc) output_root = argv[++i];
        else if (!std::strcmp(argv[i], "--registration") && i + 1 < argc) registration = argv[++i];
        else if (!std::strcmp(argv[i], "--seed") && i + 1 < argc) seed = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "--development-run")) development = true;
        else if (!std::strcmp(argv[i], "--self-test")) self_test = true;
        else throw std::runtime_error("usage: llama-rwkv-tmix-controlled-value-decode --input-root CAPTURE.root --registration REGISTRATION.json --output-root RESULT.root --output-json RESULT.json --seed 941731 | --development-run [--self-test]");
    }
    if (self_test) {
        openblas_set_num_threads(1);
        omp_set_num_threads(1);
        if (std::abs(sign_flip_p_value(std::vector<double>(16, 1.0)) - 1.0 / 65536.0) > 1e-15) throw std::runtime_error("sign-flip self-test failed");
        const std::vector<double> holm = holm_adjust({ 0.01, 0.02, 0.5 });
        if (holm[0] != 0.03 || holm[1] != 0.04 || holm[2] != 0.5) throw std::runtime_error("Holm self-test failed");
        // A held-out carrier must change a fold-local transform; otherwise preprocessing leaks.
        std::vector<float> synthetic(512 * k_rank);
        for (size_t row = 0; row < 512; ++row) for (int column = 0; column < k_rank; ++column) synthetic[row + 512 * column] = float(row % k_rank == size_t(column) ? 1000 : 0);
        const feature_pipeline first_fold = fit_feature_pipeline(synthetic, k_rank, 0);
        const feature_pipeline second_fold = fit_feature_pipeline(synthetic, k_rank, 1);
        if (!(std::abs(first_fold.mean[0] - second_fold.mean[0]) > 1.0)) throw std::runtime_error("fold-local preprocessing leakage self-test failed");
        std::vector<float> separable(16 * 16 * 16, 0.0f);
        std::vector<int> separable_labels;
        for (int label = 0; label < 16; ++label) for (int replicate = 0; replicate < 16; ++replicate) {
            separable[(label * 16 + replicate) * 16 + label] = 1.0f;
            separable_labels.push_back(label);
        }
        const softmax_model known_solution = fit_softmax(separable, separable_labels, 16, 10.0);
        if (!known_solution.converged || known_solution.gradient_norm > 1e-5 ||
            cross_entropy(known_solution, separable, separable_labels) >= std::log(16.0)) {
            throw std::runtime_error("softmax convergence self-test failed");
        }
        if (mechanical_outcome(true, true, true, false) != "relative_coordinate_advantage" ||
            mechanical_outcome(true, false, true, false) != "coordinate_dependent_relative_signal" ||
            mechanical_outcome(false, false, true, false) != "additive_coordinates_match_or_win" ||
            mechanical_outcome(false, false, false, true) != "holistic_endpoint_only" ||
            mechanical_outcome(false, false, false, false) != "no_detected_layer_local_increment") {
            throw std::runtime_error("mechanical outcome truth-table self-test failed");
        }
        std::puts("self-tests=passed");
        return 0;
    }
    if (input.empty() || output_json.empty() || std::filesystem::exists(output_json) ||
        (development && (!registration.empty() || !output_root.empty())) ||
        (!development && (registration.empty() || output_root.empty() || seed != k_seed || std::filesystem::exists(output_root)))) {
        throw std::runtime_error("development input/new JSON or classified input/registration/new ROOT/new JSON/--seed 941731 are required");
    }
    if (!development) verify_classified_capture(input, registration);
    const std::vector<behavior_summary> behavior_summaries = development ? std::vector<behavior_summary>{} : summarize_behavior(input);
    // CV parallelism is at the independent carrier-fold level; prevent nested BLAS threads.
    openblas_set_num_threads(1);
    std::printf("progress=parallel_workers workers=%d openblas_threads=1\n", omp_get_max_threads());
    std::fflush(stdout);
    auto reader = rwkv_activation_store::activation_dataset_reader::open(input);
    if (reader.entries() != 768 || reader.sources().size() != 12 || reader.dimension() == 0) {
        throw std::runtime_error("capture schema does not match the controlled-value development factorial");
    }
    uint64_t rows = 0;
    // The complete capture is only 9 MiB raw. Retain it to avoid repeated ROOT scans.
    std::vector<std::vector<float>> captured_sources(reader.sources().size());
    for (size_t source = 0; source < reader.sources().size(); ++source) {
        reader.for_each_source_batch(source, 16, [&](rwkv_activation_store::source_batch && batch) {
            if (batch.dimension != reader.dimension()) throw std::runtime_error("inconsistent activation width");
            for (float value : batch.values) if (!std::isfinite(value)) throw std::runtime_error("non-finite activation");
            if (source == 0) rows += batch.metadata.size();
            captured_sources[source].insert(captured_sources[source].end(), batch.values.begin(), batch.values.end());
        });
        if (captured_sources[source].size() != 768 * reader.dimension()) throw std::runtime_error("missing source activation rows");
    }
    if (rows != 768) throw std::runtime_error("missing activation rows");
    std::printf("progress=validated_capture rows=%llu sources=%zu width=%zu\n", (unsigned long long) rows, reader.sources().size(), reader.dimension());
    std::fflush(stdout);
    double max_identity_error = 0.0;
    uint64_t geometry_rows = 0;
    double max_angle = 0.0;
    std::vector<double> reference_norms;
    uint64_t transport_rows = 0;
    double max_basis_orthogonality_error = 0.0;
    double raw_decoder_weight_norm = 0.0;
    double raw_held_out_cross_entropy = 0.0;
    double incoming_held_out_cross_entropy = 0.0;
    double ambient_held_out_cross_entropy = 0.0, transported_held_out_cross_entropy = 0.0;
    double endpoint_held_out_cross_entropy = 0.0;
    double additive_held_out_cross_entropy = 0.0;
    double relative_ambient_held_out_cross_entropy = 0.0, relative_transport_held_out_cross_entropy = 0.0;
    std::array<double, 4> additive_gain_p{}, ambient_gain_p{}, transport_gain_p{}, endpoint_gain_p{}, ambient_advantage_p{}, transport_advantage_p{};
    std::array<double, 4> additive_gain_holm{}, ambient_gain_holm{}, transport_gain_holm{}, endpoint_gain_holm{}, ambient_advantage_holm{}, transport_advantage_holm{};
    std::array<std::array<double, 16>, 4> additive_gains{}, ambient_gains{}, transport_gains{}, endpoint_gains{}, ambient_advantages{}, transport_advantages{};
    std::array<paired_t_result, 4> additive_t{}, ambient_gain_t{}, transport_gain_t{}, endpoint_t{}, ambient_advantage_t{}, transport_advantage_t{};
    std::array<std::array<cv_selection, 5>, 4> cv_results;
    std::array<std::array<decoder_fit_record, 5>, 4> decoder_fits;
    std::array<std::array<std::vector<decoder_prediction>, 5>, 4> predictions;
    std::array<std::array<double, 5>, 4> held_out_ce{};
    std::array<double, 4> layer_basis_error{};
    for (size_t layer = 0; layer < 4; ++layer) {
        const int layer_number = layer == 0 ? 15 : layer == 1 ? 30 : layer == 2 ? 45 : 60;
        std::printf("progress=layer_start layer=%d index=%zu/4\n", layer_number, layer + 1);
        std::fflush(stdout);
        const std::vector<float> & resid_in = captured_sources[layer * 3];
        const std::vector<float> & time_out = captured_sources[layer * 3 + 1];
        const std::vector<float> & resid_time = captured_sources[layer * 3 + 2];
        if (resid_in.size() != time_out.size() || resid_in.size() != resid_time.size()) throw std::runtime_error("inconsistent layer row shape");
        for (size_t i = 0; i < resid_in.size(); ++i) {
            max_identity_error = std::max(max_identity_error,
                std::abs(double(resid_in[i]) + time_out[i] - resid_time[i]));
        }
        if (max_identity_error > 1e-3) throw std::runtime_error("residual identity error exceeds 1e-3");
        const size_t width = reader.dimension();
        std::vector<double> reference(width);
        for (size_t row = 0; row < 512; ++row) {
            double mean_x = 0.0, xx = 0.0;
            for (size_t j = 0; j < width; ++j) mean_x += resid_in[row * width + j];
            mean_x /= width;
            for (size_t j = 0; j < width; ++j) {
                const double x = resid_in[row * width + j] - mean_x;
                xx += x * x;
            }
            if (!(xx > 1e-24)) throw std::runtime_error("degenerate train incoming residual");
            const double scale = 1.0 / std::sqrt(xx);
            for (size_t j = 0; j < width; ++j) reference[j] += (resid_in[row * width + j] - mean_x) * scale / 512.0;
        }
        double reference_norm = 0.0;
        for (double value : reference) reference_norm += value * value;
        reference_norm = std::sqrt(reference_norm);
        if (!(reference_norm > 1e-12)) throw std::runtime_error("degenerate train-only transport reference");
        reference_norms.push_back(reference_norm);
        for (double & value : reference) value /= reference_norm;
        std::vector<float> raw_update_train(512 * width);
        std::vector<float> incoming_train(512 * width);
        std::vector<float> ambient_train(512 * width), transported_train(512 * width), endpoint_train(512 * width);
        for (size_t row = 0; row < 512; ++row) {
            double mean_x = 0.0, mean_w = 0.0, xx = 0.0;
            for (size_t j = 0; j < width; ++j) {
                mean_x += resid_in[row * width + j];
                mean_w += time_out[row * width + j];
            }
            mean_x /= width; mean_w /= width;
            for (size_t j = 0; j < width; ++j) {
                const double x = resid_in[row * width + j] - mean_x;
                xx += x * x;
            }
            const double radius = std::sqrt(xx);
            if (!(radius > 1e-12)) throw std::runtime_error("degenerate train raw-update scale");
            for (size_t j = 0; j < width; ++j) {
                raw_update_train[row + 512 * j] = float((time_out[row * width + j] - mean_w) / radius);
                incoming_train[row + 512 * j] = float((resid_in[row * width + j] - mean_x) / radius);
            }
            double mean_y = 0.0, yy = 0.0, xy = 0.0;
            for (size_t j = 0; j < width; ++j) mean_y += resid_time[row * width + j];
            mean_y /= width;
            for (size_t j = 0; j < width; ++j) {
                const double x = (resid_in[row * width + j] - mean_x) / radius;
                const double y = resid_time[row * width + j] - mean_y;
                yy += y * y; xy += x * y;
            }
            const double cosine = std::clamp(xy / std::sqrt(yy), -1.0, 1.0);
            const double theta = std::acos(cosine), sine = std::sqrt(std::max(1e-24, 1.0 - cosine * cosine));
            double tangent_reference_dot = 0.0;
            double incoming_reference_dot = 0.0;
            for (size_t j = 0; j < width; ++j) {
                const double x = (resid_in[row * width + j] - mean_x) / radius;
                const double y = (resid_time[row * width + j] - mean_y) / std::sqrt(yy);
                tangent_reference_dot += ((y - cosine * x) / sine * theta) * reference[j];
                incoming_reference_dot += x * reference[j];
            }
            if (!(1.0 + incoming_reference_dot > 1e-12)) throw std::runtime_error("near-antipodal train transport denominator");
            for (size_t j = 0; j < width; ++j) {
                const double x = (resid_in[row * width + j] - mean_x) / radius;
                const double y = (resid_time[row * width + j] - mean_y) / std::sqrt(yy);
                const double tangent = (y - cosine * x) / sine * theta;
                ambient_train[row + 512 * j] = float(tangent);
                transported_train[row + 512 * j] = float(tangent - tangent_reference_dot * (x + reference[j]) / (1.0 + incoming_reference_dot));
                endpoint_train[row + 512 * j] = float(y);
            }
        }
        const feature_pipeline raw_pipeline = fit_feature_pipeline(raw_update_train, width, -1);
        const feature_pipeline incoming_pipeline = fit_feature_pipeline(incoming_train, width, -1);
        const feature_pipeline ambient_pipeline = fit_feature_pipeline(ambient_train, width, -1);
        const feature_pipeline transported_pipeline = fit_feature_pipeline(transported_train, width, -1);
        const feature_pipeline endpoint_pipeline = fit_feature_pipeline(endpoint_train, width, -1);
        // Reuse these exact fold-local transforms across all registered decoder models.
        const local_coordinate_cache raw_cv_cache = build_local_coordinate_cache(raw_update_train, width);
        const local_coordinate_cache incoming_cv_cache = build_local_coordinate_cache(incoming_train, width);
        const local_coordinate_cache ambient_cv_cache = build_local_coordinate_cache(ambient_train, width);
        const local_coordinate_cache transported_cv_cache = build_transported_local_coordinate_cache(incoming_train, ambient_train, width);
        const local_coordinate_cache endpoint_cv_cache = build_local_coordinate_cache(endpoint_train, width);
        std::printf("progress=fold_preprocessing_complete layer=%d folds=32 representations=5\n", layer_number);
        std::fflush(stdout);
        const std::vector<float> & raw_basis = raw_pipeline.basis;
        const std::vector<float> & incoming_basis = incoming_pipeline.basis;
        const std::vector<float> & ambient_basis = ambient_pipeline.basis;
        const std::vector<float> & transported_basis = transported_pipeline.basis;
        const std::vector<float> & endpoint_basis = endpoint_pipeline.basis;
        const std::vector<float> raw_train_coordinates = transform_feature_pipeline(raw_update_train, 512, width, raw_pipeline);
        std::vector<int> train_labels(512);
        for (size_t row = 0; row < train_labels.size(); ++row) train_labels[row] = int(row % 16);
        // Standalone update/tangent decoders are development diagnostics, not registered models.
        const cv_selection raw_cv{ 1e-5, {} };
        const softmax_model raw_decoder = fit_softmax(raw_train_coordinates, train_labels, k_rank, raw_cv.lambda);
        for (double weight : raw_decoder.weights) raw_decoder_weight_norm += weight * weight;
        std::vector<float> raw_update_test(256 * width);
        for (size_t row = 0; row < 256; ++row) {
            const size_t source_row = row + 512;
            double mean_x = 0.0, mean_w = 0.0, xx = 0.0;
            for (size_t j = 0; j < width; ++j) { mean_x += resid_in[source_row * width + j]; mean_w += time_out[source_row * width + j]; }
            mean_x /= width; mean_w /= width;
            for (size_t j = 0; j < width; ++j) { const double x = resid_in[source_row * width + j] - mean_x; xx += x * x; }
            const double radius = std::sqrt(xx);
            if (!(radius > 1e-12)) throw std::runtime_error("degenerate held-out raw-update scale");
            for (size_t j = 0; j < width; ++j) raw_update_test[row + 256 * j] = float((time_out[source_row * width + j] - mean_w) / radius);
        }
        const std::vector<float> raw_test_coordinates = transform_feature_pipeline(raw_update_test, 256, width, raw_pipeline);
        std::vector<int> test_labels(256);
        for (size_t row = 0; row < test_labels.size(); ++row) test_labels[row] = int(row % 16);
        raw_held_out_cross_entropy += cross_entropy(raw_decoder, raw_test_coordinates, test_labels);
        const std::vector<float> incoming_train_coordinates = transform_feature_pipeline(incoming_train, 512, width, incoming_pipeline);
        const cv_selection incoming_cv = select_lambda_carrier_folds_cached({ &incoming_cv_cache }, "incoming");
        const softmax_model incoming_decoder = fit_softmax(incoming_train_coordinates, train_labels, k_rank, incoming_cv.lambda);
        std::vector<float> incoming_test(256 * width);
        for (size_t row = 0; row < 256; ++row) {
            const size_t source_row = row + 512;
            double mean_x = 0.0, xx = 0.0;
            for (size_t j = 0; j < width; ++j) mean_x += resid_in[source_row * width + j];
            mean_x /= width;
            for (size_t j = 0; j < width; ++j) { const double x = resid_in[source_row * width + j] - mean_x; xx += x * x; }
            const double radius = std::sqrt(xx);
            for (size_t j = 0; j < width; ++j) incoming_test[row + 256 * j] = float((resid_in[source_row * width + j] - mean_x) / radius);
        }
        const std::vector<float> incoming_test_coordinates = transform_feature_pipeline(incoming_test, 256, width, incoming_pipeline);
        incoming_held_out_cross_entropy += cross_entropy(incoming_decoder, incoming_test_coordinates, test_labels);
        std::vector<float> ambient_test(256 * width), transported_test(256 * width);
        for (size_t row = 0; row < 256; ++row) {
            const size_t source_row = row + 512;
            double mean_x = 0.0, mean_y = 0.0, xx = 0.0, yy = 0.0, xy = 0.0;
            for (size_t j = 0; j < width; ++j) { mean_x += resid_in[source_row * width + j]; mean_y += resid_time[source_row * width + j]; }
            mean_x /= width; mean_y /= width;
            for (size_t j = 0; j < width; ++j) {
                const double x = resid_in[source_row * width + j] - mean_x;
                const double y = resid_time[source_row * width + j] - mean_y;
                xx += x * x; yy += y * y; xy += x * y;
            }
            const double rx = std::sqrt(xx), ry = std::sqrt(yy), cosine = std::clamp(xy / (rx * ry), -1.0, 1.0);
            const double theta = std::acos(cosine), sine = std::sqrt(std::max(1e-24, 1.0 - cosine * cosine));
            double dot_reference = 0.0;
            double incoming_reference_dot = 0.0;
            for (size_t j = 0; j < width; ++j) {
                const double x = (resid_in[source_row * width + j] - mean_x) / rx;
                const double y = (resid_time[source_row * width + j] - mean_y) / ry;
                dot_reference += ((y - cosine * x) / sine * theta) * reference[j];
                incoming_reference_dot += x * reference[j];
            }
            if (!(1.0 + incoming_reference_dot > 1e-12)) throw std::runtime_error("near-antipodal held-out transport denominator");
            for (size_t j = 0; j < width; ++j) {
                const double x = (resid_in[source_row * width + j] - mean_x) / rx;
                const double y = (resid_time[source_row * width + j] - mean_y) / ry;
                const double tangent = (y - cosine * x) / sine * theta;
                ambient_test[row + 256 * j] = float(tangent);
                transported_test[row + 256 * j] = float(tangent - dot_reference * (x + reference[j]) / (1.0 + incoming_reference_dot));
            }
        }
        const std::vector<float> ambient_train_coordinates = transform_feature_pipeline(ambient_train, 512, width, ambient_pipeline);
        const std::vector<float> transported_train_coordinates = transform_feature_pipeline(transported_train, 512, width, transported_pipeline);
        const std::vector<float> ambient_test_coordinates = transform_feature_pipeline(ambient_test, 256, width, ambient_pipeline);
        const std::vector<float> transported_test_coordinates = transform_feature_pipeline(transported_test, 256, width, transported_pipeline);
        const cv_selection ambient_cv = select_lambda_carrier_folds_cached({ &ambient_cv_cache }, "ambient");
        const cv_selection transported_cv = select_lambda_carrier_folds_cached({ &transported_cv_cache }, "transported");
        const softmax_model ambient_decoder = fit_softmax(ambient_train_coordinates, train_labels, k_rank, ambient_cv.lambda);
        const softmax_model transported_decoder = fit_softmax(transported_train_coordinates, train_labels, k_rank, transported_cv.lambda);
        ambient_held_out_cross_entropy += cross_entropy(ambient_decoder, ambient_test_coordinates, test_labels);
        transported_held_out_cross_entropy += cross_entropy(transported_decoder, transported_test_coordinates, test_labels);
        std::vector<float> endpoint_test(256 * width);
        for (size_t row = 0; row < 256; ++row) {
            const size_t source_row = row + 512;
            double mean_y = 0.0, yy = 0.0;
            for (size_t j = 0; j < width; ++j) mean_y += resid_time[source_row * width + j];
            mean_y /= width;
            for (size_t j = 0; j < width; ++j) { const double y = resid_time[source_row * width + j] - mean_y; yy += y * y; }
            const double radius = std::sqrt(yy);
            for (size_t j = 0; j < width; ++j) endpoint_test[row + 256 * j] = float((resid_time[source_row * width + j] - mean_y) / radius);
        }
        const std::vector<float> endpoint_train_coordinates = transform_feature_pipeline(endpoint_train, 512, width, endpoint_pipeline);
        const std::vector<float> endpoint_test_coordinates = transform_feature_pipeline(endpoint_test, 256, width, endpoint_pipeline);
        const cv_selection endpoint_cv = select_lambda_carrier_folds_cached({ &endpoint_cv_cache }, "endpoint");
        const softmax_model endpoint_decoder = fit_softmax(endpoint_train_coordinates, train_labels, k_rank, endpoint_cv.lambda);
        endpoint_held_out_cross_entropy += cross_entropy(endpoint_decoder, endpoint_test_coordinates, test_labels);
        const std::vector<float> additive_train_features = concatenate_features(incoming_train_coordinates, raw_train_coordinates, 512);
        const cv_selection additive_cv = select_lambda_carrier_folds_cached({ &incoming_cv_cache, &raw_cv_cache }, "additive");
        const softmax_model additive_decoder = fit_softmax(additive_train_features, train_labels, 2 * k_rank, additive_cv.lambda);
        additive_held_out_cross_entropy += cross_entropy(additive_decoder,
            concatenate_features(incoming_test_coordinates, raw_test_coordinates, 256), test_labels);
        const std::vector<float> ambient_relative_train_features = concatenate_features(incoming_train_coordinates, ambient_train_coordinates, 512);
        const std::vector<float> transported_relative_train_features = concatenate_features(incoming_train_coordinates, transported_train_coordinates, 512);
        const cv_selection ambient_relative_cv = select_lambda_carrier_folds_cached({ &incoming_cv_cache, &ambient_cv_cache }, "relative_ambient");
        const cv_selection transported_relative_cv = select_lambda_carrier_folds_cached({ &incoming_cv_cache, &transported_cv_cache }, "relative_transported");
        const softmax_model ambient_relative_decoder = fit_softmax(ambient_relative_train_features, train_labels, 2 * k_rank, ambient_relative_cv.lambda);
        const softmax_model transported_relative_decoder = fit_softmax(transported_relative_train_features, train_labels, 2 * k_rank, transported_relative_cv.lambda);
        relative_ambient_held_out_cross_entropy += cross_entropy(ambient_relative_decoder,
            concatenate_features(incoming_test_coordinates, ambient_test_coordinates, 256), test_labels);
        relative_transport_held_out_cross_entropy += cross_entropy(transported_relative_decoder,
            concatenate_features(incoming_test_coordinates, transported_test_coordinates, 256), test_labels);
        held_out_ce[layer] = { cross_entropy(incoming_decoder, incoming_test_coordinates, test_labels),
                                cross_entropy(additive_decoder, concatenate_features(incoming_test_coordinates, raw_test_coordinates, 256), test_labels),
                                cross_entropy(ambient_relative_decoder, concatenate_features(incoming_test_coordinates, ambient_test_coordinates, 256), test_labels),
                                cross_entropy(transported_relative_decoder, concatenate_features(incoming_test_coordinates, transported_test_coordinates, 256), test_labels),
                                cross_entropy(endpoint_decoder, endpoint_test_coordinates, test_labels) };
        predictions[layer][0] = predict_rows(incoming_decoder, incoming_test_coordinates, test_labels);
        predictions[layer][1] = predict_rows(additive_decoder, concatenate_features(incoming_test_coordinates, raw_test_coordinates, 256), test_labels);
        predictions[layer][2] = predict_rows(ambient_relative_decoder, concatenate_features(incoming_test_coordinates, ambient_test_coordinates, 256), test_labels);
        predictions[layer][3] = predict_rows(transported_relative_decoder, concatenate_features(incoming_test_coordinates, transported_test_coordinates, 256), test_labels);
        predictions[layer][4] = predict_rows(endpoint_decoder, endpoint_test_coordinates, test_labels);
        cv_results[layer] = { incoming_cv, additive_cv, ambient_relative_cv, transported_relative_cv, endpoint_cv };
        const auto record = [](const softmax_model & model, double lambda) {
            return decoder_fit_record{ lambda, model.features, model.iterations, model.gradient_norm, model.converged };
        };
        decoder_fits[layer] = { record(incoming_decoder, incoming_cv.lambda), record(additive_decoder, additive_cv.lambda),
                                record(ambient_relative_decoder, ambient_relative_cv.lambda), record(transported_relative_decoder, transported_relative_cv.lambda),
                                record(endpoint_decoder, endpoint_cv.lambda) };
        const std::vector<double> incoming_carrier_ce = carrier_cross_entropy(incoming_decoder, incoming_test_coordinates, test_labels);
        const std::vector<double> additive_carrier_ce = carrier_cross_entropy(additive_decoder,
            concatenate_features(incoming_test_coordinates, raw_test_coordinates, 256), test_labels);
        const std::vector<double> ambient_carrier_ce = carrier_cross_entropy(ambient_relative_decoder,
            concatenate_features(incoming_test_coordinates, ambient_test_coordinates, 256), test_labels);
        const std::vector<double> transport_carrier_ce = carrier_cross_entropy(transported_relative_decoder,
            concatenate_features(incoming_test_coordinates, transported_test_coordinates, 256), test_labels);
        const std::vector<double> endpoint_carrier_ce = carrier_cross_entropy(endpoint_decoder, endpoint_test_coordinates, test_labels);
        std::vector<double> additive_gain(16), ambient_gain(16), transport_gain(16), endpoint_gain(16), ambient_advantage(16), transport_advantage(16);
        for (int carrier = 0; carrier < 16; ++carrier) {
            additive_gain[carrier] = incoming_carrier_ce[carrier] - additive_carrier_ce[carrier];
            ambient_gain[carrier] = incoming_carrier_ce[carrier] - ambient_carrier_ce[carrier];
            transport_gain[carrier] = incoming_carrier_ce[carrier] - transport_carrier_ce[carrier];
            endpoint_gain[carrier] = incoming_carrier_ce[carrier] - endpoint_carrier_ce[carrier];
            ambient_advantage[carrier] = additive_carrier_ce[carrier] - ambient_carrier_ce[carrier];
            transport_advantage[carrier] = additive_carrier_ce[carrier] - transport_carrier_ce[carrier];
            additive_gains[layer][carrier] = additive_gain[carrier];
            ambient_gains[layer][carrier] = ambient_gain[carrier];
            transport_gains[layer][carrier] = transport_gain[carrier];
            endpoint_gains[layer][carrier] = endpoint_gain[carrier];
            ambient_advantages[layer][carrier] = ambient_advantage[carrier];
            transport_advantages[layer][carrier] = transport_advantage[carrier];
        }
        additive_gain_p[layer] = sign_flip_p_value(additive_gain);
        ambient_gain_p[layer] = sign_flip_p_value(ambient_gain);
        transport_gain_p[layer] = sign_flip_p_value(transport_gain);
        endpoint_gain_p[layer] = sign_flip_p_value(endpoint_gain);
        ambient_advantage_p[layer] = sign_flip_p_value(ambient_advantage);
        transport_advantage_p[layer] = sign_flip_p_value(transport_advantage);
        additive_t[layer] = paired_t_test(additive_gain);
        ambient_gain_t[layer] = paired_t_test(ambient_gain);
        transport_gain_t[layer] = paired_t_test(transport_gain);
        endpoint_t[layer] = paired_t_test(endpoint_gain);
        ambient_advantage_t[layer] = paired_t_test(ambient_advantage);
        transport_advantage_t[layer] = paired_t_test(transport_advantage);
        for (int left = 0; left < k_rank; ++left) for (int right = 0; right < k_rank; ++right) {
            double dot = 0.0;
            for (size_t j = 0; j < width; ++j) dot += double(raw_basis[j + width * left]) * raw_basis[j + width * right];
            max_basis_orthogonality_error = std::max(max_basis_orthogonality_error, std::abs(dot - (left == right ? 1.0 : 0.0)));
            layer_basis_error[layer] = std::max(layer_basis_error[layer], std::abs(dot - (left == right ? 1.0 : 0.0)));
        }
        for (size_t row = 0; row < 768; ++row) {
            double mean_x = 0.0, mean_y = 0.0;
            for (size_t j = 0; j < width; ++j) {
                mean_x += resid_in[row * width + j];
                mean_y += resid_time[row * width + j];
            }
            mean_x /= width; mean_y /= width;
            double xx = 0.0, yy = 0.0, xy = 0.0;
            for (size_t j = 0; j < width; ++j) {
                const double x = resid_in[row * width + j] - mean_x;
                const double y = resid_time[row * width + j] - mean_y;
                xx += x * x; yy += y * y; xy += x * y;
            }
            if (!(xx > 1e-24) || !(yy > 1e-24)) throw std::runtime_error("degenerate centered residual");
            const double cosine = std::clamp(xy / std::sqrt(xx * yy), -1.0, 1.0);
            const double theta = std::acos(cosine);
            if (!std::isfinite(theta)) throw std::runtime_error("non-finite log-tangent angle");
            double incoming_reference_dot = 0.0;
            for (size_t j = 0; j < width; ++j) incoming_reference_dot += (resid_in[row * width + j] - mean_x) / std::sqrt(xx) * reference[j];
            const double denominator = 1.0 + incoming_reference_dot;
            if (!(denominator > 1e-12)) throw std::runtime_error("near-antipodal transport denominator");
            // Parallel transport from u to the train-only reference preserves the tangent norm.
            double tangent_reference_dot = 0.0;
            for (size_t j = 0; j < width; ++j) {
                const double x = (resid_in[row * width + j] - mean_x) / std::sqrt(xx);
                const double y = (resid_time[row * width + j] - mean_y) / std::sqrt(yy);
                const double tangent = (y - cosine * x) / std::sqrt(std::max(1e-24, 1.0 - cosine * cosine)) * theta;
                tangent_reference_dot += tangent * reference[j];
            }
            double transported_norm_sq = 0.0;
            for (size_t j = 0; j < width; ++j) {
                const double x = (resid_in[row * width + j] - mean_x) / std::sqrt(xx);
                const double y = (resid_time[row * width + j] - mean_y) / std::sqrt(yy);
                const double tangent = (y - cosine * x) / std::sqrt(std::max(1e-24, 1.0 - cosine * cosine)) * theta;
                const double transported = tangent - tangent_reference_dot * (x + reference[j]) / denominator;
                transported_norm_sq += transported * transported;
            }
            if (!std::isfinite(tangent_reference_dot) || !std::isfinite(transported_norm_sq)) throw std::runtime_error("non-finite transported tangent");
            max_angle = std::max(max_angle, theta);
            ++geometry_rows;
            ++transport_rows;
        }
    }
    std::printf("progress=analysis_complete mode=%s rows=%llu sources=%zu width=%zu finite=yes max_residual_identity_error=%g geometry_rows=%llu transport_rows=%llu max_angle=%g raw_basis_orthogonality_error=%g raw_decoder_weight_norm=%g raw_held_out_ce_mean=%g incoming_held_out_ce_mean=%g ambient_held_out_ce_mean=%g transported_held_out_ce_mean=%g endpoint_held_out_ce_mean=%g additive_held_out_ce_mean=%g relative_ambient_held_out_ce_mean=%g relative_transport_held_out_ce_mean=%g\n",
                development ? "development" : "classified",
                (unsigned long long) rows, reader.sources().size(), reader.dimension(), max_identity_error,
                (unsigned long long) geometry_rows, (unsigned long long) transport_rows, max_angle, max_basis_orthogonality_error,
                std::sqrt(raw_decoder_weight_norm), raw_held_out_cross_entropy / 4.0, incoming_held_out_cross_entropy / 4.0,
                ambient_held_out_cross_entropy / 4.0, transported_held_out_cross_entropy / 4.0, endpoint_held_out_cross_entropy / 4.0,
                additive_held_out_cross_entropy / 4.0, relative_ambient_held_out_cross_entropy / 4.0,
                relative_transport_held_out_cross_entropy / 4.0);
    const std::vector<double> additive_holm = holm_adjust(std::vector<double>(additive_gain_p.begin(), additive_gain_p.end()));
    const std::vector<double> endpoint_holm = holm_adjust(std::vector<double>(endpoint_gain_p.begin(), endpoint_gain_p.end()));
    std::vector<double> relative_gain_p, relative_advantage_p;
    for (size_t i = 0; i < 4; ++i) { relative_gain_p.push_back(ambient_gain_p[i]); relative_gain_p.push_back(transport_gain_p[i]);
                                     relative_advantage_p.push_back(ambient_advantage_p[i]); relative_advantage_p.push_back(transport_advantage_p[i]); }
    const std::vector<double> relative_gain_holm = holm_adjust(relative_gain_p);
    const std::vector<double> relative_advantage_holm = holm_adjust(relative_advantage_p);
    for (size_t i = 0; i < 4; ++i) {
        additive_gain_holm[i] = additive_holm[i]; endpoint_gain_holm[i] = endpoint_holm[i];
        ambient_gain_holm[i] = relative_gain_holm[2 * i]; transport_gain_holm[i] = relative_gain_holm[2 * i + 1];
        ambient_advantage_holm[i] = relative_advantage_holm[2 * i]; transport_advantage_holm[i] = relative_advantage_holm[2 * i + 1];
    }
    for (size_t layer = 0; layer < reference_norms.size(); ++layer) {
        std::printf("layer=%d train_only_reference_norm=%g fixed_lambda_additive_gain_p=%g fixed_lambda_ambient_advantage_p=%g fixed_lambda_transport_advantage_p=%g\n",
                    int(layer == 0 ? 15 : layer == 1 ? 30 : layer == 2 ? 45 : 60), reference_norms[layer], additive_gain_p[layer],
                    ambient_advantage_p[layer], transport_advantage_p[layer]);
    }
    std::array<std::string, 4> layer_outcomes;
    std::array<bool, 4> ambient_predicate{}, transported_predicate{}, additive_predicate{}, endpoint_predicate{};
    bool any_relative = false, any_coordinate_dependent = false, any_additive = false;
    for (size_t layer = 0; layer < 4; ++layer) {
        const auto mean = [](const std::array<double, 16> & values) { double total = 0; for (double value : values) total += value; return total / values.size(); };
        const bool ambient = mean(ambient_gains[layer]) > 0 && ambient_gain_holm[layer] <= .05 &&
                             mean(ambient_advantages[layer]) > 0 && ambient_advantage_holm[layer] <= .05;
        const bool transported = mean(transport_gains[layer]) > 0 && transport_gain_holm[layer] <= .05 &&
                                 mean(transport_advantages[layer]) > 0 && transport_advantage_holm[layer] <= .05;
        const bool additive = mean(additive_gains[layer]) > 0 && additive_gain_holm[layer] <= .05;
        const bool endpoint = mean(endpoint_gains[layer]) > 0 && endpoint_gain_holm[layer] <= .05;
        ambient_predicate[layer] = ambient; transported_predicate[layer] = transported;
        additive_predicate[layer] = additive; endpoint_predicate[layer] = endpoint;
        layer_outcomes[layer] = mechanical_outcome(ambient, transported, additive, endpoint);
        any_relative |= layer_outcomes[layer] == "relative_coordinate_advantage";
        any_coordinate_dependent |= layer_outcomes[layer] == "coordinate_dependent_relative_signal";
        any_additive |= layer_outcomes[layer] == "additive_coordinates_match_or_win";
    }
    bool all_converged = true;
    for (const auto & layer : decoder_fits) for (const auto & record : layer) all_converged &= record.converged && std::isfinite(record.gradient_norm);
    std::string classification = development ? "suppressed_development_only" : !all_converged ? "suppressed_invalid" : any_relative ? "relative_decoding_supported" :
                                 (!any_coordinate_dependent && any_additive ? "relative_decoding_not_supported" : "mixed_or_ambiguous");
    if (!development) {
        auto predicate_model = ROOT::RNTupleModel::Create();
        auto layer_field = predicate_model->MakeField<int32_t>("layer");
        auto outcome_field = predicate_model->MakeField<std::string>("outcome");
        auto ambient_field = predicate_model->MakeField<bool>("relative_ambient_passes");
        auto transported_field = predicate_model->MakeField<bool>("relative_transported_passes");
        auto additive_field = predicate_model->MakeField<bool>("additive_passes");
        auto endpoint_field = predicate_model->MakeField<bool>("endpoint_passes");
        auto predicate = ROOT::RNTupleWriter::Recreate(std::move(predicate_model), "outcome_predicates", output_root);
        for (size_t layer = 0; layer < 4; ++layer) {
            *layer_field = int32_t(layer == 0 ? 15 : layer == 1 ? 30 : layer == 2 ? 45 : 60); *outcome_field = layer_outcomes[layer];
            *ambient_field = ambient_predicate[layer]; *transported_field = transported_predicate[layer];
            *additive_field = additive_predicate[layer]; *endpoint_field = endpoint_predicate[layer]; predicate->Fill();
        }
        predicate->CommitCluster();
        predicate.reset();
        TFile file(output_root.c_str(), "UPDATE");
        auto carrier_model = ROOT::RNTupleModel::Create();
        auto carrier_layer = carrier_model->MakeField<int32_t>("layer");
        auto carrier_id = carrier_model->MakeField<int32_t>("carrier_id");
        auto metric_name = carrier_model->MakeField<std::string>("metric_name");
        auto metric_value = carrier_model->MakeField<double>("metric_value");
        auto carrier_metrics = ROOT::RNTupleWriter::Append(std::move(carrier_model), "per_carrier_metrics", file);
        for (size_t layer = 0; layer < 4; ++layer) for (int carrier = 0; carrier < 16; ++carrier) {
            *carrier_layer = int32_t(layer == 0 ? 15 : layer == 1 ? 30 : layer == 2 ? 45 : 60); *carrier_id = carrier;
            for (const auto & [name, values] : std::array<std::pair<const char *, const std::array<std::array<double, 16>, 4> *>, 6>{{
                    { "gain_additive", &additive_gains }, { "gain_relative_ambient", &ambient_gains },
                    { "gain_relative_transported", &transport_gains }, { "gain_endpoint", &endpoint_gains },
                    { "advantage_ambient", &ambient_advantages }, { "advantage_transported", &transport_advantages } }}) {
                *metric_name = name; *metric_value = (*values)[layer][carrier]; carrier_metrics->Fill();
            }
            for (size_t model = 0; model < 5; ++model) {
                double ce = 0.0, accuracy = 0.0;
                for (int value = 0; value < 16; ++value) {
                    const auto & row = predictions[layer][model][size_t(carrier) * 16 + value];
                    ce += row.cross_entropy / 16.0; accuracy += (row.correct ? 1.0 : 0.0) / 16.0;
                }
                *metric_name = "ce_" + std::to_string(model); *metric_value = ce; carrier_metrics->Fill();
                *metric_name = "accuracy_" + std::to_string(model); *metric_value = accuracy; carrier_metrics->Fill();
            }
        }
        carrier_metrics->CommitCluster();
        auto test_model = ROOT::RNTupleModel::Create();
        auto test_layer = test_model->MakeField<int32_t>("layer");
        auto test_name = test_model->MakeField<std::string>("test_name");
        auto raw_p = test_model->MakeField<double>("raw_p_value");
        auto adjusted_p = test_model->MakeField<double>("holm_adjusted_p_value");
        auto t_statistic = test_model->MakeField<double>("posthoc_paired_t_statistic");
        auto t_p_value = test_model->MakeField<double>("posthoc_paired_t_two_sided_p_value");
        auto tests = ROOT::RNTupleWriter::Append(std::move(test_model), "statistical_tests", file);
        for (size_t layer = 0; layer < 4; ++layer) {
            *test_layer = int32_t(layer == 0 ? 15 : layer == 1 ? 30 : layer == 2 ? 45 : 60);
            for (const auto & [name, raw, adjusted, t] : std::array<std::tuple<const char *, double, double, paired_t_result>, 6>{{
                    { "gain_additive", additive_gain_p[layer], additive_gain_holm[layer], additive_t[layer] },
                    { "gain_relative_ambient", ambient_gain_p[layer], ambient_gain_holm[layer], ambient_gain_t[layer] },
                    { "gain_relative_transported", transport_gain_p[layer], transport_gain_holm[layer], transport_gain_t[layer] },
                    { "gain_endpoint", endpoint_gain_p[layer], endpoint_gain_holm[layer], endpoint_t[layer] },
                    { "advantage_ambient", ambient_advantage_p[layer], ambient_advantage_holm[layer], ambient_advantage_t[layer] },
                    { "advantage_transported", transport_advantage_p[layer], transport_advantage_holm[layer], transport_advantage_t[layer] } }}) {
                *test_name = name; *raw_p = raw; *adjusted_p = adjusted; *t_statistic = t.statistic; *t_p_value = t.two_sided_p; tests->Fill();
            }
        }
        tests->CommitCluster();
        const std::array<const char *, 5> model_names{ "incoming", "additive", "relative_ambient", "relative_transported", "endpoint" };
        auto basis_model = ROOT::RNTupleModel::Create();
        auto basis_layer = basis_model->MakeField<int32_t>("layer");
        auto basis_error = basis_model->MakeField<double>("max_orthogonality_error");
        auto basis_writer = ROOT::RNTupleWriter::Append(std::move(basis_model), "feature_basis_checks", file);
        for (size_t layer = 0; layer < 4; ++layer) { *basis_layer = int32_t(layer == 0 ? 15 : layer == 1 ? 30 : layer == 2 ? 45 : 60); *basis_error = layer_basis_error[layer]; basis_writer->Fill(); }
        basis_writer->CommitCluster();
        auto cv_model = ROOT::RNTupleModel::Create();
        auto cv_layer = cv_model->MakeField<int32_t>("layer");
        auto cv_name = cv_model->MakeField<std::string>("model");
        auto cv_lambda = cv_model->MakeField<double>("lambda");
        auto cv_score = cv_model->MakeField<double>("mean_carrier_cross_entropy");
        auto cv_writer = ROOT::RNTupleWriter::Append(std::move(cv_model), "cross_validation_scores", file);
        static constexpr std::array<double, 7> lambda_grid{ 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0 };
        for (size_t layer = 0; layer < 4; ++layer) for (size_t model = 0; model < 5; ++model) {
            if (cv_results[layer][model].scores.size() != lambda_grid.size()) throw std::runtime_error("missing cross-validation score");
            for (size_t candidate = 0; candidate < lambda_grid.size(); ++candidate) {
                *cv_layer = int32_t(layer == 0 ? 15 : layer == 1 ? 30 : layer == 2 ? 45 : 60); *cv_name = model_names[model];
                *cv_lambda = lambda_grid[candidate]; *cv_score = cv_results[layer][model].scores[candidate]; cv_writer->Fill();
            }
        }
        cv_writer->CommitCluster();
        auto fit_model = ROOT::RNTupleModel::Create();
        auto fit_layer = fit_model->MakeField<int32_t>("layer");
        auto fit_name = fit_model->MakeField<std::string>("model");
        auto fit_lambda = fit_model->MakeField<double>("lambda");
        auto fit_features = fit_model->MakeField<int32_t>("features");
        auto fit_iterations = fit_model->MakeField<int32_t>("iterations");
        auto fit_gradient = fit_model->MakeField<double>("gradient_norm");
        auto fit_converged = fit_model->MakeField<bool>("converged");
        auto fit_writer = ROOT::RNTupleWriter::Append(std::move(fit_model), "decoder_fits", file);
        for (size_t layer = 0; layer < 4; ++layer) for (size_t model = 0; model < 5; ++model) {
            const auto & record = decoder_fits[layer][model];
            *fit_layer = int32_t(layer == 0 ? 15 : layer == 1 ? 30 : layer == 2 ? 45 : 60); *fit_name = model_names[model];
            *fit_lambda = record.lambda; *fit_features = record.features; *fit_iterations = record.iterations;
            *fit_gradient = record.gradient_norm; *fit_converged = record.converged; fit_writer->Fill();
        }
        fit_writer->CommitCluster();
        auto aggregate_model = ROOT::RNTupleModel::Create();
        auto aggregate_layer = aggregate_model->MakeField<int32_t>("layer");
        auto aggregate_name = aggregate_model->MakeField<std::string>("model");
        auto aggregate_ce = aggregate_model->MakeField<double>("carrier_balanced_cross_entropy");
        auto aggregate_accuracy = aggregate_model->MakeField<double>("carrier_balanced_accuracy");
        auto aggregate_writer = ROOT::RNTupleWriter::Append(std::move(aggregate_model), "aggregate_metrics", file);
        for (size_t layer = 0; layer < 4; ++layer) for (size_t model = 0; model < 5; ++model) {
            *aggregate_layer = int32_t(layer == 0 ? 15 : layer == 1 ? 30 : layer == 2 ? 45 : 60); *aggregate_name = model_names[model];
            *aggregate_ce = held_out_ce[layer][model];
            double correct = 0.0; for (const auto & row : predictions[layer][model]) correct += row.correct ? 1.0 : 0.0;
            *aggregate_accuracy = correct / predictions[layer][model].size(); aggregate_writer->Fill();
        }
        aggregate_writer->CommitCluster();
        auto prediction_model = ROOT::RNTupleModel::Create();
        auto prediction_layer = prediction_model->MakeField<int32_t>("layer");
        auto prediction_name = prediction_model->MakeField<std::string>("model");
        auto prediction_carrier = prediction_model->MakeField<int32_t>("carrier_id");
        auto prediction_value = prediction_model->MakeField<int32_t>("value_id");
        auto prediction_true = prediction_model->MakeField<int32_t>("true_value_id");
        auto prediction_predicted = prediction_model->MakeField<int32_t>("predicted_value_id");
        auto prediction_probability = prediction_model->MakeField<double>("true_value_probability");
        auto prediction_ce = prediction_model->MakeField<double>("cross_entropy");
        auto prediction_correct = prediction_model->MakeField<bool>("correct");
        auto prediction_writer = ROOT::RNTupleWriter::Append(std::move(prediction_model), "predictions", file);
        for (size_t layer = 0; layer < 4; ++layer) for (size_t model = 0; model < 5; ++model) for (const auto & row : predictions[layer][model]) {
            *prediction_layer = int32_t(layer == 0 ? 15 : layer == 1 ? 30 : layer == 2 ? 45 : 60); *prediction_name = model_names[model];
            *prediction_carrier = row.carrier_id; *prediction_value = row.value_id; *prediction_true = row.value_id;
            *prediction_predicted = row.predicted_value_id; *prediction_probability = row.true_probability;
            *prediction_ce = row.cross_entropy; *prediction_correct = row.correct; prediction_writer->Fill();
        }
        prediction_writer->CommitCluster();
        auto behavior_model = ROOT::RNTupleModel::Create();
        auto behavior_group = behavior_model->MakeField<std::string>("group");
        auto behavior_id = behavior_model->MakeField<int32_t>("group_id");
        auto behavior_count = behavior_model->MakeField<int32_t>("count");
        auto behavior_log_probability = behavior_model->MakeField<double>("mean_expected_value_log_probability");
        auto behavior_rank = behavior_model->MakeField<double>("mean_expected_value_rank");
        auto behavior_accuracy = behavior_model->MakeField<double>("native_top1_accuracy");
        auto behavior_writer = ROOT::RNTupleWriter::Append(std::move(behavior_model), "behavior_diagnostics", file);
        for (const auto & row : behavior_summaries) {
            *behavior_group = row.group; *behavior_id = row.id; *behavior_count = row.count;
            *behavior_log_probability = row.log_probability; *behavior_rank = row.rank; *behavior_accuracy = row.top1_accuracy; behavior_writer->Fill();
        }
        behavior_writer->CommitCluster();
    }
    const bool post_write_audit_passed = development || audit_result_root(output_root);
    const bool artifact_valid = !development && all_converged && post_write_audit_passed;
    if (!development) {
        if (!artifact_valid) classification = "suppressed_invalid";
        TFile file(output_root.c_str(), "UPDATE");
        auto metadata_model = ROOT::RNTupleModel::Create();
        auto key = metadata_model->MakeField<std::string>("key");
        auto value = metadata_model->MakeField<std::string>("value");
        auto metadata = ROOT::RNTupleWriter::Append(std::move(metadata_model), "metadata", file);
        for (const auto & [name, text] : std::array<std::pair<std::string, std::string>, 5>{{
                 { "status", artifact_valid ? "valid" : "invalid" }, { "classification", classification },
                 { "registration", registration }, { "input_root", input },
                 { "post_write_audit", post_write_audit_passed ? "passed" : "failed" } }}) {
            *key = name; *value = text; metadata->Fill();
        }
        metadata->CommitCluster();
    }
    std::ofstream output(output_json);
    if (!output) throw std::runtime_error("cannot create development output JSON");
    const std::string registration_text = development ? "" : read_text(registration);
    output << std::setprecision(17) << "{\n  \"schema_version\": 2,\n  \"status\": \"" << (development ? "development_only" : artifact_valid ? "valid" : "invalid")
           << "\",\n  \"classification\": \"" << classification << "\",\n  \"seed\": " << k_seed
            << ",\n  \"input_root\": \"" << input << "\",\n  \"output_root\": \"" << output_root << "\",\n  \"registration\": \"" << registration
           << "\",\n  \"frozen_hashes\": {\"model_sha256\":\"" << (development ? "" : json_string(registration_text, "model_sha256"))
           << "\",\"corpus_sha256\":\"" << (development ? "" : json_string(registration_text, "corpus_sha256"))
           << "\",\"manifest_sha256\":\"" << (development ? "" : json_string(registration_text, "manifest_sha256"))
           << "\",\"candidate_list_sha256\":\"" << (development ? "" : json_string(registration_text, "candidate_list_sha256"))
           << "\",\"template_sha256\":\"" << (development ? "" : json_string(registration_text, "template_sha256")) << "\"},"
           << "\n  \"token_positions\": {\"read_token_id\":" << (development ? 0 : json_integer(registration_text, "read_token_id"))
           << ",\"read_position\":" << (development ? 0 : json_integer(registration_text, "read_position"))
           << ",\"value_position\":" << (development ? 0 : json_integer(registration_text, "value_position"))
           << ",\"carrier_position\":" << (development ? 0 : json_integer(registration_text, "carrier_position")) << "},"
           << "\n  \"sample_counts\": {\"capture_rows\":768,\"behavior_rows\":768,\"sample_rows\":3072,\"prediction_rows\":5120,\"decoder_fit_rows\":20,\"cv_rows\":140,\"statistical_test_rows\":24},"
            << "\n  \"numerical_checks\": {\"max_residual_identity_error\":" << max_identity_error << ",\"residual_identity_threshold\":0.001,\"max_basis_orthogonality_error\":" << max_basis_orthogonality_error << ",\"basis_orthogonality_threshold\":1e-05,\"geometry_rows\":" << geometry_rows << ",\"transport_rows\":" << transport_rows << ",\"post_write_audit_passed\":" << (post_write_audit_passed ? "true" : "false") << "},"
           << "\n  \"layers\": [";
    const std::array<const char *, 5> json_models{ "incoming", "additive", "relative_ambient", "relative_transported", "endpoint" };
    for (size_t layer = 0; layer < 4; ++layer) {
        if (layer) output << ',';
        output << "\n    {\"layer\":" << (layer == 0 ? 15 : layer == 1 ? 30 : layer == 2 ? 45 : 60) << ",\"outcome\":\"" << layer_outcomes[layer]
               << "\",\"predicates\":{\"relative_ambient_passes\":" << (ambient_predicate[layer] ? "true" : "false")
               << ",\"relative_transported_passes\":" << (transported_predicate[layer] ? "true" : "false")
               << ",\"additive_passes\":" << (additive_predicate[layer] ? "true" : "false")
               << ",\"endpoint_passes\":" << (endpoint_predicate[layer] ? "true" : "false") << "},\"models\":{";
        for (size_t model = 0; model < 5; ++model) {
            if (model) output << ',';
            const auto & fit = decoder_fits[layer][model];
            double accuracy = 0.0; for (const auto & row : predictions[layer][model]) accuracy += (row.correct ? 1.0 : 0.0) / predictions[layer][model].size();
            output << "\"" << json_models[model] << "\":{\"cross_entropy\":" << held_out_ce[layer][model] << ",\"accuracy\":" << accuracy
                   << ",\"selected_lambda\":" << fit.lambda << ",\"iterations\":" << fit.iterations << ",\"gradient_norm\":" << fit.gradient_norm << ",\"converged\":" << (fit.converged ? "true" : "false") << '}';
        }
        output << "},\"statistics\":{\"gain_additive\":{\"raw_p\":" << additive_gain_p[layer] << ",\"holm_p\":" << additive_gain_holm[layer]
               << ",\"posthoc_paired_t\":{\"statistic\":" << additive_t[layer].statistic << ",\"two_sided_p\":" << additive_t[layer].two_sided_p << "}}"
               << ",\"gain_relative_ambient\":{\"raw_p\":" << ambient_gain_p[layer] << ",\"holm_p\":" << ambient_gain_holm[layer]
               << ",\"posthoc_paired_t\":{\"statistic\":" << ambient_gain_t[layer].statistic << ",\"two_sided_p\":" << ambient_gain_t[layer].two_sided_p << "}}"
               << ",\"gain_relative_transported\":{\"raw_p\":" << transport_gain_p[layer] << ",\"holm_p\":" << transport_gain_holm[layer]
               << ",\"posthoc_paired_t\":{\"statistic\":" << transport_gain_t[layer].statistic << ",\"two_sided_p\":" << transport_gain_t[layer].two_sided_p << "}}"
               << ",\"advantage_ambient\":{\"raw_p\":" << ambient_advantage_p[layer] << ",\"holm_p\":" << ambient_advantage_holm[layer]
               << ",\"posthoc_paired_t\":{\"statistic\":" << ambient_advantage_t[layer].statistic << ",\"two_sided_p\":" << ambient_advantage_t[layer].two_sided_p << "}}"
               << ",\"advantage_transported\":{\"raw_p\":" << transport_advantage_p[layer] << ",\"holm_p\":" << transport_advantage_holm[layer]
               << ",\"posthoc_paired_t\":{\"statistic\":" << transport_advantage_t[layer].statistic << ",\"two_sided_p\":" << transport_advantage_t[layer].two_sided_p << "}}}}";
    }
    output << "\n  ],\n  \"cross_validation_scores\": [";
    for (size_t layer = 0; layer < 4; ++layer) for (size_t model = 0; model < 5; ++model) {
        if (layer || model) output << ',';
        output << "\n    {\"layer\":" << (layer == 0 ? 15 : layer == 1 ? 30 : layer == 2 ? 45 : 60)
               << ",\"model\":\"" << json_models[model] << "\",\"scores\":[";
        for (size_t candidate = 0; candidate < cv_results[layer][model].scores.size(); ++candidate) {
            if (candidate) output << ',';
            output << "{\"lambda\":" << k_lambda_grid[candidate] << ",\"mean_carrier_cross_entropy\":" << cv_results[layer][model].scores[candidate] << '}';
        }
        output << "]}";
    }
    output << "\n  ],\n  \"per_carrier_metrics\": [";
    for (size_t layer = 0; layer < 4; ++layer) for (int carrier = 0; carrier < 16; ++carrier) {
        if (layer || carrier) output << ',';
        output << "\n    {\"layer\":" << (layer == 0 ? 15 : layer == 1 ? 30 : layer == 2 ? 45 : 60) << ",\"carrier_id\":" << carrier << ",\"models\":{";
        for (size_t model = 0; model < 5; ++model) {
            if (model) output << ',';
            double ce = 0.0, accuracy = 0.0;
            for (int value = 0; value < 16; ++value) {
                const auto & row = predictions[layer][model][size_t(carrier) * 16 + value];
                ce += row.cross_entropy / 16.0; accuracy += (row.correct ? 1.0 : 0.0) / 16.0;
            }
            output << "\"" << json_models[model] << "\":{\"cross_entropy\":" << ce << ",\"accuracy\":" << accuracy << '}';
        }
        output << "},\"gains\":{\"additive\":" << additive_gains[layer][carrier]
               << ",\"relative_ambient\":" << ambient_gains[layer][carrier]
               << ",\"relative_transported\":" << transport_gains[layer][carrier]
               << ",\"endpoint\":" << endpoint_gains[layer][carrier]
               << "},\"advantages\":{\"ambient\":" << ambient_advantages[layer][carrier]
               << ",\"transported\":" << transport_advantages[layer][carrier] << "}}";
    }
    output << "\n  ],\n  \"behavior_diagnostics\": [";
    for (size_t i = 0; i < behavior_summaries.size(); ++i) {
        if (i) output << ',';
        const auto & row = behavior_summaries[i];
        output << "\n    {\"group\":\"" << row.group << "\",\"group_id\":" << row.id << ",\"count\":" << row.count
               << ",\"mean_expected_value_log_probability\":" << row.log_probability << ",\"mean_expected_value_rank\":" << row.rank
               << ",\"native_top1_accuracy\":" << row.top1_accuracy << '}';
    }
    output << "\n  ],\n  \"factorial\": {\"value_count\":16,\"carrier_count\":48,\"train_carrier_count\":32,\"test_carrier_count\":16,\"prompt_count\":768,\"complete\":true},"
           << "\n  \"validity_checks\": ["
           << "{\"name\":\"residual_identity\",\"threshold\":0.001,\"observed\":" << max_identity_error << ",\"passed\":" << (max_identity_error <= 1e-3 ? "true" : "false") << "},"
           << "{\"name\":\"basis_orthogonality\",\"threshold\":1e-05,\"observed\":" << max_basis_orthogonality_error << ",\"passed\":" << (max_basis_orthogonality_error <= 1e-5 ? "true" : "false") << "},"
           << "{\"name\":\"complete_geometry\",\"expected\":3072,\"observed\":" << geometry_rows << ",\"passed\":" << (geometry_rows == 3072 ? "true" : "false") << "},"
           << "{\"name\":\"complete_transport\",\"expected\":3072,\"observed\":" << transport_rows << ",\"passed\":" << (transport_rows == 3072 ? "true" : "false") << "},"
           << "{\"name\":\"post_write_audit\",\"expected\":true,\"observed\":" << (post_write_audit_passed ? "true" : "false") << ",\"passed\":" << (post_write_audit_passed ? "true" : "false") << "}],"
           << "\n  \"artifact_rows\": {\"metadata\":5,\"feature_basis_checks\":4,\"cross_validation_scores\":140,\"decoder_fits\":20,\"per_carrier_metrics\":1024,\"aggregate_metrics\":20,\"statistical_tests\":24,\"outcome_predicates\":4,\"predictions\":5120,\"behavior_diagnostics\":65},"
           << "\n  \"model\": {\"path\":\"" << (development ? "" : json_string(registration_text, "model_path")) << "\",\"backend\":\"" << (development ? "" : json_string(registration_text, "backend")) << "\",\"n_gpu_layers\":" << (development ? 0 : json_integer(registration_text, "n_gpu_layers")) << "},"
           << "\n  \"protocol_deviations\": []\n}\n";
    if (!output) throw std::runtime_error("cannot write development output JSON");
    return 0;
} catch (const std::exception & error) {
    std::fprintf(stderr, "error: %s\n", error.what());
    return 1;
}
