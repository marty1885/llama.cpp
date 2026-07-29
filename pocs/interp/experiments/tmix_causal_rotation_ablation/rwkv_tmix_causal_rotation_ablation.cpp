#include "rwkv_experiment.hpp"

#include "llama-model.h"

#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleReader.hxx>
#include <ROOT/RNTupleWriter.hxx>
#include <TFile.h>
#include <openssl/sha.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <numeric>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

constexpr int k_layer = 45;
constexpr int k_value_id = 0;
constexpr int k_train_carriers = 32;
constexpr int k_test_carriers = 16;
constexpr int k_controls = 3;
constexpr uint64_t k_seed = 315971;
constexpr double k_norm_epsilon = 1e-12;
constexpr double k_angle_epsilon = 1e-4;
constexpr double k_endpoint_tolerance = 2e-3;
constexpr double k_repeat_cosine = 0.99999;
constexpr double k_repeat_relative_norm = 1e-4;

using vec = std::vector<double>;

struct args {
    enum class mode { none, self_test, calibration, classified } run_mode = mode::none;
    std::string model;
    std::string prompts;
    std::string manifest;
    std::string registration;
    std::string output_root;
    std::string output_json;
    uint64_t seed = 0;
    bool have_seed = false;
    int n_gpu_layers = 0;
};

struct manifest_data {
    int token_count = 0;
    int read_position = 0;
    int read_token_id = 0;
    int carrier_position = 0;
    int value_position = 0;
    std::array<int32_t, 16> value_tokens{};
    std::array<int32_t, 48> carrier_tokens{};
};

struct geometry {
    double mean = 0;
    double radius = 0;
    double theta = 0;
    vec u;
    vec v;
    vec h;
};

struct logits_summary {
    double log_probability = 0;
    int32_t rank = 0;
    int32_t top1 = 0;
};

struct native_sample {
    int carrier_id = 0;
    int32_t carrier_token_id = 0;
    int32_t expected_token_id = 0;
    int32_t read_token_id = 0;
    int32_t read_position = 0;
    geometry geom;
    std::vector<float> resid_in;
    std::vector<float> time_out;
    std::vector<float> resid_time;
    double log_probability = 0;
    int32_t rank = 0;
    int32_t top1 = 0;
};

struct intervention_row {
    int carrier_id = 0;
    int32_t carrier_token_id = 0;
    int value_id = k_value_id;
    int32_t expected_token_id = 0;
    std::string condition;
    int control_index = -1;
    int donor_carrier_id = -1;
    uint64_t random_seed = 0;
    double native_log_probability = 0;
    double perturbed_log_probability = 0;
    double behavior_loss = 0;
    int32_t expected_rank = 0;
    int32_t top1_token_id = 0;
    bool top1_correct = false;
    double kl_native_perturbed = 0;
    double requested_angle = 0;
    double actual_angle = 0;
    double requested_endpoint_distance = 0;
    double actual_endpoint_distance = 0;
    double mean_error = 0;
    double radius_relative_error = 0;
    double angle_error = 0;
    double unrotation_error = -1;
    double time_out_reconstruction_error = 0;
    double ffn_norm_delta_norm = 0;
    double channel_out_delta_norm = 0;
    double resid_out_delta_norm = 0;
    bool numerical_pass = false;
};

struct carrier_metric {
    int carrier_id = 0;
    double native_unrotation_loss = 0;
    double foreign_mean_loss = 0;
    double random_mean_loss = 0;
    double delta_foreign = 0;
    double delta_random = 0;
};

struct test_result {
    std::string family;
    double mean_delta = 0;
    double raw_p = 1;
    double adjusted_p = 1;
};

void quiet_llama_logs(ggml_log_level, const char *, void *) {}

void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s --self-test | --calibration -m MODEL --prompts PROMPTS --manifest MANIFEST --registration REGISTRATION --output-json FILE --seed 315971 -ngl 99 | --classified -m MODEL --prompts PROMPTS --manifest MANIFEST --registration REGISTRATION --output-root FILE --output-json FILE --seed 315971 -ngl 99\n",
        argv0);
}

args parse_args(int argc, char ** argv) {
    args a;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--self-test") && a.run_mode == args::mode::none) a.run_mode = args::mode::self_test;
        else if (!std::strcmp(argv[i], "--calibration") && a.run_mode == args::mode::none) a.run_mode = args::mode::calibration;
        else if (!std::strcmp(argv[i], "--classified") && a.run_mode == args::mode::none) a.run_mode = args::mode::classified;
        else if (!std::strcmp(argv[i], "-m") && i + 1 < argc) a.model = argv[++i];
        else if (!std::strcmp(argv[i], "--prompts") && i + 1 < argc) a.prompts = argv[++i];
        else if (!std::strcmp(argv[i], "--manifest") && i + 1 < argc) a.manifest = argv[++i];
        else if (!std::strcmp(argv[i], "--registration") && i + 1 < argc) a.registration = argv[++i];
        else if (!std::strcmp(argv[i], "--output-root") && i + 1 < argc) a.output_root = argv[++i];
        else if (!std::strcmp(argv[i], "--output-json") && i + 1 < argc) a.output_json = argv[++i];
        else if (!std::strcmp(argv[i], "--seed") && i + 1 < argc) { a.seed = std::strtoull(argv[++i], nullptr, 10); a.have_seed = true; }
        else if (!std::strcmp(argv[i], "-ngl") && i + 1 < argc) a.n_gpu_layers = std::atoi(argv[++i]);
        else throw std::runtime_error("unknown or incomplete argument: " + std::string(argv[i]));
    }
    if (a.run_mode == args::mode::self_test) {
        if (argc != 2) throw std::runtime_error("--self-test accepts no other arguments");
        return a;
    }
    if (a.run_mode == args::mode::none || a.model.empty() || a.prompts.empty() || a.manifest.empty() ||
        a.registration.empty() || a.output_json.empty() || !a.have_seed || a.seed != k_seed || a.n_gpu_layers != 99 ||
        (a.run_mode == args::mode::classified && a.output_root.empty()) ||
        (a.run_mode == args::mode::calibration && !a.output_root.empty())) {
        throw std::runtime_error("invalid or incomplete arguments");
    }
    if (std::filesystem::exists(a.output_json) || (!a.output_root.empty() && std::filesystem::exists(a.output_root))) {
        throw std::runtime_error("output path already exists; overwrite cannot classify");
    }
    return a;
}

std::string read_text(const std::string & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot read " + path);
    return { std::istreambuf_iterator<char>(input), {} };
}

std::string sha256_file(const std::string & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot hash " + path);
    SHA256_CTX context;
    SHA256_Init(&context);
    std::array<char, 1024 * 1024> chunk{};
    while (input.read(chunk.data(), chunk.size()) || input.gcount()) SHA256_Update(&context, chunk.data(), size_t(input.gcount()));
    std::array<unsigned char, SHA256_DIGEST_LENGTH> digest{};
    SHA256_Final(digest.data(), &context);
    static constexpr char hex[] = "0123456789abcdef";
    std::string result(64, '0');
    for (size_t i = 0; i < digest.size(); ++i) { result[2 * i] = hex[digest[i] >> 4]; result[2 * i + 1] = hex[digest[i] & 15]; }
    return result;
}

std::string json_string(const std::string & text, const char * key) {
    std::smatch match;
    if (!std::regex_search(text, match, std::regex(std::string("\\\"") + key + "\\\"\\s*:\\s*\\\"([^\\\"]+)\\\""))) {
        throw std::runtime_error(std::string("JSON lacks ") + key);
    }
    return match[1];
}

int json_integer(const std::string & text, const char * key) {
    std::smatch match;
    if (!std::regex_search(text, match, std::regex(std::string("\\\"") + key + "\\\"\\s*:\\s*([0-9]+)"))) {
        throw std::runtime_error(std::string("JSON lacks ") + key);
    }
    return std::stoi(match[1]);
}

manifest_data verify_frozen_inputs(const args & a, std::string & model_hash, std::string & prompts_hash,
                                   std::string & manifest_hash, std::string & registration_hash) {
    const std::string registration = read_text(a.registration);
    const std::string manifest = read_text(a.manifest);
    model_hash = sha256_file(a.model);
    prompts_hash = sha256_file(a.prompts);
    manifest_hash = sha256_file(a.manifest);
    registration_hash = sha256_file(a.registration);
    if (json_string(registration, "status") != "frozen" || json_string(registration, "backend") != "Vulkan" ||
        json_integer(registration, "n_gpu_layers") != 99 || json_string(registration, "model_sha256") != model_hash ||
        json_string(registration, "corpus_sha256") != prompts_hash || json_string(registration, "manifest_sha256") != manifest_hash ||
        json_string(manifest, "status") != "frozen" || json_integer(manifest, "prompt_count") != 768 ||
        json_string(manifest, "model_sha256") != model_hash || json_string(manifest, "corpus_sha256") != prompts_hash) {
        throw std::runtime_error("frozen registration, manifest, model, or prompt hash mismatch");
    }
    manifest_data data;
    data.token_count = json_integer(manifest, "token_count");
    data.read_position = json_integer(manifest, "read_position");
    data.read_token_id = json_integer(manifest, "read_token_id");
    data.carrier_position = json_integer(manifest, "carrier_position");
    data.value_position = json_integer(manifest, "value_position");
    std::vector<int32_t> ids;
    const std::regex pattern("\\\"token_id\\\"\\s*:\\s*([0-9]+)");
    for (std::sregex_iterator it(manifest.begin(), manifest.end(), pattern), end; it != end; ++it) ids.push_back(std::stoi((*it)[1]));
    if (ids.size() != 64) throw std::runtime_error("manifest token arrays are incomplete");
    std::copy_n(ids.begin(), 16, data.value_tokens.begin());
    std::copy_n(ids.begin() + 16, 48, data.carrier_tokens.begin());
    return data;
}

double dot(const vec & a, const vec & b) {
    if (a.size() != b.size()) throw std::runtime_error("vector size mismatch");
    double result = 0;
    for (size_t i = 0; i < a.size(); ++i) result += a[i] * b[i];
    return result;
}

double norm(const vec & x) { return std::sqrt(dot(x, x)); }

double mean(const vec & x) { return std::accumulate(x.begin(), x.end(), 0.0) / x.size(); }

vec to_double(const std::vector<float> & x) { return vec(x.begin(), x.end()); }

vec center(const vec & x) {
    const double mu = mean(x);
    vec result(x.size());
    for (size_t i = 0; i < x.size(); ++i) result[i] = x[i] - mu;
    return result;
}

vec unit(vec x) {
    const double n = norm(x);
    if (!std::isfinite(n) || n < k_norm_epsilon) throw std::runtime_error("non-finite or degenerate vector norm");
    for (double & value : x) value /= n;
    return x;
}

double angle(const vec & a, const vec & b) {
    return std::acos(std::clamp(dot(unit(a), unit(b)), -1.0, 1.0));
}

double distance(const vec & a, const vec & b) {
    if (a.size() != b.size()) throw std::runtime_error("vector size mismatch");
    vec delta(a.size());
    for (size_t i = 0; i < a.size(); ++i) delta[i] = a[i] - b[i];
    return norm(delta);
}

geometry make_geometry(const std::vector<float> & resid_in, const std::vector<float> & resid_time) {
    const vec a = to_double(resid_in), y = to_double(resid_time);
    if (a.size() != y.size() || a.empty()) throw std::runtime_error("invalid residual geometry dimensions");
    geometry g;
    g.mean = mean(y);
    g.radius = norm(center(y));
    g.u = unit(center(a));
    g.v = unit(center(y));
    const double cosine = std::clamp(dot(g.u, g.v), -1.0, 1.0);
    g.theta = std::acos(cosine);
    vec tangent(g.u.size());
    for (size_t i = 0; i < tangent.size(); ++i) tangent[i] = g.u[i] - cosine * g.v[i];
    g.h = unit(std::move(tangent));
    if (!std::isfinite(g.mean) || !std::isfinite(g.radius) || !std::isfinite(g.theta) || g.radius < k_norm_epsilon) {
        throw std::runtime_error("non-finite native geometry");
    }
    return g;
}

vec endpoint(const geometry & g, const vec & direction) {
    const vec target = unit(direction);
    vec result(target.size());
    for (size_t i = 0; i < result.size(); ++i) result[i] = g.mean + g.radius * target[i];
    return result;
}

vec direction_at_angle(const geometry & g, const vec & tangent, double alpha = 1.0) {
    vec result(g.v.size());
    const double radians = alpha * g.theta;
    for (size_t i = 0; i < result.size(); ++i) result[i] = std::cos(radians) * g.v[i] + std::sin(radians) * tangent[i];
    return unit(std::move(result));
}

vec transport_tangent(const geometry & donor, const geometry & target) {
    const double denominator = 1.0 + dot(donor.v, target.v);
    if (!std::isfinite(denominator) || denominator < 1e-4) throw std::runtime_error("foreign transport denominator rejected");
    vec result(donor.h.size());
    const double scale = dot(donor.h, target.v) / denominator;
    for (size_t i = 0; i < result.size(); ++i) result[i] = donor.h[i] - scale * (donor.v[i] + target.v[i]);
    return unit(std::move(result));
}

uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

double uniform_open(uint64_t & state) {
    state = splitmix64(state);
    return (double((state >> 11) + 1) / double((uint64_t(1) << 53) + 1));
}

vec gaussian(size_t width, uint64_t seed, int carrier_id, int control_index, int redraw) {
    uint64_t state = seed ^ (uint64_t(carrier_id) << 32) ^ (uint64_t(control_index) << 16) ^ uint64_t(redraw);
    vec result(width);
    for (size_t i = 0; i < width; i += 2) {
        const double u1 = uniform_open(state), u2 = uniform_open(state);
        const double radius = std::sqrt(-2.0 * std::log(u1));
        const double phase = 2.0 * std::acos(-1.0) * u2;
        result[i] = radius * std::cos(phase);
        if (i + 1 < width) result[i + 1] = radius * std::sin(phase);
    }
    return result;
}

std::pair<vec, uint64_t> random_tangent(const geometry & target, int carrier_id, int control_index) {
    for (int redraw = 0; redraw < 1000; ++redraw) {
        vec h = gaussian(target.v.size(), k_seed, carrier_id, control_index, redraw);
        const double projection = dot(h, target.v);
        for (size_t i = 0; i < h.size(); ++i) h[i] -= projection * target.v[i];
        h = unit(std::move(h));
        if (std::abs(dot(h, target.h)) <= 0.999) {
            return { std::move(h), splitmix64(k_seed ^ (uint64_t(carrier_id) << 32) ^ (uint64_t(control_index) << 16) ^ uint64_t(redraw)) };
        }
    }
    throw std::runtime_error("unable to draw random tangent control");
}

std::vector<int> donor_order(int carrier_id) {
    std::vector<int> donors(k_train_carriers);
    std::iota(donors.begin(), donors.end(), 0);
    std::sort(donors.begin(), donors.end(), [carrier_id](int left, int right) {
        const uint64_t salt = k_seed ^ (uint64_t(carrier_id) << 32) ^ 0x464f524549474eULL;
        const uint64_t a = splitmix64(salt ^ uint64_t(left));
        const uint64_t b = splitmix64(salt ^ uint64_t(right));
        return a == b ? left < right : a < b;
    });
    return donors;
}

std::vector<int> select_donors(const geometry & target, const std::array<native_sample, k_train_carriers> & train,
                               int carrier_id) {
    std::vector<int> selected;
    for (int donor : donor_order(carrier_id)) {
        try {
            const vec h = transport_tangent(train[donor].geom, target);
            if (std::abs(dot(h, target.h)) > 0.999) continue;
            selected.push_back(donor);
            if (selected.size() == k_controls) return selected;
        } catch (const std::runtime_error &) {
        }
    }
    throw std::runtime_error("unable to construct three distinct foreign donors");
}

double vector_cosine(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) throw std::runtime_error("capture width mismatch");
    double ab = 0, aa = 0, bb = 0;
    for (size_t i = 0; i < a.size(); ++i) { ab += double(a[i]) * b[i]; aa += double(a[i]) * a[i]; bb += double(b[i]) * b[i]; }
    return ab / std::sqrt(aa * bb);
}

double float_norm(const std::vector<float> & x) {
    double result = 0;
    for (float value : x) result += double(value) * value;
    return std::sqrt(result);
}

double float_delta_norm(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) throw std::runtime_error("capture width mismatch");
    double result = 0;
    for (size_t i = 0; i < a.size(); ++i) { const double d = double(a[i]) - b[i]; result += d * d; }
    return std::sqrt(result);
}

double float_max_delta(const std::vector<float> & a, const vec & b) {
    if (a.size() != b.size()) throw std::runtime_error("capture width mismatch");
    double result = 0;
    for (size_t i = 0; i < a.size(); ++i) result = std::max(result, std::abs(double(a[i]) - b[i]));
    return result;
}

logits_summary summarize_logits(const std::vector<float> & logits, int32_t expected) {
    if (expected < 0 || expected >= int32_t(logits.size())) throw std::runtime_error("expected token outside vocabulary");
    const float maximum = *std::max_element(logits.begin(), logits.end());
    double sum = 0;
    int32_t rank = 1;
    for (float value : logits) sum += std::exp(double(value) - maximum);
    for (float value : logits) if (value > logits[expected]) ++rank;
    const int32_t top1 = int32_t(std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
    const double logp = double(logits[expected]) - maximum - std::log(sum);
    if (!std::isfinite(logp)) throw std::runtime_error("non-finite expected-token log probability");
    return { logp, rank, top1 };
}

double kl_divergence(const std::vector<float> & native, const std::vector<float> & perturbed) {
    if (native.size() != perturbed.size()) throw std::runtime_error("logit size mismatch");
    const double native_max = *std::max_element(native.begin(), native.end());
    const double perturbed_max = *std::max_element(perturbed.begin(), perturbed.end());
    double native_sum = 0, perturbed_sum = 0;
    for (size_t i = 0; i < native.size(); ++i) { native_sum += std::exp(double(native[i]) - native_max); perturbed_sum += std::exp(double(perturbed[i]) - perturbed_max); }
    const double native_log_z = native_max + std::log(native_sum), perturbed_log_z = perturbed_max + std::log(perturbed_sum);
    double result = 0;
    for (size_t i = 0; i < native.size(); ++i) {
        const double log_p = native[i] - native_log_z, log_q = perturbed[i] - perturbed_log_z;
        result += std::exp(log_p) * (log_p - log_q);
    }
    return std::max(0.0, result);
}

double sign_flip_p_value(const std::vector<double> & deltas) {
    if (deltas.size() != k_test_carriers) throw std::runtime_error("sign-flip test requires 16 carriers");
    const double observed_sum = std::accumulate(deltas.begin(), deltas.end(), 0.0);
    uint64_t count = 0;
    for (uint64_t mask = 0; mask < (uint64_t(1) << deltas.size()); ++mask) {
        double permuted_sum = 0;
        for (size_t i = 0; i < deltas.size(); ++i) permuted_sum += ((mask >> i) & 1 ? 1.0 : -1.0) * deltas[i];
        if (permuted_sum >= observed_sum - 1e-15) ++count;
    }
    return double(count) / double(uint64_t(1) << deltas.size());
}

std::array<double, 2> holm_adjust(const std::array<double, 2> & raw) {
    const int first = raw[0] <= raw[1] ? 0 : 1, second = 1 - first;
    std::array<double, 2> adjusted{};
    adjusted[first] = std::min(1.0, 2.0 * raw[first]);
    adjusted[second] = std::max(adjusted[first], raw[second]);
    return adjusted;
}

std::string classify(double foreign_mean, double random_mean, double foreign_adjusted, double random_adjusted) {
    if (foreign_mean > 0 && random_mean > 0 && foreign_adjusted <= 0.05 && random_adjusted <= 0.05) {
        return "native_rotation_behaviorally_privileged";
    }
    if (foreign_mean <= 0 && random_mean <= 0) return "no_detected_native_rotation_privilege";
    return "ambiguous";
}

const llama_interp_activation & capture(const llama_interp::activation_set & captures, const std::string & name) {
    return rwkv_experiment::require_capture(captures, name);
}

std::vector<std::string> capture_names(int last_layer) {
    const std::string prefix = "rwkv.layer.45.";
    return { prefix + "resid.in", prefix + "time.out", prefix + "resid.time", prefix + "ffn.norm",
             prefix + "channel.out", prefix + "resid.out", "rwkv.layer." + std::to_string(last_layer) + ".resid.out" };
}

llama_interp::activation_set run_step(llama_interp::runtime & runtime, const llama_interp::rwkv_state & state,
                                      llama_token token, const std::vector<std::string> & taps,
                                      const std::vector<ggml_fp16_t> * correction = nullptr) {
    llama_interp::activation_set captures;
    auto operation = runtime.prefill_tokens(state, { token });
    if (correction) operation.perturb("^rwkv.layer.45.time.out$", llama_interp::runtime::add_head(-1, *correction));
    for (const std::string & tap : taps) operation.capture_f32("^" + tap + "$", captures);
    operation.discard_state();
    auto task = [&]() -> llama_interp::task<> { (void) co_await operation; }();
    runtime.run();
    task.rethrow_if_failed();
    return captures;
}

std::vector<float> readout(llama_context * ctx, const llama_interp::activation_set & captures, const std::string & final_tap,
                           int vocabulary_size) {
    std::vector<float> logits(vocabulary_size);
    if (!llama_interp_rwkv_final_readout(ctx, capture(captures, final_tap).data_f32.data(), 1, logits.data())) {
        throw std::runtime_error("final residual readout failed");
    }
    if (!std::all_of(logits.begin(), logits.end(), [](float x) { return std::isfinite(x); })) throw std::runtime_error("non-finite final logits");
    return logits;
}

std::vector<ggml_fp16_t> correction_for(const native_sample & native, const vec & target) {
    const vec y_target = endpoint(native.geom, target);
    std::vector<ggml_fp16_t> result(target.size());
    for (size_t i = 0; i < result.size(); ++i) {
        const double time_target = y_target[i] - native.resid_in[i];
        result[i] = ggml_fp32_to_fp16(float(time_target - native.time_out[i]));
    }
    return result;
}

native_sample make_native_sample(int carrier_id, const manifest_data & manifest, const llama_interp::activation_set & captures,
                                 const std::vector<float> & logits, const std::vector<std::string> & taps) {
    native_sample sample;
    sample.carrier_id = carrier_id;
    sample.carrier_token_id = manifest.carrier_tokens[carrier_id];
    sample.expected_token_id = manifest.value_tokens[k_value_id];
    sample.read_token_id = manifest.read_token_id;
    sample.read_position = manifest.read_position;
    sample.resid_in = capture(captures, "rwkv.layer.45.resid.in").data_f32;
    sample.time_out = capture(captures, "rwkv.layer.45.time.out").data_f32;
    sample.resid_time = capture(captures, "rwkv.layer.45.resid.time").data_f32;
    sample.geom = make_geometry(sample.resid_in, sample.resid_time);
    if (sample.geom.theta < k_angle_epsilon) throw std::runtime_error("native angle below classified threshold");
    double identity = 0;
    for (size_t i = 0; i < sample.resid_in.size(); ++i) identity = std::max(identity, std::abs(double(sample.resid_in[i]) + sample.time_out[i] - sample.resid_time[i]));
    if (identity > 1e-3) throw std::runtime_error("native residual identity failed");
    const logits_summary summary = summarize_logits(logits, sample.expected_token_id);
    sample.log_probability = summary.log_probability;
    sample.rank = summary.rank;
    sample.top1 = summary.top1;
    for (const std::string & tap : taps) (void) capture(captures, tap);
    return sample;
}

intervention_row evaluate_intervention(llama_interp::runtime & runtime, llama_context * ctx,
                                       const llama_interp::rwkv_state & state, llama_token read_token,
                                       const std::vector<std::string> & taps, const native_sample & native,
                                       const llama_interp::activation_set & native_captures,
                                       const std::vector<float> & native_logits, const vec & target,
                                       std::string condition, int control_index, int donor, uint64_t random_seed) {
    const auto correction = correction_for(native, target);
    const llama_interp::activation_set actual = run_step(runtime, state, read_token, taps, &correction);
    const std::vector<float> logits = readout(ctx, actual, taps.back(), int(native_logits.size()));
    const logits_summary summary = summarize_logits(logits, native.expected_token_id);
    const vec actual_y = to_double(capture(actual, "rwkv.layer.45.resid.time").data_f32);
    const vec actual_direction = unit(center(actual_y));
    const vec requested_y = endpoint(native.geom, target);
    const vec native_y = to_double(native.resid_time);
    const double actual_angle = angle(actual_direction, native.geom.v);
    const double requested_angle = angle(target, native.geom.v);
    intervention_row row;
    row.carrier_id = native.carrier_id;
    row.carrier_token_id = native.carrier_token_id;
    row.expected_token_id = native.expected_token_id;
    row.condition = std::move(condition);
    row.control_index = control_index;
    row.donor_carrier_id = donor;
    row.random_seed = random_seed;
    row.native_log_probability = native.log_probability;
    row.perturbed_log_probability = summary.log_probability;
    row.behavior_loss = native.log_probability - summary.log_probability;
    row.expected_rank = summary.rank;
    row.top1_token_id = summary.top1;
    row.top1_correct = summary.top1 == native.expected_token_id;
    row.kl_native_perturbed = kl_divergence(native_logits, logits);
    row.requested_angle = requested_angle;
    row.actual_angle = actual_angle;
    row.requested_endpoint_distance = distance(requested_y, native_y);
    row.actual_endpoint_distance = distance(actual_y, native_y);
    row.mean_error = std::abs(mean(actual_y) - native.geom.mean);
    row.radius_relative_error = std::abs(norm(center(actual_y)) - native.geom.radius) / std::max(native.geom.radius, k_norm_epsilon);
    row.angle_error = std::abs(actual_angle - requested_angle);
    row.unrotation_error = row.condition == "native_unrotation" ? angle(actual_direction, native.geom.u) : -1;
    row.time_out_reconstruction_error = float_max_delta(capture(actual, "rwkv.layer.45.time.out").data_f32,
        [&]() { vec expected(native.time_out.size()); for (size_t i = 0; i < expected.size(); ++i) expected[i] = double(native.time_out[i]) + ggml_fp16_to_fp32(correction[i]); return expected; }());
    row.ffn_norm_delta_norm = float_delta_norm(capture(actual, "rwkv.layer.45.ffn.norm").data_f32,
                                               capture(native_captures, "rwkv.layer.45.ffn.norm").data_f32);
    row.channel_out_delta_norm = float_delta_norm(capture(actual, "rwkv.layer.45.channel.out").data_f32,
                                                  capture(native_captures, "rwkv.layer.45.channel.out").data_f32);
    row.resid_out_delta_norm = float_delta_norm(capture(actual, "rwkv.layer.45.resid.out").data_f32,
                                                capture(native_captures, "rwkv.layer.45.resid.out").data_f32);
    row.numerical_pass = std::isfinite(row.behavior_loss) && std::isfinite(row.kl_native_perturbed) &&
        row.mean_error <= k_endpoint_tolerance && row.radius_relative_error <= k_endpoint_tolerance &&
        row.angle_error <= k_endpoint_tolerance && (row.unrotation_error < 0 || row.unrotation_error <= k_endpoint_tolerance);
    return row;
}

void self_test() {
    const std::vector<float> a{ 1, -1, 0, 0 }, y{ 0, -1, 1, 0 };
    const geometry g = make_geometry(a, y);
    const vec unrotated = endpoint(g, g.u);
    if (std::abs(mean(unrotated) - g.mean) > 1e-14 || std::abs(norm(center(unrotated)) - g.radius) > 1e-14 ||
        angle(unit(center(unrotated)), g.u) > 1e-7) throw std::runtime_error("geometry/unrotation self-test failed");
    geometry donor = g;
    donor.v = unit(vec{ 1, 0, -1, 0 });
    donor.h = unit(vec{ 0, 1, 0, -1 });
    const vec transported = transport_tangent(donor, g);
    if (std::abs(dot(transported, g.v)) > 1e-12 || std::abs(norm(transported) - 1) > 1e-12) throw std::runtime_error("transport self-test failed");
    const vec foreign_endpoint = direction_at_angle(g, transported);
    if (std::abs(angle(foreign_endpoint, g.v) - g.theta) > 1e-12) throw std::runtime_error("foreign endpoint self-test failed");
    const auto random1 = random_tangent(g, 32, 0), random2 = random_tangent(g, 32, 0);
    if (random1.second != random2.second || distance(random1.first, random2.first) != 0 ||
        std::abs(dot(random1.first, g.v)) > 1e-12 || std::abs(angle(direction_at_angle(g, random1.first), g.v) - g.theta) > 1e-12) {
        throw std::runtime_error("random control self-test failed");
    }
    const std::vector<int> donors32 = donor_order(32);
    const std::vector<int> donors32_repeat = donor_order(32);
    const std::vector<int> donors33 = donor_order(33);
    if (donors32 != donors32_repeat || donors32 == donors33 ||
        std::any_of(donors32.begin(), donors32.end(), [](int id) { return id < 0 || id >= 32; })) {
        throw std::runtime_error("donor selection self-test failed");
    }
    const std::vector<float> logits{ -2, 0, 1, 4 };
    const logits_summary summary = summarize_logits(logits, 3);
    if (summary.rank != 1 || summary.top1 != 3 || !(summary.log_probability < 0) || std::abs(kl_divergence(logits, logits)) > 1e-14) {
        throw std::runtime_error("logit metric self-test failed");
    }
    const std::vector<double> positive(16, 1.0), zero(16, 0.0);
    if (std::abs(sign_flip_p_value(positive) - 1.0 / 65536.0) > 1e-15 || sign_flip_p_value(zero) != 1.0) {
        throw std::runtime_error("sign-flip self-test failed");
    }
    const auto adjusted = holm_adjust({ 0.01, 0.04 });
    if (adjusted[0] != 0.02 || adjusted[1] != 0.04 ||
        classify(1, 1, 0.01, 0.02) != "native_rotation_behaviorally_privileged" ||
        classify(0, -1, 1, 1) != "no_detected_native_rotation_privilege" ||
        classify(1, -1, 1, 1) != "ambiguous") throw std::runtime_error("Holm/outcome self-test failed");
    bool suppressed = false;
    try { (void) sign_flip_p_value(std::vector<double>(15, 1)); } catch (const std::runtime_error &) { suppressed = true; }
    if (!suppressed) throw std::runtime_error("missing-row suppression self-test failed");
}

std::vector<std::string> read_prompts(const std::string & path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot read prompt corpus");
    std::vector<std::string> result;
    for (std::string line; std::getline(input, line);) result.push_back(line);
    if (result.size() != 768) throw std::runtime_error("prompt corpus does not contain exactly 768 rows");
    return result;
}

void validate_prompt(const std::vector<llama_token> & tokens, int carrier_id, const manifest_data & manifest) {
    if (tokens.size() != size_t(manifest.token_count) || tokens.size() <= size_t(std::max(manifest.carrier_position, manifest.value_position)) ||
        tokens[manifest.read_position] != manifest.read_token_id || tokens[manifest.carrier_position] != manifest.carrier_tokens[carrier_id] ||
        tokens[manifest.value_position] != manifest.value_tokens[k_value_id]) throw std::runtime_error("prompt tokenization differs from frozen manifest");
}

void write_calibration_json(const args & a, const std::string & model_hash, const std::string & prompts_hash,
                            const std::string & manifest_hash, const std::string & registration_hash,
                            double repeat_cosine, double repeat_relative_norm, double alpha0_cosine,
                            double alpha0_relative_norm, const std::array<double, 4> & distances,
                            const std::array<double, 4> & losses) {
    std::ofstream out(a.output_json);
    if (!out) throw std::runtime_error("cannot create calibration JSON");
    out << std::setprecision(17) << "{\n  \"schema_version\": 1,\n  \"status\": \"calibration_only\",\n"
        << "  \"classification\": null,\n  \"model_sha256\": \"" << model_hash << "\",\n"
        << "  \"corpus_sha256\": \"" << prompts_hash << "\",\n  \"manifest_sha256\": \"" << manifest_hash
        << "\",\n  \"registration_sha256\": \"" << registration_hash << "\",\n  \"seed\": " << k_seed
        << ",\n  \"layer\": 45,\n  \"carrier_id\": 0,\n  \"value_id\": 0,\n"
        << "  \"native_repeat_cosine\": " << repeat_cosine << ",\n  \"native_repeat_relative_norm_difference\": " << repeat_relative_norm
        << ",\n  \"alpha0_cosine\": " << alpha0_cosine << ",\n  \"alpha0_relative_norm_difference\": " << alpha0_relative_norm
        << ",\n  \"points\": [";
    const std::array<double, 4> alphas{ 0, 0.25, 0.5, 1 };
    for (size_t i = 0; i < alphas.size(); ++i) {
        if (i) out << ',';
        out << "{\"alpha\":" << alphas[i] << ",\"endpoint_distance\":" << distances[i] << ",\"behavior_loss\":" << losses[i] << '}';
    }
    out << "],\n  \"all_gates_passed\": true,\n  \"protocol_deviations\": []\n}\n";
}

void write_root(const std::string & path, const std::map<std::string, std::string> & metadata,
                const std::vector<native_sample> & natives, const std::vector<intervention_row> & rows,
                const std::vector<carrier_metric> & carriers, const std::array<test_result, 2> & tests,
                const std::string & classification) {
    {
        auto model = ROOT::RNTupleModel::Create();
        auto carrier = model->MakeField<int32_t>("carrier_id");
        auto carrier_token = model->MakeField<int32_t>("carrier_token_id");
        auto expected = model->MakeField<int32_t>("expected_token_id");
        auto read_token = model->MakeField<int32_t>("read_token_id");
        auto read_position = model->MakeField<int32_t>("read_position");
        auto theta = model->MakeField<double>("theta");
        auto mu = model->MakeField<double>("post_tmix_mean");
        auto radius = model->MakeField<double>("post_tmix_centered_radius");
        auto logp = model->MakeField<double>("expected_log_probability");
        auto rank = model->MakeField<int32_t>("expected_rank");
        auto top1 = model->MakeField<int32_t>("top1_token_id");
        auto resid_in = model->MakeField<std::vector<float>>("resid_in");
        auto time_out = model->MakeField<std::vector<float>>("time_out");
        auto resid_time = model->MakeField<std::vector<float>>("resid_time");
        auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "native_samples", path);
        for (const auto & row : natives) {
            *carrier = row.carrier_id; *carrier_token = row.carrier_token_id; *expected = row.expected_token_id;
            *read_token = row.read_token_id; *read_position = row.read_position; *theta = row.geom.theta; *mu = row.geom.mean;
            *radius = row.geom.radius; *logp = row.log_probability; *rank = row.rank; *top1 = row.top1;
            *resid_in = row.resid_in; *time_out = row.time_out; *resid_time = row.resid_time; writer->Fill();
        }
        writer->CommitCluster();
    }
    {
        TFile file(path.c_str(), "UPDATE");
        auto model = ROOT::RNTupleModel::Create();
#define FIELD(type, name) auto name = model->MakeField<type>(#name)
        FIELD(int32_t, carrier_id); FIELD(int32_t, carrier_token_id); FIELD(int32_t, value_id); FIELD(int32_t, expected_token_id);
        FIELD(std::string, condition); FIELD(int32_t, control_index); FIELD(int32_t, donor_carrier_id); FIELD(uint64_t, random_seed);
        FIELD(double, native_log_probability); FIELD(double, perturbed_log_probability); FIELD(double, behavior_loss);
        FIELD(int32_t, expected_rank); FIELD(int32_t, top1_token_id); FIELD(bool, top1_correct); FIELD(double, kl_native_perturbed);
        FIELD(double, requested_angle); FIELD(double, actual_angle); FIELD(double, requested_endpoint_distance); FIELD(double, actual_endpoint_distance);
        FIELD(double, mean_error); FIELD(double, radius_relative_error); FIELD(double, angle_error); FIELD(double, unrotation_error);
        FIELD(double, time_out_reconstruction_error); FIELD(double, ffn_norm_delta_norm); FIELD(double, channel_out_delta_norm);
        FIELD(double, resid_out_delta_norm); FIELD(bool, numerical_pass);
#undef FIELD
        auto writer = ROOT::RNTupleWriter::Append(std::move(model), "interventions", file);
        int previous_carrier = -1;
        for (const auto & row : rows) {
            if (previous_carrier >= 0 && previous_carrier != row.carrier_id) writer->CommitCluster();
            previous_carrier = row.carrier_id;
#define SET(name) *name = row.name
            SET(carrier_id); SET(carrier_token_id); SET(value_id); SET(expected_token_id); SET(condition); SET(control_index);
            SET(donor_carrier_id); SET(random_seed); SET(native_log_probability); SET(perturbed_log_probability); SET(behavior_loss);
            SET(expected_rank); SET(top1_token_id); SET(top1_correct); SET(kl_native_perturbed); SET(requested_angle); SET(actual_angle);
            SET(requested_endpoint_distance); SET(actual_endpoint_distance); SET(mean_error); SET(radius_relative_error); SET(angle_error);
            SET(unrotation_error); SET(time_out_reconstruction_error); SET(ffn_norm_delta_norm); SET(channel_out_delta_norm);
            SET(resid_out_delta_norm); SET(numerical_pass);
#undef SET
            writer->Fill();
        }
        writer->CommitCluster();
    }
    {
        TFile file(path.c_str(), "UPDATE");
        auto model = ROOT::RNTupleModel::Create();
        auto carrier = model->MakeField<int32_t>("carrier_id");
        auto native = model->MakeField<double>("native_unrotation_loss");
        auto foreign = model->MakeField<double>("foreign_mean_loss");
        auto random = model->MakeField<double>("random_mean_loss");
        auto delta_foreign = model->MakeField<double>("delta_foreign");
        auto delta_random = model->MakeField<double>("delta_random");
        auto writer = ROOT::RNTupleWriter::Append(std::move(model), "carrier_metrics", file);
        for (const auto & row : carriers) { *carrier = row.carrier_id; *native = row.native_unrotation_loss; *foreign = row.foreign_mean_loss;
            *random = row.random_mean_loss; *delta_foreign = row.delta_foreign; *delta_random = row.delta_random; writer->Fill(); }
        writer->CommitCluster();
    }
    {
        TFile file(path.c_str(), "UPDATE");
        auto model = ROOT::RNTupleModel::Create();
        auto family = model->MakeField<std::string>("control_family");
        auto mean_delta = model->MakeField<double>("mean_delta");
        auto raw_p = model->MakeField<double>("raw_p_value");
        auto adjusted_p = model->MakeField<double>("holm_adjusted_p_value");
        auto writer = ROOT::RNTupleWriter::Append(std::move(model), "statistical_tests", file);
        for (const auto & row : tests) { *family = row.family; *mean_delta = row.mean_delta; *raw_p = row.raw_p; *adjusted_p = row.adjusted_p; writer->Fill(); }
        writer->CommitCluster();
    }
    {
        TFile file(path.c_str(), "UPDATE");
        auto model = ROOT::RNTupleModel::Create();
        auto predicate = model->MakeField<std::string>("predicate");
        auto pass = model->MakeField<bool>("pass");
        auto writer = ROOT::RNTupleWriter::Append(std::move(model), "outcome_predicates", file);
        const std::array<std::pair<std::string, bool>, 7> predicates{{
            { "foreign_mean_positive", tests[0].mean_delta > 0 }, { "random_mean_positive", tests[1].mean_delta > 0 },
            { "foreign_holm_significant", tests[0].adjusted_p <= 0.05 }, { "random_holm_significant", tests[1].adjusted_p <= 0.05 },
            { "both_means_non_positive", tests[0].mean_delta <= 0 && tests[1].mean_delta <= 0 },
            { "classification_privileged", classification == "native_rotation_behaviorally_privileged" },
            { "classification_no_detected_privilege", classification == "no_detected_native_rotation_privilege" } }};
        for (const auto & row : predicates) { *predicate = row.first; *pass = row.second; writer->Fill(); }
        writer->CommitCluster();
    }
    {
        TFile file(path.c_str(), "UPDATE");
        auto model = ROOT::RNTupleModel::Create();
        auto check = model->MakeField<std::string>("check_name");
        auto observed = model->MakeField<double>("observed_maximum");
        auto threshold = model->MakeField<double>("threshold");
        auto pass = model->MakeField<bool>("pass");
        auto writer = ROOT::RNTupleWriter::Append(std::move(model), "numerical_checks", file);
        const auto maximum = [&rows](auto member) { double value = 0; for (const auto & row : rows) value = std::max(value, row.*member); return value; };
        const std::array<std::tuple<std::string, double, double>, 4> checks{{
            { "post_tmix_mean_error", maximum(&intervention_row::mean_error), k_endpoint_tolerance },
            { "post_tmix_radius_relative_error", maximum(&intervention_row::radius_relative_error), k_endpoint_tolerance },
            { "endpoint_angle_error", maximum(&intervention_row::angle_error), k_endpoint_tolerance },
            { "native_unrotation_error", [&rows]() { double x = 0; for (const auto & r : rows) if (r.unrotation_error >= 0) x = std::max(x, r.unrotation_error); return x; }(), k_endpoint_tolerance } }};
        for (const auto & row : checks) { *check = std::get<0>(row); *observed = std::get<1>(row); *threshold = std::get<2>(row); *pass = *observed <= *threshold; writer->Fill(); }
        writer->CommitCluster();
    }
    {
        TFile file(path.c_str(), "UPDATE");
        auto model = ROOT::RNTupleModel::Create();
        auto key = model->MakeField<std::string>("key");
        auto value = model->MakeField<std::string>("value");
        auto writer = ROOT::RNTupleWriter::Append(std::move(model), "metadata", file);
        for (const auto & row : metadata) { *key = row.first; *value = row.second; writer->Fill(); }
        writer->CommitCluster();
    }
}

bool audit_root(const std::string & path, std::vector<carrier_metric> & audited_carriers,
                std::array<test_result, 2> & audited_tests, std::string & audited_classification,
                bool & all_numerical_pass) {
    auto native = ROOT::RNTupleReader::Open("native_samples", path);
    auto interventions = ROOT::RNTupleReader::Open("interventions", path);
    auto carriers = ROOT::RNTupleReader::Open("carrier_metrics", path);
    auto tests = ROOT::RNTupleReader::Open("statistical_tests", path);
    auto predicates = ROOT::RNTupleReader::Open("outcome_predicates", path);
    auto checks = ROOT::RNTupleReader::Open("numerical_checks", path);
    auto metadata = ROOT::RNTupleReader::Open("metadata", path);
    if (native->GetNEntries() != 48 || interventions->GetNEntries() != 112 || carriers->GetNEntries() != 16 ||
        tests->GetNEntries() != 2 || predicates->GetNEntries() != 7 || checks->GetNEntries() != 4 || metadata->GetNEntries() < 10) return false;
    auto carrier = interventions->GetView<int32_t>("carrier_id");
    auto condition = interventions->GetView<std::string>("condition");
    auto control_index = interventions->GetView<int32_t>("control_index");
    auto donor = interventions->GetView<int32_t>("donor_carrier_id");
    auto loss = interventions->GetView<double>("behavior_loss");
    auto numerical = interventions->GetView<bool>("numerical_pass");
    std::map<int, std::vector<double>> native_loss, foreign_loss, random_loss;
    std::set<std::tuple<int, std::string, int>> seen;
    all_numerical_pass = true;
    for (auto entry : interventions->GetEntryRange()) {
        const int c = carrier(entry), index = control_index(entry);
        const std::string family = condition(entry);
        if (c < 32 || c >= 48 || !std::isfinite(loss(entry)) || !seen.emplace(c, family, index).second) return false;
        all_numerical_pass = all_numerical_pass && numerical(entry);
        if (family == "native_unrotation" && index == -1 && donor(entry) == -1) native_loss[c].push_back(loss(entry));
        else if (family == "foreign" && index >= 0 && index < 3 && donor(entry) >= 0 && donor(entry) < 32) foreign_loss[c].push_back(loss(entry));
        else if (family == "random" && index >= 0 && index < 3 && donor(entry) == -1) random_loss[c].push_back(loss(entry));
        else return false;
    }
    audited_carriers.clear();
    std::vector<double> foreign_deltas, random_deltas;
    for (int c = 32; c < 48; ++c) {
        if (native_loss[c].size() != 1 || foreign_loss[c].size() != 3 || random_loss[c].size() != 3) return false;
        const double f = std::accumulate(foreign_loss[c].begin(), foreign_loss[c].end(), 0.0) / 3;
        const double r = std::accumulate(random_loss[c].begin(), random_loss[c].end(), 0.0) / 3;
        audited_carriers.push_back({ c, native_loss[c][0], f, r, native_loss[c][0] - f, native_loss[c][0] - r });
        foreign_deltas.push_back(native_loss[c][0] - f); random_deltas.push_back(native_loss[c][0] - r);
    }
    const std::array<double, 2> raw{ sign_flip_p_value(foreign_deltas), sign_flip_p_value(random_deltas) };
    const auto adjusted = holm_adjust(raw);
    audited_tests = {{ { "foreign", std::accumulate(foreign_deltas.begin(), foreign_deltas.end(), 0.0) / 16, raw[0], adjusted[0] },
                       { "random", std::accumulate(random_deltas.begin(), random_deltas.end(), 0.0) / 16, raw[1], adjusted[1] } }};
    audited_classification = classify(audited_tests[0].mean_delta, audited_tests[1].mean_delta, adjusted[0], adjusted[1]);
    auto check_pass = checks->GetView<bool>("pass");
    for (auto entry : checks->GetEntryRange()) all_numerical_pass = all_numerical_pass && check_pass(entry);
    return true;
}

void write_classified_json(const args & a, const std::map<std::string, std::string> & hashes,
                           const std::vector<carrier_metric> & carriers, const std::array<test_result, 2> & tests,
                           const std::string & classification, bool valid) {
    const std::string claim = classification == "native_rotation_behaviorally_privileged" ?
        "At layer 45 for this fixed controlled value and held-out carrier panel, native unrotation damaged expected completion probability more than matched foreign and random rotations. The native direction was behaviorally special." :
        classification == "no_detected_native_rotation_privilege" ?
        "No causal privilege of native unrotation was detected beyond generic matched directional sensitivity at this layer and controlled value." :
        "The gate did not distinguish native directional privilege from generic sensitivity.";
    std::ofstream out(a.output_json);
    if (!out) throw std::runtime_error("cannot create classified JSON");
    out << std::setprecision(17) << "{\n  \"schema_version\": 1,\n  \"status\": \"" << (valid ? "valid" : "invalid")
        << "\",\n  \"classification\": ";
    if (valid) out << '"' << classification << '"'; else out << "null";
    out << ",\n  \"diagnostic_unclassified_outcome\": \"" << classification
        << "\",\n  \"seed\": " << k_seed << ",\n  \"layer\": 45,\n  \"value_id\": 0,\n  \"train_carriers\": 32,\n"
        << "  \"test_carriers\": 16,\n  \"foreign_controls_per_carrier\": 3,\n  \"random_controls_per_carrier\": 3,\n"
        << "  \"hashes\": {";
    bool first = true;
    for (const auto & row : hashes) { if (!first) out << ','; out << "\n    \"" << row.first << "\": \"" << row.second << '"'; first = false; }
    out << "\n  },\n  \"carrier_metrics\": [";
    for (size_t i = 0; i < carriers.size(); ++i) {
        if (i) out << ',';
        const auto & row = carriers[i];
        out << "\n    {\"carrier_id\":" << row.carrier_id << ",\"native_unrotation_loss\":" << row.native_unrotation_loss
            << ",\"foreign_mean_loss\":" << row.foreign_mean_loss << ",\"random_mean_loss\":" << row.random_mean_loss
            << ",\"delta_foreign\":" << row.delta_foreign << ",\"delta_random\":" << row.delta_random << '}';
    }
    out << "\n  ],\n  \"statistical_tests\": [";
    for (size_t i = 0; i < tests.size(); ++i) { if (i) out << ','; const auto & row = tests[i];
        out << "{\"control_family\":\"" << row.family << "\",\"mean_delta\":" << row.mean_delta << ",\"raw_p_value\":" << row.raw_p
            << ",\"holm_adjusted_p_value\":" << row.adjusted_p << '}'; }
    out << "],\n  \"outcome_predicates\": {\"both_means_positive\":" << (tests[0].mean_delta > 0 && tests[1].mean_delta > 0 ? "true" : "false")
        << ",\"both_holm_significant\":" << (tests[0].adjusted_p <= 0.05 && tests[1].adjusted_p <= 0.05 ? "true" : "false")
        << ",\"both_means_non_positive\":" << (tests[0].mean_delta <= 0 && tests[1].mean_delta <= 0 ? "true" : "false") << "},\n"
        << "  \"numerical_checks\": {\"all_interventions_passed\":" << (valid ? "true" : "false") << ",\"post_write_audit_passed\":true},\n"
        << "  \"root_row_counts\": {\"metadata\":11,\"native_samples\":48,\"interventions\":112,\"carrier_metrics\":16,\"statistical_tests\":2,\"outcome_predicates\":7,\"numerical_checks\":4},\n"
        << "  \"permitted_claim\": "; rwkv_experiment::write_json_string(out, valid ? claim : "None; numerical validation failed and classification is suppressed.");
    out << ",\n  \"protocol_deviations\": []\n}\n";
}

} // namespace

int main(int argc, char ** argv) try {
    llama_log_set(quiet_llama_logs, nullptr);
    const args a = parse_args(argc, argv);
    self_test();
    if (a.run_mode == args::mode::self_test) { std::puts("self-tests=passed"); return 0; }

    std::string model_hash, prompts_hash, manifest_hash, registration_hash;
    const manifest_data manifest = verify_frozen_inputs(a, model_hash, prompts_hash, manifest_hash, registration_hash);
    const std::vector<std::string> prompts = read_prompts(a.prompts);
    ggml_backend_load_all();
    llama_backend_init();
    llama_model_params params = llama_model_default_params(); params.n_gpu_layers = a.n_gpu_layers;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(a.model.c_str(), params), llama_model_free);
    if (!model || model->arch != LLM_ARCH_RWKV7 || model->hparams.n_layer() <= k_layer) throw std::runtime_error("the frozen RWKV-7 model is required");
    llama_context_params cp = llama_context_default_params(); cp.n_ctx = 64; cp.n_batch = 64; cp.n_ubatch = 64;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
    if (!ctx) throw std::runtime_error("context creation failed");
    llama_interp::runtime runtime(ctx.get(), 1);
    const llama_vocab * vocab = llama_model_get_vocab(model.get());
    const int vocabulary_size = llama_vocab_n_tokens(vocab);
    const std::vector<std::string> taps = capture_names(model->hparams.n_layer() - 1);

    const auto prepare = [&](int carrier_id) {
        const std::vector<llama_token> tokens = common_tokenize(vocab, prompts[size_t(carrier_id) * 16], false, false);
        validate_prompt(tokens, carrier_id, manifest);
        llama_interp::rwkv_state state = runtime.make_state();
        const std::vector<llama_token> prefix(tokens.begin(), tokens.begin() + manifest.read_position);
        auto task = [&]() -> llama_interp::task<> { state = co_await runtime.prefill_tokens(state, prefix); }();
        runtime.run(); task.rethrow_if_failed();
        return std::make_pair(std::move(state), tokens[manifest.read_position]);
    };

    if (a.run_mode == args::mode::calibration) {
        auto [state, read_token] = prepare(0);
        const auto native_capture = run_step(runtime, state, read_token, taps);
        const auto native_logits = readout(ctx.get(), native_capture, taps.back(), vocabulary_size);
        const native_sample native = make_native_sample(0, manifest, native_capture, native_logits, taps);
        const auto repeat_capture = run_step(runtime, state, read_token, taps);
        const auto repeat_logits = readout(ctx.get(), repeat_capture, taps.back(), vocabulary_size);
        double repeat_cosine = 1, repeat_relative_norm = 0;
        for (const std::string & tap : taps) {
            const auto & x = capture(native_capture, tap).data_f32; const auto & y = capture(repeat_capture, tap).data_f32;
            repeat_cosine = std::min(repeat_cosine, vector_cosine(x, y));
            repeat_relative_norm = std::max(repeat_relative_norm, std::abs(float_norm(x) - float_norm(y)) / std::max(float_norm(x), k_norm_epsilon));
        }
        if (repeat_cosine < k_repeat_cosine || repeat_relative_norm > k_repeat_relative_norm ||
            !std::isfinite(summarize_logits(repeat_logits, native.expected_token_id).log_probability)) throw std::runtime_error("native repeat calibration gate failed");
        std::array<double, 4> distances{}, losses{};
        double alpha0_cosine = 0, alpha0_relative_norm = 0;
        const std::array<double, 4> alphas{ 0, 0.25, 0.5, 1 };
        for (size_t i = 0; i < alphas.size(); ++i) {
            const vec target = direction_at_angle(native.geom, native.geom.h, alphas[i]);
            const intervention_row row = evaluate_intervention(runtime, ctx.get(), state, read_token, taps, native, native_capture,
                native_logits, target, alphas[i] == 1 ? "native_unrotation" : "calibration", int(i), -1, 0);
            distances[i] = row.actual_endpoint_distance; losses[i] = row.behavior_loss;
            if (i == 0) {
                const auto correction = correction_for(native, target);
                const auto alpha0 = run_step(runtime, state, read_token, taps, &correction);
                alpha0_cosine = 1; alpha0_relative_norm = 0;
                for (const std::string & tap : taps) {
                    const auto & x = capture(native_capture, tap).data_f32; const auto & y = capture(alpha0, tap).data_f32;
                    alpha0_cosine = std::min(alpha0_cosine, vector_cosine(x, y));
                    alpha0_relative_norm = std::max(alpha0_relative_norm, std::abs(float_norm(x) - float_norm(y)) / std::max(float_norm(x), k_norm_epsilon));
                }
            }
        }
        for (size_t i = 1; i < distances.size(); ++i) if (distances[i] + k_endpoint_tolerance < distances[i - 1]) throw std::runtime_error("calibration endpoint distance is not non-decreasing");
        if (alpha0_cosine < k_repeat_cosine || alpha0_relative_norm > k_repeat_relative_norm) throw std::runtime_error("alpha=0 calibration reproduction failed");
        write_calibration_json(a, model_hash, prompts_hash, manifest_hash, registration_hash, repeat_cosine, repeat_relative_norm,
                               alpha0_cosine, alpha0_relative_norm, distances, losses);
        std::printf("status=calibration_only all_gates_passed=true output=%s\n", a.output_json.c_str());
    } else {
        std::array<native_sample, k_train_carriers> train{};
        std::vector<native_sample> natives;
        natives.reserve(48);
        for (int carrier_id = 0; carrier_id < k_train_carriers; ++carrier_id) {
            auto [state, read_token] = prepare(carrier_id);
            const auto captures = run_step(runtime, state, read_token, taps);
            const auto logits = readout(ctx.get(), captures, taps.back(), vocabulary_size);
            train[carrier_id] = make_native_sample(carrier_id, manifest, captures, logits, taps);
            natives.push_back(train[carrier_id]);
            std::printf("native_donor=%d/32\n", carrier_id + 1); std::fflush(stdout);
        }
        std::vector<intervention_row> rows;
        std::vector<carrier_metric> metrics;
        rows.reserve(112); metrics.reserve(16);
        for (int carrier_id = 32; carrier_id < 48; ++carrier_id) {
            auto [state, read_token] = prepare(carrier_id);
            const auto native_capture = run_step(runtime, state, read_token, taps);
            const auto native_logits = readout(ctx.get(), native_capture, taps.back(), vocabulary_size);
            const native_sample native = make_native_sample(carrier_id, manifest, native_capture, native_logits, taps);
            natives.push_back(native);
            rows.push_back(evaluate_intervention(runtime, ctx.get(), state, read_token, taps, native, native_capture, native_logits,
                                                 native.geom.u, "native_unrotation", -1, -1, 0));
            double foreign_sum = 0, random_sum = 0;
            const std::vector<int> donors = select_donors(native.geom, train, carrier_id);
            for (int control = 0; control < k_controls; ++control) {
                const vec h = transport_tangent(train[donors[control]].geom, native.geom);
                rows.push_back(evaluate_intervention(runtime, ctx.get(), state, read_token, taps, native, native_capture, native_logits,
                                                     direction_at_angle(native.geom, h), "foreign", control, donors[control], 0));
                foreign_sum += rows.back().behavior_loss;
            }
            for (int control = 0; control < k_controls; ++control) {
                auto [h, random_seed] = random_tangent(native.geom, carrier_id, control);
                rows.push_back(evaluate_intervention(runtime, ctx.get(), state, read_token, taps, native, native_capture, native_logits,
                                                     direction_at_angle(native.geom, h), "random", control, -1, random_seed));
                random_sum += rows.back().behavior_loss;
            }
            const double foreign_mean = foreign_sum / k_controls, random_mean = random_sum / k_controls;
            metrics.push_back({ carrier_id, rows[rows.size() - 7].behavior_loss, foreign_mean, random_mean,
                                rows[rows.size() - 7].behavior_loss - foreign_mean, rows[rows.size() - 7].behavior_loss - random_mean });
            const bool carrier_valid = std::all_of(rows.end() - 7, rows.end(), [](const intervention_row & row) { return row.numerical_pass; });
            std::printf("classified_carrier=%d/16 numerical_checks=%s\n", carrier_id - 31, carrier_valid ? "passed" : "failed"); std::fflush(stdout);
        }
        std::vector<double> foreign_deltas, random_deltas;
        for (const auto & row : metrics) { foreign_deltas.push_back(row.delta_foreign); random_deltas.push_back(row.delta_random); }
        const std::array<double, 2> raw{ sign_flip_p_value(foreign_deltas), sign_flip_p_value(random_deltas) };
        const auto adjusted = holm_adjust(raw);
        std::array<test_result, 2> tests{{
            { "foreign", std::accumulate(foreign_deltas.begin(), foreign_deltas.end(), 0.0) / 16, raw[0], adjusted[0] },
            { "random", std::accumulate(random_deltas.begin(), random_deltas.end(), 0.0) / 16, raw[1], adjusted[1] } }};
        const std::string classification = classify(tests[0].mean_delta, tests[1].mean_delta, tests[0].adjusted_p, tests[1].adjusted_p);
        const bool valid = std::all_of(rows.begin(), rows.end(), [](const intervention_row & row) { return row.numerical_pass; });
        const std::map<std::string, std::string> metadata{
            { "status", valid ? "valid" : "invalid" }, { "classification", valid ? classification : "suppressed" }, { "model_sha256", model_hash }, { "corpus_sha256", prompts_hash },
            { "manifest_sha256", manifest_hash }, { "registration_sha256", registration_hash }, { "backend", "Vulkan" }, { "n_gpu_layers", "99" },
            { "seed", std::to_string(k_seed) }, { "layer", "45" }, { "value_id", "0" } };
        write_root(a.output_root, metadata, natives, rows, metrics, tests, valid ? classification : "suppressed");
        std::vector<carrier_metric> audited_metrics;
        std::array<test_result, 2> audited_tests;
        std::string audited_classification;
        bool audited_numerical_pass = false;
        if (!audit_root(a.output_root, audited_metrics, audited_tests, audited_classification, audited_numerical_pass) || audited_classification != classification ||
            audited_numerical_pass != valid ||
            audited_metrics.size() != metrics.size() || std::abs(audited_tests[0].raw_p - tests[0].raw_p) > 1e-15 ||
            std::abs(audited_tests[1].raw_p - tests[1].raw_p) > 1e-15) throw std::runtime_error("post-write independent ROOT audit failed");
        const std::map<std::string, std::string> hashes{{ "model_sha256", model_hash }, { "corpus_sha256", prompts_hash },
            { "manifest_sha256", manifest_hash }, { "registration_sha256", registration_hash }, { "output_root_sha256", sha256_file(a.output_root) }};
        write_classified_json(a, hashes, audited_metrics, audited_tests, audited_classification, valid);
        std::printf("status=%s classification=%s diagnostic_outcome=%s foreign_mean_delta=%.17g foreign_adjusted_p=%.17g random_mean_delta=%.17g random_adjusted_p=%.17g\n",
                    valid ? "valid" : "invalid", valid ? classification.c_str() : "suppressed", classification.c_str(),
                    tests[0].mean_delta, tests[0].adjusted_p, tests[1].mean_delta, tests[1].adjusted_p);
    }
    ctx.reset(); model.reset(); llama_backend_free(); return 0;
} catch (const std::exception & error) {
    std::fprintf(stderr, "error: %s\n", error.what());
    llama_backend_free();
    return 1;
}
