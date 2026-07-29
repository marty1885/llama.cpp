#include <cblas.h>
#include <omp.h>
#include <openblas/lapacke.h>
#include <TFile.h>
#include <TMatrixD.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <numeric>
#include <random>
#include <regex>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleReader.hxx>
#include <ROOT/RNTupleWriter.hxx>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {
constexpr int      k_layer     = 15;
constexpr int      k_rank      = 32;
constexpr int      k_repeats   = 99;
constexpr uint64_t k_seed      = 817263;
constexpr double   k_min_norm  = 1e-12;
constexpr double   k_min_angle = 1e-6;

struct args {
    std::string input, corpus, manifest, registration, output_root, output_json;
    bool        development = false, exploratory = false, overwrite = false, self_test = false;
    int         threads = 0, layer = k_layer;
};

struct row {
    uint64_t           prompt_id = 0;
    bool               train     = false;
    int32_t            position = 0, token_id = 0;
    std::vector<float> x, w, y, u, tangent;
    double             radius = 0, write_norm = 0, theta = 0;
};

struct checks {
    bool                                             valid = true;
    std::map<std::string, std::pair<double, double>> values;
    std::vector<std::string>                         failures;

    void require(const std::string & name, double observed, double threshold, bool pass) {
        values[name] = { observed, threshold };
        if (!pass) {
            valid = false;
            failures.push_back(name);
        }
    }
};

struct distribution {
    std::vector<double> values;
    double              mean = 0, sd = 0, min = 0, p5 = 0, median = 0, p95 = 0, max = 0;
};

struct metric {
    double                     value = NAN;
    std::map<uint64_t, double> prompts;
};

using basis = TMatrixD;

void usage(const char * program) {
    std::fprintf(stderr,
                 "usage: %s --input-root CAPTURE.root --corpus CORPUS.txt --manifest SPLIT.json --registration "
                 "REGISTRATION.json "
                 "--output-root RESULT.root --output-json RESULT.json --seed 817263 [--threads N] [--development-run] "
                 "[--exploratory-layer N] "
                 "[--overwrite] [--self-test]\n",
                 program);
}

args parse_args(int argc, char ** argv) {
    args     result;
    uint64_t seed = 0;
    for (int i = 1; i < argc; ++i) {
        const auto value = [&](std::string & target) {
            if (i + 1 == argc) {
                throw std::runtime_error("missing argument value");
            }
            target = argv[++i];
        };
        if (!std::strcmp(argv[i], "--input-root")) {
            value(result.input);
        } else if (!std::strcmp(argv[i], "--corpus")) {
            value(result.corpus);
        } else if (!std::strcmp(argv[i], "--manifest")) {
            value(result.manifest);
        } else if (!std::strcmp(argv[i], "--registration")) {
            value(result.registration);
        } else if (!std::strcmp(argv[i], "--output-root")) {
            value(result.output_root);
        } else if (!std::strcmp(argv[i], "--output-json")) {
            value(result.output_json);
        } else if (!std::strcmp(argv[i], "--seed") && i + 1 < argc) {
            seed = std::strtoull(argv[++i], nullptr, 10);
        } else if (!std::strcmp(argv[i], "--threads") && i + 1 < argc) {
            result.threads = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--exploratory-layer") && i + 1 < argc) {
            result.layer       = std::atoi(argv[++i]);
            result.exploratory = true;
        } else if (!std::strcmp(argv[i], "--development-run")) {
            result.development = true;
        } else if (!std::strcmp(argv[i], "--overwrite")) {
            result.overwrite = true;
        } else if (!std::strcmp(argv[i], "--self-test")) {
            result.self_test = true;
        } else {
            throw std::runtime_error("unknown argument");
        }
    }
    if (result.self_test) {
        return result;
    }
    if (result.threads < 0) {
        throw std::runtime_error("--threads must be positive");
    }
    if (result.exploratory && (result.layer < 0 || result.layer == k_layer)) {
        throw std::runtime_error("--exploratory-layer must name a non-layer-15 layer");
    }
    if (result.input.empty() || result.corpus.empty() || result.manifest.empty() ||
        (!result.development && result.registration.empty()) || result.output_root.empty() ||
        result.output_json.empty() || seed != k_seed) {
        throw std::runtime_error("all fixed inputs and --seed 817263 are required");
    }
    if (!result.overwrite &&
        (std::filesystem::exists(result.output_root) || std::filesystem::exists(result.output_json))) {
        throw std::runtime_error("output path exists; --overwrite is a recorded protocol deviation");
    }
    return result;
}

std::string read_text(const std::string & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot read " + path);
    }
    return { std::istreambuf_iterator<char>(input), {} };
}

// Small self-contained SHA-256 implementation keeps the classified artifact independent of external tools.
std::string sha256(const std::string & path) {
    const std::string                         data = read_text(path);
    std::array<uint32_t, 8>                   h{ 0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                                 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 };
    static constexpr std::array<uint32_t, 64> k{
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
    };
    std::vector<uint8_t> bytes(data.begin(), data.end());
    const uint64_t       bits = uint64_t(bytes.size()) * 8;
    bytes.push_back(0x80);
    while (bytes.size() % 64 != 56) {
        bytes.push_back(0);
    }
    for (int i = 7; i >= 0; --i) {
        bytes.push_back(uint8_t(bits >> (i * 8)));
    }
    const auto rotr = [](uint32_t x, int n) {
        return (x >> n) | (x << (32 - n));
    };
    for (size_t offset = 0; offset < bytes.size(); offset += 64) {
        std::array<uint32_t, 64> w{};
        for (int i = 0; i < 16; ++i) {
            w[i] = (uint32_t(bytes[offset + 4 * i]) << 24) | (uint32_t(bytes[offset + 4 * i + 1]) << 16) |
                   (uint32_t(bytes[offset + 4 * i + 2]) << 8) | bytes[offset + 4 * i + 3];
        }
        for (int i = 16; i < 64; ++i) {
            w[i] = (rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3)) + w[i - 16] +
                   (rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10)) + w[i - 7];
        }
        auto [a, b, c, d, e, f, g, hh] = h;
        for (int i = 0; i < 64; ++i) {
            const uint32_t t1 = hh + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ ((~e) & g)) + k[i] + w[i];
            const uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            hh                = g;
            g                 = f;
            f                 = e;
            e                 = d + t1;
            d                 = c;
            c                 = b;
            b                 = a;
            a                 = t1 + t2;
        }
        h[0] += a;
        h[1] += b;
        h[2] += c;
        h[3] += d;
        h[4] += e;
        h[5] += f;
        h[6] += g;
        h[7] += hh;
    }
    std::ostringstream output;
    for (uint32_t x : h) {
        output << std::hex << std::setw(8) << std::setfill('0') << x;
    }
    return output.str();
}

double dot(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("vector width mismatch");
    }
    double total = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        total += double(a[i]) * b[i];
    }
    return total;
}

double norm(const std::vector<float> & a) {
    return std::sqrt(dot(a, a));
}

bool finite(const std::vector<float> & a) {
    return std::all_of(a.begin(), a.end(), [](float x) { return std::isfinite(x); });
}

void center(std::vector<float> & a) {
    double mean = 0;
    for (float x : a) {
        mean += x;
    }
    mean /= a.size();
    for (float & x : a) {
        x = float(x - mean);
    }
}

std::vector<float> unit(std::vector<float> a) {
    const double n = norm(a);
    if (!(n > k_min_norm) || !std::isfinite(n)) {
        throw std::runtime_error("non-finite or zero vector");
    }
    for (float & x : a) {
        x = float(x / n);
    }
    return a;
}

double max_abs(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("vector width mismatch");
    }
    double result = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        result = std::max(result, std::abs(double(a[i]) - b[i]));
    }
    return result;
}

std::unordered_map<uint64_t, bool> read_manifest(const std::string & path) {
    const std::string text = read_text(path);
    const std::regex  entry(
        "\\{[^}]*\\\"line_index\\\"\\s*:\\s*([0-9]+)[^}]*\\\"split\\\"\\s*:\\s*\\\"(train|test)\\\"[^}]*\\}");
    std::unordered_map<uint64_t, bool> result;
    for (std::sregex_iterator it(text.begin(), text.end(), entry), end; it != end; ++it) {
        // The collector persists one-based corpus_line as the stable prompt identifier.
        const auto [found, inserted] = result.emplace(std::stoull((*it)[1]) + 1, (*it)[2] == "train");
        if (!inserted) {
            throw std::runtime_error("manifest repeats prompt ID");
        }
    }
    if (result.empty()) {
        throw std::runtime_error("manifest has no prompt split entries");
    }
    return result;
}

std::set<uint64_t> corpus_ids(const std::string & path) {
    std::set<uint64_t> ids;
    std::istringstream lines(read_text(path));
    std::string        line;
    uint64_t           id = 0;
    while (std::getline(lines, line)) {
        ++id;
        if (!line.empty()) {
            ids.insert(id);
        }
    }
    return ids;
}

std::set<std::string> corpus_prompts(const std::string & path) {
    std::set<std::string> prompts;
    std::istringstream    lines(read_text(path));
    std::string           line;
    while (std::getline(lines, line)) {
        if (!line.empty()) {
            prompts.insert(line);
        }
    }
    return prompts;
}

std::string json_string(const std::string & text, const char * key) {
    const std::regex pattern(std::string("\\\"") + key + "\\\"\\s*:\\s*\\\"([^\\\"]+)\\\"");
    std::smatch      match;
    if (!std::regex_search(text, match, pattern)) {
        throw std::runtime_error(std::string("registration lacks ") + key);
    }
    return match[1];
}

uint64_t json_integer(const std::string & text, const char * key) {
    const std::regex pattern(std::string("\\\"") + key + "\\\"\\s*:\\s*([0-9]+)");
    std::smatch      match;
    if (!std::regex_search(text, match, pattern)) {
        throw std::runtime_error(std::string("registration lacks numeric ") + key);
    }
    return std::stoull(match[1]);
}

void verify_registration(const args & a, bool classified) {
    if (a.registration.empty()) {
        if (!classified) {
            return;
        }
        throw std::runtime_error("classified analysis requires a registration");
    }
    const std::string registration = read_text(a.registration);
    if (json_string(registration, "corpus_sha256") != sha256(a.corpus) ||
        json_string(registration, "manifest_sha256") != sha256(a.manifest)) {
        throw std::runtime_error("frozen corpus or manifest hash does not match registration");
    }
    if (json_integer(registration, "seed") != k_seed) {
        throw std::runtime_error("registration seed differs from protocol");
    }
    for (const char * key :
         { "source_provenance", "rotor_subspace_prompts_sha256", "destination_subspace_expanded_prompts_sha256",
           "destination_subspace_pile10k_prompts_sha256" }) {
        (void) json_string(registration, key);
    }
    if (classified) {
        const std::array<std::pair<const char *, const char *>, 3> prohibited{
            { { "rotor_subspace_prompts_sha256", "pocs/interp/rotor_subspace_prompts.txt" },
             { "destination_subspace_expanded_prompts_sha256",
                "pocs/interp/destination_subspace_expanded_prompts.txt" },
             { "destination_subspace_pile10k_prompts_sha256",
                "pocs/interp/destination_subspace_pile10k_prompts.txt" } }
        };
        const auto candidate_prompts = corpus_prompts(a.corpus);
        for (const auto & [key, path] : prohibited) {
            if (json_string(registration, key) != sha256(path)) {
                throw std::runtime_error("registration prohibited-corpus hash mismatch");
            }
            const auto prior_prompts = corpus_prompts(path);
            for (const std::string & prompt : candidate_prompts) {
                if (prior_prompts.contains(prompt)) {
                    throw std::runtime_error("classified corpus has prompt overlap with a prohibited corpus");
                }
            }
        }
    }
}

std::vector<row> read_capture(const args & a, const std::unordered_map<uint64_t, bool> & split, checks & check) {
    auto                             reader       = ROOT::RNTupleReader::Open("rwkv_activations", a.input);
    const auto                       names        = reader->GetView<std::vector<std::string>>("source_names")(0);
    const std::string                layer_prefix = "rwkv.layer." + std::to_string(a.layer) + ".";
    const std::array<std::string, 3> expected{ layer_prefix + "resid.in", layer_prefix + "time.out",
                                               layer_prefix + "resid.time" };
    std::array<int, 3>               source{};
    for (size_t i = 0; i < expected.size(); ++i) {
        const auto found = std::find(names.begin(), names.end(), expected[i]);
        if (found == names.end()) {
            throw std::runtime_error("capture lacks required layer tap: " + expected[i]);
        }
        source[i] = int(found - names.begin());
    }
    auto         prompt       = reader->GetView<uint64_t>("corpus_line");
    auto         position     = reader->GetView<int32_t>("token_position");
    auto         token        = reader->GetView<int32_t>("token_id");
    auto         f0           = reader->GetView<std::vector<float>>("activation_000");
    const size_t width        = f0(0).size();
    const auto   source_field = [](int index) {
        char field[32];
        std::snprintf(field, sizeof(field), "activation_%03d", index);
        return std::string(field);
    };
    if (!width || width <= k_rank) {
        throw std::runtime_error("capture width is too small for rank 32");
    }
    std::array<ROOT::RNTupleView<std::vector<float>>, 3> fields{
        reader->GetView<std::vector<float>>(source_field(source[0])),
        reader->GetView<std::vector<float>>(source_field(source[1])),
        reader->GetView<std::vector<float>>(source_field(source[2]))
    };
    std::vector<row> result;
    size_t           excluded_train = 0, excluded_test = 0, all_train = 0, all_test = 0;
    double           identity = 0, unit_error = 0, tangent_error = 0, rotor_error = 0;
    for (auto entry : reader->GetEntryRange()) {
        const auto found = split.find(prompt(entry));
        if (found == split.end()) {
            throw std::runtime_error("capture prompt is absent from manifest");
        }
        const bool train = found->second;
        train ? ++all_train : ++all_test;
        row r{};
        r.prompt_id = prompt(entry);
        r.train     = train;
        r.position  = position(entry);
        r.token_id  = token(entry);
        std::array<std::vector<float> *, 3> vectors{ &r.x, &r.w, &r.y };
        for (size_t s = 0; s < fields.size(); ++s) {
            const auto & encoded = fields[s](entry);
            if (encoded.size() != width) {
                throw std::runtime_error("capture width changes between rows");
            }
            vectors[s]->reserve(width);
            vectors[s]->assign(encoded.begin(), encoded.end());
        }
        std::vector<float> raw_sum(width);
        for (size_t i = 0; i < width; ++i) {
            raw_sum[i] = r.x[i] + r.w[i];
        }
        identity = std::max(identity, max_abs(raw_sum, r.y));
        center(r.x);
        center(r.w);
        center(r.y);
        if (!finite(r.x) || !finite(r.w) || !finite(r.y) || norm(r.x) <= k_min_norm || norm(r.w) <= k_min_norm ||
            norm(r.y) <= k_min_norm) {
            train ? ++excluded_train : ++excluded_test;
            continue;
        }
        r.radius            = norm(r.x);
        r.write_norm        = norm(r.w);
        r.u                 = unit(std::move(r.x));
        const auto   v      = unit(r.y);
        const double cosine = std::clamp(dot(r.u, v), -1.0, 1.0);
        r.theta             = std::acos(cosine);
        if (r.theta < k_min_angle) {
            train ? ++excluded_train : ++excluded_test;
            continue;
        }
        r.tangent.resize(width);
        for (size_t i = 0; i < width; ++i) {
            r.tangent[i] = float(v[i] - cosine * r.u[i]);
        }
        r.tangent     = unit(std::move(r.tangent));
        unit_error    = std::max({ unit_error, std::abs(norm(r.u) - 1), std::abs(norm(r.tangent) - 1) });
        tangent_error = std::max(tangent_error, std::abs(dot(r.u, r.tangent)));
        std::vector<float> reconstructed(width);
        for (size_t i = 0; i < width; ++i) {
            reconstructed[i] = float(std::cos(r.theta) * r.u[i] + std::sin(r.theta) * r.tangent[i]);
        }
        rotor_error = std::max(rotor_error, max_abs(reconstructed, v));
        result.push_back(std::move(r));
    }
    check.require("residual_identity", identity, 1e-3, identity <= 1e-3);
    check.require("unit_norm_error", unit_error, 1e-5, unit_error <= 1e-5);
    check.require("tangent_orthogonality", tangent_error, 1e-5, tangent_error <= 1e-5);
    check.require("finite_rotor_reconstruction", rotor_error, 1e-5, rotor_error <= 1e-5);
    check.require("train_exclusions", all_train ? double(excluded_train) / all_train : 1, .01,
                  all_train && double(excluded_train) / all_train <= .01);
    check.require("test_exclusions", all_test ? double(excluded_test) / all_test : 1, .01,
                  all_test && double(excluded_test) / all_test <= .01);
    return result;
}

basis fit_basis(const std::vector<const std::vector<float> *> & rows, const std::vector<double> & weights) {
    if (rows.size() < k_rank || rows.size() != weights.size()) {
        throw std::runtime_error("insufficient weighted train rows");
    }
    const int               m = int(rows.size()), n = int(rows[0]->size());
    std::vector<float>      matrix(size_t(m) * n), gram(size_t(m) * m), eigen(k_rank), vectors(size_t(m) * k_rank);
    std::vector<lapack_int> support(2 * k_rank);
    for (int c = 0; c < n; ++c) {
        for (int r = 0; r < m; ++r) {
            matrix[r + size_t(m) * c] = float((*rows[r])[c] * weights[r]);
        }
    }
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, m, m, n, 1, matrix.data(), m, matrix.data(), m, 0, gram.data(),
                m);
    lapack_int selected = 0;
    if (LAPACKE_ssyevr(LAPACK_COL_MAJOR, 'V', 'I', 'U', m, gram.data(), m, 0, 0, m - k_rank + 1, m, 0, &selected,
                       eigen.data(), vectors.data(), m, support.data()) != 0 ||
        selected != k_rank) {
        throw std::runtime_error("Gram eigendecomposition failed");
    }
    basis result(n, k_rank);
    for (int c = 0; c < k_rank; ++c) {
        const int    component = k_rank - 1 - c;
        const double singular  = std::sqrt(std::max(0.0f, eigen[component]));
        if (!(singular > 1e-10)) {
            throw std::runtime_error("rank-deficient tangent basis");
        }
        for (int j = 0; j < n; ++j) {
            double value = 0;
            for (int r = 0; r < m; ++r) {
                value += matrix[r + size_t(m) * j] * vectors[r + size_t(m) * component];
            }
            result(j, c) = value / singular;
        }
        for (int prior = 0; prior < c; ++prior) {
            double product = 0;
            for (int j = 0; j < n; ++j) {
                product += result(j, prior) * result(j, c);
            }
            for (int j = 0; j < n; ++j) {
                result(j, c) -= product * result(j, prior);
            }
        }
        double length = 0;
        for (int j = 0; j < n; ++j) {
            length += result(j, c) * result(j, c);
        }
        length = std::sqrt(length);
        if (!(length > k_min_norm)) {
            throw std::runtime_error("basis reconstruction failed");
        }
        for (int j = 0; j < n; ++j) {
            result(j, c) /= length;
        }
    }
    return result;
}

// Repeated null fits use a deterministic rank-48 range finder; native bases stay exact.
basis fit_basis_randomized(const std::vector<const std::vector<float> *> & rows,
                           const std::vector<double> &                     weights,
                           uint64_t                                        seed) {
    constexpr int oversampling     = 16;
    constexpr int power_iterations = 1;
    const int     m = int(rows.size()), n = int(rows[0]->size()), l = k_rank + oversampling;
    if (m < l || rows.size() != weights.size()) {
        throw std::runtime_error("insufficient rows for randomized control basis");
    }
    std::vector<float> a(size_t(m) * n), omega(size_t(n) * l), q(size_t(m) * l), z(size_t(n) * l), b(size_t(l) * n),
        gram(size_t(l) * l), eigen(k_rank), vectors(size_t(l) * k_rank);
    for (int c = 0; c < n; ++c) {
        for (int r = 0; r < m; ++r) {
            a[r + size_t(m) * c] = float((*rows[r])[c] * weights[r]);
        }
    }
    std::mt19937_64                 rng(seed);
    std::normal_distribution<float> normal;
    for (float & value : omega) {
        value = normal(rng);
    }
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, l, n, 1, a.data(), m, omega.data(), n, 0, q.data(), m);
    for (int iteration = 0; iteration < power_iterations; ++iteration) {
        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, n, l, m, 1, a.data(), m, q.data(), m, 0, z.data(), n);
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, l, n, 1, a.data(), m, z.data(), n, 0, q.data(), m);
    }
    for (int c = 0; c < l; ++c) {
        for (int prior = 0; prior < c; ++prior) {
            const float product = cblas_sdot(m, q.data() + size_t(m) * prior, 1, q.data() + size_t(m) * c, 1);
            cblas_saxpy(m, -product, q.data() + size_t(m) * prior, 1, q.data() + size_t(m) * c, 1);
        }
        const float length = cblas_snrm2(m, q.data() + size_t(m) * c, 1);
        if (!(length > 1e-8f)) {
            throw std::runtime_error("randomized control basis is rank deficient");
        }
        cblas_sscal(m, 1 / length, q.data() + size_t(m) * c, 1);
    }
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, l, n, m, 1, q.data(), m, a.data(), m, 0, b.data(), l);
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, l, l, n, 1, b.data(), l, b.data(), l, 0, gram.data(), l);
    std::vector<lapack_int> support(2 * k_rank);
    lapack_int              selected = 0;
    if (LAPACKE_ssyevr(LAPACK_COL_MAJOR, 'V', 'I', 'U', l, gram.data(), l, 0, 0, l - k_rank + 1, l, 0, &selected,
                       eigen.data(), vectors.data(), l, support.data()) != 0 ||
        selected != k_rank) {
        throw std::runtime_error("randomized control eigendecomposition failed");
    }
    basis result(n, k_rank);
    for (int c = 0; c < k_rank; ++c) {
        const int    component = k_rank - 1 - c;
        const double singular  = std::sqrt(std::max(0.0f, eigen[component]));
        if (!(singular > 1e-10)) {
            throw std::runtime_error("randomized control singular value is zero");
        }
        for (int r = 0; r < n; ++r) {
            double value = 0;
            for (int j = 0; j < l; ++j) {
                value += b[j + size_t(l) * r] * vectors[j + size_t(l) * component];
            }
            result(r, c) = value / singular;
        }
        for (int prior = 0; prior < c; ++prior) {
            double product = 0;
            for (int r = 0; r < n; ++r) {
                product += result(r, prior) * result(r, c);
            }
            for (int r = 0; r < n; ++r) {
                result(r, c) -= product * result(r, prior);
            }
        }
        double length = 0;
        for (int r = 0; r < n; ++r) {
            length += result(r, c) * result(r, c);
        }
        length = std::sqrt(length);
        if (!(length > k_min_norm)) {
            throw std::runtime_error("randomized control basis reconstruction failed");
        }
        for (int r = 0; r < n; ++r) {
            result(r, c) /= length;
        }
    }
    return result;
}

double orthogonality(const basis & p) {
    double maximum = 0;
    for (int i = 0; i < p.GetNcols(); ++i) {
        for (int j = 0; j < p.GetNcols(); ++j) {
            double value = 0;
            for (int r = 0; r < p.GetNrows(); ++r) {
                value += p(r, i) * p(r, j);
            }
            maximum = std::max(maximum, std::abs(value - (i == j)));
        }
    }
    return maximum;
}

double energy(const basis & p, const std::vector<float> & z) {
    double total = 0;
    for (int c = 0; c < p.GetNcols(); ++c) {
        double coordinate = 0;
        for (int r = 0; r < p.GetNrows(); ++r) {
            coordinate += p(r, c) * z[r];
        }
        total += coordinate * coordinate;
    }
    return total;
}

std::vector<float> transport(const std::vector<float> & u,
                             const std::vector<float> & t,
                             const std::vector<float> & reference) {
    const double denominator = 1 + dot(u, reference);
    if (denominator <= 1e-4) {
        throw std::runtime_error("transport antipode margin failed");
    }
    const double       scale = dot(t, reference) / denominator;
    std::vector<float> result(t.size());
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = float(t[i] - scale * (u[i] + reference[i]));
    }
    return unit(std::move(result));
}

std::vector<float> reference_direction(const std::vector<row> & rows) {
    std::map<uint64_t, std::vector<double>> means;
    for (const row & r : rows) {
        if (r.train) {
            auto & mean = means[r.prompt_id];
            if (mean.empty()) {
                mean.assign(r.u.size() + 1, 0);
            }
            for (size_t i = 0; i < r.u.size(); ++i) {
                mean[i] += r.u[i];
            }
            ++mean.back();
        }
    }
    if (means.empty()) {
        throw std::runtime_error("train split has no prompts");
    }
    std::vector<float> raw(means.begin()->second.size() - 1, 0);
    for (auto & [id, mean] : means) {
        for (size_t i = 0; i < raw.size(); ++i) {
            raw[i] += float(mean[i] / mean.back());
        }
    }
    for (float & x : raw) {
        x /= means.size();
    }
    if (norm(raw) <= 1e-6) {
        throw std::runtime_error("reference extrinsic mean is degenerate");
    }
    return unit(std::move(raw));
}

std::vector<double> prompt_weights(const std::vector<row> & rows, const std::vector<size_t> & indices) {
    std::map<uint64_t, size_t> counts;
    for (size_t i : indices) {
        ++counts[rows[i].prompt_id];
    }
    std::vector<double> result;
    result.reserve(indices.size());
    for (size_t i : indices) {
        result.push_back(1.0 / std::sqrt(counts[rows[i].prompt_id]));
    }
    return result;
}

metric held_out(const basis &                           p,
                const std::vector<std::vector<float>> & representation,
                const std::vector<row> &                rows,
                bool                                    train) {
    std::map<uint64_t, std::pair<double, size_t>> sums;
    for (size_t i = 0; i < rows.size(); ++i) {
        if (rows[i].train == train) {
            const double value = energy(p, representation[i]);
            sums[rows[i].prompt_id].first += value;
            ++sums[rows[i].prompt_id].second;
        }
    }
    metric result;
    result.value = 0;
    for (const auto & [id, sum] : sums) {
        result.prompts[id] = sum.first / sum.second;
    }
    for (const auto & [id, value] : result.prompts) {
        result.value += value;
    }
    result.value /= result.prompts.size();
    return result;
}

distribution summarize(std::vector<double> values) {
    if (values.size() != k_repeats) {
        throw std::runtime_error("control replicate count is not 99");
    }
    std::sort(values.begin(), values.end());
    distribution d;
    d.values            = std::move(values);
    d.min               = d.values.front();
    d.max               = d.values.back();
    const auto quantile = [&](double q) {
        return d.values[size_t(std::ceil(q * (d.values.size() - 1)))];
    };
    d.p5     = quantile(.05);
    d.median = quantile(.5);
    d.p95    = quantile(.95);
    d.mean   = std::accumulate(d.values.begin(), d.values.end(), 0.0) / d.values.size();
    for (double x : d.values) {
        d.sd += (x - d.mean) * (x - d.mean);
    }
    d.sd = std::sqrt(d.sd / d.values.size());
    return d;
}

double p_value(const distribution & d, double native) {
    return (1.0 + std::count_if(d.values.begin(), d.values.end(), [&](double x) { return x >= native; })) /
           (1.0 + d.values.size());
}

uint64_t sub_seed(const char * family, int replicate) {
    uint64_t h = k_seed;
    for (const char * p = family; *p; ++p) {
        h = (h ^ uint8_t(*p)) * 1099511628211ULL;
    }
    return (h ^ uint64_t(replicate)) * 1099511628211ULL;
}

std::vector<size_t> shuffled_indices(const std::vector<row> & rows,
                                     bool                     train,
                                     bool                     within_prompt,
                                     std::mt19937_64 &        rng) {
    std::vector<size_t> result(rows.size());
    std::iota(result.begin(), result.end(), 0);
    if (within_prompt) {
        std::map<uint64_t, std::vector<size_t>> groups;
        for (size_t i = 0; i < rows.size(); ++i) {
            if (rows[i].train == train) {
                groups[rows[i].prompt_id].push_back(i);
            }
        }
        for (auto & [id, group] : groups) {
            std::shuffle(group.begin(), group.end(), rng), [&] {
                for (size_t i : group) {
                    result[i] = i;
                }
            }();
        }
        for (auto & [id, group] : groups) {
            auto perm = group;
            std::shuffle(perm.begin(), perm.end(), rng);
            for (size_t i = 0; i < group.size(); ++i) {
                result[group[i]] = perm[i];
            }
        }
    } else {
        std::vector<size_t> group;
        for (size_t i = 0; i < rows.size(); ++i) {
            if (rows[i].train == train) {
                group.push_back(i);
            }
        }
        auto perm = group;
        std::shuffle(perm.begin(), perm.end(), rng);
        for (size_t i = 0; i < group.size(); ++i) {
            result[group[i]] = perm[i];
        }
    }
    return result;
}

std::vector<std::vector<float>> make_representation(const std::vector<row> &    rows,
                                                    const std::vector<size_t> * permutation,
                                                    bool                        isotropic,
                                                    bool                        transported,
                                                    uint64_t                    seed,
                                                    const std::vector<float> &  reference,
                                                    checks &                    check) {
    std::mt19937_64                 rng(seed);
    std::normal_distribution<float> normal;
    std::vector<std::vector<float>> result;
    result.reserve(rows.size());
    double transport_dot = 0, transport_norm = 0;
    for (size_t i = 0; i < rows.size(); ++i) {
        const row &        r = rows[i];
        std::vector<float> write;
        if (isotropic) {
            write.resize(r.u.size());
            for (float & x : write) {
                x = normal(rng);
            }
            write = unit(std::move(write));
            for (float & x : write) {
                x = float(x * r.write_norm);
            }
        } else {
            write = rows[permutation ? (*permutation)[i] : i].w;
        }
        std::vector<float> endpoint(r.u.size());
        for (size_t j = 0; j < endpoint.size(); ++j) {
            endpoint[j] = float(r.radius * r.u[j] + write[j]);
        }
        const auto   v = unit(std::move(endpoint));
        const double c = std::clamp(dot(r.u, v), -1.0, 1.0), theta = std::acos(c);
        if (theta < k_min_angle) {
            throw std::runtime_error("control created excluded near-zero angle; fail closed");
        }
        std::vector<float> t(r.u.size());
        for (size_t j = 0; j < t.size(); ++j) {
            t[j] = float(v[j] - c * r.u[j]);
        }
        t = unit(std::move(t));
        if (transported) {
            auto z         = transport(r.u, t, reference);
            transport_dot  = std::max(transport_dot, std::abs(dot(z, reference)));
            transport_norm = std::max(transport_norm, std::abs(norm(z) - 1));
            result.push_back(std::move(z));
        } else {
            result.push_back(std::move(t));
        }
    }
    check.require("transport_reference_orthogonality", transport_dot, 1e-5, transport_dot <= 1e-5);
    check.require("transport_norm_error", transport_norm, 1e-5, transport_norm <= 1e-5);
    return result;
}

void run_self_tests() {
    const auto         u     = unit(std::vector<float>{ 1, 0, 0, 0 });
    const auto         v     = unit(std::vector<float>{ 1, .5f, 0, 0 });
    const double       theta = std::acos(dot(u, v));
    std::vector<float> t{ float(v[0] - std::cos(theta)), v[1], 0, 0 };
    t = unit(std::move(t));
    std::vector<float> reconstruction(4);
    for (int i = 0; i < 4; ++i) {
        reconstruction[i] = float(std::cos(theta) * u[i] + std::sin(theta) * t[i]);
    }
    if (max_abs(reconstruction, v) > 1e-6 || std::abs(dot(u, t)) > 1e-6) {
        throw std::runtime_error("finite tangent self-test failed");
    }
    const auto ref = unit(std::vector<float>{ .5f, .5f, 0, 0 });
    const auto z   = transport(u, t, ref);
    if (std::abs(dot(z, ref)) > 1e-5 || std::abs(norm(z) - 1) > 1e-5) {
        throw std::runtime_error("transport self-test failed");
    }
    std::vector<double> controls(k_repeats, .1);
    if (std::abs(p_value(summarize(controls), .2) - .01) > 1e-12) {
        throw std::runtime_error("p-value self-test failed");
    }
    if (std::string("relative_geometry_null_rejected").empty()) {
        throw std::runtime_error("classification self-test failed");
    }
}

void merge_checks(checks & destination, const checks & source, const std::string & prefix) {
    for (const auto & [name, value] : source.values) {
        destination.require(prefix + name, value.first, value.second,
                            std::find(source.failures.begin(), source.failures.end(), name) == source.failures.end());
    }
}

void write_metadata(const std::string & path, const std::map<std::string, std::string> & values) {
    auto model  = ROOT::RNTupleModel::Create();
    auto key    = model->MakeField<std::string>("key");
    auto value  = model->MakeField<std::string>("value");
    auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "metadata", path);
    for (const auto & [name, text] : values) {
        *key   = name;
        *value = text;
        writer->Fill();
    }
    writer->CommitCluster();
}

void write_root_outputs(const std::string &                                path,
                        const metric &                                     native_ambient,
                        const metric &                                     native_transport,
                        const metric &                                     write_native,
                        const std::map<std::string, double> &              values,
                        const std::map<std::string, std::vector<double>> & raw_controls,
                        const checks &                                     check,
                        const std::map<std::string, bool> &                predicates) {
    {
        TFile file(path.c_str(), "UPDATE");
        auto  model          = ROOT::RNTupleModel::Create();
        auto  representation = model->MakeField<std::string>("representation");
        auto  prompt_id      = model->MakeField<uint64_t>("prompt_id");
        auto  metric_name    = model->MakeField<std::string>("metric_name");
        auto  metric_value   = model->MakeField<double>("metric_value");
        auto  writer         = ROOT::RNTupleWriter::Append(std::move(model), "per_prompt_metrics", file);
        for (const auto & [name, metrics] : std::array<std::pair<const char *, const metric *>, 3>{
                 { { "ambient", &native_ambient },
                  { "transported", &native_transport },
                  { "raw_write", &write_native } }
        }) {
            for (const auto & [id, value] : metrics->prompts) {
                *representation = name;
                *prompt_id      = id;
                *metric_name    = "held_out_energy";
                *metric_value   = value;
                writer->Fill();
            }
        }
        writer->CommitCluster();
    }
    {
        TFile file(path.c_str(), "UPDATE");
        auto  model          = ROOT::RNTupleModel::Create();
        auto  representation = model->MakeField<std::string>("representation");
        auto  evaluation     = model->MakeField<std::string>("evaluation_mode");
        auto  family         = model->MakeField<std::string>("control_family");
        auto  metric_name    = model->MakeField<std::string>("metric_name");
        auto  metric_value   = model->MakeField<double>("metric_value");
        auto  writer         = ROOT::RNTupleWriter::Append(std::move(model), "aggregate_metrics", file);
        for (const auto & [key, value] : values) {
            *representation = "all";
            *evaluation     = "native";
            *family         = "native";
            *metric_name    = key;
            *metric_value   = value;
            writer->Fill();
        }
        writer->CommitCluster();
    }
    {
        TFile file(path.c_str(), "UPDATE");
        auto  model          = ROOT::RNTupleModel::Create();
        auto  representation = model->MakeField<std::string>("representation");
        auto  evaluation     = model->MakeField<std::string>("evaluation_mode");
        auto  family         = model->MakeField<std::string>("control_family");
        auto  replicate      = model->MakeField<int32_t>("replicate");
        auto  seed           = model->MakeField<uint64_t>("sub_seed");
        auto  metric_name    = model->MakeField<std::string>("metric_name");
        auto  metric_value   = model->MakeField<double>("metric_value");
        auto  writer         = ROOT::RNTupleWriter::Append(std::move(model), "control_values", file);
        for (const auto & [key, control_values] : raw_controls) {
            *representation = key.substr(0, key.find('_'));
            *evaluation     = key.rfind("_self") != std::string::npos ? "self" : "native_basis";
            *family         = key.find("global_shuffle") != std::string::npos ? "global_shuffle" :
                              key.find("within_prompt") != std::string::npos  ? "within_prompt_shuffle" :
                                                                                "isotropic_write";
            for (int index = 0; index < k_repeats; ++index) {
                *replicate    = index;
                *seed         = sub_seed(family->c_str(), index);
                *metric_name  = "held_out_energy";
                *metric_value = control_values[index];
                writer->Fill();
            }
        }
        writer->CommitCluster();
    }
    {
        TFile file(path.c_str(), "UPDATE");
        auto  model    = ROOT::RNTupleModel::Create();
        auto  name     = model->MakeField<std::string>("check_name");
        auto  pass     = model->MakeField<std::string>("pass");
        auto  observed = model->MakeField<double>("observed");
        auto  writer   = ROOT::RNTupleWriter::Append(std::move(model), "numerical_checks", file);
        for (const auto & [key, pair] : check.values) {
            *name = key;
            *pass =
                std::find(check.failures.begin(), check.failures.end(), key) == check.failures.end() ? "true" : "false";
            *observed = pair.first;
            writer->Fill();
        }
        writer->CommitCluster();
    }
    {
        TFile file(path.c_str(), "UPDATE");
        auto  model  = ROOT::RNTupleModel::Create();
        auto  name   = model->MakeField<std::string>("predicate");
        auto  pass   = model->MakeField<std::string>("pass");
        auto  writer = ROOT::RNTupleWriter::Append(std::move(model), "outcome_predicates", file);
        for (const auto & [key, value] : predicates) {
            *name = key;
            *pass = value ? "true" : "false";
            writer->Fill();
        }
        writer->CommitCluster();
    }
}

void write_json(const args &                                a,
                const checks &                              check,
                const std::string &                         status,
                const std::string &                         classification,
                const std::map<std::string, double> &       values,
                const std::map<std::string, distribution> & controls) {
    std::ofstream out(a.output_json);
    if (!out) {
        throw std::runtime_error("cannot create JSON output");
    }
    out << std::setprecision(17) << "{\n  \"schema_version\": 1,\n  \"status\": \"" << status
        << "\",\n  \"classification\": \"" << classification
        << "\",\n  \"confirmatory\": " << (!a.development && !a.exploratory && !a.overwrite ? "true" : "false")
        << ",\n  \"seed\": 817263,\n  \"layer\": " << a.layer
        << ",\n  \"rank\": 32,\n  \"control_repeats\": 99,\n  "
           "\"input_root_sha256\": \""
        << sha256(a.input) << "\",\n  \"corpus_sha256\": \"" << sha256(a.corpus) << "\",\n  \"manifest_sha256\": \""
        << sha256(a.manifest) << "\",\n  \"metrics\": {";
    bool first = true;
    for (const auto & [name, value] : values) {
        out << (first ? "\n" : ",\n") << "    \"" << name << "\": " << value;
        first = false;
    }
    out << "\n  },\n  \"checks\": {";
    first = true;
    for (const auto & [name, pair] : check.values) {
        out << (first ? "\n" : ",\n") << "    \"" << name << "\": {\"observed\": " << pair.first
            << ", \"threshold\": " << pair.second << ", \"pass\": "
            << (std::find(check.failures.begin(), check.failures.end(), name) == check.failures.end() ? "true" :
                                                                                                        "false")
            << "}";
        first = false;
    }
    out << "\n  },\n  \"controls\": {";
    first = true;
    for (const auto & [name, d] : controls) {
        out << (first ? "\n" : ",\n") << "    \"" << name << "\": {\"count\": " << d.values.size()
            << ", \"mean\": " << d.mean << ", \"standard_deviation\": " << d.sd << ", \"minimum\": " << d.min
            << ", \"p5\": " << d.p5 << ", \"median\": " << d.median << ", \"p95\": " << d.p95
            << ", \"maximum\": " << d.max << "}";
        first = false;
    }
    out << "\n  },\n  \"protocol_deviations\": [" << (a.overwrite ? "\"overwrite\"" : "") << "]\n}\n";
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const args a       = parse_args(argc, argv);
        const int  threads = a.threads ? a.threads : std::min(4, omp_get_max_threads());
        // Replicates remain serial; this only parallelizes each bounded BLAS decomposition.
        omp_set_num_threads(threads);
        openblas_set_num_threads(threads);
        run_self_tests();
        if (a.self_test) {
            std::puts("synthetic self-tests passed");
            return 0;
        }
        verify_registration(a, !a.development);
        checks     check;
        const auto split = read_manifest(a.manifest);
        const auto ids   = corpus_ids(a.corpus);
        if (ids != std::set<uint64_t>([&] {
                std::set<uint64_t> x;
                for (auto [id, value] : split) {
                    x.insert(id);
                }
                return x;
            }())) {
            throw std::runtime_error("manifest and corpus prompt IDs differ");
        }
        std::vector<row>   rows = read_capture(a, split, check);
        std::set<uint64_t> train_prompts, test_prompts;
        size_t             train_count = 0, test_count = 0;
        for (const auto & r : rows) {
            (r.train ? train_prompts : test_prompts).insert(r.prompt_id);
            r.train ? ++train_count : ++test_count;
        }
        check.require("minimum_train_tokens", train_count, 512, train_count >= 512);
        check.require("minimum_test_tokens", test_count, 512, test_count >= 512);
        check.require("minimum_train_prompts", train_prompts.size(), 8, train_prompts.size() >= 8);
        check.require("minimum_test_prompts", test_prompts.size(), 8, test_prompts.size() >= 8);
        std::vector<float> reference = reference_direction(rows);
        double             antipode  = 2;
        for (const auto & r : rows) {
            antipode = std::min(antipode, 1 + dot(r.u, reference));
        }
        check.require("reference_antipode_margin", antipode, 1e-4, antipode > 1e-4);
        std::vector<std::vector<float>> ambient, transported;
        ambient.reserve(rows.size());
        transported.reserve(rows.size());
        double native_transport_dot = 0, native_transport_norm = 0;
        for (const auto & r : rows) {
            ambient.push_back(r.tangent);
            transported.push_back(transport(r.u, r.tangent, reference));
            native_transport_dot  = std::max(native_transport_dot, std::abs(dot(transported.back(), reference)));
            native_transport_norm = std::max(native_transport_norm, std::abs(norm(transported.back()) - 1));
        }
        check.require("native_transport_reference_orthogonality", native_transport_dot, 1e-5,
                      native_transport_dot <= 1e-5);
        check.require("native_transport_norm_error", native_transport_norm, 1e-5, native_transport_norm <= 1e-5);
        std::vector<size_t> train_indices;
        for (size_t i = 0; i < rows.size(); ++i) {
            if (rows[i].train) {
                train_indices.push_back(i);
            }
        }
        const auto weights = prompt_weights(rows, train_indices);
        auto       fit     = [&](const std::vector<std::vector<float>> & z) {
            std::vector<const std::vector<float> *> refs;
            for (size_t i : train_indices) {
                refs.push_back(&z[i]);
            }
            return fit_basis(refs, weights);
        };
        auto fit_control = [&](const std::vector<std::vector<float>> & z, uint64_t seed) {
            std::vector<const std::vector<float> *> refs;
            for (size_t i : train_indices) {
                refs.push_back(&z[i]);
            }
            return fit_basis_randomized(refs, weights, seed);
        };
        const basis native_a = fit(ambient), native_t = fit(transported);
        check.require("ambient_basis_orthogonality", orthogonality(native_a), 1e-5, orthogonality(native_a) <= 1e-5);
        check.require("transport_basis_orthogonality", orthogonality(native_t), 1e-5, orthogonality(native_t) <= 1e-5);
        const metric                    native_ambient   = held_out(native_a, ambient, rows, false),
                                        native_transport = held_out(native_t, transported, rows, false);
        std::vector<std::vector<float>> write_hat;
        write_hat.reserve(rows.size());
        for (const auto & r : rows) {
            write_hat.push_back(unit(r.w));
        }
        const basis write_basis = fit(write_hat);
        check.require("write_basis_orthogonality", orthogonality(write_basis), 1e-5,
                      orthogonality(write_basis) <= 1e-5);
        const metric                  write_native = held_out(write_basis, write_hat, rows, false);
        std::map<std::string, double> values{
            { "G_ambient_native",   native_ambient.value   },
            { "G_transport_native", native_transport.value },
            { "G_write_native",     write_native.value     }
        };
        std::map<std::string, std::vector<double>> raw_controls;
        constexpr int                              k_control_fits          = k_repeats * 3 * 2;
        int                                        control_fits            = 0;
        const auto                                 control_started         = std::chrono::steady_clock::now();
        const auto                                 report_control_progress = [&] {
            ++control_fits;
            if (control_fits % 10 != 0 && control_fits != k_control_fits) {
                return;
            }
            const double elapsed =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - control_started).count();
            const double eta = elapsed * (k_control_fits - control_fits) / control_fits;
            std::printf("control_fits=%d/%d %.1f%% elapsed=%.1fs eta=%.1fs\n", control_fits, k_control_fits,
                        100.0 * control_fits / k_control_fits, elapsed, eta);
            std::fflush(stdout);
        };
        const std::array<std::pair<const char *, bool>, 2> families{
            { { "global_shuffle", false }, { "within_prompt_shuffle", true } }
        };
        for (const auto & [family, within] : families) {
            for (int rep = 0; rep < k_repeats; ++rep) {
                std::mt19937_64 rng(sub_seed(family, rep));
                const auto      train_perm = shuffled_indices(rows, true, within, rng),
                                test_perm  = shuffled_indices(rows, false, within, rng);
                for (const auto & [name, native_basis] : std::array<std::pair<const char *, const basis *>, 2>{
                         { { "ambient", &native_a }, { "transported", &native_t } }
                }) {
                    const bool is_transported = std::string(name) == "transported";
                    checks     local;
                    const auto train_z = make_representation(rows, &train_perm, false, is_transported,
                                                             sub_seed(family, rep), reference, local);
                    const auto test_z  = make_representation(rows, &test_perm, false, is_transported,
                                                             sub_seed(family, rep + 1000), reference, local);
                    merge_checks(check, local, std::string("control_") + family + "_" + name + "_");
                    const basis self = fit_control(train_z, sub_seed(family, rep));
                    raw_controls[std::string(name) + "_" + family + "_self"].push_back(
                        held_out(self, test_z, rows, false).value);
                    raw_controls[std::string(name) + "_" + family + "_native_basis"].push_back(
                        held_out(*native_basis, test_z, rows, false).value);
                    report_control_progress();
                }
            }
        }
        for (int rep = 0; rep < k_repeats; ++rep) {
            for (const auto & [name, native_basis] : std::array<std::pair<const char *, const basis *>, 2>{
                     { { "ambient", &native_a }, { "transported", &native_t } }
            }) {
                const bool is_transported = std::string(name) == "transported";
                checks     local;
                const auto train_z = make_representation(rows, nullptr, true, is_transported,
                                                         sub_seed("isotropic_train", rep), reference, local);
                const auto test_z  = make_representation(rows, nullptr, true, is_transported,
                                                         sub_seed("isotropic_test", rep), reference, local);
                merge_checks(check, local, std::string("control_isotropic_") + name + "_");
                const basis self = fit_control(train_z, sub_seed("isotropic_write", rep));
                raw_controls[std::string(name) + "_isotropic_write_self"].push_back(
                    held_out(self, test_z, rows, false).value);
                raw_controls[std::string(name) + "_isotropic_write_native_basis"].push_back(
                    held_out(*native_basis, test_z, rows, false).value);
                report_control_progress();
            }
        }
        std::map<std::string, distribution> controls;
        for (auto & [key, value] : raw_controls) {
            controls.emplace(key, summarize(value));
        }
        const auto passes = [&](const char * representation, double native, const char * family, const char * mode) {
            const auto & d = controls.at(std::string(representation) + "_" + family + "_" + mode);
            values[std::string("p_") + representation + "_" + family + "_" + mode] = p_value(d, native);
            return native > d.p95 && p_value(d, native) <= .05;
        };
        const bool a_global_self   = passes("ambient", native_ambient.value, "global_shuffle", "self");
        const bool a_global_native = passes("ambient", native_ambient.value, "global_shuffle", "native_basis");
        const bool t_global_self   = passes("transported", native_transport.value, "global_shuffle", "self");
        const bool t_global_native = passes("transported", native_transport.value, "global_shuffle", "native_basis");
        const bool a_within_self   = passes("ambient", native_ambient.value, "within_prompt_shuffle", "self");
        const bool a_within_native = passes("ambient", native_ambient.value, "within_prompt_shuffle", "native_basis");
        const bool t_within_self   = passes("transported", native_transport.value, "within_prompt_shuffle", "self");
        const bool t_within_native =
            passes("transported", native_transport.value, "within_prompt_shuffle", "native_basis");
        const bool  a_iso_self     = passes("ambient", native_ambient.value, "isotropic_write", "self");
        const bool  a_iso_native   = passes("ambient", native_ambient.value, "isotropic_write", "native_basis");
        const bool  t_iso_self     = passes("transported", native_transport.value, "isotropic_write", "self");
        const bool  t_iso_native   = passes("transported", native_transport.value, "isotropic_write", "native_basis");
        const bool  a_global       = a_global_self && a_global_native;
        const bool  t_global       = t_global_self && t_global_native;
        const bool  a_within       = a_within_self && a_within_native;
        const bool  t_within       = t_within_self && t_within_native;
        const bool  a_iso          = a_iso_self && a_iso_native;
        const bool  t_iso          = t_iso_self && t_iso_native;
        std::string classification = "ambiguous";
        if (!a_iso || !t_iso) {
            classification = "no_detected_low_rank_relative_geometry";
        } else if (!a_global || !t_global) {
            classification = "structured_addition_not_distinguished";
        } else if (!a_within || !t_within) {
            classification = "prompt_conditioned_relative_structure";
        } else {
            classification = "relative_geometry_null_rejected";
        }
        if (a.development) {
            classification = "suppressed_development_only";
        }
        if (a.exploratory && check.valid) {
            classification = "exploratory_" + classification;
        }
        if (!check.valid) {
            classification = "suppressed_invalid";
        }
        const std::string status = check.valid ? (a.development ? "development_only" :
                                                  a.exploratory ? "exploratory" :
                                                                  "valid") :
                                                 "invalid";
        write_metadata(a.output_root, {
                                          { "status",            status                  },
                                          { "classification",    classification          },
                                          { "layer",             std::to_string(a.layer) },
                                          { "input_root_sha256", sha256(a.input)         },
                                          { "corpus_sha256",     sha256(a.corpus)        },
                                          { "manifest_sha256",   sha256(a.manifest)      }
        });
        write_root_outputs(a.output_root, native_ambient, native_transport, write_native, values, raw_controls, check,
                           {
                               { "ambient_global",      a_global },
                               { "transport_global",    t_global },
                               { "ambient_within",      a_within },
                               { "transport_within",    t_within },
                               { "ambient_isotropic",   a_iso    },
                               { "transport_isotropic", t_iso    }
        });
        write_json(a, check, status, classification, values, controls);
        return check.valid ? 0 : 2;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
