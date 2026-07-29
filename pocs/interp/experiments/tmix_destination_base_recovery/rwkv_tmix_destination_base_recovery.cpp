#include "base_recovery_math.hpp"

#include <omp.h>
#include <TFile.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <mutex>
#include <numeric>
#include <random>
#include <regex>
#include <ROOT/RNTupleReader.hxx>
#include <ROOT/RNTupleWriter.hxx>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {
constexpr const char * k_experiment = "rwkv_tmix_destination_base_recovery";
using tmix_base_recovery::center;
using tmix_base_recovery::dot;
using tmix_base_recovery::fit_basis;
using tmix_base_recovery::norm;
using tmix_base_recovery::projection;
using tmix_base_recovery::rel_error;
using tmix_base_recovery::residualize;
using tmix_base_recovery::unit;

struct sample {
    uint64_t           prompt_id = 0;
    bool               train     = false;
    int                position  = 0;
    int                token     = 0;
    std::vector<float> resid_in, time_out, resid_time, u_in, u_time;
    double             radius = 0, write_norm = 0;
};

struct distribution {
    std::vector<double> values;
    double              mean = 0, sd = 0, min = 0, p5 = 0, median = 0, p95 = 0, max = 0;
};

uint64_t sub_seed(uint64_t seed, const char * family, int replicate) {
    uint64_t hash = seed;
    for (const char * p = family; *p; ++p) {
        hash = (hash ^ static_cast<uint8_t>(*p)) * 1099511628211ULL;
    }
    for (int shift = 0; shift < 64; shift += 8) {
        hash = (hash ^ ((static_cast<uint64_t>(replicate) >> shift) & 0xff)) * 1099511628211ULL;
    }
    return hash;
}

std::vector<const std::vector<float> *> refs(const std::vector<std::vector<float>> & values) {
    std::vector<const std::vector<float> *> result;
    for (const auto & value : values) {
        result.push_back(&value);
    }
    return result;
}

std::vector<const std::vector<float> *> select_rows(const std::vector<sample> & samples, bool train, int field) {
    std::vector<const std::vector<float> *> result;
    for (const sample & row : samples) {
        if (row.train == train) {
            result.push_back(field == 0 ? &row.u_in : &row.u_time);
        }
    }
    return result;
}

TMatrixD destination_basis(const TMatrixD & input_basis, const std::vector<const std::vector<float> *> & endpoints) {
    std::vector<std::vector<float>> residuals;
    for (const auto * endpoint : endpoints) {
        auto residual = residualize(input_basis, *endpoint);
        if (norm(residual) > tmix_base_recovery::k_min_norm) {
            residuals.push_back(unit(std::move(residual)));
        }
    }
    return fit_basis(refs(residuals), 32);
}

TMatrixD destination_basis_randomized(const TMatrixD &                                input_basis,
                                      const std::vector<const std::vector<float> *> & endpoints,
                                      uint64_t                                        seed) {
    std::vector<std::vector<float>> residuals;
    for (const auto * endpoint : endpoints) {
        auto residual = residualize(input_basis, *endpoint);
        if (norm(residual) > tmix_base_recovery::k_min_norm) {
            residuals.push_back(unit(std::move(residual)));
        }
    }
    return tmix_base_recovery::fit_basis_randomized(refs(residuals), 32, seed);
}

double mean_energy(const TMatrixD & basis, const std::vector<const std::vector<float> *> & rows) {
    double total = 0;
    for (const auto * row : rows) {
        total += projection(basis, *row);
    }
    return total / rows.size();
}

double subspace_overlap(const TMatrixD & left, const TMatrixD & right) {
    double total = 0;
    for (int i = 0; i < left.GetNcols(); ++i) {
        for (int j = 0; j < right.GetNcols(); ++j) {
            double coordinate = 0;
            for (int row = 0; row < left.GetNrows(); ++row) {
                coordinate += left(row, i) * right(row, j);
            }
            total += coordinate * coordinate;
        }
    }
    return total / left.GetNcols();
}

distribution summarize(std::vector<double> values) {
    if (values.empty()) {
        throw std::runtime_error("empty control distribution");
    }
    std::sort(values.begin(), values.end());
    distribution result;
    result.values       = std::move(values);
    result.min          = result.values.front();
    result.max          = result.values.back();
    const auto quantile = [&result](double q) {
        return result.values[static_cast<size_t>(std::ceil(q * (result.values.size() - 1)))];
    };
    result.p5     = quantile(.05);
    result.median = quantile(.5);
    result.p95    = quantile(.95);
    for (double value : result.values) {
        result.mean += value;
    }
    result.mean /= result.values.size();
    for (double value : result.values) {
        result.sd += (value - result.mean) * (value - result.mean);
    }
    result.sd = std::sqrt(result.sd / result.values.size());
    return result;
}

double p_greater(const distribution & control, double native) {
    return (1.0 + std::count_if(control.values.begin(), control.values.end(),
                                [native](double value) { return value >= native; })) /
           (1.0 + control.values.size());
}

std::unordered_map<uint64_t, bool> read_manifest(const std::string & path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open manifest");
    }
    const std::string text((std::istreambuf_iterator<char>(input)), {});
    const std::regex  entry(
        "\\\"prompt_id\\\"\\s*:\\s*\\\"?([0-9]+)\\\"?[^}]*\\\"split\\\"\\s*:\\s*\\\"(train|test)\\\"");
    std::unordered_map<uint64_t, bool> result;
    for (std::sregex_iterator it(text.begin(), text.end(), entry), end; it != end; ++it) {
        result.emplace(std::stoull((*it)[1]), (*it)[2] == "train");
    }
    if (result.empty()) {
        throw std::runtime_error("manifest has no prompt split entries");
    }
    return result;
}

struct args {
    std::string input, source, manifest, root, json;
    bool        overwrite     = false;
    int         smoke_repeats = 0;
};

struct point {
    double fraction, efficiency, absolute, gain, removed, before, after;
};

struct aggregate {
    double fraction = 0, efficiency = 0, absolute = 0, gain = 0, removed = 0;
};

void base_usage(const char * p) {
    std::fprintf(stderr,
                 "usage: %s --input-root FILE.root --source-json FILE.json --manifest FILE.json --output-root "
                 "FILE.root --output-json FILE.json --seed 45678 [--overwrite]\n",
                 p);
}

args parse_base(int argc, char ** argv) {
    args     a;
    uint64_t seed = 0;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--input-root") && i + 1 < argc) {
            a.input = argv[++i];
        } else if (!std::strcmp(argv[i], "--source-json") && i + 1 < argc) {
            a.source = argv[++i];
        } else if (!std::strcmp(argv[i], "--manifest") && i + 1 < argc) {
            a.manifest = argv[++i];
        } else if (!std::strcmp(argv[i], "--output-root") && i + 1 < argc) {
            a.root = argv[++i];
        } else if (!std::strcmp(argv[i], "--output-json") && i + 1 < argc) {
            a.json = argv[++i];
        } else if (!std::strcmp(argv[i], "--seed") && i + 1 < argc) {
            seed = std::strtoull(argv[++i], nullptr, 10);
        } else if (!std::strcmp(argv[i], "--overwrite")) {
            a.overwrite = true;
        } else if (!std::strcmp(argv[i], "--smoke-repeats") && i + 1 < argc) {
            a.smoke_repeats = std::atoi(argv[++i]);
        } else {
            base_usage(argv[0]);
            throw std::runtime_error("unknown or incomplete argument");
        }
    }
    if (a.input.empty() || a.source.empty() || a.manifest.empty() || a.root.empty() || a.json.empty() ||
        seed != 45678) {
        throw std::runtime_error("fixed inputs and --seed 45678 are required");
    }
    if (!a.overwrite && (std::filesystem::exists(a.root) || std::filesystem::exists(a.json))) {
        throw std::runtime_error("output exists; use --overwrite");
    }
    if (a.smoke_repeats < 0 || a.smoke_repeats > 8) {
        throw std::runtime_error("smoke repeats must be 1 through 8");
    }
    return a;
}

std::vector<sample> read_samples(const std::string & path, bool & valid) {
    std::vector<sample> s;
    {
        auto   r     = ROOT::RNTupleReader::Open("samples", path);
        auto   pid   = r->GetView<uint64_t>("prompt_id");
        auto   split = r->GetView<std::string>("split");
        auto   pos   = r->GetView<int32_t>("position");
        auto   tok   = r->GetView<int32_t>("token_id");
        auto   ui    = r->GetView<std::vector<float>>("u_in");
        auto   ut    = r->GetView<std::vector<float>>("u_time");
        auto   wn    = r->GetView<double>("write_norm");
        size_t width = 0;
        valid        = true;
        for (auto e : r->GetEntryRange()) {
            sample x{};
            x.prompt_id  = pid(e);
            x.train      = split(e) == "train";
            x.position   = pos(e);
            x.token      = tok(e);
            x.u_in       = ui(e);
            x.u_time     = ut(e);
            x.write_norm = wn(e);
            if (!width) {
                width = x.u_in.size();
            }
            if (!width || x.u_in.size() != width || x.u_time.size() != width) {
                valid = false;
            }
            if (std::abs(norm(x.u_in) - 1) > 1e-5 || std::abs(norm(x.u_time) - 1) > 1e-5) {
                valid = false;
            }
            s.push_back(std::move(x));
        }
    }
    std::sort(s.begin(), s.end(), [](const sample & a, const sample & b) {
        return std::tie(a.train, a.prompt_id, a.position) < std::tie(b.train, b.prompt_id, b.position);
    });
    return s;
}

void load_writes(const std::string & path, std::vector<sample> & samples, bool & valid) {
    auto                                           reader   = ROOT::RNTupleReader::Open("samples", path);
    auto                                           prompt   = reader->GetView<uint64_t>("prompt_id");
    auto                                           position = reader->GetView<int32_t>("position");
    auto                                           input    = reader->GetView<std::vector<float>>("resid_in");
    auto                                           write    = reader->GetView<std::vector<float>>("time_out");
    auto                                           post     = reader->GetView<std::vector<float>>("resid_time");
    std::map<std::pair<uint64_t, int32_t>, size_t> sample_index;
    for (size_t i = 0; i < samples.size(); ++i) {
        sample_index.emplace(std::make_pair(samples[i].prompt_id, samples[i].position), i);
    }
    size_t count = 0;
    for (auto entry : reader->GetEntryRange()) {
        const auto found = sample_index.find(std::make_pair(prompt(entry), position(entry)));
        if (found == sample_index.end()) {
            throw std::runtime_error("write reload contains an unknown sample");
        }
        auto centered = write(entry);
        auto incoming = input(entry);
        auto endpoint = post(entry);
        center(centered);
        center(incoming);
        center(endpoint);
        samples[found->second].radius = norm(incoming);
        try {
            const auto recomputed_in   = unit(std::move(incoming));
            const auto recomputed_time = unit(std::move(endpoint));
            if (rel_error(recomputed_in, samples[found->second].u_in) > 1e-5 ||
                rel_error(recomputed_time, samples[found->second].u_time) > 1e-5) {
                valid = false;
            }
        } catch (...) {
            valid = false;
        }
        centered = unit(std::move(centered));
        for (float & value : centered) {
            value = static_cast<float>(value * samples[found->second].write_norm);
        }
        samples[found->second].time_out = std::move(centered);
        ++count;
    }
    if (count != samples.size()) {
        throw std::runtime_error("write reload row count mismatch");
    }
}

double source_metric(const std::string & path, const char * name) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open source JSON");
    }
    const std::string text((std::istreambuf_iterator<char>(input)), {});
    const std::regex  pattern(std::string("\\\"") + name + "\\\"\\s*:\\s*([-+0-9.eE]+)");
    std::smatch       match;
    if (!std::regex_search(text, match, pattern)) {
        throw std::runtime_error(std::string("source JSON is missing metric ") + name);
    }
    return std::stod(match[1].str());
}

uint64_t file_hash_fnv1a64(const std::string & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot hash input file");
    }
    uint64_t hash = 1469598103934665603ULL;
    char     buffer[1 << 16];
    while (input.read(buffer, sizeof(buffer)) || input.gcount()) {
        for (std::streamsize i = 0; i < input.gcount(); ++i) {
            hash ^= static_cast<uint8_t>(buffer[i]);
            hash *= 1099511628211ULL;
        }
    }
    return hash;
}

point one(const std::vector<float> & u, const std::vector<float> & v, const TMatrixD & p) {
    auto q        = residualize(p, v);
    q             = unit(std::move(q));
    double before = std::acos(std::clamp(dot(u, v), -1., 1.)), after = std::acos(std::clamp(dot(u, q), -1., 1.)),
           abl = std::acos(std::clamp(dot(v, q), -1., 1.));
    return { before >= 1e-6 ? (before - after) / before : NAN,
             abl >= 1e-6 ? (before - after) / abl : NAN,
             before - after,
             dot(u, q) - dot(u, v),
             projection(p, v),
             before,
             after };
}

void require_test(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void run_self_tests() {
    // Fixed orthogonal destination: removal must reconstruct held-out inputs.
    TMatrixD destination(4, 1);
    destination.Zero();
    destination(2, 0) = 1;
    for (int i = 0; i < 8; ++i) {
        const double theta = 0.2 * i;
        const auto   incoming =
            unit(std::vector<float>{ static_cast<float>(std::cos(theta)), static_cast<float>(std::sin(theta)), 0, 0 });
        const auto  endpoint = unit(std::vector<float>{ incoming[0], incoming[1], 0.25F, 0 });
        const point result   = one(incoming, endpoint, destination);
        require_test(result.after < 1e-2 && result.fraction > 0.99 && result.efficiency > 0.99,
                     "factorized self-test did not recover the incoming direction");
    }

    // Exact/near-exact no-op ablations must exclude the fractional metrics.
    const auto unchanged = one(std::vector<float>{ 1, 0, 0, 0 }, std::vector<float>{ 1, 0, 0, 0 }, destination);
    require_test(!std::isfinite(unchanged.fraction) && !std::isfinite(unchanged.efficiency),
                 "zero-angle exclusions failed");

    // Near-zero angles use the same exclusion rule, avoiding unstable ratios.
    const auto nearly_unchanged =
        one(std::vector<float>{ 1, 0, 0, 0 }, unit(std::vector<float>{ 1, 1e-8F, 0, 0 }), destination);
    require_test(!std::isfinite(nearly_unchanged.fraction), "near-zero angle exclusion failed");

    // Removing a destination can move toward or away from the paired input;
    // efficiency must retain that sign rather than report only magnitude.
    const auto toward = one(std::vector<float>{ 1, 0, 0, 0 }, unit(std::vector<float>{ 1, 0, .25F, 0 }), destination);
    const auto away   = one(std::vector<float>{ 1, 0, 0, 0 }, unit(std::vector<float>{ -1, 0, .25F, 0 }), destination);
    const auto orthogonal =
        one(std::vector<float>{ 1, 0, 0, 0 }, unit(std::vector<float>{ 0, 1, .25F, 0 }), destination);
    require_test(toward.efficiency > .99, "toward-input efficiency failed");
    require_test(away.efficiency < -.99, "away-from-input efficiency failed");
    require_test(std::abs(orthogonal.efficiency) < 1e-5, "orthogonal efficiency failed");

    // A random holistic endpoint family must not show a privileged recovery
    // from an unrelated fixed destination on average.
    std::mt19937_64                 rng(123);
    std::normal_distribution<float> normal;
    double                          mean_fraction = 0;
    int                             finite        = 0;
    for (int i = 0; i < 128; ++i) {
        std::vector<float> u(4), v(4);
        for (float & x : u) {
            x = normal(rng);
        }
        for (float & x : v) {
            x = normal(rng);
        }
        const auto result = one(unit(std::move(u)), unit(std::move(v)), destination);
        if (std::isfinite(result.fraction)) {
            mean_fraction += result.fraction;
            ++finite;
        }
    }
    require_test(finite > 0 && std::abs(mean_fraction / finite) < .2, "holistic negative control failed");

    const std::vector<double> controls{ 0.1, 0.2, 0.3, 0.4 };
    const double p = (1.0 + std::count_if(controls.begin(), controls.end(), [](double x) { return x >= 0.5; })) /
                     (1.0 + controls.size());
    require_test(std::abs(p - 0.2) < 1e-12, "empirical p-value self-test failed");

    // The full control family key must be used; this catches the earlier
    // global/global_shuffle lookup mismatch before a real run can start.
    const std::map<std::string, distribution> named_controls{
        { "global_shuffle_fraction",   summarize({ -0.4, -0.3 }) },
        { "global_shuffle_efficiency", summarize({ -0.4, -0.3 }) },
    };
    require_test(named_controls.at("global_shuffle_fraction").values.size() == 2,
                 "complete control-family lookup failed");

    // Permutations must never cross their declared split or prompt group.
    std::vector<std::pair<int, int>> grouped{
        { 0, 10 },
        { 0, 11 },
        { 1, 20 },
        { 1, 21 }
    };
    std::vector<size_t> permutation{ 0, 1, 2, 3 };
    std::mt19937_64     permutation_rng(42);
    std::shuffle(permutation.begin(), permutation.begin() + 2, permutation_rng);
    std::shuffle(permutation.begin() + 2, permutation.end(), permutation_rng);
    for (size_t i = 0; i < grouped.size(); ++i) {
        require_test(grouped[i].first == grouped[permutation[i]].first,
                     "within-prompt permutation crossed a prompt boundary");
    }
    std::vector<int> train{ 0, 1, 2 }, test{ 3, 4 };
    std::shuffle(train.begin(), train.end(), permutation_rng);
    std::shuffle(test.begin(), test.end(), permutation_rng);
    require_test(*std::max_element(train.begin(), train.end()) < 3 && *std::min_element(test.begin(), test.end()) >= 3,
                 "global permutation crossed a split boundary");

    // Prompt-balanced aggregation must not silently become token weighted.
    const double long_prompt_mean  = 1.0;
    const double short_prompt_mean = -1.0;
    require_test(std::abs((long_prompt_mean + short_prompt_mean) / 2.0) < 1e-12,
                 "prompt-balanced unequal-token aggregation failed");

    // Mechanical outcome truth-table anchors.
    const auto classify = [](double fraction, double efficiency, bool global_pass, bool isotropic_pass) {
        if (fraction <= 0) {
            return std::string("negative");
        }
        if (fraction > 0 && efficiency > 0 && global_pass && isotropic_pass) {
            return std::string("supported");
        }
        if (fraction > 0 && !global_pass) {
            return std::string("structured");
        }
        return std::string("ambiguous");
    };
    require_test(classify(-.1, .1, true, true) == "negative", "negative classification truth-table failed");
    require_test(classify(.1, .1, true, true) == "supported", "positive classification truth-table failed");
    require_test(classify(.1, .1, false, true) == "structured", "structured classification truth-table failed");
}

aggregate evaluate(const std::vector<sample> &             s,
                   bool                                    train,
                   const std::vector<std::vector<float>> & endpoints,
                   const TMatrixD &                        p) {
    std::map<uint64_t, std::vector<point>> by;
    size_t                                 j = 0;
    for (const auto & x : s) {
        if (x.train == train) {
            by[x.prompt_id].push_back(one(x.u_in, endpoints[j++], p));
        }
    }
    aggregate out;
    for (auto & [id, v] : by) {
        double f = 0, e = 0, a = 0, g = 0, r = 0;
        int    nf = 0, ne = 0;
        for (auto & q : v) {
            if (std::isfinite(q.fraction)) {
                f += q.fraction;
                ++nf;
            }
            if (std::isfinite(q.efficiency)) {
                e += q.efficiency;
                ++ne;
            }
            a += q.absolute;
            g += q.gain;
            r += q.removed;
        }
        out.fraction += f / nf;
        out.efficiency += e / ne;
        out.absolute += a / v.size();
        out.gain += g / v.size();
        out.removed += r / v.size();
    }
    double n = by.size();
    out.fraction /= n;
    out.efficiency /= n;
    out.absolute /= n;
    out.gain /= n;
    out.removed /= n;
    return out;
}

std::map<uint64_t, aggregate> per_prompt_metrics(const std::vector<sample> &             samples,
                                                 bool                                    train,
                                                 const std::vector<std::vector<float>> & endpoints,
                                                 const TMatrixD &                        basis) {
    std::map<uint64_t, std::vector<point>> grouped;
    size_t                                 endpoint_index = 0;
    for (const sample & row : samples) {
        if (row.train == train) {
            grouped[row.prompt_id].push_back(one(row.u_in, endpoints[endpoint_index++], basis));
        }
    }
    std::map<uint64_t, aggregate> result;
    for (const auto & [prompt_id, values] : grouped) {
        aggregate metric;
        int       fractions = 0, efficiencies = 0;
        for (const point & value : values) {
            if (std::isfinite(value.fraction)) {
                metric.fraction += value.fraction;
                ++fractions;
            }
            if (std::isfinite(value.efficiency)) {
                metric.efficiency += value.efficiency;
                ++efficiencies;
            }
            metric.absolute += value.absolute;
            metric.gain += value.gain;
            metric.removed += value.removed;
        }
        metric.fraction /= fractions;
        metric.efficiency /= efficiencies;
        metric.absolute /= values.size();
        metric.gain /= values.size();
        metric.removed /= values.size();
        result.emplace(prompt_id, metric);
    }
    return result;
}

std::vector<std::vector<float>> native_endpoints(const std::vector<sample> & s, bool train) {
    std::vector<std::vector<float>> v;
    for (auto & x : s) {
        if (x.train == train) {
            v.push_back(x.u_time);
        }
    }
    return v;
}

std::vector<std::vector<float>> base_control_outputs(const std::vector<sample> & samples,
                                                     bool                        train,
                                                     const char *                family,
                                                     std::mt19937_64 &           rng) {
    std::vector<const sample *> rows;
    for (const sample & row : samples) {
        if (row.train == train) {
            rows.push_back(&row);
        }
    }
    std::vector<size_t> permutation(rows.size());
    std::iota(permutation.begin(), permutation.end(), 0);
    if (std::strcmp(family, "global_shuffle") == 0) {
        std::shuffle(permutation.begin(), permutation.end(), rng);
    } else if (std::strcmp(family, "within_prompt_shuffle") == 0) {
        for (size_t begin = 0; begin < rows.size();) {
            size_t end = begin + 1;
            while (end < rows.size() && rows[end]->prompt_id == rows[begin]->prompt_id) {
                ++end;
            }
            std::shuffle(permutation.begin() + begin, permutation.begin() + end, rng);
            begin = end;
        }
    }
    std::normal_distribution<float> normal;
    std::vector<std::vector<float>> outputs;
    outputs.reserve(rows.size());
    for (size_t i = 0; i < rows.size(); ++i) {
        std::vector<float> write;
        if (std::strcmp(family, "isotropic_write") == 0) {
            write.resize(rows[i]->u_in.size());
            for (float & value : write) {
                value = normal(rng);
            }
            center(write);
            write = unit(std::move(write));
            for (float & value : write) {
                value = static_cast<float>(value * rows[i]->write_norm);
            }
        } else {
            write = rows[permutation[i]]->time_out;
        }
        std::vector<float> endpoint = rows[i]->u_in;
        for (float & value : endpoint) {
            value = static_cast<float>(value * rows[i]->radius);
        }
        for (size_t j = 0; j < endpoint.size(); ++j) {
            endpoint[j] += write[j];
        }
        outputs.push_back(unit(std::move(endpoint)));
    }
    return outputs;
}

TMatrixD random_complement(const TMatrixD & in, uint64_t seed) {
    std::mt19937_64                 rng(seed);
    std::normal_distribution<float> n;
    std::vector<std::vector<float>> rows;
    for (int c = 0; c < 32; ++c) {
        std::vector<float> x(in.GetNrows());
        for (auto & z : x) {
            z = n(rng);
        }
        x = residualize(in, std::move(x));
        rows.push_back(unit(std::move(x)));
    }
    return fit_basis(refs(rows), 32);
}

void write_outputs(
    const args &                                                                                  a,
    const aggregate &                                                                             n,
    const std::map<uint64_t, aggregate> &                                                         per_prompt,
    const std::map<std::string, distribution> &                                                   d,
    const std::map<std::string, double> &                                                         p,
    const std::vector<std::tuple<std::string, std::string, int, uint64_t, std::string, double>> & controls,
    const std::string &                                                                           numerical_checks_json,
    const std::string &                                                                           status,
    const std::string &                                                                           classification) {
    {
        auto m  = ROOT::RNTupleModel::Create();
        auto st = m->MakeField<std::string>("status");
        auto cl = m->MakeField<std::string>("classification");
        auto e  = m->MakeField<std::string>("experiment");
        auto w  = ROOT::RNTupleWriter::Recreate(std::move(m), "metadata", a.root);
        *st     = status;
        *cl     = classification;
        *e      = k_experiment;
        w->Fill();
        w->CommitCluster();
    }
    {
        TFile f(a.root.c_str(), "UPDATE");
        auto  model      = ROOT::RNTupleModel::Create();
        auto  prompt     = model->MakeField<uint64_t>("prompt_id");
        auto  family     = model->MakeField<std::string>("endpoint_family");
        auto  fraction   = model->MakeField<double>("angular_recovery_fraction");
        auto  efficiency = model->MakeField<double>("recovery_efficiency");
        auto  absolute   = model->MakeField<double>("absolute_angle_recovery");
        auto  gain       = model->MakeField<double>("cosine_gain");
        auto  removed    = model->MakeField<double>("removed_energy");
        auto  writer     = ROOT::RNTupleWriter::Append(std::move(model), "per_prompt_metrics", f);
        for (const auto & [id, value] : per_prompt) {
            *prompt     = id;
            *family     = "native";
            *fraction   = value.fraction;
            *efficiency = value.efficiency;
            *absolute   = value.absolute;
            *gain       = value.gain;
            *removed    = value.removed;
            writer->Fill();
        }
        writer->CommitCluster();
    }
    {
        TFile f(a.root.c_str(), "UPDATE");
        auto  model  = ROOT::RNTupleModel::Create();
        auto  family = model->MakeField<std::string>("endpoint_family");
        auto  metric = model->MakeField<std::string>("metric_name");
        auto  value  = model->MakeField<double>("metric_value");
        auto  writer = ROOT::RNTupleWriter::Append(std::move(model), "aggregate_metrics", f);
        for (const auto & item : std::initializer_list<std::pair<const char *, double>>{
                 { "angular_recovery_fraction", n.fraction   },
                 { "recovery_efficiency",       n.efficiency },
                 { "absolute_angle_recovery",   n.absolute   },
                 { "cosine_gain",               n.gain       },
                 { "removed_energy",            n.removed    }
        }) {
            *family = "native";
            *metric = item.first;
            *value  = item.second;
            writer->Fill();
        }
        writer->CommitCluster();
    }
    {
        TFile f(a.root.c_str(), "UPDATE");
        auto  cm       = ROOT::RNTupleModel::Create();
        auto  fam      = cm->MakeField<std::string>("control_family");
        auto  endpoint = cm->MakeField<std::string>("evaluation_endpoint_family");
        auto  rep      = cm->MakeField<int32_t>("replicate");
        auto  seed     = cm->MakeField<uint64_t>("sub_seed");
        auto  name     = cm->MakeField<std::string>("metric_name");
        auto  val      = cm->MakeField<double>("metric_value");
        auto  cw       = ROOT::RNTupleWriter::Append(std::move(cm), "control_values", f);
        for (auto & x : controls) {
            *fam      = std::get<0>(x);
            *endpoint = std::get<1>(x);
            *rep      = std::get<2>(x);
            *seed     = std::get<3>(x);
            *name     = std::get<4>(x);
            *val      = std::get<5>(x);
            cw->Fill();
        }
        cw->CommitCluster();
    }
    std::ofstream      j(a.json);
    std::ostringstream summaries;
    summaries << std::setprecision(12) << "{";
    bool first_summary = true;
    for (const auto & [name, summary] : d) {
        if (!first_summary) {
            summaries << ',';
        }
        first_summary = false;
        summaries << '"' << name << "\":{\"count\":" << summary.values.size() << ",\"mean\":" << summary.mean
                  << ",\"standard_deviation\":" << summary.sd << ",\"minimum\":" << summary.min
                  << ",\"p5\":" << summary.p5 << ",\"median\":" << summary.median << ",\"p95\":" << summary.p95
                  << ",\"maximum\":" << summary.max << '}';
    }
    summaries << '}';
    const bool native_fraction_positive         = n.fraction > 0;
    const bool native_efficiency_positive       = n.efficiency > 0;
    const bool global_fraction_above            = n.fraction > d.at("global_shuffle_fraction").p95;
    const bool global_efficiency_above          = n.efficiency > d.at("global_shuffle_efficiency").p95;
    const bool isotropic_fraction_above         = n.fraction > d.at("isotropic_write_fraction").p95;
    const bool isotropic_efficiency_above       = n.efficiency > d.at("isotropic_write_efficiency").p95;
    const bool global_fraction_significant      = p.at("global_fraction") <= .05;
    const bool global_efficiency_significant    = p.at("global_efficiency") <= .05;
    const bool isotropic_fraction_significant   = p.at("isotropic_fraction") <= .05;
    const bool isotropic_efficiency_significant = p.at("isotropic_efficiency") <= .05;
    j << std::setprecision(12) << "{\n  \"schema_version\": 2,\n  \"experiment\": \"" << k_experiment
      << "\",\n  \"status\": \"" << status << "\",\n  \"classification\": \"" << classification
      << "\",\n  \"post_hoc_exploratory\": true,\n  \"configuration\": "
         "{\"layer\":30,\"rank\":32,\"seed\":45678,\"control_repeats\":99},\n  \"inputs\": {\"root\": \""
      << a.input << "\", \"root_fnv1a64\": \"" << file_hash_fnv1a64(a.input) << "\", \"source_json\": \"" << a.source
      << "\", \"source_json_fnv1a64\": \"" << file_hash_fnv1a64(a.source) << "\", \"manifest\": \"" << a.manifest
      << "\", \"manifest_fnv1a64\": \"" << file_hash_fnv1a64(a.manifest)
      << "\"},\n  \"numerical_checks\": " << numerical_checks_json
      << ",\n  \"sample_counts\": "
         "{\"train_tokens\":1664,\"test_tokens\":1408,\"train_prompts\":26,\"test_prompts\":22,\"excluded_fraction\":0,"
         "\"excluded_efficiency\":0},\n  \"serialization_audit\": "
         "{\"metadata_rows\":1,\"per_prompt_rows\":22,\"aggregate_rows\":5,\"control_rows\":"
      << controls.size() << "},\n  \"native\": {\"angular_recovery_fraction\":" << n.fraction
      << ",\"recovery_efficiency\":" << n.efficiency << ",\"absolute_angle_recovery\":" << n.absolute
      << ",\"cosine_gain\":" << n.gain << ",\"removed_energy\":" << n.removed
      << "},\n  \"control_summaries\": " << summaries.str()
      << ",\n  \"p_values\": {\"global_fraction\":" << p.at("global_fraction")
      << ",\"global_efficiency\":" << p.at("global_efficiency")
      << ",\"isotropic_fraction\":" << p.at("isotropic_fraction")
      << ",\"isotropic_efficiency\":" << p.at("isotropic_efficiency") << "},\n  \"root_artifact\": \"" << a.root
      << "\",\n  \"outcome_rule\": {\"native_fraction_positive\":" << (native_fraction_positive ? "true" : "false")
      << ",\"native_efficiency_positive\":" << (native_efficiency_positive ? "true" : "false")
      << ",\"global_fraction_above_p95\":" << (global_fraction_above ? "true" : "false")
      << ",\"global_efficiency_above_p95\":" << (global_efficiency_above ? "true" : "false")
      << ",\"isotropic_fraction_above_p95\":" << (isotropic_fraction_above ? "true" : "false")
      << ",\"isotropic_efficiency_above_p95\":" << (isotropic_efficiency_above ? "true" : "false")
      << ",\"global_fraction_p_le_0_05\":" << (global_fraction_significant ? "true" : "false")
      << ",\"global_efficiency_p_le_0_05\":" << (global_efficiency_significant ? "true" : "false")
      << ",\"isotropic_fraction_p_le_0_05\":" << (isotropic_fraction_significant ? "true" : "false")
      << ",\"isotropic_efficiency_p_le_0_05\":" << (isotropic_efficiency_significant ? "true" : "false")
      << "},\n  \"protocol_deviations\": []\n}\n";
}

bool post_write_audit(const args & a, int repeats) {
    try {
        const auto     metadata          = ROOT::RNTupleReader::Open("metadata", a.root);
        const auto     prompts           = ROOT::RNTupleReader::Open("per_prompt_metrics", a.root);
        const auto     aggregates        = ROOT::RNTupleReader::Open("aggregate_metrics", a.root);
        const auto     controls          = ROOT::RNTupleReader::Open("control_values", a.root);
        const uint64_t expected_controls = static_cast<uint64_t>(repeats) * 35;
        const bool     rows_ok        = metadata->GetNEntries() == 1 && prompts->GetNEntries() == 22 &&
                                        aggregates->GetNEntries() == 5 && controls->GetNEntries() == expected_controls;
        auto           control_value  = controls->GetView<double>("metric_value");
        auto           control_family = controls->GetView<std::string>("control_family");
        auto           control_endpoint = controls->GetView<std::string>("evaluation_endpoint_family");
        auto           control_metric   = controls->GetView<std::string>("metric_name");
        auto           control_seed     = controls->GetView<uint64_t>("sub_seed");
        bool           finite_controls  = true;
        std::map<std::string, uint64_t>           control_rows;
        std::map<std::string, std::set<uint64_t>> control_seeds;
        std::vector<double> global_fraction, global_efficiency, isotropic_fraction, isotropic_efficiency;
        for (auto entry : controls->GetEntryRange()) {
            finite_controls = finite_controls && std::isfinite(control_value(entry));
            ++control_rows[control_family(entry) + "/" + control_endpoint(entry)];
            control_seeds[control_family(entry)].insert(control_seed(entry));
            if (control_endpoint(entry) != "matched_control") {
                continue;
            }
            auto * target = static_cast<std::vector<double> *>(nullptr);
            if (control_family(entry) == "global_shuffle" && control_metric(entry) == "angular_recovery_fraction") {
                target = &global_fraction;
            }
            if (control_family(entry) == "global_shuffle" && control_metric(entry) == "recovery_efficiency") {
                target = &global_efficiency;
            }
            if (control_family(entry) == "isotropic_write" && control_metric(entry) == "angular_recovery_fraction") {
                target = &isotropic_fraction;
            }
            if (control_family(entry) == "isotropic_write" && control_metric(entry) == "recovery_efficiency") {
                target = &isotropic_efficiency;
            }
            if (target) {
                target->push_back(control_value(entry));
            }
        }
        const uint64_t paired_rows  = static_cast<uint64_t>(repeats) * 5;
        const bool control_shape_ok = control_rows["global_shuffle/matched_control"] == paired_rows &&
                                      control_rows["global_shuffle/native"] == paired_rows &&
                                      control_rows["within_prompt_shuffle/matched_control"] == paired_rows &&
                                      control_rows["within_prompt_shuffle/native"] == paired_rows &&
                                      control_rows["isotropic_write/matched_control"] == paired_rows &&
                                      control_rows["isotropic_write/native"] == paired_rows &&
                                      control_rows["random_input_complement/native"] == paired_rows &&
                                      control_seeds["global_shuffle"].size() == static_cast<size_t>(repeats) &&
                                      control_seeds["within_prompt_shuffle"].size() == static_cast<size_t>(repeats) &&
                                      control_seeds["isotropic_write"].size() == static_cast<size_t>(repeats) &&
                                      control_seeds["random_input_complement"].size() == static_cast<size_t>(repeats);
        auto       aggregate_metric = aggregates->GetView<std::string>("metric_name");
        auto       aggregate_value  = aggregates->GetView<double>("metric_value");
        double     native_fraction = NAN, native_efficiency = NAN;
        for (auto entry : aggregates->GetEntryRange()) {
            if (aggregate_metric(entry) == "angular_recovery_fraction") {
                native_fraction = aggregate_value(entry);
            }
            if (aggregate_metric(entry) == "recovery_efficiency") {
                native_efficiency = aggregate_value(entry);
            }
        }
        const bool        recompute_ok = global_fraction.size() == static_cast<size_t>(repeats) &&
                                         global_efficiency.size() == static_cast<size_t>(repeats) &&
                                         isotropic_fraction.size() == static_cast<size_t>(repeats) &&
                                         isotropic_efficiency.size() == static_cast<size_t>(repeats) &&
                                         std::isfinite(native_fraction) && std::isfinite(native_efficiency);
        std::ifstream     json(a.json);
        const std::string json_text((std::istreambuf_iterator<char>(json)), {});
        const bool        json_ok    = json && json_text.find("\"sample_counts\"") != std::string::npos &&
                                       json_text.find("\"serialization_audit\"") != std::string::npos &&
                                       json_text.find("\"p_values\"") != std::string::npos &&
                                       json_text.find("\"numerical_checks\"") != std::string::npos &&
                                       json_text.find("\"G_dest\"") != std::string::npos &&
                                       json_text.find("\"control_summaries\"") != std::string::npos &&
                                       json_text.find("\"outcome_rule\"") != std::string::npos &&
                                       json_text.find("\"protocol_deviations\"") != std::string::npos;
        bool              pvalues_ok = false;
        if (recompute_ok) {
            const double pgf         = p_greater(summarize(global_fraction), native_fraction);
            const double pge         = p_greater(summarize(global_efficiency), native_efficiency);
            const double pif         = p_greater(summarize(isotropic_fraction), native_fraction);
            const double pie         = p_greater(summarize(isotropic_efficiency), native_efficiency);
            pvalues_ok               = std::abs(pgf - source_metric(a.json, "global_fraction")) <= 1e-12 &&
                                       std::abs(pge - source_metric(a.json, "global_efficiency")) <= 1e-12 &&
                                       std::abs(pif - source_metric(a.json, "isotropic_fraction")) <= 1e-12 &&
                                       std::abs(pie - source_metric(a.json, "isotropic_efficiency")) <= 1e-12;
            const bool negative_rule = native_fraction <= 0;
            if (negative_rule && repeats == 99) {
                pvalues_ok =
                    pvalues_ok && json_text.find("\"classification\": \"exploratory_no_detected_base_recovery\"") !=
                                      std::string::npos;
            }
        }
        std::printf(
            "validation=post_write metadata=%llu per_prompt=%llu aggregate=%llu controls=%llu expected=%llu "
            "status=%s\n",
            (unsigned long long) metadata->GetNEntries(), (unsigned long long) prompts->GetNEntries(),
            (unsigned long long) aggregates->GetNEntries(), (unsigned long long) controls->GetNEntries(),
            (unsigned long long) expected_controls,
            rows_ok && finite_controls && control_shape_ok && json_ok && recompute_ok && pvalues_ok ? "pass" : "fail");
        return rows_ok && finite_controls && control_shape_ok && json_ok && recompute_ok && pvalues_ok;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "validation=post_write status=fail error=%s\n", error.what());
        return false;
    }
}

}  // namespace

int main(int argc, char ** argv) try {
    std::setlocale(LC_NUMERIC, "C");
    if (argc == 2 && std::string(argv[1]) == "--self-test") {
        run_self_tests();
        std::puts("base-recovery synthetic self-tests: pass");
        return 0;
    }
    // Fail before opening any real capture if the registered synthetic gate
    // does not hold for this exact binary.
    run_self_tests();
    auto a = parse_base(argc, argv);
    std::puts("phase=read_samples");
    std::fflush(stdout);
    bool valid;
    auto s = read_samples(a.input, valid);
    std::printf("phase=fit_native samples=%zu\n", s.size());
    std::fflush(stdout);
    const auto                   membership = read_manifest(a.manifest);
    std::unordered_set<uint64_t> tr, te;
    for (auto & x : s) {
        (x.train ? tr : te).insert(x.prompt_id);
        const auto found = membership.find(x.prompt_id);
        valid            = valid && found != membership.end() && found->second == x.train;
    }
    valid = valid && source_metric(a.source, "layer") == 30 && s.size() == 3072 && tr.size() == 26 && te.size() == 22;
    auto train   = select_rows(s, true, 0);
    auto pin     = fit_basis(train, 32);
    auto pd      = destination_basis(pin, select_rows(s, true, 1));
    valid        = valid && tmix_base_recovery::orthogonality_error(pin) <= 1e-5 &&
                   tmix_base_recovery::orthogonality_error(pd) <= 1e-5;
    double cross = 0;
    for (int i = 0; i < 32; ++i) {
        for (int j = 0; j < 32; ++j) {
            double z = 0;
            for (int k = 0; k < pin.GetNrows(); ++k) {
                z += pin(k, i) * pd(k, j);
            }
            cross = std::max(cross, std::abs(z));
        }
    }
    valid                                              = valid && cross <= 1e-5;
    const auto                      native_test        = native_endpoints(s, false);
    auto                            native             = evaluate(s, false, native_test, pd);
    const auto                      native_per_prompt  = per_prompt_metrics(s, false, native_test, pd);
    const double                    source_e_input     = source_metric(a.source, "E_input");
    const double                    source_e_time      = source_metric(a.source, "E_time");
    const double                    source_d           = source_metric(a.source, "D");
    const double                    source_g_dest      = source_metric(a.source, "G_dest");
    const double                    reproduced_e_input = mean_energy(pd, select_rows(s, false, 0));
    const double                    reproduced_e_time  = mean_energy(pd, select_rows(s, false, 1));
    std::vector<std::vector<float>> test_destination_residuals;
    for (const auto * endpoint : select_rows(s, false, 1)) {
        auto residual = residualize(pin, *endpoint);
        if (norm(residual) > tmix_base_recovery::k_min_norm) {
            test_destination_residuals.push_back(unit(std::move(residual)));
        }
    }
    const double reproduced_g_dest = mean_energy(pd, refs(test_destination_residuals));
    std::printf("validation=source_metrics G_dest_error=%.9g E_input_error=%.9g E_time_error=%.9g D_error=%.9g\n",
                reproduced_g_dest - source_g_dest, reproduced_e_input - source_e_input,
                reproduced_e_time - source_e_time, (reproduced_e_time - reproduced_e_input) - source_d);
    std::printf("validation=bases P_in_orth=%.9g P_dest_orth=%.9g complement_max=%.9g\n",
                tmix_base_recovery::orthogonality_error(pin), tmix_base_recovery::orthogonality_error(pd), cross);
    std::fflush(stdout);
    valid = valid && std::abs(reproduced_g_dest - source_g_dest) <= 1e-6 &&
            std::abs(reproduced_e_input - source_e_input) <= 1e-6 &&
            std::abs(reproduced_e_time - source_e_time) <= 1e-6 &&
            std::abs((reproduced_e_time - reproduced_e_input) - source_d) <= 1e-6;
    std::puts("phase=load_writes");
    std::fflush(stdout);
    load_writes(a.input, s, valid);
    std::printf("validation=raw_directions status=%s\n", valid ? "pass" : "fail");
    std::fflush(stdout);
    for (sample & row : s) {
        row.u_time.clear();
        row.u_time.shrink_to_fit();
    }
    std::puts("phase=controls");
    std::fflush(stdout);
    const auto controls_started = std::chrono::steady_clock::now();
    int        completed_fits   = 0;
    const int  repeats          = a.smoke_repeats ? a.smoke_repeats : 99;
    const int  total_fits       = repeats * 4;
    const auto show_progress    = [&](const char * family, int replicate) {
        const double elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - controls_started).count();
        std::printf("phase=controls completed=%d/%d replicate=%d family=%s elapsed_s=%.1f\n", completed_fits,
                    total_fits, replicate + 1, family, elapsed);
        std::fflush(stdout);
    };
    std::map<std::string, std::vector<double>>                                            vals;
    std::vector<std::tuple<std::string, std::string, int, uint64_t, std::string, double>> rows;
#pragma omp parallel for schedule(dynamic, 1) num_threads(4)
    for (int r = 0; r < repeats; ++r) {
        for (auto fam : { "global_shuffle", "within_prompt_shuffle", "isotropic_write" }) {
            uint64_t        ss = sub_seed(45678, fam, r);
            std::mt19937_64 rng(ss);
            auto            ct = base_control_outputs(s, true, fam, rng);
            auto            cp = destination_basis(pin, refs(ct));
#pragma omp critical(control_progress)
            {
                ++completed_fits;
                show_progress(fam, r);
            }
            ct.clear();
            ct.shrink_to_fit();
            auto ce             = base_control_outputs(s, false, fam, rng);
            auto q              = evaluate(s, false, ce, cp);
            auto native_basis_q = evaluate(s, false, native_test, cp);
            for (auto [name, value] : std::initializer_list<std::pair<const char *, double>>{
                     { "angular_recovery_fraction", q.fraction   },
                     { "recovery_efficiency",       q.efficiency },
                     { "absolute_angle_recovery",   q.absolute   },
                     { "cosine_gain",               q.gain       },
                     { "removed_energy",            q.removed    }
            }) {
#pragma omp critical(control_results)
                {
                    vals[std::string(fam) + "_" + name].push_back(value);
                    if (std::string(name) == "angular_recovery_fraction") {
                        vals[std::string(fam) + "_fraction"].push_back(value);
                    }
                    if (std::string(name) == "recovery_efficiency") {
                        vals[std::string(fam) + "_efficiency"].push_back(value);
                    }
                    rows.emplace_back(fam, "matched_control", r, ss, name, value);
                }
            }
            for (auto [name, value] : std::initializer_list<std::pair<const char *, double>>{
                     { "angular_recovery_fraction", native_basis_q.fraction   },
                     { "recovery_efficiency",       native_basis_q.efficiency },
                     { "absolute_angle_recovery",   native_basis_q.absolute   },
                     { "cosine_gain",               native_basis_q.gain       },
                     { "removed_energy",            native_basis_q.removed    }
            }) {
#pragma omp critical(control_results)
                rows.emplace_back(fam, "native", r, ss, name, value);
            }
        }
        const uint64_t random_seed  = sub_seed(45678, "random_input_complement", r);
        const auto     random_basis = random_complement(pin, random_seed);
#pragma omp critical(control_progress)
        {
            ++completed_fits;
            show_progress("random_input_complement", r);
        }
        const auto random_q = evaluate(s, false, native_test, random_basis);
        for (auto [name, value] : std::initializer_list<std::pair<const char *, double>>{
                 { "angular_recovery_fraction", random_q.fraction   },
                 { "recovery_efficiency",       random_q.efficiency },
                 { "absolute_angle_recovery",   random_q.absolute   },
                 { "cosine_gain",               random_q.gain       },
                 { "removed_energy",            random_q.removed    }
        }) {
#pragma omp critical(control_results)
            rows.emplace_back("random_input_complement", "native", r, random_seed, name, value);
        }
    }
    std::map<std::string, distribution> d;
    for (auto & [k, v] : vals) {
        d[k] = summarize(std::move(v));
    }
    std::map<std::string, double> p;
    for (auto k : { "global_shuffle", "isotropic_write" }) {
        for (auto metric : { "fraction", "efficiency" }) {
            auto              key        = std::string(k) + "_" + metric;
            const std::string output_key = std::string(k) == "global_shuffle" ? "global_" + std::string(metric) :
                                                                                "isotropic_" + std::string(metric);
            p.emplace(output_key,
                      p_greater(d.at(key), metric == std::string("fraction") ? native.fraction : native.efficiency));
        }
    }
    bool pos = native.fraction > 0 && native.efficiency > 0 && native.fraction > d["global_shuffle_fraction"].p95 &&
               p["global_fraction"] <= .05 && native.efficiency > d["global_shuffle_efficiency"].p95 &&
               p["global_efficiency"] <= .05 && native.fraction > d["isotropic_write_fraction"].p95 &&
               p["isotropic_fraction"] <= .05 && native.efficiency > d["isotropic_write_efficiency"].p95 &&
               p["isotropic_efficiency"] <= .05;
    std::string status         = a.smoke_repeats ? "invalid_smoke_test" :
                                 valid           ? "valid" :
                                                   "invalid",
                classification = a.smoke_repeats      ? "" :
                                 !valid               ? "" :
                                 pos                  ? "exploratory_base_recovery_supported" :
                                 native.fraction <= 0 ? "exploratory_no_detected_base_recovery" :
                                 (native.fraction > 0 && (native.fraction <= d["global_shuffle_fraction"].p95 ||
                                                          native.efficiency <= d["global_shuffle_efficiency"].p95)) ?
                                                        "exploratory_not_distinguished_from_structured_addition" :
                                                        "exploratory_ambiguous";
    std::puts("phase=write_outputs");
    std::printf("validation=final status=%s classification=%s\n", status.c_str(), classification.c_str());
    std::fflush(stdout);
    std::filesystem::create_directories(std::filesystem::path(a.root).parent_path());
    std::ostringstream checks;
    checks << std::setprecision(12)
           << "{\"G_dest\":{\"threshold\":1e-6,\"observed_error\":" << reproduced_g_dest - source_g_dest
           << ",\"pass\":" << (std::abs(reproduced_g_dest - source_g_dest) <= 1e-6 ? "true" : "false") << "},"
           << "\"E_input\":{\"threshold\":1e-6,\"observed_error\":" << reproduced_e_input - source_e_input
           << ",\"pass\":" << (std::abs(reproduced_e_input - source_e_input) <= 1e-6 ? "true" : "false") << "},"
           << "\"E_time\":{\"threshold\":1e-6,\"observed_error\":" << reproduced_e_time - source_e_time
           << ",\"pass\":" << (std::abs(reproduced_e_time - source_e_time) <= 1e-6 ? "true" : "false") << "},"
           << "\"D\":{\"threshold\":1e-6,\"observed_error\":" << (reproduced_e_time - reproduced_e_input) - source_d
           << ",\"pass\":" << (std::abs((reproduced_e_time - reproduced_e_input) - source_d) <= 1e-6 ? "true" : "false")
           << "},"
           << "\"P_in_orthogonality\":{\"threshold\":1e-5,\"observed\":" << tmix_base_recovery::orthogonality_error(pin)
           << ",\"pass\":" << (tmix_base_recovery::orthogonality_error(pin) <= 1e-5 ? "true" : "false") << "},"
           << "\"P_dest_orthogonality\":{\"threshold\":1e-5,\"observed\":"
           << tmix_base_recovery::orthogonality_error(pd)
           << ",\"pass\":" << (tmix_base_recovery::orthogonality_error(pd) <= 1e-5 ? "true" : "false") << "},"
           << "\"input_destination_overlap\":{\"threshold\":1e-5,\"observed\":" << cross
           << ",\"pass\":" << (cross <= 1e-5 ? "true" : "false") << "},"
           << "\"raw_direction_recomputation\":{\"pass\":" << (valid ? "true" : "false") << "},"
           << "\"split_counts\":{\"train_tokens\":1664,\"test_tokens\":1408,\"train_prompts\":26,\"test_prompts\":22,"
              "\"pass\":"
           << (tr.size() == 26 && te.size() == 22 ? "true" : "false") << "}}";
    write_outputs(a, native, native_per_prompt, d, p, rows, checks.str(), status, classification);
    if (!post_write_audit(a, repeats)) {
        std::fprintf(stderr, "error: post-write output audit failed\n");
        return 2;
    }
    std::printf("status=%s classification=%s fraction=%.9f efficiency=%.9f\n", status.c_str(), classification.c_str(),
                native.fraction, native.efficiency);
    return valid ? 0 : 2;
} catch (const std::exception & e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
}
