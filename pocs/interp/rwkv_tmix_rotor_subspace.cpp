#include "rwkv_experiment.hpp"

#include "llama-model.h"

#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleWriter.hxx>
#include <TFile.h>
#include <TMatrixD.h>
#include <TMatrixDSym.h>
#include <TMatrixDSymEigen.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <clocale>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

constexpr int k_primary_rank = 32;
constexpr int k_control_repeats = 99;
constexpr double k_tangent_epsilon = 1e-8;

void quiet_llama_logs(ggml_log_level, const char *, void *) {}

struct options {
    std::string model, corpus, rationale, output_root, output_json;
    int layer = -1, ngl = 0, max_tokens = 1024, max_tokens_per_prompt = 128;
    uint64_t seed = 0;
};

struct sample {
    uint64_t prompt_id;
    int position;
    llama_token token;
    bool train;
    std::vector<float> resid_in, time_out, resid_time, channel_out, resid_out;
    std::vector<float> u, w, q, xi, channel_q, channel_xi;
    double angle = 0, radial = 0, tangent_norm = 0, radius = 0;
    double resid_time_error = 0, resid_out_error = 0, orthogonality_error = 0, angle_error = 0, reconstruction_error = 0;
};

double dot(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) throw std::runtime_error("vector size mismatch");
    double result = 0;
    for (size_t i = 0; i < a.size(); ++i) result += (double) a[i] * b[i];
    return result;
}
double norm(const std::vector<float> & a) { return std::sqrt(dot(a, a)); }
void center(std::vector<float> & a) {
    double mean = 0;
    for (float x : a) mean += x;
    mean /= a.size();
    for (float & x : a) x -= (float) mean;
}
std::vector<float> unit(std::vector<float> a) {
    const double n = norm(a);
    if (!(n > 0) || !std::isfinite(n)) throw std::runtime_error("zero or non-finite vector");
    for (float & x : a) x = (float) (x / n);
    return a;
}
double rel_error(const std::vector<float> & actual, const std::vector<float> & expected) {
    std::vector<float> difference(actual.size());
    for (size_t i = 0; i < actual.size(); ++i) difference[i] = actual[i] - expected[i];
    return norm(difference) / std::max(norm(actual), 1e-12);
}
uint64_t fnv1a(uint64_t value, uint64_t seed) {
    uint64_t hash = 1469598103934665603ULL ^ seed;
    for (unsigned shift = 0; shift < 64; shift += 8) { hash ^= (value >> shift) & 0xff; hash *= 1099511628211ULL; }
    return hash;
}
uint64_t split_seed(uint64_t seed, const char * name, int replicate) {
    uint64_t hash = seed;
    for (const char * p = name; *p; ++p) { hash ^= (unsigned char) *p; hash *= 1099511628211ULL; }
    return fnv1a((uint64_t) replicate, hash);
}

const llama_interp_activation & get_capture(const llama_interp::activation_set & captures, const std::string & name) {
    return rwkv_experiment::require_capture(captures, name);
}

llama_interp::task<> capture_token(llama_interp::runtime & runtime, const llama_interp::rwkv_state & before,
        llama_token token, const std::vector<std::string> & taps, llama_interp::rwkv_state & after,
        llama_interp::activation_set & captures) {
    auto call = runtime.prefill_tokens(before, { token });
    for (const auto & tap : taps) call.capture_f32("^" + tap + "$", captures);
    after = co_await call;
}

void usage(const char * argv0) {
    std::fprintf(stderr, "usage: %s -m MODEL --corpus PROMPTS.txt --layer L --layer-selection-rationale TEXT --output-root FILE.root --output-json FILE.json --seed N [-ngl N] [--max-tokens N] [--max-tokens-per-prompt N]\n", argv0);
}

options parse_options(int argc, char ** argv) {
    options out;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-m") && i + 1 < argc) out.model = argv[++i];
        else if (!std::strcmp(argv[i], "--corpus") && i + 1 < argc) out.corpus = argv[++i];
        else if (!std::strcmp(argv[i], "--layer") && i + 1 < argc) out.layer = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--layer-selection-rationale") && i + 1 < argc) out.rationale = argv[++i];
        else if (!std::strcmp(argv[i], "--output-root") && i + 1 < argc) out.output_root = argv[++i];
        else if (!std::strcmp(argv[i], "--output-json") && i + 1 < argc) out.output_json = argv[++i];
        else if (!std::strcmp(argv[i], "--seed") && i + 1 < argc) out.seed = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-ngl") && i + 1 < argc) out.ngl = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--max-tokens") && i + 1 < argc) out.max_tokens = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--max-tokens-per-prompt") && i + 1 < argc) out.max_tokens_per_prompt = std::atoi(argv[++i]);
        else { usage(argv[0]); throw std::runtime_error("unknown or incomplete argument"); }
    }
    if (out.model.empty() || out.corpus.empty() || out.rationale.empty() || out.output_root.empty() || out.output_json.empty() ||
        out.layer < 0 || out.max_tokens < 1 || out.max_tokens_per_prompt < 1 || std::filesystem::exists(out.output_root) || std::filesystem::exists(out.output_json)) {
        throw std::runtime_error("missing required argument, invalid limit, or existing output");
    }
    return out;
}

TMatrixD fit_basis(const std::vector<const std::vector<float> *> & rows, int rank) {
    if ((int) rows.size() < rank) throw std::runtime_error("insufficient samples for requested rank");
    // The Gram eigensystem gives the thin SVD right vectors without a dense width x width covariance.
    TMatrixDSym gram(rows.size());
    for (int i = 0; i < gram.GetNrows(); ++i) for (int j = 0; j <= i; ++j) {
        const double value = dot(*rows[i], *rows[j]);
        gram(i, j) = value;
        if (i != j) gram(j, i) = value;
    }
    TMatrixDSymEigen eigensystem(gram);
    const auto eigenvalues = eigensystem.GetEigenValues();
    const TMatrixD eigenvectors = eigensystem.GetEigenVectors();
    TMatrixD basis(rows[0]->size(), rank);
    for (int component = 0; component < rank; ++component) {
        const int eigen_index = eigenvalues.GetNrows() - 1 - component;
        const double singular = std::sqrt(std::max(0.0, eigenvalues[eigen_index]));
        if (!(singular > 1e-12) || !std::isfinite(singular)) throw std::runtime_error("insufficient nonzero singular values");
        for (int coordinate = 0; coordinate < basis.GetNrows(); ++coordinate) {
            double value = 0;
            for (int row = 0; row < eigenvectors.GetNrows(); ++row) value += (*rows[row])[coordinate] * eigenvectors(row, eigen_index);
            basis(coordinate, component) = value / singular;
        }
    }
    return basis;
}
double energy(const TMatrixD & basis, const std::vector<const std::vector<float> *> & rows) {
    double result = 0;
    for (const auto * row : rows) {
        const double row_norm = norm(*row);
        if (!(row_norm > 0)) throw std::runtime_error("zero analysis vector");
        double projection = 0;
        for (int c = 0; c < basis.GetNcols(); ++c) { double x = 0; for (int r = 0; r < basis.GetNrows(); ++r) x += basis(r, c) * (*row)[r]; projection += x * x; }
        result += projection / (row_norm * row_norm);
    }
    return result / rows.size();
}
double orthogonality(const TMatrixD & basis) {
    double maximum = 0;
    for (int i = 0; i < basis.GetNcols(); ++i) for (int j = 0; j < basis.GetNcols(); ++j) {
        double value = 0; for (int k = 0; k < basis.GetNrows(); ++k) value += basis(k, i) * basis(k, j);
        maximum = std::max(maximum, std::abs(value - (i == j)));
    }
    return maximum;
}
std::vector<const std::vector<float> *> select(const std::vector<sample> & samples, bool train, char representation) {
    std::vector<const std::vector<float> *> result;
    for (const auto & s : samples) if (s.train == train) {
        if (representation == 'U') result.push_back(&s.u); else if (representation == 'W') result.push_back(&s.w);
        else if (representation == 'Q') result.push_back(&s.q); else result.push_back(&s.xi);
    }
    return result;
}
double percentile(std::vector<double> values, double fraction) {
    std::sort(values.begin(), values.end());
    return values[(size_t) std::ceil(fraction * (values.size() - 1))];
}
double p_greater(const std::vector<double> & controls, double native) {
    return (1.0 + std::count_if(controls.begin(), controls.end(), [native](double x) { return x >= native; })) / (1.0 + controls.size());
}

std::vector<std::vector<float>> shuffled_q(const std::vector<sample> & samples, bool train, std::mt19937_64 & rng) {
    std::vector<const sample *> selected;
    for (const auto & s : samples) if (s.train == train) selected.push_back(&s);
    std::vector<size_t> permutation(selected.size()); std::iota(permutation.begin(), permutation.end(), 0); std::shuffle(permutation.begin(), permutation.end(), rng);
    std::vector<std::vector<float>> out; out.reserve(selected.size());
    for (size_t i = 0; i < selected.size(); ++i) {
        std::vector<float> w = selected[permutation[i]]->time_out; center(w);
        const auto & u = selected[i]->u; const double radial = dot(w, u);
        for (size_t j = 0; j < w.size(); ++j) w[j] -= (float) (radial * u[j]);
        if (norm(w) <= k_tangent_epsilon * selected[i]->radius) throw std::runtime_error("degenerate shuffled tangent");
        out.push_back(unit(std::move(w)));
    }
    return out;
}
std::vector<std::vector<float>> random_q(const std::vector<sample> & samples, bool train, std::mt19937_64 & rng) {
    std::normal_distribution<float> normal;
    std::vector<std::vector<float>> out;
    for (const auto & s : samples) if (s.train == train) {
        std::vector<float> g(s.u.size()); for (float & x : g) x = normal(rng); center(g);
        const double radial = dot(g, s.u); for (size_t i = 0; i < g.size(); ++i) g[i] -= (float) (radial * s.u[i]);
        out.push_back(unit(std::move(g)));
    }
    return out;
}
std::vector<std::vector<float>> random_ambient(const std::vector<sample> & samples, bool train, std::mt19937_64 & rng) {
    std::normal_distribution<float> normal;
    std::vector<std::vector<float>> out;
    for (const auto & s : samples) if (s.train == train) {
        std::vector<float> g(s.u.size());
        for (float & x : g) x = normal(rng);
        center(g);
        out.push_back(unit(std::move(g)));
    }
    return out;
}
std::vector<std::vector<float>> shuffled_q_within_prompt(const std::vector<sample> & samples, bool train, std::mt19937_64 & rng) {
    std::vector<std::vector<const sample *>> groups;
    for (const auto & s : samples) if (s.train == train) {
        auto group = std::find_if(groups.begin(), groups.end(), [&s](const auto & value) { return value.front()->prompt_id == s.prompt_id; });
        if (group == groups.end()) groups.push_back({ &s }); else group->push_back(&s);
    }
    std::vector<std::vector<float>> out;
    for (auto & group : groups) {
        std::vector<size_t> permutation(group.size()); std::iota(permutation.begin(), permutation.end(), 0); std::shuffle(permutation.begin(), permutation.end(), rng);
        for (size_t i = 0; i < group.size(); ++i) {
            std::vector<float> w = group[permutation[i]]->time_out; center(w);
            const auto & u = group[i]->u; const double radial = dot(w, u);
            for (size_t j = 0; j < w.size(); ++j) w[j] -= (float) (radial * u[j]);
            if (norm(w) <= k_tangent_epsilon * group[i]->radius) throw std::runtime_error("degenerate within-prompt shuffled tangent");
            out.push_back(unit(std::move(w)));
        }
    }
    return out;
}
std::vector<const std::vector<float> *> pointers(const std::vector<std::vector<float>> & rows) {
    std::vector<const std::vector<float> *> out; for (const auto & row : rows) out.push_back(&row); return out;
}

void write_root(const std::string & path, const std::vector<sample> & samples, int width, int layer,
        double gu, double gw, double gt, const std::vector<double> & isotropic_t) {
    auto model = ROOT::RNTupleModel::Create();
    auto prompt_id = model->MakeField<uint64_t>("prompt_id"); auto split = model->MakeField<std::string>("split");
    auto position = model->MakeField<int32_t>("position"); auto token = model->MakeField<int32_t>("token_id"); auto sample_layer = model->MakeField<int32_t>("layer");
    auto resid_in = model->MakeField<std::vector<float>>("resid_in"); auto time_out = model->MakeField<std::vector<float>>("time_out");
    auto resid_time = model->MakeField<std::vector<float>>("resid_time"); auto channel_out = model->MakeField<std::vector<float>>("channel_out"); auto resid_out = model->MakeField<std::vector<float>>("resid_out");
    auto u = model->MakeField<std::vector<float>>("u_in"); auto q = model->MakeField<std::vector<float>>("q_tmix"); auto xi = model->MakeField<std::vector<float>>("xi_tmix");
    auto angle = model->MakeField<double>("angle"); auto tangent = model->MakeField<double>("tangent_write_norm");
    {
        auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "samples", path);
        for (const auto & s : samples) { *prompt_id = s.prompt_id; *split = s.train ? "train" : "test"; *position = s.position; *token = s.token; *sample_layer = layer;
            *resid_in = s.resid_in; *time_out = s.time_out; *resid_time = s.resid_time; *channel_out = s.channel_out; *resid_out = s.resid_out; *u = s.u; *q = s.q; *xi = s.xi; *angle = s.angle; *tangent = s.tangent_norm; writer->Fill(); }
        writer->CommitCluster();
    }
    TFile file(path.c_str(), "UPDATE");
    if (file.IsZombie()) throw std::runtime_error("failed to reopen ROOT output");
    auto metadata_model = ROOT::RNTupleModel::Create();
    auto metadata_layer = metadata_model->MakeField<int32_t>("layer"); auto metadata_width = metadata_model->MakeField<int32_t>("width");
    {
        auto writer = ROOT::RNTupleWriter::Append(std::move(metadata_model), "metadata", file);
        *metadata_layer = layer; *metadata_width = width; writer->Fill(); writer->CommitCluster();
    }
    auto metrics_model = ROOT::RNTupleModel::Create();
    auto representation = metrics_model->MakeField<std::string>("representation"); auto control = metrics_model->MakeField<std::string>("control_family");
    auto repeat = metrics_model->MakeField<int32_t>("control_replicate"); auto value = metrics_model->MakeField<double>("self_generalization");
    {
        auto writer = ROOT::RNTupleWriter::Append(std::move(metrics_model), "subspace_metrics", file);
        for (const auto & native : std::array<std::pair<const char *, double>, 3>{ { { "U", gu }, { "W", gw }, { "Q", gt } } }) { *representation = native.first; *control = "native"; *repeat = -1; *value = native.second; writer->Fill(); }
        for (size_t i = 0; i < isotropic_t.size(); ++i) { *representation = "Q"; *control = "isotropic_tangent"; *repeat = i; *value = isotropic_t[i]; writer->Fill(); }
        writer->CommitCluster();
    }
    file.Write();
    file.Close();
}

void geometry_self_test() {
    std::vector<float> x = { 1, -1, 0 }, w = { 0, 1, -1 }; center(x); center(w);
    const auto u = unit(x); const double radial = dot(w, u); for (size_t i = 0; i < w.size(); ++i) w[i] -= (float) (radial * u[i]);
    if (std::abs(dot(u, w)) > 1e-6) throw std::runtime_error("startup tangent orthogonality self-test failed");
    if (fnv1a(7, 11) != fnv1a(7, 11)) throw std::runtime_error("startup hash self-test failed");
}

} // namespace

int main(int argc, char ** argv) try {
    std::setlocale(LC_NUMERIC, "C"); llama_log_set(quiet_llama_logs, nullptr); geometry_self_test();
    const options opt = parse_options(argc, argv);
    std::ifstream input(opt.corpus, std::ios::binary); if (!input) throw std::runtime_error("failed to open corpus");
    std::vector<std::string> prompts; std::string line;
    while (std::getline(input, line)) if (!line.empty()) prompts.push_back(line);
    if (prompts.empty()) throw std::runtime_error("corpus has no nonempty prompts");
    ggml_backend_load_all(); llama_backend_init();
    llama_model_params mp = llama_model_default_params(); mp.n_gpu_layers = opt.ngl;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(opt.model.c_str(), mp), llama_model_free);
    if (!model || model->arch != LLM_ARCH_RWKV7) throw std::runtime_error("an RWKV-7 model is required");
    if (opt.layer >= model->hparams.n_layer()) throw std::runtime_error("layer is outside model depth");
    llama_context_params cp = llama_context_default_params(); cp.n_ctx = std::max(512, opt.max_tokens_per_prompt + 8); cp.n_batch = cp.n_ctx; cp.n_ubatch = cp.n_ctx;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
    if (!ctx) throw std::runtime_error("failed to create context");
    llama_interp::runtime runtime(ctx.get(), 1); const llama_vocab * vocab = llama_model_get_vocab(model.get());
    const std::string prefix = "rwkv.layer." + std::to_string(opt.layer) + ".";
    const std::vector<std::string> taps = { prefix + "resid.in", prefix + "time.out", prefix + "resid.time", prefix + "channel.out", prefix + "resid.out" };
    std::vector<sample> samples; size_t degenerate = 0; int width = 0;
    std::printf("phase=capture layer=%d max_tokens=%d max_tokens_per_prompt=%d\n", opt.layer, opt.max_tokens, opt.max_tokens_per_prompt);
    std::fflush(stdout);
    for (uint64_t prompt_id = 0; prompt_id < prompts.size() && (int) samples.size() < opt.max_tokens; ++prompt_id) {
        const auto tokens = common_tokenize(vocab, prompts[prompt_id], false, true); if (tokens.empty()) continue;
        llama_interp::rwkv_state state = runtime.make_state(); const bool train = (fnv1a(prompt_id, opt.seed) >> 63) == 0;
        for (size_t position = 0; position < tokens.size() && position < (size_t) opt.max_tokens_per_prompt && (int) samples.size() < opt.max_tokens; ++position) {
            llama_interp::activation_set captures; llama_interp::rwkv_state after;
            auto task = capture_token(runtime, state, tokens[position], taps, after, captures); runtime.run(); task.rethrow_if_failed(); state = std::move(after);
            sample s; s.prompt_id = prompt_id; s.position = position; s.token = tokens[position]; s.train = train;
            s.resid_in = get_capture(captures, taps[0]).data_f32; s.time_out = get_capture(captures, taps[1]).data_f32; s.resid_time = get_capture(captures, taps[2]).data_f32; s.channel_out = get_capture(captures, taps[3]).data_f32; s.resid_out = get_capture(captures, taps[4]).data_f32;
            width = s.resid_in.size(); std::vector<float> expected_time(width), expected_out(width);
            for (int i = 0; i < width; ++i) { expected_time[i] = s.resid_in[i] + s.time_out[i]; expected_out[i] = s.resid_time[i] + s.channel_out[i]; }
            s.resid_time_error = rel_error(s.resid_time, expected_time); s.resid_out_error = rel_error(s.resid_out, expected_out);
            std::vector<float> x = s.resid_in, w = s.time_out, y = s.resid_time; center(x); center(w); center(y); s.radius = norm(x);
            if (!(s.radius > 0) || norm(y) == 0) throw std::runtime_error("degenerate centered residual");
            s.u = unit(x); const auto u_time = unit(y); s.radial = dot(w, s.u); for (int i = 0; i < width; ++i) w[i] -= (float) (s.radial * s.u[i]); s.tangent_norm = norm(w);
            if (s.tangent_norm <= k_tangent_epsilon * s.radius) { ++degenerate; continue; }
            s.q = unit(w); const double acos_angle = std::acos(std::clamp(dot(s.u, u_time), -1.0, 1.0)); s.angle = std::atan2(s.tangent_norm, s.radius + s.radial);
            s.xi = s.q; for (float & v : s.xi) v = (float) (v * s.angle); s.w = s.time_out; center(s.w); s.w = unit(std::move(s.w));
            s.orthogonality_error = std::abs(dot(s.u, s.q)); s.angle_error = std::abs(acos_angle - s.angle); std::vector<float> reconstructed(width);
            for (int i = 0; i < width; ++i) reconstructed[i] = (float) (std::cos(s.angle) * s.u[i] + std::sin(s.angle) * s.q[i]); s.reconstruction_error = rel_error(u_time, reconstructed);
            samples.push_back(std::move(s));
            if (samples.size() % 16 == 0 || samples.size() == (size_t) opt.max_tokens) {
                std::printf("phase=capture samples=%zu/%d prompt=%llu position=%zu\n", samples.size(), opt.max_tokens, (unsigned long long) prompt_id, position);
                std::fflush(stdout);
            }
        }
    }
    const auto train_u = select(samples, true, 'U'), test_u = select(samples, false, 'U'), train_w = select(samples, true, 'W'), test_w = select(samples, false, 'W'), train_q = select(samples, true, 'Q'), test_q = select(samples, false, 'Q');
    std::unordered_set<uint64_t> train_prompts, test_prompts; for (const auto & s : samples) (s.train ? train_prompts : test_prompts).insert(s.prompt_id);
    bool valid = train_u.size() >= 256 && test_u.size() >= 256 && train_prompts.size() >= 4 && test_prompts.size() >= 4 && degenerate <= samples.size() / 100;
    double max_identity = 0, max_geometry = 0; for (const auto & s : samples) { max_identity = std::max({ max_identity, s.resid_time_error, s.resid_out_error }); max_geometry = std::max({ max_geometry, s.orthogonality_error, s.angle_error, s.reconstruction_error }); }
    valid = valid && max_identity <= 1e-3 && max_geometry <= 1e-5 && train_u.size() > 64 && test_u.size() > 64;
    double gu = 0, gw = 0, gt = 0, complementarity = 0, pairing_gap = 0, p_pairing = 1, p_shuffle_self = 1, p_complementarity = 1;
    std::vector<double> ambient_u, ambient_w, isotropic_t, pairing, shuffle_self, shuffled_complementarity, within_prompt_self;
    if (valid) {
        std::printf("phase=native_subspaces train=%zu test=%zu rank=%d\n", train_u.size(), test_u.size(), k_primary_rank);
        std::fflush(stdout);
        const TMatrixD pu = fit_basis(train_u, k_primary_rank), pw = fit_basis(train_w, k_primary_rank), pt = fit_basis(train_q, k_primary_rank);
        valid = orthogonality(pu) <= 1e-5 && orthogonality(pw) <= 1e-5 && orthogonality(pt) <= 1e-5;
        gu = energy(pu, test_u); gw = energy(pw, test_w); gt = energy(pt, test_q); complementarity = std::min(gt - energy(pu, test_q), gu - energy(pt, test_u));
        for (int repeat = 0; repeat < k_control_repeats; ++repeat) {
            std::printf("phase=controls replicate=%d/%d\n", repeat + 1, k_control_repeats);
            std::fflush(stdout);
            std::mt19937_64 rng(split_seed(opt.seed, "controls", repeat));
            auto shuffled_train = shuffled_q(samples, true, rng); auto shuffled_test = shuffled_q(samples, false, rng); auto pst = fit_basis(pointers(shuffled_train), k_primary_rank);
            pairing.push_back(energy(pt, pointers(shuffled_test))); shuffle_self.push_back(energy(pst, pointers(shuffled_test)));
            shuffled_complementarity.push_back(std::min(energy(pst, pointers(shuffled_test)) - energy(pu, pointers(shuffled_test)), gu - energy(pst, test_u)));
            auto within_train = shuffled_q_within_prompt(samples, true, rng); auto within_test = shuffled_q_within_prompt(samples, false, rng);
            within_prompt_self.push_back(energy(fit_basis(pointers(within_train), k_primary_rank), pointers(within_test)));
            auto random_train = random_q(samples, true, rng); auto random_test = random_q(samples, false, rng); isotropic_t.push_back(energy(fit_basis(pointers(random_train), k_primary_rank), pointers(random_test)));
            auto random_u_train = random_ambient(samples, true, rng); auto random_u_test = random_ambient(samples, false, rng);
            auto random_w_train = random_ambient(samples, true, rng); auto random_w_test = random_ambient(samples, false, rng);
            ambient_u.push_back(energy(fit_basis(pointers(random_u_train), k_primary_rank), pointers(random_u_test)));
            ambient_w.push_back(energy(fit_basis(pointers(random_w_train), k_primary_rank), pointers(random_w_test)));
        }
        pairing_gap = gt - percentile(pairing, .5); p_pairing = p_greater(pairing, gt); p_shuffle_self = p_greater(shuffle_self, gt); p_complementarity = p_greater(shuffled_complementarity, complementarity);
    }
    std::string classification = "suppressed";
    if (valid) {
        if (gu > percentile(ambient_u, .95) && gt > percentile(isotropic_t, .95) && pairing_gap > 0 && p_pairing <= .05 && p_shuffle_self <= .05 && complementarity > 0 && complementarity > percentile(shuffled_complementarity, .95) && p_complementarity <= .05) classification = "residual_relative_tangent_structure";
        else if (gw > percentile(ambient_w, .95) && gt > percentile(isotropic_t, .95) && p_pairing > .10 && p_shuffle_self > .10 && complementarity <= percentile(shuffled_complementarity, .95)) classification = "global_additive_structure_consistent";
        else if (gw <= percentile(ambient_w, .95) && gt <= percentile(isotropic_t, .95)) classification = "no_detected_low_rank_structure";
        else classification = "ambiguous";
    }
    std::printf("phase=write_artifacts status=%s\n", valid ? "valid" : "invalid");
    std::fflush(stdout);
    const std::filesystem::path root_parent = std::filesystem::path(opt.output_root).parent_path();
    if (!root_parent.empty()) std::filesystem::create_directories(root_parent);
    write_root(opt.output_root, samples, width, opt.layer, gu, gw, gt, isotropic_t);
    std::ofstream json(opt.output_json); if (!json) throw std::runtime_error("failed to create JSON output"); json << std::setprecision(10);
    json << "{\n  \"schema_version\": 1,\n  \"experiment\": \"rwkv_tmix_rotor_subspace\",\n  \"status\": \"" << (valid ? "valid" : "invalid_insufficient_samples") << "\",\n  \"classification\": \"" << classification << "\",\n  \"scientific_question\": \"Are native TMix tangents held-out-stable, complementary to incoming residuals, and dependent on the native residual/write pairing?\",\n  \"maximum_permitted_claim\": \"Distributional organization at one layer only; no semantic, memory-content, causal-use, logit, or output-embedding claim.\",\n  \"metadata\": {\"model\": "; rwkv_experiment::write_json_string(json, opt.model); json << ", \"corpus\": "; rwkv_experiment::write_json_string(json, opt.corpus); json << ", \"layer\": " << opt.layer << ", \"layer_selection_rationale\": "; rwkv_experiment::write_json_string(json, opt.rationale); json << ", \"seed\": " << opt.seed << ", \"primary_rank\": 32, \"control_repeats\": 99, \"split_hash\": \"FNV-1a-64\"},\n  \"sample_counts\": {\"valid_tokens\": " << samples.size() << ", \"degenerate_tokens\": " << degenerate << ", \"train_tokens\": " << train_u.size() << ", \"test_tokens\": " << test_u.size() << ", \"train_prompts\": " << train_prompts.size() << ", \"test_prompts\": " << test_prompts.size() << "},\n  \"sanity_checks\": {\"max_residual_identity_error\": " << max_identity << ", \"max_geometry_error\": " << max_geometry << "},\n  \"primary_metrics\": {\"G_U\": " << gu << ", \"G_W\": " << gw << ", \"G_T\": " << gt << ", \"pairing_gap\": " << pairing_gap << ", \"p_pairing\": " << p_pairing << ", \"p_shuffle_self\": " << p_shuffle_self << ", \"complementarity\": " << complementarity << ", \"p_complementarity\": " << p_complementarity << "},\n  \"control_distributions\": {\"ambient_U_p95\": " << (ambient_u.empty() ? 0 : percentile(ambient_u, .95)) << ", \"ambient_W_p95\": " << (ambient_w.empty() ? 0 : percentile(ambient_w, .95)) << ", \"isotropic_tangent_p95\": " << (isotropic_t.empty() ? 0 : percentile(isotropic_t, .95)) << ", \"global_shuffle_complementarity_p95\": " << (shuffled_complementarity.empty() ? 0 : percentile(shuffled_complementarity, .95)) << ", \"within_prompt_shuffle_self_median\": " << (within_prompt_self.empty() ? 0 : percentile(within_prompt_self, .5)) << "},\n  \"root_artifact\": "; rwkv_experiment::write_json_string(json, opt.output_root); json << "\n}\n";
    if (!json) throw std::runtime_error("failed to write JSON");
    std::printf("status=%s\nclassification=%s\nsamples=%zu train=%zu test=%zu\nprimary_rank=32\nG_U=%.6f G_W=%.6f G_T=%.6f\npairing_gap=%.6f p_pairing=%.4f p_shuffle_self=%.4f\ncomplementarity=%.6f p=%.4f\nROOT=%s\nJSON=%s\n", valid ? "valid" : "invalid", classification.c_str(), samples.size(), train_u.size(), test_u.size(), gu, gw, gt, pairing_gap, p_pairing, p_shuffle_self, complementarity, p_complementarity, opt.output_root.c_str(), opt.output_json.c_str());
    llama_backend_free(); return valid ? 0 : 2;
} catch (const std::exception & error) { std::fprintf(stderr, "error: %s\n", error.what()); return 1; }
