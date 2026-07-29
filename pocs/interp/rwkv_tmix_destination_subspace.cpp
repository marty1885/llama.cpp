#include "rwkv_experiment.hpp"
#include "llama-model.h"

#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleWriter.hxx>
#include <TFile.h>
#include <TMatrixD.h>
#include <TMatrixDSym.h>
#include <TMatrixDSymEigen.h>

#include <cblas.h>
#include <openblas/lapacke.h>

#include <algorithm>
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
#include <regex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {
constexpr int k_layer = 30, k_primary_rank = 32, k_repeats = 99;
constexpr double k_min_norm = 1e-12;

struct options { std::string model, corpus, manifest, root, json; uint64_t seed = 0; int ngl = 0, layer = k_layer, max_tokens = 2048, per_prompt = 128, repeats = k_repeats; bool overwrite = false, exploratory = false; };
struct sample {
    uint64_t prompt_id; bool train; int position; llama_token token;
    std::vector<float> resid_in, time_out, resid_time, u_in, u_time, w_hat;
    double radius = 0, write_norm = 0, identity_error = 0, direction_error = 0;
};
struct metrics { double g = 0, ein = 0, etime = 0, d = 0, input_self = 0, write_self = 0, iw = 0, wd = 0, id = 0, orth_in = 0, orth_dest = 0; };
struct distribution { std::vector<double> values; double mean = 0, sd = 0, median = 0, p5 = 0, p95 = 0, min = 0, max = 0; };

void quiet_logs(ggml_log_level, const char *, void *) {}
uint64_t fnv_bytes(const void * data, size_t size, uint64_t h = 1469598103934665603ULL) { const auto * p = static_cast<const uint8_t *>(data); for (size_t i = 0; i < size; ++i) { h ^= p[i]; h *= 1099511628211ULL; } return h; }
uint64_t prompt_hash(uint64_t index, const std::string & text) { uint8_t n[8]; for (int i = 0; i < 8; ++i) n[i] = index >> (8 * i); return fnv_bytes(text.data(), text.size(), fnv_bytes(n, sizeof(n))); }
uint64_t sub_seed(uint64_t seed, const char * family, int replicate) { return fnv_bytes(&replicate, sizeof(replicate), fnv_bytes(family, std::strlen(family), seed)); }
double dot(const std::vector<float> & a, const std::vector<float> & b) { double r = 0; for (size_t i = 0; i < a.size(); ++i) r += (double) a[i] * b[i]; return r; }
double norm(const std::vector<float> & a) { return std::sqrt(dot(a, a)); }
double cosine(const std::vector<float> & a, const std::vector<float> & b) { return dot(a, b) / std::max(norm(a) * norm(b), k_min_norm); }
void center(std::vector<float> & a) { double m = 0; for (float x : a) m += x; m /= a.size(); for (float & x : a) x -= (float) m; }
std::vector<float> unit(std::vector<float> a) { const double n = norm(a); if (!(n > k_min_norm) || !std::isfinite(n)) throw std::runtime_error("zero or non-finite analysis vector"); for (float & x : a) x = (float) (x / n); return a; }
double rel_error(const std::vector<float> & a, const std::vector<float> & b) { std::vector<float> d(a.size()); for (size_t i = 0; i < a.size(); ++i) d[i] = a[i] - b[i]; return norm(d) / std::max(norm(a), k_min_norm); }

void usage(const char * p) { std::fprintf(stderr, "usage: %s -m MODEL --corpus PROMPTS.txt --split-manifest MANIFEST.json --output-root FILE.root --output-json FILE.json --seed N [-ngl N] [--max-tokens N] [--max-tokens-per-prompt N] [--control-repeats N] [--overwrite]\n", p); }
options parse(int argc, char ** argv) {
    options o;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-m") && i + 1 < argc) o.model = argv[++i];
        else if (!std::strcmp(argv[i], "--corpus") && i + 1 < argc) o.corpus = argv[++i];
        else if (!std::strcmp(argv[i], "--split-manifest") && i + 1 < argc) o.manifest = argv[++i];
        else if (!std::strcmp(argv[i], "--output-root") && i + 1 < argc) o.root = argv[++i];
        else if (!std::strcmp(argv[i], "--output-json") && i + 1 < argc) o.json = argv[++i];
        else if (!std::strcmp(argv[i], "--seed") && i + 1 < argc) o.seed = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-ngl") && i + 1 < argc) o.ngl = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--max-tokens") && i + 1 < argc) o.max_tokens = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--max-tokens-per-prompt") && i + 1 < argc) o.per_prompt = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--control-repeats") && i + 1 < argc) o.repeats = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--layer") && i + 1 < argc) { o.layer = std::atoi(argv[++i]); o.exploratory = o.layer != k_layer; }
        else if (!std::strcmp(argv[i], "--overwrite")) o.overwrite = true;
        else { usage(argv[0]); throw std::runtime_error("unknown or incomplete argument"); }
    }
    if (o.model.empty() || o.corpus.empty() || o.manifest.empty() || o.root.empty() || o.json.empty() || o.layer < 0 || o.max_tokens < 1 || o.per_prompt < 1 || o.repeats != k_repeats) throw std::runtime_error("missing required argument or non-pre-registered limit");
    if (!o.overwrite && (std::filesystem::exists(o.root) || std::filesystem::exists(o.json))) throw std::runtime_error("output exists; use --overwrite (recorded protocol deviation)");
    return o;
}

std::unordered_map<uint64_t, bool> read_manifest(const std::string & path) {
    std::ifstream f(path, std::ios::binary); if (!f) throw std::runtime_error("failed to open split manifest");
    const std::string s((std::istreambuf_iterator<char>(f)), {});
    std::unordered_map<uint64_t, bool> out;
    const std::regex entries("\\\"prompt_id\\\"\\s*:\\s*\\\"?([0-9]+)\\\"?[^}]*\\\"split\\\"\\s*:\\s*\\\"(train|test)\\\"");
    for (std::sregex_iterator i(s.begin(), s.end(), entries), e; i != e; ++i) out.emplace(std::strtoull((*i)[1].str().c_str(), nullptr, 10), (*i)[2] == "train");
    if (out.empty()) throw std::runtime_error("manifest must assign every prompt with prompt_id and split=train|test");
    return out;
}

TMatrixD fit_basis(const std::vector<const std::vector<float> *> & rows, int rank) {
    if ((int) rows.size() < rank) throw std::runtime_error("insufficient rows for rank");
    const int sample_count = rows.size(), width = rows[0]->size();
    // A A^T is only samples x samples; OpenBLAS/LAPACK avoids ROOT's unstable large Gram eigensolver.
    std::vector<float> matrix((size_t) sample_count * width), gram((size_t) sample_count * sample_count), eigenvalues(sample_count);
    for (int column = 0; column < width; ++column) for (int row = 0; row < sample_count; ++row) matrix[row + (size_t) sample_count * column] = (*rows[row])[column];
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, sample_count, sample_count, width, 1.0f, matrix.data(), sample_count, matrix.data(), sample_count, 0.0f, gram.data(), sample_count);
    if (LAPACKE_ssyevd(LAPACK_COL_MAJOR, 'V', 'U', sample_count, gram.data(), sample_count, eigenvalues.data()) != 0) throw std::runtime_error("OpenBLAS symmetric eigendecomposition failed");
    TMatrixD p(width, rank);
    for (int c = 0; c < rank; ++c) {
        const int k = sample_count - 1 - c;
        const double singular = std::sqrt(std::max(0.0f, eigenvalues[k]));
        if (!(singular > 1e-15) || !std::isfinite(singular)) throw std::runtime_error("rank-deficient train matrix at component " + std::to_string(c));
        for (int r = 0; r < p.GetNrows(); ++r) { double x = 0; for (int n = 0; n < sample_count; ++n) x += (*rows[n])[r] * gram[n + (size_t) sample_count * k]; p(r, c) = x / singular; }
        // Gram reconstruction accumulates roundoff at the larger sample count; restore an orthonormal basis.
        for (int prior = 0; prior < c; ++prior) { double projection = 0; for (int r = 0; r < p.GetNrows(); ++r) projection += p(r, prior) * p(r, c); for (int r = 0; r < p.GetNrows(); ++r) p(r, c) -= projection * p(r, prior); }
        double column_norm = 0; for (int r = 0; r < p.GetNrows(); ++r) column_norm += p(r, c) * p(r, c); column_norm = std::sqrt(column_norm);
        if (!(column_norm > 1e-12)) throw std::runtime_error("rank-deficient reconstructed basis at component " + std::to_string(c));
        for (int r = 0; r < p.GetNrows(); ++r) p(r, c) /= column_norm;
    }
    return p;
}
double orth(const TMatrixD & p) { double m = 0; for (int i = 0; i < p.GetNcols(); ++i) for (int j = 0; j < p.GetNcols(); ++j) { double x = 0; for (int r = 0; r < p.GetNrows(); ++r) x += p(r, i) * p(r, j); m = std::max(m, std::abs(x - (i == j))); } return m; }
double projection(const TMatrixD & p, const std::vector<float> & x) { double e = 0; for (int c = 0; c < p.GetNcols(); ++c) { double z = 0; for (int r = 0; r < p.GetNrows(); ++r) z += p(r, c) * x[r]; e += z * z; } return e; }
std::vector<float> residualize(const TMatrixD & p, std::vector<float> x) { for (int c = 0; c < p.GetNcols(); ++c) { double z = 0; for (int r = 0; r < p.GetNrows(); ++r) z += p(r, c) * x[r]; for (int r = 0; r < p.GetNrows(); ++r) x[r] -= (float) (z * p(r, c)); } return x; }
double mean_energy(const TMatrixD & p, const std::vector<const std::vector<float> *> & rows) { double x = 0; for (auto r : rows) x += projection(p, *r); return x / rows.size(); }
double overlap(const TMatrixD & a, const TMatrixD & b) { double x = 0; for (int i = 0; i < a.GetNcols(); ++i) for (int j = 0; j < b.GetNcols(); ++j) { double z = 0; for (int r = 0; r < a.GetNrows(); ++r) z += a(r, i) * b(r, j); x += z * z; } return x / a.GetNcols(); }
std::vector<const std::vector<float> *> refs(const std::vector<std::vector<float>> & x) { std::vector<const std::vector<float> *> r; for (const auto & v : x) r.push_back(&v); return r; }
std::vector<const std::vector<float> *> select(const std::vector<sample> & s, bool train, int which) { std::vector<const std::vector<float> *> r; for (const auto & x : s) if (x.train == train) r.push_back(which == 0 ? &x.u_in : which == 1 ? &x.u_time : &x.w_hat); return r; }
TMatrixD destination(const TMatrixD & in, const std::vector<const std::vector<float> *> & time, int rank) { std::vector<std::vector<float>> z; for (auto x : time) { auto v = residualize(in, *x); if (norm(v) > k_min_norm) z.push_back(unit(std::move(v))); } if ((int) z.size() < rank) throw std::runtime_error("destination residualization retained fewer rows than rank"); return fit_basis(refs(z), rank); }

std::vector<std::vector<float>> control_outputs(const std::vector<sample> & all, bool train, const char * family, std::mt19937_64 & rng) {
    std::vector<const sample *> s; for (const auto & x : all) if (x.train == train) s.push_back(&x);
    std::vector<size_t> perm(s.size()); std::iota(perm.begin(), perm.end(), 0);
    if (std::strcmp(family, "within_prompt_shuffle") == 0) { size_t begin = 0; while (begin < s.size()) { size_t end = begin + 1; while (end < s.size() && s[end]->prompt_id == s[begin]->prompt_id) ++end; std::shuffle(perm.begin() + begin, perm.begin() + end, rng); begin = end; } } else if (std::strcmp(family, "global_shuffle") == 0) std::shuffle(perm.begin(), perm.end(), rng);
    std::normal_distribution<float> normal; std::vector<std::vector<float>> out; out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) { std::vector<float> w; if (std::strcmp(family, "isotropic_write") == 0) { w.resize(s[i]->u_in.size()); for (float & v : w) v = normal(rng); center(w); w = unit(std::move(w)); for (float & v : w) v = (float) (v * s[i]->write_norm); } else { w = s[perm[i]]->time_out; center(w); } std::vector<float> y = s[i]->resid_in; center(y); for (size_t j = 0; j < y.size(); ++j) y[j] += w[j]; out.push_back(unit(std::move(y))); }
    return out;
}
distribution summarize(std::vector<double> x) { distribution d; d.values = std::move(x); if (d.values.empty()) return d; std::sort(d.values.begin(), d.values.end()); d.min = d.values.front(); d.max = d.values.back(); auto at = [&d](double q) { return d.values[(size_t) std::ceil(q * (d.values.size() - 1))]; }; d.p5 = at(.05); d.median = at(.5); d.p95 = at(.95); for (double v : d.values) d.mean += v; d.mean /= d.values.size(); for (double v : d.values) d.sd += (v - d.mean) * (v - d.mean); d.sd = std::sqrt(d.sd / d.values.size()); return d; }
double p_greater(const distribution & d, double x) { return (1.0 + std::count_if(d.values.begin(), d.values.end(), [x](double v) { return v >= x; })) / (1.0 + d.values.size()); }

llama_interp::task<> capture_token(llama_interp::runtime & runtime, const llama_interp::rwkv_state & before, llama_token token,
        const std::vector<std::string> & taps, llama_interp::rwkv_state & after, llama_interp::activation_set & captures) {
    auto call = runtime.prefill_tokens(before, { token });
    for (const auto & tap : taps) call.capture_f32("^" + tap + "$", captures);
    after = co_await call;
}

void write_root(const std::string & path, const std::vector<sample> & s, const std::vector<std::tuple<int, std::string, int, metrics, uint64_t>> & rows, const std::vector<std::tuple<int, std::string, int, std::string, double, uint64_t>> & controls, const std::string & status, const std::string & classification) {
    auto sm = ROOT::RNTupleModel::Create(); auto pid = sm->MakeField<uint64_t>("prompt_id"); auto split = sm->MakeField<std::string>("split"); auto pos = sm->MakeField<int32_t>("position"); auto tok = sm->MakeField<int32_t>("token_id"); auto rin = sm->MakeField<std::vector<float>>("resid_in"); auto tout = sm->MakeField<std::vector<float>>("time_out"); auto rtime = sm->MakeField<std::vector<float>>("resid_time"); auto uin = sm->MakeField<std::vector<float>>("u_in"); auto utime = sm->MakeField<std::vector<float>>("u_time"); auto what = sm->MakeField<std::vector<float>>("w_hat"); auto radius = sm->MakeField<double>("radius_in"); auto wn = sm->MakeField<double>("write_norm"); auto ie = sm->MakeField<double>("identity_error");
    { auto w = ROOT::RNTupleWriter::Recreate(std::move(sm), "samples", path); for (const auto & x : s) { *pid=x.prompt_id; *split=x.train?"train":"test"; *pos=x.position; *tok=x.token; *rin=x.resid_in; *tout=x.time_out; *rtime=x.resid_time; *uin=x.u_in; *utime=x.u_time; *what=x.w_hat; *radius=x.radius; *wn=x.write_norm; *ie=x.identity_error; w->Fill(); } w->CommitCluster(); }
    TFile f(path.c_str(), "UPDATE");
    auto mm=ROOT::RNTupleModel::Create(); auto rank=mm->MakeField<int32_t>("rank"); auto family=mm->MakeField<std::string>("control_family"); auto rep=mm->MakeField<int32_t>("replicate"); auto g=mm->MakeField<double>("G_dest"); auto ein=mm->MakeField<double>("E_input"); auto et=mm->MakeField<double>("E_time"); auto d=mm->MakeField<double>("D"); auto seed=mm->MakeField<uint64_t>("sub_seed"); { auto w=ROOT::RNTupleWriter::Append(std::move(mm),"destination_metrics",f); for (const auto & x:rows) { *rank=std::get<0>(x); *family=std::get<1>(x); *rep=std::get<2>(x); const auto & m=std::get<3>(x); *g=m.g; *ein=m.ein; *et=m.etime; *d=m.d; *seed=std::get<4>(x); w->Fill(); } w->CommitCluster(); }
    auto cm=ROOT::RNTupleModel::Create(); auto cr=cm->MakeField<int32_t>("rank"); auto cf=cm->MakeField<std::string>("control_family"); auto cp=cm->MakeField<int32_t>("replicate"); auto name=cm->MakeField<std::string>("metric_name"); auto value=cm->MakeField<double>("metric_value"); auto cs=cm->MakeField<uint64_t>("sub_seed"); { auto w=ROOT::RNTupleWriter::Append(std::move(cm),"control_values",f); for (const auto & x:controls) { *cr=std::get<0>(x); *cf=std::get<1>(x); *cp=std::get<2>(x); *name=std::get<3>(x); *value=std::get<4>(x); *cs=std::get<5>(x); w->Fill(); } w->CommitCluster(); }
    auto md=ROOT::RNTupleModel::Create(); auto st=md->MakeField<std::string>("status"); auto cl=md->MakeField<std::string>("classification"); { auto w=ROOT::RNTupleWriter::Append(std::move(md),"metadata",f); *st=status; *cl=classification; w->Fill(); w->CommitCluster(); } f.Close();
}
} // namespace

int main(int argc, char ** argv) try {
    std::setlocale(LC_NUMERIC, "C"); llama_log_set(quiet_logs, nullptr); const options o=parse(argc,argv); const auto membership=read_manifest(o.manifest);
    std::ifstream corpus(o.corpus, std::ios::binary); if (!corpus) throw std::runtime_error("failed to open corpus"); std::vector<std::pair<uint64_t,std::string>> prompts; std::string line; uint64_t index=0; while (std::getline(corpus,line)) { if (!line.empty()) prompts.emplace_back(prompt_hash(index,line),line); ++index; }
    ggml_backend_load_all(); llama_backend_init(); llama_model_params mp=llama_model_default_params(); mp.n_gpu_layers=o.ngl; std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(o.model.c_str(),mp),llama_model_free); if (!model || model->arch != LLM_ARCH_RWKV7 || model->hparams.n_layer() <= o.layer) throw std::runtime_error("RWKV-7 model does not contain requested layer"); llama_context_params cp=llama_context_default_params(); cp.n_ctx=std::max(512,o.per_prompt+8); cp.n_batch=cp.n_ctx; cp.n_ubatch=cp.n_ctx; std::unique_ptr<llama_context,decltype(&llama_free)> ctx(llama_init_from_model(model.get(),cp),llama_free); if(!ctx) throw std::runtime_error("context creation failed"); llama_interp::runtime runtime(ctx.get(),1); const llama_vocab * vocab=llama_model_get_vocab(model.get()); const std::string prefix="rwkv.layer."+std::to_string(o.layer)+"."; const std::vector<std::string> taps={prefix+"resid.in",prefix+"time.out",prefix+"resid.time"};
    std::vector<sample> samples; size_t excluded=0; int width=0, repeat_positions=0; double min_repeat_cosine=1, max_repeat_norm_difference=0; std::printf("phase=capture layer=%d exploratory=%s max_tokens=%d max_tokens_per_prompt=%d\n", o.layer, o.exploratory ? "true" : "false", o.max_tokens, o.per_prompt); std::fflush(stdout);
    for (size_t prompt_index = 0; prompt_index < prompts.size() && (int) samples.size() < o.max_tokens; ++prompt_index) {
        const auto & prompt = prompts[prompt_index];
        const auto it = membership.find(prompt.first);
        if (it == membership.end()) throw std::runtime_error("corpus prompt missing from split manifest");
        const auto tokens = common_tokenize(vocab, prompt.second, false, true);
        llama_interp::rwkv_state state = runtime.make_state();
        for (size_t p = 0; p < tokens.size() && p < (size_t) o.per_prompt && (int) samples.size() < o.max_tokens; ++p) {
            llama_interp::activation_set c; llama_interp::rwkv_state after;
            auto task = capture_token(runtime, state, tokens[p], taps, after, c); runtime.run(); task.rethrow_if_failed();
            if (prompt_index < 2 && p < 8) {
                llama_interp::activation_set repeated; llama_interp::rwkv_state repeated_after;
                auto repeat_task = capture_token(runtime, state, tokens[p], taps, repeated_after, repeated); runtime.run(); repeat_task.rethrow_if_failed();
                for (const auto & tap : taps) {
                    const auto & first = rwkv_experiment::require_capture(c, tap).data_f32;
                    const auto & second = rwkv_experiment::require_capture(repeated, tap).data_f32;
                    min_repeat_cosine = std::min(min_repeat_cosine, cosine(first, second));
                    max_repeat_norm_difference = std::max(max_repeat_norm_difference, std::abs(norm(first) - norm(second)) / std::max(norm(first), k_min_norm));
                }
                ++repeat_positions;
            }
            state = std::move(after);
            sample x{prompt.first, it->second, (int) p, tokens[p]}; x.resid_in = rwkv_experiment::require_capture(c, taps[0]).data_f32; x.time_out = rwkv_experiment::require_capture(c, taps[1]).data_f32; x.resid_time = rwkv_experiment::require_capture(c, taps[2]).data_f32; width = x.resid_in.size(); std::vector<float> expected(width); for (int j = 0; j < width; ++j) expected[j] = x.resid_in[j] + x.time_out[j]; x.identity_error = rel_error(x.resid_time, expected); auto a = x.resid_in, b = x.time_out, y = x.resid_time; center(a); center(b); center(y); x.radius = norm(a); x.write_norm = norm(b); if (!(x.radius > k_min_norm && x.write_norm > k_min_norm && norm(y) > k_min_norm)) { ++excluded; continue; } x.u_in = unit(std::move(a)); x.w_hat = unit(std::move(b)); x.u_time = unit(std::move(y)); x.direction_error = std::max({std::abs(norm(x.u_in) - 1), std::abs(norm(x.w_hat) - 1), std::abs(norm(x.u_time) - 1)}); samples.push_back(std::move(x));
        }
        std::printf("phase=capture prompt=%zu/%zu samples=%zu\n", prompt_index + 1, prompts.size(), samples.size()); std::fflush(stdout);
    }
    const auto train_in=select(samples,true,0), test_in=select(samples,false,0), train_time=select(samples,true,1), test_time=select(samples,false,1), train_write=select(samples,true,2), test_write=select(samples,false,2); std::unordered_set<uint64_t> tr,te; for(const auto & x:samples)(x.train?tr:te).insert(x.prompt_id); std::printf("phase=capture_complete samples=%zu train=%zu test=%zu train_prompts=%zu test_prompts=%zu\n", samples.size(), train_in.size(), test_in.size(), tr.size(), te.size()); std::fflush(stdout); bool enough=train_in.size()>=512 && test_in.size()>=512 && tr.size()>=8 && te.size()>=8 && train_in.size()>64 && test_in.size()>64; bool valid=enough; double max_identity=0,max_direction=0; for(const auto & x:samples){max_identity=std::max(max_identity,x.identity_error);max_direction=std::max(max_direction,x.direction_error);} valid=valid && repeat_positions >= 16 && min_repeat_cosine >= .99999 && max_repeat_norm_difference <= 1e-4 && max_identity<=1e-3 && max_direction<=1e-5;
    std::vector<std::tuple<int,std::string,int,metrics,uint64_t>> metric_rows; std::vector<std::tuple<int,std::string,int,std::string,double,uint64_t>> control_rows; metrics native32; distribution global,within,isotropic,random_dest; std::string classification="suppressed";
    if(valid) { for(int rank: {8,16,32,64}) { TMatrixD pin=fit_basis(train_in,rank), pdest=destination(pin,train_time,rank), pwrite=fit_basis(train_write,rank); valid=valid && orth(pin)<=1e-5 && orth(pdest)<=1e-5 && orth(pwrite)<=1e-5 && overlap(pin,pdest)<=1e-10; metrics n; n.ein=mean_energy(pdest,test_in);n.etime=mean_energy(pdest,test_time);n.d=n.etime-n.ein; std::vector<std::vector<float>> z; for(auto x:test_time){auto v=residualize(pin,*x);if(norm(v)>k_min_norm)z.push_back(unit(std::move(v)));} n.g=mean_energy(pdest,refs(z));n.input_self=mean_energy(pin,test_in);n.write_self=mean_energy(pwrite,test_write);n.iw=overlap(pin,pwrite);n.wd=overlap(pwrite,pdest);n.id=overlap(pin,pdest);n.orth_in=orth(pin);n.orth_dest=orth(pdest); metric_rows.emplace_back(rank,"native",-1,n,0); if(rank==k_primary_rank)native32=n;
            std::vector<double> gd,wd,id,rd;
            if (rank == k_primary_rank) for(int r=0;r<o.repeats;++r) { std::printf("phase=controls rank=32 replicate=%d/%d\n",r+1,o.repeats); std::fflush(stdout); for(const char * family: {"global_shuffle","within_prompt_shuffle","isotropic_write"}) { const uint64_t ss=sub_seed(o.seed,family,r); std::mt19937_64 rng(ss); auto train=control_outputs(samples,true,family,rng); auto test=control_outputs(samples,false,family,rng); metrics m; m.ein=n.ein;m.etime=mean_energy(pdest,refs(test));m.d=m.etime-m.ein; metric_rows.emplace_back(rank,family,r,m,ss); control_rows.emplace_back(rank,family,r,"D",m.d,ss); if(std::strcmp(family,"global_shuffle")==0)gd.push_back(m.d); else if(std::strcmp(family,"within_prompt_shuffle")==0)wd.push_back(m.d); else id.push_back(m.d); if(std::strcmp(family,"isotropic_write")==0){TMatrixD random=destination(pin,refs(train),rank);metrics rm;rm.g=mean_energy(random,refs(test));metric_rows.emplace_back(rank,"isotropic_destination_basis",r,rm,ss);control_rows.emplace_back(rank,"isotropic_destination_basis",r,"G_dest",rm.g,ss);rd.push_back(rm.g);} } } if(rank==32){global=summarize(std::move(gd));within=summarize(std::move(wd));isotropic=summarize(std::move(id));random_dest=summarize(std::move(rd));} } }
    const double pg=p_greater(global,native32.d), pi=p_greater(isotropic,native32.d), ps=p_greater(random_dest,native32.g), psi=(1.0+std::count_if(isotropic.values.begin(),isotropic.values.end(),[&](double v){return v>=global.median;}))/(1.0+isotropic.values.size()); const double ag=native32.d-global.median, ai=native32.d-isotropic.median, asi=global.median-isotropic.median;
    const bool stable = native32.g > random_dest.p95 && ps <= .05;
    const bool h1 = stable && native32.d > 0 && ag > 0 && pg <= .05 && ai > 0 && pi <= .05;
    const bool h2 = stable && native32.d > 0 && pg > .10 && global.median > isotropic.median && psi <= .05;
    const bool h0 = native32.g <= random_dest.p95;
    if(valid) { if(h1)classification="native_destination_routing"; else if(h2)classification="uncoordinated_structured_addition"; else if(h0)classification="no_detected_destination_channel"; else classification="ambiguous"; if(o.exploratory) classification="exploratory_"+classification; }
    const std::string status = valid ? "valid" : (enough ? "invalid" : "invalid_insufficient_samples");
    const auto        parent = std::filesystem::path(o.root).parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
    write_root(o.root, samples, metric_rows, control_rows, status, classification);
    std::ofstream j(o.json);
    if (!j) {
        throw std::runtime_error("cannot write JSON");
    }
    j << std::setprecision(12)
      << "{\n  \"schema_version\": 1,\n  \"experiment\": \"rwkv_tmix_destination_subspace\",\n  \"status\": \""
      << status << "\",\n  \"classification\": \"" << classification
      << "\",\n  \"scientific_question\": \"At layer 30, does native TMix addition route normalized residuals into a "
         "held-out-stable complementary destination, and does native pairing matter?\",\n  "
         "\"maximum_permitted_claim\": \"Distributional destination routing at one layer only; no semantic, "
          "memory-content, causal-use, logit, or output-embedding claim.\",\n  \"metadata\": "
          "{\"layer\":"
       << o.layer << ",\"confirmatory\":" << (o.exploratory ? "false" : "true")
       << ",\"primary_rank\":32,\"control_repeats\":99,\"seed\":"
      << o.seed << ",\"split_manifest\":";
    rwkv_experiment::write_json_string(j, o.manifest);
    j << ",\"protocol_deviation_overwrite\":" << (o.overwrite ? "true" : "false")
      << "},\n  \"sample_counts\": {\"valid_tokens\":" << samples.size() << ",\"excluded_tokens\":" << excluded
      << ",\"train_tokens\":" << train_in.size() << ",\"test_tokens\":" << test_in.size()
      << ",\"train_prompts\":" << tr.size() << ",\"test_prompts\":" << te.size()
      << "},\n  \"sanity_checks\": {\"max_residual_identity_error\":" << max_identity
      << ",\"max_direction_error\":" << max_direction << ",\"pass\":" << (valid ? "true" : "false")
      << "},\n  \"native_metrics\": {\"G_dest\":" << native32.g << ",\"E_input\":" << native32.ein
      << ",\"E_time\":" << native32.etime << ",\"D\":" << native32.d
      << "},\n  \"global_shuffle\": {\"median\":" << global.median << ",\"p95\":" << global.p95
      << ",\"empirical_p\":" << pg << "},\n  \"within_prompt_shuffle\": {\"median\":" << within.median
      << ",\"p95\":" << within.p95 << "},\n  \"isotropic_write\": {\"median\":" << isotropic.median
      << ",\"p95\":" << isotropic.p95 << ",\"empirical_p\":" << pi
      << "},\n  \"destination_stability\": {\"isotropic_p95\":" << random_dest.p95 << ",\"empirical_p\":" << ps
      << "},\n  \"outcome_rule_evaluation\": {\"native_advantage_global\":" << ag
      << ",\"native_advantage_isotropic\":" << ai << ",\"shuffle_advantage_isotropic\":" << asi
      << ",\"p_global\":" << pg << ",\"p_isotropic\":" << pi << ",\"p_destination_stability\":" << ps
      << ",\"p_shuffle_vs_isotropic\":" << psi << ",\"stable\":" << (stable ? "true" : "false")
      << ",\"H1\":" << (h1 ? "true" : "false") << ",\"H2\":" << (h2 ? "true" : "false")
      << ",\"H0\":" << (h0 ? "true" : "false") << "},\n  \"root_artifact\":";
    rwkv_experiment::write_json_string(j, o.root);
    j << "\n}\n";
    if (!j) {
        throw std::runtime_error("JSON write failed");
    }
    std::printf(
        "status=%s\nclassification=%s\nconfirmatory=true\nsamples=train:%zu "
        "test:%zu\nprimary_rank=32\nG_dest_native=%.6f isotropic_p95=%.6f\nD_native=%.6f\nnative_advantage_global=%.6f "
        "p_global=%.4f\nnative_advantage_isotropic=%.6f p_isotropic=%.4f\nshuffle_advantage_isotropic=%.6f "
        "p_shuffle_vs_isotropic=%.4f\nsanity=%s\nROOT=%s\nJSON=%s\n",
        status.c_str(), classification.c_str(), train_in.size(), test_in.size(), native32.g, random_dest.p95,
        native32.d, ag, pg, ai, pi, asi, psi, valid ? "pass" : "fail", o.root.c_str(), o.json.c_str());
    llama_backend_free();
    return valid ? 0 : 2;
} catch(const std::exception & e) { std::fprintf(stderr,"error: %s\n",e.what()); return 1; }
