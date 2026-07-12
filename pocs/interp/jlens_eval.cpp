#include "interp.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

struct artifact {
    int source_layer = -1;
    int target_layer = -1;
    std::string source_activation = "residual";
    size_t input_dimension = 0;
    size_t output_dimension = 0;
    size_t rank = 0;
    std::vector<float> directions;
    std::vector<float> responses;
};

static std::string residual_name(int layer) {
    return "rwkv.layer." + std::to_string(layer) + ".resid.out";
}

static std::string source_name(int layer, const std::string & activation) {
    if (activation == "residual") return residual_name(layer);
    if (activation == "time-wkv") return "rwkv.layer." + std::to_string(layer) + ".time.wkv";
    throw std::runtime_error("unknown source activation in operator: " + activation);
}

static void read_f32(const std::string & path, std::vector<float> & values, size_t count) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("failed to open operator data: " + path);
    values.resize(count);
    input.read((char *) values.data(), (std::streamsize) (count * sizeof(float)));
    if (input.gcount() != (std::streamsize) (count * sizeof(float)) || input.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error("unexpected operator data size: " + path);
    }
}

static artifact load_operator(const std::string & prefix) {
    artifact out;
    std::ifstream input(prefix.ends_with(".txt") ? prefix : prefix + ".txt");
    if (!input) throw std::runtime_error("failed to open operator manifest");
    std::string directions_path;
    std::string responses_path;
    std::string line;
    while (std::getline(input, line)) {
        const size_t p = line.find('=');
        if (p == std::string::npos) continue;
        const std::string key = line.substr(0, p);
        const std::string value = line.substr(p + 1);
        if (key == "source_layer") out.source_layer = std::stoi(value);
        else if (key == "target_layer") out.target_layer = std::stoi(value);
        else if (key == "source_activation") out.source_activation = value;
        else if (key == "input_dimension") out.input_dimension = std::stoull(value);
        else if (key == "output_dimension") out.output_dimension = std::stoull(value);
        else if (key == "rank") out.rank = std::stoull(value);
        else if (key == "directions_f32") directions_path = value;
        else if (key == "responses_f32") responses_path = value;
    }
    if (out.source_layer < 0 || out.target_layer < 0 || !out.input_dimension || !out.output_dimension || !out.rank ||
        directions_path.empty() || responses_path.empty()) throw std::runtime_error("incomplete operator manifest");
    read_f32(directions_path, out.directions, out.rank * out.input_dimension);
    read_f32(responses_path, out.responses, out.rank * out.output_dimension);
    return out;
}

static std::vector<float> apply_operator(const artifact & op, const std::vector<float> & input) {
    if (input.size() != op.input_dimension) throw std::runtime_error("operator input dimension mismatch");
    std::vector<float> out(op.output_dimension);
    const double scale = (double) op.input_dimension / op.rank;
    for (size_t k = 0; k < op.rank; ++k) {
        double dot = 0.0;
        for (size_t i = 0; i < input.size(); ++i) dot += (double) op.directions[k * op.input_dimension + i] * input[i];
        for (size_t i = 0; i < out.size(); ++i) out[i] += (float) (scale * dot * op.responses[k * op.output_dimension + i]);
    }
    return out;
}

static const std::vector<float> & capture_f32(const llama_interp::activation_set & captures, const std::string & name) {
    for (const auto & capture : captures) if (capture.name == name && !capture.data_f32.empty()) return capture.data_f32;
    throw std::runtime_error("missing capture: " + name);
}

static std::vector<ggml_fp16_t> as_fp16(const std::vector<float> & values) {
    std::vector<ggml_fp16_t> out;
    out.reserve(values.size());
    for (float value : values) out.push_back(ggml_fp32_to_fp16(value));
    return out;
}

static llama_interp::task<> read_head(llama_interp::runtime & rt, const llama_interp::rwkv_state & initial,
                                      llama_token carrier, int target_layer, std::vector<ggml_fp16_t> value) {
    llama_interp_perturb_spec replace;
    replace.op = LLAMA_INTERP_PERTURB_REPLACE;
    replace.head = -1;
    replace.data = std::move(value);
    auto path = rt.prefill_tokens(initial, { carrier });
    path.perturb("^" + residual_name(target_layer) + "$", std::move(replace));
    path.discard_state();
    co_await path;
}

static llama_interp::task<> capture_source(llama_interp::runtime & rt, const llama_interp::rwkv_state & initial,
                                            const std::vector<llama_token> & tokens, const std::string & source,
                                            llama_interp::activation_set & captures) {
    auto path = rt.prefill_tokens(initial, tokens);
    path.capture_f32("^" + source + "$", captures);
    path.discard_state();
    co_await path;
}

static int rank_of(const std::vector<float> & logits, int token) {
    int rank = 1;
    for (float logit : logits) rank += logit > logits[token];
    return rank;
}

static double l2(const std::vector<float> & values) {
    double sum = 0.0;
    for (float value : values) sum += (double) value * value;
    return std::sqrt(sum);
}

static uint64_t splitmix64(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

static void usage(const char * argv0) {
    std::fprintf(stderr, "usage: %s -m MODEL --operator PREFIX --cases FILE --output FILE [-ngl N] [--random-seed N]\n"
                         "cases are TSV rows: id<TAB>prompt<TAB>single-token intermediate\n", argv0);
}

int main(int argc, char ** argv) {
    std::string model_path, operator_path, cases_path, output_path;
    int n_gpu_layers = 0;
    uint64_t random_seed = 1;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (std::strcmp(argv[i], "--operator") == 0 && i + 1 < argc) operator_path = argv[++i];
        else if (std::strcmp(argv[i], "--cases") == 0 && i + 1 < argc) cases_path = argv[++i];
        else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) output_path = argv[++i];
        else if (std::strcmp(argv[i], "-ngl") == 0 && i + 1 < argc) n_gpu_layers = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--random-seed") == 0 && i + 1 < argc) random_seed = std::strtoull(argv[++i], nullptr, 10);
        else { usage(argv[0]); return 1; }
    }
    if (model_path.empty() || operator_path.empty() || cases_path.empty() || output_path.empty()) { usage(argv[0]); return 1; }

    const artifact op = load_operator(operator_path);
    const bool residual_source = op.source_activation == "residual";
    ggml_backend_load_all();
    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) throw std::runtime_error("failed to load model");
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 256;
    cparams.n_batch = 256;
    cparams.n_ubatch = 256;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) throw std::runtime_error("failed to create context");
    llama_interp::runtime rt(ctx, 1);
    const auto initial = rt.make_state();

    std::ifstream cases(cases_path);
    std::ofstream output(output_path);
    if (!cases || !output) throw std::runtime_error("failed to open cases or output");
    output << "id\texpected\tj_rank\t" << (residual_source ? "logit_rank" : "source_rank") << "\trandom_rank\n";
    std::string line;
    int count = 0, j_top10 = 0, logit_top10 = 0, random_top10 = 0;
    while (std::getline(cases, line)) {
        const size_t a = line.find('\t');
        const size_t b = a == std::string::npos ? a : line.find('\t', a + 1);
        if (a == std::string::npos || b == std::string::npos) throw std::runtime_error("invalid case row");
        const std::string id = line.substr(0, a);
        const std::string prompt = line.substr(a + 1, b - a - 1);
        const std::string expected = line.substr(b + 1);
        const auto tokens = common_tokenize(vocab, prompt, false, true);
        const auto expected_tokens = common_tokenize(vocab, expected, false, false);
        if (tokens.empty() || expected_tokens.size() != 1) throw std::runtime_error("case requires nonempty prompt and one token intermediate: " + id);
        llama_interp::activation_set captures;
        const std::string source = source_name(op.source_layer, op.source_activation);
        auto capture = capture_source(rt, initial, tokens, source, captures);
        rt.run(); capture.rethrow_if_failed();
        const auto & query = capture_f32(captures, source);
        const std::vector<float> action = apply_operator(op, query);
        auto j_task = read_head(rt, initial, tokens.back(), op.target_layer, as_fp16(action));
        rt.run(); j_task.rethrow_if_failed();
        const float * j_logits = llama_get_logits_ith(ctx, 0);
        std::vector<float> j(j_logits, j_logits + llama_vocab_n_tokens(vocab));
        auto logit_task = read_head(rt, initial, tokens.back(), (int) initial.n_layer - 1, as_fp16(query));
        rt.run(); logit_task.rethrow_if_failed();
        const float * logit_logits = llama_get_logits_ith(ctx, 0);
        std::vector<float> logit(logit_logits, logit_logits + llama_vocab_n_tokens(vocab));
        std::vector<float> random(action.size());
        for (size_t i = 0; i < random.size(); ++i) random[i] = (splitmix64(random_seed + count * random.size() + i) & 1) ? 1.0f : -1.0f;
        const double scale = l2(action) / std::max(l2(random), 1e-30);
        for (float & value : random) value = (float) (value * scale);
        auto random_task = read_head(rt, initial, tokens.back(), op.target_layer, as_fp16(random));
        rt.run(); random_task.rethrow_if_failed();
        const float * random_logits = llama_get_logits_ith(ctx, 0);
        std::vector<float> random_readout(random_logits, random_logits + llama_vocab_n_tokens(vocab));
        const int token = expected_tokens[0];
        const int jr = rank_of(j, token), lr = rank_of(logit, token), rr = rank_of(random_readout, token);
        output << id << '\t' << expected << '\t' << jr << '\t' << lr << '\t' << rr << '\n';
        j_top10 += jr <= 10; logit_top10 += lr <= 10; random_top10 += rr <= 10; ++count;
        if (count == 1 || count % 8 == 0) {
            std::fprintf(stderr, "jlens eval case=%d id=%s j_rank=%d %s=%d random_rank=%d\n",
                count, id.c_str(), jr, residual_source ? "logit_rank" : "source_rank", lr, rr);
        }
    }
    std::printf("cases=%d j_top10=%d %s_top10=%d random_top10=%d wrote %s\n",
        count, j_top10, residual_source ? "logit" : "source", logit_top10, random_top10, output_path.c_str());
    llama_free(ctx); llama_model_free(model); llama_backend_free();
    return 0;
}
