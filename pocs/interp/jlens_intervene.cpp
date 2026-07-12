#include "interp.hpp"
#include "llama-model.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

struct artifact {
    int source_layer = -1, target_layer = -1;
    std::string source_activation = "residual";
    size_t input_dimension = 0, output_dimension = 0, rank = 0;
    std::vector<float> directions, responses;
};

static std::string residual_name(int layer) { return "rwkv.layer." + std::to_string(layer) + ".resid.out"; }

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
    if (input.gcount() != (std::streamsize) (count * sizeof(float)) || input.peek() != std::char_traits<char>::eof()) throw std::runtime_error("unexpected operator data size");
}

static artifact load_operator(const std::string & prefix) {
    artifact out;
    std::ifstream input(prefix.ends_with(".txt") ? prefix : prefix + ".txt");
    if (!input) throw std::runtime_error("failed to open operator manifest");
    std::string directions_path, responses_path, line;
    while (std::getline(input, line)) {
        const size_t p = line.find('='); if (p == std::string::npos) continue;
        const std::string key = line.substr(0, p), value = line.substr(p + 1);
        if (key == "source_layer") out.source_layer = std::stoi(value);
        else if (key == "target_layer") out.target_layer = std::stoi(value);
        else if (key == "source_activation") out.source_activation = value;
        else if (key == "input_dimension") out.input_dimension = std::stoull(value);
        else if (key == "output_dimension") out.output_dimension = std::stoull(value);
        else if (key == "rank") out.rank = std::stoull(value);
        else if (key == "directions_f32") directions_path = value;
        else if (key == "responses_f32") responses_path = value;
    }
    if (out.source_layer < 0 || out.target_layer < 0 || !out.input_dimension || !out.output_dimension || !out.rank || directions_path.empty() || responses_path.empty()) throw std::runtime_error("incomplete operator manifest");
    read_f32(directions_path, out.directions, out.rank * out.input_dimension);
    read_f32(responses_path, out.responses, out.rank * out.output_dimension);
    return out;
}

static const std::vector<float> & capture_f32(const llama_interp::activation_set & captures, const std::string & name) {
    for (const auto & capture : captures) if (capture.name == name && !capture.data_f32.empty()) return capture.data_f32;
    throw std::runtime_error("missing capture: " + name);
}

static std::vector<ggml_fp16_t> as_fp16(const std::vector<float> & values) {
    std::vector<ggml_fp16_t> out; out.reserve(values.size());
    for (float value : values) out.push_back(ggml_fp32_to_fp16(value));
    return out;
}

static double dot(const std::vector<float> & a, const std::vector<float> & b) {
    double out = 0.0; for (size_t i = 0; i < a.size(); ++i) out += (double) a[i] * b[i]; return out;
}

static double l2(const std::vector<float> & values) {
    return std::sqrt(dot(values, values));
}

static uint64_t splitmix64(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

static std::vector<float> output_column(const llama_model * model, int token, size_t n_embd) {
    if (!model->output || model->output->ne[0] != (int64_t) n_embd || token < 0 || token >= model->output->ne[1]) throw std::runtime_error("unexpected RWKV output tensor");
    std::vector<float> out(n_embd);
    const size_t offset = (size_t) token * model->output->nb[1];
    if (model->output->type == GGML_TYPE_BF16) {
        std::vector<ggml_bf16_t> row(n_embd);
        ggml_backend_tensor_get(model->output, row.data(), offset, row.size() * sizeof(ggml_bf16_t));
        for (size_t i = 0; i < n_embd; ++i) out[i] = ggml_bf16_to_fp32(row[i]);
    } else if (model->output->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> row(n_embd);
        ggml_backend_tensor_get(model->output, row.data(), offset, row.size() * sizeof(ggml_fp16_t));
        for (size_t i = 0; i < n_embd; ++i) out[i] = ggml_fp16_to_fp32(row[i]);
    } else if (model->output->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(model->output, out.data(), offset, out.size() * sizeof(float));
    } else throw std::runtime_error("unsupported output tensor type for J-coordinate intervention");
    return out;
}

static std::vector<float> token_vector(const artifact & op, const std::vector<float> & output) {
    std::vector<float> out(op.input_dimension);
    const double scale = (double) op.input_dimension / op.rank;
    for (size_t k = 0; k < op.rank; ++k) {
        double response_dot = 0.0;
        for (size_t j = 0; j < op.output_dimension; ++j) response_dot += (double) op.responses[k * op.output_dimension + j] * output[j];
        for (size_t i = 0; i < op.input_dimension; ++i) out[i] += (float) (scale * response_dot * op.directions[k * op.input_dimension + i]);
    }
    return out;
}

static int rank_of(const float * logits, int n_vocab, int token) {
    int rank = 1; for (int i = 0; i < n_vocab; ++i) rank += logits[i] > logits[token]; return rank;
}

static llama_interp::task<> prefill_state(llama_interp::runtime & rt, const llama_interp::rwkv_state & initial,
                                          const std::vector<llama_token> & tokens, llama_interp::rwkv_state & output) {
    output = co_await rt.prefill_tokens(initial, tokens);
}

static llama_interp::task<> capture_source(llama_interp::runtime & rt, const llama_interp::rwkv_state & initial,
                                            llama_token token, const std::string & source, llama_interp::activation_set & captures) {
    auto path = rt.prefill_tokens(initial, { token });
    path.capture_f32("^" + source + "$", captures);
    path.discard_state();
    co_await path;
}

static llama_interp::task<> prefill_perturbed(llama_interp::runtime & rt, const llama_interp::rwkv_state & initial,
                                               llama_token token, const std::vector<ggml_fp16_t> * delta,
                                               const std::string & source, llama_interp::rwkv_state & output) {
    auto path = rt.prefill_tokens(initial, { token });
    if (delta) path.perturb("^" + source + "$", llama_interp::runtime::add_head(-1, *delta));
    output = co_await path;
}

static void usage(const char * argv0) {
    std::fprintf(stderr, "usage: %s -m MODEL --operator PREFIX --source-token TOKEN --target-token TOKEN [-p PROMPT] [-ngl N] [--scale N]\n", argv0);
}

int main(int argc, char ** argv) {
    std::string model_path, operator_path, source_text, target_text, prompt = "Think of an animal. Answer with one word:";
    int n_gpu_layers = 0; float scale = 1.0f;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (std::strcmp(argv[i], "--operator") == 0 && i + 1 < argc) operator_path = argv[++i];
        else if (std::strcmp(argv[i], "--source-token") == 0 && i + 1 < argc) source_text = argv[++i];
        else if (std::strcmp(argv[i], "--target-token") == 0 && i + 1 < argc) target_text = argv[++i];
        else if (std::strcmp(argv[i], "-p") == 0 && i + 1 < argc) prompt = argv[++i];
        else if (std::strcmp(argv[i], "-ngl") == 0 && i + 1 < argc) n_gpu_layers = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--scale") == 0 && i + 1 < argc) scale = std::strtof(argv[++i], nullptr);
        else { usage(argv[0]); return 1; }
    }
    if (model_path.empty() || operator_path.empty() || source_text.empty() || target_text.empty() || scale == 0.0f) { usage(argv[0]); return 1; }
    const artifact op = load_operator(operator_path);
    ggml_backend_load_all(); llama_backend_init();
    llama_model_params mparams = llama_model_default_params(); mparams.n_gpu_layers = n_gpu_layers;
    llama_model * public_model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!public_model) throw std::runtime_error("failed to load model");
    const llama_vocab * vocab = llama_model_get_vocab(public_model);
    const auto source_tokens = common_tokenize(vocab, source_text, false, false);
    const auto target_tokens = common_tokenize(vocab, target_text, false, false);
    if (source_tokens.size() != 1 || target_tokens.size() != 1) throw std::runtime_error("source and target must each tokenize to one token");
    llama_context_params cparams = llama_context_default_params(); cparams.n_ctx = 256; cparams.n_batch = 256; cparams.n_ubatch = 256;
    llama_context * ctx = llama_init_from_model(public_model, cparams);
    if (!ctx) throw std::runtime_error("failed to create context");
    llama_interp::runtime rt(ctx, 1); const auto initial = rt.make_state();
    if (op.target_layer != (int) initial.n_layer - 1) throw std::runtime_error("intervention requires a final-residual target operator");
    const auto prompt_tokens = common_tokenize(vocab, prompt, false, true);
    if (prompt_tokens.empty()) throw std::runtime_error("prompt tokenized to zero tokens");
    const std::vector<llama_token> prefix(prompt_tokens.begin(), prompt_tokens.end() - 1);
    llama_interp::rwkv_state before_source = initial;
    if (!prefix.empty()) { auto task = prefill_state(rt, initial, prefix, before_source); rt.run(); task.rethrow_if_failed(); }
    const std::string source = source_name(op.source_layer, op.source_activation);
    llama_interp::activation_set captures;
    auto capture = capture_source(rt, before_source, prompt_tokens.back(), source, captures); rt.run(); capture.rethrow_if_failed();
    const auto & activation = capture_f32(captures, source);
    const auto * model = reinterpret_cast<const ::llama_model *>(public_model);
    const std::vector<float> v_source = token_vector(op, output_column(model, source_tokens[0], op.output_dimension));
    const std::vector<float> v_target = token_vector(op, output_column(model, target_tokens[0], op.output_dimension));
    const double aa = dot(v_source, v_source), ab = dot(v_source, v_target), bb = dot(v_target, v_target);
    const double determinant = aa * bb - ab * ab;
    if (determinant <= 1e-20) throw std::runtime_error("J-vectors are linearly dependent");
    const double ah = dot(v_source, activation), bh = dot(v_target, activation);
    const double c_source = (bb * ah - ab * bh) / determinant;
    const double c_target = (aa * bh - ab * ah) / determinant;
    std::vector<float> delta(activation.size());
    for (size_t i = 0; i < delta.size(); ++i) delta[i] = (float) (scale * ((c_target - c_source) * v_source[i] + (c_source - c_target) * v_target[i]));
    llama_interp::rwkv_state clean;
    auto clean_task = prefill_perturbed(rt, before_source, prompt_tokens.back(), nullptr, source, clean); rt.run(); clean_task.rethrow_if_failed();
    const float * clean_logits = llama_get_logits_ith(ctx, 0);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int clean_target_rank = rank_of(clean_logits, n_vocab, target_tokens[0]);
    llama_interp::rwkv_state patched;
    const auto delta_f16 = as_fp16(delta);
    auto patched_task = prefill_perturbed(rt, before_source, prompt_tokens.back(), &delta_f16, source, patched); rt.run(); patched_task.rethrow_if_failed();
    const float * patched_logits = llama_get_logits_ith(ctx, 0);
    const int patched_target_rank = rank_of(patched_logits, n_vocab, target_tokens[0]);
    std::vector<float> random_delta(delta.size());
    for (size_t i = 0; i < random_delta.size(); ++i) random_delta[i] = (splitmix64(i) & 1) ? 1.0f : -1.0f;
    const double random_scale = l2(delta) / std::max(l2(random_delta), 1e-30);
    for (float & value : random_delta) value = (float) (value * random_scale);
    llama_interp::rwkv_state random;
    const auto random_f16 = as_fp16(random_delta);
    auto random_task = prefill_perturbed(rt, before_source, prompt_tokens.back(), &random_f16, source, random); rt.run(); random_task.rethrow_if_failed();
    const int random_target_rank = rank_of(llama_get_logits_ith(ctx, 0), n_vocab, target_tokens[0]);
    std::printf("source=%s target=%s activation=%s layer=%d scale=%g clean_target_rank=%d patched_target_rank=%d random_target_rank=%d clean=%s patched=%s\n", source_text.c_str(), target_text.c_str(), source.c_str(), op.source_layer, scale, clean_target_rank, patched_target_rank, random_target_rank, common_token_to_piece(vocab, clean.next_token, true).c_str(), common_token_to_piece(vocab, patched.next_token, true).c_str());
    llama_free(ctx); llama_model_free(public_model); llama_backend_free();
    return 0;
}
