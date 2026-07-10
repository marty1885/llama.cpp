#include "interp.hpp"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

struct path_result {
    llama_interp::rwkv_state state;
    llama_interp::activation_set captures;
    std::vector<float> logits;
};

struct trace_file {
    int source_layer = -1;
    float epsilon = 0.0f;
    std::vector<float> perturbation;
    path_result continuous;
    path_result split;
    path_result clean_injection;
    path_result plus_injection;
    path_result repeat_clean_injection;
    path_result repeat_plus_injection;
};

struct difference {
    double l2 = 0.0;
    double relative_l2 = 0.0;
    double max_abs = 0.0;
};

static void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s -m MODEL --output TRACE [-p PROMPT] [-ngl N] [--layer N] [--epsilon E]\n"
        "       %s --compare CPU_TRACE VULKAN_TRACE\n"
        "\n"
        "Captures clean RWKV FP32 checkpoints and validates state handoff and perturbation injection.\n",
        argv0, argv0);
}

static std::string residual_name(int layer) {
    return "rwkv.layer." + std::to_string(layer) + ".resid.out";
}

static std::string wkv_name(int layer) {
    return "rwkv.layer." + std::to_string(layer) + ".time.wkv";
}

static const llama_interp_activation & require_capture(
        const llama_interp::activation_set & captures,
        const std::string & name) {
    const llama_interp_activation * found = nullptr;
    for (const auto & capture : captures) {
        if (capture.name != name) {
            continue;
        }
        if (found) {
            throw std::runtime_error("duplicate activation capture: " + name);
        }
        found = &capture;
    }
    if (!found || found->data_f32.empty()) {
        throw std::runtime_error("missing FP32 activation capture: " + name);
    }
    return *found;
}

static std::vector<float> copy_logits(const llama_context * ctx, int n_vocab) {
    const float * logits = llama_get_logits_ith(const_cast<llama_context *>(ctx), 0);
    if (!logits) {
        throw std::runtime_error("missing logits");
    }
    return std::vector<float>(logits, logits + n_vocab);
}

static double l2_norm(const std::vector<float> & values) {
    double sum = 0.0;
    for (float value : values) {
        sum += (double) value * value;
    }
    return std::sqrt(sum);
}

static std::vector<ggml_fp16_t> normalized_perturbation(
        const std::vector<float> & activation,
        float epsilon) {
    const double norm = l2_norm(activation);
    if (norm == 0.0) {
        throw std::runtime_error("cannot perturb a zero activation");
    }

    std::vector<ggml_fp16_t> out;
    out.reserve(activation.size());
    for (float value : activation) {
        out.push_back(ggml_fp32_to_fp16((float) (epsilon * value / norm)));
    }
    return out;
}

static llama_interp::task<> run_path(
        llama_interp::runtime & rt,
        const llama_interp::rwkv_state & initial_state,
        const std::vector<llama_token> & tokens,
        const std::vector<std::string> & capture_regexes,
        const llama_interp_perturb_spec * perturbation,
        path_result & out) {
    auto path = rt.prefill_tokens(initial_state, tokens);
    if (perturbation) {
        path.perturb(perturbation->regex, *perturbation);
    }
    for (const auto & regex : capture_regexes) {
        path.capture_f32(regex, out.captures);
    }
    out.state = co_await path;
}

static path_result evaluate_path(
        llama_interp::runtime & rt,
        llama_context * ctx,
        int n_vocab,
        const llama_interp::rwkv_state & initial_state,
        const std::vector<llama_token> & tokens,
        const std::vector<std::string> & capture_regexes,
        const llama_interp_perturb_spec * perturbation = nullptr) {
    path_result out;
    auto task = run_path(rt, initial_state, tokens, capture_regexes, perturbation, out);
    rt.run();
    task.rethrow_if_failed();
    out.logits = copy_logits(ctx, n_vocab);
    return out;
}

static void write_u32(std::ofstream & out, uint32_t value) {
    out.write((const char *) &value, sizeof(value));
}

static uint32_t read_u32(std::ifstream & in) {
    uint32_t value = 0;
    in.read((char *) &value, sizeof(value));
    return value;
}

static void write_string(std::ofstream & out, const std::string & value) {
    write_u32(out, (uint32_t) value.size());
    out.write(value.data(), (std::streamsize) value.size());
}

static std::string read_string(std::ifstream & in) {
    const uint32_t size = read_u32(in);
    std::string value(size, '\0');
    in.read(value.data(), size);
    return value;
}

static void write_f32_vector(std::ofstream & out, const std::vector<float> & values) {
    write_u32(out, (uint32_t) values.size());
    out.write((const char *) values.data(), (std::streamsize) (values.size() * sizeof(float)));
}

static std::vector<float> read_f32_vector(std::ifstream & in) {
    const uint32_t size = read_u32(in);
    std::vector<float> values(size);
    in.read((char *) values.data(), (std::streamsize) (values.size() * sizeof(float)));
    return values;
}

static void write_path(std::ofstream & out, const path_result & path) {
    write_f32_vector(out, path.logits);
    write_u32(out, (uint32_t) path.captures.size());
    for (const auto & capture : path.captures) {
        if (capture.data_f32.empty()) {
            throw std::runtime_error("trace contains a non-FP32 capture");
        }
        write_string(out, capture.name);
        write_string(out, capture.backend);
        write_f32_vector(out, capture.data_f32);
    }
}

static path_result read_path(std::ifstream & in) {
    path_result path;
    path.logits = read_f32_vector(in);
    const uint32_t n_captures = read_u32(in);
    path.captures.resize(n_captures);
    for (auto & capture : path.captures) {
        capture.name = read_string(in);
        capture.backend = read_string(in);
        capture.data_f32 = read_f32_vector(in);
    }
    return path;
}

static void write_trace(const std::string & filename, const trace_file & trace) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open trace for writing: " + filename);
    }
    const char magic[] = "JPARITY1";
    out.write(magic, sizeof(magic) - 1);
    write_u32(out, (uint32_t) trace.source_layer);
    out.write((const char *) &trace.epsilon, sizeof(trace.epsilon));
    write_f32_vector(out, trace.perturbation);
    write_path(out, trace.continuous);
    write_path(out, trace.split);
    write_path(out, trace.clean_injection);
    write_path(out, trace.plus_injection);
    write_path(out, trace.repeat_clean_injection);
    write_path(out, trace.repeat_plus_injection);
    if (!out) {
        throw std::runtime_error("failed to write trace: " + filename);
    }
}

static trace_file read_trace(const std::string & filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open trace: " + filename);
    }
    char magic[8];
    in.read(magic, sizeof(magic));
    if (std::memcmp(magic, "JPARITY1", sizeof(magic)) != 0) {
        throw std::runtime_error("invalid trace format: " + filename);
    }
    trace_file trace;
    trace.source_layer = (int) read_u32(in);
    in.read((char *) &trace.epsilon, sizeof(trace.epsilon));
    trace.perturbation = read_f32_vector(in);
    trace.continuous = read_path(in);
    trace.split = read_path(in);
    trace.clean_injection = read_path(in);
    trace.plus_injection = read_path(in);
    trace.repeat_clean_injection = read_path(in);
    trace.repeat_plus_injection = read_path(in);
    if (!in) {
        throw std::runtime_error("truncated trace: " + filename);
    }
    return trace;
}

static difference compare_vectors(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("vector dimensions differ");
    }
    double squared_difference = 0.0;
    double squared_reference = 0.0;
    double max_abs = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double delta = (double) a[i] - b[i];
        squared_difference += delta * delta;
        squared_reference += (double) b[i] * b[i];
        max_abs = std::max(max_abs, std::abs(delta));
    }
    return {
        std::sqrt(squared_difference),
        std::sqrt(squared_difference) / std::max(std::sqrt(squared_reference), 1e-30),
        max_abs,
    };
}

static std::map<std::string, const llama_interp_activation *> capture_index(const path_result & path) {
    std::map<std::string, const llama_interp_activation *> index;
    for (const auto & capture : path.captures) {
        if (!index.emplace(capture.name, &capture).second) {
            throw std::runtime_error("duplicate capture in trace: " + capture.name);
        }
    }
    return index;
}

static void print_path_difference(const char * label, const path_result & a, const path_result & b) {
    const auto a_index = capture_index(a);
    const auto b_index = capture_index(b);
    if (a_index.size() != b_index.size()) {
        throw std::runtime_error("capture counts differ for " + std::string(label));
    }
    for (const auto & [name, left] : a_index) {
        const auto right = b_index.find(name);
        if (right == b_index.end()) {
            throw std::runtime_error("missing capture " + name + " for " + label);
        }
        const difference diff = compare_vectors(left->data_f32, right->second->data_f32);
        std::printf("%s name=%s backend_a=%s backend_b=%s l2=%.6e relative_l2=%.6e max_abs=%.6e\n",
            label, name.c_str(), left->backend.c_str(), right->second->backend.c_str(),
            diff.l2, diff.relative_l2, diff.max_abs);
    }
    const difference logits = compare_vectors(a.logits, b.logits);
    std::printf("%s logits l2=%.6e relative_l2=%.6e max_abs=%.6e\n",
        label, logits.l2, logits.relative_l2, logits.max_abs);
}

static void print_injection_report(const char * label, const trace_file & trace) {
    const std::string source = residual_name(trace.source_layer);
    const auto & clean_pre = require_capture(trace.clean_injection.captures, source + ".pre");
    const auto & plus_pre = require_capture(trace.plus_injection.captures, source + ".pre");
    const auto & plus_post = require_capture(trace.plus_injection.captures, source);
    if (plus_post.data_f32.size() != trace.perturbation.size()) {
        throw std::runtime_error("perturbation dimensions differ");
    }

    std::vector<float> actual(trace.perturbation.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        actual[i] = plus_post.data_f32[i] - plus_pre.data_f32[i];
    }
    const difference source_pre = compare_vectors(clean_pre.data_f32, plus_pre.data_f32);
    const difference addition = compare_vectors(actual, trace.perturbation);
    const difference repeat_clean = compare_vectors(
        trace.clean_injection.logits, trace.repeat_clean_injection.logits);
    const difference repeat_plus = compare_vectors(
        trace.plus_injection.logits, trace.repeat_plus_injection.logits);
    std::printf("%s injection source_pre_relative_l2=%.6e addition_l2_error=%.6e addition_relative_l2_error=%.6e\n",
        label, source_pre.relative_l2, addition.l2, addition.relative_l2);
    std::printf("%s repeat_clean_logit_max_abs=%.6e repeat_plus_logit_max_abs=%.6e\n",
        label, repeat_clean.max_abs, repeat_plus.max_abs);
}

static trace_file make_trace(
        llama_context * ctx,
        const llama_vocab * vocab,
        const std::vector<llama_token> & tokens,
        int source_layer,
        float epsilon) {
    const int n_vocab = llama_vocab_n_tokens(vocab);
    llama_interp::runtime rt(ctx, 1);
    const llama_interp::rwkv_state initial_state = rt.make_state();
    const std::vector<std::string> checkpoints = {
        "^rwkv\\.layer\\.[0-9]+\\.resid\\.out$",
        "^rwkv\\.layer\\.[0-9]+\\.time\\.r$",
        "^rwkv\\.layer\\.[0-9]+\\.time\\.v$",
        "^rwkv\\.layer\\.[0-9]+\\.time\\.wkv$",
    };

    trace_file trace;
    trace.source_layer = source_layer;
    trace.epsilon = epsilon;
    trace.continuous = evaluate_path(rt, ctx, n_vocab, initial_state, tokens, checkpoints);
    if (tokens.size() < 2) {
        throw std::runtime_error("prompt needs at least two tokens for the state handoff check");
    }

    const std::vector<llama_token> prefix(tokens.begin(), tokens.end() - 1);
    const std::vector<llama_token> suffix(tokens.end() - 1, tokens.end());
    const path_result prefix_result = evaluate_path(rt, ctx, n_vocab, initial_state, prefix, {});
    trace.split = evaluate_path(rt, ctx, n_vocab, prefix_result.state, suffix, checkpoints);

    const std::string source = residual_name(source_layer);
    const auto & query = require_capture(trace.continuous.captures, source).data_f32;
    const std::vector<ggml_fp16_t> perturbation = normalized_perturbation(query, epsilon);
    trace.perturbation.reserve(perturbation.size());
    for (ggml_fp16_t value : perturbation) {
        trace.perturbation.push_back(ggml_fp16_to_fp32(value));
    }

    const std::vector<llama_token> injection_token = { tokens.back() };
    const std::vector<std::string> clean_injection_captures = { "^" + source + "\\.pre$" };
    const std::vector<std::string> plus_injection_captures = {
        "^" + source + "\\.pre$",
        "^" + source + "$",
    };
    llama_interp_perturb_spec add = llama_interp::runtime::add_head(-1, perturbation);
    add.regex = "^" + source + "$";
    trace.clean_injection = evaluate_path(
        rt, ctx, n_vocab, initial_state, injection_token, clean_injection_captures);
    trace.repeat_clean_injection = evaluate_path(
        rt, ctx, n_vocab, initial_state, injection_token, clean_injection_captures);
    trace.plus_injection = evaluate_path(
        rt, ctx, n_vocab, initial_state, injection_token, plus_injection_captures, &add);
    trace.repeat_plus_injection = evaluate_path(
        rt, ctx, n_vocab, initial_state, injection_token, plus_injection_captures, &add);
    return trace;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::string model_path;
    std::string output_path;
    std::string prompt = "The Eiffel Tower is located in";
    std::string compare_a;
    std::string compare_b;
    int n_gpu_layers = 0;
    int source_layer = -1;
    float epsilon = 0.2f;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_path = argv[++i];
        } else if (std::strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (std::strcmp(argv[i], "-ngl") == 0 && i + 1 < argc) {
            n_gpu_layers = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--layer") == 0 && i + 1 < argc) {
            source_layer = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--epsilon") == 0 && i + 1 < argc) {
            epsilon = std::strtof(argv[++i], nullptr);
        } else if (std::strcmp(argv[i], "--compare") == 0 && i + 2 < argc) {
            compare_a = argv[++i];
            compare_b = argv[++i];
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    if (!compare_a.empty()) {
        if (!model_path.empty() || !output_path.empty()) {
            usage(argv[0]);
            return 1;
        }
        const trace_file a = read_trace(compare_a);
        const trace_file b = read_trace(compare_b);
        if (a.source_layer != b.source_layer || a.epsilon != b.epsilon) {
            throw std::runtime_error("traces use different source layers or epsilons");
        }
        print_path_difference("backend_parity", a.continuous, b.continuous);
        print_path_difference("state_handoff_a", a.continuous, a.split);
        print_path_difference("state_handoff_b", b.continuous, b.split);
        print_injection_report("injection_a", a);
        print_injection_report("injection_b", b);
        return 0;
    }

    if (model_path.empty() || output_path.empty() || epsilon <= 0.0f) {
        usage(argv[0]);
        return 1;
    }

    ggml_backend_load_all();
    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        std::fprintf(stderr, "failed to load model: %s\n", model_path.c_str());
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const std::vector<llama_token> tokens = common_tokenize(vocab, prompt, false, true);
    if (tokens.size() < 2) {
        throw std::runtime_error("prompt tokenized to fewer than two tokens");
    }
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = std::max(128, (int) tokens.size() + 8);
    cparams.n_batch = (int) tokens.size();
    cparams.n_ubatch = (int) tokens.size();
    cparams.n_seq_max = 1;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::fprintf(stderr, "failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    llama_interp::runtime setup(ctx, 1);
    const uint32_t n_layer = setup.make_state().n_layer;
    if (source_layer < 0) {
        source_layer = (int) n_layer / 2;
    }
    if (source_layer < 0 || source_layer >= (int) n_layer) {
        throw std::runtime_error("invalid source layer");
    }
    const trace_file trace = make_trace(ctx, vocab, tokens, source_layer, epsilon);
    write_trace(output_path, trace);
    std::printf("wrote %s source_layer=%d n_gpu_layers=%d\n", output_path.c_str(), source_layer, n_gpu_layers);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
