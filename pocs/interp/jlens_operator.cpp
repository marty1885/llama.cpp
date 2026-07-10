#include "interp.hpp"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

struct operator_results {
    llama_interp::activation_set clean_caps;
    llama_interp::activation_set repeat_clean_caps;
    llama_interp::activation_set plus_caps;
    llama_interp::activation_set minus_caps;
    llama_interp::activation_set repeat_plus_caps;
    llama_interp::decode_result clean;
    llama_interp::decode_result repeat_clean;
    llama_interp::decode_result plus;
    llama_interp::decode_result minus;
    llama_interp::decode_result repeat_plus;
    std::vector<ggml_fp16_t> perturbation;
};

static void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s -m MODEL [-p PROMPT] [-ngl N] [--layer N] [--epsilon E] [--compare-epsilon E]\n"
        "\n"
        "Runs one symmetric finite-difference check from rwkv.layer.L.resid.out "
        "to the final residual on the next decoded token.\n",
        argv0);
}

static std::string residual_name(int layer) {
    return "rwkv.layer." + std::to_string(layer) + ".resid.out";
}

static const llama_interp_activation & require_capture(
        const llama_interp::activation_set & caps,
        const std::string & name) {
    const llama_interp_activation * found = nullptr;
    for (const auto & cap : caps) {
        if (cap.name == name) {
            if (found) {
                throw std::runtime_error("duplicate activation capture: " + name);
            }
            found = &cap;
        }
    }
    if (!found) {
        throw std::runtime_error("missing activation capture: " + name);
    }
    return *found;
}

static std::vector<ggml_fp16_t> rademacher_perturbation(size_t n, float epsilon) {
    std::vector<ggml_fp16_t> out(n);
    const float scale = epsilon / std::sqrt((float) n);
    for (size_t i = 0; i < n; ++i) {
        out[i] = ggml_fp32_to_fp16((i & 1) ? scale : -scale);
    }
    return out;
}

static std::vector<ggml_fp16_t> negate(const std::vector<ggml_fp16_t> & values) {
    std::vector<ggml_fp16_t> out;
    out.reserve(values.size());
    for (ggml_fp16_t value : values) {
        out.push_back(ggml_fp32_to_fp16(-ggml_fp16_to_fp32(value)));
    }
    return out;
}

static double l2_norm(const std::vector<ggml_fp16_t> & values) {
    double sum = 0.0;
    for (ggml_fp16_t value : values) {
        const double x = ggml_fp16_to_fp32(value);
        sum += x*x;
    }
    return std::sqrt(sum);
}

static double difference_l2(
        const std::vector<ggml_fp16_t> & a,
        const std::vector<ggml_fp16_t> & b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("activation dimensions differ");
    }
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double x = ggml_fp16_to_fp32(a[i]) - ggml_fp16_to_fp32(b[i]);
        sum += x*x;
    }
    return std::sqrt(sum);
}

static double symmetry_error(
        const std::vector<ggml_fp16_t> & clean,
        const std::vector<ggml_fp16_t> & plus,
        const std::vector<ggml_fp16_t> & minus) {
    if (clean.size() != plus.size() || clean.size() != minus.size()) {
        throw std::runtime_error("activation dimensions differ");
    }

    double numerator = 0.0;
    double denominator = 0.0;
    for (size_t i = 0; i < clean.size(); ++i) {
        const double y0 = ggml_fp16_to_fp32(clean[i]);
        const double yp = ggml_fp16_to_fp32(plus[i]);
        const double ym = ggml_fp16_to_fp32(minus[i]);
        const double even = (yp - y0) + (ym - y0);
        const double odd = (yp - y0) - (ym - y0);
        numerator += even*even;
        denominator += odd*odd;
    }
    return std::sqrt(numerator) / std::max(std::sqrt(denominator), 1e-30);
}

static double perturbation_error_l2(
        const std::vector<ggml_fp16_t> & base,
        const std::vector<ggml_fp16_t> & perturbed,
        const std::vector<ggml_fp16_t> & expected) {
    if (base.size() != perturbed.size() || base.size() != expected.size()) {
        throw std::runtime_error("activation dimensions differ");
    }
    double sum = 0.0;
    for (size_t i = 0; i < base.size(); ++i) {
        const double error = ggml_fp16_to_fp32(perturbed[i]) - ggml_fp16_to_fp32(base[i]) -
                             ggml_fp16_to_fp32(expected[i]);
        sum += error*error;
    }
    return std::sqrt(sum);
}

struct derivative_comparison {
    double cosine = 0.0;
    double relative_l2_difference = 0.0;
};

static derivative_comparison compare_central_derivatives(
        const std::vector<ggml_fp16_t> & plus_a,
        const std::vector<ggml_fp16_t> & minus_a,
        double epsilon_a,
        const std::vector<ggml_fp16_t> & plus_b,
        const std::vector<ggml_fp16_t> & minus_b,
        double epsilon_b) {
    if (plus_a.size() != minus_a.size() || plus_a.size() != plus_b.size() || plus_a.size() != minus_b.size()) {
        throw std::runtime_error("activation dimensions differ");
    }

    double dot = 0.0;
    double norm_a = 0.0;
    double norm_b = 0.0;
    double difference = 0.0;
    for (size_t i = 0; i < plus_a.size(); ++i) {
        const double a = (ggml_fp16_to_fp32(plus_a[i]) - ggml_fp16_to_fp32(minus_a[i])) / (2.0 * epsilon_a);
        const double b = (ggml_fp16_to_fp32(plus_b[i]) - ggml_fp16_to_fp32(minus_b[i])) / (2.0 * epsilon_b);
        dot += a*b;
        norm_a += a*a;
        norm_b += b*b;
        const double d = a - b;
        difference += d*d;
    }

    derivative_comparison out;
    out.cosine = dot / std::max(std::sqrt(norm_a * norm_b), 1e-30);
    out.relative_l2_difference = std::sqrt(difference) / std::max(std::sqrt(norm_b), 1e-30);
    return out;
}

static llama_interp::task<> prefill_prompt(
        llama_interp::runtime & rt,
        const llama_interp::rwkv_state & initial_state,
        const std::string & prompt,
        llama_interp::rwkv_state & out) {
    out = co_await rt.prefill(initial_state, prompt);
}

static llama_interp::task<> run_symmetric_check(
        llama_interp::runtime & rt,
        const llama_interp::rwkv_state & prompt_state,
        uint32_t n_layer,
        int source_layer,
        float epsilon,
        operator_results & out) {
    using namespace llama_interp;

    const std::string source = residual_name(source_layer);
    const std::string final = residual_name((int) n_layer - 1);

    auto clean = rt.decode(prompt_state, 1);
    clean.capture_f16("^" + source + "\\.pre$", out.clean_caps);
    clean.capture_f16("^" + final + "$", out.clean_caps);
    out.clean = co_await clean;

    const auto & source_activation = require_capture(out.clean_caps, source + ".pre");
    out.perturbation = rademacher_perturbation(source_activation.data.size(), epsilon);

    auto repeat_clean = rt.decode(prompt_state, 1);
    repeat_clean.capture_f16("^" + final + "$", out.repeat_clean_caps);
    out.repeat_clean = co_await repeat_clean;

    auto plus = rt.decode(prompt_state, 1);
    plus.perturb("^" + source + "$", runtime::add_head(-1, out.perturbation));
    if (source != final) {
        plus.capture_f16("^" + source + "$", out.plus_caps);
    }
    plus.capture_f16("^" + final + "$", out.plus_caps);
    out.plus = co_await plus;

    auto minus = rt.decode(prompt_state, 1);
    minus.perturb("^" + source + "$", runtime::add_head(-1, negate(out.perturbation)));
    minus.capture_f16("^" + final + "$", out.minus_caps);
    out.minus = co_await minus;

    auto repeat_plus = rt.decode(prompt_state, 1);
    repeat_plus.perturb("^" + source + "$", runtime::add_head(-1, out.perturbation));
    repeat_plus.capture_f16("^" + final + "$", out.repeat_plus_caps);
    out.repeat_plus = co_await repeat_plus;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::string model_path;
    std::string prompt = "The Eiffel Tower is located in";
    int n_gpu_layers = 0;
    int source_layer = -1;
    float epsilon = 0.1f;
    float compare_epsilon = 0.0f;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (std::strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (std::strcmp(argv[i], "-ngl") == 0 && i + 1 < argc) {
            n_gpu_layers = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--layer") == 0 && i + 1 < argc) {
            source_layer = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--epsilon") == 0 && i + 1 < argc) {
            epsilon = std::strtof(argv[++i], nullptr);
        } else if (std::strcmp(argv[i], "--compare-epsilon") == 0 && i + 1 < argc) {
            compare_epsilon = std::strtof(argv[++i], nullptr);
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    if (model_path.empty() || epsilon <= 0.0f || compare_epsilon < 0.0f) {
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
    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), (int) prompt.size(), nullptr, 0, false, true);
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = std::max(128, n_prompt + 8);
    cparams.n_batch = n_prompt;
    cparams.n_ubatch = n_prompt;
    cparams.n_seq_max = 1;
    cparams.no_perf = false;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::fprintf(stderr, "failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    llama_interp::runtime rt(ctx, 1);
    const llama_interp::rwkv_state initial_state = rt.make_state();
    if (source_layer < 0) {
        source_layer = (int) initial_state.n_layer / 2;
    }
    if (source_layer < 0 || source_layer >= (int) initial_state.n_layer) {
        std::fprintf(stderr, "invalid source layer %d for %u-layer model\n", source_layer, initial_state.n_layer);
        return 1;
    }

    llama_interp::rwkv_state prompt_state;
    auto prefill_task = prefill_prompt(rt, initial_state, prompt, prompt_state);
    rt.run();
    prefill_task.rethrow_if_failed();

    operator_results results;
    auto task = run_symmetric_check(rt, prompt_state, initial_state.n_layer, source_layer, epsilon, results);
    rt.run();
    task.rethrow_if_failed();

    const std::string final = residual_name((int) initial_state.n_layer - 1);
    const auto & clean = require_capture(results.clean_caps, final);
    const auto & repeat_clean = require_capture(results.repeat_clean_caps, final);
    const auto & clean_source = require_capture(results.clean_caps, residual_name(source_layer) + ".pre");
    const auto & plus_source = require_capture(results.plus_caps, residual_name(source_layer));
    const auto & plus = require_capture(results.plus_caps, final);
    const auto & minus = require_capture(results.minus_caps, final);
    const auto & repeat_plus = require_capture(results.repeat_plus_caps, final);
    const double actual_epsilon = l2_norm(results.perturbation);
    const double central_derivative = difference_l2(plus.data, minus.data) / (2.0 * actual_epsilon);

    std::printf("source=%s target=%s\n", residual_name(source_layer).c_str(), final.c_str());
    std::printf("epsilon requested=%g actual_l2=%g perturb_elements=%zu\n",
        epsilon, actual_epsilon, results.perturbation.size());
    std::printf("target elements=%zu backend=%s\n", clean.data.size(), clean.backend.c_str());
    std::printf("plus delta l2=%.6e minus delta l2=%.6e\n",
        difference_l2(plus.data, clean.data), difference_l2(minus.data, clean.data));
    std::printf("source addition l2=%.6e error_l2=%.6e\n",
        difference_l2(plus_source.data, clean_source.data),
        perturbation_error_l2(clean_source.data, plus_source.data, results.perturbation));
    std::printf("central derivative l2=%.6e\n", central_derivative);
    std::printf("relative symmetry error=%.6e\n", symmetry_error(clean.data, plus.data, minus.data));
    std::printf("repeat clean exact=%s\n",
        clean.data == repeat_clean.data && results.clean.tokens == results.repeat_clean.tokens ? "yes" : "no");
    std::printf("repeat plus exact=%s token=%s\n",
        plus.data == repeat_plus.data && results.plus.tokens == results.repeat_plus.tokens ? "yes" : "no",
        results.clean.to_string().c_str());

    if (compare_epsilon > 0.0f) {
        operator_results comparison_results;
        auto comparison_task = run_symmetric_check(
            rt, prompt_state, initial_state.n_layer, source_layer, compare_epsilon, comparison_results);
        rt.run();
        comparison_task.rethrow_if_failed();

        const auto & comparison_clean = require_capture(comparison_results.clean_caps, final);
        const auto & comparison_plus = require_capture(comparison_results.plus_caps, final);
        const auto & comparison_minus = require_capture(comparison_results.minus_caps, final);
        const double comparison_actual_epsilon = l2_norm(comparison_results.perturbation);
        const derivative_comparison comparison = compare_central_derivatives(
            plus.data, minus.data, actual_epsilon,
            comparison_plus.data, comparison_minus.data, comparison_actual_epsilon);

        std::printf("compare epsilon requested=%g actual_l2=%g\n", compare_epsilon, comparison_actual_epsilon);
        std::printf("shared-state clean exact=%s\n",
            clean.data == comparison_clean.data && results.clean.tokens == comparison_results.clean.tokens ? "yes" : "no");
        std::printf("central derivative cosine=%.6e relative_l2_difference=%.6e\n",
            comparison.cosine, comparison.relative_l2_difference);
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
