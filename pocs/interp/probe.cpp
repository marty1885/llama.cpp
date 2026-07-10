#include "interp.hpp"

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>

struct probe_results {
    llama_interp::rwkv_state prefill_state;
    llama_interp::decode_result baseline_one;
    llama_interp::decode_result baseline;
    llama_interp::decode_result zero_state;
    llama_interp::decode_result zero_v_one;
    llama_interp::decode_result zero_v_full;
    llama_interp::activation_set zero_v_caps;
};

static void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s -m MODEL [-p PROMPT] [-n N] [-ngl N] [--parallel N]\n"
        "\n"
        "example:\n"
        "  %s -m rwkv.gguf -ngl 999 -p \"The Eiffel Tower is\" -n 64\n",
        argv0, argv0);
}

static int token_divergence(const std::vector<llama_token> & a, const std::vector<llama_token> & b) {
    const size_t n = std::min(a.size(), b.size());
    int diff = 0;
    for (size_t i = 0; i < n; ++i) {
        diff += a[i] != b[i];
    }
    return diff + (int) std::max(a.size(), b.size()) - (int) n;
}

static float fp16_abs_max(const std::vector<ggml_fp16_t> & v) {
    float res = 0.0f;
    for (ggml_fp16_t x : v) {
        res = std::max(res, std::abs(ggml_fp16_to_fp32(x)));
    }
    return res;
}

static double state_l1_diff(const llama_interp::rwkv_state & a, const llama_interp::rwkv_state & b) {
    double diff = 0.0;
    const size_t nl = std::min(a.layers.size(), b.layers.size());
    for (size_t il = 0; il < nl; ++il) {
        const auto & ar = a.layers[il].r;
        const auto & br = b.layers[il].r;
        for (size_t i = 0; i < std::min(ar.size(), br.size()); ++i) {
            diff += std::abs(ggml_fp16_to_fp32(ar[i]) - ggml_fp16_to_fp32(br[i]));
        }

        const auto & as = a.layers[il].s;
        const auto & bs = b.layers[il].s;
        for (size_t i = 0; i < std::min(as.size(), bs.size()); ++i) {
            diff += std::abs(ggml_fp16_to_fp32(as[i]) - ggml_fp16_to_fp32(bs[i]));
        }
    }
    return diff;
}

static llama_interp::task<> run_probe(
        llama_interp::runtime & rt,
        const std::string & prompt,
        int n_predict,
        probe_results & out) {
    using namespace llama_interp;

    rwkv_state s0 = rt.make_state();
    out.prefill_state = co_await rt.prefill(s0, prompt);

    out.baseline_one = co_await rt.decode(out.prefill_state, 1);
    out.baseline = co_await rt.decode(out.prefill_state, n_predict);

    rwkv_state zero = out.prefill_state;
    for (auto & layer : zero.layers) {
        std::fill(layer.r.begin(), layer.r.end(), ggml_fp32_to_fp16(0.0f));
        std::fill(layer.s.begin(), layer.s.end(), ggml_fp32_to_fp16(0.0f));
    }
    out.zero_state = co_await rt.decode(zero, n_predict);

    llama_interp_perturb_spec zero_v;
    zero_v.op = LLAMA_INTERP_PERTURB_REPLACE;
    zero_v.head = -1;

    auto proof = rt.decode(out.prefill_state, 1);
    proof.perturb("rwkv\\.layer\\.[0-9]+\\.time\\.v", zero_v);
    proof.capture_f16("rwkv\\.layer\\.[0-9]+\\.time\\.v", out.zero_v_caps);
    out.zero_v_one = co_await proof;

    auto full = rt.decode(out.prefill_state, n_predict);
    full.perturb("rwkv\\.layer\\.[0-9]+\\.time\\.v", zero_v);
    out.zero_v_full = co_await full;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::string model_path;
    std::string prompt = "The Eiffel Tower is located in";
    int n_predict = 64;
    int n_gpu_layers = 0;
    int max_parallel = 4;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (std::strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n_predict = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-ngl") == 0 && i + 1 < argc) {
            n_gpu_layers = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--parallel") == 0 && i + 1 < argc) {
            max_parallel = std::atoi(argv[++i]);
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    if (model_path.empty() || n_predict <= 0 || max_parallel <= 0) {
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
    cparams.n_ctx = std::max(128, n_prompt + n_predict + 8);
    cparams.n_batch = std::max(n_prompt, max_parallel);
    cparams.n_ubatch = cparams.n_batch;
    cparams.n_seq_max = max_parallel;
    cparams.no_perf = false;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::fprintf(stderr, "failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    std::printf("system: %s\n", llama_print_system_info());
    std::printf("prompt: %s\n", prompt.c_str());
    std::printf("n_predict=%d n_gpu_layers=%d max_parallel=%d\n", n_predict, n_gpu_layers, max_parallel);

    llama_interp::runtime rt(ctx, (uint32_t) max_parallel);
    probe_results results;
    auto t = run_probe(rt, prompt, n_predict, results);
    rt.run();
    t.rethrow_if_failed();

    std::set<std::string> capture_backends;
    float cap_max_abs = 0.0f;
    for (const auto & cap : results.zero_v_caps) {
        capture_backends.insert(cap.backend);
        cap_max_abs = std::max(cap_max_abs, fp16_abs_max(cap.data));
    }

    std::printf("\n--- baseline ---\n%s\n", results.baseline.to_string().c_str());
    std::printf("\n--- zero state ---\n%s\n", results.zero_state.to_string().c_str());
    std::printf("\n--- zero-v first-step perturb ---\n%s\n", results.zero_v_full.to_string().c_str());

    std::printf("\n--- proof ---\n");
    std::printf("zero-state token divergence: %d/%zu\n",
        token_divergence(results.baseline.tokens, results.zero_state.tokens), results.baseline.tokens.size());
    std::printf("zero-v token divergence: %d/%zu\n",
        token_divergence(results.baseline.tokens, results.zero_v_full.tokens), results.baseline.tokens.size());
    std::printf("zero-v one-step state L1 diff from baseline one-step: %.6e\n",
        state_l1_diff(results.zero_v_one.state, results.baseline_one.state));
    std::printf("zero-v post-perturb captures: %zu tensors, max_abs=%g\n", results.zero_v_caps.size(), cap_max_abs);
    std::printf("capture backends:");
    for (const auto & backend : capture_backends) {
        std::printf(" %s", backend.c_str());
    }
    std::printf("\n");

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
