// rwkv_probe/lens.h — static logit-lens graph (LN + unembed + softmax).
//
// Lifts lens_state + lens_init + lens_run + lens_free from the original
// rwkv-probe.cpp (lines 247-389).
//
// Builds the small ggml compute graph ONCE at construction. Reuses it on
// every run() call — gallocr-allocated buffers are reserved once. Weights
// (output_norm.weight/.bias and output.weight) are loaded into a CPU backend
// buffer at construction time and never copied again.
#pragma once

#include "util.h"
#include "model.h"

#include <string>
#include <vector>

struct ggml_backend;
struct ggml_backend_buffer;
struct ggml_context;
struct gguf_context;
struct ggml_gallocr;
struct ggml_cgraph;
struct ggml_tensor;

namespace rwkv_probe {

class LogitLens {
public:
    // Self-contained: re-opens the gguf into a CPU backend and copies the
    // three relevant tensors into a freshly allocated backend buffer. This
    // mirrors the existing lens_init pathway exactly.
    LogitLens(const std::string & model_path,
              const ModelGeometry & geom,
              float eps = 1e-5f);

    ~LogitLens();

    LogitLens(const LogitLens &)             = delete;
    LogitLens & operator=(const LogitLens &) = delete;

    bool ok() const { return m_ok; }

    int n_vocab() const { return m_n_vocab; }
    int n_embd()  const { return m_n_embd;  }

    // run the static graph on one residual vector; returns softmax probs
    // over vocab. throws if the graph isn't ok().
    std::vector<float> run(span<const float> residual);

    // same, but writes into a caller-provided buffer to avoid the alloc.
    void run_into(span<const float> residual, span<float> probs_out);

private:
    void free_all();

    bool m_ok = false;

    ggml_backend *        m_backend     = nullptr;
    ggml_backend_buffer * m_buf_w       = nullptr;
    ggml_context        * m_ctx_weights = nullptr;
    ggml_context        * m_ctx_compute = nullptr;
    gguf_context        * m_gctx        = nullptr;
    ggml_gallocr        * m_galloc      = nullptr;
    ggml_cgraph         * m_graph       = nullptr;

    ggml_tensor * m_input = nullptr;
    ggml_tensor * m_probs = nullptr;

    int   m_n_embd  = 0;
    int   m_n_vocab = 0;
    float m_eps     = 1e-5f;
};

}  // namespace rwkv_probe
