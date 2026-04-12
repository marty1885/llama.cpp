// rwkv_probe/lens.cpp — LogitLens implementation, lifted near-verbatim from
// the original rwkv-probe.cpp lens_state / lens_init / lens_run / lens_free.
#include "rwkv_probe/lens.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace rwkv_probe {

LogitLens::LogitLens(const std::string & model_path,
                     const ModelGeometry & geom,
                     float eps)
    : m_n_embd(geom.n_embd), m_n_vocab(geom.n_vocab), m_eps(eps) {

    // 1. open gguf, populate ctx_weights with tensor metadata only
    {
        ggml_init_params ip = {
            /*.mem_size   =*/ ggml_tensor_overhead() * 1024,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        m_ctx_weights = ggml_init(ip);
        if (!m_ctx_weights) {
            std::fprintf(stderr, "lens: ggml_init(ctx_weights) failed\n");
            free_all(); return;
        }

        gguf_init_params gp = {
            /*.no_alloc =*/ true,
            /*.ctx      =*/ &m_ctx_weights,
        };
        m_gctx = gguf_init_from_file(model_path.c_str(), gp);
        if (!m_gctx) {
            std::fprintf(stderr, "lens: gguf_init_from_file failed\n");
            free_all(); return;
        }
    }

    ggml_tensor * w_norm = ggml_get_tensor(m_ctx_weights, "output_norm.weight");
    ggml_tensor * b_norm = ggml_get_tensor(m_ctx_weights, "output_norm.bias");
    ggml_tensor * w_out  = ggml_get_tensor(m_ctx_weights, "output.weight");
    if (!w_norm || !b_norm || !w_out) {
        std::fprintf(stderr,
                     "lens: missing tensors (output_norm.weight=%p .bias=%p output.weight=%p)\n",
                     (void*) w_norm, (void*) b_norm, (void*) w_out);
        free_all(); return;
    }
    if (w_norm->ne[0] != m_n_embd || b_norm->ne[0] != m_n_embd ||
        w_out->ne[0]  != m_n_embd || w_out->ne[1]  != m_n_vocab) {
        std::fprintf(stderr,
                     "lens: shape mismatch (n_embd=%d n_vocab=%d, w_norm[%lld] b_norm[%lld] w_out[%lld,%lld])\n",
                     m_n_embd, m_n_vocab,
                     (long long) w_norm->ne[0], (long long) b_norm->ne[0],
                     (long long) w_out->ne[0],  (long long) w_out->ne[1]);
        free_all(); return;
    }
    std::fprintf(stderr, "lens: w_norm type=%s   w_out type=%s\n",
                 ggml_type_name(w_norm->type), ggml_type_name(w_out->type));

    // 2. cpu backend + allocate weights into a backend buffer
    m_backend = ggml_backend_cpu_init();
    if (!m_backend) {
        std::fprintf(stderr, "lens: cpu backend init failed\n");
        free_all(); return;
    }
    m_buf_w = ggml_backend_alloc_ctx_tensors(m_ctx_weights, m_backend);
    if (!m_buf_w) {
        std::fprintf(stderr, "lens: alloc_ctx_tensors failed\n");
        free_all(); return;
    }

    // 3. read raw tensor bytes from gguf and copy into the allocated buffer
    {
        FILE * f = std::fopen(model_path.c_str(), "rb");
        if (!f) {
            std::fprintf(stderr, "lens: open %s failed\n", model_path.c_str());
            free_all(); return;
        }
        auto load_one = [&](ggml_tensor * t) -> bool {
            int64_t idx = gguf_find_tensor(m_gctx, ggml_get_name(t));
            if (idx < 0) {
                std::fprintf(stderr, "lens: tensor %s not found in gguf\n", ggml_get_name(t));
                return false;
            }
            const std::size_t offs = gguf_get_data_offset(m_gctx)
                                   + gguf_get_tensor_offset(m_gctx, idx);
            const std::size_t sz   = ggml_nbytes(t);
            std::vector<uint8_t> tmp(sz);
            if (std::fseek(f, (long) offs, SEEK_SET) != 0) {
                std::fprintf(stderr, "lens: fseek failed\n"); return false;
            }
            if (std::fread(tmp.data(), 1, sz, f) != sz) {
                std::fprintf(stderr, "lens: short read\n"); return false;
            }
            ggml_backend_tensor_set(t, tmp.data(), 0, sz);
            return true;
        };
        bool ok = load_one(w_norm) && load_one(b_norm) && load_one(w_out);
        std::fclose(f);
        if (!ok) { free_all(); return; }
    }

    // 4. build the lens compute graph in a separate context
    {
        ggml_init_params ip = {
            /*.mem_size   =*/ ggml_tensor_overhead() * 32 + ggml_graph_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        m_ctx_compute = ggml_init(ip);
        if (!m_ctx_compute) {
            std::fprintf(stderr, "lens: ggml_init(ctx_compute) failed\n");
            free_all(); return;
        }
    }

    m_input = ggml_new_tensor_1d(m_ctx_compute, GGML_TYPE_F32, m_n_embd);
    ggml_set_name(m_input, "lens_input");
    ggml_set_input(m_input);

    ggml_tensor * h = ggml_norm(m_ctx_compute, m_input, m_eps);
    h = ggml_mul(m_ctx_compute, h, w_norm);
    h = ggml_add(m_ctx_compute, h, b_norm);
    ggml_tensor * logits = ggml_mul_mat(m_ctx_compute, w_out, h);
    m_probs = ggml_soft_max(m_ctx_compute, logits);
    ggml_set_name(m_probs, "lens_probs");
    ggml_set_output(m_probs);

    m_graph = ggml_new_graph(m_ctx_compute);
    ggml_build_forward_expand(m_graph, m_probs);

    m_galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m_backend));
    if (!m_galloc) {
        std::fprintf(stderr, "lens: gallocr_new failed\n");
        free_all(); return;
    }
    if (!ggml_gallocr_alloc_graph(m_galloc, m_graph)) {
        std::fprintf(stderr, "lens: gallocr_alloc_graph failed\n");
        free_all(); return;
    }

    m_ok = true;
}

LogitLens::~LogitLens() {
    free_all();
}

void LogitLens::free_all() {
    if (m_galloc)      { ggml_gallocr_free(m_galloc);          m_galloc      = nullptr; }
    if (m_ctx_compute) { ggml_free(m_ctx_compute);             m_ctx_compute = nullptr; }
    if (m_buf_w)       { ggml_backend_buffer_free(m_buf_w);    m_buf_w       = nullptr; }
    if (m_ctx_weights) { ggml_free(m_ctx_weights);             m_ctx_weights = nullptr; }
    if (m_gctx)        { gguf_free(m_gctx);                    m_gctx        = nullptr; }
    if (m_backend)     { ggml_backend_free(m_backend);         m_backend     = nullptr; }
    m_ok = false;
}

std::vector<float> LogitLens::run(span<const float> residual) {
    std::vector<float> probs(static_cast<std::size_t>(m_n_vocab));
    run_into(residual, span<float>(probs.data(), probs.size()));
    return probs;
}

void LogitLens::run_into(span<const float> residual, span<float> probs_out) {
    if (!m_ok) {
        die("LogitLens::run on un-initialized lens (ok()=false)");
    }
    if (static_cast<int>(residual.size()) != m_n_embd) {
        die("LogitLens::run: residual.size()=" + std::to_string(residual.size())
            + " expected n_embd=" + std::to_string(m_n_embd));
    }
    if (static_cast<int>(probs_out.size()) != m_n_vocab) {
        die("LogitLens::run: probs_out.size()=" + std::to_string(probs_out.size())
            + " expected n_vocab=" + std::to_string(m_n_vocab));
    }

    ggml_backend_tensor_set(m_input, residual.data(), 0,
                            static_cast<std::size_t>(m_n_embd) * sizeof(float));
    ggml_backend_graph_compute(m_backend, m_graph);
    ggml_backend_tensor_get(m_probs, probs_out.data(), 0,
                            static_cast<std::size_t>(m_n_vocab) * sizeof(float));
}

}  // namespace rwkv_probe
