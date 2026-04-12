// rwkv_probe/model.cpp — Backend / Model / Context implementations.
#include "rwkv_probe/model.h"
#include "rwkv_probe/capture.h"

#include "llama.h"

namespace rwkv_probe {

ModelGeometry query_geometry(const llama_model * m) {
    ModelGeometry g;
    g.n_layer  = llama_model_n_layer(m);
    g.n_embd   = llama_model_n_embd(m);
    g.n_embd_r = llama_model_n_embd_r(m);
    g.n_embd_s = llama_model_n_embd_s(m);
    g.n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(m));

    if (g.n_embd > 0) {
        g.head_size = g.n_embd_s / g.n_embd;
    }
    if (g.head_size > 0) {
        g.n_head = g.n_embd / g.head_size;
    }
    g.n_half = g.n_embd_r / 2;

    return g;
}

// ── Backend ───────────────────────────────────────────────────────────────────
Backend::Backend()  { llama_backend_init(); }
Backend::~Backend() { llama_backend_free(); }

// ── Model ─────────────────────────────────────────────────────────────────────
Model::Model(const std::string & path, llama_model_params params) : m_path(path) {
    m_model = llama_model_load_from_file(path.c_str(), params);
    if (!m_model) {
        die("failed to load model from " + path);
    }
    m_geom = query_geometry(m_model);
}

Model::~Model() {
    if (m_model) {
        llama_model_free(m_model);
        m_model = nullptr;
    }
}

const llama_vocab * Model::vocab() const {
    return llama_model_get_vocab(m_model);
}

bool Model::is_recurrent() const {
    return llama_model_is_recurrent(m_model);
}

// ── Context ───────────────────────────────────────────────────────────────────
Context::Context(Model & m, llama_context_params params, CaptureRegistry * caps) {
    if (caps) {
        params.cb_eval           = &CaptureRegistry::cb_eval_trampoline;
        params.cb_eval_user_data = caps;
    }
    m_ctx = llama_init_from_model(m.raw(), params);
    if (!m_ctx) {
        die("failed to create llama_context");
    }
}

Context::~Context() {
    if (m_ctx) {
        llama_free(m_ctx);
        m_ctx = nullptr;
    }
}

void Context::clear_memory() {
    llama_memory_clear(llama_get_memory(m_ctx), true);
}

int Context::decode(span<const llama_token> tokens) {
    if (tokens.empty()) {
        return 0;
    }
    // llama_batch_get_one wants a non-const pointer, but it does not modify
    // the buffer in practice. const_cast is the existing pattern in rwkv-probe.cpp.
    llama_batch batch = llama_batch_get_one(
        const_cast<llama_token *>(tokens.data()),
        static_cast<int32_t>(tokens.size()));
    return llama_decode(m_ctx, batch);
}

int Context::decode_one(llama_token tok) {
    llama_batch batch = llama_batch_get_one(&tok, 1);
    return llama_decode(m_ctx, batch);
}

}  // namespace rwkv_probe
