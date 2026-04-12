// rwkv_probe/model.h — model + context lifecycle for RWKV experiments.
//
// Lifts lines 510-567 of the original rwkv-probe.cpp: backend init, model load,
// geometry query, context creation with optional cb_eval wiring.
#pragma once

#include "util.h"

#include "llama.h"

#include <string>

namespace rwkv_probe {

class CaptureRegistry;  // capture.h — fwd decl, optional dependency

// Geometry pulled from a loaded model. Cached so experiments don't re-query
// every loop iteration.
struct ModelGeometry {
    int n_layer  = 0;
    int n_embd   = 0;
    int n_embd_r = 0;   // token-shift state size per layer (= 2 * n_embd for RWKV7)
    int n_embd_s = 0;   // WKV state size per layer        (= n_embd * head_size)
    int n_vocab  = 0;

    // derived
    int head_size = 0;  // n_embd_s / n_embd
    int n_head    = 0;  // n_embd / head_size
    int n_half    = 0;  // n_embd_r / 2 (r_att / r_ffn split)

    bool operator==(const ModelGeometry & o) const {
        return n_layer == o.n_layer && n_embd == o.n_embd
            && n_embd_r == o.n_embd_r && n_embd_s == o.n_embd_s
            && n_vocab == o.n_vocab;
    }
    bool operator!=(const ModelGeometry & o) const { return !(*this == o); }
};

ModelGeometry query_geometry(const llama_model * m);

// RAII wrapper around llama_backend_init / llama_backend_free.
// Construct exactly one of these per process, before any Model.
class Backend {
public:
    Backend();
    ~Backend();

    Backend(const Backend &)             = delete;
    Backend & operator=(const Backend &) = delete;
};

// RAII wrapper around llama_model_load_from_file.
class Model {
public:
    explicit Model(const std::string & path,
                   llama_model_params params = llama_model_default_params());
    ~Model();

    Model(const Model &)             = delete;
    Model & operator=(const Model &) = delete;

    llama_model *       raw()        { return m_model; }
    const llama_model * raw() const  { return m_model; }
    const llama_vocab * vocab() const;
    const ModelGeometry & geom() const { return m_geom; }
    const std::string & path() const { return m_path; }

    bool is_recurrent() const;

private:
    llama_model * m_model = nullptr;
    ModelGeometry m_geom{};
    std::string   m_path;
};

// RAII wrapper around llama_init_from_model. If `caps` is non-null, its
// cb_eval trampoline + user_data are installed into the context params before
// creation. Caller is responsible for keeping `caps` alive at least as long as
// this Context.
class Context {
public:
    Context(Model & m,
            llama_context_params params = llama_context_default_params(),
            CaptureRegistry * caps      = nullptr);
    ~Context();

    Context(const Context &)             = delete;
    Context & operator=(const Context &) = delete;

    llama_context *       raw()       { return m_ctx; }
    const llama_context * raw() const { return m_ctx; }

    // wraps llama_memory_clear(llama_get_memory(ctx), true)
    void clear_memory();

    // wraps llama_batch_get_one + llama_decode for a span of tokens
    // returns 0 on success, non-zero on llama_decode failure.
    int decode(span<const llama_token> tokens);
    int decode_one(llama_token tok);

private:
    llama_context * m_ctx = nullptr;
};

}  // namespace rwkv_probe
