// rwkv_probe/capture.h — multi-hook cb_eval registry for residual stream
// captures and mutations.
//
// Generalizes the lens_cb_eval pattern from the original rwkv-probe.cpp
// (lines 218-245) so that multiple consumers can register read or write hooks
// against tensors produced inside the llama_decode forward pass.
#pragma once

#include "util.h"
#include "model.h"

#include "ggml.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace rwkv_probe {

// Lightweight, non-owning view of a ggml_tensor produced by cb_eval. The
// underlying tensor lives on whatever backend the model is using; reading the
// data requires ggml_backend_tensor_get (which copies it to CPU).
struct TensorView {
    const ggml_tensor * tensor = nullptr;
    int                 layer  = -1;     // parsed from "<name>-<il>" if present
    const char *        name   = nullptr;

    // helper: copy the last token's column (last row of dim 1) into dst.
    // dst.size() must equal tensor->ne[0]. throws on size mismatch.
    void copy_last_token(span<float> dst) const;

    // helper: copy the entire tensor into dst (size = ne[0] * ne[1]).
    void copy_all(span<float> dst) const;
};

// Same as TensorView but the read+write hook is allowed to write back via
// ggml_backend_tensor_set after returning. The mutator should leave the
// underlying buffer in a valid state.
struct MutableTensorView : TensorView {
    // copies dst back into the backend at the same location read from.
    // Must match the shape used in the corresponding copy_*. Throws on
    // size mismatch.
    void store_last_token(span<const float> src) const;
    void store_all(span<const float> src) const;
};

using HookId = std::uint32_t;
constexpr HookId kInvalidHookId = 0;

class CaptureRegistry {
public:
    using ReadHook  = std::function<void(const TensorView &)>;
    using WriteHook = std::function<void(MutableTensorView &)>;

    explicit CaptureRegistry(const ModelGeometry & geom);

    // pattern is matched against the tensor name. supported forms:
    //   "exact"            — strcmp
    //   "prefix-"          — name starts with "prefix-"
    //   "l_out-*"          — l_out- followed by an integer (layer)
    HookId on_tensor(std::string pattern, ReadHook hook);

    // shortcut: match l_out-<layer> exactly. layer < 0 means all layers.
    HookId on_residual(int layer, ReadHook hook);
    HookId on_residual_all(ReadHook hook);

    // shortcut: write hook on l_out-<layer>. After the hook returns, the
    // residual is copied back to the backend tensor. layer < 0 = all layers.
    HookId on_residual_mutate(int layer, WriteHook hook);

    void remove(HookId id);
    void enable(HookId id, bool on);
    void clear();

    // The trampoline that gets installed into llama_context_params.cb_eval.
    // user_data must be the CaptureRegistry pointer.
    static bool cb_eval_trampoline(ggml_tensor * t, bool ask, void * user_data);

private:
    struct Entry {
        HookId      id      = 0;
        std::string pattern;
        int         layer   = -1;       // parsed from pattern if applicable
        bool        is_l_out = false;   // shortcut for the common case
        bool        is_write = false;
        bool        enabled  = true;
        ReadHook    read;
        WriteHook   write;
    };

    bool dispatch(ggml_tensor * t, bool ask);

    ModelGeometry      m_geom;
    std::vector<Entry> m_entries;
    HookId             m_next_id = 1;
};

}  // namespace rwkv_probe
