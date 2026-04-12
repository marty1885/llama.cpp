// rwkv_probe/state.h — recurrent state buffer with garbage-proof accessors.
//
// Lifts the flat r/s buffers + capture_state lambda + patching/steering math
// from the original rwkv-probe.cpp (lines 593-727).
//
// SAFETY CONTRACT (the "no garbage" guarantee the user asked for):
//   1. StateBuf is sized at construction from a ModelGeometry; sizes are never
//      mutated afterwards.
//   2. Every per-layer / per-head accessor bounds-checks its index and throws
//      on out-of-range.
//   3. load_from / store_to validate that the buffer sizes still match what
//      llama_recurrent_state_*_f32 expects for the given context.
//   4. copy_layers_from rejects donors with a different ModelGeometry — this
//      is the exact bug class that caused the "cocaine -> ransomware" stale
//      cache incident, just at a different layer.
//   5. assert_finite() optionally scans for NaN/inf before store_to.
//   6. No default constructor — every StateBuf is born with a geometry.
#pragma once

#include "util.h"
#include "model.h"

#include "llama.h"

#include <vector>

namespace rwkv_probe {

class StateBuf {
public:
    explicit StateBuf(const ModelGeometry & geom);

    // copy and move are both fine; the buffers are owned vectors.
    StateBuf(const StateBuf &)             = default;
    StateBuf(StateBuf &&)                  = default;
    StateBuf & operator=(const StateBuf &) = default;
    StateBuf & operator=(StateBuf &&)      = default;

    const ModelGeometry & geom() const { return m_geom; }

    // ── whole-state I/O against a context ────────────────────────────────────
    // wraps llama_recurrent_state_get_f32 / set_f32. seq_id defaults to 0,
    // matching the existing rwkv-probe usage.
    void load_from (llama_context * ctx, int seq_id = 0);
    void store_to  (llama_context * ctx, int seq_id = 0) const;

    // ── flat views (the format llama_recurrent_state_*_f32 wants) ────────────
    span<float>       r_flat()       { return span<float>      (m_r.data(), m_r.size()); }
    span<float>       s_flat()       { return span<float>      (m_s.data(), m_s.size()); }
    span<const float> r_flat() const { return span<const float>(m_r.data(), m_r.size()); }
    span<const float> s_flat() const { return span<const float>(m_s.data(), m_s.size()); }

    // ── per-layer split views ────────────────────────────────────────────────
    // r_att and r_ffn are the two halves of the layer's r state.
    // bounds-checked; throws on layer out of range.
    span<float>       r_att(int layer);
    span<float>       r_ffn(int layer);
    span<float>       s_wkv(int layer);
    span<const float> r_att(int layer) const;
    span<const float> r_ffn(int layer) const;
    span<const float> s_wkv(int layer) const;

    // single attention head's WKV state slice (head_size * head_size floats).
    // bounds-checked on both layer and head.
    span<float>       s_head(int layer, int head);
    span<const float> s_head(int layer, int head) const;

    // ── partial restore (state replay / activation patching) ─────────────────
    // copies selected layers from src into this buffer. throws if the donor
    // geometry differs from this buffer's geometry, or if any layer index is
    // out of range. include_r controls whether r_att/r_ffn are also copied
    // (default: only S, matching the original --patch-include-r=false default).
    void copy_layers_from(const StateBuf &        src,
                          span<const int>         layers,
                          bool                    include_r = false);

    // ── safety helpers ───────────────────────────────────────────────────────
    // throws if any element is NaN or inf. Off the hot path; call only when
    // an experiment has reason to suspect bad math upstream.
    void assert_finite() const;

    // throws if `other`'s geometry differs from this buffer's. used by
    // copy_layers_from and as a defensive check in experiments that mix
    // states from multiple sources.
    void assert_geometry_matches(const StateBuf & other) const;

private:
    void check_layer(int layer) const;
    void check_head (int head)  const;

    ModelGeometry      m_geom;
    std::vector<float> m_r;   // n_layer * n_embd_r
    std::vector<float> m_s;   // n_layer * n_embd_s
};

}  // namespace rwkv_probe
