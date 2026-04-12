// rwkv_probe/state.cpp — StateBuf implementation.
#include "rwkv_probe/state.h"

#include "llama.h"

#include <cmath>
#include <cstddef>
#include <string>

namespace rwkv_probe {

StateBuf::StateBuf(const ModelGeometry & geom) : m_geom(geom) {
    if (geom.n_layer <= 0 || geom.n_embd_r <= 0 || geom.n_embd_s <= 0) {
        die("StateBuf: invalid geometry (n_layer=" + std::to_string(geom.n_layer)
            + " n_embd_r=" + std::to_string(geom.n_embd_r)
            + " n_embd_s=" + std::to_string(geom.n_embd_s) + ")");
    }
    m_r.assign(static_cast<std::size_t>(geom.n_layer) * geom.n_embd_r, 0.0f);
    m_s.assign(static_cast<std::size_t>(geom.n_layer) * geom.n_embd_s, 0.0f);
}

// ── safety helpers ────────────────────────────────────────────────────────────
void StateBuf::check_layer(int layer) const {
    if (layer < 0 || layer >= m_geom.n_layer) {
        die("StateBuf: layer " + std::to_string(layer) + " out of range [0, "
            + std::to_string(m_geom.n_layer) + ")");
    }
}

void StateBuf::check_head(int head) const {
    if (head < 0 || head >= m_geom.n_head) {
        die("StateBuf: head " + std::to_string(head) + " out of range [0, "
            + std::to_string(m_geom.n_head) + ")");
    }
}

void StateBuf::assert_geometry_matches(const StateBuf & other) const {
    if (m_geom != other.m_geom) {
        die("StateBuf: geometry mismatch (this n_layer=" + std::to_string(m_geom.n_layer)
            + " n_embd_r=" + std::to_string(m_geom.n_embd_r)
            + " n_embd_s=" + std::to_string(m_geom.n_embd_s)
            + " vs other n_layer=" + std::to_string(other.m_geom.n_layer)
            + " n_embd_r=" + std::to_string(other.m_geom.n_embd_r)
            + " n_embd_s=" + std::to_string(other.m_geom.n_embd_s) + ")");
    }
}

void StateBuf::assert_finite() const {
    for (std::size_t i = 0; i < m_r.size(); ++i) {
        if (!std::isfinite(m_r[i])) {
            die("StateBuf: non-finite value in r_flat at index " + std::to_string(i));
        }
    }
    for (std::size_t i = 0; i < m_s.size(); ++i) {
        if (!std::isfinite(m_s[i])) {
            die("StateBuf: non-finite value in s_flat at index " + std::to_string(i));
        }
    }
}

// ── whole-state I/O ───────────────────────────────────────────────────────────
void StateBuf::load_from(llama_context * ctx, int seq_id) {
    // sanity: buffer sizes must still match what we'll ask the API to write into.
    const std::size_t expected_r = static_cast<std::size_t>(m_geom.n_layer) * m_geom.n_embd_r;
    const std::size_t expected_s = static_cast<std::size_t>(m_geom.n_layer) * m_geom.n_embd_s;
    if (m_r.size() != expected_r || m_s.size() != expected_s) {
        die("StateBuf::load_from: buffer size mismatch (have r=" + std::to_string(m_r.size())
            + " s=" + std::to_string(m_s.size())
            + " expected r=" + std::to_string(expected_r)
            + " s=" + std::to_string(expected_s) + ")");
    }
    // ensure GPU work is complete before reading state tensors
    llama_synchronize(ctx);
    llama_recurrent_state_get_f32(ctx, seq_id, m_r.data(), m_s.data());
}

void StateBuf::store_to(llama_context * ctx, int seq_id) const {
    const std::size_t expected_r = static_cast<std::size_t>(m_geom.n_layer) * m_geom.n_embd_r;
    const std::size_t expected_s = static_cast<std::size_t>(m_geom.n_layer) * m_geom.n_embd_s;
    if (m_r.size() != expected_r || m_s.size() != expected_s) {
        die("StateBuf::store_to: buffer size mismatch (have r=" + std::to_string(m_r.size())
            + " s=" + std::to_string(m_s.size())
            + " expected r=" + std::to_string(expected_r)
            + " s=" + std::to_string(expected_s) + ")");
    }
    // ensure GPU work is complete before overwriting state tensors
    llama_synchronize(ctx);
    bool ok = llama_recurrent_state_set_f32(ctx, seq_id, m_r.data(), m_s.data());
    if (!ok) {
        die("StateBuf::store_to: llama_recurrent_state_set_f32 returned false "
            "(seq_id=" + std::to_string(seq_id) + " — no cell found?)");
    }
}

// ── per-layer split views ─────────────────────────────────────────────────────
span<float> StateBuf::r_att(int layer) {
    check_layer(layer);
    const std::size_t off = static_cast<std::size_t>(layer) * m_geom.n_embd_r;
    return span<float>(m_r.data() + off, static_cast<std::size_t>(m_geom.n_half));
}
span<float> StateBuf::r_ffn(int layer) {
    check_layer(layer);
    const std::size_t off = static_cast<std::size_t>(layer) * m_geom.n_embd_r
                          + static_cast<std::size_t>(m_geom.n_half);
    return span<float>(m_r.data() + off, static_cast<std::size_t>(m_geom.n_half));
}
span<float> StateBuf::s_wkv(int layer) {
    check_layer(layer);
    const std::size_t off = static_cast<std::size_t>(layer) * m_geom.n_embd_s;
    return span<float>(m_s.data() + off, static_cast<std::size_t>(m_geom.n_embd_s));
}
span<float> StateBuf::s_head(int layer, int head) {
    check_layer(layer);
    check_head(head);
    const std::size_t off = static_cast<std::size_t>(layer) * m_geom.n_embd_s
                          + static_cast<std::size_t>(head)  * m_geom.head_size * m_geom.head_size;
    return span<float>(m_s.data() + off,
                       static_cast<std::size_t>(m_geom.head_size) * m_geom.head_size);
}

span<const float> StateBuf::r_att(int layer) const {
    check_layer(layer);
    const std::size_t off = static_cast<std::size_t>(layer) * m_geom.n_embd_r;
    return span<const float>(m_r.data() + off, static_cast<std::size_t>(m_geom.n_half));
}
span<const float> StateBuf::r_ffn(int layer) const {
    check_layer(layer);
    const std::size_t off = static_cast<std::size_t>(layer) * m_geom.n_embd_r
                          + static_cast<std::size_t>(m_geom.n_half);
    return span<const float>(m_r.data() + off, static_cast<std::size_t>(m_geom.n_half));
}
span<const float> StateBuf::s_wkv(int layer) const {
    check_layer(layer);
    const std::size_t off = static_cast<std::size_t>(layer) * m_geom.n_embd_s;
    return span<const float>(m_s.data() + off, static_cast<std::size_t>(m_geom.n_embd_s));
}
span<const float> StateBuf::s_head(int layer, int head) const {
    check_layer(layer);
    check_head(head);
    const std::size_t off = static_cast<std::size_t>(layer) * m_geom.n_embd_s
                          + static_cast<std::size_t>(head)  * m_geom.head_size * m_geom.head_size;
    return span<const float>(m_s.data() + off,
                             static_cast<std::size_t>(m_geom.head_size) * m_geom.head_size);
}

// ── partial restore (state replay) ────────────────────────────────────────────
void StateBuf::copy_layers_from(const StateBuf & src,
                                span<const int>  layers,
                                bool             include_r) {
    assert_geometry_matches(src);
    for (std::size_t i = 0; i < layers.size(); ++i) {
        const int il = layers[i];
        check_layer(il);

        // S state
        const std::size_t s_off = static_cast<std::size_t>(il) * m_geom.n_embd_s;
        for (int j = 0; j < m_geom.n_embd_s; ++j) {
            m_s[s_off + j] = src.m_s[s_off + j];
        }

        if (include_r) {
            const std::size_t r_off = static_cast<std::size_t>(il) * m_geom.n_embd_r;
            for (int j = 0; j < m_geom.n_embd_r; ++j) {
                m_r[r_off + j] = src.m_r[r_off + j];
            }
        }
    }
}

}  // namespace rwkv_probe
