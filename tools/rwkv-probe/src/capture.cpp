// rwkv_probe/capture.cpp — CaptureRegistry implementation.
//
// Generalizes the lens_cb_eval pattern from the original rwkv-probe.cpp into
// a multi-hook registry that supports both read and read-modify-write hooks
// against tensors produced inside the llama_decode forward pass.
#include "rwkv_probe/capture.h"

#include "ggml-backend.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace rwkv_probe {

// ── helpers ───────────────────────────────────────────────────────────────────

// returns the integer suffix of "<prefix>-<int>", or -1 if no match.
static int parse_int_suffix(const char * name, const char * prefix) {
    const std::size_t pl = std::strlen(prefix);
    if (std::strncmp(name, prefix, pl) != 0) {
        return -1;
    }
    const char * p = name + pl;
    if (*p == '\0') {
        return -1;
    }
    char * end = nullptr;
    long v = std::strtol(p, &end, 10);
    if (end == p || *end != '\0') {
        return -1;
    }
    return static_cast<int>(v);
}

static bool name_matches(const char * tensor_name, const std::string & pattern,
                         int * out_layer) {
    if (pattern.size() >= 2 && pattern.compare(pattern.size() - 2, 2, "-*") == 0) {
        const std::string prefix = pattern.substr(0, pattern.size() - 1);  // keep trailing '-'
        const int il = parse_int_suffix(tensor_name, prefix.c_str());
        if (il < 0) {
            return false;
        }
        if (out_layer) *out_layer = il;
        return true;
    }
    if (std::strcmp(tensor_name, pattern.c_str()) == 0) {
        if (out_layer) {
            // try to parse a trailing -N if present
            const char * dash = std::strrchr(tensor_name, '-');
            if (dash) {
                char * end = nullptr;
                long v = std::strtol(dash + 1, &end, 10);
                if (end != dash + 1 && *end == '\0') {
                    *out_layer = static_cast<int>(v);
                }
            }
        }
        return true;
    }
    return false;
}

// ── TensorView helpers ────────────────────────────────────────────────────────
void TensorView::copy_last_token(span<float> dst) const {
    const int64_t ne0 = tensor->ne[0];
    const int64_t ne1 = tensor->ne[1];
    if (static_cast<int64_t>(dst.size()) != ne0) {
        die("TensorView::copy_last_token: dst.size()=" + std::to_string(dst.size())
            + " expected " + std::to_string(ne0));
    }
    if (ne1 < 1) {
        die("TensorView::copy_last_token: tensor has ne1 < 1");
    }
    const std::size_t nb1 = tensor->nb[1];
    const std::size_t off = static_cast<std::size_t>(ne1 - 1) * nb1;
    ggml_backend_tensor_get(tensor, dst.data(), off,
                            static_cast<std::size_t>(ne0) * sizeof(float));
}

void TensorView::copy_all(span<float> dst) const {
    const int64_t ne0 = tensor->ne[0];
    const int64_t ne1 = tensor->ne[1];
    const int64_t total = ne0 * ne1;
    if (static_cast<int64_t>(dst.size()) != total) {
        die("TensorView::copy_all: dst.size()=" + std::to_string(dst.size())
            + " expected " + std::to_string(total));
    }
    ggml_backend_tensor_get(tensor, dst.data(), 0,
                            static_cast<std::size_t>(total) * sizeof(float));
}

void MutableTensorView::store_last_token(span<const float> src) const {
    const int64_t ne0 = tensor->ne[0];
    const int64_t ne1 = tensor->ne[1];
    if (static_cast<int64_t>(src.size()) != ne0) {
        die("MutableTensorView::store_last_token: src.size()=" + std::to_string(src.size())
            + " expected " + std::to_string(ne0));
    }
    if (ne1 < 1) {
        die("MutableTensorView::store_last_token: tensor has ne1 < 1");
    }
    const std::size_t nb1 = tensor->nb[1];
    const std::size_t off = static_cast<std::size_t>(ne1 - 1) * nb1;
    // const_cast: ggml_backend_tensor_set takes a non-const ggml_tensor *.
    // The view itself stores a const pointer for type safety in read paths.
    ggml_backend_tensor_set(const_cast<ggml_tensor *>(tensor), src.data(), off,
                            static_cast<std::size_t>(ne0) * sizeof(float));
}

void MutableTensorView::store_all(span<const float> src) const {
    const int64_t ne0 = tensor->ne[0];
    const int64_t ne1 = tensor->ne[1];
    const int64_t total = ne0 * ne1;
    if (static_cast<int64_t>(src.size()) != total) {
        die("MutableTensorView::store_all: src.size()=" + std::to_string(src.size())
            + " expected " + std::to_string(total));
    }
    ggml_backend_tensor_set(const_cast<ggml_tensor *>(tensor), src.data(), 0,
                            static_cast<std::size_t>(total) * sizeof(float));
}

// ── CaptureRegistry ───────────────────────────────────────────────────────────
CaptureRegistry::CaptureRegistry(const ModelGeometry & geom) : m_geom(geom) {}

HookId CaptureRegistry::on_tensor(std::string pattern, ReadHook hook) {
    Entry e;
    e.id       = m_next_id++;
    e.pattern  = std::move(pattern);
    e.is_l_out = (e.pattern == "l_out-*"
                  || (e.pattern.rfind("l_out-", 0) == 0));
    e.is_write = false;
    e.read     = std::move(hook);
    m_entries.push_back(std::move(e));
    return m_entries.back().id;
}

HookId CaptureRegistry::on_residual(int layer, ReadHook hook) {
    Entry e;
    e.id       = m_next_id++;
    e.pattern  = (layer < 0) ? "l_out-*" : ("l_out-" + std::to_string(layer));
    e.layer    = layer;
    e.is_l_out = true;
    e.is_write = false;
    e.read     = std::move(hook);
    m_entries.push_back(std::move(e));
    return m_entries.back().id;
}

HookId CaptureRegistry::on_residual_all(ReadHook hook) {
    return on_residual(-1, std::move(hook));
}

HookId CaptureRegistry::on_residual_mutate(int layer, WriteHook hook) {
    Entry e;
    e.id       = m_next_id++;
    e.pattern  = (layer < 0) ? "l_out-*" : ("l_out-" + std::to_string(layer));
    e.layer    = layer;
    e.is_l_out = true;
    e.is_write = true;
    e.write    = std::move(hook);
    m_entries.push_back(std::move(e));
    return m_entries.back().id;
}

void CaptureRegistry::remove(HookId id) {
    for (auto it = m_entries.begin(); it != m_entries.end(); ++it) {
        if (it->id == id) {
            m_entries.erase(it);
            return;
        }
    }
}

void CaptureRegistry::enable(HookId id, bool on) {
    for (auto & e : m_entries) {
        if (e.id == id) {
            e.enabled = on;
            return;
        }
    }
}

void CaptureRegistry::clear() {
    m_entries.clear();
}

// hot path. called twice per tensor: ask=true (consult), ask=false (fire).
bool CaptureRegistry::dispatch(ggml_tensor * t, bool ask) {
    if (m_entries.empty()) {
        return ask ? false : true;
    }
    const char * name = t->name;
    if (!name) {
        return ask ? false : true;
    }

    // fast-path: most hooks are on l_out-<il>. extract il once.
    int il_for_l_out = -1;
    bool is_l_out = false;
    if (std::strncmp(name, "l_out-", 6) == 0) {
        is_l_out = true;
        il_for_l_out = parse_int_suffix(name, "l_out-");
    }

    if (ask) {
        for (const auto & e : m_entries) {
            if (!e.enabled) continue;
            if (e.is_l_out && is_l_out) {
                if (e.layer < 0 || e.layer == il_for_l_out) {
                    return true;
                }
            } else {
                int dummy = -1;
                if (name_matches(name, e.pattern, &dummy)) {
                    return true;
                }
            }
        }
        return false;
    }

    // fire phase
    for (auto & e : m_entries) {
        if (!e.enabled) continue;

        int matched_layer = -1;
        bool match = false;
        if (e.is_l_out && is_l_out) {
            if (e.layer < 0 || e.layer == il_for_l_out) {
                match = true;
                matched_layer = il_for_l_out;
            }
        } else if (name_matches(name, e.pattern, &matched_layer)) {
            match = true;
        }
        if (!match) continue;

        if (e.is_write) {
            MutableTensorView v;
            v.tensor = t;
            v.layer  = matched_layer;
            v.name   = name;
            e.write(v);
        } else {
            TensorView v;
            v.tensor = t;
            v.layer  = matched_layer;
            v.name   = name;
            e.read(v);
        }
    }
    return true;
}

bool CaptureRegistry::cb_eval_trampoline(ggml_tensor * t, bool ask, void * user_data) {
    auto * self = static_cast<CaptureRegistry *>(user_data);
    if (!self) {
        return ask ? false : true;
    }
    return self->dispatch(t, ask);
}

}  // namespace rwkv_probe
