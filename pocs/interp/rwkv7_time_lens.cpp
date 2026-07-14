#include "common.h"
#include "llama-context.h"
#include "llama-memory-recurrent.h"
#include "llama-model.h"
#include "models/models.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct rwkv_state {
    std::vector<std::vector<float>> r;
    std::vector<std::vector<float>> s;
};

struct scored_token {
    int id;
    std::string piece;
    float logit;
};

struct reported_token {
    int id;
    std::string piece;
    int rank;
    float logit;
};

struct source_readout {
    std::string source;
    std::vector<scored_token> top_tokens;
    std::vector<reported_token> reported_tokens;
    float baseline_cosine;
    int target_rank;
    float target_logit;
    int random_target_rank;
    float random_target_logit;
};

struct step_result {
    llama_token input;
    llama_token next;
    std::vector<source_readout> readouts;
    std::vector<reported_token> final_reported_tokens;
    std::vector<float> state_cosine;
};

struct snapshot {
    rwkv_state next_state;
    std::vector<std::vector<float>> resid_in;
    struct raw_layer {
        std::vector<float> r;
        std::vector<float> w;
        std::vector<float> k;
        std::vector<float> v;
        std::vector<float> a;
        std::vector<float> a_pre;
        std::vector<float> g;
        std::vector<float> k0;
        std::vector<float> kk;
        std::vector<float> wkv;
        std::vector<float> rkv;
        std::vector<float> channel_out;
    };
    std::vector<raw_layer> raw;
    std::vector<float> logits;
    std::vector<float> projected_logits;
};

struct projection_workspace {
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor * packed = nullptr;
    size_t n_columns = 0;
};

enum class source_kind { r, w, k, v, a, a_pre, g, k0, kk, wkv, rkv, channel_out };

static constexpr size_t projection_source_count = 12;

enum class run_mode { project, local_lens };

enum steer_component : uint32_t {
    steer_r = 1 << 0,
    steer_w = 1 << 1,
    steer_k = 1 << 2,
    steer_v = 1 << 3,
    steer_a = 1 << 4,
};

static constexpr uint32_t steer_default_components = steer_k | steer_v;

static const char * steer_component_name(steer_component component) {
    switch (component) {
        case steer_r: return "r";
        case steer_w: return "w";
        case steer_k: return "k";
        case steer_v: return "v";
        case steer_a: return "a";
    }
    GGML_ABORT("unknown steering component");
}

static uint32_t parse_steer_components(const std::string & value) {
    uint32_t result = 0;
    size_t begin = 0;
    while (begin < value.size()) {
        const size_t end = value.find(',', begin);
        const std::string item = value.substr(begin, end == std::string::npos ? end : end - begin);
        steer_component component;
        if (item == "r") component = steer_r;
        else if (item == "w") component = steer_w;
        else if (item == "k") component = steer_k;
        else if (item == "v") component = steer_v;
        else if (item == "a") component = steer_a;
        else throw std::runtime_error("--steer-edit must be a comma-separated subset of r,w,k,v,a");
        if (result & component) throw std::runtime_error("duplicate --steer-edit component");
        result |= component;
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    if (!result) throw std::runtime_error("empty --steer-edit");
    return result;
}

static std::string steer_components_label(uint32_t components) {
    if (!components) return "clean";
    std::string label;
    for (steer_component component : { steer_r, steer_w, steer_k, steer_v, steer_a }) {
        if (!(components & component)) continue;
        if (!label.empty()) label += "-";
        label += steer_component_name(component);
    }
    return label;
}

static std::string sweep_json_path(const std::string & path, uint32_t components) {
    const size_t extension = path.rfind('.');
    const std::string suffix = "-" + steer_components_label(components);
    if (extension == std::string::npos) return path + suffix;
    return path.substr(0, extension) + suffix + path.substr(extension);
}

struct activation_patch {
    int layer;
    std::vector<float> k;
    std::vector<float> kk;
    std::vector<float> v;
};

enum class swap_source { k, v, kv };

static swap_source parse_swap_source(const std::string & value) {
    if (value == "k") return swap_source::k;
    if (value == "v") return swap_source::v;
    if (value == "kv") return swap_source::kv;
    throw std::runtime_error("--swap-source must be k, v, or kv");
}

static const char * source_name(source_kind source) {
    switch (source) {
        case source_kind::r:   return "r";
        case source_kind::w:   return "w";
        case source_kind::k:   return "k";
        case source_kind::v:   return "v";
        case source_kind::a:   return "a";
        case source_kind::a_pre: return "a_pre";
        case source_kind::g:   return "g";
        case source_kind::k0:  return "k0";
        case source_kind::kk:  return "kk";
        case source_kind::wkv: return "wkv";
        case source_kind::rkv: return "rkv";
        case source_kind::channel_out: return "channel.out";
    }
    GGML_ABORT("unknown source");
}

static source_kind parse_source(const std::string & value) {
    for (source_kind source : { source_kind::r, source_kind::w, source_kind::k, source_kind::v,
                                source_kind::a, source_kind::a_pre, source_kind::g, source_kind::k0,
                                source_kind::kk, source_kind::wkv, source_kind::rkv, source_kind::channel_out }) {
        if (value == source_name(source)) return source;
    }
    throw std::runtime_error("unknown --source: " + value);
}

static std::vector<source_kind> all_source_kinds() {
    return { source_kind::r, source_kind::w, source_kind::k, source_kind::v,
             source_kind::a, source_kind::a_pre, source_kind::g, source_kind::k0,
             source_kind::kk, source_kind::wkv, source_kind::rkv, source_kind::channel_out };
}

static std::vector<source_kind> parse_project_sources(const std::string & value) {
    if (value == "all") return all_source_kinds();
    std::vector<source_kind> result;
    size_t begin = 0;
    while (begin < value.size()) {
        const size_t end = value.find(',', begin);
        const std::string item = value.substr(begin, end == std::string::npos ? end : end - begin);
        if (item.empty()) throw std::runtime_error("empty --project-sources item");
        const source_kind source = parse_source(item);
        if (std::find(result.begin(), result.end(), source) == result.end()) result.push_back(source);
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    if (result.empty()) throw std::runtime_error("empty --project-sources");
    return result;
}

static run_mode parse_mode(const std::string & value) {
    if (value == "project") return run_mode::project;
    if (value == "local-lens") return run_mode::local_lens;
    throw std::runtime_error("unknown --mode: " + value);
}

struct direct_graph : llm_build_rwkv7_base {
    direct_graph(const llama_model & model, const llm_graph_params & params) : llm_build_rwkv7_base(model, params) {}
};

static void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s -m MODEL [-p PROMPT] [-ngl N] [--mode project|local-lens] [--source r|w|k|v|a|a_pre|g|k0|kk|wkv|rkv|channel.out] [--project-sources all|k,v,a,a_pre,k0,kk,channel.out] [--no-in-graph-unembed] [--random-baseline] [--capture-prompt-last] [--track-text TEXT] [--report-text TEXT]... [--forced-text TEXT] [--steps N] [--epsilon E] [--top N] [--steer-source TEXT --steer-target TEXT [--steer-edit r,w,k,v,a|--steer-sweep r,w,k,v,a] [--steer-start N] [--steer-steps N]] [--swap-text TEXT [--swap-source k|v|kv] [--swap-step N]] [--json FILE]\n"
        "\n"
        "project (default) batches all 732 captured vectors through the native\n"
        "output normalization and output matrix. local-lens perturbs one source and runs its\n"
        "local downstream closure as a diagnostic. --steer-source/--steer-target applies\n"
        "x - l2norm(l2norm(sum(source)) - l2norm(sum(target))) * max(dot(x, l2norm(sum(source))), 0) to selected components at every RWKV layer. --steer-edit selects r,w,k,v,a (default: k,v); --steer-sweep writes every subset while reusing the loaded model. Raw projections are fused into the decode graph unless --no-in-graph-unembed is set. W and A are clamped after steering. --steer-start and --steer-steps limit it to generation steps (default: all). --swap-text replaces\n"
        "the strongest selected source whose tracked token is in its top-k raw projection.\n",
        argv0);
}

static llama_token greedy_token(const std::vector<float> & logits) {
    return (llama_token) std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()));
}

static std::vector<reported_token> report_logits(
        const llama_vocab * vocab, const std::vector<float> & logits, const std::vector<llama_token> & tokens) {
    std::vector<reported_token> reports;
    reports.reserve(tokens.size());
    for (llama_token token : tokens) {
        const float logit = logits[token];
        int rank = 1;
        for (float value : logits) rank += value > logit;
        reports.push_back({ token, common_token_to_piece(vocab, token, true), rank, logit });
    }
    return reports;
}

static std::vector<float> tensor_row_f32(ggml_tensor * tensor, uint32_t row, uint32_t width) {
    if (!tensor || tensor->ne[0] != width || row >= (uint32_t) tensor->ne[1] || tensor->type != GGML_TYPE_F32) {
        throw std::runtime_error("RWKV direct lens requires F32 recurrent state tensors");
    }
    std::vector<float> values(width);
    ggml_backend_tensor_get(tensor, values.data(), row * tensor->nb[1], values.size() * sizeof(float));
    return values;
}

static rwkv_state copy_context_state(llama_context * ctx) {
    // The native prompt decode may still own the recurrent buffers on the GPU.
    ctx->synchronize();
    auto * memory = dynamic_cast<llama_memory_recurrent *>(ctx->get_memory());
    if (!memory || memory->cells.empty() || memory->cells[0].tail < 0) {
        throw std::runtime_error("prompt did not produce an RWKV recurrent state");
    }

    const auto & hparams = ctx->get_model().hparams;
    const uint32_t tail = (uint32_t) memory->cells[0].tail;
    const uint32_t row = memory->n_rs_seq == 0 ? tail : memory->rs_idx[0] * memory->size + tail;
    rwkv_state state;
    state.r.reserve(hparams.n_layer());
    state.s.reserve(hparams.n_layer());
    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        state.r.push_back(tensor_row_f32(memory->r_l[il], row, hparams.n_embd_r()));
        state.s.push_back(tensor_row_f32(memory->s_l[il], row, hparams.n_embd_s()));
    }
    return state;
}

static llm_graph_params graph_params(llama_context * ctx, llm_graph_result * result) {
    const auto & model = ctx->get_model();
    llama_ubatch ubatch = {};
    ubatch.b_equal_seqs = true;
    ubatch.n_tokens = 1;
    ubatch.n_seq_tokens = 1;
    ubatch.n_seqs = 1;
    ubatch.n_seqs_unq = 1;
    ubatch.n_pos = 1;

    llm_graph_params params = {};
    params.arch = model.arch;
    params.hparams = model.hparams;
    params.cparams = ctx->get_cparams();
    params.ubatch = ubatch;
    params.gtype = LLM_GRAPH_TYPE_DEFAULT;
    params.sched = ctx->get_sched();
    static const llama_adapter_loras no_loras;
    params.loras = &no_loras;
    params.n_outputs = 1;
    params.res = result;
    return params;
}

static projection_workspace make_projection_workspace(const llama_model & model) {
    const size_t n_columns = model.hparams.n_layer() * projection_source_count;
    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    projection_workspace workspace;
    workspace.ctx.reset(ggml_init(params));
    if (!workspace.ctx) throw std::runtime_error("failed to create projection workspace context");

    workspace.packed = ggml_new_tensor_2d(workspace.ctx.get(), GGML_TYPE_F32, model.hparams.n_embd, n_columns);
    const ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(model.output->buffer);
    workspace.buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(workspace.ctx.get(), buft));
    if (!workspace.buffer) throw std::runtime_error("failed to allocate projection workspace");
    workspace.n_columns = n_columns;

    return workspace;
}

static ggml_tensor * input_f32_1d(direct_graph & graph, size_t n) {
    ggml_tensor * tensor = ggml_new_tensor_1d(graph.ctx0, GGML_TYPE_F32, n);
    ggml_set_input(tensor);
    return tensor;
}

static ggml_tensor * input_f32_2d(direct_graph & graph, int64_t n0, int64_t n1) {
    ggml_tensor * tensor = ggml_new_tensor_2d(graph.ctx0, GGML_TYPE_F32, n0, n1);
    ggml_set_input(tensor);
    return tensor;
}

static ggml_tensor * input_f32_3d(direct_graph & graph, int64_t n0, int64_t n1, int64_t n2) {
    ggml_tensor * tensor = ggml_new_tensor_3d(graph.ctx0, GGML_TYPE_F32, n0, n1, n2);
    ggml_set_input(tensor);
    return tensor;
}

static ggml_tensor * steer_away_from_source(
        direct_graph & graph,
        ggml_tensor * values,
        ggml_tensor * source_vector,
        ggml_tensor * source_minus_target) {
    if (!source_vector) return values;
    ggml_tensor * alignment = ggml_sum_rows(graph.ctx0, ggml_mul(graph.ctx0, values, source_vector));
    alignment = ggml_clamp(graph.ctx0, alignment, 0.0f, std::numeric_limits<float>::max());
    return ggml_sub(graph.ctx0, values,
        ggml_mul(graph.ctx0, source_minus_target, ggml_repeat(graph.ctx0, alignment, values)));
}

struct raw_tensors {
    ggml_tensor * r;
    ggml_tensor * w;
    ggml_tensor * k;
    ggml_tensor * v;
    ggml_tensor * a;
    ggml_tensor * a_pre;
    ggml_tensor * g;
    ggml_tensor * k0;
    ggml_tensor * kk;
    ggml_tensor * wkv;
    ggml_tensor * rkv;
};

struct time_result {
    ggml_tensor * output;
    ggml_tensor * next_state;
    raw_tensors raw;
};

static time_result build_time_mix(
        direct_graph & graph,
        const llama_model & model,
        ggml_tensor * cur,
        ggml_tensor * x_prev,
        ggml_tensor *& first_layer_value,
        ggml_tensor * prior_state,
        int il,
        ggml_tensor * k_override,
        ggml_tensor * kk_override,
        ggml_tensor * v_override,
        uint32_t steer_components,
        ggml_tensor * source_vector,
        ggml_tensor * source_minus_target) {
    const auto & hparams = model.hparams;
    const auto & layer = model.layers[il];
    const int64_t n_embd = hparams.n_embd;
    const int64_t head_size = hparams.wkv_head_size;
    const int64_t head_count = n_embd / head_size;
    const bool has_gating = layer.time_mix_g1 && layer.time_mix_g2;

    ggml_tensor * sx = ggml_sub(graph.ctx0, x_prev, cur);
    ggml_tensor * dummy = ggml_new_tensor_4d(graph.ctx0, GGML_TYPE_F32, n_embd, 1, 1, has_gating ? 6 : 5);
    sx = ggml_repeat(graph.ctx0, sx, dummy);
    ggml_tensor * xxx = ggml_add(graph.ctx0, ggml_mul(graph.ctx0, sx, layer.time_mix_lerp_fused), cur);

    ggml_tensor * xr = ggml_view_2d(graph.ctx0, xxx, n_embd, 1, xxx->nb[1], 0);
    ggml_tensor * xw = ggml_view_2d(graph.ctx0, xxx, n_embd, 1, xxx->nb[1], n_embd * sizeof(float));
    ggml_tensor * xk = ggml_view_2d(graph.ctx0, xxx, n_embd, 1, xxx->nb[1], n_embd * 2 * sizeof(float));
    ggml_tensor * xv = ggml_view_2d(graph.ctx0, xxx, n_embd, 1, xxx->nb[1], n_embd * 3 * sizeof(float));
    ggml_tensor * xa = ggml_view_2d(graph.ctx0, xxx, n_embd, 1, xxx->nb[1], n_embd * 4 * sizeof(float));
    ggml_tensor * xg = has_gating ? ggml_view_2d(graph.ctx0, xxx, n_embd, 1, xxx->nb[1], n_embd * 5 * sizeof(float)) : nullptr;

    ggml_tensor * r = ggml_mul_mat(graph.ctx0, layer.time_mix_receptance, xr);
    if (steer_components & steer_r) r = steer_away_from_source(graph, r, source_vector, source_minus_target);
    ggml_tensor * w = ggml_add(graph.ctx0,
        ggml_mul_mat(graph.ctx0, layer.time_mix_w2, ggml_tanh(graph.ctx0, ggml_mul_mat(graph.ctx0, layer.time_mix_w1, xw))),
        layer.time_mix_w0);
    w = ggml_exp(graph.ctx0, ggml_scale(graph.ctx0, ggml_sigmoid(graph.ctx0, w), -0.606531));
    if (steer_components & steer_w) {
        w = steer_away_from_source(graph, w, source_vector, source_minus_target);
        w = ggml_clamp(graph.ctx0, w, std::exp(-0.606531f), 1.0f);
    }

    ggml_tensor * k0 = ggml_mul_mat(graph.ctx0, layer.time_mix_key, xk);
    ggml_tensor * k = steer_away_from_source(graph, k0, (steer_components & steer_k) ? source_vector : nullptr, source_minus_target);
    ggml_tensor * v = ggml_mul_mat(graph.ctx0, layer.time_mix_value, xv);
    if (first_layer_value == nullptr) {
        v = steer_away_from_source(graph, v, (steer_components & steer_v) ? source_vector : nullptr, source_minus_target);
        first_layer_value = v;
    } else {
        v = ggml_add(graph.ctx0, v,
            ggml_mul(graph.ctx0, ggml_sub(graph.ctx0, first_layer_value, v),
                ggml_sigmoid(graph.ctx0, ggml_add(graph.ctx0,
                    ggml_mul_mat(graph.ctx0, layer.time_mix_v2, ggml_mul_mat(graph.ctx0, layer.time_mix_v1, xv)),
                    layer.time_mix_v0))));
        v = steer_away_from_source(graph, v, (steer_components & steer_v) ? source_vector : nullptr, source_minus_target);
    }
    ggml_tensor * g = has_gating ? ggml_mul_mat(graph.ctx0, layer.time_mix_g2,
        ggml_sigmoid(graph.ctx0, ggml_mul_mat(graph.ctx0, layer.time_mix_g1, xg))) : nullptr;
    ggml_tensor * a_pre = ggml_add(graph.ctx0,
        ggml_mul_mat(graph.ctx0, layer.time_mix_a2, ggml_mul_mat(graph.ctx0, layer.time_mix_a1, xa)), layer.time_mix_a0);
    ggml_tensor * a = ggml_sigmoid(graph.ctx0, a_pre);
    if (steer_components & steer_a) {
        a = steer_away_from_source(graph, a, source_vector, source_minus_target);
        a = ggml_clamp(graph.ctx0, a, 0.0f, 1.0f);
    }

    ggml_tensor * kk = ggml_reshape_3d(graph.ctx0, ggml_mul(graph.ctx0, k, layer.time_mix_k_k), head_size, head_count, 1);
    kk = ggml_l2_norm(graph.ctx0, kk, 1e-12);
    ggml_tensor * ka = ggml_mul(graph.ctx0, k, layer.time_mix_k_a);
    k = ggml_add(graph.ctx0, k, ggml_sub(graph.ctx0, ggml_mul(graph.ctx0, a, ka), ka));
    if (k_override) k = k_override;
    if (kk_override) kk = ggml_reshape_3d(graph.ctx0, kk_override, head_size, head_count, 1);
    if (v_override) v = v_override;

    r = ggml_reshape_3d(graph.ctx0, r, head_size, head_count, 1);
    w = ggml_reshape_3d(graph.ctx0, w, head_size, head_count, 1);
    k = ggml_reshape_3d(graph.ctx0, k, head_size, head_count, 1);
    v = ggml_reshape_3d(graph.ctx0, v, head_size, head_count, 1);
    a = ggml_reshape_3d(graph.ctx0, a, head_size, head_count, 1);

    ggml_tensor * wkv = ggml_rwkv_wkv7(graph.ctx0, r, w, k, v, ggml_neg(graph.ctx0, kk), ggml_mul(graph.ctx0, kk, a), prior_state);
    ggml_tensor * next_state = ggml_view_2d(graph.ctx0, wkv, hparams.n_embd_s(), 1, hparams.n_embd_s() * sizeof(float), n_embd * sizeof(float));
    ggml_tensor * wkv_value = ggml_view_2d(graph.ctx0, wkv, n_embd, 1, n_embd * sizeof(float), 0);
    cur = wkv_value;
    cur = ggml_reshape_3d(graph.ctx0, cur, n_embd / head_count, head_count, 1);
    cur = ggml_norm(graph.ctx0, cur, 64e-5f);
    cur = ggml_reshape_2d(graph.ctx0, cur, n_embd, 1);
    cur = ggml_add(graph.ctx0, ggml_mul(graph.ctx0, cur, layer.time_mix_ln), layer.time_mix_ln_b);
    ggml_tensor * rk = ggml_sum_rows(graph.ctx0, ggml_mul(graph.ctx0,
        ggml_mul(graph.ctx0, k, r), ggml_reshape_2d(graph.ctx0, layer.time_mix_r_k, head_size, head_count)));
    cur = ggml_add(graph.ctx0, cur, ggml_reshape_2d(graph.ctx0, ggml_mul(graph.ctx0, v, rk), n_embd, 1));
    ggml_tensor * rkv = cur;
    if (g) cur = ggml_mul(graph.ctx0, cur, g);
    cur = ggml_mul_mat(graph.ctx0, layer.time_mix_output, cur);
    return {
        ggml_reshape_3d(graph.ctx0, cur, n_embd, 1, 1),
        next_state,
        { r, w, k, v, a, a_pre, g, k0, kk, wkv_value, rkv },
    };
}

static void add_output(direct_graph & graph, ggml_tensor * tensor) {
    ggml_set_output(tensor);
    ggml_build_forward_expand(graph.gf, tensor);
}

static ggml_tensor * materialize_output(direct_graph & graph, ggml_tensor * tensor) {
    // Output views do not own storage, so retain an independent value for readback.
    ggml_tensor * result = ggml_cont(graph.ctx0, tensor);
    add_output(graph, result);
    return result;
}

static snapshot run_snapshot(
        llama_context * ctx,
        llama_token token,
        const rwkv_state & state,
        const std::vector<activation_patch> * patches = nullptr,
        const std::vector<llama_token> * steer_sources = nullptr,
        const std::vector<llama_token> * steer_targets = nullptr,
        uint32_t steer_components = steer_default_components,
        const std::vector<source_kind> * in_graph_project_sources = nullptr,
        projection_workspace * workspace = nullptr) {
    const auto & model = ctx->get_model();
    const auto & hparams = model.hparams;
    const int n_layer = hparams.n_layer();
    if ((int) state.r.size() != n_layer || (int) state.s.size() != n_layer) {
        throw std::runtime_error("invalid RWKV state for snapshot");
    }
    const bool steering_enabled = steer_sources && !steer_sources->empty() && steer_targets && !steer_targets->empty();
    if ((steer_sources && !steer_sources->empty()) != (steer_targets && !steer_targets->empty())) {
        throw std::runtime_error("semantic steering needs both token vectors");
    }
    if (in_graph_project_sources && !workspace) {
        throw std::runtime_error("in-graph projection needs a persistent workspace");
    }
    // Research traces retain raw activations even when their projection is fused into decode.
    const bool capture_raw = true;

    ctx->synchronize();
    ggml_backend_sched_reset(ctx->get_sched());
    llm_graph_result result(ctx->graph_max_nodes(1));
    direct_graph graph(model, graph_params(ctx, &result));
    ggml_tensor * token_input = ggml_new_tensor_1d(graph.ctx0, GGML_TYPE_I32, 1);
    ggml_set_input(token_input);
    ggml_tensor * steer_tokens = nullptr;
    ggml_tensor * source_vector = nullptr;
    ggml_tensor * source_minus_targets = nullptr;
    if (steering_enabled) {
        steer_tokens = ggml_new_tensor_1d(graph.ctx0, GGML_TYPE_I32, steer_sources->size() + steer_targets->size());
        ggml_set_input(steer_tokens);
        ggml_tensor * steer_vectors = ggml_get_rows(graph.ctx0, model.output, steer_tokens);
        for (size_t i = 0; i < steer_sources->size(); ++i) {
            ggml_tensor * source = ggml_view_2d(graph.ctx0, steer_vectors, hparams.n_embd, 1,
                steer_vectors->nb[1], i * steer_vectors->nb[1]);
            source_vector = source_vector ? ggml_add(graph.ctx0, source_vector, source) : source;
        }
        source_vector = ggml_l2_norm(graph.ctx0, source_vector, 1e-12f);
        ggml_tensor * target_sum = nullptr;
        for (size_t i = 0; i < steer_targets->size(); ++i) {
            ggml_tensor * target = ggml_view_2d(graph.ctx0, steer_vectors, hparams.n_embd, 1,
                steer_vectors->nb[1], (steer_sources->size() + i) * steer_vectors->nb[1]);
            target_sum = target_sum ? ggml_add(graph.ctx0, target_sum, target) : target;
        }
        target_sum = ggml_l2_norm(graph.ctx0, target_sum, 1e-12f);
        source_minus_targets = ggml_sub(graph.ctx0, source_vector, target_sum);
        source_minus_targets = ggml_l2_norm(graph.ctx0, source_minus_targets, 1e-12f);
    }
    ggml_tensor * cur = ggml_get_rows(graph.ctx0, model.tok_embd, token_input);
    cur = graph.build_norm(cur, model.tok_norm, model.tok_norm_b, LLM_NORM, 0);
    cur = ggml_reshape_3d(graph.ctx0, cur, hparams.n_embd, 1, 1);

    snapshot out;
    out.next_state.r.resize(n_layer);
    out.next_state.s.resize(n_layer);
    if (capture_raw) {
        out.resid_in.resize(n_layer);
        out.raw.resize(n_layer);
    }
    std::vector<ggml_tensor *> r_inputs(n_layer);
    std::vector<ggml_tensor *> s_inputs(n_layer);
    std::vector<ggml_tensor *> resid_outputs(n_layer);
    std::vector<raw_tensors> raw_outputs(n_layer);
    std::vector<ggml_tensor *> state_outputs(n_layer);
    std::vector<ggml_tensor *> shift_outputs(n_layer);
    std::vector<ggml_tensor *> channel_outputs(n_layer);
    std::vector<ggml_tensor *> in_graph_projection_vectors;
    if (in_graph_project_sources) {
        in_graph_projection_vectors.reserve(n_layer * in_graph_project_sources->size());
    }
    std::vector<const activation_patch *> patches_by_layer(n_layer);
    if (patches) {
        for (const activation_patch & patch : *patches) {
            const bool has_k = !patch.k.empty() || !patch.kk.empty();
            if (patch.layer < 0 || patch.layer >= n_layer || patches_by_layer[patch.layer] ||
                (has_k && (patch.k.size() != (size_t) hparams.n_embd || patch.kk.size() != (size_t) hparams.n_embd)) ||
                (!patch.v.empty() && patch.v.size() != (size_t) hparams.n_embd) || (!has_k && patch.v.empty())) {
                throw std::runtime_error("invalid activation patch");
            }
            patches_by_layer[patch.layer] = &patch;
        }
    }
    for (int il = 0; il < n_layer; ++il) {
        const activation_patch * patch = patches_by_layer[il];
        const int n_override_values = patch ? ((!patch->k.empty() ? 2 : 0) + (!patch->v.empty() ? 1 : 0)) : 0;
        const int n_shift_values = hparams.token_shift_count + n_override_values;
        r_inputs[il] = input_f32_3d(graph, hparams.n_embd, n_shift_values, 1);
        s_inputs[il] = input_f32_2d(graph, hparams.n_embd_s(), 1);
    }

    ggml_tensor * first_layer_value = nullptr;
    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];
        if (capture_raw) {
            out.resid_in[il].resize(hparams.n_embd);
            resid_outputs[il] = materialize_output(graph, cur);
        }

        ggml_tensor * att_prev = ggml_view_3d(graph.ctx0, r_inputs[il], hparams.n_embd, 1, 1,
            r_inputs[il]->nb[1], r_inputs[il]->nb[2], 0);
        ggml_tensor * ffn_prev = ggml_view_3d(graph.ctx0, r_inputs[il], hparams.n_embd, 1, 1,
            r_inputs[il]->nb[1], r_inputs[il]->nb[2], hparams.n_embd * sizeof(float));
        ggml_tensor * att_norm = graph.build_norm(cur, layer.attn_norm, layer.attn_norm_b, LLM_NORM, il);
        ggml_tensor * k_override = nullptr;
        ggml_tensor * kk_override = nullptr;
        ggml_tensor * v_override = nullptr;
        if (patches_by_layer[il]) {
            size_t override_offset = hparams.token_shift_count * hparams.n_embd * sizeof(float);
            if (!patches_by_layer[il]->k.empty()) {
                k_override = ggml_view_2d(graph.ctx0, r_inputs[il], hparams.n_embd, 1, r_inputs[il]->nb[1], override_offset);
                override_offset += hparams.n_embd * sizeof(float);
                kk_override = ggml_view_2d(graph.ctx0, r_inputs[il], hparams.n_embd, 1, r_inputs[il]->nb[1], override_offset);
                override_offset += hparams.n_embd * sizeof(float);
            }
            if (!patches_by_layer[il]->v.empty()) {
                v_override = ggml_view_2d(graph.ctx0, r_inputs[il], hparams.n_embd, 1, r_inputs[il]->nb[1], override_offset);
            }
        }
        time_result time = build_time_mix(graph, model, att_norm, att_prev, first_layer_value, s_inputs[il], il,
            k_override, kk_override, v_override, steer_components, source_vector, source_minus_targets);
        out.next_state.s[il].resize(hparams.n_embd_s());
        add_output(graph, time.output);
        state_outputs[il] = materialize_output(graph, time.next_state);
        if (capture_raw) {
            raw_outputs[il] = {
                materialize_output(graph, time.raw.r),
                materialize_output(graph, time.raw.w),
                materialize_output(graph, time.raw.k),
                materialize_output(graph, time.raw.v),
                materialize_output(graph, time.raw.a),
                materialize_output(graph, time.raw.a_pre),
                materialize_output(graph, time.raw.g),
                materialize_output(graph, time.raw.k0),
                materialize_output(graph, time.raw.kk),
                materialize_output(graph, time.raw.wkv),
                materialize_output(graph, time.raw.rkv),
            };
            auto & raw = out.raw[il];
            for (std::vector<float> * values : { &raw.r, &raw.w, &raw.k, &raw.v, &raw.a,
                                                   &raw.a_pre, &raw.g, &raw.k0, &raw.kk,
                                                   &raw.wkv, &raw.rkv }) {
                values->resize(hparams.n_embd);
            }
            raw.channel_out.resize(hparams.n_embd);
        }

        ggml_tensor * ffn_inp = ggml_add(graph.ctx0, time.output, cur);
        ggml_tensor * ffn_norm = graph.build_norm(ffn_inp, layer.attn_norm_2, layer.attn_norm_2_b, LLM_NORM, il);
        ggml_tensor * channel = graph.build_rwkv7_channel_mix(&layer, ffn_norm, ffn_prev, LLM_ARCH_RWKV7);
        if (capture_raw) {
            channel_outputs[il] = materialize_output(graph, channel);
        }
        if (in_graph_project_sources) {
            const raw_tensors & raw = raw_outputs[il];
            for (source_kind kind : *in_graph_project_sources) {
                ggml_tensor * vector = nullptr;
                switch (kind) {
                    case source_kind::r: vector = raw.r; break;
                    case source_kind::w: vector = raw.w; break;
                    case source_kind::k: vector = raw.k; break;
                    case source_kind::v: vector = raw.v; break;
                    case source_kind::a: vector = raw.a; break;
                    case source_kind::a_pre: vector = raw.a_pre; break;
                    case source_kind::g: vector = raw.g; break;
                    case source_kind::k0: vector = raw.k0; break;
                    case source_kind::kk: vector = raw.kk; break;
                    case source_kind::wkv: vector = raw.wkv; break;
                    case source_kind::rkv: vector = raw.rkv; break;
                    case source_kind::channel_out: vector = channel_outputs[il]; break;
                }
                if (!vector) throw std::runtime_error("in-graph projection source is unavailable");
                in_graph_projection_vectors.push_back(vector);
            }
        }
        cur = ggml_add(graph.ctx0, channel, ffn_inp);
        out.next_state.r[il].resize(hparams.n_embd_r());
        ggml_tensor * next_shift = ggml_concat(graph.ctx0, att_norm, ffn_norm, 1);
        add_output(graph, next_shift);
        shift_outputs[il] = next_shift;
    }
    cur = ggml_reshape_2d(graph.ctx0, cur, hparams.n_embd, 1);
    cur = graph.build_norm(cur, model.output_norm, model.output_norm_b, LLM_NORM, -1);
    ggml_tensor * logits = ggml_mul_mat(graph.ctx0, model.output, cur);
    if (model.output_s) {
        logits = ggml_mul(graph.ctx0, logits, model.output_s);
    }
    add_output(graph, logits);
    ggml_tensor * projected_logits = nullptr;
    if (in_graph_project_sources) {
        if (in_graph_projection_vectors.size() > workspace->n_columns) {
            throw std::runtime_error("projection workspace is too small");
        }
        ggml_tensor * vectors = ggml_view_2d(graph.ctx0, workspace->packed, hparams.n_embd,
            in_graph_projection_vectors.size(), workspace->packed->nb[1], 0);
        for (size_t column = 0; column < in_graph_projection_vectors.size(); ++column) {
            ggml_tensor * destination = ggml_view_2d(graph.ctx0, workspace->packed, hparams.n_embd, 1,
                workspace->packed->nb[1], column * workspace->packed->nb[1]);
            ggml_tensor * source = ggml_reshape_2d(graph.ctx0, in_graph_projection_vectors[column], hparams.n_embd, 1);
            // KV-cache-style writes must be appended before their persistent-buffer consumer.
            ggml_tensor * copied = ggml_cpy(graph.ctx0, source, destination);
            add_output(graph, copied);
        }
        ggml_tensor * normed = graph.build_norm(vectors, model.output_norm, model.output_norm_b, LLM_NORM, -1);
        projected_logits = ggml_mul_mat(graph.ctx0, model.output, normed);
        if (model.output_s) projected_logits = ggml_mul(graph.ctx0, projected_logits, model.output_s);
        add_output(graph, projected_logits);
    }

    if (!ggml_backend_sched_alloc_graph(ctx->get_sched(), graph.gf)) {
        throw std::runtime_error("failed to allocate GGML snapshot graph");
    }
    ggml_backend_tensor_set(token_input, &token, 0, sizeof(token));
    if (steer_tokens) {
        std::vector<llama_token> steer_token_ids = *steer_sources;
        steer_token_ids.insert(steer_token_ids.end(), steer_targets->begin(), steer_targets->end());
        ggml_backend_tensor_set(steer_tokens, steer_token_ids.data(), 0, steer_token_ids.size() * sizeof(llama_token));
    }
    for (int il = 0; il < n_layer; ++il) {
        ggml_backend_tensor_set(r_inputs[il], state.r[il].data(), 0, state.r[il].size() * sizeof(float));
        ggml_backend_tensor_set(s_inputs[il], state.s[il].data(), 0, state.s[il].size() * sizeof(float));
        if (patches_by_layer[il]) {
            const activation_patch & patch = *patches_by_layer[il];
            size_t override_offset = hparams.token_shift_count * hparams.n_embd * sizeof(float);
            if (!patch.k.empty()) {
                ggml_backend_tensor_set(r_inputs[il], patch.k.data(), override_offset, patch.k.size() * sizeof(float));
                override_offset += hparams.n_embd * sizeof(float);
                ggml_backend_tensor_set(r_inputs[il], patch.kk.data(), override_offset, patch.kk.size() * sizeof(float));
                override_offset += hparams.n_embd * sizeof(float);
            }
            if (!patch.v.empty()) {
                ggml_backend_tensor_set(r_inputs[il], patch.v.data(), override_offset, patch.v.size() * sizeof(float));
            }
        }
    }
    if (ctx->graph_compute(graph.gf, false) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("GGML snapshot graph failed");
    }
    ctx->synchronize();

    for (int il = 0; il < n_layer; ++il) {
        if (capture_raw) ggml_backend_tensor_get(resid_outputs[il], out.resid_in[il].data(), 0, hparams.n_embd * sizeof(float));
        ggml_backend_tensor_get(state_outputs[il], out.next_state.s[il].data(), 0, hparams.n_embd_s() * sizeof(float));
        ggml_backend_tensor_get(shift_outputs[il], out.next_state.r[il].data(), 0, hparams.n_embd_r() * sizeof(float));
        if (!capture_raw) continue;
        const auto & tensors = raw_outputs[il];
        auto & raw = out.raw[il];
        const std::vector<ggml_tensor *> sources = { tensors.r, tensors.w, tensors.k, tensors.v, tensors.a,
                                                       tensors.a_pre, tensors.g, tensors.k0, tensors.kk,
                                                       tensors.wkv, tensors.rkv };
        const std::vector<std::vector<float> *> values = { &raw.r, &raw.w, &raw.k, &raw.v, &raw.a,
                                                             &raw.a_pre, &raw.g, &raw.k0, &raw.kk,
                                                             &raw.wkv, &raw.rkv };
        for (size_t source = 0; source < sources.size(); ++source) {
            ggml_backend_tensor_get(sources[source], values[source]->data(), 0, hparams.n_embd * sizeof(float));
        }
        ggml_backend_tensor_get(channel_outputs[il], raw.channel_out.data(), 0, hparams.n_embd * sizeof(float));
    }
    out.logits.resize(llama_vocab_n_tokens(&model.vocab));
    ggml_backend_tensor_get(logits, out.logits.data(), 0, out.logits.size() * sizeof(float));
    if (projected_logits) {
        out.projected_logits.resize(llama_vocab_n_tokens(&model.vocab) * in_graph_projection_vectors.size());
        ggml_backend_tensor_get(projected_logits, out.projected_logits.data(), 0, out.projected_logits.size() * sizeof(float));
    }
    return out;
}

static std::vector<scored_token> lens_readout(
        llama_context * ctx,
        int layer_id,
        const std::vector<float> & resid,
        const snapshot::raw_layer & clean,
        const std::vector<float> & prior_shift,
        const std::vector<float> & prior_state,
        source_kind source,
        float epsilon,
        int top) {
    const auto & model = ctx->get_model();
    const auto & hparams = model.hparams;
    const auto & layer = model.layers[layer_id];
    snapshot::raw_layer raw = clean;
    std::vector<float> * perturbed = nullptr;
    switch (source) {
        case source_kind::r:   perturbed = &raw.r; break;
        case source_kind::w:   perturbed = &raw.w; break;
        case source_kind::k:   perturbed = &raw.k; break;
        case source_kind::v:   perturbed = &raw.v; break;
        case source_kind::a:   perturbed = &raw.a; break;
        case source_kind::a_pre:
        case source_kind::k0:
        case source_kind::kk:
        case source_kind::channel_out:
            throw std::runtime_error("local-lens does not support derived source " + std::string(source_name(source)));
        case source_kind::g:   perturbed = &raw.g; break;
        case source_kind::wkv: perturbed = &raw.wkv; break;
        case source_kind::rkv: perturbed = &raw.rkv; break;
    }
    const float scale = epsilon / std::sqrt((float) perturbed->size());
    for (size_t i = 0; i < perturbed->size(); ++i) (*perturbed)[i] += (i & 1) ? scale : -scale;

    ctx->synchronize();
    ggml_backend_sched_reset(ctx->get_sched());
    llm_graph_result result(ctx->graph_max_nodes(1));
    direct_graph graph(model, graph_params(ctx, &result));
    ggml_tensor * resid_input = input_f32_3d(graph, hparams.n_embd, 1, 1);
    ggml_tensor * shift_input = input_f32_1d(graph, hparams.n_embd);
    ggml_tensor * state_input = input_f32_2d(graph, hparams.n_embd_s(), 1);
    ggml_tensor * r_input = input_f32_3d(graph, hparams.wkv_head_size, hparams.n_embd / hparams.wkv_head_size, 1);
    ggml_tensor * w_input = input_f32_3d(graph, hparams.wkv_head_size, hparams.n_embd / hparams.wkv_head_size, 1);
    ggml_tensor * k_input = input_f32_3d(graph, hparams.wkv_head_size, hparams.n_embd / hparams.wkv_head_size, 1);
    ggml_tensor * v_input = input_f32_3d(graph, hparams.wkv_head_size, hparams.n_embd / hparams.wkv_head_size, 1);
    ggml_tensor * a_input = input_f32_3d(graph, hparams.wkv_head_size, hparams.n_embd / hparams.wkv_head_size, 1);
    ggml_tensor * g_input = input_f32_2d(graph, hparams.n_embd, 1);
    ggml_tensor * k0_input = input_f32_2d(graph, hparams.n_embd, 1);
    ggml_tensor * kk_input = input_f32_3d(graph, hparams.wkv_head_size, hparams.n_embd / hparams.wkv_head_size, 1);
    ggml_tensor * wkv_input = input_f32_2d(graph, hparams.n_embd, 1);
    ggml_tensor * rkv_input = input_f32_2d(graph, hparams.n_embd, 1);

    ggml_tensor * k = k_input;
    if (source == source_kind::a) {
        ggml_tensor * k0 = k0_input;
        ggml_tensor * ka = ggml_mul(graph.ctx0, k0, layer.time_mix_k_a);
        k = ggml_add(graph.ctx0, k0, ggml_sub(graph.ctx0, ggml_mul(graph.ctx0, a_input, ka), ka));
        k = ggml_reshape_3d(graph.ctx0, k, hparams.wkv_head_size, hparams.n_embd / hparams.wkv_head_size, 1);
    }

    ggml_tensor * time = nullptr;
    if (source == source_kind::r || source == source_kind::w || source == source_kind::k ||
        source == source_kind::v || source == source_kind::a) {
        ggml_tensor * wkv = ggml_rwkv_wkv7(graph.ctx0, r_input, w_input, k, v_input,
            ggml_neg(graph.ctx0, kk_input), ggml_mul(graph.ctx0, kk_input, a_input), state_input);
        time = ggml_view_2d(graph.ctx0, wkv, hparams.n_embd, 1, hparams.n_embd * sizeof(float), 0);
    } else {
        time = wkv_input;
    }

    ggml_tensor * rkv = nullptr;
    if (source == source_kind::rkv) {
        rkv = rkv_input;
    } else {
        const int64_t head_count = hparams.n_embd / hparams.wkv_head_size;
        time = ggml_reshape_3d(graph.ctx0, time, hparams.n_embd / head_count, head_count, 1);
        time = ggml_norm(graph.ctx0, time, 64e-5f);
        time = ggml_reshape_2d(graph.ctx0, time, hparams.n_embd, 1);
        time = ggml_add(graph.ctx0, ggml_mul(graph.ctx0, time, layer.time_mix_ln), layer.time_mix_ln_b);
        ggml_tensor * rk = ggml_sum_rows(graph.ctx0, ggml_mul(graph.ctx0,
            ggml_mul(graph.ctx0, k, r_input), ggml_reshape_2d(graph.ctx0, layer.time_mix_r_k, hparams.wkv_head_size, head_count)));
        rkv = ggml_add(graph.ctx0, time, ggml_reshape_2d(graph.ctx0, ggml_mul(graph.ctx0, v_input, rk), hparams.n_embd, 1));
    }
    time = ggml_mul(graph.ctx0, rkv, g_input);
    time = ggml_mul_mat(graph.ctx0, layer.time_mix_output, time);
    time = ggml_reshape_3d(graph.ctx0, time, hparams.n_embd, 1, 1);

    ggml_tensor * ffn_inp = ggml_add(graph.ctx0, resid_input, time);
    ggml_tensor * ffn_norm = graph.build_norm(ffn_inp, layer.attn_norm_2, layer.attn_norm_2_b, LLM_NORM, layer_id);
    ggml_tensor * ffn_prev = ggml_reshape_3d(graph.ctx0, shift_input, hparams.n_embd, 1, 1);
    ggml_tensor * channel = graph.build_rwkv7_channel_mix(&layer, ffn_norm, ffn_prev, LLM_ARCH_RWKV7);
    channel = ggml_reshape_2d(graph.ctx0, channel, hparams.n_embd, 1);
    ggml_tensor * lens = graph.build_norm(channel, model.output_norm, model.output_norm_b, LLM_NORM, -1);
    ggml_tensor * logits = ggml_mul_mat(graph.ctx0, model.output, lens);
    if (model.output_s) {
        logits = ggml_mul(graph.ctx0, logits, model.output_s);
    }
    add_output(graph, logits);

    if (!ggml_backend_sched_alloc_graph(ctx->get_sched(), graph.gf)) {
        throw std::runtime_error("failed to allocate GGML lens graph");
    }
    const auto set_input = [](ggml_tensor * tensor, const float * values, size_t count) {
        if (tensor->buffer) {
            ggml_backend_tensor_set(tensor, values, 0, count * sizeof(float));
        }
    };
    set_input(resid_input, resid.data(), resid.size());
    set_input(shift_input, prior_shift.data() + hparams.n_embd, hparams.n_embd);
    set_input(state_input, prior_state.data(), prior_state.size());
    set_input(r_input, raw.r.data(), raw.r.size());
    set_input(w_input, raw.w.data(), raw.w.size());
    set_input(k_input, raw.k.data(), raw.k.size());
    set_input(v_input, raw.v.data(), raw.v.size());
    set_input(a_input, raw.a.data(), raw.a.size());
    set_input(g_input, raw.g.data(), raw.g.size());
    set_input(k0_input, raw.k0.data(), raw.k0.size());
    set_input(kk_input, raw.kk.data(), raw.kk.size());
    set_input(wkv_input, raw.wkv.data(), raw.wkv.size());
    set_input(rkv_input, raw.rkv.data(), raw.rkv.size());
    if (ctx->graph_compute(graph.gf, false) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("GGML lens graph failed");
    }
    ctx->synchronize();

    std::vector<float> values(llama_vocab_n_tokens(&model.vocab));
    ggml_backend_tensor_get(logits, values.data(), 0, values.size() * sizeof(float));
    std::vector<int> ids(values.size());
    for (int i = 0; i < (int) ids.size(); ++i) ids[i] = i;
    top = std::min(top, (int) ids.size());
    std::partial_sort(ids.begin(), ids.begin() + top, ids.end(), [&values](int a, int b) { return values[a] > values[b]; });
    std::vector<scored_token> result_tokens;
    result_tokens.reserve(top);
    for (int i = 0; i < top; ++i) {
        result_tokens.push_back({ ids[i], common_token_to_piece(&model.vocab, ids[i], true), values[ids[i]] });
    }
    return result_tokens;
}

static std::string raw_source_name(int layer, source_kind source) {
    if (source == source_kind::channel_out) {
        return "rwkv.layer." + std::to_string(layer) + ".channel.out";
    }
    return "rwkv.layer." + std::to_string(layer) + ".time." + source_name(source);
}

static const std::vector<float> & raw_values(const snapshot::raw_layer & raw, source_kind source) {
    switch (source) {
        case source_kind::r:   return raw.r;
        case source_kind::w:   return raw.w;
        case source_kind::k:   return raw.k;
        case source_kind::v:   return raw.v;
        case source_kind::a:   return raw.a;
        case source_kind::a_pre: return raw.a_pre;
        case source_kind::g:   return raw.g;
        case source_kind::k0:  return raw.k0;
        case source_kind::kk:  return raw.kk;
        case source_kind::wkv: return raw.wkv;
        case source_kind::rkv: return raw.rkv;
        case source_kind::channel_out: return raw.channel_out;
    }
    GGML_ABORT("unknown source");
}

static float cosine_similarity(const std::vector<float> & left, const std::vector<float> & right) {
    double dot = 0.0;
    double left_norm = 0.0;
    double right_norm = 0.0;
    for (size_t i = 0; i < left.size(); ++i) {
        dot += (double) left[i] * right[i];
        left_norm += (double) left[i] * left[i];
        right_norm += (double) right[i] * right[i];
    }
    return (float) (dot / std::sqrt(std::max(left_norm * right_norm, 1e-30)));
}

static std::vector<source_readout> summarize_projected_logits(
        const llama_model & model, const std::vector<float> & projected,
        const std::vector<source_kind> & kinds, llama_token target,
        const std::vector<llama_token> & report_tokens, int top) {
    const size_t n_vocab = llama_vocab_n_tokens(&model.vocab);
    const size_t n_vectors = model.hparams.n_layer() * kinds.size();
    if (projected.size() != n_vocab * n_vectors) throw std::runtime_error("invalid in-graph projection size");
    top = std::min(top, (int) n_vocab);
    std::vector<source_readout> readouts;
    readouts.reserve(n_vectors);
    for (size_t vector = 0; vector < n_vectors; ++vector) {
        const float * column_values = projected.data() + vector * n_vocab;
        std::vector<int> ids(n_vocab);
        for (int id = 0; id < (int) n_vocab; ++id) ids[id] = id;
        std::partial_sort(ids.begin(), ids.begin() + top, ids.end(), [column_values](int a, int b) {
            return column_values[a] > column_values[b];
        });
        const size_t layer = vector / kinds.size();
        const source_kind kind = kinds[vector % kinds.size()];
        int target_rank = -1;
        float target_logit = 0.0f;
        if (target >= 0 && (size_t) target < n_vocab) {
            target_logit = column_values[target];
            target_rank = 1;
            for (size_t id = 0; id < n_vocab; ++id) target_rank += column_values[id] > target_logit;
        }
        source_readout readout = { raw_source_name((int) layer, kind), {}, {}, 1.0f, target_rank, target_logit, -1, 0.0f };
        readout.reported_tokens.reserve(report_tokens.size());
        for (llama_token report_token : report_tokens) {
            const float report_logit = column_values[report_token];
            int report_rank = 1;
            for (size_t id = 0; id < n_vocab; ++id) report_rank += column_values[id] > report_logit;
            readout.reported_tokens.push_back({ report_token, common_token_to_piece(&model.vocab, report_token, true), report_rank, report_logit });
        }
        readout.top_tokens.reserve(top);
        for (int rank = 0; rank < top; ++rank) {
            const int id = ids[rank];
            readout.top_tokens.push_back({ id, common_token_to_piece(&model.vocab, id, true), column_values[id] });
        }
        readouts.push_back(std::move(readout));
    }
    return readouts;
}

static std::vector<source_readout> project_raw_vectors(
        llama_context * ctx, const std::vector<snapshot::raw_layer> & raw,
        const std::vector<source_kind> & kinds, const std::vector<snapshot::raw_layer> * baseline,
        llama_token target, const std::vector<llama_token> & report_tokens, bool random_baseline, int top) {
    const auto & model = ctx->get_model();
    const auto & hparams = model.hparams;
    const size_t n_vectors = raw.size() * kinds.size();
    const size_t n_columns = random_baseline ? n_vectors * 2 : n_vectors;
    std::vector<float> values(hparams.n_embd * n_columns);
    std::vector<std::string> sources;
    sources.reserve(n_vectors);
    size_t column = 0;
    for (int layer = 0; layer < (int) raw.size(); ++layer) {
        for (source_kind kind : kinds) {
            const auto & vector = raw_values(raw[layer], kind);
            std::copy(vector.begin(), vector.end(), values.begin() + column * hparams.n_embd);
            sources.push_back(raw_source_name(layer, kind));
            ++column;
        }
    }
    if (random_baseline) {
        uint32_t state = 0x6d2b79f5;
        for (size_t i = hparams.n_embd * n_vectors; i < values.size(); ++i) {
            state = state * 1664525u + 1013904223u;
            values[i] = (state >> 31) ? 1.0f : -1.0f;
        }
    }

    ctx->synchronize();
    ggml_backend_sched_reset(ctx->get_sched());
    llm_graph_result result(ctx->graph_max_nodes(1));
    direct_graph graph(model, graph_params(ctx, &result));
    ggml_tensor * vectors = input_f32_2d(graph, hparams.n_embd, n_columns);
    ggml_tensor * normed = graph.build_norm(vectors, model.output_norm, model.output_norm_b, LLM_NORM, -1);
    ggml_tensor * logits = ggml_mul_mat(graph.ctx0, model.output, normed);
    if (model.output_s) {
        logits = ggml_mul(graph.ctx0, logits, model.output_s);
    }
    add_output(graph, logits);
    if (!ggml_backend_sched_alloc_graph(ctx->get_sched(), graph.gf)) {
        throw std::runtime_error("failed to allocate raw projection graph");
    }
    ggml_backend_tensor_set(vectors, values.data(), 0, values.size() * sizeof(float));
    if (ctx->graph_compute(graph.gf, false) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("GGML raw projection graph failed");
    }
    ctx->synchronize();

    const size_t n_vocab = llama_vocab_n_tokens(&model.vocab);
    std::vector<float> projected(n_vocab * n_columns);
    ggml_backend_tensor_get(logits, projected.data(), 0, projected.size() * sizeof(float));
    top = std::min(top, (int) n_vocab);
    std::vector<source_readout> readouts;
    readouts.reserve(n_vectors);
    for (size_t vector = 0; vector < n_vectors; ++vector) {
        const float * column_values = projected.data() + vector * n_vocab;
        std::vector<int> ids(n_vocab);
        for (int id = 0; id < (int) n_vocab; ++id) ids[id] = id;
        std::partial_sort(ids.begin(), ids.begin() + top, ids.end(), [column_values](int a, int b) {
            return column_values[a] > column_values[b];
        });
        const size_t layer = vector / kinds.size();
        const source_kind kind = kinds[vector % kinds.size()];
        const float baseline_cosine = baseline ? cosine_similarity(raw_values(raw[layer], kind), raw_values((*baseline)[layer], kind)) : 1.0f;
        int target_rank = -1;
        float target_logit = 0.0f;
        if (target >= 0 && (size_t) target < n_vocab) {
            target_logit = column_values[target];
            target_rank = 1;
            for (size_t id = 0; id < n_vocab; ++id) target_rank += column_values[id] > target_logit;
        }
        int random_target_rank = -1;
        float random_target_logit = 0.0f;
        if (random_baseline && target >= 0 && (size_t) target < n_vocab) {
            const float * random_values = projected.data() + (vector + n_vectors) * n_vocab;
            random_target_logit = random_values[target];
            random_target_rank = 1;
            for (size_t id = 0; id < n_vocab; ++id) random_target_rank += random_values[id] > random_target_logit;
        }
        source_readout readout = { sources[vector], {}, {}, baseline_cosine, target_rank, target_logit, random_target_rank, random_target_logit };
        readout.reported_tokens.reserve(report_tokens.size());
        for (llama_token report_token : report_tokens) {
            const float report_logit = column_values[report_token];
            int report_rank = 1;
            for (size_t id = 0; id < n_vocab; ++id) report_rank += column_values[id] > report_logit;
            readout.reported_tokens.push_back({ report_token, common_token_to_piece(&model.vocab, report_token, true), report_rank, report_logit });
        }
        readout.top_tokens.reserve(top);
        for (int rank = 0; rank < top; ++rank) {
            const int id = ids[rank];
            readout.top_tokens.push_back({ id, common_token_to_piece(&model.vocab, id, true), column_values[id] });
        }
        readouts.push_back(std::move(readout));
    }
    return readouts;
}

static void write_json_string(std::ostream & output, const std::string & value) {
    static const char hex[] = "0123456789abcdef";
    output << '"';
    for (unsigned char c : value) {
        switch (c) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (c < 0x20) {
                    output << "\\u00" << hex[c >> 4] << hex[c & 0x0f];
                } else {
                    output << c;
                }
        }
    }
    output << '"';
}

static void write_trace(const std::string & path, const std::string & model_path, const std::string & prompt,
                        int n_gpu_layers, int top, const std::string & trace_kind,
                        const std::vector<std::string> & sources, llama_token tracked_token,
                        const std::vector<llama_token> & steer_sources, const std::vector<llama_token> & steer_targets,
                        uint32_t steer_components, int steer_start, int steer_steps,
                        const std::vector<llama_token> & report_tokens,
                        const std::string & stopped, const llama_vocab * vocab, const std::vector<step_result> & steps) {
    std::ofstream output(path);
    if (!output) throw std::runtime_error("failed to open JSON output");
    output << std::setprecision(9) << "{\n  \"schema_version\": 1,\n  \"trace_kind\": ";
    write_json_string(output, trace_kind);
    output << ",\n  \"model_path\": ";
    write_json_string(output, model_path);
    output << ",\n  \"n_gpu_layers\": " << n_gpu_layers << ",\n  \"top_k\": " << top << ",\n  \"sources\": [";
    for (size_t source = 0; source < sources.size(); ++source) {
        if (source) output << ", ";
        write_json_string(output, sources[source]);
    }
    output << "]";
    if (tracked_token >= 0) {
        output << ",\n  \"tracked_token\": {\"id\": " << tracked_token << ", \"piece\": ";
        write_json_string(output, common_token_to_piece(vocab, tracked_token, true));
        output << "}";
    }
    if (!report_tokens.empty()) {
        output << ",\n  \"reported_tokens\": [";
        for (size_t i = 0; i < report_tokens.size(); ++i) {
            if (i) output << ",";
            output << "{\"id\": " << report_tokens[i] << ", \"piece\": ";
            write_json_string(output, common_token_to_piece(vocab, report_tokens[i], true));
            output << "}";
        }
        output << "]";
    }
    if (!steer_sources.empty()) {
        output << ",\n  \"static_steering\": {\"source_tokens\": [";
        for (size_t i = 0; i < steer_sources.size(); ++i) {
            if (i) output << ",";
            output << "{\"id\": " << steer_sources[i] << ", \"piece\": ";
            write_json_string(output, common_token_to_piece(vocab, steer_sources[i], true));
            output << "}";
        }
        output << "], \"target_tokens\": [";
        for (size_t i = 0; i < steer_targets.size(); ++i) {
            if (i) output << ",";
            output << "{\"id\": " << steer_targets[i] << ", \"piece\": ";
            write_json_string(output, common_token_to_piece(vocab, steer_targets[i], true));
            output << "}";
        }
        output << "], \"components\": [";
        bool needs_comma = false;
        for (steer_component component : { steer_r, steer_w, steer_k, steer_v, steer_a }) {
            if (!(steer_components & component)) continue;
            if (needs_comma) output << ",";
            write_json_string(output, steer_component_name(component));
            needs_comma = true;
        }
        output << "], \"formula\": \"x - l2norm(l2norm(sum(source)) - l2norm(sum(target))) * max(dot(x, l2norm(sum(source))), 0)\", \"start_step\": " << steer_start << ", \"step_count\": ";
        if (steer_steps < 0) output << "null";
        else output << steer_steps;
        output << "}";
    }
    output << ",\n  \"runs\": [{\n    \"prompt\": ";
    write_json_string(output, prompt);
    output << ",\n    \"prompt_token_count\": 0,\n    \"stopped\": ";
    write_json_string(output, stopped);
    output << ",\n    \"steps\": [";
    for (size_t step_index = 0; step_index < steps.size(); ++step_index) {
        const auto & step = steps[step_index];
        if (step_index) output << ",";
        output << "\n      {\"index\": " << step_index << ", \"input\": {\"id\": " << step.input << ", \"piece\": ";
        write_json_string(output, common_token_to_piece(vocab, step.input, true));
        output << "}, \"next\": {\"id\": " << step.next << ", \"piece\": ";
        write_json_string(output, common_token_to_piece(vocab, step.next, true));
        output << "}, \"final_round_trip_error\": 0, \"state_cosine\": [";
        for (size_t il = 0; il < step.state_cosine.size(); ++il) {
            if (il) output << ",";
            output << step.state_cosine[il];
        }
        output << "], \"final_reported_tokens\": [";
        for (size_t token_index = 0; token_index < step.final_reported_tokens.size(); ++token_index) {
            if (token_index) output << ",";
            const auto & token = step.final_reported_tokens[token_index];
            output << "{\"id\": " << token.id << ", \"piece\": ";
            write_json_string(output, token.piece);
            output << ", \"rank\": " << token.rank << ", \"logit\": " << token.logit << "}";
        }
        output << "], \"readouts\": [";
        for (size_t il = 0; il < step.readouts.size(); ++il) {
            if (il) output << ",";
            output << "{\"source\": ";
            write_json_string(output, step.readouts[il].source);
            output << ", \"baseline_cosine\": " << step.readouts[il].baseline_cosine;
            if (step.readouts[il].target_rank >= 0) {
                output << ", \"target_rank\": " << step.readouts[il].target_rank
                       << ", \"target_logit\": " << step.readouts[il].target_logit;
            }
            if (step.readouts[il].random_target_rank >= 0) {
                output << ", \"random_target_rank\": " << step.readouts[il].random_target_rank
                       << ", \"random_target_logit\": " << step.readouts[il].random_target_logit;
            }
            output << ", \"reported_tokens\": [";
            for (size_t token_index = 0; token_index < step.readouts[il].reported_tokens.size(); ++token_index) {
                if (token_index) output << ",";
                const auto & token = step.readouts[il].reported_tokens[token_index];
                output << "{\"id\": " << token.id << ", \"piece\": ";
                write_json_string(output, token.piece);
                output << ", \"rank\": " << token.rank << ", \"logit\": " << token.logit << "}";
            }
            output << "]";
            output << ", \"top_tokens\": [";
            for (size_t rank = 0; rank < step.readouts[il].top_tokens.size(); ++rank) {
                if (rank) output << ",";
                const auto & token = step.readouts[il].top_tokens[rank];
                output << "{\"rank\": " << rank + 1 << ", \"id\": " << token.id << ", \"piece\": ";
                write_json_string(output, token.piece);
                output << ", \"logit\": " << token.logit << "}";
            }
            output << "]}";
        }
        output << "]}";
    }
    output << "\n    ]\n  }]\n}\n";
}

} // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    std::string model_path;
    std::string prompt = "The Eiffel Tower is located in";
    std::string json_path = "rwkv7-time-lens.json";
    std::string forced_text;
    std::string track_text;
    std::vector<std::string> report_texts;
    std::string swap_text;
    std::vector<std::string> steer_source_texts;
    std::vector<std::string> steer_target_texts;
    int n_gpu_layers = 0;
    int steps = 3;
    int top = 5;
    int swap_step = 0;
    int steer_start = 0;
    int steer_steps = -1;
    uint32_t steer_components = steer_default_components;
    uint32_t steer_sweep_components = 0;
    float epsilon = 0.1f;
    source_kind source = source_kind::r;
    swap_source swap_kind = swap_source::v;
    run_mode mode = run_mode::project;
    std::vector<source_kind> project_sources = all_source_kinds();
    bool capture_prompt_last = false;
    bool random_baseline = false;
    bool in_graph_unembed = true;
    bool steer_edit_set = false;
    bool steer_sweep_set = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (std::strcmp(argv[i], "-p") == 0 && i + 1 < argc) prompt = argv[++i];
        else if (std::strcmp(argv[i], "-ngl") == 0 && i + 1 < argc) n_gpu_layers = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc) mode = parse_mode(argv[++i]);
        else if (std::strcmp(argv[i], "--source") == 0 && i + 1 < argc) source = parse_source(argv[++i]);
        else if (std::strcmp(argv[i], "--project-sources") == 0 && i + 1 < argc) project_sources = parse_project_sources(argv[++i]);
        else if (std::strcmp(argv[i], "--no-in-graph-unembed") == 0) in_graph_unembed = false;
        else if (std::strcmp(argv[i], "--capture-prompt-last") == 0) capture_prompt_last = true;
        else if (std::strcmp(argv[i], "--random-baseline") == 0) random_baseline = true;
        else if (std::strcmp(argv[i], "--track-text") == 0 && i + 1 < argc) track_text = argv[++i];
        else if (std::strcmp(argv[i], "--report-text") == 0 && i + 1 < argc) report_texts.push_back(argv[++i]);
        else if (std::strcmp(argv[i], "--forced-text") == 0 && i + 1 < argc) forced_text = argv[++i];
        else if (std::strcmp(argv[i], "--swap-text") == 0 && i + 1 < argc) swap_text = argv[++i];
        else if (std::strcmp(argv[i], "--swap-source") == 0 && i + 1 < argc) swap_kind = parse_swap_source(argv[++i]);
        else if (std::strcmp(argv[i], "--swap-step") == 0 && i + 1 < argc) swap_step = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--steer-source") == 0 && i + 1 < argc) steer_source_texts.push_back(argv[++i]);
        else if (std::strcmp(argv[i], "--steer-target") == 0 && i + 1 < argc) steer_target_texts.push_back(argv[++i]);
        else if (std::strcmp(argv[i], "--steer-edit") == 0 && i + 1 < argc) { steer_components = parse_steer_components(argv[++i]); steer_edit_set = true; }
        else if (std::strcmp(argv[i], "--steer-sweep") == 0 && i + 1 < argc) { steer_sweep_components = parse_steer_components(argv[++i]); steer_sweep_set = true; }
        else if (std::strcmp(argv[i], "--steer-start") == 0 && i + 1 < argc) steer_start = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--steer-steps") == 0 && i + 1 < argc) steer_steps = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--steer-smiling") == 0 && i + 1 < argc) steer_source_texts.push_back(argv[++i]);
        else if (std::strcmp(argv[i], "--steer-crying") == 0 && i + 1 < argc) steer_target_texts.push_back(argv[++i]);
        else if (std::strcmp(argv[i], "--steps") == 0 && i + 1 < argc) steps = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--epsilon") == 0 && i + 1 < argc) epsilon = std::strtof(argv[++i], nullptr);
        else if (std::strcmp(argv[i], "--top") == 0 && i + 1 < argc) top = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--json") == 0 && i + 1 < argc) json_path = argv[++i];
        else { usage(argv[0]); return 1; }
    }
    const bool swap_enabled = !swap_text.empty();
    const bool steering_enabled = !steer_source_texts.empty() || !steer_target_texts.empty();
    const bool fused_projection = mode == run_mode::project && in_graph_unembed && !random_baseline && !swap_enabled;
    if (model_path.empty() || steps <= 0 || top <= 0 || epsilon <= 0.0f || swap_step < 0 || steer_start < 0 || steer_steps == 0 || steer_steps < -1 ||
        (swap_enabled && track_text.empty()) ||
        (steering_enabled && (steer_source_texts.empty() || steer_target_texts.empty() || swap_enabled || (steer_edit_set && steer_sweep_set))) ||
        (!steering_enabled && (steer_edit_set || steer_sweep_set || steer_start != 0 || steer_steps >= 0))) {
        usage(argv[0]);
        return 1;
    }

    ggml_backend_load_all();
    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) throw std::runtime_error("failed to load model");
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const std::vector<llama_token> prompt_tokens = common_tokenize(vocab, prompt, false, true);
    const std::vector<llama_token> forced_tokens = forced_text.empty() ? std::vector<llama_token>() : common_tokenize(vocab, forced_text, false, true);
    const std::vector<llama_token> tracked_tokens = track_text.empty() ? std::vector<llama_token>() : common_tokenize(vocab, track_text, false, true);
    std::vector<llama_token> report_tokens;
    for (const std::string & report_text : report_texts) {
        const std::vector<llama_token> tokens = common_tokenize(vocab, report_text, false, true);
        if (tokens.size() != 1) throw std::runtime_error("--report-text must name one token");
        if (std::find(report_tokens.begin(), report_tokens.end(), tokens[0]) == report_tokens.end()) report_tokens.push_back(tokens[0]);
    }
    const std::vector<llama_token> swap_tokens = swap_text.empty() ? std::vector<llama_token>() : common_tokenize(vocab, swap_text, false, true);
    std::vector<llama_token> steer_source_tokens;
    for (const std::string & steer_source_text : steer_source_texts) {
        const std::vector<llama_token> tokens = common_tokenize(vocab, steer_source_text, false, true);
        if (tokens.size() != 1) throw std::runtime_error("--steer-source must name one token");
        if (std::find(steer_source_tokens.begin(), steer_source_tokens.end(), tokens[0]) == steer_source_tokens.end()) steer_source_tokens.push_back(tokens[0]);
    }
    std::vector<llama_token> steer_target_tokens;
    for (const std::string & steer_target_text : steer_target_texts) {
        const std::vector<llama_token> tokens = common_tokenize(vocab, steer_target_text, false, true);
        if (tokens.size() != 1) throw std::runtime_error("--steer-target must name one token");
        if (std::find(steer_target_tokens.begin(), steer_target_tokens.end(), tokens[0]) == steer_target_tokens.end()) steer_target_tokens.push_back(tokens[0]);
    }
    if (steering_enabled && steer_source_tokens.size() != steer_target_tokens.size()) {
        throw std::runtime_error("--steer-source and --steer-target must name equally sized token sets");
    }
    if (!forced_text.empty() && forced_tokens.empty()) throw std::runtime_error("--forced-text produced no tokens");
    if (capture_prompt_last && prompt_tokens.size() < 2) throw std::runtime_error("--capture-prompt-last needs at least two prompt tokens");
    if (!track_text.empty() && tracked_tokens.size() != 1) {
        throw std::runtime_error("--track-text must name one token; omit it to track the model's next token automatically");
    }
    if (swap_enabled && swap_tokens.size() != 1) {
        throw std::runtime_error("--swap-text must name one token");
    }
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = std::max(128, (int) prompt_tokens.size() + steps + 8);
    cparams.n_batch = prompt_tokens.size();
    cparams.n_ubatch = cparams.n_batch;
    cparams.n_seq_max = 1;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) throw std::runtime_error("failed to create context");
    projection_workspace workspace;
    if (fused_projection) workspace = make_projection_workspace(ctx->get_model());
    const size_t prompt_decode_tokens = capture_prompt_last ? prompt_tokens.size() - 1 : prompt_tokens.size();
    llama_batch batch = llama_batch_init((int32_t) prompt_decode_tokens, 0, 1);
    common_batch_clear(batch);
    for (size_t i = 0; i < prompt_decode_tokens; ++i) common_batch_add(batch, prompt_tokens[i], (llama_pos) i, { 0 }, true);
    if (llama_decode(ctx, batch) != 0) throw std::runtime_error("prompt decode failed");
    llama_batch_free(batch);

    const rwkv_state initial_state = copy_context_state(ctx);
    llama_token initial_carrier;
    if (capture_prompt_last) {
        initial_carrier = prompt_tokens.back();
    } else {
        std::vector<float> prompt_logits(llama_vocab_n_tokens(vocab));
        std::copy(llama_get_logits_ith(ctx, (int32_t) prompt_tokens.size() - 1),
                  llama_get_logits_ith(ctx, (int32_t) prompt_tokens.size() - 1) + prompt_logits.size(), prompt_logits.begin());
        initial_carrier = greedy_token(prompt_logits);
    }
    if (!forced_tokens.empty()) initial_carrier = forced_tokens[0];
    const llama_token fixed_tracked_token = !tracked_tokens.empty() ? tracked_tokens[0] : forced_tokens.empty() ? LLAMA_TOKEN_NULL : forced_tokens[0];
    llama_token tracked_token = fixed_tracked_token;
    const int run_steps = steps;
    if (swap_enabled && swap_step >= run_steps) throw std::runtime_error("--swap-step is outside this run");
    if (steering_enabled && steer_start >= run_steps) throw std::runtime_error("--steer-start is outside this run");
    std::vector<uint32_t> sweep_runs;
    if (steer_sweep_set) {
        std::vector<steer_component> sweep_components;
        for (steer_component component : { steer_r, steer_w, steer_k, steer_v, steer_a }) {
            if (steer_sweep_components & component) sweep_components.push_back(component);
        }
        sweep_runs.reserve(1u << sweep_components.size());
        for (uint32_t subset = 0; subset < (1u << sweep_components.size()); ++subset) {
            uint32_t components = 0;
            for (size_t i = 0; i < sweep_components.size(); ++i) {
                if (subset & (1u << i)) components |= sweep_components[i];
            }
            sweep_runs.push_back(components);
        }
    } else {
        sweep_runs.push_back(steering_enabled ? steer_components : 0);
    }
    const std::vector<llama_token> no_steer_tokens;
    for (uint32_t active_components : sweep_runs) {
    rwkv_state state = initial_state;
    llama_token carrier = initial_carrier;
        std::vector<step_result> trace;
        trace.reserve(run_steps);
        std::vector<std::string> trace_sources;
        std::vector<snapshot::raw_layer> baseline_raw;
        std::vector<std::vector<float>> baseline_state;
    const bool steering_this_run = steering_enabled && active_components != 0;
    if (steering_this_run) {
        std::printf("semantic steering (");
        bool first_component = true;
        for (steer_component component : { steer_r, steer_w, steer_k, steer_v, steer_a }) {
            if (!(active_components & component)) continue;
            std::printf("%s%s", first_component ? "" : ",", steer_component_name(component));
            first_component = false;
        }
        std::printf(") at every layer for generation steps %d", steer_start);
        if (steer_steps < 0) std::printf(" onward:");
        else std::printf(" through %d:", steer_start + steer_steps - 1);
        for (llama_token source : steer_source_tokens) std::printf(" %s", common_token_to_piece(vocab, source, true).c_str());
        std::printf(" ->");
        for (llama_token target : steer_target_tokens) std::printf(" %s", common_token_to_piece(vocab, target, true).c_str());
        std::printf("\n");
    }
    for (int step = 0; step < run_steps; ++step) {
        snapshot clean;
        step_result result;
        result.input = carrier;
        const bool steer_this_step = steering_this_run && step >= steer_start && (steer_steps < 0 || step - steer_start < steer_steps);
        if (swap_enabled && step == swap_step) {
            snapshot base = run_snapshot(ctx, carrier, state);
            snapshot replacement = run_snapshot(ctx, swap_tokens[0], state);
            const std::vector<source_readout> readouts = project_raw_vectors(ctx, base.raw, { source_kind::k, source_kind::v }, nullptr, tracked_tokens[0], {}, false, top);
            std::vector<activation_patch> patches;
            int best_layer = -1;
            for (int layer = 0; layer < (int) base.raw.size(); ++layer) {
                const source_readout & k_readout = readouts[layer * 2];
                const source_readout & v_readout = readouts[layer * 2 + 1];
                const bool has_k = k_readout.target_rank >= 1 && k_readout.target_rank <= top;
                const bool has_v = v_readout.target_rank >= 1 && v_readout.target_rank <= top;
                const bool eligible = swap_kind == swap_source::k ? has_k : swap_kind == swap_source::v ? has_v : has_k && has_v;
                const int rank = swap_kind == swap_source::k ? k_readout.target_rank : swap_kind == swap_source::v ? v_readout.target_rank : k_readout.target_rank + v_readout.target_rank;
                const float score = swap_kind == swap_source::k ? k_readout.target_logit : swap_kind == swap_source::v ? v_readout.target_logit : k_readout.target_logit + v_readout.target_logit;
                const int best_rank = best_layer < 0 ? 0 : swap_kind == swap_source::k ? readouts[best_layer * 2].target_rank : swap_kind == swap_source::v ? readouts[best_layer * 2 + 1].target_rank : readouts[best_layer * 2].target_rank + readouts[best_layer * 2 + 1].target_rank;
                const float best_score = best_layer < 0 ? 0.0f : swap_kind == swap_source::k ? readouts[best_layer * 2].target_logit : swap_kind == swap_source::v ? readouts[best_layer * 2 + 1].target_logit : readouts[best_layer * 2].target_logit + readouts[best_layer * 2 + 1].target_logit;
                if (eligible && (best_layer < 0 || rank < best_rank || (rank == best_rank && score > best_score))) {
                    best_layer = layer;
                }
            }
            if (best_layer >= 0) {
                const bool swap_k = swap_kind == swap_source::k || swap_kind == swap_source::kv;
                const bool swap_v = swap_kind == swap_source::v || swap_kind == swap_source::kv;
                patches.push_back({ best_layer,
                    swap_k ? replacement.raw[best_layer].k : std::vector<float>(),
                    swap_k ? replacement.raw[best_layer].kk : std::vector<float>(),
                    swap_v ? replacement.raw[best_layer].v : std::vector<float>() });
                const char * kind = swap_kind == swap_source::k ? "k" : swap_kind == swap_source::v ? "v" : "k+v";
                std::printf("swap step %d: layer %d %s replaced from %s\n", step, best_layer, kind,
                    common_token_to_piece(vocab, swap_tokens[0], true).c_str());
                clean = run_snapshot(ctx, carrier, state, &patches);
                if (swap_k) std::printf("swap k cosine to replacement: %.6f\n", cosine_similarity(clean.raw[best_layer].k, replacement.raw[best_layer].k));
                if (swap_v) std::printf("swap v cosine to replacement: %.6f\n", cosine_similarity(clean.raw[best_layer].v, replacement.raw[best_layer].v));
            } else {
                std::printf("swap step %d: no v source ranked %d or better for %s\n", step, top,
                    common_token_to_piece(vocab, tracked_tokens[0], true).c_str());
                clean = std::move(base);
            }
        } else {
            clean = run_snapshot(ctx, carrier, state, nullptr,
                steer_this_step ? &steer_source_tokens : nullptr, steer_this_step ? &steer_target_tokens : nullptr, active_components,
                fused_projection ? &project_sources : nullptr, fused_projection ? &workspace : nullptr);
        }
        result.next = !forced_tokens.empty() && step + 1 < (int) forced_tokens.size() ? forced_tokens[step + 1] : greedy_token(clean.logits);
        result.final_reported_tokens = report_logits(vocab, clean.logits, report_tokens);
        if (baseline_state.empty()) {
            baseline_state = clean.next_state.s;
        }
        if (!fused_projection && baseline_raw.empty()) baseline_raw = clean.raw;
        result.state_cosine.reserve(clean.next_state.s.size());
        for (size_t il = 0; il < clean.next_state.s.size(); ++il) {
            result.state_cosine.push_back(cosine_similarity(clean.next_state.s[il], baseline_state[il]));
        }
        const llama_token step_target = fixed_tracked_token != LLAMA_TOKEN_NULL ? fixed_tracked_token : greedy_token(clean.logits);
        if (mode == run_mode::project) {
            result.readouts = fused_projection ? summarize_projected_logits(ctx->get_model(), clean.projected_logits, project_sources, step_target, report_tokens, top) :
                project_raw_vectors(ctx, clean.raw, project_sources, &baseline_raw, step_target, report_tokens, random_baseline, top);
        } else {
            result.readouts.reserve(clean.raw.size());
            for (int il = 0; il < (int) clean.raw.size(); ++il) {
                result.readouts.push_back({ raw_source_name(il, source),
                    lens_readout(ctx, il, clean.resid_in[il], clean.raw[il], state.r[il], state.s[il], source, epsilon, top), {}, 1.0f, -1, 0.0f, -1, 0.0f });
            }
        }
        if (trace_sources.empty()) {
            trace_sources.reserve(result.readouts.size());
            for (const auto & readout : result.readouts) {
                trace_sources.push_back(readout.source);
            }
        }
        std::printf("step %d: %s -> %s\n", step,
            common_token_to_piece(vocab, result.input, true).c_str(), common_token_to_piece(vocab, result.next, true).c_str());
        carrier = result.next;
        state = std::move(clean.next_state);
        trace.push_back(std::move(result));
    }
    const std::string output_path = steer_sweep_set ? sweep_json_path(json_path, active_components) : json_path;
    const std::string trace_kind = steer_sweep_set ? (steering_this_run ? "rwkv_component_sweep" : "rwkv_raw_projection") :
        steering_enabled ? (steer_components == steer_default_components && steer_start == 0 && steer_steps < 0 ? "rwkv_kv_static_steer" : "rwkv_component_steer") :
        swap_enabled ? "rwkv_kv_swap" : mode == run_mode::project ? "rwkv_raw_projection" : "rwkv_time_lens";
    write_trace(output_path, model_path, prompt, n_gpu_layers, top, trace_kind, trace_sources, tracked_token,
        steering_this_run ? steer_source_tokens : no_steer_tokens, steering_this_run ? steer_target_tokens : no_steer_tokens,
        active_components, steer_start, steer_steps, report_tokens, forced_tokens.empty() ? "steps" : "forced_prefix", vocab, trace);
    std::printf("trace=%s\n", output_path.c_str());
    }
    std::printf("viewer=pocs/interp/rwkv_unembed_viewer.html\n");
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
