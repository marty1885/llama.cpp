#include "common.h"
#include "llama-context.h"
#include "llama-memory-recurrent.h"
#include "llama-model.h"
#include "models/models.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
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

struct source_readout {
    std::string source;
    std::vector<scored_token> top_tokens;
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
        std::vector<float> g;
        std::vector<float> k0;
        std::vector<float> kk;
        std::vector<float> wkv;
        std::vector<float> rkv;
    };
    std::vector<raw_layer> raw;
    std::vector<float> logits;
};

enum class source_kind { r, w, k, v, a, g, wkv, rkv };

enum class run_mode { project, local_lens };

static const char * source_name(source_kind source) {
    switch (source) {
        case source_kind::r:   return "r";
        case source_kind::w:   return "w";
        case source_kind::k:   return "k";
        case source_kind::v:   return "v";
        case source_kind::a:   return "a";
        case source_kind::g:   return "g";
        case source_kind::wkv: return "wkv";
        case source_kind::rkv: return "rkv";
    }
    GGML_ABORT("unknown source");
}

static source_kind parse_source(const std::string & value) {
    for (source_kind source : { source_kind::r, source_kind::w, source_kind::k, source_kind::v,
                                source_kind::a, source_kind::g, source_kind::wkv, source_kind::rkv }) {
        if (value == source_name(source)) return source;
    }
    throw std::runtime_error("unknown --source: " + value);
}

static std::vector<source_kind> all_source_kinds() {
    return { source_kind::r, source_kind::w, source_kind::k, source_kind::v,
             source_kind::a, source_kind::g, source_kind::wkv, source_kind::rkv };
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
        "usage: %s -m MODEL [-p PROMPT] [-ngl N] [--mode project|local-lens] [--source r|w|k|v|a|g|wkv|rkv] [--project-sources all|k,v,a] [--random-baseline] [--capture-prompt-last] [--track-text TEXT] [--forced-text TEXT] [--steps N] [--epsilon E] [--top N] [--json FILE]\n"
        "\n"
        "project (default) batches all 488 captured raw time-mix vectors through the native\n"
        "output normalization and output matrix. local-lens perturbs one source and runs its\n"
        "local downstream closure as a diagnostic.\n",
        argv0);
}

static llama_token greedy_token(const std::vector<float> & logits) {
    return (llama_token) std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()));
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

struct raw_tensors {
    ggml_tensor * r;
    ggml_tensor * w;
    ggml_tensor * k;
    ggml_tensor * v;
    ggml_tensor * a;
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
        int il) {
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
    ggml_tensor * w = ggml_add(graph.ctx0,
        ggml_mul_mat(graph.ctx0, layer.time_mix_w2, ggml_tanh(graph.ctx0, ggml_mul_mat(graph.ctx0, layer.time_mix_w1, xw))),
        layer.time_mix_w0);
    w = ggml_exp(graph.ctx0, ggml_scale(graph.ctx0, ggml_sigmoid(graph.ctx0, w), -0.606531));

    ggml_tensor * k0 = ggml_mul_mat(graph.ctx0, layer.time_mix_key, xk);
    ggml_tensor * k = k0;
    ggml_tensor * v = ggml_mul_mat(graph.ctx0, layer.time_mix_value, xv);
    if (first_layer_value == nullptr) {
        first_layer_value = v;
    } else {
        v = ggml_add(graph.ctx0, v,
            ggml_mul(graph.ctx0, ggml_sub(graph.ctx0, first_layer_value, v),
                ggml_sigmoid(graph.ctx0, ggml_add(graph.ctx0,
                    ggml_mul_mat(graph.ctx0, layer.time_mix_v2, ggml_mul_mat(graph.ctx0, layer.time_mix_v1, xv)),
                    layer.time_mix_v0))));
    }
    ggml_tensor * g = has_gating ? ggml_mul_mat(graph.ctx0, layer.time_mix_g2,
        ggml_sigmoid(graph.ctx0, ggml_mul_mat(graph.ctx0, layer.time_mix_g1, xg))) : nullptr;
    ggml_tensor * a = ggml_sigmoid(graph.ctx0, ggml_add(graph.ctx0,
        ggml_mul_mat(graph.ctx0, layer.time_mix_a2, ggml_mul_mat(graph.ctx0, layer.time_mix_a1, xa)), layer.time_mix_a0));

    ggml_tensor * kk = ggml_reshape_3d(graph.ctx0, ggml_mul(graph.ctx0, k, layer.time_mix_k_k), head_size, head_count, 1);
    kk = ggml_l2_norm(graph.ctx0, kk, 1e-12);
    ggml_tensor * ka = ggml_mul(graph.ctx0, k, layer.time_mix_k_a);
    k = ggml_add(graph.ctx0, k, ggml_sub(graph.ctx0, ggml_mul(graph.ctx0, a, ka), ka));

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
        { r, w, k, v, a, g, k0, kk, wkv_value, rkv },
    };
}

static void add_output(direct_graph & graph, ggml_tensor * tensor) {
    ggml_set_output(tensor);
    ggml_build_forward_expand(graph.gf, tensor);
}

static snapshot run_snapshot(llama_context * ctx, llama_token token, const rwkv_state & state) {
    const auto & model = ctx->get_model();
    const auto & hparams = model.hparams;
    const int n_layer = hparams.n_layer();
    if ((int) state.r.size() != n_layer || (int) state.s.size() != n_layer) {
        throw std::runtime_error("invalid RWKV state for snapshot");
    }

    ctx->synchronize();
    ggml_backend_sched_reset(ctx->get_sched());
    llm_graph_result result(ctx->graph_max_nodes(1));
    direct_graph graph(model, graph_params(ctx, &result));
    ggml_tensor * token_input = ggml_new_tensor_1d(graph.ctx0, GGML_TYPE_I32, 1);
    ggml_set_input(token_input);
    ggml_tensor * cur = ggml_get_rows(graph.ctx0, model.tok_embd, token_input);
    cur = graph.build_norm(cur, model.tok_norm, model.tok_norm_b, LLM_NORM, 0);
    cur = ggml_reshape_3d(graph.ctx0, cur, hparams.n_embd, 1, 1);

    snapshot out;
    out.next_state.r.resize(n_layer);
    out.next_state.s.resize(n_layer);
    out.resid_in.resize(n_layer);
    out.raw.resize(n_layer);
    std::vector<ggml_tensor *> r_inputs(n_layer);
    std::vector<ggml_tensor *> s_inputs(n_layer);
    std::vector<ggml_tensor *> resid_outputs(n_layer);
    std::vector<raw_tensors> raw_outputs(n_layer);
    std::vector<ggml_tensor *> state_outputs(n_layer);
    std::vector<ggml_tensor *> shift_outputs(n_layer);
    for (int il = 0; il < n_layer; ++il) {
        r_inputs[il] = input_f32_3d(graph, hparams.n_embd, hparams.token_shift_count, 1);
        s_inputs[il] = input_f32_2d(graph, hparams.n_embd_s(), 1);
    }

    ggml_tensor * first_layer_value = nullptr;
    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];
        out.resid_in[il].resize(hparams.n_embd);
        add_output(graph, cur);
        resid_outputs[il] = cur;

        ggml_tensor * att_prev = ggml_view_3d(graph.ctx0, r_inputs[il], hparams.n_embd, 1, 1,
            r_inputs[il]->nb[1], r_inputs[il]->nb[2], 0);
        ggml_tensor * ffn_prev = ggml_view_3d(graph.ctx0, r_inputs[il], hparams.n_embd, 1, 1,
            r_inputs[il]->nb[1], r_inputs[il]->nb[2], hparams.n_embd * sizeof(float));
        ggml_tensor * att_norm = graph.build_norm(cur, layer.attn_norm, layer.attn_norm_b, LLM_NORM, il);
        time_result time = build_time_mix(graph, model, att_norm, att_prev, first_layer_value, s_inputs[il], il);
        out.next_state.s[il].resize(hparams.n_embd_s());
        add_output(graph, time.output);
        add_output(graph, time.next_state);
        raw_outputs[il] = time.raw;
        state_outputs[il] = time.next_state;
        for (ggml_tensor * tensor : { time.raw.r, time.raw.w, time.raw.k, time.raw.v, time.raw.a,
                                      time.raw.g, time.raw.k0, time.raw.kk, time.raw.wkv, time.raw.rkv }) {
            add_output(graph, tensor);
        }
        auto & raw = out.raw[il];
        for (std::vector<float> * values : { &raw.r, &raw.w, &raw.k, &raw.v, &raw.a,
                                             &raw.g, &raw.k0, &raw.kk, &raw.wkv, &raw.rkv }) {
            values->resize(hparams.n_embd);
        }

        ggml_tensor * ffn_inp = ggml_add(graph.ctx0, time.output, cur);
        ggml_tensor * ffn_norm = graph.build_norm(ffn_inp, layer.attn_norm_2, layer.attn_norm_2_b, LLM_NORM, il);
        ggml_tensor * channel = graph.build_rwkv7_channel_mix(&layer, ffn_norm, ffn_prev, LLM_ARCH_RWKV7);
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

    if (!ggml_backend_sched_alloc_graph(ctx->get_sched(), graph.gf)) {
        throw std::runtime_error("failed to allocate GGML snapshot graph");
    }
    ggml_backend_tensor_set(token_input, &token, 0, sizeof(token));
    for (int il = 0; il < n_layer; ++il) {
        ggml_backend_tensor_set(r_inputs[il], state.r[il].data(), 0, state.r[il].size() * sizeof(float));
        ggml_backend_tensor_set(s_inputs[il], state.s[il].data(), 0, state.s[il].size() * sizeof(float));
    }
    if (ctx->graph_compute(graph.gf, false) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("GGML snapshot graph failed");
    }
    ctx->synchronize();

    for (int il = 0; il < n_layer; ++il) {
        ggml_backend_tensor_get(resid_outputs[il], out.resid_in[il].data(), 0, hparams.n_embd * sizeof(float));
        ggml_backend_tensor_get(state_outputs[il], out.next_state.s[il].data(), 0, hparams.n_embd_s() * sizeof(float));
        ggml_backend_tensor_get(shift_outputs[il], out.next_state.r[il].data(), 0, hparams.n_embd_r() * sizeof(float));
        const auto & tensors = raw_outputs[il];
        auto & raw = out.raw[il];
        const std::vector<ggml_tensor *> sources = { tensors.r, tensors.w, tensors.k, tensors.v, tensors.a,
                                                      tensors.g, tensors.k0, tensors.kk, tensors.wkv, tensors.rkv };
        const std::vector<std::vector<float> *> values = { &raw.r, &raw.w, &raw.k, &raw.v, &raw.a,
                                                            &raw.g, &raw.k0, &raw.kk, &raw.wkv, &raw.rkv };
        for (size_t source = 0; source < sources.size(); ++source) {
            ggml_backend_tensor_get(sources[source], values[source]->data(), 0, hparams.n_embd * sizeof(float));
        }
    }
    out.logits.resize(llama_vocab_n_tokens(&model.vocab));
    ggml_backend_tensor_get(logits, out.logits.data(), 0, out.logits.size() * sizeof(float));
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
    return "rwkv.layer." + std::to_string(layer) + ".time." + source_name(source);
}

static const std::vector<float> & raw_values(const snapshot::raw_layer & raw, source_kind source) {
    switch (source) {
        case source_kind::r:   return raw.r;
        case source_kind::w:   return raw.w;
        case source_kind::k:   return raw.k;
        case source_kind::v:   return raw.v;
        case source_kind::a:   return raw.a;
        case source_kind::g:   return raw.g;
        case source_kind::wkv: return raw.wkv;
        case source_kind::rkv: return raw.rkv;
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

static std::vector<source_readout> project_raw_vectors(
        llama_context * ctx, const std::vector<snapshot::raw_layer> & raw,
        const std::vector<source_kind> & kinds, const std::vector<snapshot::raw_layer> * baseline,
        llama_token target, bool random_baseline, int top) {
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
            values[i] = (state & 1) ? 1.0f : -1.0f;
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
        source_readout readout = { sources[vector], {}, baseline_cosine, target_rank, target_logit, random_target_rank, random_target_logit };
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
    int n_gpu_layers = 0;
    int steps = 3;
    int top = 5;
    float epsilon = 0.1f;
    source_kind source = source_kind::r;
    run_mode mode = run_mode::project;
    std::vector<source_kind> project_sources = all_source_kinds();
    bool capture_prompt_last = false;
    bool random_baseline = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (std::strcmp(argv[i], "-p") == 0 && i + 1 < argc) prompt = argv[++i];
        else if (std::strcmp(argv[i], "-ngl") == 0 && i + 1 < argc) n_gpu_layers = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc) mode = parse_mode(argv[++i]);
        else if (std::strcmp(argv[i], "--source") == 0 && i + 1 < argc) source = parse_source(argv[++i]);
        else if (std::strcmp(argv[i], "--project-sources") == 0 && i + 1 < argc) project_sources = parse_project_sources(argv[++i]);
        else if (std::strcmp(argv[i], "--capture-prompt-last") == 0) capture_prompt_last = true;
        else if (std::strcmp(argv[i], "--random-baseline") == 0) random_baseline = true;
        else if (std::strcmp(argv[i], "--track-text") == 0 && i + 1 < argc) track_text = argv[++i];
        else if (std::strcmp(argv[i], "--forced-text") == 0 && i + 1 < argc) forced_text = argv[++i];
        else if (std::strcmp(argv[i], "--steps") == 0 && i + 1 < argc) steps = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--epsilon") == 0 && i + 1 < argc) epsilon = std::strtof(argv[++i], nullptr);
        else if (std::strcmp(argv[i], "--top") == 0 && i + 1 < argc) top = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--json") == 0 && i + 1 < argc) json_path = argv[++i];
        else { usage(argv[0]); return 1; }
    }
    if (model_path.empty() || steps <= 0 || top <= 0 || epsilon <= 0.0f) { usage(argv[0]); return 1; }

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
    if (!forced_text.empty() && forced_tokens.empty()) throw std::runtime_error("--forced-text produced no tokens");
    if (capture_prompt_last && prompt_tokens.size() < 2) throw std::runtime_error("--capture-prompt-last needs at least two prompt tokens");
    if (!track_text.empty() && tracked_tokens.size() != 1) {
        throw std::runtime_error("--track-text must name one token; omit it to track the model's next token automatically");
    }
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = std::max(128, (int) prompt_tokens.size() + steps + 8);
    cparams.n_batch = prompt_tokens.size();
    cparams.n_ubatch = prompt_tokens.size();
    cparams.n_seq_max = 1;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) throw std::runtime_error("failed to create context");

    const size_t prompt_decode_tokens = capture_prompt_last ? prompt_tokens.size() - 1 : prompt_tokens.size();
    llama_batch batch = llama_batch_init((int32_t) prompt_decode_tokens, 0, 1);
    common_batch_clear(batch);
    for (size_t i = 0; i < prompt_decode_tokens; ++i) common_batch_add(batch, prompt_tokens[i], (llama_pos) i, { 0 }, true);
    if (llama_decode(ctx, batch) != 0) throw std::runtime_error("prompt decode failed");
    llama_batch_free(batch);

    rwkv_state state = copy_context_state(ctx);
    llama_token carrier;
    if (capture_prompt_last) {
        carrier = prompt_tokens.back();
    } else {
        std::vector<float> prompt_logits(llama_vocab_n_tokens(vocab));
        std::copy(llama_get_logits_ith(ctx, (int32_t) prompt_tokens.size() - 1),
                  llama_get_logits_ith(ctx, (int32_t) prompt_tokens.size() - 1) + prompt_logits.size(), prompt_logits.begin());
        carrier = greedy_token(prompt_logits);
    }
    if (!forced_tokens.empty()) carrier = forced_tokens[0];
    const llama_token fixed_tracked_token = !tracked_tokens.empty() ? tracked_tokens[0] : forced_tokens.empty() ? LLAMA_TOKEN_NULL : forced_tokens[0];
    llama_token tracked_token = fixed_tracked_token;
    const int run_steps = forced_tokens.empty() ? steps : std::min(steps, (int) forced_tokens.size());
    std::vector<step_result> trace;
    trace.reserve(run_steps);
    std::vector<std::string> trace_sources;
    std::vector<snapshot::raw_layer> baseline_raw;
    std::vector<std::vector<float>> baseline_state;
    for (int step = 0; step < run_steps; ++step) {
        snapshot clean = run_snapshot(ctx, carrier, state);
        const llama_token next = !forced_tokens.empty() && step + 1 < (int) forced_tokens.size() ? forced_tokens[step + 1] : greedy_token(clean.logits);
        step_result result = { carrier, next, {}, {} };
        if (baseline_raw.empty()) {
            baseline_raw = clean.raw;
            baseline_state = clean.next_state.s;
        }
        result.state_cosine.reserve(clean.next_state.s.size());
        for (size_t il = 0; il < clean.next_state.s.size(); ++il) {
            result.state_cosine.push_back(cosine_similarity(clean.next_state.s[il], baseline_state[il]));
        }
        const llama_token step_target = fixed_tracked_token != LLAMA_TOKEN_NULL ? fixed_tracked_token : greedy_token(clean.logits);
        if (mode == run_mode::project) {
            result.readouts = project_raw_vectors(ctx, clean.raw, project_sources, &baseline_raw, step_target, random_baseline, top);
        } else {
            result.readouts.reserve(clean.raw.size());
            for (int il = 0; il < (int) clean.raw.size(); ++il) {
                result.readouts.push_back({ raw_source_name(il, source),
                    lens_readout(ctx, il, clean.resid_in[il], clean.raw[il], state.r[il], state.s[il], source, epsilon, top), 1.0f, -1, 0.0f, -1, 0.0f });
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
    write_trace(json_path, model_path, prompt, n_gpu_layers, top,
        mode == run_mode::project ? "rwkv_raw_projection" : "rwkv_time_lens", trace_sources, tracked_token,
        forced_tokens.empty() ? "steps" : "forced", vocab, trace);
    std::printf("trace=%s\nviewer=pocs/interp/rwkv_unembed_viewer.html\n", json_path.c_str());
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
