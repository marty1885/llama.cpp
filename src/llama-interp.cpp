#include "llama-context.h"
#include "llama-memory-recurrent.h"
#include "llama-model.h"

#include <algorithm>
#include <cstring>

static llama_memory_recurrent * interp_get_recurrent_memory(const llama_context * ctx) {
    if (!ctx) {
        return nullptr;
    }
    return dynamic_cast<llama_memory_recurrent *>(ctx->get_memory());
}

static bool interp_tensor_row_to_fp16(ggml_tensor * tensor, uint32_t row, uint32_t n, std::vector<ggml_fp16_t> & dst) {
    if (!tensor || row >= (uint32_t) tensor->ne[1] || n != (uint32_t) tensor->ne[0]) {
        return false;
    }

    dst.resize(n);
    const size_t row_offset = row * tensor->nb[1];

    if (tensor->type == GGML_TYPE_F16) {
        ggml_backend_tensor_get(tensor, dst.data(), row_offset, n * sizeof(ggml_fp16_t));
        return true;
    }
    if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> tmp(n);
        ggml_backend_tensor_get(tensor, tmp.data(), row_offset, n * sizeof(float));
        for (uint32_t i = 0; i < n; ++i) {
            dst[i] = ggml_fp32_to_fp16(tmp[i]);
        }
        return true;
    }

    return false;
}

static bool interp_fp16_to_tensor_row(ggml_tensor * tensor, uint32_t row, uint32_t n, const std::vector<ggml_fp16_t> & src) {
    if (!tensor || row >= (uint32_t) tensor->ne[1] || n != (uint32_t) tensor->ne[0] || src.size() != n) {
        return false;
    }

    const size_t row_offset = row * tensor->nb[1];

    if (tensor->type == GGML_TYPE_F16) {
        ggml_backend_tensor_set(tensor, src.data(), row_offset, n * sizeof(ggml_fp16_t));
        return true;
    }
    if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> tmp(n);
        for (uint32_t i = 0; i < n; ++i) {
            tmp[i] = ggml_fp16_to_fp32(src[i]);
        }
        ggml_backend_tensor_set(tensor, tmp.data(), row_offset, n * sizeof(float));
        return true;
    }

    return false;
}

bool llama_interp_rwkv_state_init(const llama_context * ctx, llama_interp_rwkv_state * state) {
    if (!ctx || !state) {
        return false;
    }

    const auto & hparams = ctx->get_model().hparams;
    state->pos = -1;
    state->has_next = false;
    state->next_token = LLAMA_TOKEN_NULL;
    state->n_layer = hparams.n_layer();
    state->n_embd_r = hparams.n_embd_r();
    state->n_embd_s = hparams.n_embd_s();
    state->layers.assign(state->n_layer, {});
    for (auto & layer : state->layers) {
        layer.r.assign(state->n_embd_r, ggml_fp32_to_fp16(0.0f));
        layer.s.assign(state->n_embd_s, ggml_fp32_to_fp16(0.0f));
    }

    return true;
}

bool llama_interp_rwkv_state_export(const llama_context * ctx, llama_seq_id seq_id, llama_interp_rwkv_state * state) {
    if (!ctx || !state) {
        return false;
    }

    auto * mem = interp_get_recurrent_memory(ctx);
    if (!mem || seq_id < 0 || (uint32_t) seq_id >= mem->size) {
        return false;
    }

    const int32_t tail = mem->cells[seq_id].tail;
    if (tail < 0) {
        return llama_interp_rwkv_state_init(ctx, state);
    }

    const auto & hparams = ctx->get_model().hparams;
    state->pos = mem->cells[tail].pos;
    state->n_layer = hparams.n_layer();
    state->n_embd_r = hparams.n_embd_r();
    state->n_embd_s = hparams.n_embd_s();
    state->layers.assign(state->n_layer, {});

    const uint32_t row = mem->n_rs_seq == 0 ? (uint32_t) tail : mem->rs_idx[seq_id] * mem->size + (uint32_t) tail;
    for (uint32_t il = 0; il < state->n_layer; ++il) {
        if (!interp_tensor_row_to_fp16(mem->r_l[il], row, state->n_embd_r, state->layers[il].r)) {
            return false;
        }
        if (!interp_tensor_row_to_fp16(mem->s_l[il], row, state->n_embd_s, state->layers[il].s)) {
            return false;
        }
    }

    return true;
}

bool llama_interp_rwkv_state_import(llama_context * ctx, llama_seq_id seq_id, const llama_interp_rwkv_state * state) {
    if (!ctx || !state) {
        return false;
    }

    auto * mem = interp_get_recurrent_memory(ctx);
    if (!mem || seq_id < 0 || (uint32_t) seq_id >= mem->size) {
        return false;
    }

    const auto & hparams = ctx->get_model().hparams;
    if (state->n_layer != (uint32_t) hparams.n_layer() ||
        state->n_embd_r != hparams.n_embd_r() ||
        state->n_embd_s != hparams.n_embd_s() ||
        state->layers.size() != state->n_layer) {
        return false;
    }

    ctx->synchronize();
    mem->seq_rm(seq_id, -1, -1);

    const uint32_t row = (uint32_t) seq_id;
    for (uint32_t il = 0; il < state->n_layer; ++il) {
        if (!interp_fp16_to_tensor_row(mem->r_l[il], row, state->n_embd_r, state->layers[il].r)) {
            return false;
        }
        if (!interp_fp16_to_tensor_row(mem->s_l[il], row, state->n_embd_s, state->layers[il].s)) {
            return false;
        }
    }

    auto & cell = mem->cells[row];
    cell.pos = state->pos;
    cell.src = row;
    cell.src0 = row;
    cell.tail = row;
    cell.seq_id.clear();
    cell.seq_id.insert(seq_id);
    mem->cells[seq_id].tail = row;

    if ((size_t) seq_id < mem->rs_idx.size()) {
        mem->rs_idx[seq_id] = 0;
    }

    mem->used = std::count_if(mem->cells.begin(), mem->cells.end(), [](const llama_memory_recurrent::mem_cell & c) {
        return !c.is_empty();
    });
    mem->head = row;
    mem->n = 1;

    return true;
}
