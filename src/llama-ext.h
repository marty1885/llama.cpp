#pragma once

// this is a staging header for new llama.cpp API
// breaking changes and C++ are allowed. everything here should be considered WIP
// try as much as possible to not include this header in the rest of the codebase

#include "llama.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

// Reserve a new compute graph. It is valid until the next call to llama_graph_reserve.
LLAMA_API struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs);

// Get the default ggml_type for a given ftype.
LLAMA_API ggml_type llama_ftype_get_default_type(llama_ftype ftype);

struct quantize_state_impl;

LLAMA_API quantize_state_impl * llama_quant_init(
        const llama_model * model,
        const llama_model_quantize_params * params);

LLAMA_API void llama_quant_free(quantize_state_impl * qs);

// Descriptor for constructing a mock model for quantization testing.
struct llama_quant_model_desc {
    const char * architecture;
    uint32_t n_embd;
    uint32_t n_ff;
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_expert;
    uint32_t n_embd_head_k;
    uint32_t n_embd_head_v;
};

// Create a mock model from a metadata descriptor (for testing).
// The returned model must be freed with llama_model_free().
LLAMA_API llama_model * llama_quant_model_from_metadata(const llama_quant_model_desc * desc);

// Returns true if this tensor should be quantized (based on name, dims, params).
LLAMA_API bool llama_quant_tensor_allows_quantization(
        const quantize_state_impl * qs,
        const ggml_tensor * tensor);

// Compute quantization type assignments for a list of tensors.
// All tensors should be quantizable (use llama_quant_tensor_allows_quantization to filter).
// result_types: caller-allocated array of n_tensors elements, filled with assigned types.
LLAMA_API void llama_quant_compute_types(
        quantize_state_impl * qs,
        llama_ftype ftype,
        ggml_tensor ** tensors,
        ggml_type * result_types,
        size_t n_tensors);

//
// device memory querying
//

// "memory" as in physical memory for a buffer type, in bytes
struct llama_memory_breakdown_data {
    size_t model   = 0; // memory allocated for the model
    size_t context = 0; // memory allocated for the context
    size_t compute = 0; // memory allocated for temporary compute buffers

    size_t total() const {
        return model + context + compute;
    }
};

struct llama_device_memory_data {
    int64_t total;
    int64_t free;
    llama_memory_breakdown_data mb;
};

// TODO: convert to C-style data structure
using llama_memory_breakdown = std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data>;

LLAMA_API int32_t llama_model_n_expert (const struct llama_model * model);
LLAMA_API int32_t llama_model_n_devices(const struct llama_model * model);

LLAMA_API ggml_backend_dev_t llama_model_get_device(const struct llama_model * model, int i);

LLAMA_API llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx);

// Set whether the context outputs nextn embeddings or not
// If masked == true,  output the embeddings only for the tokens with batch.logits != 0
// If masked == false, output the embeddings for all tokens in the batch regardless of batch.logits
LLAMA_API void llama_set_embeddings_nextn(struct llama_context * ctx, bool value, bool masked);

// Select which appended NextN block the DECODER_MTP graph runs (offset past
// the trunk: il = n_layer() + offset). Used by the speculative NextN driver to
// chain multiple trained NextN heads. Default 0 (first head).
LLAMA_API void llama_set_nextn_layer_offset(struct llama_context * ctx, int32_t offset);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_nextn(struct llama_context * ctx);

// LLAMA_API float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);
LLAMA_API float * llama_get_embeddings_nextn_ith(struct llama_context * ctx, int32_t i);

// Set whether the context outputs the input embeddings of a specific layer
LLAMA_API void llama_set_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid, bool value);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid);

LLAMA_API llama_context * llama_get_ctx_other(struct llama_context * ctx);

//
// model/context data extraction
//

// returns pointer to the target-model layer indices
LLAMA_API const int32_t * llama_model_target_layer_ids  (const struct llama_model * model);
// returns the number of extracted layers from target model
LLAMA_API uint32_t        llama_model_target_layer_ids_n(const struct llama_model * model);

//
// interpretability research API
//

struct llama_interp_rwkv_layer_state {
    std::vector<float> r;
    std::vector<float> s;
};

struct llama_interp_rwkv_state {
    llama_pos pos = -1;
    bool has_next = false;
    llama_token next_token = LLAMA_TOKEN_NULL;

    uint32_t n_layer  = 0;
    uint32_t n_embd_r = 0;
    uint32_t n_embd_s = 0;

    std::vector<llama_interp_rwkv_layer_state> layers;
};

struct llama_interp_activation {
    std::string name;
    std::string backend;
    int32_t layer = -1;
    int32_t head_size = 0;
    int32_t n_head = 0;
    std::vector<int64_t> shape;
    std::vector<ggml_fp16_t> data;
    std::vector<float> data_f32;
};

using llama_interp_activation_set = std::vector<llama_interp_activation>;

enum llama_interp_perturb_op {
    LLAMA_INTERP_PERTURB_ADD = 0,
    LLAMA_INTERP_PERTURB_REPLACE = 1,
};

struct llama_interp_capture_spec {
    std::string regex;
    llama_interp_activation_set * dst = nullptr;
    bool f32 = true;
};

struct llama_interp_perturb_spec {
    std::string regex;
    llama_interp_perturb_op op = LLAMA_INTERP_PERTURB_ADD;
    int32_t head = -1; // -1 means whole tensor; otherwise RWKV head index
    std::vector<ggml_fp16_t> data;
};

struct llama_interp_request {
    uint64_t id = 0;
    bool enable_captures = false;
    bool enable_perturbations = false;

    std::vector<llama_interp_capture_spec> captures;
    std::vector<llama_interp_perturb_spec> perturbations;
};

LLAMA_API bool llama_interp_rwkv_state_init(
        const struct llama_context * ctx,
              llama_interp_rwkv_state * state);

LLAMA_API bool llama_interp_rwkv_state_export(
        const struct llama_context * ctx,
              llama_seq_id seq_id,
              llama_interp_rwkv_state * state);

LLAMA_API bool llama_interp_rwkv_state_import(
              struct llama_context * ctx,
              llama_seq_id seq_id,
        const llama_interp_rwkv_state * state);

LLAMA_API void llama_interp_set_request(
        struct llama_context * ctx,
        const llama_interp_request * request);

// Applies the native RWKV7 final control vector, output normalization, and output head to
// F32 final-residual rows without evaluating any recurrent blocks.
LLAMA_API bool llama_interp_rwkv_final_readout(
        struct llama_context * ctx,
        const float * residuals,
        uint32_t n_rows,
        float * logits);
