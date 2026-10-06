#pragma once

#include "llama.h"
#include "llama-memory-recurrent.h"
#include "ggml-backend.h"

#include <algorithm>
#include <coroutine>
#include <exception>
#include <memory>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace rwkv {

enum class state_kind { wkv, s = wkv, shift_att, shift_ffn };

struct wkv_activation {
    int32_t layer;
    size_t microbatch;
    int64_t n_tokens;
    int64_t n_seqs;
    std::vector<float> values;
};

class wkv_trace {
public:
    int32_t n_layers;
    std::vector<int32_t> selected_layers;
    std::vector<wkv_activation> records;
    std::exception_ptr error;

    explicit wkv_trace(int32_t n_layers) : n_layers(n_layers) {
        if (n_layers <= 0) throw std::runtime_error("invalid trace layer count");
    }
    void clear() { records.clear(); error = {}; cursor = 0; }

    static bool callback(ggml_tensor * t, bool ask, void * user) noexcept {
        auto & trace = *static_cast<wkv_trace *>(user);
        if (t->op != GGML_OP_RWKV_WKV7 && t->op != GGML_OP_RWKV_WKV6) return !ask;
        if (ask) {
            trace.current = trace.cursor++;
            const auto layer = int32_t(trace.current % trace.n_layers);
            return trace.selected_layers.empty() || std::find(trace.selected_layers.begin(), trace.selected_layers.end(), layer) != trace.selected_layers.end();
        }
        try {
            const int64_t tokens = t->src[0]->ne[2];
            const int state_index = t->op == GGML_OP_RWKV_WKV7 ? 6 : 5;
            wkv_activation result{int32_t(trace.current % trace.n_layers), trace.current/trace.n_layers, tokens, t->src[state_index]->ne[1], {}};
            result.values.resize(size_t(t->ne[0])*tokens);
            ggml_backend_tensor_get(t, result.values.data(), 0, result.values.size()*sizeof(float));
            trace.records.push_back(std::move(result));
            return true;
        } catch (...) {
            trace.error = std::current_exception();
            return false;
        }
    }
private:
    size_t cursor = 0;
    size_t current = 0;
};

struct state_tensor {
    int32_t layer;
    state_kind kind;
    std::vector<float> values;
};

struct checkpoint {
    const llama_model * model;
    llama_pos next_pos;
    std::vector<uint8_t> bytes;
};

struct step {
    llama_seq_id seq;
    llama_pos pos;
    llama_token token;
    std::vector<float> logits;
};

class prefix_cache {
public:
    struct entry {
        std::vector<llama_token> tokens;
        checkpoint state;
        std::vector<float> logits;
    };
    explicit prefix_cache(size_t byte_budget) : budget(byte_budget) {}
    size_t size_bytes() const { return used; }
    std::shared_ptr<const entry> find(const llama_model * model, std::span<const llama_token> tokens) const {
        std::shared_ptr<const entry> result;
        for (const auto & item : entries) {
            if (item->state.model == model && item->tokens.size() <= tokens.size() &&
                (!result || item->tokens.size() > result->tokens.size()) &&
                std::equal(item->tokens.begin(), item->tokens.end(), tokens.begin())) result = item;
        }
        return result;
    }
    bool put(entry item) {
        if (item.state.next_pos != llama_pos(item.tokens.size())) throw std::runtime_error("prefix position mismatch");
        const size_t bytes = item.state.bytes.size() + item.tokens.size()*sizeof(llama_token) + item.logits.size()*sizeof(float);
        if (bytes > budget) return false;
        while (used > budget - bytes && !entries.empty()) {
            const auto & old = entries.front();
            used -= old->state.bytes.size() + old->tokens.size()*sizeof(llama_token) + old->logits.size()*sizeof(float);
            entries.erase(entries.begin());
        }
        entries.push_back(std::make_shared<const entry>(std::move(item)));
        used += bytes;
        return true;
    }
private:
    size_t budget;
    size_t used = 0;
    std::vector<std::shared_ptr<const entry>> entries;
};

class stream {
public:
    struct promise_type {
        step value;
        std::exception_ptr error;
        stream get_return_object() { return stream{std::coroutine_handle<promise_type>::from_promise(*this)}; }
        std::suspend_always initial_suspend() noexcept { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        std::suspend_always yield_value(step result) { value = std::move(result); return {}; }
        void return_void() {}
        void unhandled_exception() { error = std::current_exception(); }
    };
    explicit stream(std::coroutine_handle<promise_type> handle) : handle(handle) {}
    stream(const stream &) = delete;
    stream(stream && other) noexcept : handle(std::exchange(other.handle, {})) {}
    ~stream() { if (handle) handle.destroy(); }
    bool next() {
        if (!handle || handle.done()) return false;
        handle.resume();
        if (handle.promise().error) std::rethrow_exception(handle.promise().error);
        return !handle.done();
    }
    const step & value() const { return handle.promise().value; }
private:
    std::coroutine_handle<promise_type> handle;
};

// The context and model must outlive the session and its streams.
class session {
public:
    explicit session(llama_context * ctx) : ctx(ctx), model(ctx ? llama_get_model(ctx) : nullptr), embd(model ? llama_model_n_embd(model) : 0) {
        if (!ctx || !model) throw std::runtime_error("null context or model");
        char arch[32];
        llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch));
        if (std::string(arch) != "rwkv6" && std::string(arch) != "rwkv7") throw std::runtime_error("expected RWKV6/7");
        mem = dynamic_cast<llama_memory_recurrent *>(llama_get_memory(ctx));
        if (!mem || embd <= 0 || mem->s_l.empty()) throw std::runtime_error("expected pure RWKV recurrent memory");
        if (mem->n_rs_seq != 0) throw std::runtime_error("intervention sessions require n_rs_seq = 0; use explicit checkpoints");
        if (!mem->s_l.front()) throw std::runtime_error("missing state layer");
        head_size = mem->s_l.front()->ne[0]/embd;
        if (head_size <= 0) throw std::runtime_error("invalid head size");
        for (size_t i = 0; i < mem->s_l.size(); ++i) {
            if (!mem->s_l[i] || !mem->r_l[i] || mem->s_l[i]->ne[0] != int64_t(embd)*head_size || mem->r_l[i]->ne[0] != 2*int64_t(embd) ||
                mem->s_l[i]->type != GGML_TYPE_F32 || mem->r_l[i]->type != GGML_TYPE_F32) {
                throw std::runtime_error("unsupported RWKV state layout; use F32 state");
            }
        }
    }

    llama_pos position(llama_seq_id seq) const {
        check_seq(seq);
        return llama_memory_seq_pos_max(llama_get_memory(ctx), seq) + 1;
    }

    checkpoint capture(llama_seq_id seq) {
        check_seq(seq);
        llama_synchronize(ctx);
        checkpoint result{model, position(seq), {}};
        result.bytes.resize(llama_state_seq_get_size(ctx, seq));
        if (llama_state_seq_get_data(ctx, result.bytes.data(), result.bytes.size(), seq) != result.bytes.size()) {
            throw std::runtime_error("checkpoint capture failed");
        }
        return result;
    }

    void restore(llama_seq_id seq, const checkpoint & state) {
        check_seq(seq);
        if (state.model != model || state.bytes.empty()) throw std::runtime_error("incompatible checkpoint");
        if (llama_state_seq_set_data(ctx, state.bytes.data(), state.bytes.size(), seq) != state.bytes.size()) {
            throw std::runtime_error("checkpoint restore failed");
        }
        if (position(seq) != state.next_pos) throw std::runtime_error("checkpoint position mismatch");
    }

    state_tensor capture(llama_seq_id seq, int32_t layer, state_kind kind) {
        llama_synchronize(ctx);
        const auto view = locate(seq, layer, kind);
        state_tensor result{layer, kind, std::vector<float>(view.count)};
        capture_into(seq, layer, kind, result.values);
        return result;
    }

    void capture_into(llama_seq_id seq, int32_t layer, state_kind kind, std::span<float> destination) {
        llama_synchronize(ctx);
        const auto view = locate(seq, layer, kind);
        if (destination.size() != view.count) throw std::runtime_error("capture shape mismatch");
        ggml_backend_tensor_get(view.tensor, destination.data(), view.offset, view.count*sizeof(float));
    }

    void replace(llama_seq_id seq, std::span<const state_tensor> edits) {
        if (edits.empty()) return;
        llama_synchronize(ctx);
        for (const auto & edit : edits) {
            const auto view = locate(seq, edit.layer, edit.kind);
            if (edit.values.size() != view.count) throw std::runtime_error("replacement shape mismatch");
        }
        // Materialize a private current row before writes to shared or rollback state.
        check_seq(seq);
        const int32_t tail = mem->cells[seq].tail;
        const auto & cell = mem->cells[tail];
        if (cell.seq_id.size() > 1 || mem->rs_idx[seq] != 0 || (cell.src >= 0 && cell.src != tail)) {
            const auto saved = capture(seq);
            restore(seq, saved);
        }
        for (const auto & edit : edits) {
            const auto view = locate(seq, edit.layer, edit.kind);
            ggml_backend_tensor_set(view.tensor, edit.values.data(), view.offset, view.count*sizeof(float));
        }
        // Old rollback planes describe the unedited trajectory.
        mem->set_rs_idx(seq, 0);
    }

    struct request { llama_seq_id seq; std::vector<llama_token> tokens; };

    std::vector<step> prefill(std::span<const request> requests) {
        std::vector<size_t> consumed(requests.size(), 0);
        std::vector<step> final(requests.size());
        for (size_t i = 0; i < requests.size(); ++i) {
            check_seq(requests[i].seq);
            if (requests[i].tokens.empty()) throw std::runtime_error("empty prefill");
            if (requests[i].tokens.size() > size_t(std::numeric_limits<llama_pos>::max() - position(requests[i].seq))) throw std::runtime_error("position overflow");
            for (size_t j = 0; j < i; ++j) {
                if (requests[j].seq == requests[i].seq) throw std::runtime_error("duplicate sequence");
            }
        }
        while (true) {
            size_t capacity = llama_n_batch(ctx);
            std::vector<request> chunk;
            std::vector<size_t> mapping;
            for (size_t i = 0; i < requests.size() && capacity; ++i) {
                const size_t count = std::min(capacity, requests[i].tokens.size() - consumed[i]);
                if (!count) continue;
                const auto first = requests[i].tokens.begin() + consumed[i];
                chunk.push_back({requests[i].seq, {first, first + count}});
                mapping.push_back(i);
                consumed[i] += count;
                capacity -= count;
            }
            if (chunk.empty()) return final;
            auto outputs = prefill_batch(chunk);
            for (size_t i = 0; i < outputs.size(); ++i) final[mapping[i]] = std::move(outputs[i]);
        }
    }

    step prefill_cached(llama_seq_id seq, std::vector<llama_token> tokens, prefix_cache & cache) {
        check_seq(seq);
        if (tokens.empty()) throw std::runtime_error("empty prefix");
        const auto hit = cache.find(model, tokens);
        size_t consumed = 0;
        if (hit) {
            restore(seq, hit->state);
            consumed = hit->tokens.size();
            if (consumed == tokens.size()) return {seq, position(seq) - 1, tokens.back(), hit->logits};
        } else {
            if (!llama_memory_seq_rm(llama_get_memory(ctx), seq, 0, -1)) throw std::runtime_error("sequence reset failed");
        }
        request tail{seq, {tokens.begin() + consumed, tokens.end()}};
        auto output = prefill(std::span(&tail, 1)).front();
        cache.put({std::move(tokens), capture(seq), output.logits});
        return output;
    }

private:
    std::vector<step> prefill_batch(std::span<const request> requests) {
        size_t count = 0;
        for (const auto & r : requests) {
            check_seq(r.seq);
            if (r.tokens.empty()) throw std::runtime_error("empty prefill");
            for (const auto & other : requests) {
                if (&other != &r && other.seq == r.seq) throw std::runtime_error("duplicate sequence");
            }
            count += r.tokens.size();
        }
        if (count == 0 || count > llama_n_batch(ctx)) throw std::runtime_error("prefill exceeds batch capacity");
        std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)> batch(llama_batch_ext_init(ctx), llama_batch_ext_free);
        if (!batch) throw std::runtime_error("batch allocation failed");
        std::vector<step> results;
        std::vector<int32_t> output_indices;
        for (const auto & r : requests) {
            const auto start = position(r.seq);
            for (size_t j = 0; j < r.tokens.size(); ++j) {
                const int32_t i = llama_batch_ext_add_token(batch.get(), r.seq, r.tokens[j]);
                const llama_pos pos = start + j;
                if (j + 1 == r.tokens.size()) output_indices.push_back(i);
                if (i < 0 || !llama_batch_ext_set_pos(batch.get(), i, &pos) ||
                    !llama_batch_ext_set_output_logits(batch.get(), i, j + 1 == r.tokens.size())) {
                    throw std::runtime_error("invalid batch token");
                }
            }
            results.push_back({r.seq, start + llama_pos(r.tokens.size()) - 1, r.tokens.back(), {}});
        }
        const int rc = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
        if (rc != 0) throw std::runtime_error("decode failed: " + std::to_string(rc));
        const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        for (size_t i = 0; i < results.size(); ++i) {
            const float * logits = llama_get_logits_ith(ctx, output_indices[i]);
            if (!logits) throw std::runtime_error("missing logits");
            results[i].logits.assign(logits, logits + n_vocab);
        }
        return results;
    }

public:
    stream run(llama_seq_id seq, std::vector<llama_token> tokens) {
        for (auto token : tokens) {
            const request r{seq, {token}};
            auto results = prefill(std::span(&r, 1));
            co_yield std::move(results.front());
        }
    }

private:
    struct view { ggml_tensor * tensor; size_t offset; size_t count; };
    llama_context * ctx;
    const llama_model * model;
    llama_memory_recurrent * mem;
    int32_t embd;
    int32_t head_size;

    void check_seq(llama_seq_id seq) const {
        if (seq < 0 || size_t(seq) >= mem->rs_idx.size()) throw std::runtime_error("invalid sequence");
    }

    view locate(llama_seq_id seq, int32_t layer, state_kind kind) const {
        check_seq(seq);
        if (layer < 0 || size_t(layer) >= mem->s_l.size()) throw std::runtime_error("invalid layer");
        const int32_t tail = mem->cells[seq].tail;
        if (tail < 0) throw std::runtime_error("sequence has no state; prefill first");
        const auto & cell = mem->cells[tail];
        const size_t row = size_t(mem->rs_idx[seq])*mem->size + (cell.src >= 0 ? cell.src : tail);
        if (kind != state_kind::wkv && kind != state_kind::shift_att && kind != state_kind::shift_ffn) throw std::runtime_error("invalid state kind");
        auto * tensor = kind == state_kind::wkv ? mem->s_l[layer] : mem->r_l[layer];
        const size_t count = kind == state_kind::wkv ? size_t(embd)*head_size : size_t(embd);
        const size_t offset = row*tensor->nb[1] + (kind == state_kind::shift_ffn ? size_t(embd)*sizeof(float) : 0);
        if (offset > ggml_nbytes(tensor) || count > (ggml_nbytes(tensor) - offset)/sizeof(float)) throw std::runtime_error("state row outside tensor");
        return {tensor, offset, count};
    }
};

}
