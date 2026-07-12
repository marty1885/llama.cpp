#pragma once

#include "common.h"
#include "llama-ext.h"

#include <coroutine>
#include <deque>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace llama_interp {

using rwkv_state = llama_interp_rwkv_state;
using activation_set = llama_interp_activation_set;

struct decode_result {
    std::vector<llama_token> tokens;
    rwkv_state state;
    const llama_context * ctx = nullptr;

    std::string to_string() const {
        std::string out;
        for (llama_token tok : tokens) {
            out += common_token_to_piece(ctx, tok, true);
        }
        return out;
    }
};

template <typename T = void>
struct task;

template <>
struct task<void> {
    struct promise_type {
        std::exception_ptr error;

        task get_return_object() { return task(std::coroutine_handle<promise_type>::from_promise(*this)); }
        std::suspend_never initial_suspend() noexcept { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        void return_void() noexcept {}
        void unhandled_exception() { error = std::current_exception(); }
    };

    explicit task(std::coroutine_handle<promise_type> h) : h(h) {}
    task(task && other) noexcept : h(std::exchange(other.h, {})) {}
    task(const task &) = delete;
    ~task() { if (h) h.destroy(); }

    void rethrow_if_failed() const {
        if (h && h.promise().error) {
            std::rethrow_exception(h.promise().error);
        }
    }

    std::coroutine_handle<promise_type> h;
};

class runtime {
public:
    runtime(llama_context * ctx, uint32_t max_parallel_experiments)
        : ctx(ctx), max_parallel_experiments(max_parallel_experiments) {
        if (!ctx || max_parallel_experiments == 0) {
            throw std::runtime_error("invalid interp runtime");
        }
        batch = llama_batch_init((int32_t) max_parallel_experiments, 0, 1);
    }

    ~runtime() {
        llama_batch_free(batch);
    }

    rwkv_state make_state() const {
        rwkv_state state;
        if (!llama_interp_rwkv_state_init(ctx, &state)) {
            throw std::runtime_error("failed to initialize RWKV interp state");
        }
        return state;
    }

    class prefill_op;
    class decode_op;

    prefill_op prefill(const rwkv_state & state, std::string text);
    prefill_op prefill_tokens(const rwkv_state & state, std::vector<llama_token> tokens);
    decode_op decode(const rwkv_state & state, int32_t n_tokens);

    void run();

private:
    enum class kind { prefill, decode };

    struct op_base {
        op_base(runtime & rt, kind op_kind, const rwkv_state & state) : rt(rt), op_kind(op_kind), input(state) {}
        virtual ~op_base() = default;

        bool await_ready() const noexcept { return false; }
        void await_suspend(std::coroutine_handle<> h) {
            continuation = h;
            rt.queue.push_back(this);
        }

        bool has_perturb() const { return !request.perturbations.empty(); }

        void resume() {
            if (continuation) {
                continuation.resume();
            }
        }

        runtime & rt;
        kind op_kind;
        rwkv_state input;
        llama_interp_request request;
        std::exception_ptr error;
        std::coroutine_handle<> continuation;
    };

public:
    class prefill_op : public op_base {
    public:
        prefill_op(runtime & rt, const rwkv_state & state, std::string text)
            : op_base(rt, kind::prefill, state), text(std::move(text)) {
            tokens = common_tokenize(rt.ctx, this->text, false, true);
        }

        prefill_op(runtime & rt, const rwkv_state & state, std::vector<llama_token> tokens)
            : op_base(rt, kind::prefill, state), tokens(std::move(tokens)) {}

        prefill_op & capture(std::string regex, activation_set & dst) {
            request.captures.push_back({std::move(regex), &dst, true});
            return *this;
        }

        prefill_op & capture_f16(std::string regex, activation_set & dst) {
            request.captures.push_back({std::move(regex), &dst, false});
            return *this;
        }

        prefill_op & capture_f32(std::string regex, activation_set & dst) {
            request.captures.push_back({std::move(regex), &dst, true});
            return *this;
        }

        prefill_op & discard_state() {
            export_state = false;
            return *this;
        }

        prefill_op & perturb(std::string regex, llama_interp_perturb_spec spec) {
            spec.regex = std::move(regex);
            request.perturbations.push_back(std::move(spec));
            return *this;
        }

        rwkv_state await_resume() {
            if (error) std::rethrow_exception(error);
            return output;
        }

        std::string text;
        std::vector<llama_token> tokens;
        rwkv_state output;
        bool export_state = true;
    };

    class decode_op : public op_base {
    public:
        decode_op(runtime & rt, const rwkv_state & state, int32_t n_tokens)
            : op_base(rt, kind::decode, state), n_tokens(n_tokens) {}

        decode_op & capture(std::string regex, activation_set & dst) {
            request.captures.push_back({std::move(regex), &dst, true});
            return *this;
        }

        decode_op & capture_f16(std::string regex, activation_set & dst) {
            request.captures.push_back({std::move(regex), &dst, false});
            return *this;
        }

        decode_op & capture_f32(std::string regex, activation_set & dst) {
            request.captures.push_back({std::move(regex), &dst, true});
            return *this;
        }

        decode_op & perturb(std::string regex, llama_interp_perturb_spec spec) {
            spec.regex = std::move(regex);
            request.perturbations.push_back(std::move(spec));
            return *this;
        }

        decode_result await_resume() {
            if (error) std::rethrow_exception(error);
            return output;
        }

        int32_t n_tokens;
        decode_result output;
    };

    static llama_interp_perturb_spec add_head(int32_t head, std::vector<ggml_fp16_t> data) {
        llama_interp_perturb_spec spec;
        spec.op = LLAMA_INTERP_PERTURB_ADD;
        spec.head = head;
        spec.data = std::move(data);
        return spec;
    }

private:
    static llama_token greedy_token(llama_context * ctx, int32_t row) {
        const llama_model * model = llama_get_model(ctx);
        const llama_vocab * vocab = llama_model_get_vocab(model);
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);
        const float * logits = llama_get_logits_ith(ctx, row);

        llama_token best = 0;
        float best_v = logits[0];
        for (int32_t i = 1; i < n_vocab; ++i) {
            if (logits[i] > best_v) {
                best_v = logits[i];
                best = i;
            }
        }
        return best;
    }

    llama_interp_request make_request(const std::vector<op_base *> & group, bool perturb, bool capture) {
        llama_interp_request req;
        req.id = ++request_id;
        req.enable_perturbations = perturb;
        req.enable_captures = capture;
        for (op_base * op : group) {
            if (perturb) {
                req.perturbations.insert(req.perturbations.end(), op->request.perturbations.begin(), op->request.perturbations.end());
            }
            if (capture) {
                req.captures.insert(req.captures.end(), op->request.captures.begin(), op->request.captures.end());
            }
        }
        return req;
    }

    void import_group(const std::vector<op_base *> & group) {
        for (size_t i = 0; i < group.size(); ++i) {
            if (!llama_interp_rwkv_state_import(ctx, (llama_seq_id) i, &group[i]->input)) {
                throw std::runtime_error("failed to import RWKV interp state");
            }
        }
    }

    void decode_step(const std::vector<op_base *> & group, const std::vector<llama_token> & tokens, bool perturb, bool capture) {
        llama_batch_free(batch);
        batch = llama_batch_init((int32_t) group.size(), 0, 1);
        common_batch_clear(batch);

        for (size_t i = 0; i < group.size(); ++i) {
            const llama_pos pos = group[i]->input.pos + 1;
            common_batch_add(batch, tokens[i], pos, { (llama_seq_id) i }, true);
        }

        auto req = make_request(group, perturb, capture);
        const bool active_request = (req.enable_perturbations && !req.perturbations.empty()) ||
                                    (req.enable_captures      && !req.captures.empty());
        llama_interp_set_request(ctx, active_request ? &req : nullptr);
        if (llama_decode(ctx, batch) != 0) {
            throw std::runtime_error("llama_decode failed");
        }
        llama_interp_set_request(ctx, nullptr);
    }

    void run_prefill(const std::vector<op_base *> & base_group) {
        std::vector<op_base *> group = base_group;
        import_group(group);

        auto * first = static_cast<prefill_op *>(group.front());
        const size_t n = first->tokens.size();
        if (n == 0) {
            throw std::runtime_error("prefill text tokenized to zero tokens");
        }

        for (size_t t = 0; t < n; ++t) {
            std::vector<llama_token> toks(group.size());
            for (size_t i = 0; i < group.size(); ++i) {
                toks[i] = static_cast<prefill_op *>(group[i])->tokens[t];
            }
            decode_step(group, toks, t == 0, t + 1 == n);
            for (op_base * op : group) {
                op->input.pos += 1;
            }
        }

        for (size_t i = 0; i < group.size(); ++i) {
            auto * op = static_cast<prefill_op *>(group[i]);
            if (op->export_state) {
                if (!llama_interp_rwkv_state_export(ctx, (llama_seq_id) i, &op->output)) {
                    throw std::runtime_error("failed to export RWKV interp state");
                }
                op->output.has_next = true;
                op->output.next_token = greedy_token(ctx, (int32_t) i);
            }
        }
    }

    void run_decode(const std::vector<op_base *> & base_group) {
        std::vector<op_base *> group = base_group;
        import_group(group);

        int32_t n = static_cast<decode_op *>(group.front())->n_tokens;
        for (op_base * op_base : group) {
            auto * op = static_cast<decode_op *>(op_base);
            if (op->n_tokens != n || !op->input.has_next) {
                throw std::runtime_error("decode requires equal n_tokens and a state with next_token");
            }
            op->output.ctx = ctx;
            op->output.state = op->input;
        }

        std::vector<llama_token> toks(group.size());
        for (int32_t t = 0; t < n; ++t) {
            for (size_t i = 0; i < group.size(); ++i) {
                auto * op = static_cast<decode_op *>(group[i]);
                toks[i] = op->output.state.next_token;
                op->output.tokens.push_back(toks[i]);
            }

            decode_step(group, toks, t == 0, t + 1 == n);

            for (size_t i = 0; i < group.size(); ++i) {
                auto * op = static_cast<decode_op *>(group[i]);
                if (!llama_interp_rwkv_state_export(ctx, (llama_seq_id) i, &op->output.state)) {
                    throw std::runtime_error("failed to export RWKV interp state");
                }
                op->output.state.has_next = true;
                op->output.state.next_token = greedy_token(ctx, (int32_t) i);
                group[i]->input = op->output.state;
            }
        }
    }

    llama_context * ctx;
    uint32_t max_parallel_experiments;
    llama_batch batch;
    uint64_t request_id = 0;
    std::deque<op_base *> queue;
};

inline runtime::prefill_op runtime::prefill(const rwkv_state & state, std::string text) {
    return prefill_op(*this, state, std::move(text));
}

inline runtime::prefill_op runtime::prefill_tokens(const rwkv_state & state, std::vector<llama_token> tokens) {
    return prefill_op(*this, state, std::move(tokens));
}

inline runtime::decode_op runtime::decode(const rwkv_state & state, int32_t n_tokens) {
    return decode_op(*this, state, n_tokens);
}

inline void runtime::run() {
    while (!queue.empty()) {
        op_base * first = queue.front();
        const kind k = first->op_kind;

        std::vector<op_base *> group;
        for (auto it = queue.begin(); it != queue.end() && group.size() < max_parallel_experiments;) {
            op_base * op = *it;
            if (op->op_kind != k || (op->has_perturb() && !group.empty()) || (!group.empty() && group.front()->has_perturb())) {
                ++it;
                continue;
            }
            if (k == kind::prefill && static_cast<prefill_op *>(op)->tokens.size() != static_cast<prefill_op *>(first)->tokens.size()) {
                ++it;
                continue;
            }
            group.push_back(op);
            it = queue.erase(it);
        }

        try {
            if (k == kind::prefill) {
                run_prefill(group);
            } else {
                run_decode(group);
            }
        } catch (...) {
            for (op_base * op : group) {
                op->error = std::current_exception();
            }
        }

        for (op_base * op : group) {
            op->resume();
        }
    }
    llama_interp_set_request(ctx, nullptr);
}

} // namespace llama_interp
