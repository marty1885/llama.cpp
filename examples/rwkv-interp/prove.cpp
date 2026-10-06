#include "rwkv-session.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>

static void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
    std::printf("PASS %s\n", message);
    std::fflush(stdout);
}

static double difference(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) throw std::runtime_error("shape mismatch");
    double result = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) throw std::runtime_error("nonfinite result");
        result = std::max(result, std::abs(double(a[i]) - b[i]));
    }
    return result;
}


int main(int argc, char ** argv) {
    try {
        if (argc != 2 && !(argc == 3 && std::string(argv[2]) == "--vulkan")) throw std::runtime_error("usage: prove MODEL.gguf [--vulkan]");
        const bool vulkan = argc == 3;
        if (vulkan) {
#ifdef RWKV_INTERP_BACKEND_DIR
            ggml_backend_load_all_from_path(RWKV_INTERP_BACKEND_DIR);
#else
            ggml_backend_load_all();
#endif
        }
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = vulkan ? 99 : 0;
        ggml_backend_dev_t devices[] = {vulkan ? ggml_backend_dev_by_name("Vulkan0") : nullptr, nullptr};
        if (vulkan) {
            if (!devices[0]) throw std::runtime_error("Vulkan0 unavailable");
            mp.devices = devices;
        }
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], mp), llama_model_free);
        if (!model) throw std::runtime_error("model load failed");
        char arch[64];
        llama_model_meta_val_str(model.get(), "general.architecture", arch, sizeof(arch));
        const std::string architecture(arch);
        if (architecture != "rwkv7" && architecture != "rwkv6") throw std::runtime_error("expected RWKV6/7");
        char value[64];
        llama_model_meta_val_str(model.get(), (architecture + ".wkv.head_size").c_str(), value, sizeof(value));
        require(std::stoi(value) > 0, "model head size is valid");
        rwkv::wkv_trace act(llama_model_n_layer(model.get()));
        auto cp = llama_context_default_params();
        cp.n_ctx = 128;
        cp.n_batch = 32;
        cp.n_ubatch = 32;
        cp.n_seq_max = 4;
        cp.n_rs_seq = 0;
        cp.n_threads = 8;
        cp.n_threads_batch = 8;
        cp.type_k = GGML_TYPE_F32;
        cp.type_v = GGML_TYPE_F32;
        cp.cb_eval = rwkv::wkv_trace::callback;
        cp.cb_eval_user_data = &act;
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
        if (!ctx) throw std::runtime_error("context init failed");
        rwkv::session s(ctx.get());
        using request = rwkv::session::request;
        const auto decode = [&](int seq, std::vector<llama_token> tokens) {
            request r{seq, std::move(tokens)};
            act.clear();
            return s.prefill(std::span(&r, 1)).front().logits;
        };
        const auto begin = std::chrono::steady_clock::now();
        const auto prefix_logits = decode(0, {1, 2, 3});
        const auto prefix = s.capture(0);
        require(prefix.next_pos == 3, "prefill advances position");
        require(act.records.size() == size_t(llama_model_n_layer(model.get())), "transient WKV captured for every layer");
        require(act.records.front().values.size() == size_t(3*llama_model_n_embd(model.get())), "transient WKV contains every prefill token");
        std::vector<rwkv::state_tensor> states;
        for (auto kind : {rwkv::state_kind::wkv, rwkv::state_kind::shift_att, rwkv::state_kind::shift_ffn}) states.push_back(s.capture(0, 0, kind));
        const auto reference = decode(0, {4});
        s.restore(0, prefix);
        require(difference(reference, decode(0, {4})) == 0, "checkpoint continuation reproduces logits exactly");
        s.restore(0, prefix);
        s.replace(0, states);
        require(difference(reference, decode(0, {4})) == 0, "no-op selective replacement reproduces logits exactly");
        for (const auto & state : states) {
            s.restore(0, prefix);
            auto edit = state;
            std::fill(edit.values.begin(), edit.values.end(), 0.0f);
            s.replace(0, std::span(&edit, 1));
            require(difference(edit.values, s.capture(0, edit.layer, edit.kind).values) == 0, "replacement readback equals supplied values");
            const auto changed = decode(0, {4});
            const auto delta = difference(reference, changed);
            std::printf("intervention kind=%d max_logit_delta=%g\n", int(edit.kind), delta);
            require(delta > 0, "selective state intervention changes continuation");
        }
        s.restore(0, prefix);
        auto invalid = states.front();
        invalid.values.pop_back();
        bool rejected = false;
        try { s.replace(0, std::span(&invalid, 1)); } catch (const std::runtime_error &) { rejected = true; }
        require(rejected && s.capture(0).bytes == prefix.bytes, "invalid replacement leaves checkpoint unchanged");
        s.restore(1, prefix);
        const std::vector<request> batch{{0, {4}}, {1, {4}}};
        act.clear();
        const auto batched = s.prefill(batch);
        const double batch_delta = difference(reference, batched[0].logits);
        std::printf("batch_vs_single_max_delta=%g\n", batch_delta);
        double squared_error = 0, squared_ref = 0;
        for (size_t i = 0; i < reference.size(); ++i) {
            const double d = double(reference[i]) - batched[0].logits[i];
            squared_error += d*d;
            squared_ref += double(reference[i])*reference[i];
        }
        std::printf("batch_pair_delta=%g batch_nmse=%g\n", difference(batched[0].logits, batched[1].logits), squared_error/squared_ref);
        require(difference(batched[0].logits, batched[1].logits) == 0, "batched identical prefix branches match exactly");
        s.restore(0, prefix);
        s.restore(1, prefix);
        auto native = llama_batch_ext_init(ctx.get());
        for (llama_seq_id seq : {0, 1}) {
            const auto index = llama_batch_ext_add_token(native, seq, 4);
            const llama_pos pos = 3;
            llama_batch_ext_set_pos(native, index, &pos);
            llama_batch_ext_set_output_logits(native, index, true);
        }
        require(llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, native) == 0, "native batch control decodes");
        llama_batch_ext_free(native);
        std::vector<float> native_logits(llama_get_logits_ith(ctx.get(), 0), llama_get_logits_ith(ctx.get(), 0) + reference.size());
        require(difference(native_logits, batched[0].logits) == 0, "wrapper batch equals native API exactly");
        s.restore(0, prefix);
        s.restore(1, prefix);
        const auto replay_batch = s.prefill(batch);
        require(difference(batched[0].logits, replay_batch[0].logits) == 0, "batched prefix replay matches exactly");
        s.restore(0, prefix);
        s.restore(1, prefix);
        const std::vector<request> divergent{{0, {5}}, {1, {4}}};
        const auto independent = s.prefill(divergent);
        require(difference(independent[1].logits, batched[1].logits) == 0, "batched sequence continuation is independent of other sequence token");
        s.restore(0, prefix);
        s.restore(1, prefix);
        const std::vector<request> ragged{{1, {4, 5}}, {0, {4}}};
        const auto uneven = s.prefill(ragged);
        require(uneven[0].seq == 1 && uneven[0].pos == 4 && uneven[1].seq == 0 && uneven[1].pos == 3, "ragged batch outputs retain request identities and positions");
        difference(uneven[0].logits, uneven[0].logits);
        difference(uneven[1].logits, uneven[1].logits);
        s.restore(0, prefix);
        s.restore(1, prefix);
        const auto branch1 = s.capture(1);
        auto edit = states.front();
        std::fill(edit.values.begin(), edit.values.end(), 0.0f);
        s.replace(0, std::span(&edit, 1));
        require(s.capture(1).bytes == branch1.bytes, "branch intervention leaves other prefix branch unchanged");
        llama_memory_seq_cp(llama_get_memory(ctx.get()), 1, 2, -1, -1);
        s.replace(2, std::span(&edit, 1));
        require(s.capture(1).bytes == branch1.bytes, "replacement detaches a shared recurrent row");
        s.restore(0, prefix);
        auto continuation = s.run(0, {4, 5});
        require(continuation.next() && difference(reference, continuation.value().logits) == 0, "coroutine first yield matches synchronous decode");
        s.restore(0, prefix);
        require(continuation.next() && continuation.value().pos == 3, "state replacement between coroutine yields takes effect");
        require(!continuation.next(), "coroutine completes");
        s.restore(0, prefix);
        { auto cancelled = s.run(0, {4, 5}); require(cancelled.next(), "coroutine starts before cancellation"); }
        require(s.position(0) == 4, "coroutine cancellation performs no extra decode");
        rwkv::prefix_cache cache(2*prefix.bytes.size() + 4*reference.size()*sizeof(float));
        const auto cached_first = s.prefill_cached(3, {1, 2, 3}, cache);
        act.clear();
        const auto cached_hit = s.prefill_cached(3, {1, 2, 3}, cache);
        require(difference(cached_first.logits, cached_hit.logits) == 0 && act.records.empty(), "exact prefix cache hit performs no model evaluation");
        const auto cached_tail = s.prefill_cached(3, {1, 2, 3, 4}, cache);
        require(difference(reference, cached_tail.logits) == 0, "longest prefix cache reuse decodes only suffix");
        require(cache.size_bytes() <= 2*prefix.bytes.size() + 4*reference.size()*sizeof(float), "prefix cache respects byte budget");
        s.restore(3, prefix);
        const auto long_output = decode(3, std::vector<llama_token>(35, 4));
        require(s.position(3) == 38 && !long_output.empty(), "prefill chunks prompts larger than batch capacity");
        act.selected_layers = {0};
        decode(3, {5});
        require(act.records.size() == 1 && act.records[0].layer == 0, "transient capture selects one layer");
        std::vector<float> reuse(states.front().values.size());
        s.capture_into(3, 0, rwkv::state_kind::s, reuse);
        require(difference(reuse, s.capture(3, 0, rwkv::state_kind::wkv).values) == 0, "S aliases persistent WKV and supports reusable capture buffers");
        const auto capture_start = std::chrono::steady_clock::now();
        for (int i = 0; i < 20; ++i) s.capture_into(3, 0, rwkv::state_kind::s, reuse);
        const double capture_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - capture_start).count()/20;
        std::printf("selected_state_bytes=%zu capture_into_mean_us=%.3f\n", reuse.size()*sizeof(float), capture_us);
        std::printf("checkpoint_bytes=%zu elapsed_seconds=%.3f\n", prefix.bytes.size(), std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count());
        std::puts("ALL CHECKS PASSED");
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL %s\n", error.what());
        return 1;
    }
}
