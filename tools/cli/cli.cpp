#include "chat.h"
#include "common.h"
#include "arg.h"
#include "console.h"
// #include "log.h"

#include "server-context.h"
#include "server-task.h"

#include "llama.h"

#include <array>
#include <atomic>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>
#include <thread>
#include <signal.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

const char * LLAMA_ASCII_LOGO = R"(
▄▄ ▄▄
██ ██
██ ██  ▀▀█▄ ███▄███▄  ▀▀█▄    ▄████ ████▄ ████▄
██ ██ ▄█▀██ ██ ██ ██ ▄█▀██    ██    ██ ██ ██ ██
██ ██ ▀█▄██ ██ ██ ██ ▀█▄██ ██ ▀████ ████▀ ████▀
                                    ██    ██
                                    ▀▀    ▀▀
)";

// ── /rwkv-edit helpers ──────────────────────────────────────────────────────

// parse shell-style quoted arguments: /rwkv-edit "from" "to" [alpha] [method] ["layers"]
static std::vector<std::string> parse_shell_args(const std::string & s) {
    std::vector<std::string> args;
    std::size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && s[i] == ' ') ++i;
        if (i >= s.size()) break;

        if (s[i] == '"') {
            ++i; // skip opening quote
            std::string arg;
            while (i < s.size() && s[i] != '"') {
                if (s[i] == '\\' && i + 1 < s.size()) { arg += s[++i]; }
                else { arg += s[i]; }
                ++i;
            }
            if (i < s.size()) ++i; // skip closing quote
            args.push_back(std::move(arg));
        } else {
            std::size_t start = i;
            while (i < s.size() && s[i] != ' ') ++i;
            args.push_back(s.substr(start, i - start));
        }
    }
    return args;
}

static std::vector<int> parse_layer_range(const std::string & s, int n_layer) {
    std::vector<int> out;
    auto dash = s.find('-');
    if (dash != std::string::npos) {
        int lo = std::atoi(s.substr(0, dash).c_str());
        int hi = std::atoi(s.substr(dash + 1).c_str());
        for (int i = lo; i <= hi && i < n_layer; ++i) out.push_back(i);
    } else {
        out.push_back(std::atoi(s.c_str()));
    }
    return out;
}

// rank-1 decomposition via power iteration: returns σ₁, u₁, v₁
struct Rank1Decomp {
    float sigma = 0.0f;
    std::vector<float> u; // left singular vector (value direction)
    std::vector<float> v; // right singular vector (address direction)
};

static Rank1Decomp power_iter_decomp(const float * mat, int hs, int iters = 20) {
    Rank1Decomp d;
    d.u.resize(hs, 0.0f);
    d.v.assign(hs, 1.0f / std::sqrt((float)hs));

    for (int it = 0; it < iters; ++it) {
        // u = M * v
        for (int r = 0; r < hs; ++r) {
            float s = 0.0f;
            for (int c = 0; c < hs; ++c) s += mat[r * hs + c] * d.v[c];
            d.u[r] = s;
        }
        float norm_u = 0.0f;
        for (int r = 0; r < hs; ++r) norm_u += d.u[r] * d.u[r];
        norm_u = std::sqrt(norm_u);
        if (norm_u < 1e-12f) return d;
        for (int r = 0; r < hs; ++r) d.u[r] /= norm_u;

        // v = M^T * u
        for (int c = 0; c < hs; ++c) {
            float s = 0.0f;
            for (int r = 0; r < hs; ++r) s += mat[r * hs + c] * d.u[r];
            d.v[c] = s;
        }
        float norm_v = 0.0f;
        for (int c = 0; c < hs; ++c) norm_v += d.v[c] * d.v[c];
        norm_v = std::sqrt(norm_v);
        if (norm_v < 1e-12f) return d;
        for (int c = 0; c < hs; ++c) d.v[c] /= norm_v;
    }
    // σ₁ ≈ ||M * v||
    float sigma_sq = 0.0f;
    for (int r = 0; r < hs; ++r) {
        float s = 0.0f;
        for (int c = 0; c < hs; ++c) s += mat[r * hs + c] * d.v[c];
        sigma_sq += s * s;
    }
    d.sigma = std::sqrt(sigma_sq);
    return d;
}

// back-compat wrapper
static float power_iter_sigma1(const float * mat, int hs, int iters = 20) {
    return power_iter_decomp(mat, hs, iters).sigma;
}

// Frobenius norm of an hs×hs matrix
static float frob_norm(const float * mat, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += mat[i] * mat[i];
    return std::sqrt(s);
}

// Precompute a weighted delta vector ready to be applied via on_prompt_done hook.
// Returns the per-element delta to add to the S state (already weighted per-head).
// Empty vector on failure.
static std::vector<float> precompute_rwkv_edit(
        llama_context * ctx, const llama_vocab * vocab,
        const std::string & from_prompt, const std::string & to_prompt,
        float alpha, const std::string & method,
        const std::vector<int> & layers) {
    const llama_model * mdl = llama_get_model(ctx);
    const int n_layer     = llama_model_n_layer(mdl);
    const int n_embd      = llama_model_n_embd(mdl);
    const int n_embd_r    = llama_model_n_embd_r(mdl);
    const int n_embd_s    = llama_model_n_embd_s(mdl);
    const int head_size   = (n_embd > 0) ? n_embd_s / n_embd : 0;
    const int n_head      = (head_size > 0) ? n_embd / head_size : 0;
    const int hs2         = head_size * head_size;

    if (n_embd_s == 0 || n_embd_r == 0) {
        fprintf(stderr, "[rwkv-edit] ERROR: model is not an SSM/recurrent model (no recurrent state)\n");
        return {};
    }

    if (n_head == 0 || hs2 == 0) {
        fprintf(stderr, "[rwkv-edit] ERROR: cannot determine head geometry (n_embd_s=%d n_embd=%d)\n",
                n_embd_s, n_embd);
        return {};
    }

    (void)n_embd_r;
    (void)ctx; // main context is NOT touched — we use a temporary one
    const llama_seq_id seq_id = 0;
    const size_t total_s = (size_t)n_layer * n_embd_s;

    // create a temporary context on the same model for captures
    // this avoids disturbing the main context's memory/position tracking
    llama_context_params tmp_params = llama_context_default_params();
    tmp_params.n_ctx   = 512; // small, just enough for the edit prompts
    tmp_params.n_batch = 512;
    llama_context * tmp_ctx = llama_init_from_model(const_cast<llama_model *>(mdl), tmp_params);
    if (!tmp_ctx) {
        fprintf(stderr, "[rwkv-edit] ERROR: failed to create temporary context\n");
        return {};
    }

    // helper: decode a prompt on tmp_ctx, capture S state
    auto capture_s = [&](const std::string & prompt) -> std::vector<float> {
        llama_memory_clear(llama_get_memory(tmp_ctx), true);

        auto toks = common_tokenize(vocab, prompt, true, false);
        llama_batch batch = llama_batch_init(std::max((int)toks.size(), 1), 0, 1);
        for (size_t i = 0; i < toks.size(); ++i) {
            common_batch_add(batch, toks[i], i, {seq_id}, false);
        }
        if (llama_decode(tmp_ctx, batch) != 0) {
            fprintf(stderr, "[rwkv-edit] ERROR: decode failed for prompt\n");
            llama_batch_free(batch);
            return {};
        }
        llama_batch_free(batch);

        std::vector<float> s(total_s);
        llama_synchronize(tmp_ctx);
        llama_recurrent_state_get_f32(tmp_ctx, seq_id, nullptr, s.data());
        return s;
    };

    fprintf(stderr, "[rwkv-edit] capturing states...\n");
    auto s_from = capture_s(from_prompt);
    if (s_from.empty()) { llama_free(tmp_ctx); return {}; }

    auto s_to = capture_s(to_prompt);
    if (s_to.empty()) { llama_free(tmp_ctx); return {}; }

    llama_free(tmp_ctx);

    // compute weighted delta — zero out heads/layers that shouldn't be edited
    std::vector<float> weighted_delta(total_s, 0.0f);

    int heads_edited = 0;
    for (int layer : layers) {
        if (layer < 0 || layer >= n_layer) continue;
        for (int h = 0; h < n_head; ++h) {
            size_t offset = (size_t)layer * n_embd_s + (size_t)h * hs2;
            const float * d_from = s_from.data() + offset;
            const float * d_to   = s_to.data()   + offset;

            // compute per-head delta
            std::vector<float> head_delta(hs2);
            for (int i = 0; i < hs2; ++i) {
                head_delta[i] = d_to[i] - d_from[i];
            }

            float w = alpha;

            if (method == "R1-WT" || method == "ISOLATE" || method == "R1-ADAPTIVE") {
                float s1   = power_iter_sigma1(head_delta.data(), head_size);
                float fnrm = frob_norm(head_delta.data(), hs2);
                float rank1ness = (fnrm > 1e-10f) ? s1 / fnrm : 0.0f;
                w = rank1ness * alpha;
            }

            if (std::abs(w) < 1e-10f) continue;

            float * dst = weighted_delta.data() + offset;
            for (int i = 0; i < hs2; ++i) {
                dst[i] = w * head_delta[i];
            }
            heads_edited++;
        }
    }

    fprintf(stderr, "[rwkv-edit] precomputed %s alpha=%.2f layers=%d-%d heads=%d\n",
            method.c_str(), alpha, layers.front(), layers.back(), heads_edited);
    return weighted_delta;
}

// ── HYBRID edit data: offline u₁/σ per head, online v₁ computed in hook ──────

struct HybridHeadCal {
    int    layer  = 0;
    int    head   = 0;
    size_t offset = 0;       // offset into S state flat array
    float  sigma  = 0.0f;    // offline σ₁
    std::vector<float> u;    // offline u₁ (value direction)
    std::vector<float> v;    // offline v₁ (address direction)
};

struct HybridEditData {
    std::vector<HybridHeadCal> heads;  // per-head offline calibration
    std::vector<int> layers;
    float alpha    = 1.0f;
    int n_embd_s   = 0;
    int head_size  = 0;
    int n_head     = 0;
    // for online probe: decode these from live state to get online v₁/σ
    std::string from_prompt;
    std::string to_prompt;
    // method variant: "HYB-U", "HYB-UV", "ONLINE"
    std::string method = "HYB-U";
};

// Online probe helper: decode from/to prompts from a live state snapshot
// in a temporary context, returning the two post-decode S states.
struct OnlineProbeResult {
    std::vector<float> s_from;
    std::vector<float> s_to;
};

static OnlineProbeResult online_probe(
        const llama_model * mdl, const llama_vocab * vocab,
        const std::vector<float> & s_live,
        const std::string & from_prompt, const std::string & to_prompt) {
    const int n_layer  = llama_model_n_layer(mdl);
    const int n_embd_s = llama_model_n_embd_s(mdl);
    const size_t total_s = (size_t)n_layer * n_embd_s;

    auto toks_from = common_tokenize(vocab, from_prompt, false, false);
    auto toks_to   = common_tokenize(vocab, to_prompt,   false, false);
    int max_toks = (int)std::max(toks_from.size(), toks_to.size()) + 2;

    llama_context_params tp = llama_context_default_params();
    tp.n_ctx = (uint32_t)max_toks; tp.n_batch = (uint32_t)max_toks;
    llama_context * tmp = llama_init_from_model(const_cast<llama_model *>(mdl), tp);
    if (!tmp) {
        fprintf(stderr, "[rwkv-edit] hook: temp ctx failed\n");
        return {};
    }

    llama_token bos = llama_vocab_bos(vocab);
    auto decode_prompt = [&](const std::vector<llama_token> & toks) -> std::vector<float> {
        llama_memory_clear(llama_get_memory(tmp), true);
        llama_batch b = llama_batch_init(1, 0, 1);
        common_batch_add(b, bos, 0, {0}, false);
        llama_decode(tmp, b);
        llama_batch_free(b);
        llama_synchronize(tmp);
        llama_recurrent_state_set_f32(tmp, 0, nullptr, s_live.data());
        b = llama_batch_init((int)toks.size(), 0, 1);
        for (size_t i = 0; i < toks.size(); ++i) {
            common_batch_add(b, toks[i], (int)(i + 1), {0}, false);
        }
        llama_decode(tmp, b);
        llama_batch_free(b);
        std::vector<float> s(total_s);
        llama_synchronize(tmp);
        llama_recurrent_state_get_f32(tmp, 0, nullptr, s.data());
        return s;
    };

    OnlineProbeResult result;
    result.s_from = decode_prompt(toks_from);
    result.s_to   = decode_prompt(toks_to);
    llama_free(tmp);
    return result;
}

// Precompute offline decomposition for HYBRID method.
// Captures from/to prompts, SVDs per-head delta, stores u₁/v₁/σ.
static std::shared_ptr<HybridEditData> precompute_hybrid(
        llama_context * ctx, const llama_vocab * vocab,
        const std::string & from_prompt, const std::string & to_prompt,
        float alpha, const std::vector<int> & layers) {
    const llama_model * mdl = llama_get_model(ctx);
    const int n_layer   = llama_model_n_layer(mdl);
    const int n_embd    = llama_model_n_embd(mdl);
    const int n_embd_s  = llama_model_n_embd_s(mdl);
    const int head_size = (n_embd > 0) ? n_embd_s / n_embd : 0;
    const int n_head    = (head_size > 0) ? n_embd / head_size : 0;
    const int hs2       = head_size * head_size;

    if (n_embd_s == 0 || n_head == 0) {
        fprintf(stderr, "[hybrid] ERROR: not an SSM model or bad geometry\n");
        return nullptr;
    }

    // capture isolated states
    const size_t total_s = (size_t)n_layer * n_embd_s;
    const llama_seq_id seq_id = 0;

    llama_context_params tmp_params = llama_context_default_params();
    tmp_params.n_ctx   = 512;
    tmp_params.n_batch = 512;
    llama_context * tmp_ctx = llama_init_from_model(const_cast<llama_model *>(mdl), tmp_params);
    if (!tmp_ctx) {
        fprintf(stderr, "[hybrid] ERROR: failed to create temp context\n");
        return nullptr;
    }

    auto capture_s = [&](const std::string & prompt) -> std::vector<float> {
        llama_memory_clear(llama_get_memory(tmp_ctx), true);
        auto toks = common_tokenize(vocab, prompt, true, false);
        llama_batch batch = llama_batch_init(std::max((int)toks.size(), 1), 0, 1);
        for (size_t i = 0; i < toks.size(); ++i) {
            common_batch_add(batch, toks[i], i, {seq_id}, false);
        }
        if (llama_decode(tmp_ctx, batch) != 0) {
            llama_batch_free(batch);
            return {};
        }
        llama_batch_free(batch);
        std::vector<float> s(total_s);
        llama_synchronize(tmp_ctx);
        llama_recurrent_state_get_f32(tmp_ctx, seq_id, nullptr, s.data());
        return s;
    };

    fprintf(stderr, "[hybrid] capturing offline states...\n");
    auto s_from = capture_s(from_prompt);
    auto s_to   = capture_s(to_prompt);
    llama_free(tmp_ctx);

    if (s_from.empty() || s_to.empty()) return nullptr;

    // decompose per head
    auto data = std::make_shared<HybridEditData>();
    data->layers       = layers;
    data->from_prompt  = from_prompt;
    data->to_prompt    = to_prompt;
    data->alpha     = alpha;
    data->n_embd_s  = n_embd_s;
    data->head_size = head_size;
    data->n_head    = n_head;

    int heads_stored = 0;
    for (int layer : layers) {
        if (layer < 0 || layer >= n_layer) continue;
        for (int h = 0; h < n_head; ++h) {
            size_t offset = (size_t)layer * n_embd_s + (size_t)h * hs2;

            std::vector<float> delta(hs2);
            for (int i = 0; i < hs2; ++i) {
                delta[i] = s_to[offset + i] - s_from[offset + i];
            }

            auto decomp = power_iter_decomp(delta.data(), head_size);
            if (decomp.sigma < 1e-10f) continue;

            HybridHeadCal hc;
            hc.layer  = layer;
            hc.head   = h;
            hc.offset = offset;
            hc.sigma  = decomp.sigma;
            hc.u      = std::move(decomp.u);
            hc.v      = std::move(decomp.v);
            data->heads.push_back(std::move(hc));
            heads_stored++;
        }
    }

    fprintf(stderr, "[hybrid] offline: %d heads, alpha=%.2f\n",
            heads_stored, alpha);
    return data;
}

static std::atomic<bool> g_is_interrupted = false;
static bool should_stop() {
    return g_is_interrupted.load();
}

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
static void signal_handler(int) {
    if (g_is_interrupted.load()) {
        // second Ctrl+C - exit immediately
        // make sure to clear colors before exiting (not using LOG or console.cpp here to avoid deadlock)
        fprintf(stdout, "\033[0m\n");
        fflush(stdout);
        std::exit(130);
    }
    g_is_interrupted.store(true);
}
#endif

struct cli_context {
    server_context ctx_server;
    json messages = json::array();
    std::vector<raw_buffer> input_files;
    task_params defaults;
    bool verbose_prompt;
    int reasoning_budget = -1;
    std::string reasoning_budget_message;

    // thread for showing "loading" animation
    std::atomic<bool> loading_show;

    cli_context(const common_params & params) {
        defaults.sampling    = params.sampling;
        defaults.speculative = params.speculative;
        defaults.n_keep      = params.n_keep;
        defaults.n_predict   = params.n_predict;
        defaults.antiprompt  = params.antiprompt;

        defaults.stream = true; // make sure we always use streaming mode
        defaults.timings_per_token = true; // in order to get timings even when we cancel mid-way
        // defaults.return_progress = true; // TODO: show progress

        verbose_prompt = params.verbose_prompt;
        reasoning_budget = params.reasoning_budget;
        reasoning_budget_message = params.reasoning_budget_message;
    }

    std::string generate_completion(result_timings & out_timings) {
        server_response_reader rd = ctx_server.get_response_reader();
        auto chat_params = format_chat();
        {
            // TODO: reduce some copies here in the future
            server_task task = server_task(SERVER_TASK_TYPE_COMPLETION);
            task.id         = rd.get_new_id();
            task.index      = 0;
            task.params     = defaults;           // copy
            task.cli_prompt = chat_params.prompt; // copy
            task.cli_files  = input_files;        // copy
            task.cli        = true;

            // chat template settings
            task.params.chat_parser_params = common_chat_parser_params(chat_params);
            task.params.chat_parser_params.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
            if (!chat_params.parser.empty()) {
                task.params.chat_parser_params.parser.load(chat_params.parser);
            }

            // reasoning budget sampler
            if (!chat_params.thinking_end_tag.empty()) {
                const llama_vocab * vocab = llama_model_get_vocab(
                    llama_get_model(ctx_server.get_llama_context()));

                task.params.sampling.reasoning_budget_tokens = reasoning_budget;
                task.params.sampling.generation_prompt = chat_params.generation_prompt;

                if (!chat_params.thinking_start_tag.empty()) {
                    task.params.sampling.reasoning_budget_start =
                        common_tokenize(vocab, chat_params.thinking_start_tag, false, true);
                }
                task.params.sampling.reasoning_budget_end =
                    common_tokenize(vocab, chat_params.thinking_end_tag, false, true);
                task.params.sampling.reasoning_budget_forced =
                    common_tokenize(vocab, reasoning_budget_message + chat_params.thinking_end_tag, false, true);
            }

            rd.post_task({std::move(task)});
        }

        if (verbose_prompt) {
            console::set_display(DISPLAY_TYPE_PROMPT);
            console::log("%s\n\n", chat_params.prompt.c_str());
            console::set_display(DISPLAY_TYPE_RESET);
        }

        // wait for first result
        console::spinner::start();
        server_task_result_ptr result = rd.next(should_stop);

        console::spinner::stop();
        std::string curr_content;
        bool is_thinking = false;

        while (result) {
            if (should_stop()) {
                break;
            }
            if (result->is_error()) {
                json err_data = result->to_json();
                if (err_data.contains("message")) {
                    console::error("Error: %s\n", err_data["message"].get<std::string>().c_str());
                } else {
                    console::error("Error: %s\n", err_data.dump().c_str());
                }
                return curr_content;
            }
            auto res_partial = dynamic_cast<server_task_result_cmpl_partial *>(result.get());
            if (res_partial) {
                out_timings = std::move(res_partial->timings);
                for (const auto & diff : res_partial->oaicompat_msg_diffs) {
                    if (!diff.content_delta.empty()) {
                        if (is_thinking) {
                            console::log("\n[End thinking]\n\n");
                            console::set_display(DISPLAY_TYPE_RESET);
                            is_thinking = false;
                        }
                        curr_content += diff.content_delta;
                        console::log("%s", diff.content_delta.c_str());
                        console::flush();
                    }
                    if (!diff.reasoning_content_delta.empty()) {
                        console::set_display(DISPLAY_TYPE_REASONING);
                        if (!is_thinking) {
                            console::log("[Start thinking]\n");
                        }
                        is_thinking = true;
                        console::log("%s", diff.reasoning_content_delta.c_str());
                        console::flush();
                    }
                }
            }
            auto res_final = dynamic_cast<server_task_result_cmpl_final *>(result.get());
            if (res_final) {
                out_timings = std::move(res_final->timings);
                break;
            }
            result = rd.next(should_stop);
        }
        g_is_interrupted.store(false);
        // server_response_reader automatically cancels pending tasks upon destruction
        return curr_content;
    }

    // TODO: support remote files in the future (http, https, etc)
    std::string load_input_file(const std::string & fname, bool is_media) {
        std::ifstream file(fname, std::ios::binary);
        if (!file) {
            return "";
        }
        if (is_media) {
            raw_buffer buf;
            buf.assign((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            input_files.push_back(std::move(buf));
            return mtmd_default_marker();
        } else {
            std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            return content;
        }
    }

    common_chat_params format_chat() {
        auto meta = ctx_server.get_meta();
        auto & chat_params = meta.chat_params;

        common_chat_templates_inputs inputs;
        inputs.messages              = common_chat_msgs_parse_oaicompat(messages);
        inputs.tools                 = {}; // TODO
        inputs.tool_choice           = COMMON_CHAT_TOOL_CHOICE_NONE;
        inputs.json_schema           = ""; // TODO
        inputs.grammar               = ""; // TODO
        inputs.use_jinja             = chat_params.use_jinja;
        inputs.parallel_tool_calls   = false;
        inputs.add_generation_prompt = true;
        inputs.reasoning_format      = COMMON_REASONING_FORMAT_DEEPSEEK;
        inputs.force_pure_content    = chat_params.force_pure_content;
        inputs.enable_thinking       = chat_params.enable_thinking ? common_chat_templates_support_enable_thinking(chat_params.tmpls.get()) : false;

        // Apply chat template to the list of messages
        return common_chat_templates_apply(chat_params.tmpls.get(), inputs);
    }
};

// TODO?: Make this reusable, enums, docs
static const std::array<const std::string, 11> cmds = {
    "/audio ",
    "/clear",
    "/exit",
    "/glob ",
    "/image ",
    "/read ",
    "/regen",
    "/rwkv-edit ",
    "/rwkv-forget ",
    "/rwkv-nuke ",
};

static std::vector<std::pair<std::string, size_t>> auto_completion_callback(std::string_view line, size_t cursor_byte_pos) {
    std::vector<std::pair<std::string, size_t>> matches;
    std::string cmd;

    if (line.length() > 1 && line[0] == '/' && !std::any_of(cmds.begin(), cmds.end(), [line](const std::string & prefix) {
        return string_starts_with(line, prefix);
    })) {
        auto it = cmds.begin();

        while ((it = std::find_if(it, cmds.end(), [line](const std::string & cmd_line) {
            return string_starts_with(cmd_line, line);
        })) != cmds.end()) {
            matches.emplace_back(*it, (*it).length());
            ++it;
        }
    } else {
        auto it = std::find_if(cmds.begin(), cmds.end(), [line](const std::string & prefix) {
            return prefix.back() == ' ' && string_starts_with(line, prefix);
        });

        if (it != cmds.end()) {
            cmd = *it;
        }
    }

    if (!cmd.empty() && cmd != "/glob " && line.length() >= cmd.length() && cursor_byte_pos >= cmd.length()) {
        const std::string path_prefix  = std::string(line.substr(cmd.length(), cursor_byte_pos - cmd.length()));
        const std::string path_postfix = std::string(line.substr(cursor_byte_pos));
        auto cur_dir = std::filesystem::current_path();
        std::string cur_dir_str = cur_dir.string();
        std::string expanded_prefix = path_prefix;

#if !defined(_WIN32)
        if (string_starts_with(path_prefix, "~")) {
            const char * home = std::getenv("HOME");
            if (home && home[0]) {
                expanded_prefix = std::string(home) + path_prefix.substr(1);
            }
        }
        if (string_starts_with(expanded_prefix, "/")) {
#else
        if (std::isalpha(expanded_prefix[0]) && expanded_prefix.find(':') == 1) {
#endif
            cur_dir = std::filesystem::path(expanded_prefix).parent_path();
            cur_dir_str = "";
        } else if (!path_prefix.empty()) {
            cur_dir /= std::filesystem::path(path_prefix).parent_path();
        }

        std::error_code ec;
        for (const auto & entry : std::filesystem::directory_iterator(cur_dir, ec)) {
            if (ec) {
                break;
            }
            if (!entry.exists(ec)) {
                ec.clear();
                continue;
            }

            const std::string path_full = entry.path().string();
            std::string path_entry = !cur_dir_str.empty() && string_starts_with(path_full, cur_dir_str) ? path_full.substr(cur_dir_str.length() + 1) : path_full;

            if (entry.is_directory(ec)) {
                path_entry.push_back(std::filesystem::path::preferred_separator);
            }

            if (expanded_prefix.empty() || string_starts_with(path_entry, expanded_prefix)) {
                std::string updated_line = cmd + path_entry;
                matches.emplace_back(updated_line + path_postfix, updated_line.length());
            }

            if (ec) {
                ec.clear();
            }
        }

        if (matches.empty()) {
            std::string updated_line = cmd + path_prefix;
            matches.emplace_back(updated_line + path_postfix, updated_line.length());
        }

        // Add the longest common prefix
        if (!expanded_prefix.empty() && matches.size() > 1) {
            const std::string_view match0(matches[0].first);
            const std::string_view match1(matches[1].first);
            auto it = std::mismatch(match0.begin(), match0.end(), match1.begin(), match1.end());
            size_t len = it.first - match0.begin();

            for (size_t i = 2; i < matches.size(); ++i) {
                const std::string_view matchi(matches[i].first);
                auto cmp = std::mismatch(match0.begin(), match0.end(), matchi.begin(), matchi.end());
                len = std::min(len, static_cast<size_t>(cmp.first - match0.begin()));
            }

            std::string updated_line = std::string(match0.substr(0, len));
            matches.emplace_back(updated_line + path_postfix, updated_line.length());
        }

        std::sort(matches.begin(), matches.end(), [](const auto & a, const auto & b) {
            return a.first.compare(0, a.second, b.first, 0, b.second) < 0;
        });
    }

    return matches;
}

static constexpr size_t FILE_GLOB_MAX_RESULTS = 100;

int main(int argc, char ** argv) {
    common_params params;

    params.verbosity = LOG_LEVEL_ERROR; // by default, less verbose logs

    common_init();

    // extract --steer args before common_params_parse (which rejects unknown flags)
    // modes: nudge   — add alpha * direction to head state (default)
    //        zero    — zero out the head state
    //        scale   — multiply head state by alpha
    //        rotate  — random permutation + sign flips (preserves norm, destroys direction)
    //        random  — nudge with a random unit vector
    std::string steer_path;
    std::string steer_mode  = "nudge";
    int         steer_head  = 9;
    int         steer_layer = 0;
    float       steer_alpha = 1.0f;
    {
        std::vector<char *> filtered;
        for (int i = 0; i < argc; ++i) {
            std::string a = argv[i];
            if      (a == "--steer"        && i+1 < argc) { steer_path  = argv[++i]; }
            else if (a == "--steer-mode"   && i+1 < argc) { steer_mode  = argv[++i]; }
            else if (a == "--steer-head"   && i+1 < argc) { steer_head  = std::atoi(argv[++i]); }
            else if (a == "--steer-layer"  && i+1 < argc) { steer_layer = std::atoi(argv[++i]); }
            else if (a == "--steer-alpha"  && i+1 < argc) { steer_alpha = (float) std::atof(argv[++i]); }
            else { filtered.push_back(argv[i]); }
        }
        argc = (int) filtered.size();
        for (int i = 0; i < argc; ++i) { argv[i] = filtered[i]; }
    }
    const bool do_steer = !steer_path.empty() || steer_mode == "zero" || steer_mode == "scale" || steer_mode == "rotate" || steer_mode == "random";

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_CLI)) {
        return 1;
    }

    // TODO: maybe support it later?
    if (params.conversation_mode == COMMON_CONVERSATION_MODE_DISABLED) {
        console::error("--no-conversation is not supported by llama-cli\n");
        console::error("please use llama-completion instead\n");
    }

    // struct that contains llama context and inference
    cli_context ctx_cli(params);

    llama_backend_init();
    llama_numa_init(params.numa);

    // TODO: avoid using atexit() here by making `console` a singleton
    console::init(params.simple_io, params.use_color);
    atexit([]() { console::cleanup(); });

    console::set_display(DISPLAY_TYPE_RESET);
    console::set_completion_callback(auto_completion_callback);

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = signal_handler;
    sigemptyset (&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
    sigaction(SIGTERM, &sigint_action, NULL);
#elif defined (_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        return (ctrl_type == CTRL_C_EVENT) ? (signal_handler(SIGINT), true) : false;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif

    console::log("\nLoading model... "); // followed by loading animation
    console::spinner::start();
    if (!ctx_cli.ctx_server.load_model(params)) {
        console::spinner::stop();
        console::error("\nFailed to load the model\n");
        return 1;
    }

    console::spinner::stop();
    console::log("\n");

    // ---- steering setup (optional) ------------------------------------------
    if (do_steer) {
        llama_context * lctx = ctx_cli.ctx_server.get_llama_context();
        const llama_model * lmodel = llama_get_model(lctx);

        if (!llama_model_is_recurrent(lmodel)) {
            console::error("--steer requires a recurrent model (RWKV/Mamba)\n");
            return 1;
        }

        const int n_layer   = llama_model_n_layer(lmodel);
        const int n_embd    = llama_model_n_embd(lmodel);
        const int n_embd_s  = llama_model_n_embd_s(lmodel);
        const int head_size = n_embd_s / n_embd;
        const int n_head    = n_embd / head_size;
        const int hs2       = head_size * head_size;

        const bool steer_all_heads  = (steer_head  == -1);
        const bool steer_all_layers = (steer_layer == -1);
        if (!steer_all_heads && (steer_head < 0 || steer_head >= n_head)) {
            console::error("--steer-head %d out of range [0, %d) or -1 for all\n", steer_head, n_head); return 1;
        }
        if (!steer_all_layers && (steer_layer < 0 || steer_layer >= n_layer)) {
            console::error("--steer-layer %d out of range [0, %d) or -1 for all\n", steer_layer, n_layer); return 1;
        }

        // load direction vector (only needed for "nudge" mode)
        // auto-detect granularity from file size:
        //   hs2              = one head   (head_size * head_size)
        //   n_embd_s         = one layer  (all heads)
        //   n_layer*n_embd_s = full state (all layers, all heads)
        enum steer_scope_t { SCOPE_HEAD, SCOPE_LAYER, SCOPE_FULL };
        steer_scope_t steer_scope = SCOPE_HEAD;
        std::vector<float> direction;
        if (steer_mode == "nudge") {
            if (steer_path.empty()) { console::error("--steer FILE required for nudge mode\n"); return 1; }
            std::ifstream ifs(steer_path);
            if (!ifs) { console::error("cannot open --steer file: %s\n", steer_path.c_str()); return 1; }
            float v; while (ifs >> v) { direction.push_back(v); }
            const int dir_size = (int) direction.size();
            if (dir_size == n_layer * n_embd_s) {
                steer_scope = SCOPE_FULL;
                console::log("direction: full state (%d values, all layers)\n", dir_size);
            } else if (dir_size == n_embd_s) {
                steer_scope = SCOPE_LAYER;
                console::log("direction: one layer (%d values, all %d heads)\n", dir_size, n_head);
            } else if (dir_size == hs2) {
                steer_scope = SCOPE_HEAD;
                console::log("direction: one head (%d values)\n", dir_size);
            } else {
                console::error("direction has %d values, expected %d (head), %d (layer), or %d (full)\n",
                               dir_size, hs2, n_embd_s, n_layer * n_embd_s);
                return 1;
            }
            float norm = 0;
            for (float x : direction) { norm += x * x; }
            norm = sqrtf(norm);
            if (norm > 0) { for (float & x : direction) { x /= norm; } }
        } else if (steer_mode == "random") {
            direction.resize(hs2);
            std::mt19937 rng(42);
            std::normal_distribution<float> dist(0.0f, 1.0f);
            for (float & x : direction) { x = dist(rng); }
            float norm = 0;
            for (float x : direction) { norm += x * x; }
            norm = sqrtf(norm);
            if (norm > 0) { for (float & x : direction) { x /= norm; } }
        }

        console::log("steering: mode=%s layer=%d head=%d alpha=%.3f\n",
                      steer_mode.c_str(), steer_layer, steer_head, steer_alpha);

        ctx_cli.ctx_server.on_prompt_done(
            [direction, steer_mode, steer_scope, steer_layer, steer_head, steer_alpha,
             steer_all_heads, steer_all_layers, n_embd_s, n_head, hs2]
            (llama_context * ctx, llama_token, llama_seq_id seq_id) {
                const int n_layer_total = llama_model_n_layer(llama_get_model(ctx));
                const size_t total_s = (size_t) n_layer_total * n_embd_s;
                std::vector<float> s_flat(total_s);

                if (!llama_recurrent_state_get_f32(ctx, seq_id, nullptr, s_flat.data())) {
                    fprintf(stderr, "[steer] ERROR: get_f32 failed\n"); return;
                }

                float norm_before = 0;
                for (size_t j = 0; j < total_s; ++j) { norm_before += s_flat[j] * s_flat[j]; }
                norm_before = sqrtf(norm_before);

                if (steer_mode == "nudge" && !direction.empty()) {
                    // apply direction based on its scope
                    if (steer_scope == SCOPE_FULL) {
                        // direction covers all layers × all heads
                        for (size_t j = 0; j < total_s && j < direction.size(); ++j) {
                            s_flat[j] += steer_alpha * direction[j];
                        }
                    } else if (steer_scope == SCOPE_LAYER) {
                        // direction covers one layer (all heads) — apply to steer_layer
                        float * layer_s = s_flat.data() + steer_layer * n_embd_s;
                        for (int j = 0; j < n_embd_s && j < (int) direction.size(); ++j) {
                            layer_s[j] += steer_alpha * direction[j];
                        }
                    } else {
                        // direction covers one head — apply to steer_layer/steer_head
                        float * head_s = s_flat.data() + steer_layer * n_embd_s + steer_head * hs2;
                        for (int j = 0; j < hs2 && j < (int) direction.size(); ++j) {
                            head_s[j] += steer_alpha * direction[j];
                        }
                    }
                } else {
                    // zero / scale / rotate / random — apply per head as before
                    const int l_start = steer_all_layers ? 0 : steer_layer;
                    const int l_end   = steer_all_layers ? n_layer_total : steer_layer + 1;
                    const int h_start = steer_all_heads  ? 0 : steer_head;
                    const int h_end   = steer_all_heads  ? n_head : steer_head + 1;

                    for (int l = l_start; l < l_end; ++l) {
                        float * layer_s = s_flat.data() + l * n_embd_s;
                        for (int h = h_start; h < h_end; ++h) {
                            float * head_s = layer_s + h * hs2;
                            if (steer_mode == "zero") {
                                for (int j = 0; j < hs2; ++j) { head_s[j] = 0.0f; }
                            } else if (steer_mode == "scale") {
                                for (int j = 0; j < hs2; ++j) { head_s[j] *= steer_alpha; }
                            } else if (steer_mode == "rotate") {
                                std::mt19937 rng(42 + l * n_head + h);
                                std::vector<float> tmp(head_s, head_s + hs2);
                                for (int j = hs2 - 1; j > 0; --j) {
                                    std::uniform_int_distribution<int> dist(0, j);
                                    std::swap(tmp[j], tmp[dist(rng)]);
                                }
                                std::uniform_int_distribution<int> coin(0, 1);
                                for (int j = 0; j < hs2; ++j) {
                                    head_s[j] = coin(rng) ? tmp[j] : -tmp[j];
                                }
                            } else if (steer_mode == "random" && !direction.empty()) {
                                for (int j = 0; j < hs2; ++j) {
                                    head_s[j] += steer_alpha * direction[j];
                                }
                            }
                        }
                    }
                }

                float norm_after = 0;
                for (size_t j = 0; j < total_s; ++j) { norm_after += s_flat[j] * s_flat[j]; }
                norm_after = sqrtf(norm_after);

                fprintf(stderr, "[steer] %s alpha=%.1f  total_norm: %.2f -> %.2f\n",
                        steer_mode.c_str(), steer_alpha, norm_before, norm_after);

                if (!llama_recurrent_state_set_f32(ctx, seq_id, nullptr, s_flat.data())) {
                    fprintf(stderr, "[steer] ERROR: set_f32 failed\n");
                }
            });
    }

    std::thread inference_thread([&ctx_cli]() {
        ctx_cli.ctx_server.start_loop();
    });

    auto inf = ctx_cli.ctx_server.get_meta();
    std::string modalities = "text";
    if (inf.has_inp_image) {
        modalities += ", vision";
    }
    if (inf.has_inp_audio) {
        modalities += ", audio";
    }

    auto add_system_prompt = [&]() {
        if (!params.system_prompt.empty()) {
            ctx_cli.messages.push_back({
                {"role",    "system"},
                {"content", params.system_prompt}
            });
        }
    };
    add_system_prompt();

    console::log("\n");
    console::log("%s\n", LLAMA_ASCII_LOGO);
    console::log("build      : %s\n", inf.build_info.c_str());
    console::log("model      : %s\n", inf.model_name.c_str());
    console::log("modalities : %s\n", modalities.c_str());
    if (!params.system_prompt.empty()) {
        console::log("using custom system prompt\n");
    }
    console::log("\n");
    console::log("available commands:\n");
    console::log("  /exit or Ctrl+C     stop or exit\n");
    console::log("  /regen              regenerate the last response\n");
    console::log("  /clear              clear the chat history\n");
    console::log("  /read <file>        add a text file\n");
    console::log("  /glob <pattern>     add text files using globbing pattern\n");
    if (inf.has_inp_image) {
        console::log("  /image <file>       add an image file\n");
    }
    if (inf.has_inp_audio) {
        console::log("  /audio <file>       add an audio file\n");
    }
    console::log("  /rwkv-edit \"from\" \"to\" [ratio] [method] [\"layers\"]\n");
    console::log("                        edit RWKV state (methods: HYB-U, HYB-UV, ONLINE; default HYB-U)\n");
    console::log("  /rwkv-forget \"fact\" \"entity\" [ratio] [method] [\"layers\"]\n");
    console::log("                        forget a fact but keep the entity (ratio=1.0, UNIFORM)\n");
    console::log("                        e.g. /rwkv-forget \"Bob is in Paris\" \"Bob\"\n");
    console::log("  /rwkv-nuke \"entity\" [ratio] [\"layers\"]\n");
    console::log("                        remove the existence of an entity (ratio=0.35)\n");
    console::log("                        e.g. /rwkv-nuke \"Bob\" — Bob never existed\n");
    console::log("\n");

    // interactive loop
    std::string cur_msg;

    auto add_text_file = [&](const std::string & fname) -> bool {
        std::string marker = ctx_cli.load_input_file(fname, false);
        if (marker.empty()) {
            console::error("file does not exist or cannot be opened: '%s'\n", fname.c_str());
            return false;
        }
        if (inf.fim_sep_token != LLAMA_TOKEN_NULL) {
            cur_msg += common_token_to_piece(ctx_cli.ctx_server.get_llama_context(), inf.fim_sep_token, true);
            cur_msg += fname;
            cur_msg.push_back('\n');
        } else {
            cur_msg += "--- File: ";
            cur_msg += fname;
            cur_msg += " ---\n";
        }
        cur_msg += marker;
        console::log("Loaded text from '%s'\n", fname.c_str());
        return true;
    };

    while (true) {
        std::string buffer;
        console::set_display(DISPLAY_TYPE_USER_INPUT);
        if (params.prompt.empty()) {
            console::log("\n> ");
            std::string line;
            bool another_line = true;
            do {
                another_line = console::readline(line, params.multiline_input);
                buffer += line;
            } while (another_line);
        } else {
            // process input prompt from args
            for (auto & fname : params.image) {
                std::string marker = ctx_cli.load_input_file(fname, true);
                if (marker.empty()) {
                    console::error("file does not exist or cannot be opened: '%s'\n", fname.c_str());
                    break;
                }
                console::log("Loaded media from '%s'\n", fname.c_str());
                cur_msg += marker;
            }
            buffer = params.prompt;
            if (buffer.size() > 500) {
                console::log("\n> %s ... (truncated)\n", buffer.substr(0, 500).c_str());
            } else {
                console::log("\n> %s\n", buffer.c_str());
            }
            params.prompt.clear(); // only use it once
        }
        console::set_display(DISPLAY_TYPE_RESET);
        console::log("\n");

        if (should_stop()) {
            g_is_interrupted.store(false);
            break;
        }

        // remove trailing newline
        if (!buffer.empty() &&buffer.back() == '\n') {
            buffer.pop_back();
        }

        // skip empty messages
        if (buffer.empty()) {
            continue;
        }

        bool add_user_msg = true;

        // process commands
        if (string_starts_with(buffer, "/exit")) {
            break;
        } else if (string_starts_with(buffer, "/regen")) {
            if (ctx_cli.messages.size() >= 2) {
                size_t last_idx = ctx_cli.messages.size() - 1;
                ctx_cli.messages.erase(last_idx);
                add_user_msg = false;
            } else {
                console::error("No message to regenerate.\n");
                continue;
            }
        } else if (string_starts_with(buffer, "/clear")) {
            ctx_cli.messages.clear();
            add_system_prompt();

            ctx_cli.input_files.clear();
            ctx_cli.ctx_server.on_prompt_done(nullptr);
            console::log("Chat history cleared.\n");
            continue;
        } else if (
                (string_starts_with(buffer, "/image ") && inf.has_inp_image) ||
                (string_starts_with(buffer, "/audio ") && inf.has_inp_audio)) {
            // just in case (bad copy-paste for example), we strip all trailing/leading spaces
            std::string fname = string_strip(buffer.substr(7));
            std::string marker = ctx_cli.load_input_file(fname, true);
            if (marker.empty()) {
                console::error("file does not exist or cannot be opened: '%s'\n", fname.c_str());
                continue;
            }
            cur_msg += marker;
            console::log("Loaded media from '%s'\n", fname.c_str());
            continue;
        } else if (string_starts_with(buffer, "/read ")) {
            std::string fname = string_strip(buffer.substr(6));
            add_text_file(fname);
            continue;
        } else if (string_starts_with(buffer, "/rwkv-edit ")) {
            auto args = parse_shell_args(buffer.substr(11));
            if (args.size() < 2) {
                console::error("usage: /rwkv-edit \"<from>\" \"<to>\" [strength] [method] [\"layers\"]\n");
                console::error("  Edits RWKV recurrent state by applying the delta between two prompts.\n");
                console::error("  Methods:\n");
                console::error("    HYB-U     (default) offline u1 + online v1/sigma + adaptive (ratio=0.20)\n");
                console::error("    HYB-UV              offline u1/sigma + online v1 (ratio=0.20)\n");
                console::error("    ONLINE              full online rank-1 probe + adaptive (ratio=0.20)\n");
                console::error("    ADAPTIVE            norm-ratio scaling. strength = ratio (default 0.20)\n");
                console::error("    UNIFORM             linear addition. strength = alpha (default 1.0)\n");
                console::error("    R1-WT               per-head rank-1 weighting. strength = alpha\n");
                console::error("    R1-ADAPTIVE         R1-WT + auto-scaling. strength = ratio\n");
                console::error("  Example: /rwkv-edit \"Bob lives in Austin\" \"Bob lives in Chicago\"\n");
                console::error("  Example: /rwkv-edit \"Bob lives in Austin\" \"Bob lives in Chicago\" 0.8 HYB-UV\n");
                console::error("  Example: /rwkv-edit \"Bob lives in Austin\" \"Bob lives in Chicago\" 0.2 ONLINE\n");
                continue;
            }
            std::string from_prompt = args[0];
            std::string to_prompt   = args[1];

            // parse remaining args: [strength] [method] [layers]
            // strength is numeric, method is alpha, layers is "N-M"
            auto is_numeric = [](const std::string & s) {
                return !s.empty() && (std::isdigit(s[0]) || s[0] == '.' || s[0] == '-');
            };

            float edit_alpha = -1.0f;  // sentinel: use default
            std::string edit_method;
            std::string edit_layers;
            for (size_t ai = 2; ai < args.size(); ++ai) {
                if (edit_alpha < 0 && is_numeric(args[ai])) {
                    edit_alpha = std::stof(args[ai]);
                } else if (edit_method.empty() && !is_numeric(args[ai])) {
                    edit_method = args[ai];
                } else {
                    edit_layers = args[ai];
                }
            }
            if (edit_method.empty()) edit_method = "HYB-U";
            // aliases
            if (edit_method == "HYBRID") edit_method = "HYB-U";
            if (edit_method == "ONLINE") edit_method = "ONLINE";

            if (edit_alpha < 0) {
                if (edit_method == "HYB-U" || edit_method == "HYB-UV" || edit_method == "ONLINE") {
                    edit_alpha = 0.20f;
                } else if (edit_method == "ADAPTIVE" || edit_method == "R1-ADAPTIVE") {
                    edit_alpha = 0.20f;
                } else {
                    edit_alpha = 1.0f;
                }
            }

            if (edit_method != "UNIFORM" && edit_method != "ADAPTIVE" &&
                edit_method != "R1-WT" && edit_method != "R1-ADAPTIVE" &&
                edit_method != "HYB-U" && edit_method != "HYB-UV" &&
                edit_method != "ONLINE") {
                console::error("unknown method '%s' (use HYB-U, HYB-UV, ONLINE, ADAPTIVE, UNIFORM, R1-WT, R1-ADAPTIVE)\n",
                               edit_method.c_str());
                continue;
            }

            llama_context * lctx = ctx_cli.ctx_server.get_llama_context();
            if (!lctx) {
                console::error("no llama context available\n");
                continue;
            }
            const llama_model * mdl = llama_get_model(lctx);
            const llama_vocab * voc = llama_model_get_vocab(mdl);
            int n_layer = llama_model_n_layer(mdl);

            // default layers: upper half
            std::vector<int> layer_vec;
            if (edit_layers.empty()) {
                for (int i = n_layer / 2; i < n_layer; ++i) layer_vec.push_back(i);
            } else {
                layer_vec = parse_layer_range(edit_layers, n_layer);
            }

            if (edit_method == "HYB-U" || edit_method == "HYB-UV" || edit_method == "ONLINE") {
                std::shared_ptr<HybridEditData> hybrid;

                if (edit_method == "ONLINE") {
                    // ONLINE needs no offline calibration — just store prompts + geometry
                    const int n_embd   = llama_model_n_embd(mdl);
                    const int n_embd_s = llama_model_n_embd_s(mdl);
                    const int head_size = (n_embd > 0) ? n_embd_s / n_embd : 0;
                    const int n_head_v  = (head_size > 0) ? n_embd / head_size : 0;
                    if (n_embd_s == 0 || n_head_v == 0) {
                        console::error("Not an SSM model or bad geometry.\n");
                        continue;
                    }
                    hybrid = std::make_shared<HybridEditData>();
                    hybrid->layers      = layer_vec;
                    hybrid->from_prompt = from_prompt;
                    hybrid->to_prompt   = to_prompt;
                    hybrid->alpha       = edit_alpha;
                    hybrid->n_embd_s    = n_embd_s;
                    hybrid->head_size   = head_size;
                    hybrid->n_head      = n_head_v;
                    hybrid->method      = "ONLINE";
                    fprintf(stderr, "[rwkv-edit] ONLINE: no offline precompute needed\n");
                } else {
                    // HYB-U and HYB-UV both need offline calibration
                    hybrid = precompute_hybrid(lctx, voc, from_prompt, to_prompt,
                                                edit_alpha, layer_vec);
                    if (!hybrid) {
                        console::error("Edit precompute failed.\n");
                        continue;
                    }
                    hybrid->method = edit_method;
                }

                ctx_cli.ctx_server.on_prompt_done(
                    [hybrid](llama_context * hook_ctx, llama_token, llama_seq_id hook_seq) {
                        const llama_model * hook_mdl = llama_get_model(hook_ctx);
                        const llama_vocab * hook_vocab = llama_model_get_vocab(hook_mdl);
                        const int n_layer_total = llama_model_n_layer(hook_mdl);
                        const int n_embd_s = hybrid->n_embd_s;
                        const int hs  = hybrid->head_size;
                        const int hs2 = hs * hs;
                        const size_t total_s = (size_t)n_layer_total * n_embd_s;

                        // 1. get live state
                        std::vector<float> s_live(total_s);
                        llama_synchronize(hook_ctx);
                        if (!llama_recurrent_state_get_f32(hook_ctx, hook_seq, nullptr, s_live.data())) {
                            fprintf(stderr, "[rwkv-edit] hook: get_f32 failed\n");
                            return;
                        }

                        // 2. online probe
                        auto probe = online_probe(hook_mdl, hook_vocab, s_live,
                                                  hybrid->from_prompt, hybrid->to_prompt);
                        if (probe.s_from.empty() || probe.s_to.empty()) return;

                        if (hybrid->method == "HYB-UV") {
                            // ── HYB-UV: offline u₁ + online v₁, OFFLINE σ ──
                            // No adaptive scaling — use alpha * offline_σ directly.
                            for (const auto & hc : hybrid->heads) {
                                // get online v₁ from probe delta
                                std::vector<float> delta(hs2);
                                for (int i = 0; i < hs2; ++i) {
                                    delta[i] = probe.s_to[hc.offset + i] - probe.s_from[hc.offset + i];
                                }
                                auto on = power_iter_decomp(delta.data(), hs);
                                if (on.sigma < 1e-10f) continue;

                                float scale = hybrid->alpha * hc.sigma;  // offline σ
                                float * dst = s_live.data() + hc.offset;
                                for (int r = 0; r < hs; ++r) {
                                    float su = scale * hc.u[r];  // offline u₁
                                    for (int c = 0; c < hs; ++c) {
                                        dst[r * hs + c] += su * on.v[c];  // online v₁
                                    }
                                }
                            }
                        } else if (hybrid->method == "ONLINE") {
                            // ── ONLINE: full rank-1 from online probe, adaptive scaling ──
                            // Decompose each head from probe delta, apply with adaptive scaling.
                            float s_norm_sq = 0.0f;
                            for (size_t j = 0; j < total_s; ++j) {
                                s_norm_sq += s_live[j] * s_live[j];
                            }

                            // collect online decompositions and their total energy
                            struct OnlineHead {
                                size_t offset;
                                float sigma;
                                std::vector<float> u, v;
                            };
                            std::vector<OnlineHead> on_heads;
                            float e_norm_sq = 0.0f;

                            for (int layer : hybrid->layers) {
                                if (layer < 0 || layer >= n_layer_total) continue;
                                for (int h = 0; h < hybrid->n_head; ++h) {
                                    size_t offset = (size_t)layer * n_embd_s + (size_t)h * hs2;
                                    std::vector<float> delta(hs2);
                                    for (int i = 0; i < hs2; ++i) {
                                        delta[i] = probe.s_to[offset + i] - probe.s_from[offset + i];
                                    }
                                    auto decomp = power_iter_decomp(delta.data(), hs);
                                    if (decomp.sigma < 1e-10f) continue;
                                    e_norm_sq += decomp.sigma * decomp.sigma;
                                    on_heads.push_back({offset, decomp.sigma,
                                                        std::move(decomp.u), std::move(decomp.v)});
                                }
                            }

                            float alpha = hybrid->alpha;
                            if (e_norm_sq > 1e-20f) {
                                alpha *= std::sqrt(s_norm_sq) / std::sqrt(e_norm_sq);
                            }

                            for (const auto & oh : on_heads) {
                                float scale = alpha * oh.sigma;
                                float * dst = s_live.data() + oh.offset;
                                for (int r = 0; r < hs; ++r) {
                                    float su = scale * oh.u[r];
                                    for (int c = 0; c < hs; ++c) {
                                        dst[r * hs + c] += su * oh.v[c];
                                    }
                                }
                            }
                        } else {
                            // ── HYB-U: offline u₁ + online v₁/σ + adaptive scaling ──
                            float s_norm_sq = 0.0f;
                            for (size_t j = 0; j < total_s; ++j) {
                                s_norm_sq += s_live[j] * s_live[j];
                            }

                            float e_norm_sq = 0.0f;
                            for (const auto & hc : hybrid->heads) {
                                std::vector<float> delta(hs2);
                                for (int i = 0; i < hs2; ++i) {
                                    delta[i] = probe.s_to[hc.offset + i] - probe.s_from[hc.offset + i];
                                }
                                auto decomp = power_iter_decomp(delta.data(), hs);
                                e_norm_sq += decomp.sigma * decomp.sigma;
                            }

                            float alpha = hybrid->alpha;
                            if (e_norm_sq > 1e-20f) {
                                alpha *= std::sqrt(s_norm_sq) / std::sqrt(e_norm_sq);
                            }

                            for (const auto & hc : hybrid->heads) {
                                std::vector<float> delta(hs2);
                                for (int i = 0; i < hs2; ++i) {
                                    delta[i] = probe.s_to[hc.offset + i] - probe.s_from[hc.offset + i];
                                }
                                auto on = power_iter_decomp(delta.data(), hs);
                                if (on.sigma < 1e-10f) continue;

                                float scale = alpha * on.sigma;
                                float * dst = s_live.data() + hc.offset;
                                for (int r = 0; r < hs; ++r) {
                                    float su = scale * hc.u[r];
                                    for (int c = 0; c < hs; ++c) {
                                        dst[r * hs + c] += su * on.v[c];
                                    }
                                }
                            }
                        }

                        // write back
                        llama_synchronize(hook_ctx);
                        if (!llama_recurrent_state_set_f32(hook_ctx, hook_seq, nullptr, s_live.data())) {
                            fprintf(stderr, "[rwkv-edit] hook: set_f32 failed\n");
                        }
                    });

                console::log("Edit registered (%s, ratio=%.2f).\n", edit_method.c_str(), edit_alpha);
                continue;
            }

            // ── non-HYBRID path (existing methods) ───────────────────────
            auto weighted_delta = precompute_rwkv_edit(
                lctx, voc, from_prompt, to_prompt,
                edit_alpha, edit_method, layer_vec);

            if (weighted_delta.empty()) {
                console::error("State edit failed.\n");
                continue;
            }

            int n_embd_s_val = llama_model_n_embd_s(mdl);
            bool is_adaptive = (edit_method == "ADAPTIVE" || edit_method == "R1-ADAPTIVE");
            float strength   = edit_alpha;

            // register persistent hook: apply edit after every prompt processing
            ctx_cli.ctx_server.on_prompt_done(
                [weighted_delta, n_embd_s_val, is_adaptive, strength]
                (llama_context * hook_ctx, llama_token, llama_seq_id hook_seq) {
                    const int n_layer_total = llama_model_n_layer(llama_get_model(hook_ctx));
                    const size_t total_s = (size_t)n_layer_total * n_embd_s_val;

                    std::vector<float> s_flat(total_s);
                    llama_synchronize(hook_ctx);
                    if (!llama_recurrent_state_get_f32(hook_ctx, hook_seq, nullptr, s_flat.data())) {
                        fprintf(stderr, "[rwkv-edit] hook: get_f32 failed\n");
                        return;
                    }

                    if (!is_adaptive) {
                        // fixed alpha: simple linear addition
                        for (size_t j = 0; j < total_s && j < weighted_delta.size(); ++j) {
                            s_flat[j] += strength * weighted_delta[j];
                        }
                    } else {
                        // adaptive: auto-scale alpha based on state/delta norm ratio
                        float s_norm_sq = 0.0f, d_norm_sq = 0.0f;
                        for (size_t j = 0; j < total_s && j < weighted_delta.size(); ++j) {
                            s_norm_sq += s_flat[j] * s_flat[j];
                            d_norm_sq += weighted_delta[j] * weighted_delta[j];
                        }
                        float s_norm = std::sqrt(s_norm_sq);
                        float d_norm = std::sqrt(d_norm_sq);
                        float alpha = (d_norm > 1e-10f) ? strength * s_norm / d_norm : strength;

                        for (size_t j = 0; j < total_s && j < weighted_delta.size(); ++j) {
                            s_flat[j] += alpha * weighted_delta[j];
                        }
                    }

                    llama_synchronize(hook_ctx);
                    if (!llama_recurrent_state_set_f32(hook_ctx, hook_seq, nullptr, s_flat.data())) {
                        fprintf(stderr, "[rwkv-edit] hook: set_f32 failed\n");
                    }
                });

            console::log("Edit registered. Will apply on each generation.\n");
            continue;
        } else if (string_starts_with(buffer, "/rwkv-forget ") ||
                   string_starts_with(buffer, "/rwkv-nuke ")) {
            bool is_nuke = string_starts_with(buffer, "/rwkv-nuke ");
            std::string cmd_name = is_nuke ? "rwkv-nuke" : "rwkv-forget";
            size_t prefix_len = is_nuke ? 11 : 13;

            auto args = parse_shell_args(buffer.substr(prefix_len));
            if (args.empty() || (!is_nuke && args.size() < 2)) {
                if (is_nuke) {
                    console::error("usage: /rwkv-nuke \"<entity>\" [ratio] [\"layers\"]\n");
                    console::error("  Nukes an entity from existence in the RWKV state.\n");
                    console::error("  The entity and ALL its facts are removed.\n");
                    console::error("  Example: /rwkv-nuke \"Bob\"\n");
                    console::error("  Example: /rwkv-nuke \"Bob is a baker in Paris\" 0.4\n");
                } else {
                    console::error("usage: /rwkv-forget \"<fact>\" \"<entity>\" [ratio] [method] [\"layers\"]\n");
                    console::error("  Forgets a fact but keeps the entity.\n");
                    console::error("  Example: /rwkv-forget \"Bob is in Paris\" \"Bob\"\n");
                    console::error("  Example: /rwkv-forget \"Alice has a car\" \"Alice\" 0.8\n");
                }
                continue;
            }
            std::string fact_prompt = args[0];
            float ratio = 0.55f;
            std::string layer_spec;

            // parse optional trailing args
            size_t ai = is_nuke ? 1 : 2;
            auto is_numeric = [](const std::string & s) {
                return !s.empty() && (std::isdigit(s[0]) || s[0] == '.');
            };
            if (ai < args.size() && is_numeric(args[ai])) {
                ratio = std::stof(args[ai++]);
            }
            if (ai < args.size()) {
                layer_spec = args[ai];
            }

            llama_context * lctx = ctx_cli.ctx_server.get_llama_context();
            if (!lctx) {
                console::error("no llama context available\n");
                continue;
            }
            const llama_model * mdl = llama_get_model(lctx);

            // nuke:   replace everything with generic placeholder
            // forget: replace fact words with entity repetition to preserve structure
            //         "Bob is a baker" + entity="Bob" → "Bob is a Bob"
            std::string to_prompt;
            if (is_nuke) {
                to_prompt = "Someone is somewhere doing something.";
            } else {
                to_prompt = args[1];
            }
            int n_layer = llama_model_n_layer(mdl);

            std::vector<int> layer_vec;
            if (layer_spec.empty()) {
                for (int i = n_layer / 2; i < n_layer; ++i) layer_vec.push_back(i);
            } else {
                layer_vec = parse_layer_range(layer_spec, n_layer);
            }

            const llama_vocab * voc = llama_model_get_vocab(mdl);

            fprintf(stderr, "[%s] HYB-U: \"%s\" -> \"%s\", ratio=%.2f\n",
                    cmd_name.c_str(), fact_prompt.substr(0, 40).c_str(),
                    to_prompt.c_str(), ratio);

            auto hybrid = precompute_hybrid(lctx, voc, fact_prompt, to_prompt,
                                            ratio, layer_vec);
            if (!hybrid) {
                console::error("Failed.\n");
                continue;
            }

            hybrid->method = "HYB-U";
            ctx_cli.ctx_server.on_prompt_done(
                [hybrid, cmd_name](llama_context * hook_ctx, llama_token, llama_seq_id hook_seq) {
                    const llama_model * hook_mdl = llama_get_model(hook_ctx);
                    const llama_vocab * hook_vocab = llama_model_get_vocab(hook_mdl);
                    const int n_layer_total = llama_model_n_layer(hook_mdl);
                    const int n_embd_s = hybrid->n_embd_s;
                    const int hs  = hybrid->head_size;
                    const int hs2 = hs * hs;
                    const size_t total_s = (size_t)n_layer_total * n_embd_s;

                    std::vector<float> s_live(total_s);
                    llama_synchronize(hook_ctx);
                    if (!llama_recurrent_state_get_f32(hook_ctx, hook_seq, nullptr, s_live.data())) {
                        fprintf(stderr, "[%s] hook: get_f32 failed\n", cmd_name.c_str());
                        return;
                    }

                    auto probe = online_probe(hook_mdl, hook_vocab, s_live,
                                              hybrid->from_prompt, hybrid->to_prompt);
                    if (probe.s_from.empty() || probe.s_to.empty()) return;

                    // adaptive scaling
                    float s_norm_sq = 0.0f;
                    for (size_t j = 0; j < total_s; ++j) {
                        s_norm_sq += s_live[j] * s_live[j];
                    }

                    float e_norm_sq = 0.0f;
                    for (const auto & hc : hybrid->heads) {
                        std::vector<float> delta(hs2);
                        for (int i = 0; i < hs2; ++i) {
                            delta[i] = probe.s_to[hc.offset + i] - probe.s_from[hc.offset + i];
                        }
                        auto decomp = power_iter_decomp(delta.data(), hs);
                        e_norm_sq += decomp.sigma * decomp.sigma;
                    }

                    float alpha = hybrid->alpha;
                    if (e_norm_sq > 1e-20f) {
                        alpha *= std::sqrt(s_norm_sq) / std::sqrt(e_norm_sq);
                    }

                    // apply offline u₁ * online v₁, online σ
                    for (const auto & hc : hybrid->heads) {
                        std::vector<float> delta(hs2);
                        for (int i = 0; i < hs2; ++i) {
                            delta[i] = probe.s_to[hc.offset + i] - probe.s_from[hc.offset + i];
                        }
                        auto on = power_iter_decomp(delta.data(), hs);
                        if (on.sigma < 1e-10f) continue;
                        float scale = alpha * on.sigma;
                        float * dst = s_live.data() + hc.offset;
                        for (int r = 0; r < hs; ++r) {
                            float su = scale * hc.u[r];
                            for (int c = 0; c < hs; ++c) {
                                dst[r * hs + c] += su * on.v[c];
                            }
                        }
                    }

                    llama_synchronize(hook_ctx);
                    if (!llama_recurrent_state_set_f32(hook_ctx, hook_seq, nullptr, s_live.data())) {
                        fprintf(stderr, "[%s] hook: set_f32 failed\n", cmd_name.c_str());
                    }
                });

            if (is_nuke) {
                console::log("Nuke registered (HYB-U, ratio=%.2f).\n", ratio);
            } else {
                console::log("Forget registered (HYB-U, ratio=%.2f).\n", ratio);
            }
            continue;
        } else if (string_starts_with(buffer, "/glob ")) {
            std::error_code ec;
            size_t count = 0;
            auto curdir = std::filesystem::current_path();
            std::string pattern = string_strip(buffer.substr(6));
            std::filesystem::path rel_path;

            auto startglob = pattern.find_first_of("![*?");
            if (startglob != std::string::npos && startglob != 0) {
                auto endpath = pattern.substr(0, startglob).find_last_of('/');
                if (endpath != std::string::npos) {
                    std::string rel_pattern = pattern.substr(0, endpath);
#if !defined(_WIN32)
                    if (string_starts_with(rel_pattern, "~")) {
                        const char * home = std::getenv("HOME");
                        if (home && home[0]) {
                            rel_pattern = std::string(home) + rel_pattern.substr(1);
                        }
                    }
#endif
                    rel_path = rel_pattern;
                    pattern.erase(0, endpath + 1);
                    curdir /= rel_path;
                }
            }

            for (const auto & entry : std::filesystem::recursive_directory_iterator(curdir,
                    std::filesystem::directory_options::skip_permission_denied, ec)) {
                if (!entry.is_regular_file()) {
                    continue;
                }

                std::string rel = std::filesystem::relative(entry.path(), curdir, ec).string();
                if (ec) {
                    ec.clear();
                    continue;
                }
                std::replace(rel.begin(), rel.end(), '\\', '/');

                if (!glob_match(pattern, rel)) {
                    continue;
                }

                if (!add_text_file((rel_path / rel).string())) {
                    continue;
                }

                if (++count >= FILE_GLOB_MAX_RESULTS) {
                    console::error("Maximum number of globbed files allowed (%zu) reached.\n", FILE_GLOB_MAX_RESULTS);
                    break;
                }
            }
            continue;
        } else {
            // not a command
            cur_msg += buffer;
        }

        // generate response
        if (add_user_msg) {
            ctx_cli.messages.push_back({
                {"role",    "user"},
                {"content", cur_msg}
            });
            cur_msg.clear();
        }
        result_timings timings;
        std::string assistant_content = ctx_cli.generate_completion(timings);
        ctx_cli.messages.push_back({
            {"role",    "assistant"},
            {"content", assistant_content}
        });
        console::log("\n");

        if (params.show_timings) {
            console::set_display(DISPLAY_TYPE_INFO);
            console::log("\n");
            console::log("[ Prompt: %.1f t/s | Generation: %.1f t/s ]\n", timings.prompt_per_second, timings.predicted_per_second);
            console::set_display(DISPLAY_TYPE_RESET);
        }

        if (params.single_turn) {
            break;
        }
    }

    console::set_display(DISPLAY_TYPE_RESET);

    console::log("\nExiting...\n");
    ctx_cli.ctx_server.terminate();
    inference_thread.join();

    // bump the log level to display timings
    common_log_set_verbosity_thold(LOG_LEVEL_INFO);
    llama_memory_breakdown_print(ctx_cli.ctx_server.get_llama_context());

    return 0;
}
