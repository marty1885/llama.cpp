#include "rwkv-session.h"
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>

namespace fs = std::filesystem;
static void check(bool ok, const char * msg) { if (!ok) throw std::runtime_error(msg); }
static void write(std::ofstream & out, const std::vector<float> & x) {
    for (float v : x) check(std::isfinite(v), "nonfinite dump");
    out.write(reinterpret_cast<const char *>(x.data()), x.size()*sizeof(float));
    check(bool(out), "dump write failed");
}
static void load_backends() {
#ifdef RWKV_INTERP_BACKEND_DIR
    ggml_backend_load_all_from_path(RWKV_INTERP_BACKEND_DIR);
#else
    ggml_backend_load_all();
#endif
}
struct trace {
    int layers, embd, head, cursor = 0, norm_cursor = 0;
    std::vector<float> data;
    std::vector<bool> seen;
    std::exception_ptr error;
    bool enabled = true;
    int patch_layer = -1, patch_hits = 0;
    std::vector<float> patch;
    trace(int l, int e, int h) : layers(l), embd(e), head(h), data(size_t(l)*11*e), seen(size_t(l)*11) {}
    void clear() { cursor = norm_cursor = 0; std::fill(seen.begin(), seen.end(), false); error = {}; }
    void get(ggml_tensor * t, int layer, int field, size_t offset = 0) {
        check(layer >= 0 && layer < layers, "trace layer bounds");
        check(t->type == GGML_TYPE_F32 && ggml_is_contiguous(t), "trace requires contiguous F32");
        if (offset + size_t(embd)*sizeof(float) > ggml_nbytes(t)) throw std::runtime_error("trace shape mismatch: " + std::string(t->name) + " layer=" + std::to_string(layer) + " field=" + std::to_string(field) + " bytes=" + std::to_string(ggml_nbytes(t)));
        auto index = size_t(layer)*11 + field;
        check(!seen[index], "duplicate trace field");
        ggml_backend_tensor_get(t, data.data() + index*embd, offset, size_t(embd)*sizeof(float));
        seen[index] = true;
    }
    static bool callback(ggml_tensor * t, bool ask, void * user) noexcept {
        auto & s = *static_cast<trace *>(user);
        if (!s.enabled) return !ask;
        const std::string name = t->name;
        const bool wkv = t->op == GGML_OP_RWKV_WKV7;
        const bool norm = t->op == GGML_OP_NORM && t->ne[0] == s.head && t->ne[1] == s.embd/s.head;
        const auto dash = name.find_last_of('-');
        const bool numbered = dash != std::string::npos && dash + 1 < name.size() && std::all_of(name.begin() + dash + 1, name.end(), [](char c) { return c >= '0' && c <= '9'; });
        const bool named = numbered && (name.starts_with("attn_norm-") || name.starts_with("ffn_norm-") || name.starts_with("l_out-"));
        if (ask) return wkv || norm || named;
        try {
            if (wkv) {
                int layer = s.cursor++;
                check(t->src[0]->ne[2] == 1 && t->src[6]->ne[1] == 1, "one token and sequence required");
                for (int f = 0; f < 6; ++f) s.get(t->src[f], layer, f);
                s.get(t, layer, 6);
            } else if (norm) {
                s.get(t, s.norm_cursor++, 7);
            } else if (named) {
                int field = name.starts_with("attn_norm-") ? 8 : name.starts_with("ffn_norm-") ? 9 : 10;
                int layer = std::stoi(name.substr(name.find_last_of('-') + 1));
                if (field == 10 && layer == s.patch_layer) {
                    check(t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) && ggml_nbytes(t) == size_t(s.embd)*sizeof(float), "patch tensor shape mismatch");
                    check(s.patch.size() == size_t(s.embd), "patch vector shape mismatch");
                    for (float value : s.patch) check(std::isfinite(value), "nonfinite patch");
                    ggml_backend_tensor_set(t, s.patch.data(), 0, s.patch.size()*sizeof(float));
                    ++s.patch_hits;
                }
                s.get(t, layer, field);
            }
            return true;
        } catch (...) { s.error = std::current_exception(); return false; }
    }
    void validate() {
        if (error) std::rethrow_exception(error);
        check(cursor == layers && norm_cursor == layers, "missing recurrent operation");
        check(std::all_of(seen.begin(), seen.end(), [](bool x) { return x; }), "missing trace field");
    }
};
static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    int n = llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, false, false);
    check(n < 0, "empty or invalid tokenization");
    std::vector<llama_token> result(-n);
    n = llama_tokenize(vocab, text.data(), text.size(), result.data(), result.size(), false, false);
    check(n > 0, "tokenization failed"); result.resize(n); return result;
}
static std::string hex(const std::string & text) {
    static const char digits[] = "0123456789abcdef";
    std::string out;
    for (unsigned char c : text) { out += digits[c >> 4]; out += digits[c & 15]; }
    return out;
}
int main(int argc, char ** argv) {
    try {
        check(argc == 4 || (argc == 5 && std::string(argv[4]) == "--vulkan"), "usage: collect MODEL PLAN.tsv OUTPUT [--vulkan]");
        fs::path root(argv[3]);
        check(!fs::exists(root), "output exists; use a fresh directory");
        fs::create_directories(root);
        const bool vulkan = argc == 5;
        if (vulkan) load_backends();
        llama_backend_init();
        auto mp = llama_model_default_params(); mp.n_gpu_layers = vulkan ? 99 : 0;
        if (vulkan) {
            auto * device = ggml_backend_dev_by_name("Vulkan0");
            check(device != nullptr, "Vulkan0 unavailable");
            static ggml_backend_dev_t devices[2];
            devices[0] = device;
            mp.devices = devices;
        }
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], mp), llama_model_free);
        check(bool(model), "model load failed");
        char arch[64], head_text[64];
        llama_model_meta_val_str(model.get(), "general.architecture", arch, sizeof(arch));
        check(std::string(arch) == "rwkv7", "collector requires pure RWKV7");
        llama_model_meta_val_str(model.get(), "rwkv7.wkv.head_size", head_text, sizeof(head_text));
        const int layers = llama_model_n_layer(model.get()), embd = llama_model_n_embd(model.get()), head = std::stoi(head_text);
        check(head > 0 && embd % head == 0, "invalid head dimensions");
        trace tr(layers, embd, head);
        auto cp = llama_context_default_params();
        cp.n_ctx = 512; cp.n_batch = cp.n_ubatch = 1; cp.n_seq_max = 1; cp.n_rs_seq = 0;
        cp.n_threads = cp.n_threads_batch = 8; cp.type_k = cp.type_v = GGML_TYPE_F32;
        cp.cb_eval = trace::callback; cp.cb_eval_user_data = &tr;
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
        check(bool(ctx), "context failed");
        rwkv::session session(ctx.get());
        const auto * vocab = llama_model_get_vocab(model.get());
        const std::vector<int> selected{0, layers/4, layers/2, 3*layers/4, layers-1};
        std::ofstream config(root/"config.tsv");
        config << "layers\t" << layers << "\nembd\t" << embd << "\nhead_size\t" << head << "\nfields\tr,w,k,v,a,b,wkv_read,group_norm,attn_norm,ffn_norm,layer_out\nselected_layers\t";
        for (auto l : selected) config << l << ',';
        config << "\nmodel\t" << fs::absolute(argv[1]).string() << "\nexecution\t" << (vulkan ? "Vulkan0" : "CPU") << " F32 state; one token; one sequence; eight threads; no rollback; no BOS or chat template\n";
        config.close();
        std::ifstream plan(argv[2]); check(bool(plan), "plan open failed");
        std::ofstream tokens(root/"tokens.tsv"), segments(root/"segments.tsv"), scores(root/"scores.tsv");
        tokens << "segment\tindex\tposition\ttoken\tpiece_hex\n";
        segments << "id\tparent\tkind\tstart_pos\tn_tokens\n";
        scores << std::setprecision(17);
        scores << "segment\tcandidate_hex\tn_tokens\tlog_probability\n";
        std::vector<std::string> candidate_colors;
        rwkv::checkpoint story{}, updated{};
        std::string story_id, updated_id;
        int count = 0;
        std::string line;
        while (std::getline(plan, line)) {
            std::istringstream row(line);
            std::string id, parent, kind, text, colors;
            std::getline(row, id, '\t'); std::getline(row, parent, '\t'); std::getline(row, kind, '\t'); std::getline(row, text, '\t'); std::getline(row, colors);
            for (size_t p = 0; (p = text.find("\\n", p)) != std::string::npos;) text.replace(p, 2, "\n");
            check(!id.empty() && id.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789_-") == std::string::npos, "invalid segment id");
            if (kind == "story") {
                check(llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, 0, -1), "reset failed");
                story_id = id; updated_id.clear();
            } else if (kind == "update" || kind == "query_story") {
                check(parent == story_id, "parent does not match active story");
                session.restore(0, story);
            } else { check(kind == "query_update", "unknown segment kind"); check(parent == updated_id, "parent does not match active update"); session.restore(0, updated); }
            const auto separator = colors.find(','); check(separator != std::string::npos, "missing candidate colors");
            candidate_colors.clear();
            std::istringstream color_row(colors);
            std::string color;
            while (std::getline(color_row, color, ',')) candidate_colors.push_back(color);
            // The plan supplies independently tokenized segments; exact token IDs are the authoritative input.
            const auto input = tokenize(vocab, text);
            const int start = session.position(0);
            auto dir = root/id; fs::create_directory(dir);
            std::ofstream states(dir/"wkv.f32", std::ios::binary), shifts(dir/"shifts.f32", std::ios::binary), operands(dir/"vectors.f32", std::ios::binary), logits_file(dir/"logits.f32", std::ios::binary);
            const auto save_selected = [&]() {
                for (int l : selected) {
                    if (session.position(0) == 0) {
                        write(states, std::vector<float>(size_t(embd)*head, 0));
                        write(shifts, std::vector<float>(size_t(embd)*2, 0));
                    } else {
                        write(states, session.capture(0, l, rwkv::state_kind::wkv).values);
                        write(shifts, session.capture(0, l, rwkv::state_kind::shift_att).values);
                        write(shifts, session.capture(0, l, rwkv::state_kind::shift_ffn).values);
                    }
                }
            };
            save_selected();
            std::vector<float> logits;
            for (size_t i = 0; i < input.size(); ++i) {
                tr.clear(); rwkv::session::request r{0, {input[i]}};
                logits = session.prefill(std::span(&r, 1)).front().logits;
                tr.validate(); write(operands, tr.data); write(logits_file, logits); save_selected();
                char piece[512]; int n = llama_token_to_piece(vocab, input[i], piece, sizeof(piece), 0, true);
                check(n >= 0, "token piece too large");
                tokens << id << '\t' << i << '\t' << start + i << '\t' << input[i] << '\t' << hex(std::string(piece, n)) << '\n';
            }
            auto checkpoint = session.capture(0);
            std::ofstream boundary(dir/"boundary.f32", std::ios::binary);
            for (int l = 0; l < layers; ++l) {
                write(boundary, session.capture(0, l, rwkv::state_kind::wkv).values);
                write(boundary, session.capture(0, l, rwkv::state_kind::shift_att).values);
                write(boundary, session.capture(0, l, rwkv::state_kind::shift_ffn).values);
            }
            boundary.close();
            if (kind == "story") {
                story = checkpoint;
                if (count == 0) {
                    check(llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, 0, -1), "control reset failed");
                    tr.enabled = false;
                    std::vector<float> control;
                    for (auto token : input) {
                        rwkv::session::request r{0, {token}};
                        control = session.prefill(std::span(&r, 1)).front().logits;
                    }
                    check(control == logits && session.capture(0).bytes == checkpoint.bytes, "instrumentation changes execution");
                    tr.enabled = true;
                    session.restore(0, checkpoint);
                    std::ofstream observer(root/"observer-control.txt");
                    observer << "First story: tracing enabled/disabled gives identical final logits and full checkpoint.\n";
                }
            }
            if (kind == "update") { updated = checkpoint; updated_id = id; }
            if (kind.starts_with("query")) {
                for (const auto & color : candidate_colors) {
                    session.restore(0, checkpoint);
                    auto current = logits;
                    auto candidate = tokenize(vocab, " " + color);
                    double score = 0;
                    for (auto token : candidate) {
                        double maximum = *std::max_element(current.begin(), current.end()), sum = 0;
                        for (float x : current) sum += std::exp(double(x) - maximum);
                        score += current[token] - maximum - std::log(sum);
                        tr.clear(); rwkv::session::request r{0, {token}};
                        current = session.prefill(std::span(&r, 1)).front().logits;
                        tr.validate();
                    }
                    scores << id << '\t' << hex(color) << '\t' << candidate.size() << '\t' << score << '\n';
                }
            }
            segments << id << '\t' << parent << '\t' << kind << '\t' << start << '\t' << input.size() << '\n';
            check(bool(tokens) && bool(segments) && bool(scores), "metadata write failed");
            tokens.flush(); segments.flush(); scores.flush();
            for (auto * out : {&states, &shifts, &operands, &logits_file}) { out->close(); check(bool(*out), "dump close failed"); }
            std::ofstream done(dir/"DONE"); done << "complete\n";
            std::cout << ++count << " " << id << " tokens=" << input.size() << std::endl;
        }
        check(plan.eof(), "plan read failed");
        std::ofstream done(root/"DONE"); done << count << " segments\n";
        return 0;
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
