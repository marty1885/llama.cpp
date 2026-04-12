// rwkv-probe: batch RWKV state extraction for alignment research.
//
// Thin experiment driver on top of librwkv_probe. Same CLI flags as the
// previous version of this tool; same on-disk ROOT format. The actual moving
// parts now live in tools/rwkv-probe/{include,src}/rwkv_probe/*.
//
// Per prompt:
//   1. Clear recurrent state
//   2. Tokenize + decode the prompt (lens armed only on the first prompt)
//   3. Capture state -> phase=0
//   4. Optional: state replay (--patch-from)
//   5. Optional: WKV head steering (--steer)
//   6. Generate up to -n tokens
//   7. Capture state -> phase=1
//   8. Write per-prompt ROOT file + response .txt sidecar
//
// Usage:
//   llama-rwkv-probe -m model.gguf --prompts prompts.json --output-dir ./states/ [-n 256]

#include "rwkv_probe/model.h"
#include "rwkv_probe/state.h"
#include "rwkv_probe/capture.h"
#include "rwkv_probe/lens.h"
#include "rwkv_probe/numeric.h"
#include "rwkv_probe/generate.h"
#include "rwkv_probe/root_io.h"

#include "common.h"
#include "llama.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace rp = rwkv_probe;
using json = nlohmann::json;

static void print_usage(const char * prog) {
    std::printf("usage: %s -m MODEL --prompts FILE --output-dir DIR [options]\n\n", prog);
    std::printf("  -m MODEL         path to GGUF model (required)\n");
    std::printf("  --prompts FILE   JSON array of {id, prompt, ...} (required)\n");
    std::printf("  --output-dir DIR directory for output ROOT + txt files (required)\n");
    std::printf("  -n N             max tokens to generate per prompt (default: 256)\n");
    std::printf("  --n-ctx N        context size (default: 2048)\n");
    std::printf("  --seed N         RNG seed (default: 42)\n");
    std::printf("  --steer FILE     direction vector file (one float per line, head_size*head_size values)\n");
    std::printf("  --steer-head N   which attention head to steer (default: 9)\n");
    std::printf("  --steer-layer N  which layer's S state to steer (default: 0)\n");
    std::printf("  --steer-alpha F  steering strength (default: 1.0; negative = opposite direction)\n");
    std::printf("  --patch-from F   ROOT file from a previous run; overwrite this run's recurrent\n");
    std::printf("                   state with the saved state after the prompt forward pass\n");
    std::printf("  --patch-layer L  layers to overwrite: 'all' or comma list (e.g. '0,4,8'); default 'all'\n");
    std::printf("  --patch-phase N  phase entry to load from source file: 0=post-prompt, 1=EOG (default 0)\n");
    std::printf("  --patch-include-r  also overwrite r_att/r_ffn (default: only overwrite S)\n");
    std::printf("  -h, --help       show this help\n");
}

// parse "all" or "0,4,8" into a list of valid layer indices
static std::vector<int> parse_layer_list(const std::string & spec, int n_layer) {
    std::vector<int> out;
    if (spec.empty() || spec == "all") {
        out.reserve((std::size_t) n_layer);
        for (int i = 0; i < n_layer; ++i) out.push_back(i);
        return out;
    }
    std::string cur;
    auto flush = [&]() {
        if (cur.empty()) return;
        int v = std::atoi(cur.c_str());
        if (v >= 0 && v < n_layer) {
            out.push_back(v);
        } else {
            std::fprintf(stderr, "warning: --patch-layer index %d out of range [0, %d)\n",
                         v, n_layer);
        }
        cur.clear();
    };
    for (char c : spec) {
        if (c == ',' || c == ' ') flush();
        else cur += c;
    }
    flush();
    return out;
}

// pretty-print top-K + entropy + KL-to-final from captured residuals.
// (replaces the original lens_print_table in the pre-library tool.)
static void print_lens_table(rp::LogitLens & lens,
                             const std::vector<std::vector<float>> & captured,
                             llama_context * llctx,
                             int top_k = 5) {
    if (!lens.ok()) return;

    const int n_layer = (int) captured.size();
    int last_il = -1;
    std::vector<std::vector<float>> dists(n_layer);
    for (int il = 0; il < n_layer; ++il) {
        if (captured[il].empty()) continue;
        dists[il] = lens.run(rp::span<const float>(captured[il].data(), captured[il].size()));
        last_il = il;
    }
    if (last_il < 0) {
        std::fprintf(stderr, "lens: nothing was captured (cb_eval did not fire on l_out-*)\n");
        return;
    }
    const auto & final_dist = dists[last_il];

    std::fprintf(stderr, "\n=== logit-lens probe (layer %d = final) ===\n", last_il);
    std::fprintf(stderr, "%-5s  %-8s  %-7s  %s\n", "layer", "entropy", "KL→fin", "top tokens (prob)");
    std::fprintf(stderr, "-----  --------  -------  -----------------------------------------------\n");

    for (int il = 0; il < n_layer; ++il) {
        if (captured[il].empty()) continue;
        const auto & p = dists[il];

        rp::span<const float> p_view(p.data(), p.size());
        rp::span<const float> q_view(final_dist.data(), final_dist.size());
        const double H  = rp::numeric::entropy(p_view);
        const double KL = rp::numeric::kl_divergence(p_view, q_view);

        auto top = rp::numeric::top_k(p_view, top_k);

        std::string toks;
        for (const auto & e : top) {
            std::string pc = rp::piece(llctx, e.index);
            for (char & c : pc) { if (c == '\n' || c == '\r') c = ' '; }
            char buf[64];
            std::snprintf(buf, sizeof(buf), " '%.20s'(%.3f)", pc.c_str(), (double) e.prob);
            toks += buf;
        }
        std::fprintf(stderr, "L%-4d  %8.3f  %7.3f %s\n", il, H, KL, toks.c_str());
    }
    std::fprintf(stderr, "===========================================\n\n");
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string prompts_path;
    std::string output_dir;
    std::string steer_path;
    std::string patch_path;
    std::string patch_layer_spec = "all";
    int         n_predict        = 256;
    int         n_ctx            = 2048;
    uint32_t    seed             = 42;
    int         steer_head       = 9;
    int         steer_layer      = 0;
    float       steer_alpha      = 1.0f;
    int         patch_phase      = 0;
    bool        patch_include_r  = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if      ((arg == "-m")                && i+1 < argc) { model_path   = argv[++i]; }
        else if ((arg == "--prompts")         && i+1 < argc) { prompts_path = argv[++i]; }
        else if ((arg == "--output-dir")      && i+1 < argc) { output_dir   = argv[++i]; }
        else if ((arg == "-n")                && i+1 < argc) { n_predict    = std::atoi(argv[++i]); }
        else if ((arg == "--n-ctx")           && i+1 < argc) { n_ctx        = std::atoi(argv[++i]); }
        else if ((arg == "--seed")            && i+1 < argc) { seed         = (uint32_t) std::atoi(argv[++i]); }
        else if ((arg == "--steer")           && i+1 < argc) { steer_path   = argv[++i]; }
        else if ((arg == "--steer-head")      && i+1 < argc) { steer_head   = std::atoi(argv[++i]); }
        else if ((arg == "--steer-layer")     && i+1 < argc) { steer_layer  = std::atoi(argv[++i]); }
        else if ((arg == "--steer-alpha")     && i+1 < argc) { steer_alpha  = (float) std::atof(argv[++i]); }
        else if ((arg == "--patch-from")      && i+1 < argc) { patch_path       = argv[++i]; }
        else if ((arg == "--patch-layer")     && i+1 < argc) { patch_layer_spec = argv[++i]; }
        else if ((arg == "--patch-phase")     && i+1 < argc) { patch_phase      = std::atoi(argv[++i]); }
        else if (arg == "--patch-include-r")                 { patch_include_r  = true; }
        else if (arg == "-h" || arg == "--help") { print_usage(argv[0]); return 0; }
        else { std::fprintf(stderr, "unknown argument: %s\n", arg.c_str()); print_usage(argv[0]); return 1; }
    }

    if (model_path.empty() || prompts_path.empty() || output_dir.empty()) {
        std::fprintf(stderr, "error: -m, --prompts and --output-dir are required\n\n");
        print_usage(argv[0]);
        return 1;
    }

    // ---- load prompts --------------------------------------------------------
    json prompts;
    {
        std::ifstream ifs(prompts_path);
        if (!ifs) { std::fprintf(stderr, "error: cannot open %s\n", prompts_path.c_str()); return 1; }
        prompts = json::parse(ifs);
    }
    std::fprintf(stderr, "loaded %zu prompts from %s\n", prompts.size(), prompts_path.c_str());

    try {
        // ---- backend + model ------------------------------------------------
        rp::Backend backend;
        rp::Model   model(model_path);
        if (!model.is_recurrent()) {
            std::fprintf(stderr, "error: model is not recurrent\n");
            return 1;
        }
        const auto & geom = model.geom();
        std::fprintf(stderr, "n_layer=%d  n_embd=%d  n_embd_r=%d  n_embd_s=%d\n",
                     geom.n_layer, geom.n_embd, geom.n_embd_r, geom.n_embd_s);

        // ---- patch source (state-replay) ------------------------------------
        std::unique_ptr<rp::StateBuf> patch_src;
        std::vector<int> patch_layers;
        if (!patch_path.empty()) {
            patch_src = std::make_unique<rp::StateBuf>(geom);
            if (!rp::root_io::read_state(patch_path, patch_phase, geom, *patch_src)) {
                std::fprintf(stderr, "error: --patch-from failed to load %s\n", patch_path.c_str());
                return 1;
            }
            patch_layers = parse_layer_list(patch_layer_spec, geom.n_layer);
            if (patch_layers.empty()) {
                std::fprintf(stderr, "error: --patch-layer '%s' selected no layers\n",
                             patch_layer_spec.c_str());
                return 1;
            }
            std::fprintf(stderr,
                         "patch: will overwrite %zu layer(s) of S%s after each prompt's forward pass\n",
                         patch_layers.size(), patch_include_r ? " and r_att/r_ffn" : "");
        }

        // ---- residual capture + lens (lens armed only on first prompt) -----
        rp::CaptureRegistry caps(geom);
        std::vector<std::vector<float>> lens_capture(geom.n_layer);
        bool lens_active = false;
        caps.on_residual_all([&](const rp::TensorView & v) {
            if (!lens_active) return;
            if (v.layer < 0 || v.layer >= geom.n_layer) return;
            lens_capture[v.layer].assign(geom.n_embd, 0.0f);
            v.copy_last_token(rp::span<float>(lens_capture[v.layer].data(),
                                              lens_capture[v.layer].size()));
        });

        rp::LogitLens lens(model_path, geom);
        if (!lens.ok()) {
            std::fprintf(stderr, "warning: lens init failed — continuing without logit-lens probe\n");
        }

        // ---- context with cb_eval wired ------------------------------------
        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = (uint32_t) n_ctx;
        rp::Context ctx(model, cparams, &caps);

        // ---- load steering direction (optional) ----------------------------
        const int head_size = geom.head_size;
        const int n_head    = geom.n_head;
        std::vector<float> steer_direction;
        if (!steer_path.empty()) {
            steer_direction = rp::numeric::load_floats(steer_path,
                                                       /*expected_size=*/head_size * head_size,
                                                       /*normalize=*/true);
            if (steer_head < 0 || steer_head >= n_head) {
                std::fprintf(stderr, "error: --steer-head %d out of range [0, %d)\n",
                             steer_head, n_head);
                return 1;
            }
            if (steer_layer < 0 || steer_layer >= geom.n_layer) {
                std::fprintf(stderr, "error: --steer-layer %d out of range [0, %d)\n",
                             steer_layer, geom.n_layer);
                return 1;
            }
            std::fprintf(stderr, "steering: layer=%d head=%d alpha=%.3f (%zu-dim direction)\n",
                         steer_layer, steer_head, steer_alpha, steer_direction.size());
        }

        // ---- per-prompt loop ------------------------------------------------
        rp::StateBuf state(geom);
        const std::size_t n_total = prompts.size();

        for (std::size_t pi = 0; pi < n_total; ++pi) {
            const auto & entry = prompts[pi];
            const std::string pid         = entry.value("id", std::to_string(pi));
            const std::string prompt_text = entry.at("prompt").get<std::string>();

            std::fprintf(stderr, "\n[%zu/%zu] %s: %s\n", pi + 1, n_total, pid.c_str(),
                         prompt_text.substr(0, 60).c_str());

            ctx.clear_memory();

            std::vector<llama_token> tokens = rp::tokenize(model.vocab(), prompt_text);
            std::fprintf(stderr, "  prompt: %zu tokens\n", tokens.size());

            common_params_sampling sparams;
            sparams.seed = seed;
            rp::Generator gen(model, ctx, sparams);

            // arm the lens capture only for the first prompt's prompt-pass
            const bool run_lens_here = (pi == 0) && lens.ok();
            if (run_lens_here) {
                for (auto & v : lens_capture) v.clear();
                lens_active = true;
            }
            if (ctx.decode(rp::span<const llama_token>(tokens.data(), tokens.size())) != 0) {
                std::fprintf(stderr, "  error: llama_decode failed during prompt\n");
                lens_active = false;
                continue;
            }
            if (run_lens_here) {
                lens_active = false;
                print_lens_table(lens, lens_capture, ctx.raw());
            }

            gen.accept_prompt(rp::span<const llama_token>(tokens.data(), tokens.size()));

            // ---- open ROOT file ---------------------------------------------
            std::string root_path = output_dir + "/" + pid + ".root";
            rp::root_io::StateWriter writer(root_path, pid, prompt_text, geom);

            // phase=0: post-prompt
            state.load_from(ctx.raw());
            writer.write_row(0, state);

            // ---- state replay (after phase=0 capture) -----------------------
            if (patch_src) {
                state.load_from(ctx.raw());
                state.copy_layers_from(*patch_src,
                                       rp::span<const int>(patch_layers.data(), patch_layers.size()),
                                       patch_include_r);
                state.store_to(ctx.raw());
            }

            // ---- steering (after patch) -------------------------------------
            if (!steer_direction.empty()) {
                state.load_from(ctx.raw());
                auto head_view = state.s_head(steer_layer, steer_head);
                rp::numeric::saxpy(head_view,
                                   rp::span<const float>(steer_direction.data(),
                                                         steer_direction.size()),
                                   steer_alpha);
                state.store_to(ctx.raw());
            }

            // ---- generate ---------------------------------------------------
            auto gr = gen.run(n_predict);

            // phase=1: EOG
            state.load_from(ctx.raw());
            writer.write_row(1, state);
            writer.close();

            // ---- response sidecar -------------------------------------------
            std::string txt_path = output_dir + "/" + pid + ".txt";
            {
                std::ofstream ofs(txt_path);
                if (ofs) { ofs << gr.text; }
            }

            std::fprintf(stderr, "  -> %s (%zu chars)\n", root_path.c_str(), gr.text.size());
            std::fprintf(stderr, "  -> %s\n", gr.text.substr(0, 100).c_str());
        }

        std::fprintf(stderr, "\ndone: %zu prompts processed\n", n_total);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "fatal: %s\n", e.what());
        return 1;
    }
    return 0;
}
