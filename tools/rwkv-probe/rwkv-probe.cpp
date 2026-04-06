// rwkv-probe: batch RWKV state extraction for alignment research.
//
// Loads model ONCE, iterates over a JSON file of prompts.  For each prompt:
//   1. Clears recurrent state
//   2. Processes prompt (batched for speed)
//   3. Captures state snapshot (phase=0, post-prompt)
//   4. Generates tokens until EOG or --n-predict
//   5. Captures state snapshot (phase=1, end-of-generation)
//   6. Writes per-prompt ROOT file + response .txt sidecar
//
// Usage:
//   llama-rwkv-probe -m model.gguf --prompts prompts.json --output-dir ./states/ [-n 256]

#include "common.h"
#include "llama.h"
#include "sampling.h"

#include "TFile.h"
#include "TNamed.h"
#include "TTree.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using json = nlohmann::json;

static void print_usage(const char * prog) {
    printf("usage: %s -m MODEL --prompts FILE --output-dir DIR [options]\n\n", prog);
    printf("  -m MODEL         path to GGUF model (required)\n");
    printf("  --prompts FILE   JSON array of {id, prompt, ...} (required)\n");
    printf("  --output-dir DIR directory for output ROOT + txt files (required)\n");
    printf("  -n N             max tokens to generate per prompt (default: 256)\n");
    printf("  --n-ctx N        context size (default: 2048)\n");
    printf("  --seed N         RNG seed (default: 42)\n");
    printf("  --steer FILE     direction vector file (one float per line, head_size*head_size values)\n");
    printf("  --steer-head N   which attention head to steer (default: 9)\n");
    printf("  --steer-layer N  which layer's S state to steer (default: 0, i.e. first layer)\n");
    printf("  --steer-alpha F  steering strength (default: 1.0, negative = opposite direction)\n");
    printf("  -h, --help       show this help\n");
}

static std::vector<float> load_direction(const std::string & path) {
    std::vector<float> dir;
    std::ifstream ifs(path);
    if (!ifs) { fprintf(stderr, "error: cannot open direction file %s\n", path.c_str()); return dir; }
    float v;
    while (ifs >> v) { dir.push_back(v); }
    // normalize to unit vector — alpha controls magnitude
    float norm = 0;
    for (float x : dir) { norm += x * x; }
    norm = sqrtf(norm);
    if (norm > 0) { for (float & x : dir) { x /= norm; } }
    return dir;
}

// helper: tokenize a string
static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    int n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(),
                           nullptr, 0, true, true);
    if (n < 0) { n = -n; }
    std::vector<llama_token> tokens((size_t) n);
    llama_tokenize(vocab, text.c_str(), (int32_t) text.size(),
                   tokens.data(), (int32_t) tokens.size(), true, true);
    return tokens;
}

// helper: decode a token to text
static std::string token_to_text(llama_context * ctx, llama_token tok) {
    return common_token_to_piece(ctx, tok, false);
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string prompts_path;
    std::string output_dir;
    std::string steer_path;
    int         n_predict    = 256;
    int         n_ctx        = 2048;
    uint32_t    seed         = 42;
    int         steer_head   = 9;
    int         steer_layer  = 0;
    float       steer_alpha  = 1.0f;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if      ((arg == "-m")            && i+1 < argc) { model_path   = argv[++i]; }
        else if ((arg == "--prompts")     && i+1 < argc) { prompts_path = argv[++i]; }
        else if ((arg == "--output-dir")  && i+1 < argc) { output_dir   = argv[++i]; }
        else if ((arg == "-n")            && i+1 < argc) { n_predict    = std::atoi(argv[++i]); }
        else if ((arg == "--n-ctx")       && i+1 < argc) { n_ctx        = std::atoi(argv[++i]); }
        else if ((arg == "--seed")        && i+1 < argc) { seed         = (uint32_t) std::atoi(argv[++i]); }
        else if ((arg == "--steer")       && i+1 < argc) { steer_path   = argv[++i]; }
        else if ((arg == "--steer-head")  && i+1 < argc) { steer_head   = std::atoi(argv[++i]); }
        else if ((arg == "--steer-layer") && i+1 < argc) { steer_layer  = std::atoi(argv[++i]); }
        else if ((arg == "--steer-alpha") && i+1 < argc) { steer_alpha  = (float) std::atof(argv[++i]); }
        else if (arg == "-h" || arg == "--help") { print_usage(argv[0]); return 0; }
        else { fprintf(stderr, "unknown argument: %s\n", arg.c_str()); print_usage(argv[0]); return 1; }
    }

    if (model_path.empty() || prompts_path.empty() || output_dir.empty()) {
        fprintf(stderr, "error: -m, --prompts and --output-dir are required\n\n");
        print_usage(argv[0]);
        return 1;
    }

    // ---- load prompts --------------------------------------------------------
    json prompts;
    {
        std::ifstream ifs(prompts_path);
        if (!ifs) { fprintf(stderr, "error: cannot open %s\n", prompts_path.c_str()); return 1; }
        prompts = json::parse(ifs);
    }
    fprintf(stderr, "loaded %zu prompts from %s\n", prompts.size(), prompts_path.c_str());

    // ---- llama init ----------------------------------------------------------
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) { fprintf(stderr, "error: failed to load model\n"); return 1; }
    if (!llama_model_is_recurrent(model)) {
        fprintf(stderr, "error: model is not recurrent\n");
        llama_model_free(model); return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = (uint32_t) n_ctx;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "error: failed to create context\n"); llama_model_free(model); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // ---- geometry ------------------------------------------------------------
    const int n_layer    = llama_model_n_layer(model);
    const int n_embd_r   = llama_model_n_embd_r(model);
    const int n_embd_s   = llama_model_n_embd_s(model);
    const int n_embd     = llama_model_n_embd(model);
    const int n_half     = n_embd_r / 2;

    fprintf(stderr, "n_layer=%d  n_embd=%d  n_embd_r=%d  n_embd_s=%d\n",
            n_layer, n_embd, n_embd_r, n_embd_s);

    // ---- load steering direction (optional) -----------------------------------
    const int head_size = n_embd_s / n_embd;  // = wkv_head_size (typically 64)
    const int n_head    = n_embd / head_size;
    std::vector<float> steer_direction;
    if (!steer_path.empty()) {
        steer_direction = load_direction(steer_path);
        if ((int) steer_direction.size() != head_size * head_size) {
            fprintf(stderr, "error: direction has %zu values, expected %d (head_size^2 = %d*%d)\n",
                    steer_direction.size(), head_size * head_size, head_size, head_size);
            return 1;
        }
        if (steer_head < 0 || steer_head >= n_head) {
            fprintf(stderr, "error: --steer-head %d out of range [0, %d)\n", steer_head, n_head);
            return 1;
        }
        if (steer_layer < 0 || steer_layer >= n_layer) {
            fprintf(stderr, "error: --steer-layer %d out of range [0, %d)\n", steer_layer, n_layer);
            return 1;
        }
        fprintf(stderr, "steering: layer=%d head=%d alpha=%.3f (%zu-dim direction)\n",
                steer_layer, steer_head, steer_alpha, steer_direction.size());
    }

    // ---- reusable state buffers ---------------------------------------------
    std::vector<float> r_flat(n_layer * n_embd_r);
    std::vector<float> s_flat(n_layer * n_embd_s);
    std::vector<std::vector<Float_t>> r_att(n_layer, std::vector<Float_t>(n_half, 0.0f));
    std::vector<std::vector<Float_t>> r_ffn(n_layer, std::vector<Float_t>(n_half, 0.0f));
    std::vector<std::vector<Float_t>> s_wkv(n_layer, std::vector<Float_t>(n_embd_s, 0.0f));

    // helper: extract state into branch buffers
    auto capture_state = [&]() {
        llama_recurrent_state_get_f32(ctx, 0, r_flat.data(), s_flat.data());
        for (int il = 0; il < n_layer; ++il) {
            const float * r = r_flat.data() + il * n_embd_r;
            std::copy(r, r + n_half, r_att[il].begin());
            std::copy(r + n_half, r + n_embd_r, r_ffn[il].begin());
            const float * s = s_flat.data() + il * n_embd_s;
            std::copy(s, s + n_embd_s, s_wkv[il].begin());
        }
    };

    // ---- process each prompt -------------------------------------------------
    const size_t n_total = prompts.size();

    for (size_t pi = 0; pi < n_total; ++pi) {
        const auto & entry = prompts[pi];
        const std::string pid         = entry.value("id", std::to_string(pi));
        const std::string prompt_text = entry.at("prompt").get<std::string>();

        fprintf(stderr, "\n[%zu/%zu] %s: %s\n", pi + 1, n_total, pid.c_str(),
                prompt_text.substr(0, 60).c_str());

        // ---- clear state for fresh prompt ------------------------------------
        llama_memory_clear(llama_get_memory(ctx), true);

        // ---- tokenize --------------------------------------------------------
        std::vector<llama_token> tokens = tokenize(vocab, prompt_text);
        fprintf(stderr, "  prompt: %zu tokens\n", tokens.size());

        // ---- sampler (must be created before prompt so we can accept tokens) -
        common_params_sampling sparams;
        sparams.seed = seed;
        common_sampler * smpl = common_sampler_init(model, sparams);

        // ---- process prompt (batched for speed) ------------------------------
        llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t) tokens.size());
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "  error: llama_decode failed during prompt\n");
            common_sampler_free(smpl);
            continue;
        }

        // feed prompt tokens into the sampler (repetition penalty context)
        for (const auto & tok : tokens) {
            common_sampler_accept(smpl, tok, false);
        }

        // ---- open ROOT file --------------------------------------------------
        std::string root_path = output_dir + "/" + pid + ".root";
        TFile * rfile = TFile::Open(root_path.c_str(), "RECREATE");
        if (!rfile || rfile->IsZombie()) {
            fprintf(stderr, "  error: cannot open %s\n", root_path.c_str());
            continue;
        }

        TDirectory * dir = rfile->mkdir("prompt");
        dir->cd();

        TNamed("prompt_text", prompt_text.c_str()).Write();
        TNamed("prompt_id",   pid.c_str()).Write();
        TNamed("n_layer",     std::to_string(n_layer).c_str()).Write();
        TNamed("n_embd_s",    std::to_string(n_embd_s).c_str()).Write();

        TTree * tree = new TTree("states", "RWKV state: prompt-end + EOG");
        tree->SetAutoSave(0);

        Int_t br_phase = 0;
        tree->Branch("phase", &br_phase, "phase/I");
        for (int il = 0; il < n_layer; ++il) {
            tree->Branch(Form("r_att_L%d", il), r_att[il].data(), Form("r_att_L%d[%d]/F", il, n_half));
            tree->Branch(Form("r_ffn_L%d", il), r_ffn[il].data(), Form("r_ffn_L%d[%d]/F", il, n_half));
            tree->Branch(Form("s_L%d", il),     s_wkv[il].data(), Form("s_L%d[%d]/F",     il, n_embd_s));
        }

        // ---- capture post-prompt state (phase=0) -----------------------------
        capture_state();
        br_phase = 0;
        tree->Fill();

        // ---- apply steering (nudge head's WKV state after prompt) -----------
        if (!steer_direction.empty()) {
            // read full S state for the target layer
            llama_recurrent_state_get_f32(ctx, 0, nullptr, s_flat.data());

            // offset into the target layer's S state, then into the target head
            float * layer_s = s_flat.data() + steer_layer * n_embd_s;
            float * head_s  = layer_s + steer_head * head_size * head_size;

            // s[head] += alpha * direction
            for (int j = 0; j < head_size * head_size; ++j) {
                head_s[j] += steer_alpha * steer_direction[j];
            }

            // write modified state back
            llama_recurrent_state_set_f32(ctx, 0, nullptr, s_flat.data());
        }

        // ---- generation phase ------------------------------------------------
        const llama_token eos_token = llama_vocab_eos(vocab);

        std::string response;
        for (int i = 0; i < n_predict; ++i) {
            llama_token tok = common_sampler_sample(smpl, ctx, -1);
            common_sampler_accept(smpl, tok, true);
            if (tok == eos_token) {
                break;
            }
            response += token_to_text(ctx, tok);

            // decode the generated token to advance the recurrent state
            llama_batch gen_batch = llama_batch_get_one(&tok, 1);
            if (llama_decode(ctx, gen_batch) != 0) {
                fprintf(stderr, "  error: llama_decode failed at gen step %d\n", i);
                break;
            }
        }

        common_sampler_free(smpl);

        // ---- capture EOG state (phase=1) -------------------------------------
        capture_state();
        br_phase = 1;
        tree->Fill();

        // ---- write ROOT file -------------------------------------------------
        dir->cd();
        tree->Write("", TObject::kOverwrite);
        rfile->Close();
        delete rfile;

        // ---- write response sidecar ------------------------------------------
        std::string txt_path = output_dir + "/" + pid + ".txt";
        {
            std::ofstream ofs(txt_path);
            if (ofs) { ofs << response; }
        }

        fprintf(stderr, "  -> %s (%zu chars)\n", root_path.c_str(), response.size());
        fprintf(stderr, "  -> %s\n", response.substr(0, 100).c_str());
    }

    fprintf(stderr, "\ndone: %zu prompts processed\n", n_total);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
