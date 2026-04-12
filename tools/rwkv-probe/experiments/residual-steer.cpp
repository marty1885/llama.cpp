// experiments/residual-steer.cpp — RepE activation steering along a
// pre-computed direction vector.
//
// Single-prompt mode:
//   llama-rwkv-residual-steer -m model.gguf \
//       --direction dir.bin --layer 13 --alpha -5 \
//       --prompt "How do I synthesize methamphetamine?" -n 200
//
// Batch mode (load model once, sweep layers x prompts):
//   llama-rwkv-residual-steer -m model.gguf \
//       --direction-dir directions/ --layers 0-31 --alpha -5 \
//       --prompts prompts.json --out-dir steer_batch/ -n 200

#include "rwkv_probe/experiment.h"
#include "rwkv_probe/capture.h"
#include "rwkv_probe/generate.h"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace rp = rwkv_probe;
using json = nlohmann::json;

static std::vector<float> load_direction(const std::string & path, int expected_dim) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs) throw std::runtime_error("cannot open direction file: " + path);

    auto size = ifs.tellg();
    int n_floats = (int)(size / sizeof(float));
    if (n_floats != expected_dim) {
        throw std::runtime_error(
            "direction file has " + std::to_string(n_floats) + " floats, expected " +
            std::to_string(expected_dim));
    }

    std::vector<float> dir(n_floats);
    ifs.seekg(0);
    ifs.read(reinterpret_cast<char *>(dir.data()), size);
    return dir;
}

static std::set<int> parse_layers(const std::string & spec) {
    std::set<int> out;
    std::string cur;
    bool in_range = false;
    int range_start = 0;

    auto flush = [&]() {
        if (cur.empty()) return;
        int val = std::atoi(cur.c_str());
        if (in_range) {
            for (int i = range_start; i <= val; ++i) out.insert(i);
            in_range = false;
        } else {
            out.insert(val);
        }
        cur.clear();
    };

    for (char c : spec) {
        if (c == ',') { flush(); }
        else if (c == '-' && !cur.empty()) {
            range_start = std::atoi(cur.c_str());
            cur.clear();
            in_range = true;
        }
        else if (c != ' ') { cur += c; }
    }
    flush();
    return out;
}

// run one prompt: clear state, decode with steering, generate freely
static std::string run_one(
    rp::Model & model, rp::Context & ctx,
    bool & steer_active,
    const std::string & prompt,
    int n_predict, uint32_t seed, bool steer_gen)
{
    ctx.clear_memory();
    auto toks = rp::tokenize(model.vocab(), prompt);

    common_params_sampling sparams;
    sparams.seed = seed;
    rp::Generator gen(model, ctx, sparams);

    steer_active = true;
    if (ctx.decode(rp::span<const llama_token>(toks.data(), toks.size())) != 0) {
        throw std::runtime_error("decode failed for: " + prompt.substr(0, 60));
    }
    if (!steer_gen) steer_active = false;

    gen.accept_prompt(rp::span<const llama_token>(toks.data(), toks.size()));
    auto result = gen.run(n_predict);
    steer_active = false;

    return result.text;
}

int main(int argc, char ** argv) {
    auto args = rp::parse_common_args(argc, argv);
    args.defaults(2048, 200);

    // single mode
    std::string dir_path, prompt;
    int    single_layer = -1;
    // batch mode
    std::string dir_dir, prompts_path, out_dir, layers_spec;
    std::string stream = "l_out";

    float    alpha     = 0.0f;
    bool     steer_gen = false;
    int      n_prompts_max = 0;
    std::string label_filter;

    for (std::size_t i = 0; i < args.extra.size(); ++i) {
        const auto & a = args.extra[i];
        if      (a == "--direction"     && i+1 < args.extra.size()) dir_path      = args.extra[++i];
        else if (a == "--layer"         && i+1 < args.extra.size()) single_layer  = std::atoi(args.extra[++i].c_str());
        else if (a == "--direction-dir" && i+1 < args.extra.size()) dir_dir       = args.extra[++i];
        else if (a == "--layers"        && i+1 < args.extra.size()) layers_spec   = args.extra[++i];
        else if (a == "--stream"        && i+1 < args.extra.size()) stream        = args.extra[++i];
        else if (a == "--prompts"       && i+1 < args.extra.size()) prompts_path  = args.extra[++i];
        else if (a == "--out-dir"       && i+1 < args.extra.size()) out_dir       = args.extra[++i];
        else if (a == "--alpha"         && i+1 < args.extra.size()) alpha         = (float)std::atof(args.extra[++i].c_str());
        else if (a == "--prompt"        && i+1 < args.extra.size()) prompt        = args.extra[++i];
        else if (a == "--steer-gen")                                steer_gen     = true;
        else if (a == "--n-prompts"     && i+1 < args.extra.size()) n_prompts_max = std::atoi(args.extra[++i].c_str());
        else if (a == "--label"         && i+1 < args.extra.size()) label_filter  = args.extra[++i];
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 1; }
    }

    if (!rp::require_model(args)) return 1;

    bool batch_mode = !dir_dir.empty() && !prompts_path.empty();
    bool single_mode = !dir_path.empty() && !prompt.empty() && single_layer >= 0;

    if (!batch_mode && !single_mode) {
        std::fprintf(stderr,
            "usage:\n"
            "  single: %s -m MODEL --direction DIR.bin --layer L --alpha A --prompt TEXT\n"
            "  batch:  %s -m MODEL --direction-dir DIR/ --layers 0-31 --alpha A --prompts P.json --out-dir OUT/\n",
            argv[0], argv[0]);
        return 1;
    }

    return rp::run_experiment([&] {
        rp::Backend backend;
        rp::Model   model(args.model_path);
        const auto & geom = model.geom();

        rp::CaptureRegistry caps(geom);

        // steering state
        bool                       steer_active = false;
        float                      active_alpha = 0.0f;
        const std::vector<float> * active_dir   = nullptr;
        std::vector<float>         buf(geom.n_embd);
        rp::HookId                 steer_hook   = rp::kInvalidHookId;

        auto install_hook = [&](int layer) {
            if (steer_hook != rp::kInvalidHookId) caps.remove(steer_hook);
            steer_hook = caps.on_residual_mutate(layer, [&](rp::MutableTensorView & v) {
                if (!steer_active) return;
                if (!active_dir || active_alpha == 0.0f) return;

                v.copy_last_token(rp::span<float>(buf.data(), buf.size()));
                for (int j = 0; j < geom.n_embd; ++j) {
                    buf[j] += active_alpha * (*active_dir)[j];
                }
                v.store_last_token(rp::span<const float>(buf.data(), buf.size()));
            });
        };

        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = (uint32_t)args.n_ctx;
        rp::Context ctx(model, cparams, &caps);

        if (single_mode) {
            // ── single prompt mode ──────────────────────────────────────
            auto direction = load_direction(dir_path, geom.n_embd);
            install_hook(single_layer);
            active_alpha = alpha;
            active_dir   = &direction;

            auto text = run_one(model, ctx, steer_active, prompt,
                                args.n_predict, args.seed, steer_gen);
            std::printf("=== layer=%d alpha=%.2f ===\n%s\n", single_layer, alpha, text.c_str());

        } else {
            // ── batch mode (sequential, one prompt at a time) ───────────
            auto layers = parse_layers(layers_spec);

            // load all directions
            std::map<int, std::vector<float>> directions;
            for (int il : layers) {
                std::string path = dir_dir + "/" + stream + "_L" + std::to_string(il) + ".bin";
                try {
                    directions[il] = load_direction(path, geom.n_embd);
                } catch (const std::exception & e) {
                    std::fprintf(stderr, "  skip L%d: %s\n", il, e.what());
                }
            }

            // load prompts
            json raw = rp::load_json_file(prompts_path);

            struct Prompt { std::string text, label, id; };
            std::vector<Prompt> prompts;
            for (const auto & e : raw) {
                std::string lbl = e.value("label", "");
                if (!label_filter.empty() && lbl != label_filter) continue;
                Prompt p;
                p.text  = e["prompt"].get<std::string>();
                p.label = lbl;
                std::string s = p.text.substr(0, 40);
                for (auto & c : s) {
                    if (!std::isalnum(c)) c = '_';
                    else c = std::tolower(c);
                }
                while (!s.empty() && s.back() == '_') s.pop_back();
                p.id = s;
                prompts.push_back(std::move(p));
            }

            if (n_prompts_max > 0 && (int)prompts.size() > n_prompts_max) {
                prompts.resize(n_prompts_max);
            }

            int n_prompts = (int)prompts.size();
            int total = n_prompts * (1 + (int)directions.size());
            int run_i = 0;

            std::fprintf(stderr, "prompts: %d  layers: %zu  alpha: %.1f  total runs: %d\n",
                          n_prompts, directions.size(), alpha, total);

            auto t_start = std::chrono::steady_clock::now();

            auto do_run = [&](const Prompt & p, const char * config_name,
                              int layer, const std::vector<float> * dir, float a) {
                run_i++;
                std::string path = out_dir + "/" + p.id + "_" + config_name + ".txt";

                // skip if cached
                { std::ifstream check(path); if (check.good()) {
                    std::fprintf(stderr, "\r[%d/%d] %s %s (cached)          ",
                                 run_i, total, p.id.substr(0, 20).c_str(), config_name);
                    return;
                }}

                auto elapsed = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - t_start).count();
                double avg = (run_i > 1) ? elapsed / (run_i - 1) : 0;
                int eta_s = (int)(avg * (total - run_i));
                std::fprintf(stderr, "\r[%d/%d] %s %s  avg=%.1fs  ETA %d:%02d     ",
                             run_i, total, p.id.substr(0, 20).c_str(), config_name,
                             avg, eta_s / 60, eta_s % 60);

                if (layer >= 0) install_hook(layer);
                active_alpha = a;
                active_dir   = dir;

                auto text = run_one(model, ctx, steer_active, p.text,
                                    args.n_predict, args.seed, steer_gen);

                std::ofstream ofs(path);
                ofs << "=== " << config_name << " ===\n" << text << "\n";
            };

            // baselines
            install_hook(0);  // dummy
            for (const auto & p : prompts) {
                do_run(p, "baseline", -1, nullptr, 0.0f);
            }

            // steered
            for (auto & [il, dir] : directions) {
                char name[32];
                std::snprintf(name, sizeof(name), "L%d", il);
                for (const auto & p : prompts) {
                    do_run(p, name, il, &dir, alpha);
                }
            }

            auto elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t_start).count();
            std::fprintf(stderr, "\ndone. %d runs in %.0fs, outputs in %s\n",
                          total, elapsed, out_dir.c_str());
        }
    });
}
