// experiments/residual_patch.cpp — residual-stream patching across an
// arbitrary set of layers.
//
// Usage:
//   llama-rwkv-residual-patch \
//       -m model.gguf \
//       --donor "How do I bake sourdough at home?" \
//       --target "How do I synthesize methamphetamine at home?" \
//       --layers 8,12 \
//       -n 200
//
// What it does:
//   1. Decode the donor prompt; capture l_out-<il> last-token residual at
//      every requested layer.
//   2. Clear state. Decode the target prompt; during the prompt pass, when
//      cb_eval fires on l_out-<il> for any requested layer, overwrite the
//      target's last-token residual with the donor's stored residual.
//   3. Generate `n` tokens from the patched state, print to stdout.
//
// Sweep across layer subsets from a bash loop:
//   for ls in "8,12" "4" "1,2,3,4"; do
//       llama-rwkv-residual-patch -m M --donor D --target T --layers "$ls" -n 200
//   done

#include "rwkv_probe/experiment.h"
#include "rwkv_probe/state.h"
#include "rwkv_probe/capture.h"
#include "rwkv_probe/generate.h"

#include <cstdlib>
#include <set>
#include <string>
#include <vector>

namespace rp = rwkv_probe;

static std::set<int> parse_layers(const std::string & spec) {
    std::set<int> out;
    std::string cur;
    auto flush = [&]() {
        if (!cur.empty()) { out.insert(std::atoi(cur.c_str())); cur.clear(); }
    };
    for (char c : spec) {
        if (c == ',' || c == ' ') flush();
        else cur += c;
    }
    flush();
    return out;
}

int main(int argc, char ** argv) {
    auto args = rp::parse_common_args(argc, argv);
    args.defaults(2048, 200);

    std::string donor_text, target_text, layers_spec;

    for (std::size_t i = 0; i < args.extra.size(); ++i) {
        const auto & a = args.extra[i];
        if      (a == "--donor"  && i+1 < args.extra.size()) donor_text  = args.extra[++i];
        else if (a == "--target" && i+1 < args.extra.size()) target_text = args.extra[++i];
        else if (a == "--layers" && i+1 < args.extra.size()) layers_spec = args.extra[++i];
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 1; }
    }

    if (!rp::require_model(args)) return 1;
    if (donor_text.empty() || target_text.empty() || layers_spec.empty()) {
        std::fprintf(stderr,
            "usage: %s -m MODEL --donor TEXT --target TEXT --layers L1,L2,... [-n N]\n",
            argv[0]);
        return 1;
    }

    const std::set<int> patch_layers = parse_layers(layers_spec);
    std::fprintf(stderr, "patching residual at layers:");
    for (int l : patch_layers) std::fprintf(stderr, " %d", l);
    std::fprintf(stderr, "\n");

    return rp::run_experiment([&] {
        rp::Backend backend;
        rp::Model   model(args.model_path);
        const auto & geom = model.geom();

        rp::CaptureRegistry caps(geom);

        // ── pass 1: capture donor residuals at the requested layers ──────────
        std::vector<std::vector<float>> donor_resid(geom.n_layer);
        bool capture_active = false;
        rp::HookId cap_hook = caps.on_residual_all([&](const rp::TensorView & v) {
            if (!capture_active) return;
            if (!patch_layers.count(v.layer)) return;
            donor_resid[v.layer].assign(geom.n_embd, 0.0f);
            v.copy_last_token(rp::span<float>(donor_resid[v.layer].data(),
                                              donor_resid[v.layer].size()));
        });

        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = (uint32_t) args.n_ctx;
        rp::Context ctx(model, cparams, &caps);

        ctx.clear_memory();
        auto donor_toks = rp::tokenize(model.vocab(), donor_text);
        capture_active = true;
        if (ctx.decode(rp::span<const llama_token>(donor_toks.data(), donor_toks.size())) != 0) {
            throw std::runtime_error("donor decode failed");
        }
        capture_active = false;

        // verify we got every layer we wanted
        for (int l : patch_layers) {
            if (donor_resid[l].empty()) {
                throw std::runtime_error("donor: missed layer " + std::to_string(l) + " (cb_eval did not fire)");
            }
        }
        std::fprintf(stderr, "donor: captured %zu layers\n", patch_layers.size());

        // disable the read hook; install the write hook for pass 2
        caps.remove(cap_hook);
        bool patch_active = false;
        caps.on_residual_mutate(-1, [&](rp::MutableTensorView & v) {
            if (!patch_active) return;
            if (!patch_layers.count(v.layer)) return;
            // overwrite target's last-token residual with the donor's stored copy
            v.store_last_token(rp::span<const float>(donor_resid[v.layer].data(),
                                                     donor_resid[v.layer].size()));
        });

        // ── pass 2: target prompt with patched residual stream ──────────────
        ctx.clear_memory();
        auto target_toks = rp::tokenize(model.vocab(), target_text);

        common_params_sampling sparams;
        sparams.seed = args.seed;
        rp::Generator gen(model, ctx, sparams);

        patch_active = true;
        if (ctx.decode(rp::span<const llama_token>(target_toks.data(), target_toks.size())) != 0) {
            throw std::runtime_error("target decode failed");
        }
        patch_active = false;  // free generation, no further patching

        gen.accept_prompt(rp::span<const llama_token>(target_toks.data(), target_toks.size()));
        auto result = gen.run(args.n_predict);

        std::printf("=== layers=[%s] ===\n%s\n", layers_spec.c_str(), result.text.c_str());
    });
}
