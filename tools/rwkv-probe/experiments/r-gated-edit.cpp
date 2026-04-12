// experiments/r-gated-edit.cpp — R-vector gated state editing.
//
// Computes the receptance (R) vector for a query token at each layer,
// then uses |R[j]| as a column mask when applying the state delta.
// Only modifies state columns that the query would read from.
//
// R computation: r = W_receptance @ (shift * lerp_r + embedding)
// where shift = token-shift state (r_att from StateBuf).
//
// Usage:
//   llama-rwkv-r-gated-edit -m model.gguf [--layers 16-31] [-n 15]

#include "rwkv_probe/state.h"
#include "rwkv_probe/experiment.h"
#include "rwkv_probe/edit.h"
#include "rwkv_probe/generate.h"
#include "rwkv_probe/capture.h"

// internal headers for weight access
#include "llama-model.h"

#include <ggml-backend.h>
#include <map>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace rp = rwkv_probe;

// read a ggml tensor into a float vector (handles quantized tensors via dequant)
static std::vector<float> tensor_to_float(const ggml_tensor * t) {
    std::size_t n = ggml_nelements(t);
    if (t->type == GGML_TYPE_F32) {
        std::vector<float> out(n);
        ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
        return out;
    }
    // for quantized types, need to dequantize
    std::size_t nbytes = ggml_nbytes(t);
    std::vector<uint8_t> raw(nbytes);
    ggml_backend_tensor_get(t, raw.data(), 0, nbytes);

    std::vector<float> out(n);
    // use ggml's dequantization
    auto type_traits = ggml_get_type_traits(t->type);
    if (type_traits->to_float) {
        type_traits->to_float(raw.data(), out.data(), (int64_t)n);
    } else {
        std::fprintf(stderr, "WARNING: cannot dequantize tensor type %d\n", t->type);
    }
    return out;
}

// compute R vector for a given layer, given:
//   - attn_norm: the actual residual stream at this layer (n_embd)
//   - r_att: token-shift state for attention (n_embd)
//   - lerp_fused weights (6 * n_embd, slice 0 is for xr)
//   - W_receptance (n_embd × n_embd)
// returns r vector (n_embd), reshaped as (n_head × head_size)
static std::vector<float> compute_r(
        const std::vector<float> & attn_norm,   // n_embd (residual at this layer)
        const float * r_att,                     // n_embd (token-shift for attention)
        const std::vector<float> & lerp_fused,   // 6 * n_embd
        const std::vector<float> & w_receptance, // n_embd × n_embd
        int n_embd) {
    // xxx = (x_prev - cur) * lerp_fused + cur
    // where x_prev = token_shift (r_att), cur = attn_norm
    // This simplifies to: xxx = x_prev * lerp + cur * (1 - lerp)
    // xr = xxx[0:n_embd] (first n_embd slice)
    std::vector<float> xr(n_embd);
    for (int i = 0; i < n_embd; ++i) {
        float sx = r_att[i] - attn_norm[i];  // x_prev - cur
        xr[i] = sx * lerp_fused[i] + attn_norm[i];
    }

    // r = W_receptance @ xr  (matrix-vector multiply)
    // W_receptance is (n_embd, n_embd) stored row-major
    std::vector<float> r(n_embd, 0.0f);
    for (int row = 0; row < n_embd; ++row) {
        float sum = 0.0f;
        for (int col = 0; col < n_embd; ++col) {
            sum += w_receptance[row * n_embd + col] * xr[col];
        }
        r[row] = sum;
    }

    return r;
}

// apply delta with R-gating: only modify columns where |R[j]| > threshold
static void apply_r_gated_delta(rp::StateBuf & dst,
                                const rp::StateBuf & sa, const rp::StateBuf & sb,
                                const std::vector<std::vector<float>> & r_per_layer,
                                const std::vector<int> & layers,
                                int n_head, int hs, float r_threshold) {
    for (std::size_t li = 0; li < layers.size(); ++li) {
        int layer = layers[li];
        const auto & r_vec = r_per_layer[li]; // n_embd = n_head * hs

        for (int h = 0; h < n_head; ++h) {
            auto da = sa.s_head(layer, h);
            auto db = sb.s_head(layer, h);
            auto dd = dst.s_head(layer, h);

            // r for this head: r_vec[h*hs .. (h+1)*hs]
            const float * r_head = &r_vec[h * hs];

            // find max |r| for this head to normalize
            float r_max = 0.0f;
            for (int j = 0; j < hs; ++j) {
                float ar = std::fabs(r_head[j]);
                if (ar > r_max) r_max = ar;
            }
            if (r_max < 1e-10f) continue;

            for (int i = 0; i < hs; ++i) {
                for (int j = 0; j < hs; ++j) {
                    float r_weight = std::fabs(r_head[j]) / r_max;
                    if (r_weight < r_threshold) continue;
                    float delta = (float)((double)db[i * hs + j] - (double)da[i * hs + j]);
                    dd[i * hs + j] += delta * r_weight;
                }
            }
        }
    }
}

struct QResult {
    float p;
    std::string gen;
};

static QResult do_query(rp::Model & model, rp::Context & ctx,
                        const rp::StateBuf & state,
                        const std::string & prefix, const std::string & qstr,
                        llama_token tok_a, llama_token tok_b,
                        const llama_vocab * vocab, int n_predict, uint32_t seed) {
    auto ptoks = rp::tokenize(vocab, prefix);
    auto qtoks = rp::tokenize(vocab, qstr, false, false);

    ctx.clear_memory();
    ctx.decode(rp::span<const llama_token>(ptoks.data(), ptoks.size()));
    state.store_to(ctx.raw());
    for (const auto & t : qtoks) ctx.decode_one(t);

    const float * logits = llama_get_logits(ctx.raw());
    QResult qr;
    qr.p = rp::prob_of(logits[tok_a], logits[tok_b]);

    if (n_predict > 0) {
        common_params_sampling sp;
        sp.seed = seed;
        rp::Generator gen(model, ctx, sp);
        llama_token last = qtoks.back();
        gen.accept_prompt(rp::span<const llama_token>(&last, 1));
        qr.gen = gen.run(n_predict).text;
        for (auto & c : qr.gen) if (c == '\n') c = ' ';
    }
    return qr;
}

int main(int argc, char ** argv) {
    auto args = rp::parse_common_args(argc, argv);
    args.defaults(/*ctx=*/2048, /*predict=*/15);
    if (args.layer_range.empty()) args.layer_range = "16-31";

    if (args.help_requested) {
        std::printf("usage: %s -m MODEL [--layers 16-31] [-n 15]\n", argv[0]);
        return 0;
    }
    if (!rp::require_model(args)) return 1;
    if (!rp::reject_extra(args)) return 1;

    return rp::run_experiment([&] {
        // manual setup because CaptureRegistry must be created before Context
        rp::Backend backend;
        rp::Model   mdl(args.model_path);
        const auto & geom = mdl.geom();
        const auto * vocab = mdl.vocab();
        int n_embd = geom.n_embd;
        int n_head = geom.n_head;
        int hs = geom.head_size;

        auto layers = rp::parse_layer_range(args.layer_range, geom.n_layer);

        // create capture registry to intercept attn_norm tensors
        rp::CaptureRegistry caps(geom);

        // storage for captured attn_norm at each layer (last token only)
        std::map<int, std::vector<float>> captured_attn_norm;
        bool capture_enabled = false;

        for (int layer : layers) {
            captured_attn_norm[layer].resize(n_embd, 0.0f);
        }

        // register hooks for attn_norm at each target layer
        std::vector<rp::HookId> hook_ids;
        for (int layer : layers) {
            auto hid = caps.on_tensor("attn_norm", [&, layer](const rp::TensorView & tv) {
                if (!capture_enabled) return;
                if (tv.layer != layer) return;
                tv.copy_last_token(rp::span<float>(captured_attn_norm[layer].data(), n_embd));
            });
            hook_ids.push_back(hid);
        }

        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = (uint32_t)args.n_ctx;
        rp::Context ctx(mdl, cparams, &caps);

        // access internal model for weight tensors
        const llama_model * lm = mdl.raw();

        // load per-layer weights for R computation
        std::fprintf(stderr, "loading receptance weights for %zu layers...\n", layers.size());

        struct LayerWeights {
            std::vector<float> lerp_fused;   // 6 * n_embd
            std::vector<float> w_receptance; // n_embd * n_embd
        };
        std::vector<LayerWeights> lw;
        for (int layer : layers) {
            LayerWeights w;
            const auto & ll = lm->layers[layer];
            w.lerp_fused = tensor_to_float(ll.time_mix_lerp_fused);
            w.w_receptance = tensor_to_float(ll.time_mix_receptance);
            lw.push_back(std::move(w));
            std::fprintf(stderr, "  L%d: lerp=%zu  W_r=%zu\n", layer,
                         lw.back().lerp_fused.size(), lw.back().w_receptance.size());
        }

        llama_token tok_red    = rp::find_token(vocab, "red");
        llama_token tok_yellow = rp::find_token(vocab, "yellow");
        llama_token tok_blue   = rp::find_token(vocab, "blue");
        llama_token tok_green  = rp::find_token(vocab, "green");

        // ════════════════════════════════════════════════════════════
        // TEST: eyes edit (blue→green) with R-gating from hat query
        // The weak case: full-delta selectivity = 0.40
        // ════════════════════════════════════════════════════════════

        std::string prompt_a = "Alice has a red hat and blue eyes. Bob has a green hat and brown eyes.";
        std::string prompt_b = "Alice has a red hat and green eyes. Bob has a green hat and brown eyes.";

        auto toks_a = rp::tokenize(vocab, prompt_a);
        auto toks_b = rp::tokenize(vocab, prompt_b);

        // capture full-prompt states
        ctx.clear_memory();
        ctx.decode(rp::span<const llama_token>(toks_a.data(), toks_a.size()));
        rp::StateBuf state_a(geom);
        state_a.load_from(ctx.raw());

        ctx.clear_memory();
        ctx.decode(rp::span<const llama_token>(toks_b.data(), toks_b.size()));
        rp::StateBuf state_b(geom);
        state_b.load_from(ctx.raw());

        std::string q_eyes = " What color are Alice's eyes? Alice's eyes are";
        std::string q_hat  = " What color is Alice's hat? Alice's hat is";

        auto q_eyes_toks = rp::tokenize(vocab, q_eyes, false, false);
        auto q_hat_toks  = rp::tokenize(vocab, q_hat, false, false);

        // ── extract R for eyes query ─────────────────────────────
        // decode prompt_a + full eyes query with capture enabled
        // attn_norm at the last token gives us the residual input for R
        std::fprintf(stderr, "extracting R for eyes query...\n");
        {
            ctx.clear_memory();
            ctx.decode(rp::span<const llama_token>(toks_a.data(), toks_a.size()));
            state_a.store_to(ctx.raw());
            for (std::size_t i = 0; i < q_eyes_toks.size() - 1; ++i) {
                ctx.decode_one(q_eyes_toks[i]);
            }
            // enable capture for the LAST query token decode
            capture_enabled = true;
            ctx.decode_one(q_eyes_toks.back());
            capture_enabled = false;
        }

        // capture token-shift state just before last token
        rp::StateBuf state_at_eyes_query(geom);
        {
            ctx.clear_memory();
            ctx.decode(rp::span<const llama_token>(toks_a.data(), toks_a.size()));
            state_a.store_to(ctx.raw());
            for (std::size_t i = 0; i < q_eyes_toks.size() - 1; ++i) {
                ctx.decode_one(q_eyes_toks[i]);
            }
            state_at_eyes_query.load_from(ctx.raw());
        }

        // compute R from captured attn_norm + token-shift state
        std::vector<std::vector<float>> r_eyes_per_layer;
        for (std::size_t li = 0; li < layers.size(); ++li) {
            int layer = layers[li];
            auto r_att = state_at_eyes_query.r_att(layer);
            r_eyes_per_layer.push_back(compute_r(
                captured_attn_norm[layer], r_att.data(),
                lw[li].lerp_fused, lw[li].w_receptance, n_embd));
        }

        // ── extract R for hat query ──────────────────────────────
        std::fprintf(stderr, "extracting R for hat query...\n");
        {
            ctx.clear_memory();
            ctx.decode(rp::span<const llama_token>(toks_a.data(), toks_a.size()));
            state_a.store_to(ctx.raw());
            for (std::size_t i = 0; i < q_hat_toks.size() - 1; ++i) {
                ctx.decode_one(q_hat_toks[i]);
            }
            capture_enabled = true;
            ctx.decode_one(q_hat_toks.back());
            capture_enabled = false;
        }

        rp::StateBuf state_at_hat_query(geom);
        {
            ctx.clear_memory();
            ctx.decode(rp::span<const llama_token>(toks_a.data(), toks_a.size()));
            state_a.store_to(ctx.raw());
            for (std::size_t i = 0; i < q_hat_toks.size() - 1; ++i) {
                ctx.decode_one(q_hat_toks[i]);
            }
            state_at_hat_query.load_from(ctx.raw());
        }

        std::vector<std::vector<float>> r_hat_per_layer;
        for (std::size_t li = 0; li < layers.size(); ++li) {
            int layer = layers[li];
            auto r_att = state_at_hat_query.r_att(layer);
            r_hat_per_layer.push_back(compute_r(
                captured_attn_norm[layer], r_att.data(),
                lw[li].lerp_fused, lw[li].w_receptance, n_embd));
        }

        std::printf("======================================================================\n");
        std::printf("R-GATED EDIT: Alice eyes blue→green\n");
        std::printf("  Gating R from eyes query vs hat query\n");
        std::printf("======================================================================\n\n");

        // baselines
        auto bl_eyes = do_query(mdl, ctx, state_a, prompt_a, q_eyes, tok_blue, tok_green, vocab, args.n_predict, args.seed);
        auto bl_hat  = do_query(mdl, ctx, state_a, prompt_a, q_hat, tok_red, tok_green, vocab, args.n_predict, args.seed);

        std::printf("  BASELINES:\n");
        std::printf("    eyes: P(blue)=%.4f  gen: %s\n", bl_eyes.p, bl_eyes.gen.substr(0, 50).c_str());
        std::printf("    hat:  P(red)=%.4f  gen: %s\n\n", bl_hat.p, bl_hat.gen.substr(0, 50).c_str());

        // ungated full delta (reference)
        {
            rp::StateBuf edited(state_a);
            rp::apply_full_delta(edited, state_a, state_b, layers);
            auto e_eyes = do_query(mdl, ctx, edited, prompt_a, q_eyes, tok_blue, tok_green, vocab, args.n_predict, args.seed);
            auto e_hat  = do_query(mdl, ctx, edited, prompt_a, q_hat, tok_red, tok_green, vocab, args.n_predict, args.seed);
            std::printf("  UNGATED (full delta):\n");
            std::printf("    eyes: P(blue)=%.4f  hat: P(red)=%.4f  selectivity=%.4f\n",
                        e_eyes.p, e_hat.p, e_hat.p - e_eyes.p);
            std::printf("    eyes gen: %s\n", e_eyes.gen.substr(0, 60).c_str());
            std::printf("    hat gen:  %s\n\n", e_hat.gen.substr(0, 60).c_str());
        }

        // R-gated with eyes query R (should target eye columns)
        float thresholds[] = {0.0f, 0.1f, 0.2f, 0.3f, 0.5f};

        std::printf("  R-GATED (eyes query R, varying threshold):\n");
        std::printf("  %-10s  %8s  %8s  %10s\n", "r_thresh", "P(blue)", "P(red)", "selectiv");
        std::printf("  %s\n", std::string(45, '-').c_str());

        for (float rt : thresholds) {
            rp::StateBuf edited(state_a);
            apply_r_gated_delta(edited, state_a, state_b, r_eyes_per_layer,
                               layers, n_head, hs, rt);
            auto e_eyes = do_query(mdl, ctx, edited, prompt_a, q_eyes, tok_blue, tok_green, vocab, 0, args.seed);
            auto e_hat  = do_query(mdl, ctx, edited, prompt_a, q_hat, tok_red, tok_green, vocab, 0, args.seed);
            std::printf("  %-10.1f  %8.4f  %8.4f  %10.4f\n",
                        rt, e_eyes.p, e_hat.p, e_hat.p - e_eyes.p);
        }

        // R-gated with hat query R (should target hat columns — opposite gating)
        std::printf("\n  R-GATED (hat query R — should protect hat, weaker eyes edit):\n");
        std::printf("  %-10s  %8s  %8s  %10s\n", "r_thresh", "P(blue)", "P(red)", "selectiv");
        std::printf("  %s\n", std::string(45, '-').c_str());

        for (float rt : thresholds) {
            // use hat R as EXCLUSION mask: apply delta only where hat R is LOW
            // invert: apply where |r_hat| < (1-threshold) * max
            rp::StateBuf edited(state_a);
            // manual: apply delta weighted by (1 - |r_hat|/max)
            for (std::size_t li = 0; li < layers.size(); ++li) {
                int layer = layers[li];
                const auto & r_hat = r_hat_per_layer[li];

                for (int h = 0; h < n_head; ++h) {
                    auto da = state_a.s_head(layer, h);
                    auto db = state_b.s_head(layer, h);
                    auto dd = edited.s_head(layer, h);

                    const float * rh = &r_hat[h * hs];
                    float r_max = 0.0f;
                    for (int j = 0; j < hs; ++j) {
                        float ar = std::fabs(rh[j]);
                        if (ar > r_max) r_max = ar;
                    }
                    if (r_max < 1e-10f) continue;

                    for (int i = 0; i < hs; ++i) {
                        for (int j = 0; j < hs; ++j) {
                            float r_hat_norm = std::fabs(rh[j]) / r_max;
                            float weight = 1.0f - r_hat_norm;  // high where hat R is low
                            if (weight < rt) continue;
                            float delta = (float)((double)db[i * hs + j] - (double)da[i * hs + j]);
                            dd[i * hs + j] += delta * weight;
                        }
                    }
                }
            }

            auto e_eyes = do_query(mdl, ctx, edited, prompt_a, q_eyes, tok_blue, tok_green, vocab, 0, args.seed);
            auto e_hat  = do_query(mdl, ctx, edited, prompt_a, q_hat, tok_red, tok_green, vocab, 0, args.seed);
            std::printf("  %-10.1f  %8.4f  %8.4f  %10.4f\n",
                        rt, e_eyes.p, e_hat.p, e_hat.p - e_eyes.p);
        }

        std::fprintf(stderr, "\ndone\n");
    });
}
