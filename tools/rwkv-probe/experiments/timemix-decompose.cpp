// experiments/timemix-decompose.cpp — capture and decompose residual stream
// into time_mix (WKV/recurrent-state-driven) and channel_mix (stateless FFN)
// contributions per layer.
//
// Captures l_out-<il> and ffn_inp-<il> at every layer during the prompt pass.
// Derives per prompt:
//   time_mix[L]    = ffn_inp[L] - l_out[L-1]   (L>0; L0 skipped, no embedding)
//   channel_mix[L] = l_out[L]   - ffn_inp[L]
//
// Modes:
//   --output FILE.root   Write all captured vectors to a single ROOT file.
//                         One TTree ("decompose") with per-prompt rows and
//                         per-layer branches. This is the primary output for
//                         downstream ML analysis.
//
//   (no --output)         If prompts have pair_id/label fields, compute
//                         within-pair distance metrics and print TSV + summary.
//
// Usage:
//   # capture 1000 prompts to ROOT for offline ML
//   llama-rwkv-timemix-decompose -m model.gguf --prompts prompts.json \
//       --output /tmp/decompose.root
//
//   # quick distance analysis on a small paired set
//   llama-rwkv-timemix-decompose -m model.gguf --prompts paired_prompts.json
//
// ROOT schema (one row per prompt):
//   prompt_idx/I          sequential index
//   label/I               0 = safe, 1 = dangerous, -1 = unknown
//   pair_idx/I            pair index (or -1)
//   l_out_L<il>[n_embd]/F
//   tmix_L<il>[n_embd]/F  (il >= 1 only)
//   cmix_L<il>[n_embd]/F

#include "rwkv_probe/capture.h"
#include "rwkv_probe/generate.h"
#include "rwkv_probe/experiment.h"

#include "TFile.h"
#include "TTree.h"
#include "TNamed.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

namespace rp = rwkv_probe;
using json = nlohmann::json;

// ── distance metrics ────────────────────────────────────────────────────────

static double l2_distance(const std::vector<float> & a, const std::vector<float> & b) {
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        double d = (double) a[i] - (double) b[i];
        sum += d * d;
    }
    return std::sqrt(sum);
}

static double cosine_similarity(const std::vector<float> & a, const std::vector<float> & b) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        dot += (double) a[i] * (double) b[i];
        na  += (double) a[i] * (double) a[i];
        nb  += (double) b[i] * (double) b[i];
    }
    double denom = std::sqrt(na) * std::sqrt(nb);
    return denom > 0.0 ? dot / denom : 0.0;
}

// ── ROOT writer ─────────────────────────────────────────────────────────────

class DecomposeWriter {
public:
    DecomposeWriter(const std::string & path, int n_layer, int n_embd)
        : m_n_layer(n_layer), m_n_embd(n_embd)
    {
        m_file = TFile::Open(path.c_str(), "RECREATE");
        if (!m_file || m_file->IsZombie()) {
            throw std::runtime_error("cannot create " + path);
        }

        m_tree = new TTree("decompose", "Residual stream decomposition");
        m_tree->SetAutoSave(0);
        m_tree->SetAutoFlush(-32LL * 1024 * 1024);

        // scalar branches
        m_tree->Branch("prompt_idx", &m_prompt_idx, "prompt_idx/I");
        m_tree->Branch("label",      &m_label,      "label/I");
        m_tree->Branch("pair_idx",   &m_pair_idx,   "pair_idx/I");

        // per-layer vector branches
        m_l_out.assign(n_layer, std::vector<float>(n_embd, 0.0f));
        m_tmix.assign(n_layer, std::vector<float>(n_embd, 0.0f));
        m_cmix.assign(n_layer, std::vector<float>(n_embd, 0.0f));

        char nm[64], spec[64];
        for (int il = 0; il < n_layer; ++il) {
            std::snprintf(nm,   sizeof nm,   "l_out_L%d", il);
            std::snprintf(spec, sizeof spec, "l_out_L%d[%d]/F", il, n_embd);
            m_tree->Branch(nm, m_l_out[il].data(), spec);

            std::snprintf(nm,   sizeof nm,   "cmix_L%d", il);
            std::snprintf(spec, sizeof spec, "cmix_L%d[%d]/F", il, n_embd);
            m_tree->Branch(nm, m_cmix[il].data(), spec);

            if (il == 0) continue;  // no tmix at L0
            std::snprintf(nm,   sizeof nm,   "tmix_L%d", il);
            std::snprintf(spec, sizeof spec, "tmix_L%d[%d]/F", il, n_embd);
            m_tree->Branch(nm, m_tmix[il].data(), spec);
        }

        // metadata
        TNamed("n_layer", std::to_string(n_layer).c_str()).Write();
        TNamed("n_embd",  std::to_string(n_embd).c_str()).Write();
    }

    void fill(int prompt_idx, int label, int pair_idx,
              const std::vector<std::vector<float>> & l_out,
              const std::vector<std::vector<float>> & tmix,
              const std::vector<std::vector<float>> & cmix) {
        m_prompt_idx = prompt_idx;
        m_label      = label;
        m_pair_idx   = pair_idx;

        for (int il = 0; il < m_n_layer; ++il) {
            auto copy = [&](std::vector<float> & dst, const std::vector<float> & src) {
                if ((int) src.size() == m_n_embd) {
                    std::copy(src.begin(), src.end(), dst.begin());
                } else {
                    std::fill(dst.begin(), dst.end(), 0.0f);
                }
            };
            copy(m_l_out[il], l_out[il]);
            copy(m_cmix[il],  cmix[il]);
            if (il > 0) copy(m_tmix[il], tmix[il]);
        }
        m_tree->Fill();
    }

    void close() {
        if (!m_file) return;
        m_tree->Write("", 2 /*kOverwrite*/);
        m_file->Close();
        delete m_file;
        m_file = nullptr;
        m_tree = nullptr;
    }

    ~DecomposeWriter() { if (m_file) close(); }

    DecomposeWriter(const DecomposeWriter &) = delete;
    DecomposeWriter & operator=(const DecomposeWriter &) = delete;

private:
    int     m_n_layer;
    int     m_n_embd;
    TFile * m_file = nullptr;
    TTree * m_tree = nullptr;

    Int_t   m_prompt_idx = 0;
    Int_t   m_label      = -1;
    Int_t   m_pair_idx   = -1;

    std::vector<std::vector<float>> m_l_out;
    std::vector<std::vector<float>> m_tmix;
    std::vector<std::vector<float>> m_cmix;
};

// ── prompt entry ────────────────────────────────────────────────────────────

struct PromptEntry {
    std::string id;
    std::string pair_id;
    std::string label;
    std::string text;
};

// ── capture + derive ────────────────────────────────────────────────────────

struct PromptVectors {
    int n_layer = 0;
    int n_embd  = 0;
    std::vector<std::vector<float>> l_out;
    std::vector<std::vector<float>> ffn_inp;
    std::vector<std::vector<float>> time_mix;
    std::vector<std::vector<float>> channel_mix;

    void derive() {
        time_mix.resize(n_layer);
        channel_mix.resize(n_layer);
        for (int il = 0; il < n_layer; ++il) {
            if (ffn_inp[il].empty() || l_out[il].empty()) continue;
            channel_mix[il].resize(n_embd);
            for (int i = 0; i < n_embd; ++i)
                channel_mix[il][i] = l_out[il][i] - ffn_inp[il][i];
            if (il == 0 || l_out[il - 1].empty()) continue;
            time_mix[il].resize(n_embd);
            for (int i = 0; i < n_embd; ++i)
                time_mix[il][i] = ffn_inp[il][i] - l_out[il - 1][i];
        }
    }
};

// ── main ────────────────────────────────────────────────────────────────────

int main(int argc, char ** argv) {
    auto args = rp::parse_common_args(argc, argv);
    args.defaults(/*ctx=*/2048, /*predict=*/15);

    // parse experiment-specific args from extras
    std::string prompts_path;
    std::string output_path;
    {
        std::vector<std::string> remaining;
        for (std::size_t i = 0; i < args.extra.size(); ++i) {
            const auto & a = args.extra[i];
            if (a == "--prompts" && i+1 < args.extra.size()) {
                prompts_path = args.extra[++i];
            } else if (a == "--output" && i+1 < args.extra.size()) {
                output_path = args.extra[++i];
            } else {
                remaining.push_back(a);
            }
        }
        args.extra = std::move(remaining);
    }

    if (args.help_requested) {
        std::printf(
            "usage: %s -m MODEL --prompts FILE [options]\n\n"
            "  -m MODEL         path to GGUF model (required)\n"
            "  --prompts FILE   JSON array of {id, prompt, [pair_id, label]} (required)\n"
            "  --output FILE    write all vectors to a ROOT file\n"
            "  --n-ctx N        context size (default: 2048)\n"
            "  -h, --help       show this help\n", argv[0]);
        return 0;
    }
    if (!rp::require_model(args)) return 1;
    if (!rp::reject_extra(args)) return 1;
    if (prompts_path.empty()) {
        std::fprintf(stderr, "error: --prompts is required\n");
        return 1;
    }

    // ── load prompts ────────────────────────────────────────────────────────
    json raw = rp::load_json_file(prompts_path);

    std::vector<PromptEntry> prompts;
    for (const auto & e : raw) {
        prompts.push_back({
            e.value("id", ""),
            e.value("pair_id", ""),
            e.value("label", ""),
            e.at("prompt").get<std::string>(),
        });
    }
    std::fprintf(stderr, "loaded %zu prompts\n", prompts.size());

    // build pair_id -> index map for ROOT output
    std::map<std::string, int> pair_id_to_idx;
    {
        int next = 0;
        for (const auto & p : prompts) {
            if (!p.pair_id.empty() && !pair_id_to_idx.count(p.pair_id))
                pair_id_to_idx[p.pair_id] = next++;
        }
    }

    const bool write_root = !output_path.empty();
    const bool compute_distances = !write_root;  // distance mode when no ROOT output

    return rp::run_experiment([&] {
        // manual setup because CaptureRegistry must be created before Context
        rp::Backend backend;
        rp::Model   model(args.model_path);
        const auto & geom = model.geom();

        std::fprintf(stderr, "n_layer=%d  n_embd=%d\n", geom.n_layer, geom.n_embd);

        // ── capture hooks ───────────────────────────────────────────────────
        rp::CaptureRegistry caps(geom);

        std::vector<std::vector<float>> cur_l_out(geom.n_layer);
        std::vector<std::vector<float>> cur_ffn_inp(geom.n_layer);
        bool capturing = false;

        caps.on_tensor("l_out-*", [&](const rp::TensorView & v) {
            if (!capturing || v.layer < 0 || v.layer >= geom.n_layer) return;
            cur_l_out[v.layer].assign(geom.n_embd, 0.0f);
            v.copy_last_token(rp::span<float>(cur_l_out[v.layer].data(),
                                              cur_l_out[v.layer].size()));
        });

        caps.on_tensor("ffn_inp-*", [&](const rp::TensorView & v) {
            if (!capturing || v.layer < 0 || v.layer >= geom.n_layer) return;
            cur_ffn_inp[v.layer].assign(geom.n_embd, 0.0f);
            v.copy_last_token(rp::span<float>(cur_ffn_inp[v.layer].data(),
                                              cur_ffn_inp[v.layer].size()));
        });

        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = (uint32_t) args.n_ctx;
        rp::Context ctx(model, cparams, &caps);

        // ── ROOT writer (optional) ──────────────────────────────────────────
        std::unique_ptr<DecomposeWriter> writer;
        if (write_root) {
            writer = std::make_unique<DecomposeWriter>(output_path, geom.n_layer, geom.n_embd);
            std::fprintf(stderr, "ROOT output: %s\n", output_path.c_str());
        }

        // ── storage for distance mode ───────────────────────────────────────
        std::map<std::string, PromptVectors> all_vecs;

        // ── per-prompt loop ─────────────────────────────────────────────────
        using clock = std::chrono::steady_clock;
        const auto t_start = clock::now();
        int n_done = 0;

        for (std::size_t pi = 0; pi < prompts.size(); ++pi) {
            const auto & p = prompts[pi];
            const auto t_prompt_start = clock::now();

            ctx.clear_memory();
            for (auto & v : cur_l_out)   v.clear();
            for (auto & v : cur_ffn_inp) v.clear();

            auto tokens = rp::tokenize(model.vocab(), p.text);

            capturing = true;
            if (ctx.decode(rp::span<const llama_token>(tokens.data(), tokens.size())) != 0) {
                std::fprintf(stderr, "  [%zu/%zu] FAILED: %s\n",
                             pi + 1, prompts.size(), p.id.c_str());
                capturing = false;
                continue;
            }
            capturing = false;

            // derive contributions
            PromptVectors pv;
            pv.n_layer = geom.n_layer;
            pv.n_embd  = geom.n_embd;
            pv.l_out   = cur_l_out;
            pv.ffn_inp = cur_ffn_inp;
            pv.derive();

            // write ROOT row
            if (writer) {
                int label = (p.label == "safe") ? 0 : (p.label == "dangerous") ? 1 : -1;
                int pidx  = pair_id_to_idx.count(p.pair_id) ? pair_id_to_idx[p.pair_id] : -1;
                writer->fill((int) pi, label, pidx, pv.l_out, pv.time_mix, pv.channel_mix);
            }

            // progress display
            n_done++;
            const auto t_now = clock::now();
            double elapsed_s = std::chrono::duration<double>(t_now - t_start).count();
            double prompt_s  = std::chrono::duration<double>(t_now - t_prompt_start).count();
            double avg_s     = elapsed_s / n_done;
            int remaining    = (int) prompts.size() - n_done;
            double eta_s     = avg_s * remaining;
            int eta_m        = (int)(eta_s / 60.0);
            int eta_sec      = (int)(eta_s) % 60;
            std::fprintf(stderr, "\r[%zu/%zu] %-12s %3zut  %.1fs  avg=%.1fs  ETA %d:%02d  ",
                         pi + 1, prompts.size(), p.id.c_str(),
                         tokens.size(), prompt_s, avg_s, eta_m, eta_sec);

            // keep vectors for distance mode
            if (compute_distances) {
                all_vecs.emplace(p.id, std::move(pv));
            }
        }

        // timing summary
        {
            double total_s = std::chrono::duration<double>(clock::now() - t_start).count();
            int total_m = (int)(total_s / 60.0);
            int total_sec = (int)(total_s) % 60;
            std::fprintf(stderr, "\n%d prompts in %d:%02d (avg %.1fs/prompt)\n",
                         n_done, total_m, total_sec, n_done > 0 ? total_s / n_done : 0.0);
        }

        // ── close ROOT ──────────────────────────────────────────────────────
        if (writer) {
            writer->close();
            std::fprintf(stderr, "ROOT file written: %s  (%zu prompts)\n",
                         output_path.c_str(), prompts.size());
        }

        // ── distance analysis (only when no ROOT output) ────────────────────
        if (compute_distances) {
            struct Pair { std::string pair_id, safe_id, dang_id; };
            std::map<std::string, std::string> psafe, pdang;
            for (const auto & p : prompts) {
                if (p.label == "safe")      psafe[p.pair_id] = p.id;
                if (p.label == "dangerous") pdang[p.pair_id] = p.id;
            }
            std::vector<Pair> pairs;
            for (const auto & kv : psafe) {
                auto it = pdang.find(kv.first);
                if (it != pdang.end() &&
                    all_vecs.count(kv.second) && all_vecs.count(it->second)) {
                    pairs.push_back({kv.first, kv.second, it->second});
                }
            }
            std::sort(pairs.begin(), pairs.end(),
                      [](const Pair & a, const Pair & b) { return a.pair_id < b.pair_id; });

            std::fprintf(stderr, "\n%zu valid pairs\n\n", pairs.size());

            std::printf("%-5s  %-12s  %-11s  %10s  %10s\n",
                        "layer", "pair", "stream", "l2_dist", "cosine_sim");

            struct Acc { double l2_sum = 0; double cos_sum = 0; int n = 0; };
            std::vector<Acc> acc_lo(geom.n_layer), acc_tm(geom.n_layer), acc_cm(geom.n_layer);

            for (int il = 0; il < geom.n_layer; ++il) {
                for (const auto & pair : pairs) {
                    const auto & cs = all_vecs.at(pair.safe_id);
                    const auto & cd = all_vecs.at(pair.dang_id);

                    auto emit = [&](const char * stream, const std::vector<float> & a,
                                    const std::vector<float> & b, Acc & acc) {
                        if (a.empty() || b.empty()) return;
                        double d = l2_distance(a, b);
                        double c = cosine_similarity(a, b);
                        std::printf("L%-4d  %-12s  %-11s  %10.4f  %10.6f\n",
                                    il, pair.pair_id.c_str(), stream, d, c);
                        acc.l2_sum += d; acc.cos_sum += c; acc.n++;
                    };

                    emit("l_out",       cs.l_out[il],       cd.l_out[il],       acc_lo[il]);
                    emit("time_mix",    cs.time_mix[il],    cd.time_mix[il],    acc_tm[il]);
                    emit("channel_mix", cs.channel_mix[il], cd.channel_mix[il], acc_cm[il]);
                }
            }

            std::fprintf(stderr, "\n=== PER-LAYER MEAN COSINE SIMILARITY (across %zu pairs) ===\n",
                         pairs.size());
            std::fprintf(stderr, "%-5s  %12s  %12s  %12s\n",
                         "layer", "l_out", "time_mix", "channel_mix");
            std::fprintf(stderr, "-----  ------------  ------------  ------------\n");
            for (int il = 0; il < geom.n_layer; ++il) {
                auto f = [](const Acc & a) { return a.n > 0 ? a.cos_sum / a.n : -1.0; };
                double cl = f(acc_lo[il]);
                double ct = f(acc_tm[il]);
                double cc = f(acc_cm[il]);
                std::fprintf(stderr, "L%-4d  %12.6f  ", il, cl);
                if (ct >= 0.0) std::fprintf(stderr, "%12.6f  ", ct);
                else           std::fprintf(stderr, "     n/a      ");
                std::fprintf(stderr, "%12.6f\n", cc);
            }
        }

        std::fprintf(stderr, "\ndone\n");
    });
}
