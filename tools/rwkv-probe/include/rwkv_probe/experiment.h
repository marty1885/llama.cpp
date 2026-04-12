// rwkv_probe/experiment.h — shared utilities for experiment binaries.
//
// Concentrates boilerplate that was copy-pasted across experiments:
// CLI arg parsing, model/context setup, JSON loading, token lookup,
// binary softmax, layer-range parsing.
#pragma once

#include "model.h"

#include "llama.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace rwkv_probe {

class CaptureRegistry;  // capture.h — fwd decl

// ── common CLI args ──────────────────────────────────────────────────────────
// Shared flags recognised by parse_common_args().  Fields with -1 mean
// "not given on the command line" so experiments can fill in their own default.
struct CommonArgs {
    std::string model_path;
    int         n_ctx     = -1;
    int         n_predict = -1;
    uint32_t    seed      = 42;
    std::string layer_range;      // raw string, empty if not given
    std::string tests_path;       // empty if not given

    std::vector<std::string> extra; // args not consumed by parse_common_args
    bool help_requested = false;

    // Apply experiment-specific defaults for fields left at -1.
    void defaults(int ctx = 2048, int predict = 15) {
        if (n_ctx     < 0) n_ctx     = ctx;
        if (n_predict < 0) n_predict = predict;
    }
};

// Parse -m, --n-ctx, --seed, -n, --layers, --tests, -h/--help.
// Everything else lands in result.extra (pairs preserved: if an unknown flag
// takes a value, both flag and value go into extra).
CommonArgs parse_common_args(int argc, char ** argv);

// Validation — print an error and return false when the field is missing.
bool require_model(const CommonArgs & args);
bool require_tests(const CommonArgs & args);

// Reject any leftover extra args (convenience for experiments with no extras).
// Returns false and prints "unknown arg" on the first unknown.
bool reject_extra(const CommonArgs & args);

// ── environment (Backend + Model + Context) ──────────────────────────────────
struct ExperimentEnv {
    Backend            backend;
    Model              model;
    Context            ctx;
    const ModelGeometry & geom;
    const llama_vocab *  vocab;

    ~ExperimentEnv() = default;

private:
    friend std::unique_ptr<ExperimentEnv> make_env(
        const CommonArgs &, CaptureRegistry *);
    ExperimentEnv(const std::string & model_path, int n_ctx,
                  CaptureRegistry * caps);
};

// Construct Backend → Model → Context.  Throws on failure.
std::unique_ptr<ExperimentEnv> make_env(const CommonArgs & args,
                                         CaptureRegistry * caps = nullptr);

// ── JSON helpers ─────────────────────────────────────────────────────────────
nlohmann::json load_json_file(const std::string & path);
nlohmann::json load_tests(const CommonArgs & args);  // shorthand for load_json_file(args.tests_path)

// ── try/catch wrapper ────────────────────────────────────────────────────────
template <typename Fn>
int run_experiment(Fn && fn) {
    try {
        fn();
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "fatal: %s\n", e.what());
        return 1;
    }
}

// ── token lookup ─────────────────────────────────────────────────────────────
// Find the token id for a short word like "red" or "Paris".
// Tries " word" first (BPE leading space), then bare "word".
// Returns the first token if multi-token; returns -1 if empty.
// Prints a warning to stderr when the word requires multiple tokens.
llama_token find_token(const llama_vocab * vocab, const std::string & word);

// ── probability helpers ──────────────────────────────────────────────────────
// Binary softmax: P(a | {a, b}) from raw logits.
float prob_of(float logit_a, float logit_b);

// ── argument parsing helpers ─────────────────────────────────────────────────
// Parse "16-31" or "16" into a vector of layer indices.
// Clamps to [0, n_layer).
std::vector<int> parse_layer_range(const std::string & s, int n_layer);

// Parse "1,2,3" into a vector of ints.
std::vector<int> parse_int_list(const std::string & s);

// Parse a combined spec: "1,3-5,7" → {1,3,4,5,7}. Handles both commas and
// ranges. Clamps range endpoints to [0, max_val].
std::vector<int> parse_int_spec(const std::string & s, int max_val = 0x7fffffff);

}  // namespace rwkv_probe
