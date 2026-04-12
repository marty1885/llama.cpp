// rwkv_probe/experiment.cpp — shared experiment utilities.
#include "rwkv_probe/experiment.h"
#include "rwkv_probe/capture.h"
#include "rwkv_probe/generate.h"   // tokenize()

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>

namespace rwkv_probe {

// ── CommonArgs parsing ───────────────────────────────────────────────────────

// Flags that consume the next argv element.
static bool is_common_valued_flag(const std::string & a) {
    return a == "-m" || a == "--n-ctx" || a == "--seed"
        || a == "-n" || a == "--layers" || a == "--tests";
}

CommonArgs parse_common_args(int argc, char ** argv) {
    CommonArgs r;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-h" || a == "--help") {
            r.help_requested = true;
        } else if (a == "-m"       && i + 1 < argc) { r.model_path  = argv[++i]; }
        else if (a == "--n-ctx"    && i + 1 < argc) { r.n_ctx       = std::atoi(argv[++i]); }
        else if (a == "--seed"     && i + 1 < argc) { r.seed        = (uint32_t) std::atoi(argv[++i]); }
        else if (a == "-n"         && i + 1 < argc) { r.n_predict   = std::atoi(argv[++i]); }
        else if (a == "--layers"   && i + 1 < argc) { r.layer_range = argv[++i]; }
        else if (a == "--tests"    && i + 1 < argc) { r.tests_path  = argv[++i]; }
        else {
            // Unknown flag — push into extra.  If it looks like it takes a
            // value (next arg doesn't start with '-'), grab that too.
            r.extra.push_back(a);
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                r.extra.push_back(argv[++i]);
            }
        }
    }
    return r;
}

bool require_model(const CommonArgs & args) {
    if (args.model_path.empty()) {
        std::fprintf(stderr, "error: -m MODEL is required\n");
        return false;
    }
    return true;
}

bool require_tests(const CommonArgs & args) {
    if (args.tests_path.empty()) {
        std::fprintf(stderr, "error: --tests FILE is required\n");
        return false;
    }
    return true;
}

bool reject_extra(const CommonArgs & args) {
    if (!args.extra.empty()) {
        std::fprintf(stderr, "unknown arg: %s\n", args.extra[0].c_str());
        return false;
    }
    return true;
}

// ── ExperimentEnv ────────────────────────────────────────────────────────────

ExperimentEnv::ExperimentEnv(const std::string & model_path, int n_ctx,
                             CaptureRegistry * caps)
    : backend()
    , model(model_path)
    , ctx(model, [&]{
          llama_context_params cp = llama_context_default_params();
          cp.n_ctx = (uint32_t) n_ctx;
          return cp;
      }(), caps)
    , geom(model.geom())
    , vocab(model.vocab())
{}

std::unique_ptr<ExperimentEnv> make_env(const CommonArgs & args,
                                         CaptureRegistry * caps) {
    int ctx = args.n_ctx > 0 ? args.n_ctx : 2048;
    // std::make_unique can't access private ctor, so use new directly.
    return std::unique_ptr<ExperimentEnv>(
        new ExperimentEnv(args.model_path, ctx, caps));
}

// ── JSON helpers ─────────────────────────────────────────────────────────────

nlohmann::json load_json_file(const std::string & path) {
    std::ifstream ifs(path);
    if (!ifs) {
        throw std::runtime_error("cannot open " + path);
    }
    return nlohmann::json::parse(ifs);
}

nlohmann::json load_tests(const CommonArgs & args) {
    return load_json_file(args.tests_path);
}

// ── token helpers ────────────────────────────────────────────────────────────

llama_token find_token(const llama_vocab * vocab, const std::string & word) {
    // try with leading space (most models encode " red" as one token)
    std::string spaced = " " + word;
    auto toks = tokenize(vocab, spaced, /*add_bos=*/false, /*parse_special=*/false);
    if (toks.size() == 1) return toks[0];

    // try bare
    toks = tokenize(vocab, word, /*add_bos=*/false, /*parse_special=*/false);
    if (toks.size() == 1) return toks[0];

    // multi-token — warn
    std::fprintf(stderr, "  WARNING: '%s' is %zu tokens, using FIRST\n",
                 word.c_str(), toks.size());
    return toks.empty() ? (llama_token) -1 : toks[0];
}

float prob_of(float logit_a, float logit_b) {
    float mx = std::max(logit_a, logit_b);
    float ea = std::expf(logit_a - mx);
    float eb = std::expf(logit_b - mx);
    return ea / (ea + eb);
}

std::vector<int> parse_layer_range(const std::string & s, int n_layer) {
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

std::vector<int> parse_int_list(const std::string & s) {
    std::vector<int> out;
    std::istringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        out.push_back(std::atoi(tok.c_str()));
    }
    return out;
}

std::vector<int> parse_int_spec(const std::string & s, int max_val) {
    std::vector<int> out;
    std::istringstream ss(s);
    std::string segment;
    while (std::getline(ss, segment, ',')) {
        auto dash = segment.find('-');
        if (dash != std::string::npos && dash > 0) {
            int lo = std::atoi(segment.substr(0, dash).c_str());
            int hi = std::atoi(segment.substr(dash + 1).c_str());
            hi = std::min(hi, max_val);
            for (int i = lo; i <= hi; ++i) out.push_back(i);
        } else {
            out.push_back(std::atoi(segment.c_str()));
        }
    }
    return out;
}

}  // namespace rwkv_probe
