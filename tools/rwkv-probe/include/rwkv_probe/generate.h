// rwkv_probe/generate.h — sampler + token-by-token decode loop.
//
// Lifts the per-token generation block from the original rwkv-probe.cpp
// (lines 729-747) and the sampler init/teardown (632, 749) into a small RAII
// wrapper.
#pragma once

#include "util.h"
#include "model.h"

#include "common.h"        // llama.cpp's common.h: common_params_sampling

#include <functional>
#include <string>
#include <vector>

struct common_sampler;

namespace rwkv_probe {

// Tokenization helpers (lifted from the original tokenize() / token_to_text()).
std::vector<llama_token> tokenize(const llama_vocab * vocab,
                                  const std::string & text,
                                  bool add_bos = true,
                                  bool parse_special = true);

std::string piece(llama_context * ctx, llama_token tok);

class Generator {
public:
    Generator(Model & m, Context & c, common_params_sampling sparams = {});
    ~Generator();

    Generator(const Generator &)             = delete;
    Generator & operator=(const Generator &) = delete;

    // feed prompt tokens into the sampler (for repetition-penalty context).
    // Does NOT decode them; the caller decodes via Context::decode().
    void accept_prompt(span<const llama_token> tokens);

    // sample one token, accept it, return it. Does not decode.
    llama_token sample_one();

    // streaming generation: sample → accept → call sink → decode → repeat.
    // sink receives each token and its decoded piece; return false to stop.
    // stops on EOS or after max_tokens, whichever comes first.
    using TokenSink = std::function<bool(llama_token, const std::string & piece)>;

    struct Result {
        std::string text;
        int         n_tokens = 0;
        bool        hit_eos  = false;
    };

    Result run(int max_tokens, TokenSink sink = {});

private:
    Context *        m_ctx = nullptr;
    common_sampler * m_smpl = nullptr;
    llama_token      m_eos = 0;
};

}  // namespace rwkv_probe
