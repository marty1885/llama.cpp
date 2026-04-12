// rwkv_probe/generate.cpp — sampler + decode loop implementation.
#include "rwkv_probe/generate.h"

#include "common.h"
#include "sampling.h"
#include "llama.h"

namespace rwkv_probe {

// ── tokenization helpers ─────────────────────────────────────────────────────
std::vector<llama_token> tokenize(const llama_vocab * vocab,
                                  const std::string & text,
                                  bool add_bos,
                                  bool parse_special) {
    int n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(),
                           nullptr, 0, add_bos, parse_special);
    if (n < 0) { n = -n; }
    std::vector<llama_token> tokens((std::size_t) n);
    llama_tokenize(vocab, text.c_str(), (int32_t) text.size(),
                   tokens.data(), (int32_t) tokens.size(), add_bos, parse_special);
    return tokens;
}

std::string piece(llama_context * ctx, llama_token tok) {
    return common_token_to_piece(ctx, tok, false);
}

// ── Generator ────────────────────────────────────────────────────────────────
Generator::Generator(Model & m, Context & c, common_params_sampling sparams)
    : m_ctx(&c) {
    m_smpl = common_sampler_init(m.raw(), sparams);
    if (!m_smpl) {
        die("Generator: common_sampler_init failed");
    }
    m_eos = llama_vocab_eos(m.vocab());
}

Generator::~Generator() {
    if (m_smpl) {
        common_sampler_free(m_smpl);
        m_smpl = nullptr;
    }
}

void Generator::accept_prompt(span<const llama_token> tokens) {
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        common_sampler_accept(m_smpl, tokens[i], false);
    }
}

llama_token Generator::sample_one() {
    llama_token tok = common_sampler_sample(m_smpl, m_ctx->raw(), -1);
    common_sampler_accept(m_smpl, tok, true);
    return tok;
}

Generator::Result Generator::run(int max_tokens, TokenSink sink) {
    Result r;
    for (int i = 0; i < max_tokens; ++i) {
        llama_token tok = sample_one();
        if (tok == m_eos) {
            r.hit_eos = true;
            break;
        }
        std::string p = piece(m_ctx->raw(), tok);
        r.text += p;
        ++r.n_tokens;

        if (sink && !sink(tok, p)) {
            break;
        }

        if (m_ctx->decode_one(tok) != 0) {
            std::fprintf(stderr, "Generator::run: llama_decode failed at step %d\n", i);
            break;
        }
    }
    return r;
}

}  // namespace rwkv_probe
