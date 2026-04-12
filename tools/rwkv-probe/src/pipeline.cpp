// rwkv_probe/pipeline.cpp — state capture and probing pipeline.
#include "rwkv_probe/pipeline.h"
#include "rwkv_probe/experiment.h"  // prob_of

#include "common.h"  // common_params_sampling

#include <cstdio>

namespace rwkv_probe {

StateBuf capture(Context & ctx, const ModelGeometry & geom,
                 const llama_vocab * vocab, const std::string & text) {
    auto toks = tokenize(vocab, text);
    ctx.clear_memory();
    ctx.decode(span<const llama_token>(toks.data(), toks.size()));
    StateBuf state(geom);
    state.load_from(ctx.raw());
    return state;
}

StateBuf capture_for_probe(Context & ctx, const ModelGeometry & geom,
                           const llama_vocab * vocab, const std::string & text) {
    auto toks = tokenize(vocab, text);
    if (toks.empty()) {
        die("capture_for_probe: empty text after tokenization");
    }
    ctx.clear_memory();
    // decode all but last
    if (toks.size() > 1) {
        ctx.decode(span<const llama_token>(toks.data(), toks.size() - 1));
    }
    StateBuf state(geom);
    state.load_from(ctx.raw());
    // decode last token to make logits available
    ctx.decode_one(toks.back());
    return state;
}

ProbeResult probe(Model & model, Context & ctx,
                  const StateBuf & state,
                  const std::string & prefix,
                  const std::string & query,
                  llama_token tok_expected, llama_token tok_alt,
                  const llama_vocab * vocab,
                  int n_predict, uint32_t seed) {
    auto prefix_toks = tokenize(vocab, prefix);
    auto query_toks  = tokenize(vocab, query, /*add_bos=*/false, /*parse_special=*/false);

    // decode prefix to establish sequence position
    ctx.clear_memory();
    ctx.decode(span<const llama_token>(prefix_toks.data(), prefix_toks.size()));

    // inject edited state
    state.store_to(ctx.raw());

    // decode query tokens
    for (std::size_t i = 0; i < query_toks.size(); ++i) {
        ctx.decode_one(query_toks[i]);
    }

    const float * logits = llama_get_logits(ctx.raw());

    ProbeResult r;
    r.logit_expected = logits[tok_expected];
    r.logit_alt      = logits[tok_alt];
    r.p_binary       = prob_of(r.logit_expected, r.logit_alt);

    if (n_predict > 0) {
        common_params_sampling sp;
        sp.seed = seed;
        Generator gen(model, ctx, sp);
        llama_token last = query_toks.back();
        gen.accept_prompt(span<const llama_token>(&last, 1));
        r.generation = gen.run(n_predict).text;
    }

    return r;
}

float measure(Context & ctx, const StateBuf & state,
              const std::vector<llama_token> & toks,
              llama_token tok_a, llama_token tok_b) {
    ctx.clear_memory();
    ctx.decode(span<const llama_token>(toks.data(), toks.size() - 1));
    state.store_to(ctx.raw());
    ctx.decode_one(toks.back());
    const float * logits = llama_get_logits(ctx.raw());
    return prob_of(logits[tok_a], logits[tok_b]);
}

std::string generate_from(Model & model, Context & ctx,
                          const StateBuf & state,
                          const std::vector<llama_token> & toks,
                          int n_predict, uint32_t seed) {
    ctx.clear_memory();
    ctx.decode(span<const llama_token>(toks.data(), toks.size() - 1));
    state.store_to(ctx.raw());
    ctx.decode_one(toks.back());

    common_params_sampling sp;
    sp.seed = seed;
    Generator gen(model, ctx, sp);
    llama_token last = toks.back();
    gen.accept_prompt(span<const llama_token>(&last, 1));
    std::string text = gen.run(n_predict).text;
    for (auto & c : text) {
        if (c == '\n') c = ' ';
    }
    return text;
}

}  // namespace rwkv_probe
