// rwkv_probe/pipeline.h — state capture and probing pipeline.
//
// Concentrates the three pipeline shapes that every experiment re-derives:
//   1. capture(text) → StateBuf after full text
//   2. probe(state, prefix, query) → logit measurement + optional generation
//   3. measure() / generate_from() — quick convenience wrappers
#pragma once

#include "util.h"
#include "model.h"
#include "state.h"
#include "generate.h"

#include <cstdint>
#include <string>
#include <vector>

namespace rwkv_probe {

// ── state capture ────────────────────────────────────────────────────────────

// Decode full text and capture the resulting recurrent state.
StateBuf capture(Context & ctx, const ModelGeometry & geom,
                 const llama_vocab * vocab, const std::string & text);

// Decode all-but-last token, capture state, then decode the last token so
// logits are available via llama_get_logits(ctx.raw()).
// Returns the captured state (before the final token).
StateBuf capture_for_probe(Context & ctx, const ModelGeometry & geom,
                           const llama_vocab * vocab, const std::string & text);

// ── probing (inject state → decode query → measure) ─────────────────────────

struct ProbeResult {
    float p_binary;         // P(expected | {expected, alt})
    float logit_expected;
    float logit_alt;
    std::string generation; // empty if n_predict == 0
};

// Inject `state` into the context, decode prefix (to establish the sequence),
// inject state again, decode query tokens, and measure logits at the end.
//
// The prefix should be the text that produced `state` (so the sequence
// position matches). The query is the continuation to probe.
ProbeResult probe(Model & model, Context & ctx,
                  const StateBuf & state,
                  const std::string & prefix,
                  const std::string & query,
                  llama_token tok_expected, llama_token tok_alt,
                  const llama_vocab * vocab,
                  int n_predict = 0, uint32_t seed = 42);

// ── quick measurement ────────────────────────────────────────────────────────

// Decode tokens[:-1], inject state, decode last token, return P(a | {a,b}).
float measure(Context & ctx, const StateBuf & state,
              const std::vector<llama_token> & toks,
              llama_token tok_a, llama_token tok_b);

// Decode tokens[:-1], inject state, decode last token, generate continuation.
// Newlines in the output are replaced with spaces.
std::string generate_from(Model & model, Context & ctx,
                          const StateBuf & state,
                          const std::vector<llama_token> & toks,
                          int n_predict, uint32_t seed);

}  // namespace rwkv_probe
