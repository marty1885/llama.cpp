# RWKV GGML Instrumentation

This directory contains reusable C++ instrumentation for quantized RWKV inference on
GGML backends, including Vulkan and HIP. GGML remains the production static-graph
executor; experiments invoke the production decode graph rather than reconstructing
RWKV in a separate graph.

## Build

Enable the tools explicitly so ordinary llama.cpp builds do not require ROOT:

```bash
cmake -B build -DLLAMA_BUILD_RWKV_INTERP=ON
cmake --build build --target llama-rwkv-interp-verify llama-rwkv-activation-collect
```

## Safety Policy

Every captured tensor is a `ggml_dup` snapshot registered as a graph output before
host readback. The core capture collector rejects any capture that is not such a
materialized snapshot. Do not read an intermediate GGML tensor directly: its buffer
may be reused after graph execution.

`llama-rwkv-interp-verify` checks three invariants on a production graph:

1. A captured final residual reproduces native logits through the production final
   norm and output head.
2. Exporting/importing RWKV recurrent state reproduces the uninterrupted decode.
3. The requested named capture reproduces across that state handoff.

```bash
build/bin/llama-rwkv-interp-verify -m MODEL.gguf -ngl 99
```

## Jacobian Replay Contract

`llama-rwkv-jacobian-replay-verify` is the first validation primitive for the
corpus-averaged Jacobian-lens program. Its `LensSpec` version, exact source and
target tap names, direction seed, finite-difference scale, and offset semantics
are emitted to JSON with every run. Sources are pluggable from the beginning;
the verifier discovers the source tensor shape from a production snapshot rather
than assuming a residual width.

For source token `t` and offset `d`, it measures:

```text
(target(source + epsilon * v, t + d) - target(source - epsilon * v, t + d))
/ (2 * epsilon)
```

Both branches import the same recurrent state immediately before `t`, process the
same source token, and teacher-force the exact same suffix tokens. Offset zero is
the target activation on the source token. Before interpreting a response, run
`llama-rwkv-interp-verify` on the same backend and require this verifier's zero
perturbation checks to pass. It evaluates `0.5x`, `1x`, and `2x` of the requested
relative epsilon by default; `--epsilon-scale` replaces that set when a different
diagnostic sweep is needed, but the verifier always includes and repeats `1x`.
It fails closed before corpus collection unless every requested offset meets these
pre-registered defaults: scale-response cosine at least `0.995`, response-norm
spread at most `5%`, center-error ratio at most `5%`, and repeat-`1x` cosine at
least `0.999`. Threshold flags are recorded in JSON if deliberately changed.

```bash
build/bin/llama-rwkv-jacobian-replay-verify \
  -m MODEL.gguf -ngl 99 \
  -p "The capital of France is Paris. The capital of Germany is Berlin." \
  --source-tap rwkv.layer.60.resid.out \
  --target-tap rwkv.layer.60.resid.out \
  --offset 0 --offset 1 --offset 2 \
  --source-position 3 \
  --output build/jacobian-replay-verify.json
```

When source and target taps are identical at offset zero, the output includes an
identity-control comparison between the finite-difference response and the seeded
probe direction. This is a positive control, not a claim that all source taps
transport identically through future recurrent state.

## ROOT Capture

`llama-rwkv-activation-collect` writes selected, same-width named taps to a ROOT
RNTuple. It commits clusters every `--commit-rows` records, so completed records
survive an interrupted long run. The dataset records source names and token metadata.

```bash
build/bin/llama-rwkv-activation-collect \
  -m MODEL.gguf -ngl 99 \
  --corpus prompts.txt --output captures.root \
  --tap rwkv.layer.60.resid.out \
  --tap rwkv.layer.60.resid.in
```

The collector accepts exact production tap names only. It intentionally contains no
mapping, lens, clustering, or semantic-analysis logic.

## Experiment JSON

`rwkv_experiment.hpp` is a small helper for separate experiments: it captures selected
production taps at every input-token position and writes named top-logit rows in a
stable JSON format. It does not add experiment-specific transformations to the shared
runtime.

`llama-rwkv-inverse-cache-build` performs the expensive exact inversions once and writes
them to a disk-backed ROOT `TTree`. Each entry contains the canonical model path, layer,
projection name, and one F32 inverse matrix. The cache currently contains `time_mix_key`,
`time_mix_value`, and `time_mix_output` inverses. Cache files are large for 4096-wide
models, so place them under `build/` or another disk-backed location, never `/tmp`.

```bash
build/bin/llama-rwkv-inverse-cache-build \
  -m MODEL.gguf -ngl 99 --output build/rwkv-inverses.root
```

`llama-rwkv-key-lens` loads the cached `W_K` inverses, uploads the aggregate
`[K, M, L, 1]` inverse bank once, and asks the production graph to write
each `time.k0` into a persistent `[K, 1, L, 1]` aggregate with dependency-chained
`ggml_set_inplace` nodes, then produces `[vocab, 1, L, 1]` lens logits directly.

```bash
build/bin/llama-rwkv-key-lens \
  -m MODEL.gguf -ngl 99 -p "The capital of France is" \
  --inverse-cache build/rwkv-inverses.root --top-n 20 --output key-lens.json
```

The GGML shape of `W_K` is `[input, output]`; its usual linear-algebra shape is
`[output, input]`. `time.k0` is `W_K x^k`, so the lens reconstructs an approximation
of `x^k`, not a layer residual or an exact model continuation.

The JSON is viewer-oriented and hierarchical: inputs contain layers, and layers contain
named lenses. `top_logits` is sorted descending by raw logit.

```json
{
  "schema_version": 2,
  "experiment": "rwkv-key-pseudoinverse-logit-lens",
  "top_n": 20,
  "inputs": [
    {
      "position": 12,
      "input_token": {"id": 123, "text": " token"},
      "layers": [{
        "layer": 59,
        "lenses": [{
          "lens": "rwkv.layer.59.time.k0.pinv",
          "result": {"top_logits": [{"token_id": 456, "token": " next", "logit": 8.25}]}
        }]
      }]
    }
  ]
}
```

`llama-rwkv-key-input-lens` compares direct final-readout lenses of the attention
normalized layer input and the key time-mix input without constructing inverses:

```bash
build/bin/llama-rwkv-key-input-lens \
  -m MODEL.gguf -ngl 99 -p "The capital of France is" \
  --top-n 20 --output key-input-lens.json
```

`llama-rwkv-memory-readout-lens` directly lenses the value time-mix input and the
post-gate/pre-output-projection activation; it does not construct or load inverses:

```bash
build/bin/llama-rwkv-memory-readout-lens \
  -m MODEL.gguf -ngl 99 -p "The capital of France is" \
  --generate 32 --top-n 20 --output value-path-lens.json
```

`--generate N` greedily samples `N` production next tokens. Each sampled token is then
processed with the same named captures and appears after the prompt tokens in `inputs`.
