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
