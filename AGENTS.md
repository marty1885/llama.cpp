# RWKV GGML Instrumentation Fork

This fork develops discovery-first mechanistic interpretability tooling for quantized
RWKV models. GGML is the production static-graph executor and backend abstraction,
not a model to reimplement. Vulkan and HIP support, including AMD execution, and
standard GGUF quantization are core constraints.

## Architecture

- `src/llama-ext.h`: provisional C++ interpretability API.
- `src/llama-interp.cpp`: RWKV recurrent-state import/export.
- `src/llama-graph.cpp`: named production-graph taps and perturbations.
- `src/llama-context.cpp`: synchronized, host-side capture collection and final
  residual readout.
- `pocs/interp/interp.hpp`: C++ runtime for invoking production prefill/decode
  operations with captured state, named taps, and perturbations.
- `pocs/interp/rwkv_activation_store.*`: ROOT RNTuple persistence.
- `pocs/interp/rwkv_activation_collect.cpp`: generic named-tap ROOT collector.
- `pocs/interp/rwkv_interp_verify.cpp`: capture/state/output validation executable.

## Non-Negotiable Capture Policy

GGML intermediates may alias or be reused after graph execution. Never read an
intermediate tensor directly. Every host-readable capture must be an explicit
`ggml_dup` snapshot registered as a graph output. The core collector enforces this
before transfer. Keep this invariant if changing graph or capture code.

Validate instrumentation before interpreting any activation:

```bash
cmake -S . -B build -DLLAMA_BUILD_RWKV_INTERP=ON
cmake --build build --target llama-rwkv-interp-verify llama-rwkv-activation-collect
build/bin/llama-rwkv-interp-verify -m MODEL.gguf -ngl 99
```

The verifier checks captured-final-residual readout against native logits and checks
state handoff/capture replay. Run it on the backend used for a data collection.

## Tooling Rules

- C++ is the experiment scripting language. Do not add Python analysis pipelines.
- Persist long captures in ROOT RNTuples. Commit regularly so completed clusters
  survive process interruption.
- Build instrumentation only with `-DLLAMA_BUILD_RWKV_INTERP=ON`; ordinary builds
  must not require ROOT or VDT.
- Invoke the production decode graph through `llama_interp::runtime`. Do not add a
  manual reconstruction of RWKV layers for an experiment.
- Keep collection, validation, and a future experiment separate. Do not retain
  speculative lenses, mapping code, viewers, or one-off semantic analyses as shared
  infrastructure.

## Scope

This repository contains the llama.cpp inference stack plus a small RWKV
instrumentation layer. The instrumentation layer is intentionally limited to
production-graph execution, capture validation, and durable activation collection;
individual mechanistic experiments belong in separate C++ programs built on that base.
