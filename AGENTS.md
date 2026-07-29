# RWKV GGML Instrumentation Fork

This fork develops discovery-first mechanistic interpretability tooling for quantized
RWKV models. GGML is the production static-graph executor and backend abstraction,
not a model to reimplement. Vulkan and HIP support, including AMD execution, and
standard GGUF quantization are core constraints.

## Scientific Agent Policy

> **FOLLOW THIS. YES, YOU WILL NEED IT. DO NOT SKIP IT. READ IT AND INTERNALIZE IT.**

Follow `SCIENTIFIC_METHOD_PROMPT.md`. It is the policy for experimental reasoning,
claims, hypotheses, and proposed tests in this repository.

Before proposing or implementing an experiment, read
`pocs/interp/logs/TMIX_INVESTIGATION.md`. It is the durable ledger of observations,
failed interpretations, open questions, and superseded experiments. Do not infer
the current scientific question from executable names alone.

Discuss the scientific question, competing interpretations, and discriminating
outcomes with the user before drafting or implementing a new experiment. The user
handles agent dispatch; do not spawn subagents unless explicitly requested.

Use Markdown and Python-style pseudocode for mathematical explanations. Avoid raw
LaTeX unless the user explicitly requests it.

## Current Scientific Objective

Determine what RWKV memory/recurrence contributes to the residual at each token and
layer, and whether that contribution admits a validated map into output-embedding
coordinates.

The current representational question is not whether RWKV literally executes a
rotor instead of addition and LayerNorm. The production operation is:

```python
resid_time = resid_in + time_out
ffn_input = layer_norm(resid_time)
```

On the centered pre-affine LayerNorm sphere, this same composite operation defines
a canonical minimal rotor from the incoming direction to the resulting direction.
`ADD + LayerNorm` and this rotor description are mathematically equivalent, not
competing implementations. The falsifiable question is whether memory content is
organized and consumed coherently in rotor/tangent coordinates, or whether the
rotor is only incidental normalization geometry.

The current working hypothesis is that TMix may use this geometry to move the
normalized residual into an unused or less-used destination that marks information
contributed by memory, after which the MLP resolves it. The competing explanation
is ordinary normalization of a structured additive update. Establishing a stable
destination is an intermediate observation; proving that the model uses the
rotation requires a discriminating result beyond the guaranteed geometry.

Established interpretation boundaries:

- `time.out` is an update in residual coordinates, not a standalone residual
  activation. Applying the final output head directly to it has no established
  interpretation and often produces garbage.
- A canonical minimal rotor in high dimension is represented by its oriented
  two-plane and angle. Do not claim that high dimensionality eliminates the
  axis-like rotation object. Other rotors can map the same vector pair while also
  rotating the orthogonal complement; the minimal rotor is the relevant default.
- LayerNorm first projects into the mean-zero hyperplane, normalizes toward a
  sphere, then applies learned gain and bias. Define rotors before the learned
  affine map. After gain, the corresponding geometry is generally an ellipsoid.
- The raw residual skip remains available downstream. LayerNorm makes direction
  decisive for its normalized branch, but does not by architecture prove that the
  entire remaining model ignores mean or radius.
- The exported RWKV recurrent state is a tangled contextual superposition. State
  swaps, state deltas, rank approximations of state differences, and zero-state
  interventions do not isolate a memory item and cannot establish its content.
- Controlled memory writing must use live RWKV quantities, including live
  receptance/value information, and RWKV's native production write mechanism.
- Logits from a zero-memory or state-ablated rollout include the trained model's
  policy response to the intervention. They are causal behavior effects, not a
  policy-free decoding of memory content.
- A Jacobian or transported vector may be expressed in output-embedding
  coordinates only after numerical validity and the source representation have
  been established. Vocabulary-looking output is not validation.

## Experiment Admission Gate

Do not start an experiment until its note or implementation states all of the
following:

```text
scientific question
observation already established
competing hypotheses
intervention and measured quantity
positive outcome
negative outcome
matched controls
why the positive outcome is not guaranteed by the architecture
numerical validity criteria
claim permitted by each outcome
```

In particular, observing an angle, tangent, rotor, or output change after
`ADD + LayerNorm` is tautological and is not evidence for a rotor code. Evidence
would require content specificity, appropriate mean/radius and random controls,
causal use or recovery of independently known native writes, and held-out
generalization. If no result could distinguish the hypotheses, do not run the
experiment.

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
- Do not add a broad all-layer or corpus sweep before a single-layer, single-token
  discriminating test has passed its pre-registered gates.

## Scope

This repository contains the llama.cpp inference stack plus a small RWKV
instrumentation layer. The instrumentation layer is intentionally limited to
production-graph execution, capture validation, and durable activation collection;
individual mechanistic experiments belong in separate C++ programs built on that base.
