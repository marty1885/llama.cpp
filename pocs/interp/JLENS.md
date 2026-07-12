# RWKV J-Lens

`llama-interp-jlens-build` estimates a low-rank, current-and-future Jacobian operator from
`rwkv.layer.L.resid.out` by default, or from RWKV's addressed memory read
`rwkv.layer.L.time.wkv` with `--source-activation time-wkv`, to a selected later residual. It writes FP32 Rademacher probe
directions, their averaged FP32 responses, a manifest, and the exact corpus line and
position samples used for the build and held-out validation.

## Build

Configure and build the tool:

```bash
cmake -S . -B build
cmake --build build --target llama-interp-jlens-build -j 2
```

Run a bounded pilot on a line-delimited corpus:

```bash
JLEN_VK_BUILD_VMEM_KIB=40000000 \
JLEN_VK_BUILD_N_GPU_LAYERS=24 \
JLEN_VK_BUILD_LAYER=37 \
JLEN_VK_BUILD_RANK=64 \
JLEN_VK_BUILD_SAMPLES_PER_DIRECTION=16 \
JLEN_VK_BUILD_VALIDATION_SAMPLES=64 \
bash pocs/interp/jlens_vulkan_build.sh MODEL CORPUS OUTPUT_PREFIX
```

The corpus must have enough independent lines for both partitions. The builder assigns
one in every five lines to validation using its recorded split seed; pass a larger
corpus for a scientific result. `jlens_smoke_corpus.txt` is only an end-to-end fixture.

`--target-layer` selects the response residual and defaults to the final residual. The
readout tool can apply an artifact targeted at any later residual by replacing that residual
on a fresh carrier token and running the remaining RWKV blocks.

For a same-token memory-readout lens, use `--source-activation time-wkv` with
`--min-future 0 --future-window 0`. `time.wkv` is the current token's query-dependent
read from the recurrent WKV state, so this maps retrieved memory to the final residual
that produces the current next-token logits. Do not use the strict-future wrapper defaults
for this experiment.

Use `--all-future` to average each sampled source position over every target from
`--min-future` through `--future-window`; otherwise the builder samples one target uniformly.
The manifest records `future_aggregation`.

## Native Source Checks

`llama-interp-jlens-operator` checks a single same-token native perturbation before
building an operator. It accepts `--source-activation residual` (the default) or any
of `time-r`, `time-w`, `time-k`, `time-v`, `time-a`, `time-g`, `time-wkv`, and
`time-rkv`. It captures the source, its layer's `time.out`, and the final residual in
FP32; it then verifies source injection, plus/minus symmetry, deterministic replay,
and agreement at `--epsilon` and `--compare-epsilon`.

With `--center-root CENTROIDS.root`, a raw-source check uses the normalized direction
`x - c`, where `c` is that source's `K=1` centroid. It still perturbs around the live
activation `x`; it does not replace the activation with `c`. The centroid file uses the
same source ordering as `--all-time-mix` collection.

With `--direction-prompt TEXT`, the runner instead captures the source from `TEXT` and
uses normalized `source(primary prompt) - source(TEXT)`. The primary prompt remains the
only perturbed execution; the contrast prompt supplies a direction, not recurrent state.

```bash
build/bin/llama-interp-jlens-operator \
  -m MODEL.gguf -ngl 999 \
  --layer 36 --source-activation time-wkv \
  --center-root CENTROIDS.root \
  --epsilon 0.20 --compare-epsilon 0.10 \
  -p "A computer's central processing unit is the"
```

It also passes the plus and minus final residuals through the production final
normalization/output head and prints their central logit derivative. Rademacher
directions are numerical controls only; their token lists are not semantic evidence.

Pass `--json TRACE.json` to export a viewer-compatible perturbation trace. Open
`pocs/interp/rwkv_unembed_viewer.html` and select the JSON file; its positive and
negative cards are independently sorted signed final-logit derivatives for the raw
source direction, not direct unembeddings or generated continuation tokens.

## Hardware

Use one Vulkan process at a time. The BF16 model is about 25.5 GiB, so it cannot be
fully offloaded to the current 11.4 GiB free discrete Vulkan heap. The previous 30 GiB
virtual-memory limit also prevented deeper suffixes because CPU-mapped weights and
Vulkan mappings share process address space.

On the current machine, `-ngl 24` runs layers 38--60 on Vulkan with a 9.74 GiB model
buffer and supports source layer 37 when `JLEN_VK_BUILD_VMEM_KIB=40000000`. Lower
layers require a larger Vulkan heap. On a Strix Halo machine, first select the intended
device with `JLEN_VK_BUILD_VISIBLE_DEVICES`, then increase `JLEN_VK_BUILD_N_GPU_LAYERS`
gradually. Use `999` only after confirming that the Vulkan heap exceeds model weights,
recurrent state, and compute buffers.

## Acceptance

Treat a build as a numerical artifact only when its manifest shows low symmetry error,
exact repeat-plus replay, and strong epsilon agreement. Then increase rank and samples
until held-out `validation_mean_cosine` and `validation_mean_relative_l2` stabilize.
The rank-1 smoke result is expected to have poor held-out operator accuracy and must
not be interpreted as a usable lens.

The stored operator is applied as:

```text
Jhat(x) = (input_dimension / rank) * sum_k response[k] * dot(direction[k], x)
```

## Applied Readout

`llama-interp-jlens-readout --operator` applies a saved builder artifact to the
source activation recorded in the artifact at the final token of a prompt, then routes the resulting final
residual through RWKV's production final normalization and output head:

```bash
build/bin/llama-interp-jlens-readout \
  -m MODEL -ngl 12 \
  --operator OUTPUT_PREFIX \
  --output READOUT_PREFIX \
  -p "The capital of France is"
```

This produces `READOUT_PREFIX.action.f16`, a manifest, and a top-token readout.
A low-rank smoke artifact is only a plumbing check: do not interpret its tokens
until held-out operator accuracy converges as rank and corpus samples increase.

## Held-out Semantic Evaluation

`llama-interp-jlens-eval` scores a saved artifact against TSV rows of
`id<TAB>prompt<TAB>single-token intermediate`. It reports J-lens and a norm-matched
random-readout rank for each predeclared case. Residual-source artifacts additionally use
a logit-lens control; `time-wkv` artifacts report a raw source-readout control instead.

```bash
build/bin/llama-interp-jlens-eval \
  -m MODEL -ngl 999 \
  --operator OUTPUT_PREFIX \
  --cases CASES.tsv \
  --output RESULTS.tsv
```

`jlens_eval_smoke_cases.tsv` only verifies the format. Scientific cases must be held out
from the corpus and use known intermediates absent from prompt and final answer. Compare
rank and pass@k across increasing rank/sample counts and independent seeds; the J-lens must
beat both controls before it is interpreted.

## Coordinate Swaps

For a final-residual artifact, `llama-interp-jlens-intervene` derives two source-space
token directions from the low-rank operator and RWKV output weight, swaps their local
pseudoinverse coordinates, and reports the target token's clean and patched rank:

```bash
build/bin/llama-interp-jlens-intervene \
  -m MODEL -ngl 999 \
  --operator OUTPUT_PREFIX \
  --source-token " Paris" \
  --target-token " London" \
  -p "The capital of France is"
```

Both tokens must tokenize to one token. The runner also reports a deterministic
matched-norm random-direction control. This is an initial causal test, not a workspace
result: use same-category targets absent from the clean top-10 and add non-J controls.
Clamping later recurrent steps remains necessary to exclude re-derivation.
