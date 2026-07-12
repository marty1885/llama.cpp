# RWKV J-Lens Progress Log

Last updated: 2026-07-11

## Goal

Determine whether a Jacobian lens produces useful, held-out, verbalizable readouts for
the RWKV7 13.3B BF16 checkpoint on HIP. A numerical finite-difference pass alone is not
evidence of a usable lens. The required evidence is:

1. Stable held-out operator estimates as rank and samples increase.
2. Held-out semantic intermediate recovery better than logit-lens and random controls.
3. Causal token-coordinate interventions better than matched-norm random controls.

If a global residual lens fails after these checks, test the model's query-dependent
recurrent-memory readout (`time.wkv`) rather than treating raw state as a flat feature vector.

## Environment

- Repository branch: `rwkv-interp`.
- Model: `rwkv7-g1h_preview4743-13.3b-20260703-ctx8192-BF16.gguf` in the repository root.
- Hardware: AMD Ryzen AI MAX+ 395 / Radeon 8060S, ROCm/HIP `gfx1151`.
- All real runs use `-ngl 999`; the model reports 61 RWKV blocks indexed 0 through 60.
- CPU BF16 is not a reference backend for this work.

## Numerical Foundation

The HIP path passed the backend parity and finite-difference sanity checks after two
interpreter fixes:

- `llama_interp_rwkv_state_export()` now synchronizes before reading ROCm-resident state.
- Experimental RWKV recurrent state is retained in FP32 rather than FP16.

Reference HIP parity trace:

```text
state handoff logits: exact
source perturbation injection relative_l2_error: 1.621024e-05
clean and perturbed replay logits: exact
```

The layer-37 operator sanity check also found deterministic replay, nonzero response, and
central-derivative cosine `0.9753` between epsilon 0.20 and 0.10. This validates the
instrumentation; it does not validate a semantic lens.

## Implemented Tooling

- `llama-interp-jlens-build`
  - Produces a rank-factorized FP32 source-to-target residual operator.
  - Supports `--target-layer`, `--all-future`, and
    `--source-activation residual|time-wkv`.
  - `time-wkv` selects `rwkv.layer.L.time.wkv`, the current token's addressed read
    from that layer's recurrent WKV state.
  - `--all-future` averages each selected source position across all targets from
    `--min-future` through its available `--future-window`.
  - Emits periodic `jlens build` and `jlens validation` progress lines.
- `llama-interp-jlens-readout`
  - Applies an operator targeted at any later residual by replacing it on a carrier token.
- `llama-interp-jlens-eval`
  - Scores TSV rows: `id<TAB>prompt<TAB>single-token intermediate`.
  - Reports J-lens, raw-source, and norm-matched random-readout ranks for memory-readout
    artifacts. Raw `time.wkv` projected through the output head is not called a logit lens.
  - Emits `jlens eval` progress lines.
- `llama-interp-jlens-intervene`
  - For final-residual operators only, derives two rank-factorized token directions,
    swaps local pseudoinverse coordinates, and reports a matched-norm random control.

See `JLENS.md` for command syntax and limits.

## Data

The initial corpus is Penn Treebank newswire training text:

```text
source: https://github.com/wojzaremba/lstm
local:  /tmp/opencode/ptb-lstm/data/ptb.train.txt
lines:  42,068
bytes:  5,101,618
```

The semantic cases are generated, separate from the corpus, by:

```bash
python3 pocs/interp/generate_jlens_arithmetic_cases.py
```

Generated files:

- `pocs/interp/jlens_arithmetic_cases.tsv`: 128 hidden-intermediate arithmetic prompts.
- `pocs/interp/jlens_arithmetic_answer_control.tsv`: 16 direct-answer controls.

These arithmetic prompts are a first gate, not a sufficient general semantic benchmark.

## Completed Pilot

All artifacts below use source layer 37, final residual target layer 60, strict horizon 1,
and PTB with split seed 17. Long commands were run through:

```bash
... 2>&1 | rg --line-buffered '^(jlens |built |validation |epsilon |wrote |.*error)'
```

### Rank 64

Artifact prefix: `/tmp/opencode/rwkv7-jlens-l37-h1-r64s4`

```text
rank: 64
samples per direction: 4
build samples: 256
validation samples: 64
validation cosine: 0.0338322
validation relative_l2: 10.8941
mean symmetry: 0.00576701
```

`--compare-epsilon 0` and no repeat-plus replay were used for this compute-bounded semantic

Held-out arithmetic-intermediate evaluation:

```text
cases: 128
J mean rank: 10220.5
logit mean rank: 6633.1
random mean rank: 23715.5
J MRR: 0.00015665
logit MRR: 0.00025300
random MRR: 0.00014524
J beats random: 103/128
J beats logit: 47/128
top-10 recovery: 0 for all three methods
```

Interpretation: the rank-64 residual operator has non-random signal, but it is weaker than
the logit lens on this task. This is neither a semantic J-lens pass nor a valid failure.

### Rank 128

Artifact prefix: `/tmp/opencode/rwkv7-jlens-l37-h1-r128s4`

```text
rank: 128
samples per direction: 4
build samples: 512
validation samples: 64
validation cosine: 0.00911026
validation relative_l2: 8.17246
mean symmetry: 0.0056529
```

The rank-128 semantic evaluation has not yet been run. The held-out operator cosine did not
improve over rank 64, indicating substantial context dependence and/or estimator variance.

### Same-Token Memory Readout: Rank 64

Artifact prefix: `/tmp/opencode/rwkv7-timewkv-l37-h0-r64s4`

This is a different experiment from the strict-future residual pilot. It maps the
query-dependent WKV memory read at layer 37 on token `t` to the final residual on that same
token, which produces logits for token `t+1`:

```text
source: rwkv.layer.37.time.wkv
target: rwkv.layer.60.resid.out
rank: 64
samples per direction: 4
build samples: 256
validation samples: 64
min_future: 0
future_window: 0
validation cosine: 0.00943074
validation relative_l2: 14346.6
mean symmetry: 0.142113
```

The rank-1 end-to-end smoke artifact passed capture, perturbation, saved readout, and
semantic-evaluator plumbing. The rank-64 build used no epsilon comparison or repeat-plus
replay, so it is a semantic gate rather than a fully checked numerical artifact.

Held-out arithmetic-intermediate evaluation:

```text
cases: 128
J mean rank: 21120.2
raw time.wkv source mean rank: 31985.3
random mean rank: 23715.5
J MRR: 0.00007255
raw source MRR: 0.00003858
random MRR: 0.00014524
J beats random: 69/128
J beats raw source: 103/128
top-10 recovery: 0 for all three methods
```

Interpretation: the layer-37, rank-64 same-token memory-readout operator improves over
the raw WKV readout but is worse than the matched-norm random control on this arithmetic
gate. This rejects that exact layer/rank/task configuration; it does not show that recurrent
memory is not readable through `time.wkv` at another layer or with a model-native benchmark.

## Current Assessment

The evidence currently supports a weak source-residual signal above random, but does not
support a useful global residual J-lens. The first same-token `time.wkv` lens is weaker than
random at layer 37. Rank 64 is only 1.6% of the 4,096-dimensional source space; rank 128 is
still a low-rank, high-variance estimate. Do not conclude that RWKV has no useful J-lens
from these pilots.

The arithmetic cases also need a model-native benchmark check: their direct-answer controls
were not top-10 under the layer-37 logit lens, so poor hidden-intermediate recovery may be
task/model mismatch rather than a lens failure.

## Next Steps

1. Run `llama-interp-jlens-eval` on the rank-128 residual artifact and compare it with rank 64.
2. Build a small model-native continuation benchmark, then verify its final-layer/logit
   readout before using it to judge intermediate recovery.
3. Sweep `time.wkv` source layers on the same-token (`min_future=0`) task before increasing
   rank. The memory-editing evidence suggests the most informative WKV reads may be in a
   distributed late-layer band rather than layer 37 alone.
4. Only after a promising same-token layer is found, increase rank and compare against both
   raw WKV readout and matched-norm random controls.
