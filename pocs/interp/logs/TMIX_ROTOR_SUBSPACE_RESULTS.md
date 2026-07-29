# TMix Rotor-Subspace Run Results

## Status

This report records four completed single-layer runs of the proposed
rotor-subspace gate. It is a results summary, not evidence that RWKV memory is
encoded in rotor coordinates. No run permits a semantic, memory-content,
causal-use, logit, or output-embedding claim.

The layer-15 run was explicitly exploratory because it was selected after the
other listed results. The four outcomes must not be fitted or described as a
depth trend.

## Shared Run Configuration

| Field | Value |
| --- | --- |
| Model | `rwkv7-g1h-13.3b-20260710-ctx10240-Q4_K_M.gguf` |
| Backend | Vulkan, `-ngl 99` |
| Capture validation | `llama-rwkv-interp-verify` passed: final readout, state handoff, and capture handoff maximum errors were all `0` |
| Corpus | `pocs/interp/rotor_subspace_prompts.txt` |
| Corpus construction | 64 evenly spaced, non-overlapping 256-word excerpts from local `text8` |
| Corpus manifest | `pocs/interp/rotor_subspace_prompts.manifest.json` |
| Source/corpus FNV-1a-64 | `3342511896256184474` / `4482027086835586803` |
| Seed and split | `12345`, FNV-1a-64 prompt-ID split |
| Collected positions | 512 total: 64 positions from each of 8 fresh-state prompts |
| Train/test data | 256 tokens from 4 prompts / 256 tokens from 4 prompts |
| Primary rank | 32 |
| Controls | 99 global-pair-shuffle and isotropic-control replicates; within-prompt shuffle recorded diagnostically |

The frozen corpus has 19,162 target-model tokens in total. These runs use the
minimum pre-registered sample count, not a high-powered corpus-scale estimate.

## Numerical Validation

All four runs had zero degenerate tangents and maximum residual identity error
of zero. The reported maximum geometry error combines unit, tangent
orthogonality, angle-equality, and endpoint-reconstruction checks; the required
threshold was `1e-5`.

| Layer | Status | Maximum geometry error | ROOT artifact | JSON summary |
| --- | --- | ---: | --- | --- |
| 15 | valid, exploratory | `9.02e-07` | `build/tmix-rotor-subspace-layer15-text8-exploratory-run1.root` | `build/tmix-rotor-subspace-layer15-text8-exploratory-run1.json` |
| 30 | valid | `7.87e-07` | `build/tmix-rotor-subspace-layer30-text8-run1.root` | `build/tmix-rotor-subspace-layer30-text8-run1.json` |
| 45 | valid | `3.35e-07` | `build/tmix-rotor-subspace-layer45-text8-run1.root` | `build/tmix-rotor-subspace-layer45-text8-run1.json` |
| 60 | valid | `9.99e-08` | `build/tmix-rotor-subspace-layer60-text8-run2.root` | `build/tmix-rotor-subspace-layer60-text8-run2.json` |

Each listed ROOT file was checked to contain `samples`, `metadata`, and
`subspace_metrics` RNTuples. Each JSON file was parsed and its status, sample
counts, residual identity bound, and geometry bound were checked.

## Primary Rank-32 Outcomes

`G_W` is held-out raw centered `time.out` self-generalization. `G_T` is
held-out native tangent-direction self-generalization. The p95 values are the
matched finite-sample isotropic controls. `p_pairing` asks whether the native
tangent basis retains its held-out explanatory power after global residual/write
pairings are broken. `p_shuffle_self` asks whether shuffled pairings learn a
tangent subspace at least as coherent as the native one.

| Layer | Classification | `G_W` | Ambient `W` p95 | `G_T` | Isotropic tangent p95 | Pairing gap | `p_pairing` | `p_shuffle_self` | Complementarity | `p_complementarity` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15 | ambiguous, exploratory | 0.013613 | 0.007992 | 0.016093 | 0.008042 | -0.000101 | 0.98 | 0.02 | -0.034155 | 0.23 |
| 30 | global additive structure consistent | 0.008995 | 0.007992 | 0.013066 | 0.008040 | -0.000406 | 1.00 | 0.50 | -0.010729 | 0.71 |
| 45 | ambiguous | 0.010863 | 0.007992 | 0.011502 | 0.008039 | -0.000092 | 0.97 | 0.05 | -0.003760 | 0.01 |
| 60 | no detected low-rank structure | 0.000944 | 0.007992 | 0.001055 | 0.008039 | -0.000073 | 1.00 | 1.00 | -0.008203 | 0.99 |

## Permitted Interpretation

- Layer 30: at this layer, rank, corpus, and sample size, the result is
  consistent with stable ambient additive organization. Native pairing did not
  establish residual-relative rotor organization.
- Layer 45: raw-update and tangent self-generalization exceeded isotropic
  controls, but native pairing specificity and positive complementarity were
  absent. The hypotheses were not distinguished.
- Layer 60: no stable low-rank raw-additive or tangent organization was detected
  at this layer, rank, corpus, and sample size. This does not prove absence of
  structure.
- Layer 15: the same combination of above-control self-generalization with
  absent pairing specificity was observed, but the run is exploratory and the
  hypotheses were not distinguished.

These observations do not establish a depth trend. In particular, the four
layers were not selected as a pre-registered sweep, layer 15 was selected after
earlier results, and all runs use the same minimum-sized source-derived corpus.

## Implementation Limits

The runs enforce production-graph snapshots, residual identities, per-token
geometry checks, prompt-level splits, rank-32 control classification, and ROOT
artifact existence. The current executable does not yet implement every field
and validation item in `../experiments/TMIX_ROTOR_SUBSPACE_EXPERIMENT.md`, including the
specified 16-position capture-repeatability check, full diagnostic-rank output,
complete control-replicate serialization in the JSON review artifact, and all
descriptive/channel-mix metrics. Treat these results as preliminary run records
until those requirements are implemented and the runs are repeated.

## Non-Claims

Do not infer any of the following from this report:

- A semantic meaning for any TMix direction or subspace.
- A memory-item representation or a memory-content decoder.
- Causal use of a rotor/tangent representation.
- A mapping into final residual or output-embedding coordinates.
- A monotonic depth trend.
