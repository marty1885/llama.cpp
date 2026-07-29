# TMix Destination-Subspace Results

## Scope

This report records layer-30 destination-subspace runs for the RWKV-7 13.3B
Q4_K_M model. It distinguishes a stable post-TMix destination from evidence
that a particular native `resid.in` / `time.out` pairing is special.

The production operation is:

```python
resid_time = resid_in + time_out
u_in = unit(center(resid_in))
u_time = unit(center(resid_time))
```

The train-only destination basis is fitted from the component of `u_time`
orthogonal to a train-only rank-32 incoming-residual basis. Held-out routing is:

```python
D = mean(||P_dest.T @ u_time||^2 - ||P_dest.T @ u_in||^2)
```

The test does not identify semantic content, a memory item, causal use, a
rotor code, logits, or output-embedding coordinates.

## Shared Configuration

| Field | Value |
| --- | --- |
| Model | `rwkv7-g1h-13.3b-20260710-ctx10240-Q4_K_M.gguf` |
| Backend | Vulkan, `-ngl 99` |
| Layer | 30 |
| Primary rank | 32 |
| Controls | 99 global pair shuffles, within-prompt shuffles, isotropic norm-matched writes, and isotropic destination-basis controls |
| Production validation | `llama-rwkv-interp-verify` passed with zero reported final-readout, state-handoff, and capture-handoff errors |
| Numerics | F32 production snapshots; double-precision statistics; OpenBLAS/LAPACK thin Gram eigensystem |

All reported runs had zero excluded tokens, zero residual-identity error, and
direction errors below `4e-08`.

## Expanded Wikipedia-Derived Corpus

Corpus: `pocs/interp/destination_subspace_expanded_prompts.txt`

The corpus has 48 newly frozen offset `text8` excerpts. `text8` is
Wikipedia-derived. Its frozen manifest is
`pocs/interp/destination_subspace_expanded_prompts.manifest.json`.

| Field | Value |
| --- | --- |
| Artifact | `build/tmix-destination-subspace-layer30-expanded-run3.root` |
| JSON | `build/tmix-destination-subspace-layer30-expanded-run3.json` |
| Train/test samples | 1,408 / 1,664 |
| Train/test prompts | 22 / 26 |
| Classification | `native_destination_routing` |
| Native destination stability | `G_dest = 0.082044` |
| Isotropic destination stability p95 | `0.008835` |
| Destination stability p-value | `0.01` |
| Native routing gain | `D_native = 0.003721` |
| Global-shuffle median gain | `0.003382` |
| Native advantage over global shuffle | `0.000339`, `p = 0.01` |
| Isotropic-write median gain | `-0.000013` |
| Native advantage over isotropic writes | `0.003735`, `p = 0.01` |

Within this corpus, the native pairing passed the pre-registered routing rule:
the destination generalized, native gain was positive, and native pairing
outperformed both global empirical-write shuffles and isotropic norm-matched
writes.

## Non-Wikipedia Pile Replication

Corpus: `pocs/interp/destination_subspace_pile10k_prompts.txt`

This corpus is a frozen, source-balanced subset of the 33.3 MB Hugging Face
artifact `NeelNanda/pile-10k`, downloaded with SHA-256:

```text
a1a9475a8684ac8f1b17a36eccb2ec49c127edd7aae9beb2f240726972d93f31
```

It contains eight prompts each from `Pile-CC`, `OpenWebText2`,
`StackExchange`, `PubMed Abstracts`, `Github`, and `USPTO Backgrounds`.
`Wikipedia (en)` was excluded. Source-row IDs, component labels, prompt hashes,
and split assignments are recorded in
`pocs/interp/destination_subspace_pile10k_prompts.manifest.json`.

| Field | Value |
| --- | --- |
| Artifact | `build/tmix-destination-subspace-layer30-pile10k-run1.root` |
| JSON | `build/tmix-destination-subspace-layer30-pile10k-run1.json` |
| Train/test samples | 1,664 / 1,408 |
| Train/test prompts | 26 / 22 |
| Classification | `uncoordinated_structured_addition` |
| Native destination stability | `G_dest = 0.079503` |
| Isotropic destination stability p95 | `0.009466` |
| Destination stability p-value | `0.01` |
| Native routing gain | `D_native = 0.003303` |
| Global-shuffle median gain | `0.003285` |
| Native advantage over global shuffle | `0.000018`, `p = 0.34` |
| Isotropic-write median gain | `-0.000021` |
| Native advantage over isotropic writes | `0.003324`, `p = 0.01` |
| Global-shuffle advantage over isotropic writes | `0.003306`, `p = 0.01` |

Within this corpus, the destination generalized and structured empirical writes
outperformed isotropic writes. However, native pairing did not outperform the
global empirical-write shuffle. The H2-shaped outcome is therefore
`uncoordinated_structured_addition`.

## Cross-Corpus Result

The result that replicates across the Wikipedia-derived and non-Wikipedia Pile
corpora is:

```text
Layer-30 TMix additions route normalized residuals into a held-out-stable
destination subspace, and empirical write directions outperform isotropic
norm-matched write directions.
```

Native token-local pairing specificity did not replicate:

```text
Wikipedia-derived text8 corpus: native pairing exceeded global shuffle, p = 0.01.
Non-Wikipedia Pile corpus: native pairing did not exceed global shuffle, p = 0.34.
```

Therefore, the supported cross-corpus claim is limited to stable destination
structure associated with empirical write directions. It does not establish
that native `resid.in` / `time.out` pairing is generally special.

## Limitations

- These are single-layer, rank-32 results for one model and backend.
- The Pile corpus is a small 48-prompt balanced subset of `pile-10k`, not a
  random estimate over the full Pile.
- The current executable performs repeated-capture validation during collection,
  but its JSON schema does not yet serialize the repeatability metrics.
- No result supports semantic, memory-content, causal-intervention, rotor-code,
  logit, or output-embedding claims.
