# TMix Destination-Subspace: Exploratory Pile Depth Results

## Status

This is an exploratory same-corpus depth scan, not a pre-registered
confirmatory depth experiment. Only layer 30 was specified as the confirmatory
layer in `../experiments/TMIX_DESTINATION_SUBSPACE_EXPERIMENT.md`; layers 15, 45, and 60 were
selected after layer-30 results and are explicitly marked non-confirmatory in
their JSON artifacts.

The results must not be described as evidence of a depth trend or a
depth-localized mechanism. They are a source of hypotheses for a future
pre-registered replication.

## Shared Configuration

| Field | Value |
| --- | --- |
| Model | `rwkv7-g1h-13.3b-20260710-ctx10240-Q4_K_M.gguf` |
| Backend | Vulkan, `-ngl 99` |
| Corpus | 48 prompts from `NeelNanda/pile-10k`, excluding `Wikipedia (en)` |
| Source families | Eight prompts each from `Pile-CC`, `OpenWebText2`, `StackExchange`, `PubMed Abstracts`, `Github`, and `USPTO Backgrounds` |
| Pile artifact SHA-256 | `a1a9475a8684ac8f1b17a36eccb2ec49c127edd7aae9beb2f240726972d93f31` |
| Corpus manifest | `pocs/interp/destination_subspace_pile10k_prompts.manifest.json` |
| Seed | `45678` |
| Train/test samples | 1,664 / 1,408 tokens from 26 / 22 prompts |
| Positions per prompt | 64 |
| Primary rank | 32 |
| Control replicates | 99 |
| Controls | Global empirical-write shuffle, within-prompt empirical-write shuffle, isotropic norm-matched writes, isotropic destination-basis controls |

`llama-rwkv-interp-verify` passed on the same model/backend with zero reported
final-readout, state-handoff, and capture-handoff errors. All four completed
runs had zero residual-identity error and maximum direction error below
`4e-08`.

## Measured Quantity

For each layer, train prompts define a rank-32 incoming-residual basis and a
destination basis from the component of the post-TMix centered normalized
residual orthogonal to that incoming basis.

Held-out destination routing is:

```python
D = mean(
    ||P_dest.T @ u_time||**2
    - ||P_dest.T @ u_in||**2
)
```

Positive `D` means post-TMix residual directions occupy the fitted destination
more than incoming residual directions. Native pairing is assessed by comparing
native `D` with the distribution from globally shuffled empirical `time.out`

## Results

| Layer | Status | Classification | `G_dest` | Isotropic destination p95 | Native `D` | Global-shuffle median `D` | Native minus global | `p_global` | Isotropic-write median `D` |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15 | exploratory | `exploratory_native_destination_routing` | 0.102685 | 0.006299 | 0.001146 | 0.000796 | +0.000350 | 0.01 | +0.000013 |
| 30 | confirmatory for this experiment | `uncoordinated_structured_addition` | 0.079503 | 0.009466 | 0.003303 | 0.003285 | +0.000018 | 0.34 | -0.000021 |
| 45 | exploratory | `exploratory_uncoordinated_structured_addition` | 0.074717 | 0.025606 | 0.010358 | 0.010916 | -0.000558 | 1.00 | -0.000708 |
| 60 | exploratory | `exploratory_uncoordinated_structured_addition` | 0.461553 | 0.016077 | 0.250181 | 0.279528 | -0.029347 | 1.00 | -0.003708 |

All four layers had a held-out-stable destination basis relative to isotropic
destination-basis controls. At every layer, global shuffled empirical writes
outperformed isotropic norm-matched writes (`p = 0.01`).

Native pairing exceeded global shuffled empirical writes only at exploratory
layer 15. It did not exceed shuffled writes at layer 30, 45, or 60.

## Artifacts

| Layer | ROOT artifact | JSON summary |
| --- | --- | --- |
| 15 | `build/tmix-destination-subspace-layer15-pile10k-exploratory.root` | `build/tmix-destination-subspace-layer15-pile10k-exploratory.json` |
| 30 | `build/tmix-destination-subspace-layer30-pile10k-run1.root` | `build/tmix-destination-subspace-layer30-pile10k-run1.json` |
| 45 | `build/tmix-destination-subspace-layer45-pile10k-exploratory.root` | `build/tmix-destination-subspace-layer45-pile10k-exploratory.json` |
| 60 | `build/tmix-destination-subspace-layer60-pile10k-exploratory.root` | `build/tmix-destination-subspace-layer60-pile10k-exploratory.json` |

## Permitted Interpretation

The only cross-layer exploratory observation is that structured empirical TMix
writes route residuals into a stable destination-like subspace more strongly
than isotropic norm-matched writes at all four sampled layers.

The scan does not establish that native token-local pairing specificity is
localized to layer 15, absent at later layers, monotonic with depth, or
generalizable across corpora. A future depth claim requires a new
pre-registered set of layers and independent held-out corpus.

No result establishes semantic content, a memory item, causal use, a rotor
code, logits, or an output-embedding mapping.
