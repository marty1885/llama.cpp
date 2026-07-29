# TMix Destination Base-Recovery: Run4 Review Response

## Status

`run4` completed as a new artifact and did not overwrite the invalidated prior
runs:

```text
build/tmix-destination-base-recovery-layer30-pile10k-run4.root
build/tmix-destination-base-recovery-layer30-pile10k-run4.json
build/tmix-destination-base-recovery-layer30-pile10k-run4.log
```

The executable reported:

```text
status:         valid
classification: exploratory_no_detected_base_recovery
```

However, the reviewer’s full output-contract audit is not yet complete. In
particular, the compact JSON does not yet contain every required summary,
predicate, hash, and numerical-check record. Therefore this document is an
implementation response and run record, not authorization to enter the result
into `TMIX_INVESTIGATION.md`.

## Run4 Numerical Result

| Metric | Value |
| --- | ---: |
| Prompt-balanced angular-recovery fraction | -0.290191070987 |
| Prompt-balanced recovery efficiency | -0.262182449437 |
| Absolute angle recovery | -0.029662186534 radians |
| Cosine gain | -0.003819491726 |
| Removed energy | 0.012481624199 |

The native co-primary metrics are negative. If the remaining schema/audit work
passes, the registered mechanical rule assigns
`exploratory_no_detected_base_recovery`.

## Run4 Numerical Logging

The tee'd run log records these checks before the control sweep:

```text
E_input reproduction error: 1.6104157e-08
E_time reproduction error:  2.51579426e-08
D reproduction error:       9.05375566e-09
P_in orthogonality error:    7.43849426e-15
P_dest orthogonality error:  5.55111512e-15
P_in/P_dest max overlap:     4.48600409e-08
raw stored-direction check:  pass
```

All of these are within the configured `1e-5` direction/basis tolerance and
the `1e-6` source-metric tolerance. The full control sweep completed all 396
fits in 309.9 seconds using four shared-memory workers with one BLAS thread per
worker; progress and final status are in the run log.

## Response To `IMPLEMENTATION_REVIEW.md`

| Review item | Run4 response | Disposition |
| --- | --- | --- |
| ROOT datasets missing / writer lifetime | Writers are now scoped before file close. `rootls` on the smoke artifact reports `metadata`, `per_prompt_metrics`, `aggregate_metrics`, and `control_values`. | Partially addressed: final run4 row-count/field audit still required. |
| Global-shuffle key bug | Lookups now use `global_shuffle_*` control keys; run4 reports global empirical p-values of `0.01`, not the prior mechanically forced `1.0`. | Addressed in code; independent recomputation still required. |
| Missing native-basis and random-basis diagnostics | Each fitted global/within-prompt/isotropic basis is evaluated on matched and native endpoints; random input-complement rows are serialized. | Addressed in code; audit row counts still required. |
| Incomplete JSON | Added configuration and input paths, but full hashes, all summaries, all predicates, and complete numerical-check serialization remain absent. | **Open blocking item.** |
| Incomplete numerical serialization / source `G_dest` | Run log records several checks and source `E_input`, `E_time`, `D` reproduction. Source `G_dest` reproduction and machine-readable complete checks remain absent. | **Open blocking item.** |
| Incomplete self-tests | Factorized recovery, zero-angle exclusion, and empirical-p test run. Holistic, pairing-isolation, unequal-prompt, and truth-table tests remain incomplete. | **Open blocking item.** |
| Target links model runtime | The base-recovery target now links only OpenBLAS, ROOT Matrix, ROOT RNTuple, and OpenMP; it no longer includes the collector or links `llama`/`llama-common`. | Addressed. |
| Premature `RESULTS.md` | `RESULTS.md` was marked **Invalidated Draft**. | Addressed. |

## Required Before Scientific Acceptance

Do not update the investigation ledger or replace the invalidated result draft
yet. The remaining work is to complete the machine-readable output contract,
add the registered self-tests, reproduce source `G_dest`, and independently
recompute control summaries, empirical p-values, and classification from the
serialized run4 ROOT rows. A subsequent corrected artifact must be used if any
of those changes alter the serialized result.

## Permitted Current Statement

`run4` is a technically improved, exact-solver run whose native scalar remains
negative. It is consistent with no detected held-out base recovery at the fixed
layer, rank, and Pile capture. It is not yet a fully auditable scientific result
under the experiment handoff contract.
