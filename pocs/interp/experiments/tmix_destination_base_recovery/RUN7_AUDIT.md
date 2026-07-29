# TMix Destination Base-Recovery: Run 7 Audit and Result

## Verdict

Run 7 is the corrected fixed-scope artifact for the registered layer-30,
rank-32 Pile reanalysis. The executable's fail-fast self-test gate, numerical
reproduction checks, serialized-output checks, and post-write ROOT/JSON audit
passed.

The registered mechanical classification is:

```text
exploratory_no_detected_base_recovery
```

This is a negative result for the tested necessary condition only. It does not
show that TMix is unstructured, that there is no destination representation, or
that RWKV does not admit another geometric description.

## Artifacts

```text
build/tmix-destination-base-recovery-layer30-pile10k-run7.root
build/tmix-destination-base-recovery-layer30-pile10k-run7.json
build/tmix-destination-base-recovery-layer30-pile10k-run7.log
```

The ROOT artifact contains the required RNTuples:

```text
metadata:           1 row
per_prompt_metrics: 22 rows
aggregate_metrics:  5 rows
control_values:     3,465 rows
```

The post-write gate checked every expected family/endpoint shape, finite control
values, 99 distinct deterministic sub-seeds per control family, control-derived
p-values, and the required JSON sections.

## Numerical Validity

All registered source reproductions were below the `1e-6` tolerance:

| Check | Observed error |
| --- | ---: |
| `G_dest` | `3.13393685e-7` |
| `E_input` | `1.86331954e-8` |
| `E_time` | `2.85114842e-8` |
| `D` | `9.87825877e-9` |

Basis checks also passed: `P_in` orthogonality `5.55e-15`, destination
orthogonality `6.66e-15`, and input/destination overlap `5.37e-8`, each below
`1e-5`.

## Held-Out Native Result

Prompt-balanced held-out native metrics:

| Metric | Value |
| --- | ---: |
| angular-recovery fraction | `-0.290191003138` |
| recovery efficiency | `-0.262182409117` |
| absolute angle recovery | `-0.029662188836` |
| cosine gain | `-0.003819492382` |
| removed energy | `0.012481627553` |

The primary fraction is negative, so destination removal moved held-out endpoints
away from their paired incoming directions on average. This directly selects the
registered negative classification; favorable comparisons with some negative
control distributions do not override that rule.

## Control Context

The native fraction and efficiency are numerically above the global-shuffle and
isotropic 95th percentiles, but remain negative in absolute terms:

| Pipeline / metric | 95th percentile | Empirical upper-tail p |
| --- | ---: | ---: |
| global shuffle / fraction | `-0.335778016515` | `0.01` |
| global shuffle / efficiency | `-0.278637385975` | `0.01` |
| isotropic / fraction | `-0.290768633296` | `0.02` |
| isotropic / efficiency | `-0.336011313336` | `0.01` |

The result therefore does **not** support the simple separable-destination
necessary condition as pre-registered. The permitted claim is:

> No positive held-out recovery of incoming direction was detected after removing
> the native destination at this layer, rank, and corpus.

## Next Step

Do not expand the analysis to more layers, ranks, corpora, logits, or semantic
claims in response to this negative outcome. The appropriate follow-up is to
record this bounded negative result in the TMIX investigation ledger and decide,
from the ledger's competing hypotheses, whether a new independently admitted
experiment can discriminate a holistic residual representation from ordinary
structured addition.
