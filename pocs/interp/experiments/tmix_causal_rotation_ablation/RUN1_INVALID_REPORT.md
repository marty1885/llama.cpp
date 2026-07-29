# TMix Controlled-Completion Native-Unrotation: Run 1 Invalid Report

## Verdict

Run 1 is invalid and has no scientific classification.

The executable, synthetic self-tests, exact-model Vulkan verifier, calibration,
ROOT serialization, and post-write recomputation audit completed. However, 30 of
the 48 random-control interventions failed the registered post-TMix mean
reconstruction threshold. The classified decision rule was therefore suppressed.

The serialized diagnostic outcome is:

```text
no_detected_native_rotation_privilege
```

This is not a result. It cannot support the registered negative claim because the
random control family was not numerically matched as required.

## Frozen Run

```text
model: rwkv7-g1h_preview4743-13.3b-20260703-ctx8192-Q4_K_M.gguf
model SHA-256: 6a5c67dab16ad7c796db3f1390a883964ee678c3d6d302497552521b6223a204
backend: Vulkan, -ngl 99
layer: 45
value_id: 0
train carriers: 0 through 31
held-out carriers: 32 through 47
foreign controls per held-out carrier: 3
random controls per held-out carrier: 3
seed: 315971
```

The run reused the frozen controlled-value artifacts without regeneration:

```text
build/tmix-controlled-value-classified-prompts.txt
build/tmix-controlled-value-classified-manifest.json
build/tmix-controlled-value-classified-registration.json
```

Their hashes and the model hash matched the frozen registration.

## Implementation

The experiment-local executable is:

```text
pocs/interp/experiments/tmix_causal_rotation_ablation/
    rwkv_tmix_causal_rotation_ablation.cpp

llama-rwkv-tmix-causal-rotation-ablation
```

It provides the registered `--self-test`, `--calibration`, and `--classified`
modes. Model execution uses `llama_interp::runtime::prefill_tokens`; interventions
use `runtime::add_head` only at `rwkv.layer.45.time.out`; final logits are computed
from the captured final `resid.out` with `llama_interp_rwkv_final_readout`.

Each prompt prefix is decoded once and its native recurrent state is reused for the
native and perturbed final-read-token executions. All host-readable activations are
F32 production-graph snapshots.

The classified ROOT artifact contains:

| RNTuple | Rows |
| --- | ---: |
| `metadata` | 11 |
| `native_samples` | 48 |
| `interventions` | 112 |
| `carrier_metrics` | 16 |
| `statistical_tests` | 2 |
| `outcome_predicates` | 7 |
| `numerical_checks` | 4 |

The executable reopened these rows and independently recomputed intervention
counts, carrier means and deltas, exact sign-flip p-values, Holm adjustments, and
the diagnostic outcome before writing the JSON summary.

## Verification

Synthetic self-tests passed for:

1. Centering, normalization, endpoint construction, and native unrotation.
2. Tangent parallel transport and endpoint-angle matching.
3. Deterministic donor ordering and train/test isolation.
4. Deterministic Gaussian generation and redraw behavior.
5. Stable log-softmax, rank, top-1, and KL calculations.
6. Exact sign-flip p-values, Holm adjustment, and outcome branches.
7. Missing-row classification suppression.

The exact-model Vulkan production verifier reported:

```text
final_readout_max_abs_error=0
state_handoff_logit_max_abs_error=0
capture_handoff_max_abs_error=0
```

`git diff --check` also passed.

## Calibration

The engineering-only calibration passed and retained `calibration_only` status.

```text
native repeat minimum capture cosine:       1
native repeat maximum relative norm error:  0
alpha=0 minimum capture cosine:             0.9999999999999946
alpha=0 maximum relative norm error:        2.7055988563609712e-08
```

Endpoint distance was non-decreasing across the frozen alpha schedule:

| Alpha | Actual endpoint distance | Expected-token behavior loss |
| ---: | ---: | ---: |
| `0` | `0` | `0` |
| `0.25` | `5.0059294960` | `-0.0207789122` |
| `0.5` | `10.0079766863` | `-0.0181557819` |
| `1` | `19.9871186496` | `-0.0518761586` |

These calibration behavior values are engineering diagnostics and were not used to
change the layer, value, metric, controls, tolerances, or decision rule.

## Numerical Failure

The complete classified panel produced the following validity counts:

| Condition | Total | Passed | Failed | Maximum mean error |
| --- | ---: | ---: | ---: | ---: |
| native unrotation | 16 | 16 | 0 | `1.4935e-06` |
| same-value foreign | 48 | 48 | 0 | `2.45715e-06` |
| random tangent | 48 | 18 | 30 | `0.0143011` |

The registered absolute mean-error threshold is `0.002`. Every held-out carrier
had at least one failed random control.

The aggregate numerical checks were:

| Check | Observed maximum | Threshold | Pass |
| --- | ---: | ---: | --- |
| post-TMix mean error | `0.0143011` | `0.002` | no |
| post-TMix centered-radius relative error | `4.79292e-05` | `0.002` | yes |
| endpoint-angle error | `0.000211015` | `0.002` | yes |
| native-unrotation endpoint error | `5.6752e-05` | `0.002` | yes |

The failure is specific to the random controls' mean matching. Radius and angle
matching remained comfortably inside their registered thresholds.

## Root Cause

The registered random-control pseudocode is:

```python
g = deterministic_gaussian(width, seed, carrier_id, control_index)
h_random = unit(g - dot(g, v) * v)
v_target = cos(theta) * v + sin(theta) * h_random
```

Here `v` is in the mean-zero LayerNorm hyperplane. Projecting an unconstrained
Gaussian only off `v` does not remove the Gaussian's constant-vector component.
Consequently, `h_random` is tangent to the ambient unit sphere at `v`, but is not
generally tangent to the centered LayerNorm sphere. `v_target` can therefore have
nonzero mean, contradicting the separately registered requirement that the actual
endpoint preserve the native post-TMix mean within `0.002`.

The native and transported foreign directions already lie in the mean-zero
hyperplane, which explains why those families passed while the random family did
not.

This is an internal mismatch between the random-control construction and its
matched-control gate. It is not evidence for either behavioral hypothesis.

## Unclassified Diagnostics

For completeness, the invalid artifact contains the following mechanically
computed values:

| Control family | Mean carrier delta | Raw one-sided p | Holm-adjusted p |
| --- | ---: | ---: | ---: |
| foreign | `-0.02724867437` | `0.9927825928` | `1` |
| random | `-0.07036024860` | `0.9999237061` | `1` |

Both means are non-positive, which would select
`no_detected_native_rotation_privilege` if all numerical gates had passed. They did
not pass. These values must not be reported as a classified negative result or used
to claim that native directional movement lacks behavioral privilege.

## Permitted Conclusion

The maximum permitted statement from Run 1 is:

> The registered causal gate was not evaluable because the specified random
> tangent construction failed its matched post-TMix mean requirement in 30 of 48
> random interventions. No behavioral classification is permitted.

Run 1 does not establish whether native unrotation is behaviorally privileged. It
also does not establish a rotor code, memory content, pure recurrence isolation,
FFN decoding, or a map into output-embedding coordinates.

## Required Protocol Decision

A geometrically consistent random control would first project the Gaussian into the
mean-zero hyperplane and then remove its component along `v`:

```python
g_centered = g - mean(g)
h_random = unit(g_centered - dot(g_centered, v) * v)
```

That is a protocol amendment, not a valid post-hoc repair to Run 1. Because Run 1's
behavior diagnostics have already been observed, a rerun intended for scientific
classification should be discussed and registered explicitly, use a fresh random
seed, write new output paths, and retain this invalid artifact unchanged.

## Artifacts

```text
build/tmix-causal-rotation-ablation-calibration-run1.json
build/tmix-causal-rotation-ablation-classified-run1.root
build/tmix-causal-rotation-ablation-classified-run1.json
```

The classified ROOT SHA-256 is:

```text
329b569362b0624902d1ebe7e42eefe9f09dcbfa1801b0abde729433fc5b8602
```

`RESULTS.md` and the TMix investigation ledger remain unchanged because the
required numerical checks did not all pass.
