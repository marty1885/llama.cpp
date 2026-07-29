# TMix Destination Base-Recovery Experiment

## Status

This is a discussed and admitted post-hoc exploratory reanalysis of an existing
layer-30 Pile capture. It is an implementation handoff. It has not yet been
implemented or run.

The source data have already been inspected in prior destination experiments, so
no result from this reanalysis is confirmatory. A positive result can justify a
future independent replication but cannot establish a rotation code by itself.

Read before implementation:

- `../../../../AGENTS.md`
- `../../../../SCIENTIFIC_METHOD_PROMPT.md`
- `../../logs/TMIX_INVESTIGATION.md`
- `../TMIX_DESTINATION_SUBSPACE_EXPERIMENT.md`
- `../../logs/TMIX_DESTINATION_SUBSPACE_RESULTS.md`
- `../../logs/TMIX_DESTINATION_SUBSPACE_PILE_EXPLORATORY_DEPTH_RESULTS.md`

## Current Scientific Gate

The eventual objective is to determine what RWKV memory contributes to the
residual stream and whether that contribution can be decoded before the FFN turns
it into the model's learned response.

This experiment does not attempt that objective. It tests one necessary structural
condition of the current direction:

```text
If TMix uses ADD + LayerNorm to place added information in a complementary
destination, removing that destination from a held-out post-TMix direction should
recover the incoming normalized residual better than matched structured-addition
controls.
```

If this condition fails, the stable destination may simply be part of a different
holistic residual representation. In that case, the claim that it is a separable
added channel is unsupported.

## Scientific Question

At layer 30, does removing one native train-fitted destination subspace from
held-out post-TMix normalized residuals move them back toward their paired incoming
normalized residuals more, and more efficiently per unit angular ablation, than
destinations fitted from shuffled empirical or isotropic additions?

## Observation Already Established

The layer-30 destination experiment established on expanded text8 and
non-Wikipedia Pile captures that:

```text
a rank-32 destination fitted from train post-TMix directions is held-out stable
relative to isotropic destination-basis controls

empirical TMix writes enter that destination more than isotropic norm-matched
writes

native token-local pairing specificity did not replicate across the two corpora
```

This is evidence for stable structured addition. It does not show that the
destination is separable from a preserved incoming representation.

## Competing Hypotheses

### H1: Separable Destination Component

```text
The post-TMix normalized residual retains an identifiable incoming component and
adds a component in a stable complementary destination.

A destination fitted on train prompts therefore removes a component that points
held-out post-TMix directions away from their paired incoming directions.
```

### H2: Ordinary Structured Addition

```text
TMix writes are structured and produce a stable post-TMix subspace, but that
subspace is not a separable added channel.

Removing it does not recover the paired incoming direction better than equivalent
destinations produced by shuffled empirical additions.
```

### H0: No Detected Held-Out Base Recovery

```text
Removing the native destination does not, on average, move held-out post-TMix
directions toward their paired incoming directions.
```

## Shared Predictions And Tautologies

Both H1 and H2 permit:

- a stable low-rank post-TMix destination;
- greater destination occupancy after TMix;
- empirical writes outperforming isotropic writes;
- an FFN response to the resulting direction.

Those observations do not distinguish the hypotheses.

For each token, `u_in` and `u_time` define a token-local rotor and tangent. Removing
that exact token-local tangent can recover `u_in` by construction. Do not use a
token-local rotor, tangent, plane, or test-fitted direction as evidence in this
experiment.

The non-guaranteed prediction is that one destination fitted only on train prompts
recovers paired incoming directions on held-out prompts better than complete
structured-addition control pipelines.

## Scope

Use exactly:

```python
layer = 30
rank = 32
control_repeats = 99
seed = 45678
```

Input artifacts:

```text
build/tmix-destination-subspace-layer30-pile10k-run1.root
pocs/interp/destination_subspace_pile10k_prompts.manifest.json
```

The ROOT artifact contains 1,664 train and 1,408 test tokens from 26 and 22
prompts, respectively. Use its existing split without modification.

Do not add:

- another layer or rank;
- another corpus;
- model execution;
- FFN activations;
- logits or vocabulary output;
- semantic labels or probes;
- recurrent-state interventions;
- memory-content claims;
- a new destination definition selected after viewing results.

## Stored Per-Token Geometry

Read these fields from the `samples` RNTuple:

```text
prompt_id
split
position
token_id
resid_in
time_out
resid_time
u_in
u_time
write_norm
```

Recompute centered unit directions from the raw vectors and verify them against the
stored `u_in` and `u_time`. Use double-precision accumulation.

```python
def center(v):
    return v - mean(v)

def unit(v):
    return v / norm(v)

u_in_recomputed = unit(center(resid_in))
u_time_recomputed = unit(center(resid_time))
```

## Native Basis Construction

Fit all bases from train rows only. Do not subtract an across-token dataset mean.

```python
P_in = top_right_singular_vectors(U_in_train, rank=32)

def remove_input_basis(v):
    return v - P_in @ (P_in.T @ v)

Z_native_train = [
    unit(remove_input_basis(u_time))
    for u_time in U_time_train
    if norm(remove_input_basis(u_time)) > 1e-12
]

P_dest_native = top_right_singular_vectors(Z_native_train, rank=32)
```

Use a thin SVD, Gram method, or equivalently validated decomposition. Do not form a
dense width-by-width covariance.

## Base-Recovery Metrics

For a held-out incoming direction `u`, post-TMix direction `v`, and destination
basis `P`:

```python
removed = P @ (P.T @ v)
v_recovered = unit(v - removed)

angle_before = acos(clamp(dot(u, v), -1, 1))
angle_after = acos(clamp(dot(u, v_recovered), -1, 1))
ablation_angle = acos(clamp(dot(v, v_recovered), -1, 1))

cosine_gain = dot(u, v_recovered) - dot(u, v)
absolute_angle_recovery = angle_before - angle_after

angular_recovery_fraction = (
    absolute_angle_recovery / angle_before
)

recovery_efficiency = (
    absolute_angle_recovery / ablation_angle
)
```

`recovery_efficiency` is bounded approximately by `[-1, 1]` from the triangle
inequality and asks whether the ablation moves toward `u_in`, rather than merely
whether it makes a large change. It prevents a basis from winning solely because it
removes more endpoint energy.

Exclude only the fractional metric for a token when:

```python
angle_before < 1e-6
```

Exclude only the efficiency metric when:

```python
ablation_angle < 1e-6
```

Retain the token for all other valid metrics. Serialize every exclusion count.

Also report:

```python
removed_energy = norm(P.T @ v) ** 2
cosine_before = dot(u, v)
cosine_after = dot(u, v_recovered)
```

## Prompt-Balanced Aggregation

Tokens within a prompt are dependent. Do not treat tokens as independent samples.

For every metric:

```python
prompt_metric[prompt_id] = mean(valid_token_metrics_in_prompt)
aggregate_metric = mean(prompt_metric.values())
```

The two co-primary metrics are:

```text
prompt-balanced mean angular_recovery_fraction
prompt-balanced mean recovery_efficiency
```

Absolute angle recovery, cosine gain, removed energy, and unbalanced token means
are required diagnostics and cannot determine the outcome.

## Complete Control Pipelines

Use deterministic distinct sub-seeds for all 99 replicates. Never permute across
train/test. Every control must fit its own destination from controlled train
endpoints and evaluate on independently controlled test endpoints.

Use the same native `P_in` for every control so that only the addition and fitted
destination differ.

### Control A: Global Empirical-Write Shuffle

Within train and test independently, globally permute complete empirical centered
`time_out` vectors:

```python
v_shuffle[i] = unit(
    center(resid_in[i]) + center(time_out[permutation[i]])
)
```

For each replicate:

1. Fit `P_dest_shuffle` from shuffled train endpoints after residualizing against
   native `P_in`.
2. Evaluate base recovery on the matched shuffled test endpoints relative to each
   test token's original `u_in`.
3. Also apply `P_dest_shuffle` to native test endpoints as a basis-specificity
   diagnostic.

The complete shuffled pipeline is the primary ordinary-structured-addition
control.

### Control B: Within-Prompt Empirical-Write Shuffle

Repeat Control A while permuting writes only among positions in the same prompt.
This preserves prompt-level trajectory and topic statistics while breaking
token-local pairing.

This control is required and diagnostic. It does not determine the primary
classification.

### Control C: Isotropic Norm-Matched Writes

For each token, draw a Gaussian vector, center it, normalize it, and scale it to
that token's native centered write norm:

```python
g = unit(center(random_gaussian(width)))
w_isotropic = write_norm[i] * g
v_isotropic[i] = unit(center(resid_in[i]) + w_isotropic)
```

Fit `P_dest_isotropic` from each isotropic train replicate and evaluate its matched
independent isotropic test replicate. Also apply it to native test endpoints as a
basis-specificity diagnostic.

### Control D: Random Input-Complement Basis

Generate a random rank-32 orthonormal basis, residualize it against `P_in`, and
re-orthonormalize it. Apply it to native test endpoints.

This is a basis-orientation diagnostic. Because its removed energy need not match
the fitted destination, it cannot determine classification. Report both recovery
metrics together with removed energy.

## Empirical Statistics

For every control family and co-primary metric, serialize all 99 aggregate values
and report:

```text
count
mean
standard deviation
minimum
5th percentile
median
95th percentile
maximum
```

For a native metric `M_native` where larger is more supportive:

```python
p_control = (
    1 + count(M_control >= M_native)
) / (1 + control_repeats)
```

Do not compute token-level p-values.

## Mechanical Outcome Rules

Classification uses only the native result and the complete global-shuffle and
isotropic pipelines. Both co-primary metrics must agree.

### Necessary Condition Supported

Classify as:

```text
exploratory_base_recovery_supported
```

only if all conditions pass:

```python
native_angular_recovery_fraction > 0
native_recovery_efficiency > 0

native_angular_recovery_fraction > global_shuffle_p95_fraction
p_global_fraction <= 0.05

native_recovery_efficiency > global_shuffle_p95_efficiency
p_global_efficiency <= 0.05

native_angular_recovery_fraction > isotropic_p95_fraction
p_isotropic_fraction <= 0.05

native_recovery_efficiency > isotropic_p95_efficiency
p_isotropic_efficiency <= 0.05
```

### No Detected Base Recovery

Classify as:

```text
exploratory_no_detected_base_recovery
```

only if:

```python
native_angular_recovery_fraction <= 0
```

### Structured Addition Not Distinguished

Classify as:

```text
exploratory_not_distinguished_from_structured_addition
```

if native recovery is positive but either co-primary metric fails the complete
global-shuffle comparison.

### Ambiguous

Every other valid result is:

```text
exploratory_ambiguous
```

The executable must evaluate these rules mechanically and serialize every operand
and Boolean predicate.

## Positive, Negative, And Ambiguous Claims

### Positive Maximum Claim

```text
On the previously inspected layer-30 Pile capture, removing the native train-fitted
destination moved held-out post-TMix directions toward their paired incoming
directions more, and more efficiently per unit ablation angle, than destinations
from globally shuffled empirical and isotropic additions. This supports a
separable-destination necessary condition and nominates it for independent
replication.
```

### Structured-Addition Maximum Claim

```text
The native destination moved held-out endpoints toward incoming directions, but
the effect was not distinguished from ordinary structured empirical addition.
```

### Negative Maximum Claim

```text
No positive held-out recovery of incoming direction was detected after removing
the native destination at this layer, rank, and corpus.
```

### Claims Forbidden For Every Outcome

No outcome establishes:

- preservation or recovery of semantic meaning;
- that the destination contains memory;
- that RWKV uses a rotation code;
- that the FFN reads the destination;
- causal use;
- a depth trend;
- a valid logit lens or output-embedding map.

## Numerical Validity And Fail-Closed Criteria

Require:

```text
ROOT schema contains every required field
layer metadata is 30
train/test token and prompt counts match the source JSON
no prompt occurs in both splits
all vectors have one common width
all recomputed directions are finite
max norm error of every unit vector <= 1e-5
max stored-versus-recomputed direction L2 error <= 1e-5
P_in orthogonality error <= 1e-5
every destination-basis orthogonality error <= 1e-5
max_abs(P_in.T @ P_dest) <= 1e-5 for every destination
all 99 replicates use distinct deterministic sub-seeds
every permutation remains inside its declared split and grouping
all aggregate and per-prompt metrics are finite
all control values are serialized
```

Recompute the original native rank-32 `G_dest`, `E_input`, `E_time`, and `D` from
the source ROOT rows and require absolute agreement with the source JSON within:

```python
metric_tolerance = 1e-6
```

Any failed criterion sets status `invalid`, suppresses classification, and records
the failure.

## Implementation Handoff

Suggested source:

```text
pocs/interp/experiments/tmix_destination_base_recovery/rwkv_tmix_destination_base_recovery.cpp
```

Suggested target:

```text
llama-rwkv-tmix-destination-base-recovery
```

Required CLI:

```text
--input-root FILE.root
--source-json FILE.json
--manifest FILE.json
--output-root FILE.root
--output-json FILE.json
--seed 45678
```

Reject unknown arguments. Reject an existing output path unless `--overwrite` is
provided and recorded as a protocol deviation.

This executable reads existing captures only. It must not link the model runtime or
execute a GGML graph. Use C++ and ROOT RNTuples; do not add a Python pipeline.

## Output Contract

Write a ROOT artifact containing:

```text
metadata
per_prompt_metrics
aggregate_metrics
control_values
```

Every control replicate row must contain:

```text
control_family
evaluation_endpoint_family
replicate
sub_seed
metric_name
metric_value
```

The compact JSON must include:

```text
schema version
status and mechanical classification
post-hoc exploratory flag
all input paths and hashes
fixed layer, rank, seed, and repeat count
sample and exclusion counts
all numerical checks
native co-primary and diagnostic metrics
complete summaries for every control family
all empirical p-values
every outcome-rule operand and predicate
ROOT artifact path
```

Do not omit unfavorable metrics.

## Synthetic Self-Tests

Before reading the real artifact, test:

1. A synthetic factorized case where a fixed orthogonal destination is added to
   varied incoming directions. Native destination removal must recover the inputs.
2. A random holistic-direction case where destination removal must not show
   privileged recovery.
3. Zero and near-zero pre/post angles for metric exclusions.
4. Recovery efficiency for movement directly toward, orthogonal to, and directly
   away from the incoming direction.
5. Global and within-prompt permutation isolation.
6. Prompt-balanced aggregation with unequal token counts.
7. Empirical p-value and classification truth tables.

## Verification

Build only the analysis target, run it on the fixed inputs, validate both output
schemas and row counts, independently recompute classification from serialized
control values, and finish with:

```bash
git diff --check
```

Do not update the investigation ledger or write a result report until the output
has passed all numerical and serialization checks.
