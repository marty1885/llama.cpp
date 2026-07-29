# TMix Destination-Subspace Experiment Specification

## Status

This document specifies a proposed confirmatory single-layer experiment. It is an
implementation handoff, not a completed result.

Implement this as a new experiment. Do not silently change the question or result
schema of `TMIX_ROTOR_SUBSPACE_EXPERIMENT.md`.

Read before implementation:

- `AGENTS.md`
- `SCIENTIFIC_METHOD_PROMPT.md`
- `pocs/interp/logs/TMIX_INVESTIGATION.md`
- `pocs/interp/experiments/TMIX_ROTOR_SUBSPACE_EXPERIMENT.md`
- `pocs/interp/logs/TMIX_ROTOR_SUBSPACE_RESULTS.md`
- `pocs/interp/README.md`

## Scientific Question

At RWKV layer 30, does the native TMix addition move the centered normalized
residual into a held-out-stable destination subspace that incoming residuals occupy
less, and does this movement depend on the native `resid.in` / `time.out` pairing?

This tests a narrower hypothesis than a global rotor code:

```text
TMix may use structured addition plus LayerNorm to move the normalized residual
into a complementary destination channel. The information-bearing object need not
be the rotor, tangent vector, plane, or angle itself.
```

The production operation remains:

```python
resid_time = resid_in + time_out
ffn_input = layer_norm(resid_time)
```

The experiment does not test whether RWKV literally executes a rotor instead of
addition and LayerNorm. These are equivalent descriptions of the same composite
operation on the centered pre-affine LayerNorm sphere.

## Observation Already Established

The preliminary rotor-subspace runs reported in
`../logs/TMIX_ROTOR_SUBSPACE_RESULTS.md` found at layer 30 and primary rank 32:

```text
raw centered time.out self-generalization exceeded its isotropic control
native tangent self-generalization exceeded its isotropic control
native resid.in/time.out pairing did not improve tangent-subspace coherence
complementarity between incoming residuals and native tangents was negative
```

The mechanically assigned outcome was `global additive structure consistent`.
Those runs were preliminary because not every validation and serialization item in
the original specification had been implemented.

The present experiment asks whether the structured addition nevertheless moves
post-TMix normalized residuals into a destination channel that is not visible in a
global raw-tangent comparison.

## Competing Hypotheses

### H1: Native Destination Routing

```text
Native resid.in/time.out pairings move normalized residuals into a held-out-stable
destination subspace.

Incoming normalized residuals occupy that destination subspace less.

Breaking only the native residual/write pairing reduces movement into the
destination subspace.
```

### H2: Uncoordinated Structured Addition

```text
time.out is structured and can move residuals into a stable post-TMix subspace,
but the native residual/write pairing is not special.

Pair-shuffled empirical writes route into the candidate destination as strongly as
native writes, while isotropic norm-matched writes do not.
```

### H0: No Stable Destination Channel At This Granularity

```text
The candidate post-TMix destination does not generalize beyond finite-sample
isotropic controls at the pre-registered layer and rank.
```

### Ambiguous Outcomes

Any valid result that does not satisfy one complete rule below is ambiguous.
Failure to reject a control is not proof of absence.

## Maximum Claims

| Outcome | Maximum permitted claim |
| --- | --- |
| H1 criteria pass | At layer 30 and on the held-out corpus, native TMix pairings move normalized residuals into a stable complement of the fitted incoming-residual subspace more than matched shuffled and isotropic additions. |
| H2 criteria pass | At layer 30, a stable post-TMix destination is associated with structured empirical additions, but native residual/write pairing did not establish special routing. |
| H0 criteria pass | No held-out-stable rank-32 destination channel was detected at this layer, corpus, and sample size. |
| Ambiguous | The experiment did not distinguish the hypotheses. |

No outcome permits a semantic, memory-content, causal-use, rotor-code, logit, or
output-embedding claim.

## Experiment Admission Gate Summary

| Required field | Pre-registered answer |
| --- | --- |
| Scientific question | Does native TMix addition route normalized residuals into a stable complementary destination, and does native pairing matter? |
| Established observation | Layer-30 `time.out` and tangents showed above-isotropic structure without native-pairing specificity in a preliminary run. |
| Competing hypotheses | Native destination routing, uncoordinated structured addition, and no detected destination channel. |
| Intervention | No graph intervention. Analysis controls recompute `ADD + normalize` after empirical-write shuffling or isotropic norm-matched substitution. |
| Measured quantity | Held-out destination occupancy, native destination gain, pairing advantage, isotropic advantage, basis stability, and subspace overlap. |
| Positive outcome | Native destination gain exceeds both shuffled empirical-write and isotropic-write controls. |
| Negative outcome | Shuffled empirical writes match native routing, or no destination basis generalizes beyond isotropic controls. |
| Matched controls | Global pair shuffle, within-prompt pair shuffle, isotropic norm-matched writes, and isotropic destination-basis controls. |
| Why positive is not guaranteed | `ADD + LayerNorm` guarantees a direction change, but not a held-out complementary destination or dependence on the correct residual/write pairing. |
| Numerical validity | Backend verification, capture repeatability, residual identity, unit norms, basis orthogonality, prompt-level split isolation, sample minima, and complete control serialization. |
| Claim permitted | Distributional destination routing at one layer only. |

## Scope And Pre-Registration

The confirmatory layer is fixed:

```python
layer = 30
```

Selection rationale:

```text
Layer 30 was selected from a prior preliminary experiment because raw time.out had
held-out structure while native tangent pairing specificity was absent. This run
tests a new destination-routing prediction on a newly frozen corpus. It is not an
unbiased estimate over depth.
```

Primary and diagnostic ranks:

```python
primary_rank = 32
diagnostic_ranks = [8, 16, 32, 64]
```

Only rank 32 determines classification. Other ranks are sensitivity diagnostics.

Control count:

```python
control_repeats = 99
```

Use a new corpus that was not used by the rotor-subspace runs. Freeze and hash it
before activation collection. The corpus must contain independently decoded
prompts, and train/test splitting must be by prompt.

Minimum valid sample counts after exclusions:

```python
min_train_tokens = 512
min_test_tokens = 512
min_train_prompts = 8
min_test_prompts = 8
```

A smaller run must have status `invalid_insufficient_samples` and cannot classify
a hypothesis.

## Corpus Format And Split

Default corpus format is UTF-8 text with one prompt per nonempty line.

Define each prompt ID from:

```python
prompt_id = fnv1a64(
    little_endian_u64(nonempty_line_index)
    + exact_prompt_bytes
)
```

Do not trim prompt bytes except for the line delimiter. Record corpus and prompt
hashes.

Prefer a frozen split manifest. Otherwise assign prompts using a recorded seed and
the stable prompt ID. Never use `std::hash`.

No prompt ID may occur in both train and test. Do not move prompts after inspecting
activations.

## Production Captures

Capture these exact F32 production taps at layer 30:

```text
rwkv.layer.30.resid.in
rwkv.layer.30.time.out
rwkv.layer.30.resid.time
```

Every host-readable capture must be an explicit `ggml_dup` graph-output snapshot.

Decode each prompt from a fresh native recurrent state using
`llama_interp::runtime`. Do not reconstruct RWKV or LayerNorm manually.

## Per-Token Geometry

Use double-precision accumulation for means, norms, dot products, SVD/Gram
statistics, projected energies, and sanity checks. Stored vectors may be F32.

```python
def center(v):
    return v - mean(v)

def unit(v):
    return v / norm(v)

x = center(resid_in)
w = center(time_out)
y = center(resid_time)

radius_in = norm(x)
write_norm = norm(w)

u_in = unit(x)
u_time = unit(y)
w_hat = unit(w)
```

Reject a sample from vector analysis if any required norm is non-finite or below:

```python
minimum_norm = 1e-12
```

Retain exclusion counts and reasons in metadata.

## Basis Construction

All fitted bases use train prompts only. Do not subtract the across-sample dataset
mean in the primary analysis. A stable directional mean is part of the occupied
distribution.

Do not construct a dense `width x width` covariance. Use thin SVD, a Gram method,
or an equivalently validated decomposition of the `samples x width` matrix.

### Incoming Residual Basis

```python
P_in = top_right_singular_vectors(U_in_train, rank)
```

Rows of `U_in_train` are unit centered incoming residual directions.

### Input-Complement Projector

```python
def remove_input_subspace(v):
    return v - P_in @ (P_in.T @ v)
```

The complement is relative to the fitted rank-k incoming basis, not the complete
support of the incoming-residual distribution.

### Candidate Destination Basis

Residualize native post-TMix train directions against `P_in`:

```python
Z_dest_train = []

for u_time in U_time_train:
    z = remove_input_subspace(u_time)
    if norm(z) > minimum_norm:
        Z_dest_train.append(unit(z))

P_dest = top_right_singular_vectors(Z_dest_train, rank)
```

The candidate destination is therefore constrained to the orthogonal complement
of the fitted incoming basis.

The same `P_in` and `P_dest` must be used for native and control evaluation. Never
refit the destination basis on held-out native or control outputs for the primary
routing statistic.

### Raw Write Basis

```python
P_write = top_right_singular_vectors(W_hat_train, rank)
```

This basis describes structured ambient addition and is secondary to the routing
classification.

### Control Destination Bases

For destination-stability controls only, construct an independent candidate
destination basis from each isotropic-control training replicate by applying the
same process:

```python
P_dest_random = fit_destination_basis(
    P_in=P_in,
    post_add_directions=U_random_train,
    rank=rank,
)
```

Evaluate `P_dest_random` on its matched independent isotropic-control test
replicate. These bases define the finite-sample null for destination-basis
self-generalization. They are not used to evaluate native routing.

## Native Held-Out Metrics

For each held-out native token:

```python
input_destination_energy = norm(P_dest.T @ u_in) ** 2
time_destination_energy = norm(P_dest.T @ u_time) ** 2

destination_gain = (
    time_destination_energy
    - input_destination_energy
)
```

Aggregate:

```python
E_input_native = mean(input_destination_energy)
E_time_native = mean(time_destination_energy)
D_native = mean(destination_gain)
```

Destination-basis held-out stability:

```python
G_dest_native = projected_energy(P_dest, Z_dest_test)
```

where:

```python
Z_dest_test = [
    unit(remove_input_subspace(u_time))
    for u_time in U_time_test
    if norm(remove_input_subspace(u_time)) > minimum_norm
]
```

`D_native` is the primary routing metric. `G_dest_native` gates whether the fitted
destination itself generalizes.

## Descriptive Subspace Metrics

At every diagnostic rank, report:

```python
input_self = projected_energy(P_in, U_in_test)
write_self = projected_energy(P_write, W_hat_test)
destination_self = G_dest_native

input_write_overlap = frobenius_norm(P_in.T @ P_write) ** 2 / rank
write_destination_overlap = frobenius_norm(P_write.T @ P_dest) ** 2 / rank
input_destination_overlap = frobenius_norm(P_in.T @ P_dest) ** 2 / rank
```

Also report the principal-angle cosines for each basis pair.

`input_destination_overlap` should be numerically near zero by construction and is
a sanity check, not evidence.

## Analysis Controls

Controls require no additional model execution. Every control recomputes centered
addition and normalization exactly from captured values.

```python
def normalized_add_direction(centered_input, replacement_write):
    centered_write = center(replacement_write)
    return unit(centered_input + centered_write)
```

Do not substitute a tangent vector or reuse the native angle. This experiment tests
the complete finite `ADD + normalize` endpoint.

Generate 99 deterministic replicates per control family using distinct sub-seeds.
Never permute across train/test.

### Control A: Global Pair Shuffle

Within each split independently, globally permute empirical `time.out` vectors:

```python
u_global_shuffle[i] = normalized_add_direction(
    x[i],
    time_out[permutation[i]],
)
```

Evaluate held-out control routing with the fixed native `P_dest`:

```python
D_global_shuffle = mean(
    norm(P_dest.T @ u_global_shuffle_test[i]) ** 2
    - norm(P_dest.T @ u_in_test[i]) ** 2
)
```

This is the primary pairing control.

### Control B: Within-Prompt Pair Shuffle

Repeat Control A, but permute empirical writes only among positions in the same
prompt. This preserves prompt-level topic, style, and recurrent-trajectory
statistics while breaking token-local pairing.

This control is required and diagnostic. It does not determine the primary
classification.

### Control C: Isotropic Norm-Matched Writes

For each token, generate a random direction in the mean-zero hyperplane and scale
it to the native write norm:

```python
g = center(random_gaussian_vector(width))
w_random = write_norm * unit(g)

u_isotropic = unit(x + w_random)
```

Evaluate with the fixed native destination basis:

```python
D_isotropic = mean(
    norm(P_dest.T @ u_isotropic_test[i]) ** 2
    - norm(P_dest.T @ u_in_test[i]) ** 2
)
```

Use matched isotropic train/test control outputs to fit and evaluate
`P_dest_random` for the destination-stability null.

### Optional Diagnostic: Empirical Direction, Shuffled Norm

This diagnostic separates empirical write direction from write magnitude:

```python
w_direction_native = unit(center(time_out[i]))
w_norm_shuffled = write_norm[permutation[i]]
w_control = w_norm_shuffled * w_direction_native
```

It must not affect classification.

## Empirical Statistics

For each 99-replicate control distribution report:

```text
all replicate values in ROOT
mean
standard deviation
median
5th percentile
95th percentile
minimum
maximum
empirical one-sided p-value
```

Primary empirical values:

```python
native_advantage_global = D_native - median(D_global_shuffle)

p_global = (
    1 + count(D_global_shuffle >= D_native)
) / (1 + control_repeats)

native_advantage_isotropic = D_native - median(D_isotropic)

p_isotropic = (
    1 + count(D_isotropic >= D_native)
) / (1 + control_repeats)

p_destination_stability = (
    1 + count(G_dest_random >= G_dest_native)
) / (1 + control_repeats)
```

For H2, also compare shuffled empirical writes with isotropic writes. Pair replicate
indices by deterministic sub-seed:

```python
shuffle_advantage_isotropic = (
    median(D_global_shuffle)
    - median(D_isotropic)
)

p_shuffle_vs_isotropic = (
    1 + count(D_isotropic >= D_global_shuffle)
) / (1 + control_repeats)
```

If paired replicate comparison is used, record that exact method and preserve every
pair in ROOT.

## Pre-Registered Outcome Classification

Classification uses only layer 30, rank 32, global pair shuffle, isotropic
norm-matched writes, and isotropic destination-basis stability controls.

### H1-Supporting Result

Classify as `native_destination_routing` only if every condition passes:

```python
G_dest_native > percentile95(G_dest_random)
p_destination_stability <= 0.05

D_native > 0.0

native_advantage_global > 0.0
p_global <= 0.05

native_advantage_isotropic > 0.0
p_isotropic <= 0.05
```

### H2-Consistent Result

Classify as `uncoordinated_structured_addition` only if every condition passes:

```python
G_dest_native > percentile95(G_dest_random)
p_destination_stability <= 0.05

D_native > 0.0

p_global > 0.10

median(D_global_shuffle) > median(D_isotropic)
p_shuffle_vs_isotropic <= 0.05
```

This outcome says empirical write structure matters but native token-local pairing
was not established as special.

### H0-Consistent Result

Classify as `no_detected_destination_channel` only if:

```python
G_dest_native <= percentile95(G_dest_random)
```

### Ambiguous Result

Every other valid result is:

```text
ambiguous
```

The executable must mechanically evaluate and serialize every predicate. Do not
manually select the classification after inspecting results.

## Sanity Checks And Fail-Closed Criteria

### Backend And Instrumentation

Run `llama-rwkv-interp-verify` on the same model and backend before collection.
Record its artifact path or complete metrics.

### Capture Repeatability

Repeat all three captures for at least 16 deterministic token positions from at
least two prompts.

Required defaults:

```python
capture_cosine >= 0.99999
relative_norm_difference <= 1e-4
```

### Residual Identity

For every sample:

```python
expected = resid_in + time_out
relative_l2_error = norm(resid_time - expected) / max(norm(resid_time), 1e-12)
```

Required default:

```python
max_relative_l2_error <= 1e-3
```

Record maximum, mean, and 99th percentile.

### Direction Checks

Required defaults:

```python
abs(norm(u_in) - 1.0) <= 1e-5
abs(norm(u_time) - 1.0) <= 1e-5
abs(norm(w_hat) - 1.0) <= 1e-5
```

### Basis Checks

For every basis:

```python
orthogonality_error = max_abs(P.T @ P - identity(rank))
```

Required default:

```python
orthogonality_error <= 1e-5
max_abs(P_in.T @ P_dest) <= 1e-5
```

### Split And Control Checks

Verify:

```text
no prompt occurs in both train and test
all controls preserve original split membership
all 99 replicates use distinct deterministic sub-seeds
all required sample and prompt minima pass
rank 64 is below train and test sample counts
all native and control metrics are finite
all requested control values are serialized
```

Any failure sets status `invalid`, suppresses classification, and records the
failed checks.

## Protocol Deviations

All defaults and thresholds must be serialized. Any override must record:

```json
{
  "protocol_deviation": true,
  "field": "...",
  "old_value": "...",
  "new_value": "...",
  "reason": "..."
}
```

A deviated run must use new artifact paths and cannot overwrite the pre-registered
run.

## Suggested Executable And CLI

Source:

```text
pocs/interp/rwkv_tmix_destination_subspace.cpp
```

Target:

```text
llama-rwkv-tmix-destination-subspace
```

Required CLI:

```text
-m MODEL
--corpus PROMPTS.txt
--split-manifest MANIFEST.json
--output-root FILE.root
--output-json FILE.json
--seed N
-ngl N
```

The implementation must reject a `--layer` other than 30 for the confirmatory run.
If a reusable executable accepts other layers, any other layer must be marked
`exploratory` and cannot use the confirmatory classification label.

Optional CLI with recorded defaults:

```text
--max-tokens 2048
--max-tokens-per-prompt 128
--primary-rank 32
--diagnostic-ranks 8,16,32,64
--control-repeats 99
--exclude-special false
--pair-sample-count 100000
```

Example:

```bash
build/bin/llama-rwkv-tmix-destination-subspace \
  -m MODEL.gguf -ngl 99 \
  --corpus pocs/interp/destination_subspace_prompts.txt \
  --split-manifest pocs/interp/destination_subspace_prompts.manifest.json \
  --seed 23456 \
  --output-root build/tmix-destination-subspace-layer30.root \
  --output-json build/tmix-destination-subspace-layer30.json
```

The corpus names and seed in this example become binding only when the files are
frozen and the run is formally pre-registered.

Reject unknown arguments. Reject existing output paths unless an explicit
`--overwrite` override is recorded as a protocol deviation.

## ROOT Result Format

Use ROOT RNTuples and commit clusters regularly.

```text
schema_version = 1
```

### `metadata`

One row:

```text
schema_version
experiment_name
git_commit
working_tree_dirty
model path/hash
backend and GPU-layer configuration
layer and selection rationale
confirmatory/exploratory flag
corpus and manifest paths/hashes
seed and hash algorithm
width
prompt and token counts by split
all thresholds and defaults
protocol deviations
verifier artifact/metrics
```

### `samples`

One row per captured token:

```text
prompt_id
split
position
token_id
token_text
is_special

resid_in
time_out
resid_time

u_in
u_time
w_hat

radius_in
write_norm
input_destination_energy
time_destination_energy
destination_gain

identity and direction sanity errors
exclusion flags and reasons
```

### `destination_metrics`

One row per rank, control family, and replicate:

```text
rank
control_family: native, global_shuffle, within_prompt_shuffle,
                isotropic_write, isotropic_destination_basis,
                optional_shuffled_norm
replicate: -1 for native

G_dest
E_input
E_time
D
input_self
write_self
basis overlap metrics
principal-angle summaries
basis sanity metrics
control sub-seed
```

### `control_values`

One row per scalar control value so the complete empirical distributions can be
recomputed independently:

```text
rank
control_family
replicate
metric_name
metric_value
sub_seed
```

## JSON Summary Format

The JSON is the compact review artifact. Do not omit unfavorable metrics.

```json
{
  "schema_version": 1,
  "experiment": "rwkv_tmix_destination_subspace",
  "status": "valid|invalid|invalid_insufficient_samples",
  "classification": "native_destination_routing|uncoordinated_structured_addition|no_detected_destination_channel|ambiguous|suppressed",
  "scientific_question": "...",
  "hypotheses": {},
  "maximum_permitted_claim": "...",
  "metadata": {},
  "sample_counts": {},
  "sanity_checks": {},
  "primary_rank": 32,
  "native_metrics": {},
  "global_shuffle": {},
  "within_prompt_shuffle": {},
  "isotropic_write": {},
  "destination_stability": {},
  "diagnostic_ranks": [],
  "outcome_rules": {},
  "outcome_rule_evaluation": {},
  "root_artifact": "..."
}
```

Every control summary must include count, mean, standard deviation, median, 5th and
95th percentiles, minimum, maximum, empirical p-value, and a hash of the complete
ROOT control rows.

`outcome_rule_evaluation` must contain every Boolean predicate, observed operands,
and pass/fail value.

## Console Output

At completion print only progress and the scientific summary:

```text
status
classification
confirmatory/exploratory flag
sample counts
primary rank
G_dest_native and isotropic p95
D_native
native_advantage_global and p_global
native_advantage_isotropic and p_isotropic
shuffle_advantage_isotropic and p_shuffle_vs_isotropic
sanity pass/fail summary
ROOT path
JSON path
```

Do not print vocabulary tokens.

## Implementation Self-Tests

Add deterministic unit-level tests or startup self-tests for:

```text
centering and unit normalization
residual identity error
thin-SVD basis orthogonality
input-complement residualization
destination-basis orthogonality to P_in
projected energy
principal angles and chordal overlap
global and within-prompt permutation isolation
isotropic mean-zero write generation
empirical p-values
outcome classification truth table
stable prompt hashing and manifest parsing
```

Synthetic destination cases must include:

```text
native writes route into a fixed complement and shuffled writes do not
native and shuffled empirical writes route equally
isotropic writes equal native routing
no held-out destination stability
rank-deficient train matrix
near-zero complement residual
```

## Required Verification Sequence

```bash
cmake -S . -B build -DLLAMA_BUILD_RWKV_INTERP=ON
cmake --build build --target \
  llama-rwkv-interp-verify \
  llama-rwkv-tmix-destination-subspace

build/bin/llama-rwkv-interp-verify -m MODEL.gguf -ngl 99

build/bin/llama-rwkv-tmix-destination-subspace \
  -m MODEL.gguf -ngl 99 \
  --corpus pocs/interp/destination_subspace_prompts.txt \
  --split-manifest pocs/interp/destination_subspace_prompts.manifest.json \
  --seed 23456 \
  --output-root build/tmix-destination-subspace-layer30.root \
  --output-json build/tmix-destination-subspace-layer30.json
```

After execution, validate programmatically:

```text
JSON parses
metadata, samples, destination_metrics, and control_values RNTuples exist
row counts match metadata
all 99 required replicates are present
control-row hash matches JSON
classification matches the serialized predicate truth table
```

Then run:

```bash
git diff --check
```

## Explicit Non-Goals

Do not add:

- Rotor or tangent-subspace classification from the previous experiment.
- A direct final-head lens of `time.out` or any intermediate update.
- Vocabulary logits or output-embedding projection.
- Jacobians or path integration.
- Semantic classifiers, probes, or token labels.
- Zero-memory rollouts.
- Recurrent-state swaps, differences, rank approximations, or edits.
- Live `R`, `K`, or `V` editing.
- Causal residual or rotor interventions.
- ChannelMix analysis.
- An all-layer sweep.
- Post-hoc layer, rank, corpus, split, or exclusion selection.

These remain later questions conditional on a valid result from this gate.
