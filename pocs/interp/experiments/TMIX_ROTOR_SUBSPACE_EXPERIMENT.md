# TMix Rotor-Subspace Experiment Specification

## Status

This document specifies a proposed first distributional test. It is an
implementation handoff, not a record of a completed experiment.

Implement only the single-layer experiment described here. Do not expand it into
an all-layer sweep, semantic decoder, Jacobian lens, or vocabulary analysis.

Read these documents before implementation:

- `AGENTS.md`
- `SCIENTIFIC_METHOD_PROMPT.md`
- `pocs/interp/logs/TMIX_INVESTIGATION.md`
- `pocs/interp/README.md`

## Scientific Question

At one pre-registered RWKV layer, are native TMix updates coordinated with their
incoming residuals so that the resulting finite rotations occupy a stable tangent
subspace that is distinct from the subspace occupied by incoming residuals?

The test must distinguish residual-relative organization from two alternatives:

```text
global additive organization:
  time.out occupies a stable ambient subspace, and LayerNorm mechanically projects
  it into a tangent direction. Pairing a particular time.out with its actual
  resid.in is not important.

incidental or unstructured geometry:
  the observed tangent directions have no held-out structure beyond matched
  shuffled or isotropic controls.
```

The production computation remains:

```python
resid_time = resid_in + time_out
ffn_input = layer_norm(resid_time)
```

The experiment does not test whether RWKV literally executes a rotor instead of
this operation. The rotor is a mathematically equivalent description of the
centered pre-affine LayerNorm direction change.

## Observation Already Established

Existing production-graph captures establish that native TMix updates can induce
nonzero and sometimes large changes in centered residual direction. Tangent removal
can substantially change `ffn.norm`. These observations are guaranteed in part by
the geometry of addition followed by normalization and do not establish a rotor
code or a complementary information channel.

## Competing Hypotheses

### H1: Residual-Relative Tangent Organization

```text
Native resid.in/time.out pairings produce a held-out-stable tangent subspace.
Breaking those pairings materially reduces that structure.
The tangent subspace is measurably distinct from the occupied incoming-residual
subspace.
```

This is the distributional prediction of a mechanism that learned to rotate
normalized residuals into a complementary channel.

### H2: Global Additive Subspace

```text
time.out itself occupies a held-out-stable ambient subspace.
Native and shuffled resid.in/time.out pairings produce comparable tangent-subspace
structure.
```

This is consistent with a structured additive channel whose rotor description is
incidental to LayerNorm.

### H0: No Stable Low-Rank Organization At This Granularity

```text
Neither native tangent directions nor raw time.out directions generalize beyond
matched isotropic controls at the pre-registered layer and rank.
```

### Ambiguous Outcomes

Any result that does not satisfy a complete outcome rule below is ambiguous. In
particular, anisotropy without pairing specificity does not support H1, and failure
to reject a null does not prove the null.

## Maximum Claims

| Outcome | Maximum permitted claim |
| --- | --- |
| H1 criteria pass | TMix updates at the tested layer are distributionally coordinated with incoming residuals and occupy a held-out-stable complementary tangent subspace. |
| H2 criteria pass | Results are consistent with a stable ambient additive subspace; no residual-relative rotor organization was established. |
| H0 criteria pass | No stable low-rank additive or tangent organization was detected at the tested layer, rank, corpus, and sample size. |
| Ambiguous | The test did not distinguish the hypotheses. |

No outcome permits a claim about semantic content, memory-item identity, causal
use, logits, or output-embedding coordinates.

## Experiment Admission Gate Summary

| Required field | Pre-registered answer |
| --- | --- |
| Scientific question | Are native TMix tangents held-out-stable, complementary to incoming residuals, and dependent on the native residual/write pairing? |
| Established observation | Native TMix updates produce nonzero finite rotations and tangent-sensitive `ffn.norm` changes. |
| Competing hypotheses | Residual-relative tangent organization, global additive organization, and no detected low-rank organization. |
| Intervention | No graph intervention. Analysis controls shuffle native pairings or substitute matched isotropic directions after capture. |
| Measured quantity | Held-out subspace energy, residual/tangent cross-occupancy, principal angles, pairing gap, effective rank, and finite-rotation statistics. |
| Positive outcome | Native tangent structure exceeds isotropic and shuffled controls and occupies a complementary held-out subspace. |
| Negative outcome | Native tangents are control-like, or structure survives pair shuffling and is better explained by raw `time.out`. |
| Matched controls | Global pairing shuffle, within-prompt pairing shuffle, isotropic matched tangents, and isotropic mean-zero ambient directions. |
| Why positive is not guaranteed | `ADD + LayerNorm` guarantees a tangent direction, but it does not guarantee held-out low-rank structure, complementarity, or dependence on the correct residual/write pairing. |
| Numerical validity | Capture replay, residual identities, rotor reconstruction, tangent orthogonality, SVD orthogonality, finite sample minima, deterministic splits, and fail-closed thresholds below. |
| Claim permitted | Distributional organization at one layer only; no semantic, memory-content, causal-use, or logit claim. |

## Scope And Pre-Registration

The implementation must require an explicit layer. There must be no default layer
and no search over layers.

The layer-selection rationale must be supplied on the command line and copied into
the result metadata. Valid examples include selection from a prior artifact or an
architectural reason. Selection based on this experiment's result is invalid.

The primary analysis rank is fixed at:

```python
primary_rank = 32
```

Sensitivity ranks are reported but cannot determine the classification:

```python
diagnostic_ranks = [8, 16, 32, 64]
```

The primary control count is:

```python
control_repeats = 99
```

The random seed must be explicit and recorded.

The corpus must contain independently decoded prompts. Train/test splitting is by
prompt, never by token, so tokens from one recurrent trajectory cannot occur in
both splits.

The default corpus format is UTF-8 text with one prompt per nonempty line. Define
the stable prompt ID from the zero-based nonempty-line index and the exact prompt
bytes. Record both the corpus file hash and each prompt hash. Do not trim prompt
content except for removing the line delimiter; whitespace is part of the prompt.

Minimum valid sample counts after all exclusions:

```python
min_train_tokens = 256
min_test_tokens = 256
min_train_prompts = 4
min_test_prompts = 4
```

A smaller run may be written as a diagnostic artifact, but its result status must
be `invalid_insufficient_samples` and it must not classify a hypothesis.

## Production Captures

At the one requested layer, capture these exact production taps in F32:

```text
rwkv.layer.<L>.resid.in
rwkv.layer.<L>.time.out
rwkv.layer.<L>.resid.time
rwkv.layer.<L>.channel.out
rwkv.layer.<L>.resid.out
```

Every capture must use the existing explicit `ggml_dup` graph-output snapshot path.
Never read a reused GGML intermediate directly.

Each prompt must start from a fresh native recurrent state and be decoded by
`llama_interp::runtime`. Do not manually reconstruct RWKV or LayerNorm.

Exclude padding and positions that do not have all five valid captures. Record BOS
and other special tokens rather than silently excluding them. The CLI may provide a
pre-registered `--exclude-special` option, whose value must be recorded.

## Per-Token Geometry

Use double-precision accumulators for means, dot products, norms, covariance/Gram
statistics, and sanity checks. Stored vectors may remain F32.

### Helpers

```python
def center(v):
    return v - mean(v)

def unit(v):
    return v / norm(v)

def clamp_cosine(x):
    return max(-1.0, min(1.0, x))
```

### TMix Rotation

```python
x = center(resid_in)
w = center(time_out)
y = center(resid_time)

radius_in = norm(x)
u_in = x / radius_in
u_time = y / norm(y)

radial_scalar = dot(w, u_in)
tangent_write = w - radial_scalar * u_in
tangent_write_norm = norm(tangent_write)

cos_angle = clamp_cosine(dot(u_in, u_time))
angle_acos = acos(cos_angle)
angle_atan2 = atan2(tangent_write_norm, radius_in + radial_scalar)

q_tmix = tangent_write / tangent_write_norm
xi_tmix = angle_atan2 * q_tmix

w_direction = w / norm(w)
```

`q_tmix` is the oriented unit tangent direction. `xi_tmix` is the logarithmic
representation of the canonical minimal rotor: its direction is the oriented
two-plane direction at `u_in`, and its norm is the finite rotation angle.

Use `angle_atan2` as the canonical angle after it passes the equality check against
`angle_acos`.

Degenerate samples with a tangent norm below this relative threshold are excluded
from tangent-subspace analysis and retained in scalar summaries:

```python
tangent_write_norm <= 1e-8 * radius_in
```

The exclusion count and fraction must be recorded. More than 1% degenerate samples
invalidates classification unless the threshold was changed before execution and
recorded as a protocol deviation.

### ChannelMix Rotation

Compute a secondary, non-classifying ChannelMix rotation:

```python
x_channel = center(resid_time)
w_channel = center(channel_out)
y_channel = center(resid_out)

radius_channel_in = norm(x_channel)
u_channel_in = x_channel / radius_channel_in
u_out = y_channel / norm(y_channel)

channel_radial = dot(w_channel, u_channel_in)
channel_tangent_write = w_channel - channel_radial * u_channel_in
channel_tangent_norm = norm(channel_tangent_write)

channel_angle = atan2(
    channel_tangent_norm,
    radius_channel_in + channel_radial,
)

q_channel = channel_tangent_write / channel_tangent_norm
xi_channel = channel_angle * q_channel
```

ChannelMix metrics are descriptive only in this first experiment. They must not
affect the hypothesis classification.

### Finite-Rotation Nonlinearity

```python
linear_angle = tangent_write_norm / radius_in
exact_angle = angle_atan2

if linear_angle > 1e-8:
    nonlinearity_ratio = exact_angle / linear_angle
```

Report quantiles of `linear_angle`, `exact_angle`, and `nonlinearity_ratio`, plus
the fractions of samples above predeclared angle thresholds:

```python
angle_thresholds_radians = [0.01, 0.05, 0.10, 0.25, 0.50]
```

These metrics describe whether the layer operates in a finite-rotation regime.
They are not evidence for H1 by themselves.

## Train/Test Split

Assign each prompt deterministically from its prompt ID and the recorded seed:

```python
split_value = stable_hash(prompt_id, seed)
split = "train" if split_value is in the lower half else "test"
```

Do not use `std::hash`, because its portability is not guaranteed. Use a specified
stable hash such as FNV-1a 64-bit and record its name.

If the initial split does not meet prompt or token minima, fail closed. Do not move
individual prompts between splits after inspecting activations. A separately
specified split manifest is acceptable and preferred for exact reproducibility.

## Subspace Estimation

Do not form a dense `width x width` covariance matrix. Use a thin SVD, a Gram-matrix
method, or an equivalent numerically validated method on the `samples x width`
matrix.

Vectors have already been centered across neural coordinates. Do not subtract the
dataset mean before the primary uncentered second-moment analysis. A stable mean
direction is part of the occupied directional distribution.

For representation matrix `X`, whose rows are samples:

```python
def fit_basis(X_train, rank):
    # Return orthonormal right singular vectors with shape [width, rank].
    return top_right_singular_vectors(X_train, rank)

def projected_energy(basis, samples):
    total = 0.0
    for v in samples:
        total += norm(transpose(basis) @ v) ** 2 / norm(v) ** 2
    return total / len(samples)
```

Fit these native bases on training data:

```python
P_U = fit_basis(U_train, rank)          # u_in
P_W = fit_basis(W_train, rank)          # centered unit time.out
P_T = fit_basis(Q_train, rank)          # q_tmix
P_XI = fit_basis(XI_train, rank)        # angle-weighted xi_tmix
P_C = fit_basis(C_train, rank)          # q_channel, secondary
```

`Q` is the primary tangent representation. `XI` is a secondary amplitude-weighted
version. The hypothesis classification uses `Q`, not `XI`.

## Primary Metrics

Compute every metric at all diagnostic ranks. Classification uses rank 32 only.

### Held-Out Self-Generalization

```python
G_U = projected_energy(P_U, U_test)
G_W = projected_energy(P_W, W_test)
G_T = projected_energy(P_T, Q_test)
G_XI = projected_energy(P_XI, XI_test)
G_C = projected_energy(P_C, C_test)
```

### Residual/Tangent Cross-Occupancy

```python
T_in_U = projected_energy(P_U, Q_test)
T_in_T = projected_energy(P_T, Q_test)

U_in_U = projected_energy(P_U, U_test)
U_in_T = projected_energy(P_T, U_test)

tangent_preference = T_in_T - T_in_U
residual_preference = U_in_U - U_in_T
complementarity = min(tangent_preference, residual_preference)
```

A positive `complementarity` means each held-out distribution prefers its own
training subspace. Its magnitude must be judged against controls, not in isolation.

### Subspace Overlap And Principal Angles

```python
cross = transpose(P_U) @ P_T
singular_values = svd(cross)

principal_angle_cosines = singular_values
normalized_chordal_overlap = (
    frobenius_norm(cross) ** 2 / rank
)
```

Report every principal-angle cosine at the primary rank and summary statistics at
diagnostic ranks. Low overlap is descriptive; classification still requires
held-out and shuffled-control evidence.

### Distribution Shape

For `U`, `W`, `Q`, `XI`, and secondary `C`, report:

```text
mean resultant norm
mean signed pairwise cosine
mean absolute pairwise cosine
pairwise cosine standard deviation
top-k explained second-moment fraction
participation-ratio effective rank
entropy effective rank
```

Compute pairwise statistics exactly when affordable. Otherwise use a deterministic,
recorded sample of at least 100,000 distinct pairs.

## Controls

All controls are analysis-only. They require no additional model execution.

Generate 99 independent replicates for each control family using deterministic
sub-seeds derived from the main seed and control family name.

### Control A: Global Pair Shuffle

Within train and test splits separately, permute `time.out` across all prompts and
tokens. Never move a sample across train/test.

For incoming direction `u_in[i]` and permuted write `w[j]`:

```python
w_shuffled = center(time_out[j])
t_shuffled = w_shuffled - dot(w_shuffled, u_in[i]) * u_in[i]
q_shuffled = unit(t_shuffled)
xi_shuffled = native_angle[i] * q_shuffled
```

Using the native angle preserves the empirical angle distribution while breaking
the residual/write pairing. If a projected shuffled tangent is degenerate, redraw
the permutation target deterministically and record the redraw count.

For each replicate, fit the shuffled training basis and evaluate shuffled test
self-generalization:

```python
P_T_shuffled = fit_basis(Q_shuffled_train, rank)
G_T_shuffled = projected_energy(P_T_shuffled, Q_shuffled_test)
```

Also evaluate the native tangent basis on shuffled held-out directions:

```python
native_basis_on_shuffled = projected_energy(P_T, Q_shuffled_test)
```

Across the 99 replicates, define:

```python
pairing_gap = G_T - median(native_basis_on_shuffled)

p_pairing = (
    1 + count(native_basis_on_shuffled >= G_T)
) / (1 + control_repeats)

p_shuffle_self = (
    1 + count(G_T_shuffled >= G_T)
) / (1 + control_repeats)
```

`pairing_gap`, `p_pairing`, and `p_shuffle_self` are the primary
pairing-specific statistics. `p_pairing` asks whether the native tangent basis
continues to explain tangent directions after local pairings are broken.
`p_shuffle_self` asks whether shuffled pairings can independently learn a tangent
subspace as coherent as the native one.

For each replicate, also compute the shuffled residual/tangent cross-occupancy and
its `complementarity`. Define:

```python
p_complementarity = (
    1 + count(complementarity_global_shuffle >= complementarity)
) / (1 + control_repeats)
```

### Control B: Within-Prompt Pair Shuffle

Repeat Control A, but only permute writes among token positions from the same
prompt. This preserves prompt-level topic, style, and trajectory statistics while
breaking token-local pairing.

The global shuffle is the primary pairing control. The within-prompt shuffle is a
required diagnostic that can expose prompt-level confounding.

### Control C: Isotropic Matched Tangents

For every native `u_in`, sample a Gaussian vector using the deterministic control
RNG, center it across neural coordinates, and tangent-project it:

```python
g = center(random_gaussian_vector(width))
t_random = g - dot(g, u_in) * u_in
q_random = unit(t_random)
xi_random = native_angle * q_random
```

Fit random training bases and evaluate random held-out self-generalization. This is
the finite-sample null for tangent-space isotropy.

### Control D: Isotropic Mean-Zero Ambient Directions

Generate independent centered random unit vectors without conditioning on `u_in`.
Use them to obtain null self-generalization distributions for `U` and `W`:

```python
ambient_random = unit(center(random_gaussian_vector(width)))
```

This determines whether incoming residuals and raw updates possess held-out
low-rank structure beyond sample-size artifacts.

## Empirical Control Statistics

For each scalar native metric and its 99 control replicates, report:

```text
control mean
control standard deviation
control median
control 5th percentile
control 95th percentile
empirical one-sided p-value
```

Use the finite-sample empirical p-value:

```python
p_greater = (
    1 + count(control_value >= native_value)
) / (1 + control_repeats)
```

For metrics where smaller values are the alternative, reverse the comparison and
name the field `p_less`. Do not report Gaussian extrapolated p-values.

## Pre-Registered Outcome Classification

Classification uses only rank 32 and the global-shuffle and isotropic controls.
Within-prompt shuffle and other ranks are diagnostics.

### H1-Supporting Result

Classify as `residual_relative_tangent_structure` only when all conditions pass:

```python
G_U > percentile95(G_U_ambient_random)
G_T > percentile95(G_T_isotropic_tangent)

pairing_gap > 0.0
p_pairing <= 0.05
p_shuffle_self <= 0.05

complementarity > 0.0
complementarity > percentile95(complementarity_global_shuffle)
p_complementarity <= 0.05
```

### H2-Consistent Result

Classify as `global_additive_structure_consistent` only when:

```python
G_W > percentile95(G_W_ambient_random)
G_T > percentile95(G_T_isotropic_tangent)
p_pairing > 0.10
p_shuffle_self > 0.10
complementarity <= percentile95(complementarity_global_shuffle)
```

This classification is consistency evidence, not proof that the model uses an
additive code.

### H0-Consistent Result

Classify as `no_detected_low_rank_structure` only when:

```python
G_W <= percentile95(G_W_ambient_random)
G_T <= percentile95(G_T_isotropic_tangent)
```

### Ambiguous Result

Every other valid result is classified as:

```text
ambiguous
```

The executable must compute the classification mechanically from the recorded
metrics. Do not manually assign a favorable label after inspecting results.

## Sanity Checks And Fail-Closed Criteria

### Instrumentation Validation

Run `llama-rwkv-interp-verify` on the same model and backend before collection.
Record the verifier artifact path or its complete reported metrics in metadata.

### Residual Identities

For every sample, verify:

```python
resid_time_expected = resid_in + time_out
resid_out_expected = resid_time + channel_out

relative_l2_error = norm(actual - expected) / max(norm(actual), 1e-12)
```

Default fail-closed threshold:

```python
max_relative_l2_error <= 1e-3
```

Record maximum, mean, and 99th-percentile errors for both identities.

### Geometry Checks

Default fail-closed thresholds:

```python
abs(norm(u_in) - 1.0) <= 1e-5
abs(norm(u_time) - 1.0) <= 1e-5
abs(dot(u_in, q_tmix)) <= 1e-5
abs(angle_acos - angle_atan2) <= 1e-5

u_reconstructed = (
    cos(angle_atan2) * u_in
    + sin(angle_atan2) * q_tmix
)

norm(u_reconstructed - u_time) <= 1e-5
```

Report maxima even when all checks pass.

### SVD Checks

For every fitted basis used in classification:

```python
orthogonality_error = max_abs(
    transpose(basis) @ basis - identity(rank)
)
```

Default fail-closed threshold:

```python
orthogonality_error <= 1e-5
```

Reject NaN or infinite singular values, metrics, vectors, or control statistics.

### Repeatability

Re-run captures for a deterministic subset of at least 16 token positions from at
least two prompts. Compare each of the five taps.

Default fail-closed thresholds:

```python
capture_cosine >= 0.99999
relative_norm_difference <= 1e-4
```

### Split And Leakage Checks

Verify and record:

```text
no prompt ID occurs in both train and test
all native and control rows retain their original split
all 99 control replicates use distinct deterministic sub-seeds
sample counts satisfy minima
rank 64 is less than both train and test sample counts
```

Any failure sets result status to `invalid` and suppresses hypothesis
classification.

### Protocol Deviations

Every threshold and default must be available in metadata. If a CLI override is
provided, the result must contain:

```json
"protocol_deviation": true
```

along with the old value, new value, and user-supplied reason. A protocol-deviated
run must not overwrite the original artifact.

## Executable And CLI

Suggested executable:

```text
llama-rwkv-tmix-rotor-subspace
```

Suggested source:

```text
pocs/interp/rwkv_tmix_rotor_subspace.cpp
```

Required CLI:

```text
-m MODEL
--corpus PROMPTS.txt
--layer L
--layer-selection-rationale TEXT
--output-root FILE.root
--output-json FILE.json
--seed N
-ngl N
```

Optional CLI with recorded defaults:

```text
--max-tokens 1024
--max-tokens-per-prompt 128
--primary-rank 32
--diagnostic-ranks 8,16,32,64
--control-repeats 99
--split-manifest FILE
--exclude-special true|false
--pair-sample-count 100000
```

Reject unknown arguments. Reject a missing layer rationale. Reject output files that
already exist unless an explicit non-default `--overwrite` option is provided and
recorded as a protocol deviation.

Example invocation:

```bash
build/bin/llama-rwkv-tmix-rotor-subspace \
  -m MODEL.gguf -ngl 99 \
  --corpus pocs/interp/rotor_subspace_prompts.txt \
  --layer 30 \
  --layer-selection-rationale "selected before this run from prior native-angle artifact" \
  --seed 12345 \
  --output-root build/tmix-rotor-subspace-layer30.root \
  --output-json build/tmix-rotor-subspace-layer30.json
```

The example layer is illustrative, not a prescribed layer.

## ROOT Result Format

Use ROOT RNTuples and commit clusters regularly. Suggested schema version:

```text
schema_version = 1
```

### `metadata` RNTuple

One row containing:

```text
schema_version
experiment_name
git_commit
working_tree_dirty
model_path
model_hash if available
backend
n_gpu_layers
layer
layer_selection_rationale
corpus_path
corpus_hash
split_manifest_path/hash if present
seed
width
prompt_count
train/test prompt counts
train/test token counts
all thresholds and CLI defaults
protocol_deviation fields
verifier artifact/metrics
```

### `samples` RNTuple

One row per valid token:

```text
prompt_id
split
position
token_id
token_text
is_special
layer

resid_in
time_out
resid_time
channel_out
resid_out

u_in
u_time
w_direction
q_tmix
xi_tmix
q_channel
xi_channel

radius_in
time_out_centered_norm
radial_scalar
tangent_write_norm
angle
linear_angle
nonlinearity_ratio
channel_angle

identity and geometry sanity errors
degenerate flags
```

Vectors are fixed width and must include shape metadata. If storing all raw taps and
derived vectors is too expensive, raw taps take precedence over derived vectors
because derived values can be recomputed. Do not omit `u_in`, `q_tmix`, or the raw
five taps from a scientific artifact without documenting the storage tradeoff.

### `subspace_metrics` RNTuple

One row per representation, rank, and control replicate:

```text
representation: U, W, Q, XI, C
rank
control_family: native, global_shuffle, within_prompt_shuffle,
                isotropic_tangent, isotropic_ambient
control_replicate: -1 for native
self_generalization
cross_occupancy fields
complementarity
normalized_chordal_overlap
effective-rank metrics
SVD sanity metrics
```

Do not store every fitted basis unless needed for exact replay. If bases are stored,
place them in a separate RNTuple keyed by representation, rank, and replicate.

## JSON Summary Format

The JSON is the review artifact. It must contain no omitted unfavorable metrics.

Required top-level structure:

```json
{
  "schema_version": 1,
  "experiment": "rwkv_tmix_rotor_subspace",
  "status": "valid|invalid|invalid_insufficient_samples",
  "classification": "residual_relative_tangent_structure|global_additive_structure_consistent|no_detected_low_rank_structure|ambiguous|suppressed",
  "scientific_question": "...",
  "hypotheses": {},
  "maximum_permitted_claim": "...",
  "metadata": {},
  "sample_counts": {},
  "sanity_checks": {},
  "geometry_summary": {},
  "primary_rank": 32,
  "primary_metrics": {},
  "diagnostic_ranks": [],
  "control_distributions": {},
  "outcome_rules": {},
  "outcome_rule_evaluation": {},
  "root_artifact": "..."
}
```

For each control distribution, include all 99 values or store them in ROOT and
include their hash plus complete summary statistics in JSON.

`outcome_rule_evaluation` must show every Boolean predicate, its observed values,
and whether it passed. Do not output only the final classification.

## Console Output

Console output is progress and diagnostics only. At completion print:

```text
status
classification
sample counts
primary rank
G_U, G_W, G_T
pairing_gap, p_pairing, and p_shuffle_self
complementarity and empirical p-value
sanity pass/fail summary
ROOT path
JSON path
```

Do not print top vocabulary tokens because this experiment does not compute a
vocabulary mapping.

## Implementation Tests

Add deterministic unit-level tests or startup self-tests for:

```text
centering
radial/tangent decomposition
angle equality
rotor endpoint reconstruction
random tangent orthogonality
stable prompt hashing
projected-energy calculation
principal-angle calculation
empirical p-value calculation
outcome classification truth table
```

Synthetic geometry cases must include:

```python
pure_mean_update
pure_radial_update
small_tangent_update
large_tangent_update
mixed_radial_tangent_update
near_zero_tangent
```

The pure mean and non-sign-flipping pure radial cases should produce zero rotation
up to numerical tolerance. Include a separate sign-flipping radial edge case and
verify that it is detected as the antipodal ambiguity rather than silently assigned
an arbitrary tangent plane. The mixed case must agree with the `atan2` formula and
reconstruct its endpoint.

## Required Verification Sequence

```bash
cmake -S . -B build -DLLAMA_BUILD_RWKV_INTERP=ON
cmake --build build --target \
  llama-rwkv-interp-verify \
  llama-rwkv-tmix-rotor-subspace

build/bin/llama-rwkv-interp-verify -m MODEL.gguf -ngl 99

build/bin/llama-rwkv-tmix-rotor-subspace \
  -m MODEL.gguf -ngl 99 \
  --corpus CORPUS.txt \
  --layer LAYER \
  --layer-selection-rationale "PRE-REGISTERED REASON" \
  --seed SEED \
  --output-root OUTPUT.root \
  --output-json OUTPUT.json
```

After execution:

```bash
git diff --check
```

Also validate programmatically that the JSON parses, both ROOT RNTuples exist, row
counts match metadata, and the output classification exactly matches the recorded
predicate truth table.

## Explicit Non-Goals

Do not add any of the following to this experiment:

- A final-head lens of `time.out`, `q_tmix`, or `xi_tmix`.
- A semantic classifier or probe.
- A zero-memory rollout.
- Recurrent-state swapping, differencing, SVD, or editing.
- Live `R`, `K`, or `V` editing.
- A Jacobian to final residuals or logits.
- A causal rotor intervention.
- An all-layer sweep.
- Post-hoc selection of layer, rank, prompts, or exclusions.

Those are possible later experiments only if this distributional gate produces a
valid result that motivates them.
