# TMix Relative-Geometry Null Experiment

## Status

This is a discussed and admitted implementation handoff. It has not been
implemented or run.

This is a single-layer confirmatory test of a newly specified prediction on a new
frozen corpus. Layer 15 was selected from prior exploratory Pile results and is not
an unbiased depth selection. Existing text8 and Pile captures may be used for code
development only; they cannot classify this experiment.

Read before implementation:

- `../../../../AGENTS.md`
- `../../../../SCIENTIFIC_METHOD_PROMPT.md`
- `../../logs/TMIX_INVESTIGATION.md`
- `../TMIX_ROTOR_SUBSPACE_EXPERIMENT.md`
- `../../logs/TMIX_ROTOR_SUBSPACE_RESULTS.md`
- `../TMIX_DESTINATION_SUBSPACE_EXPERIMENT.md`
- `../../logs/TMIX_DESTINATION_SUBSPACE_RESULTS.md`
- `../tmix_destination_base_recovery/RESULTS.md`

## Current Scientific Goal

The eventual goal is to determine what RWKV memory contributes to the residual
stream and whether that contribution can be decoded before the FFN turns it into a
learned policy response.

Directly applying the final output head to `time.out` has produced incoherent output
and has no validated interpretation. Meanwhile, the architecture necessarily maps:

```python
resid_time = resid_in + time_out
u_in = unit(center(resid_in))
u_time = unit(center(resid_time))
```

into a canonical minimal rotor from `u_in` to `u_time`.

The existence of that rotor is an identity. This experiment does not treat it as
evidence. It asks only whether native residual/write pairing creates stable relative
geometric organization beyond what follows from the structured ambient write
distribution and `ADD + normalize` alone.

## Scientific Question

At layer 15 on a new frozen corpus, do native TMix pairings produce a held-out-stable
rank-32 field of finite canonical tangent directions, in both ambient and
common-reference coordinates, that exceeds complete globally shuffled,
within-prompt shuffled, and isotropic norm-matched addition pipelines?

## Observation Already Established

Prior work established:

```text
layer-30 destination occupancy is held-out stable across two corpora

empirical writes produce greater destination occupancy than isotropic writes

native layer-30 token-local pairing specificity did not replicate

an exploratory Pile depth scan found native global-shuffle pairing advantage only
at layer 15

removing the layer-30 rank-32 destination did not recover incoming directions
```

These results support structured TMix addition and motivate layer 15 as a candidate.
They do not establish a relative geometric code, semantic content, memory content,
or causal use.

## Null And Alternative

### H0: Incidental Relative Geometry

```text
TMix writes have structured ambient directions, but pairing each write with its
native incoming residual creates no additional held-out-stable low-rank relative
geometric organization.

Any tangent structure is explained by the empirical write distribution, prompt
statistics, finite-sample anisotropy, and the unavoidable geometry of ADD +
normalize.
```

### H1: Native-Pairing Relative Organization

```text
Native incoming-residual/write pairing creates a held-out-stable low-rank field of
canonical tangent directions that is weakened when the same empirical writes are
paired with other residuals.
```

H1 is deliberately weaker than a rotor-code or memory-content hypothesis. Rejecting
H0 only justifies continuing to investigate relative coordinates.

## Shared Predictions

Both H0 and H1 permit:

- a canonical rotor, angle, plane, and tangent for every token;
- structured raw `time.out` directions;
- stable post-TMix endpoint or destination occupancy;
- changes to the normalized FFN input;
- an FFN response to those changes.

None of those observations rejects H0.

## Why A Positive Result Is Not Guaranteed

`ADD + normalize` guarantees a token-local tangent. It does not guarantee that one
rank-32 basis fitted on train prompts will capture held-out native tangent directions,
that native organization will exceed independently fitted shuffled empirical-write
pipelines, or that it will survive both ambient and parallel-transport coordinate
descriptions.

## Fixed Scope

```python
layer = 15
primary_rank = 32
control_repeats = 99
seed = 817263
minimum_norm = 1e-12
minimum_angle = 1e-6
```

No diagnostic rank sweep is permitted in the classified run.

Use a newly frozen corpus that has no prompt overlap with:

```text
pocs/interp/rotor_subspace_prompts.txt
pocs/interp/destination_subspace_expanded_prompts.txt
pocs/interp/destination_subspace_pile10k_prompts.txt
```

Freeze before activation inspection:

```text
exact corpus bytes and cryptographic hash
prompt IDs
train/test split manifest
source provenance
experiment seed
```

Create a registration JSON before collection containing the exact corpus and
manifest SHA-256 hashes, seed `817263`, source provenance, and exact prompt hashes
for the three prohibited prior corpora. The classified analyzer must verify this
registration and reject any mismatch. Development mode may omit it only while
suppressing classification.

Minimum retained samples:

```python
min_train_tokens = 512
min_test_tokens = 512
min_train_prompts = 8
min_test_prompts = 8
```

Split only by prompt. No prompt may occur in both splits. Do not alter the corpus,
split, exclusions, layer, rank, or seed after inspecting activations.

## Capture Input

Use an existing validated production-graph capture path to collect these layer-15
F32 snapshots:

```text
rwkv.layer.15.resid.in
rwkv.layer.15.time.out
rwkv.layer.15.resid.time
```

Every capture must be an explicit `ggml_dup` graph-output snapshot. Run
`llama-rwkv-interp-verify` on the same model and backend before collection.

The analysis executable must read an existing ROOT artifact and must not link or
execute the model runtime. Keep collection and analysis separate.

Required per-token fields:

```text
prompt_id
split
position
token_id
resid_in
time_out
resid_time
```

## Per-Token Finite Geometry

Use double-precision accumulation:

```python
def center(x):
    return x - mean(x)

def unit(x):
    return x / norm(x)

x = center(resid_in)
w = center(time_out)
y = center(resid_time)

radius_in = norm(x)
write_norm = norm(w)

u = unit(x)
v = unit(y)

cos_theta = clamp(dot(u, v), -1, 1)
theta = acos(cos_theta)

t_raw = v - cos_theta * u
t = unit(t_raw)
q = theta * t
```

`t` is the unit tangent direction and `q` is the finite canonical log tangent. The
primary directional analysis uses `t`; `theta` and `q` norms are required
diagnostics.

Exclude a token from tangent analysis if:

```python
norm(x) <= minimum_norm
norm(w) <= minimum_norm
norm(y) <= minimum_norm
theta < minimum_angle
any required value is non-finite
```

Record every exclusion and reason. More than 1% exclusions in either split makes
the run invalid.

Do not replace the finite endpoint with a first-order projection of `w`.

## Representation A: Ambient Tangent

All model residual vectors share one ambient coordinate system. The first
representation is the unit canonical tangent `t` itself:

```python
z_ambient = t
```

This captures low-rank relative displacement directions expressed in residual
coordinates.

## Representation B: Parallel-Transported Tangent

Tangents at different `u` lie in different tangent spaces. Define one train-only
reference direction using a prompt-balanced extrinsic mean:

```python
prompt_mean[p] = mean(u for train tokens in prompt p)
reference_raw = mean(prompt_mean.values())
reference = unit(reference_raw)
```

Fail if:

```python
norm(reference_raw) <= 1e-6
min(1 + dot(u, reference)) <= 1e-4
```

Parallel transport each unit tangent along the shortest sphere geodesic from `u`
to `reference`:

```python
def transport_to_reference(u, t, reference):
    return t - (
        dot(t, reference) / (1 + dot(u, reference))
    ) * (u + reference)

z_transport = unit(
    transport_to_reference(u, t, reference)
)
```

The same native train-fitted reference must be used for native and every control.
Never refit a control-specific reference.

Required transport checks:

```python
abs(dot(z_transport, reference)) <= 1e-5
abs(norm(z_transport) - 1) <= 1e-5
abs(norm(transported_unscaled_t) - norm(t)) <= 1e-5
```

The ambient and transported tests are co-primary. Requiring both prevents a
classification based only on one arbitrary tangent-space comparison convention.

## Prompt-Balanced Basis Fitting

Fit train bases independently for ambient and transported representations.

To give every train prompt equal covariance weight, scale every unit row from
prompt `p` by:

```python
row_weight[p] = 1 / sqrt(valid_train_tokens_in_prompt[p])
```

Then fit:

```python
P_ambient = top_right_singular_vectors(
    weighted_Z_ambient_train,
    rank=32,
)

P_transport = top_right_singular_vectors(
    weighted_Z_transport_train,
    rank=32,
)
```

Do not subtract an across-token dataset mean. A stable directional mean is part of
the occupied distribution.

Use a thin SVD, Gram method, or validated deterministic randomized SVD. Do not form
a dense width-by-width covariance. If randomized SVD is used, serialize its seed,
oversampling, power iterations, residual error, and exact small-matrix validation.

## Scalability Contract

The implementation must operate on width-sized vectors and rank-sized factors. It
must never materialize:

```text
a width-by-width covariance
a width-by-width rotor matrix
an exterior-space bivector tensor
all control endpoint vectors for every replicate simultaneously
```

Precompute immutable centered inputs and writes, process control replicates
independently, and release each replicate's vectors after its scalar metrics are
serialized. Parallel replicates must remain deterministic and cap BLAS threads so
worker multiplication does not oversubscribe the host.

Record peak resident memory and wall time in metadata. A future experiment may
reuse the same analyzer at another pre-registered layer without changing the
scientific metric, but this run classifies layer 15 only.

## Held-Out Stability Metric

For representation `R` and its train-fitted basis `P_R`:

```python
token_energy[i] = norm(P_R.T @ z_R_test[i]) ** 2

prompt_energy[p] = mean(token_energy for valid test tokens in prompt p)

G_R_native = mean(prompt_energy.values())
```

The co-primary native metrics are:

```text
G_ambient_native
G_transport_native
```

Serialize every per-prompt energy.

## Complete Finite Controls

Every control recomputes the complete centered finite endpoint, angle, tangent, and
transported tangent. Never permute precomputed tangents or reuse native angles.

Use 99 deterministic replicates with distinct sub-seeds. Generate train and test
controls independently and never cross split boundaries.

### Control A: Global Empirical-Write Shuffle

Within each split, globally permute complete centered empirical writes:

```python
v_shuffle[i] = unit(x[i] + w[permutation[i]])
```

For each representation and replicate:

1. Fit a rank-32 basis on shuffled train representations.
2. Evaluate that basis on independently shuffled test representations:

   ```text
   G_global_self
   ```

3. Evaluate the fixed native basis on the shuffled test representations:

   ```text
   G_global_native_basis
   ```

The self metric controls for low-rank organization induced by the unchanged
empirical write distribution. The fixed-native-basis metric controls whether
shuffled endpoints continue to occupy the native relative field.

### Control B: Within-Prompt Empirical-Write Shuffle

Repeat Control A while permuting writes only among positions in the same prompt.
This preserves prompt topic, style, trajectory, and local norm statistics while
breaking token-local pairing.

Compute:

```text
G_within_prompt_self
G_within_prompt_native_basis
```

### Control C: Isotropic Norm-Matched Writes

For each receiving token:

```python
g = unit(center(random_gaussian(width)))
w_isotropic = write_norm[i] * g
v_isotropic = unit(x[i] + w_isotropic)
```

Fit on one isotropic train draw and evaluate on an independent isotropic test draw.
Compute self and fixed-native-basis metrics for both representations.

### Descriptive Raw-Write Baseline

Fit a prompt-balanced rank-32 basis on native unit centered writes:

```python
w_hat = unit(w)
G_write_native = held_out_prompt_balanced_energy(P_write, w_hat_test)
```

This records ambient write structure but does not determine classification because
raw-write and tangent organizations need not have equal scientific meaning. Do not
call a larger tangent score proof that tangent coordinates contain more information.

## Empirical Statistics

For every representation, control family, evaluation mode, and metric, serialize
all 99 values and report:

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

For a control distribution `C` and native score `G`:

```python
p_C = (
    1 + count(C >= G)
) / (1 + control_repeats)
```

Do not use tokens as independent statistical samples.

## Mechanical Outcomes

All classification predicates apply separately to both co-primary
representations. No rank, layer, corpus, anchor, or control may be selected after
inspection.

### Relative-Geometry Null Rejected

Classify as:

```text
relative_geometry_null_rejected
```

only if, for both ambient and transported representations:

```python
G_native > global_self_p95
p_global_self <= 0.05

G_native > global_native_basis_p95
p_global_native_basis <= 0.05

G_native > within_prompt_self_p95
p_within_prompt_self <= 0.05

G_native > within_prompt_native_basis_p95
p_within_prompt_native_basis <= 0.05

G_native > isotropic_self_p95
p_isotropic_self <= 0.05

G_native > isotropic_native_basis_p95
p_isotropic_native_basis <= 0.05
```

### Prompt-Conditioned Relative Structure

Classify as:

```text
prompt_conditioned_relative_structure
```

only if both representations pass every global and isotropic predicate above but
either representation fails a within-prompt predicate.

This outcome says global pairing matters but token-local organization was not
distinguished from prompt-level structure.

### Structured Addition Not Distinguished

Classify as:

```text
structured_addition_not_distinguished
```

if both native representations exceed their isotropic self controls, but either
representation fails a global empirical-write predicate.

### No Detected Low-Rank Relative Geometry

Classify as:

```text
no_detected_low_rank_relative_geometry
```

if either native representation does not exceed its isotropic self 95th percentile.

### Ambiguous

Every other valid result is:

```text
ambiguous
```

The executable must mechanically serialize every operand and Boolean predicate.

## Claims Permitted

### Null-Rejection Maximum Claim

```text
At layer 15 on the frozen held-out corpus, native residual/write pairing produced
rank-32 canonical tangent organization in both ambient and common-reference
coordinates that exceeded complete global, within-prompt, and isotropic finite-add
controls. This rejects the tested incidental-geometry null and supports continued
investigation of relative coordinates.
```

### Prompt-Conditioned Maximum Claim

```text
Native relative organization exceeded global and isotropic controls but was not
distinguished from prompt-preserving shuffles. The detected structure may be
prompt-conditioned rather than token-local.
```

### Structured-Addition Maximum Claim

```text
Relative tangent organization was detected above isotropic controls but was not
distinguished from the organization produced by shuffled structured empirical
writes.
```

### Negative Maximum Claim

```text
No held-out rank-32 relative tangent organization above the complete isotropic
finite-addition null was detected in both pre-registered coordinate descriptions at
this layer and corpus.
```

A negative result permits deprioritizing this specific class:

```text
held-out-stable low-rank canonical tangent-direction fields detectable in ambient
and shortest-geodesic common-reference coordinates at rank 32
```

It does not rule out nonlinear, high-rank, content-conditioned, token-specific, or
cross-layer geometric codes.

### Forbidden Claims

No outcome establishes:

- semantic content;
- that `time.out` is pure memory;
- that memory is encoded in a rotor;
- FFN consumption or causal use;
- a valid logit lens;
- an output-embedding map;
- a depth trend;
- superiority of rotor coordinates over all additive representations.

## Numerical Validity And Fail-Closed Criteria

Require and serialize:

```text
source artifact paths and cryptographic hashes
layer equals 15
corpus and manifest hashes match frozen registration
no prompt overlap with prohibited prior corpora
no train/test prompt overlap
minimum token and prompt counts pass
exclusions <= 1% in each split and control replicate
all raw vectors share one width and are finite
max residual identity error <= 1e-3
all unit norm errors <= 1e-5
max tangent orthogonality error <= 1e-5
max finite rotor reconstruction error <= 1e-5
reference norm and antipode margin pass
all transport tangent and norm checks <= 1e-5
all basis orthogonality errors <= 1e-5
all projected energies are finite and within [0, 1 + 1e-5]
all 99 sub-seeds per family are distinct
all permutations stay within their declared split/group
all required control rows and summaries are serialized
classification independently recomputes from serialized values
```

Any absent or failed criterion sets status `invalid`, suppresses classification,
and records the failure.

## Implementation Handoff

Directory:

```text
pocs/interp/experiments/tmix_relative_geometry_null/
```

Suggested source:

```text
rwkv_tmix_relative_geometry_null.cpp
```

Suggested target:

```text
llama-rwkv-tmix-relative-geometry-null
```

Required CLI:

```text
--input-root CAPTURE.root
--corpus CORPUS.txt
--manifest SPLIT.json
--registration REGISTRATION.json
--output-root RESULT.root
--output-json RESULT.json
--seed 817263
```

Optional development-only CLI:

```text
--development-run
```

`--development-run` must force status `development_only`, suppress classification,
and record that the corpus is ineligible. The classified path must reject known
prior corpus hashes.

Reject unknown arguments and existing output paths. An explicit `--overwrite`
must be recorded as a protocol deviation and cannot produce the canonical
classified artifact.

The analysis target must use only C++, ROOT RNTuples, and the required linear
algebra library. It must not link `llama`, `llama-common`, or execute GGML.

## ROOT Output Contract

Write:

```text
metadata
per_prompt_metrics
aggregate_metrics
control_values
numerical_checks
outcome_predicates
```

Every control row must include:

```text
representation: ambient | transported
evaluation_mode: self | native_basis
control_family: global_shuffle | within_prompt_shuffle | isotropic_write
replicate
sub_seed
metric_name
metric_value
```

Commit clusters regularly and close every writer before closing its `TFile`.

## JSON Output Contract

The compact review artifact must include:

```text
schema version
status and classification
confirmatory/development flag
scientific question and maximum permitted claim
all input paths, hashes, provenance, and frozen parameters
sample, prompt, and exclusion counts by split
reference construction and antipode margin
all numerical checks with threshold, observed value, and pass/fail
native ambient and transported metrics
raw-write descriptive metric
complete summaries for every control/evaluation/representation
all empirical p-values
every mechanical outcome operand and predicate
protocol deviations
ROOT artifact path and RNTuple row counts
```

Do not omit unfavorable metrics.

## Synthetic Self-Tests

Before opening real captures, test:

1. Exact finite tangent extraction and rotor reconstruction.
2. Parallel transport tangent membership and norm preservation.
3. A fixed low-rank native tangent field that survives train/test and is destroyed
   by pairing shuffle.
4. Structured ambient writes whose shuffled endpoints retain tangent organization,
   producing `structured_addition_not_distinguished`.
5. Isotropic writes producing `no_detected_low_rank_relative_geometry`.
6. A prompt-level field that survives within-prompt shuffle and produces
   `prompt_conditioned_relative_structure`.
7. Near-zero-angle and near-antipode exclusions.
8. Prompt-balanced basis fitting and metrics with unequal prompt lengths.
9. Global and within-prompt permutation isolation.
10. Empirical summaries, p-values, and complete classification truth table.
11. Missing control names, rows, fields, or predicates fail closed.

The classified executable path must run these tests before opening the input ROOT.

## Required Verification

1. Build and run the analysis self-tests.
2. Run a development-only analysis on an ineligible existing artifact.
3. Programmatically reopen ROOT and JSON outputs and validate every schema, field,
   row count, finite value, control count, and sub-seed count.
4. Independently recompute summaries, p-values, predicates, and classification from
   serialized ROOT rows.
5. Freeze and hash a new corpus and split before collecting classified activations.
6. Run the backend verifier, collect the three production taps, and record the
   verifier artifact.
7. Run the classified analysis once to new canonical paths.
8. Repeat the complete post-write audit.
9. Run `git diff --check`.

Do not write `RESULTS.md` or update the investigation ledger until all classified
artifact checks pass.
