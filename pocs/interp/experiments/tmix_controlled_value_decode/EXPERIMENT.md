# TMix Controlled-Value Coordinate Decoding Experiment

## Status

This is a discussed and admitted implementation handoff. It has not been
implemented or run.

This experiment follows the audited relative-geometry null rejection. It does not
transplant activations or edit recurrent state. It uses a controlled passive task
with the same read token at the same token position in every prompt.

Read before implementation:

- `../../../../AGENTS.md`
- `../../../../SCIENTIFIC_METHOD_PROMPT.md`
- `../../logs/TMIX_INVESTIGATION.md`
- `../tmix_relative_geometry_null/EXPERIMENT.md`
- `../tmix_relative_geometry_null/RESULTS.md`
- `../tmix_relative_geometry_null/LAYER_FOLLOWUP_RESULTS.md`

## Scientific Goal

The eventual goal is to determine what RWKV recurrence contributes to the residual
stream and whether that contribution admits a validated decoding map before the FFN
turns it into a learned policy response.

Prior work established that native `resid.in` / `time.out` pairing creates stable
relative tangent organization beyond shuffled empirical writes at sampled layers.
That does not establish that the relative geometry carries memory content.

This experiment introduces independently known content through a controlled prompt
factorial and asks which layer-local coordinate description makes that content more
linearly recoverable across held-out residual carriers.

## Scientific Question

At an identical read token and token position, after accounting for value
information already linearly decodable from `resid.in`, is a previously presented
controlled value more stably recoverable across held-out carrier contexts from:

```text
the centered raw TMix update
or
the finite TMix-induced relative log-tangent coordinates?
```

The primary comparison is equal-capacity carrier-held-out linear decoding. It is
not a direct final-head lens and does not interpret model logits as memory content.

## Observation Already Established

On a frozen Project Gutenberg corpus, native TMix pairings at layers 15, 30, 45,
and 60 produced rank-32 canonical tangent organization in ambient and
parallel-transported coordinates above global, within-prompt, and isotropic finite
addition controls.

This supports continuing to investigate relative coordinates. It does not show
whether the organized directions encode a remembered item rather than current-token
or generic contextual structure.

## Competing Interpretations

### H_relative: Relative Coordinates Expose The Controlled Value

```text
Conditional on the incoming residual representation, finite relative TMix
coordinates make the previously presented value more linearly recoverable across
held-out carriers than an equal-capacity raw-update representation.
```

### H_additive: Raw Update Coordinates Are At Least As Useful

```text
Conditional on the incoming residual representation, the centered raw TMix update
matches or exceeds relative coordinates for held-out value decoding.

The observed relative geometry may therefore be a transformation of ordinary
context-conditioned addition rather than a more useful content coordinate system.
```

### H_no_increment: No Detected Layer-Local Increment

```text
Neither raw-update nor relative coordinates improve held-out decoding over the
incoming residual baseline at the tested layer and capacity.
```

### H_holistic: Endpoint Is Useful But Neither Isolated Coordinate Is

```text
The normalized post-TMix endpoint improves value decoding, but neither the isolated
raw update nor the isolated relative coordinate provides a stable incremental
linear representation under the matched comparison.
```

## Shared Predictions And Non-Evidence

All hypotheses permit:

- the model to predict some controlled values correctly;
- `resid.in` to contain value information from lower-layer recurrence;
- raw updates and tangents to be structured;
- different values to produce different activations;
- the FFN or final logits to respond differently.

Those observations do not distinguish the coordinate hypotheses.

## Why A Positive Is Not Guaranteed

The rotor and tangent are deterministic functions of the incoming residual and raw
update. Re-expressing an update does not guarantee improved held-out linear
decoding. The relative hypothesis predicts a specific non-guaranteed advantage:
after matching decoder capacity and conditioning on incoming coordinates, relative
features should generalize across carrier contexts better than raw-update features.

## Interpretation Boundary

Earlier value tokens can influence a later RWKV read position only through the
model's recurrent mechanisms, including distributed layer states and token-shift
state. However, this experiment does not isolate a single recurrent state, memory
item, layer, or native state write.

The permitted object is therefore:

```text
a controlled value carried through RWKV recurrence and represented at a matched
read position
```

Do not call `time.out` pure memory or attribute the controlled value to one layer.

## Fixed Model And Backend

Use the exact model checkpoint from the classified relative-geometry run:

```text
rwkv7-g1h_preview4743-13.3b-20260703-ctx8192-Q4_K_M.gguf
```

Freeze and serialize its cryptographic hash before corpus generation. Use Vulkan
with `-ngl 99`, and run `llama-rwkv-interp-verify` on that exact model/backend
before collection.

Any model or backend change requires a new registration and cannot use the
canonical classified artifact path.

## Fixed Layers And Capacity

```python
layers = [15, 30, 45, 60]
representation_rank = 32
value_count = 16
train_carrier_count = 32
test_carrier_count = 16
seed = 941731
```

All four layers are pre-registered factors. Do not select the best layer after
inspection. Layer-specific outcomes and the across-layer family correction must
both be serialized.

## Controlled Full-Factorial Corpus

Use disjoint frozen sets:

```text
16 value tokens
48 carrier tokens
```

Every value and carrier replacement must tokenize as exactly one token in its
template position. Value and carrier token IDs must be disjoint.

The conceptual template is:

```text
Context: CARRIER. Remember: the value is VALUE. Query: the value is
```

The exact bytes may be adjusted before freezing only to satisfy tokenizer and
position constraints. After freezing, no wording may change.

Generate the complete factorial:

```python
for carrier in carriers:
    for value in values:
        emit(template.replace("CARRIER", carrier).replace("VALUE", value))
```

This gives:

```text
32 train carriers * 16 values = 512 train prompts
16 test carriers  * 16 values = 256 test prompts
```

Split by carrier. Every value appears exactly once in every carrier and in both
splits. Never split individual prompts independently.

## Exact Matching Requirements

Before model execution, tokenize every prompt and fail unless:

```text
all prompts have exactly the same token count
the final read token has one identical token ID in every prompt
the final read token has one identical position in every prompt
the VALUE replacement occupies one identical position and one token in every prompt
the CARRIER replacement occupies one identical position and one token in every prompt
no value token ID appears in the fixed template outside VALUE
no carrier token ID appears in the fixed template outside CARRIER
value and carrier token sets are disjoint
each carrier has all 16 values exactly once
```

The classified read position is the final token of the prompt. Record its token ID,
text, and zero-based position in registration metadata.

Do not use padding to repair token lengths. Reject unsuitable value or carrier
candidates before freezing.

## Value And Carrier Selection

Selection must occur without inspecting experiment activations or representation
decoding results.

Use a fixed candidate list committed with the experiment. Apply only mechanical
filters:

```text
single-token substitution in the exact template
ordinary non-special token
printable stable token text
disjoint value/carrier token IDs
no template collision
```

Choose the first passing candidates in deterministic seeded order. Do not select
tokens based on model retrieval accuracy, logits, activation geometry, semantic
category, or decoder performance.

Freeze:

```text
candidate-list bytes and hash
selected value strings and token IDs
selected carrier strings and token IDs
exact generated corpus and hash
carrier split and manifest hash
model hash
seed
template bytes
read/value/carrier positions
```

## Capture Procedure

Decode every prompt from a fresh native recurrent state. At the final read token,
capture F32 production snapshots at each fixed layer:

```text
rwkv.layer.<L>.resid.in
rwkv.layer.<L>.time.out
rwkv.layer.<L>.resid.time
```

Every host-readable tensor must be an explicit `ggml_dup` graph-output snapshot.
Do not reconstruct RWKV or LayerNorm.

Also record the native final logits needed only for behavioral diagnostics:

```text
expected value token log probability
expected value token rank
native top-1 token ID
whether top-1 equals the controlled value
```

Do not filter, weight, or select samples based on model behavior.

## Stored Sample Schema

One row per prompt and layer:

```text
prompt_id
carrier_id
carrier_token_id
carrier_split: train | test
value_id
value_token_id
layer
read_position
read_token_id

resid_in
time_out
resid_time

expected_value_log_probability
expected_value_rank
native_top1_token_id
native_top1_correct

residual identity and capture sanity metrics
```

Persist captures in a ROOT RNTuple and commit clusters regularly.

## Per-Sample Representations

Use double-precision accumulation:

```python
def center(x):
    return x - mean(x)

def unit(x):
    return x / norm(x)

x = center(resid_in)
w = center(time_out)
y = center(resid_time)

radius = norm(x)
u = unit(x)
v = unit(y)

w_relative_scale = w / radius

cos_theta = clamp(dot(u, v), -1, 1)
theta = acos(cos_theta)
t = unit(v - cos_theta * u)
q_ambient = theta * t
```

`w_relative_scale` and `q_ambient` are both dimensionless width-sized vectors and
retain their native finite magnitudes.

Construct the train-carrier-only prompt-balanced reference and parallel transport
exactly as in the relative-geometry null experiment:

```python
q_transport = transport_to_reference(u, q_ambient, reference)
```

Parallel transport preserves the norm `theta`; do not renormalize away the angle.
Use the same train-fitted reference for train and held-out carriers at one layer.
Each layer has its own train-only reference.

Required representations:

```text
incoming:             u
raw_update:           w_relative_scale
relative_ambient:     q_ambient
relative_transported: q_transport
endpoint:             v
```

Reject a sample from one layer if a required norm, angle, transport denominator, or
representation is non-finite or below the registered geometry threshold. Any
exclusion breaks the complete factorial and makes the classified run invalid.

## Train-Only Feature Construction

At each layer and representation independently:

1. Compute the feature mean using train carriers only, with each carrier weighted
   equally.
2. Center train and test rows by that train mean.
3. Fit a rank-32 right-singular-vector basis on train rows, weighting every carrier
   equally.
4. Project train and test rows into 32 coordinates.
5. Standardize each coordinate using train-only mean and standard deviation.

Fail if a retained coordinate has train standard deviation below `1e-12`.

Never use value labels, test carriers, model correctness, or final logits when
fitting feature bases or normalization.

## Equal-Capacity Decoder Models

Use one multinomial linear softmax decoder family with L2 regularization.

At every layer fit:

```text
M_incoming:
    32 incoming coordinates

M_additive:
    32 incoming + 32 raw-update coordinates

M_relative_ambient:
    32 incoming + 32 ambient-log-tangent coordinates

M_relative_transport:
    32 incoming + 32 transported-log-tangent coordinates

M_endpoint:
    32 endpoint coordinates
```

`M_additive`, `M_relative_ambient`, and `M_relative_transport` have exactly equal
input dimensionality and decoder parameter count. Initialize and optimize them with
the same deterministic procedure and convergence criteria.

The endpoint model is a descriptive holistic baseline and cannot determine the
raw-versus-relative classification.

## Regularization Selection

Use the same fixed grid for every model:

```python
lambda_grid = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
```

Select lambda independently per model using leave-one-train-carrier-out
cross-validation. The selection metric is mean carrier-balanced cross-entropy.

Within each cross-validation fold, refit the feature mean, rank-32 basis,
standardization, and decoder using only the remaining train carriers. Do not fit an
unsupervised basis once on all train carriers and reuse it inside held-out-carrier
lambda selection. After selecting lambda, refit features and the decoder on all
train carriers for final test-carrier evaluation.

Tie-break toward the larger lambda. Serialize every candidate score and selected
lambda. Never inspect test carriers during selection.

The optimizer must pass synthetic convergence and known-solution tests. Serialize
final gradient norm, iteration count, and convergence status for every fit. Any
failed fit invalidates that layer's classified comparison; a failed pre-registered
layer makes the overall classified run invalid.

## Held-Out Metrics

For model `M` and held-out carrier `c`:

```python
CE[M, c] = mean(
    -log_probability_of_controlled_value
    for all 16 values under carrier c
)

accuracy[M, c] = mean(correct_value_prediction)
```

Aggregate carriers equally:

```python
CE[M] = mean(CE[M, c] for test carriers c)
accuracy[M] = mean(accuracy[M, c] for test carriers c)
```

Cross-entropy is primary. Accuracy is diagnostic.

Define per-carrier incremental gains:

```python
gain_add[c] = CE[M_incoming, c] - CE[M_additive, c]

gain_rel_ambient[c] = (
    CE[M_incoming, c] - CE[M_relative_ambient, c]
)

gain_rel_transport[c] = (
    CE[M_incoming, c] - CE[M_relative_transport, c]
)

gain_endpoint[c] = CE[M_incoming, c] - CE[M_endpoint, c]
```

Define relative advantages:

```python
advantage_ambient[c] = (
    CE[M_additive, c] - CE[M_relative_ambient, c]
)

advantage_transport[c] = (
    CE[M_additive, c] - CE[M_relative_transport, c]
)
```

Positive gain means improvement over incoming residual alone. Positive advantage
means the matched relative model outperforms the raw-update model.

## Carrier-Level Statistics

The independent evaluation unit is the held-out carrier, not the prompt.

For every gain and advantage, perform an exact one-sided sign-flip randomization
test over the 16 held-out carrier values:

```python
observed = mean(delta[c])

p = count(
    mean(sign[c] * delta[c]) >= observed
    for all 2 ** 16 sign assignments
) / (2 ** 16)
```

Also report the exact two-sided sign-flip interval or a deterministic
carrier-bootstrap 95% interval as a diagnostic. Do not use prompts as independent
replicates.

## Across-Layer Multiplicity

There are three pre-registered statistical families:

```text
relative versus additive advantages: 4 layers * 2 coordinates = 8
relative versus incoming gains:       4 layers * 2 coordinates = 8
additive versus incoming gains:       4 layers                 = 4
```

Apply Holm correction separately within each family at family-wise alpha `0.05`.
Serialize raw and adjusted p-values and each complete ordering. Endpoint-versus-
incoming gains are descriptive; also serialize their four-test Holm correction for
the `holistic_endpoint_only` label.

The classified relative-coordinate claim requires corrected positive gain and
corrected positive additive advantage at the same layer.

Do not select a layer using uncorrected results.

## Mechanical Layer Outcomes

Evaluate each layer mechanically.

### Relative Coordinate Advantage

```text
relative_coordinate_advantage
```

only if both relative representations satisfy:

```python
mean(gain_relative) > 0
holm_adjusted_p_gain_relative <= 0.05

mean(advantage_relative) > 0
holm_adjusted_p_advantage <= 0.05
```

### Coordinate-Dependent Relative Signal

```text
coordinate_dependent_relative_signal
```

if exactly one relative representation satisfies all positive-gain and corrected
advantage predicates.

### Additive Coordinates Match Or Win

```text
additive_coordinates_match_or_win
```

if:

```python
mean(gain_add) > 0
holm_adjusted_p_gain_add <= 0.05
```

and neither relative representation has a significant corrected positive advantage
over `M_additive`.

### No Detected Layer-Local Increment

```text
no_detected_layer_local_increment
```

if none of additive, ambient-relative, or transported-relative models has positive
gain over incoming with its family-wise Holm-adjusted `p <= 0.05`.

### Holistic Endpoint Only

```text
holistic_endpoint_only
```

if no incremental model passes but `M_endpoint` improves over `M_incoming` with
positive mean gain and its four-test Holm-adjusted `p <= 0.05`.

### Ambiguous

Every other valid layer result is:

```text
ambiguous
```

Serialize every operand and Boolean predicate. Define and implement a fixed
precedence so only one label is assigned:

```text
relative_coordinate_advantage
coordinate_dependent_relative_signal
additive_coordinates_match_or_win
holistic_endpoint_only
no_detected_layer_local_increment
ambiguous
```

## Overall Outcome

This experiment does not require one coordinate system to win at every layer.
Report all four layer outcomes.

Classify the experiment overall as:

```text
relative_decoding_supported
```

if at least one pre-registered layer has `relative_coordinate_advantage` after the
eight-comparison Holm correction.

Classify as:

```text
relative_decoding_not_supported
```

if no layer has either `relative_coordinate_advantage` or
`coordinate_dependent_relative_signal`, and at least one layer has
`additive_coordinates_match_or_win`.

Every other valid combination is:

```text
mixed_or_ambiguous
```

## Permitted Claims

### Relative-Decoding Maximum Claim

```text
At the specified layer(s), controlled values carried through RWKV recurrence were
more linearly decodable across held-out carrier tokens from incoming-plus-relative
TMix coordinates than from an equal-capacity incoming-plus-raw-update
representation. This supports relative coordinates as a candidate memory-decoding
representation.
```

### Additive Maximum Claim

```text
Under the matched linear-decoding comparison, raw-update coordinates matched or
exceeded the tested relative coordinates for controlled-value recovery.
```

### No-Increment Maximum Claim

```text
No incremental linear decoding of the controlled value beyond incoming residual
coordinates was detected from either tested TMix update representation at the
specified layer and capacity.
```

### Forbidden Claims

No outcome establishes:

- that one layer stores the complete value;
- that `time.out` is pure memory;
- that the FFN uses or decodes a rotor;
- a causal mechanism;
- semantic generalization beyond the frozen values;
- decoding of unseen value identities;
- a validated output-embedding map;
- that nonlinear decoders would give the same ordering;
- that all memory content uses one coordinate system.

## Behavioral Diagnostics

Report native model value-token cross-entropy, expected-token rank, and top-1
accuracy by carrier and value. These metrics answer whether the model behaviorally
retrieves the task values.

Do not:

- filter incorrect examples;
- choose values or carriers based on behavior;
- use native logits as decoder features;
- call native output probabilities a decoding of layer-local memory content.

If native behavior is at chance, representation results remain mechanically valid
but the maximum claim must say only "previously presented controlled value," not
"behaviorally retrieved value."

## Numerical And Data Validity

Require and serialize:

```text
exact model path and cryptographic hash match registration
backend verifier artifact and pass status
corpus, manifest, candidate-list, template, value, and carrier hashes match
all prompts have identical token count
read/value/carrier positions and token-width checks pass
complete 48-by-16 factorial exists exactly once
carrier-only train/test split isolation
all required layer/tap rows exist exactly once per prompt
max residual identity error <= 1e-3
all finite geometry checks pass
no classified sample exclusions
all feature bases are train-only, rank 32, and orthonormal within 1e-5
all coordinate standardizations use train-only statistics
all decoder dimensions and parameter counts match registered models
all optimizer fits converge and pass finite checks
all 16 held-out carrier metrics exist per model/layer
all exact sign-flip counts, raw p-values, Holm ordering, and adjusted p-values serialize
all outcome predicates and labels independently recompute from serialized rows
```

Any absent or failed requirement sets status `invalid` and suppresses scientific
classification.

## Scalability Contract

The implementation must never form a width-by-width covariance or rotor matrix.
Use width-by-rank bases and carrier-by-feature decoder matrices.

Collection captures only one read position per prompt at four fixed layers.
Analysis streams width-sized vectors from ROOT and persists compact projected
features and metrics. Long-running collection must commit ROOT clusters regularly.

## Implementation Handoff

Directory:

```text
pocs/interp/experiments/tmix_controlled_value_decode/
```

Suggested programs:

```text
rwkv_tmix_controlled_value_corpus.cpp
rwkv_tmix_controlled_value_collect.cpp
rwkv_tmix_controlled_value_decode.cpp
```

Suggested targets:

```text
llama-rwkv-tmix-controlled-value-corpus
llama-rwkv-tmix-controlled-value-collect
llama-rwkv-tmix-controlled-value-decode
```

Keep corpus generation, production capture, and analysis separate.

Required analysis CLI:

```text
--input-root CAPTURE.root
--registration REGISTRATION.json
--output-root RESULT.root
--output-json RESULT.json
--seed 941731
```

Development mode must force `status=development_only` and suppress classification.
Reject unknown arguments and existing classified output paths.

The analysis target must not link `llama`, `llama-common`, or execute GGML. The
collector must invoke the production graph through `llama_interp::runtime` and obey
the explicit-snapshot capture policy.

## Output Contract

The capture ROOT must contain:

```text
metadata
samples
behavior
```

The analysis ROOT must contain:

```text
metadata
feature_basis_checks
cross_validation_scores
decoder_fits
per_carrier_metrics
aggregate_metrics
statistical_tests
outcome_predicates
```

The compact JSON must include all frozen hashes, token-position checks, sample and
factorial counts, numerical checks, selected lambdas, convergence records,
layer/model cross-entropies and accuracies, per-carrier gains and advantages,
sign-flip and Holm results, every outcome predicate, permitted claim, protocol
deviations, ROOT paths, and exact RNTuple row counts.

Do not omit unfavorable models, layers, values, carriers, or behavior diagnostics.

## Synthetic Self-Tests

Before opening classified data, test:

1. Exact tokenizer-position/factorial validation on a miniature corpus.
2. Centering, finite rotor/log-tangent extraction, and transport norm preservation.
3. Prompt/carrier-balanced rank-32 basis fitting with unequal synthetic row order.
4. No train/test leakage in means, bases, standardization, or lambda selection.
5. Multinomial optimizer recovery on a known separable synthetic problem.
6. Optimizer non-convergence fails closed.
7. Equal feature dimensions and parameter counts for additive and relative models.
8. A raw-update-coded synthetic value produces `additive_coordinates_match_or_win`.
9. A carrier-relative synthetic value produces `relative_coordinate_advantage`.
10. A baseline-only synthetic value produces `no_detected_layer_local_increment`.
11. An endpoint-only synthetic value produces `holistic_endpoint_only`.
12. Exact sign-flip p-values for known carrier deltas.
13. Holm correction and all layer/overall classification truth tables.
14. Missing rows, duplicated factorial cells, wrong token positions, or hash mismatch
    fail closed.

The classified execution path must run these self-tests before opening input data.

## Required Verification Sequence

1. Build and run corpus-generator, collector, and analyzer self-tests.
2. Generate a development factorial from disjoint candidate tokens and validate
   exact positions without model execution.
3. Run a development-only capture/analysis to validate schemas; it cannot classify.
4. Freeze the exact model hash, candidate list, selected tokens, template, corpus,
   carrier split, registration, seed, and expected positions.
5. Run the backend verifier on the frozen model/backend.
6. Collect the classified production snapshots once to a new ROOT path.
7. Run the classified analyzer once to new ROOT/JSON paths.
8. Reopen both artifacts and programmatically validate all schemas and row counts.
9. Independently recompute per-carrier metrics, exact sign-flip tests, Holm
   adjustment, layer outcomes, and overall classification from serialized rows.
10. Run `git diff --check`.

Do not write `RESULTS.md` or update the investigation ledger until every classified
artifact and independent audit passes.
