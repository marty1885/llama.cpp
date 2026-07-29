# TMix One-Shot Native-Unrotation Generation Gate

## Status

This is a discussed and admitted single-layer causal falsification gate. The revised
generation protocol has not been implemented or run. Its primary outcome is
generated token sequences, not logits or activation response norms.

An earlier logit-based implementation and Run 1 are documented in
`RUN1_INVALID_REPORT.md`. Run 1 has no scientific classification: its random
controls violated the registered post-TMix mean requirement. The old executable and
artifacts do not implement this revised generation protocol and must not be reused
as its classified result.

Expansion to other values, layers, prompts, or repeated per-generation-token
ablation is not admitted unless this gate produces a positive or ambiguous result
and a follow-up is discussed.

Read before implementation:

- `../../../../AGENTS.md`
- `../../../../SCIENTIFIC_METHOD_PROMPT.md`
- `../../logs/TMIX_INVESTIGATION.md`
- `../tmix_controlled_value_decode/EXPERIMENT.md`
- `../tmix_controlled_value_decode/RESULTS.md`
- `RUN1_INVALID_REPORT.md`
- `../../rwkv_tmix_residual_rotation.cpp`
- `../../rwkv_tmix_norm_geometry.cpp`

## Scientific Question

At layer 45 and the fixed final query token of the frozen controlled-value task,
does exactly removing the native TMix directional movement produce a larger change
in deterministic generated text than equally large foreign empirical or random
endpoint rotations?

The practical stopping question is:

```text
Does one native unrotation at the query token make any observable difference to the
subsequent greedy completion, and is that difference special relative to matched
rotations?
```

## Observation Already Established

The frozen 48-carrier by 16-value factorial established:

- native controlled-value top-1 accuracy of `96.09375%`;
- stable native relative geometry at layers 15, 30, 45, and 60;
- `100%` held-out linear-decoder accuracy for every tested representation;
- a small relative-coordinate cross-entropy advantage only at layer 45.

These are passive observations. They do not show that the native directional
movement affects generated behavior.

## Relevant External Evidence

No direct mechanistic-interpretability study of a pretrained Gated DeltaNet was
located during protocol review. The closest relevant results are:

1. [Gated Delta Networks](https://arxiv.org/abs/2412.06464) describes a recurrent
   matrix-valued associative memory with complementary global decay and targeted
   delta-rule updates. Its evidence is architectural and behavioral, not a causal
   interpretation of residual-space rotations.
2. [Mechanistic evaluation of Transformers and state space
   models](https://arxiv.org/abs/2505.15105) finds that small trained DeltaNet and
   Mamba models usually perform associative recall through direct retrieval at the
   query token in one layer, rather than by storing the association at the value
   token as a two-layer Transformer induction mechanism. This supports a one-shot
   query-token intervention as the first gate.
3. [Understanding the Skill Gap in Recurrent Language
   Models](https://proceedings.mlr.press/v267/bick25a.html) finds that retrieval in
   pretrained SSM language models can concentrate in a small number of mixer heads
   or channels. This warns that a geometrically interesting layer need not be a
   behaviorally critical layer.

These results motivate intervention timing and strict behavioral measurement. They
do not transfer a Gated DeltaNet state interpretation to RWKV, establish layer 45
as a retrieval bottleneck, or identify the RWKV rotor as memory.

## Competing Hypotheses

### H0: No Native Directional Privilege

```text
Native unrotation either leaves the generated continuation unchanged or changes it
no earlier or more strongly than matched foreign empirical and random rotations.
```

### H1: Native Direction Is Behaviorally Privileged

```text
Native unrotation causes consistently earlier generated-token divergence than both
matched control families.
```

H1 is not guaranteed by `ADD + LayerNorm`. The architecture guarantees the native
angle and canonical inverse but does not guarantee any token change, nor that the
native inverse changes generation more than another endpoint displacement with the
same angle, mean, and radius.

## Frozen Scope

Reuse these existing classified artifacts without regeneration or selection:

```text
build/tmix-controlled-value-classified-prompts.txt
build/tmix-controlled-value-classified-manifest.json
build/tmix-controlled-value-classified-registration.json
```

Verify their hashes against the registration and verify the model hash. The fixed
model and backend are:

```text
rwkv7-g1h_preview4743-13.3b-20260703-ctx8192-Q4_K_M.gguf
Vulkan, -ngl 99
```

The first causal gate is deliberately narrow:

```python
layer = 45
value_id = 0
train_carriers = range(0, 32)
test_carriers = range(32, 48)
generated_token_budget = 64
decoding = "greedy_argmax"
foreign_controls_per_test_carrier = 3
random_controls_per_test_carrier = 3
seed = 315972
```

`value_id = 0` is the first manifest value, fixed mechanically rather than chosen
using accuracy, logits, geometry, or perturbation response. The independent units
are the 16 held-out carriers. The 32 train carriers provide same-value donor
directions only.

Layer 45 is the first gate because it was the only tested layer with a registered
relative-coordinate decoding advantage. A null result is therefore a practical
stopping result for the rotor-channel program, not proof that every RWKV layer or
all recurrence is behaviorally irrelevant.

## Production Execution

For each prompt:

1. Decode the prefix before the final query/read token and preserve its native
   recurrent state.
2. Run the final prompt token natively from that state and retain the resulting
   native state.
3. Replay the same final token from the identical pre-token state with one
   intervention and retain the resulting perturbed state.
4. Starting with each branch's own greedy next token, generate up to 64 tokens with
   no further intervention.
5. Stop a branch early at EOS. Do not sample, alter temperature, share later states,
   or teacher-force generated tokens between branches.

The intervention occurs once, at the final prompt token. Repeated ablation at every
generated token is outside scope because it compounds interventions and asks whether
the direction is continuously necessary rather than whether the query-token event
causally affects completion.

Required host-readable captures at the intervened token are explicit F32
`ggml_dup` graph-output snapshots:

```text
rwkv.layer.45.resid.in
rwkv.layer.45.time.out
rwkv.layer.45.resid.time
```

No logits, final-head readout, output-embedding lens, decoder, or generated-token
probabilities are collected or used. Run `llama-rwkv-interp-verify` on the exact
model/backend before the experiment to validate instrumentation and state replay.

## Native Geometry

For incoming residual `a` and native post-TMix residual `y` at the final prompt
token:

```python
def center(x):
    return x - mean(x)

def unit(x):
    return x / norm(x)

mu = mean(y)
rho = norm(center(y))
u = unit(center(a))
v = unit(center(y))
theta = acos(clamp(dot(u, v), -1, 1))
h_native = unit(u - dot(u, v) * v)
```

Reject the run before classification if any required value is non-finite, a norm is
below `1e-12`, or `theta < 1e-4` for any classified test prompt.

## Exact Production Perturbation

For a target centered direction `v_target`:

```python
y_target = mu + rho * v_target
time_out_target = y_target - resid_in
delta_time_out = time_out_target - time_out_native
```

Convert the complete correction to FP16 and inject it only at layer-45 `time.out`:

```cpp
runtime::add_head(-1, fp16(delta_time_out))
```

This intervention changes the layer-45 residual contribution at one token. It does
not edit the layer-45 recurrent state update already computed inside TMix, isolate a
memory item, or remove current-token computation. Its returned branch state may
differ downstream because later layers consume the perturbed residual at that token.

### Native Unrotation

```python
v_target = u
```

This removes the full native centered directional movement while preserving the
native post-TMix mean and centered radius up to measured injection error.

### Same-Value Foreign Controls

For donor train carrier `j` with the same fixed `value_id = 0`:

```python
h_j = unit(u_j - dot(u_j, v_j) * v_j)

h_foreign = unit(
    h_j
    - dot(h_j, v_i) / (1 + dot(v_j, v_i))
    * (v_j + v_i)
)

v_target = cos(theta_i) * v_i + sin(theta_i) * h_foreign
```

Select three distinct donors per test carrier deterministically from only the 32
same-value train prompts. Reject a donor if:

```python
1 + dot(v_j, v_i) < 1e-4
abs(dot(h_foreign, h_native)) > 0.999
```

Inability to construct all three donors invalidates the run.

### Random Tangent Controls

For each test carrier and control index:

```python
g = deterministic_gaussian(width, seed, carrier_id, control_index)
g_centered = center(g)
h_random = unit(g_centered - dot(g_centered, v) * v)
v_target = cos(theta) * v + sin(theta) * h_random
```

Centering the Gaussian before tangent projection is required because `v` lies on
the mean-zero LayerNorm sphere. Run 1 projected an unconstrained Gaussian only off
`v`, leaving a constant-vector component and invalidating 30 of 48 random controls.
This amendment ensures the requested random tangent lies in the mean-zero
hyperplane. Seed `315972` is fresh and was fixed as the previous seed plus one
before any generated continuation from this revised protocol was observed.

Reject and redraw deterministically when:

```python
abs(dot(h_random, h_native)) > 0.999
```

## Matched-Control And Numerical Checks

Use the actual captured perturbed endpoint for checks. Every classified
intervention must satisfy:

```python
abs(mean(y_actual) - mu) <= 2e-3

abs(norm(center(y_actual)) - rho) / max(rho, 1e-12) <= 2e-3

abs(angle(unit(center(y_actual)), v) - theta) <= 2e-3
```

Native unrotation must additionally satisfy:

```python
angle(unit(center(y_actual)), u) <= 2e-3
```

Every branch must have a valid returned state and a deterministic finite sequence
of token IDs ending at EOS or the 64-token budget. Any failure suppresses
classification. Serialize requested and actual endpoint distances and all
reconstruction errors.

## Calibration Gate

Before classified execution, use train carrier 0 with `value_id = 0` at layer 45:

```python
alpha = [0.0, 0.25, 0.5, 1.0]

v_target(alpha) = (
    cos(alpha * theta) * v
    + sin(alpha * theta) * h_native
)
```

Calibration is engineering-only and must report `calibration_only`. Require:

```text
two native 64-token greedy continuations are token-identical
native repeat capture cosine >= 0.99999
native repeat relative norm difference <= 1e-4
alpha=0 endpoint and 64-token continuation reproduce native
all alpha endpoints pass reconstruction checks
endpoint distance is non-decreasing with alpha, with 2e-3 slack
every branch returns a valid state and valid token sequence
```

The alpha continuations may be rendered for inspection, but calibration behavior
cannot select the layer, value, generation length, metric, controls, tolerances, or
classified decision rule.

## Primary Generated-Text Response

For native token sequence `s_native` and condition sequence `s_z`, each ending at
EOS or the 64-token budget:

```python
exact_match[z] = (s_z == s_native)

lcp[z] = length_of_longest_common_token_prefix(s_native, s_z)

divergence_score[z] = (
    0
    if exact_match[z]
    else 65 - lcp[z]
)
```

The score ranges from `0` for an identical continuation to `65` for divergence at
the first generated token. Earlier divergence is treated as a stronger observable
completion effect. Token IDs, not decoded string bytes, determine equality and
prefix length.

For each held-out carrier:

```python
foreign_score = mean(divergence_score over 3 foreign controls)
random_score = mean(divergence_score over 3 random controls)

delta_foreign = divergence_score[native_unrotation] - foreign_score
delta_random = divergence_score[native_unrotation] - random_score
```

Required descriptive diagnostics, none of which can override the primary rule:

```text
first divergent token index and token IDs
generated sequence length and EOS status
token-level Levenshtein distance from native
first generated token and whether it equals the controlled value
decoded native, native-unrotated, foreign-control, and random-control text
```

Do not apply semantic judges, embedding similarity, language-model scoring,
perplexity, or post-hoc judgments of text quality. A changed greedy continuation is
not automatically a broken continuation.

## Statistical Test

If any native-unrotated continuation differs from native, run an exact one-sided
sign-flip test for each control family over the 16 held-out carrier deltas:

```python
p = count(
    mean(sign[i] * delta[i]) >= mean(delta)
    for all 2 ** 16 sign assignments
) / (2 ** 16)
```

Apply Holm correction across the two primary p-values at family-wise alpha `0.05`.
The independent unit is the held-out carrier, not generated tokens or individual
controls.

## Mechanical Outcome

Classify:

```text
no_observable_generation_effect
```

if all 16 native-unrotated continuations are token-identical to their native
continuations. This is the immediate practical stopping outcome; no significance
test is needed.

Otherwise classify:

```text
native_rotation_generation_privilege_detected
```

only if both mean deltas are positive and both Holm-adjusted p-values are at most
`0.05`.

Classify:

```text
no_detected_native_rotation_privilege
```

if both mean deltas are non-positive. Every other valid outcome is:

```text
ambiguous
```

## Claims Permitted By Each Outcome

| Outcome | Maximum permitted claim |
| --- | --- |
| `no_observable_generation_effect` | At layer 45 for this fixed value and carrier panel, one native unrotation at the query token produced no change in the next 64 greedily generated tokens. Stop the rotor-channel program at its strongest current candidate. |
| `native_rotation_generation_privilege_detected` | At layer 45 for this fixed value and carrier panel, native unrotation caused earlier deterministic completion divergence than matched foreign and random rotations. The native direction was behaviorally special under this intervention. |
| `no_detected_native_rotation_privilege` | Generated changes from native unrotation were no stronger than both matched control families. Do not expand the rotor-channel program on this result. |
| `ambiguous` | The generation gate did not distinguish native directional privilege from generic perturbation sensitivity. |

No outcome establishes that the rotation is memory content, that RWKV implements a
rotor instead of addition and LayerNorm, that `time.out` is pure memory, or that a
tangent-to-output-embedding map is valid. A positive result can still mean that
unrotation destroys an ordinarily useful context-dependent additive update.

## Implementation Limits

Implement one small executable with `--self-test`, `--calibration`, and
`--classified` modes. Reuse:

```text
llama_interp::runtime::prefill_tokens
llama_interp::runtime::add_head
rwkv_experiment::require_capture
controlled-value registration and manifest validation patterns
existing deterministic state replay and greedy generation
existing ROOT RNTuple writer patterns
```

Do not add or modify `src/` instrumentation, recurrent-state editing, a decoder,
logit readout, generic perturbation framework, semantic evaluator, or shared
geometry library.

Suggested source and target:

```text
pocs/interp/experiments/tmix_causal_rotation_ablation/
    rwkv_tmix_causal_rotation_ablation.cpp

llama-rwkv-tmix-causal-rotation-ablation
```

## CLI

```text
--self-test

--calibration -m MODEL --prompts PROMPTS.txt --manifest MANIFEST.json
    --registration REGISTRATION.json --output-json FILE.json
    --seed 315972 -ngl 99

--classified -m MODEL --prompts PROMPTS.txt --manifest MANIFEST.json
    --registration REGISTRATION.json --output-root FILE.root
    --output-json FILE.json --seed 315972 -ngl 99
```

Reject unknown arguments, wrong hashes, wrong seed/backend, and existing output
paths. An overwrite cannot produce a classified result.

## Durable Output

The classified ROOT file must contain enough information to reproduce the generated
comparison without rerunning the model:

```text
metadata
native_samples
interventions
generated_tokens
carrier_metrics
statistical_tests
outcome_predicates
numerical_checks
```

Each intervention records prompt/carrier/value IDs, condition, control index, donor
or random seed, requested and actual angle, reconstruction errors, generated length,
EOS status, exact-match flag, longest common prefix, divergence score, edit distance,
and decoded continuation. `generated_tokens` stores every token ID and position for
every branch. Commit at least once per completed held-out carrier.

The JSON summary must include frozen hashes, fixed factors, all numerical checks,
all rendered continuations, per-carrier primary deltas, raw and adjusted p-values
when applicable, predicate operands, final classification, permitted claim, ROOT
row counts, and protocol deviations.

## Synthetic Self-Tests

Before model execution, test:

1. Centering, normalization, endpoint reconstruction, and native unrotation.
2. Parallel transport tangent membership and norm preservation.
3. Foreign and random endpoint angle matching, including zero mean of every random
   tangent and random target direction.
4. Deterministic same-value donor selection and train/test isolation.
5. Deterministic Gaussian controls and redraws.
6. EOS handling, token-sequence equality, longest common prefix, divergence score,
   and token edit distance.
7. Exact sign-flip p-values, Holm correction, and every outcome truth-table branch.
8. Missing rows, invalid states, generation failures, or failed numerical checks
   suppress classification.

## Required Verification Sequence

1. Build and run synthetic self-tests.
2. Run `llama-rwkv-interp-verify` on the exact model/backend.
3. Run calibration to a new JSON path and confirm deterministic native replay,
   alpha-zero continuation identity, and every numerical gate.
4. Run the classified layer-45/value-0 generation panel to new ROOT/JSON paths.
5. Reopen every RNTuple and verify schemas, row counts, finite geometry, donor
   isolation, control counts, generation lengths, EOS handling, and token rows.
6. Independently recompute exact matches, prefix lengths, divergence scores,
   carrier deltas, exact p-values, Holm correction, predicates, and classification
   from serialized rows.
7. Render all branch continuations side by side without using subjective quality
   judgments to change classification.
8. Run `git diff --check`.
9. Write `RESULTS.md` and update the ledger only after all checks pass.
