# TMix Investigation Ledger

This is an experiment ledger for the RWKV-7 TMix investigation. It records
measurements, assumptions, and open questions separately. Do not promote an
interpretation to a result without a discriminating production-graph test.

## Model And Runs

- Model: `rwkv7-g1h-13.3b-20260710-ctx10240-Q4_K_M.gguf`, Q4_K_M.
- Backend: Vulkan with `-ngl 99`.
- Main prompt: `The capital of France is`.
- Greedy continuation used in several runs:

  ```text
   Paris.
  The capital of Germany is Berlin.
  The
  ```

## Established

### Value time mix

The production graph computes the value input as:

```cpp
xv = att_norm + (previous_att_norm - att_norm) * value_lerp
```

Equivalently:

```cpp
xv = (1 - value_lerp) * att_norm + value_lerp * previous_att_norm
```

`xv` is a direct production tap at `rwkv.layer.<L>.time.xv`.

For the input token ` is`, direct final-head lenses of `xv` ranked ` Paris`
first at layers 55 and 60.

### Layer-60 TMix output cluster

Raw `time.out` was collected for all 61 layers over five prompt tokens and 32
greedy-generated tokens: 2,257 vectors total. Cosine k-means with eight
clusters found a compact high-norm cluster:

```text
count:       39
mean cosine: 0.918
mean norm:   207.55
centroid lens: Forum / Flush / Force family
members:     all 37 sampled layer-60 vectors, plus one layer-1 and one layer-4 vector
```

For layer 60 alone, the 37 `time.out` vectors have mean pairwise cosine
`0.929`. This is a real, coherent production activation pattern, not merely
repeated top-logit labels.

### Layer-60 head structure

Layer 60 has 64 RWKV heads of width 64. Across the same 37 positions:

```text
pre_output mean pairwise cosine: 0.819
time.out mean pairwise cosine:   0.929
```

The leading `pre_output` head blocks are:

```text
head 11: 34.9% norm fraction, directional coherence 0.953
head 30: 16.5% norm fraction, directional coherence 0.983
head 19: 13.6% norm fraction, directional coherence 0.898
head 45:  9.9% norm fraction, directional coherence 0.930
```

Heads 11, 30, 19, and 45 account for about 75% of the pre-output norm.
`W_O` makes the already coherent pre-output pattern more coherent in
residual-width `time.out` space.

### Normalization geometry at layer 60

For the ` is` probe token, a unit direction was estimated from normalized
layer-60 `time.out` vectors. Its projected component was removed at the
production `time.out` tap, and the native versus ablated runs captured
`resid.time`, `ffn.norm`, `channel.out`, and `resid.out`.

The stable component was decomposed relative to native `resid.time` into:

```text
mean:    1.6% norm fraction
radial: 34.0% norm fraction
tangent:94.0% norm fraction
```

These are orthogonal-component norm fractions, not additive percentages.

Observed effects:

```text
mean removal:
  ffn.norm change: effectively zero
  max final-logit change: 0.000004

radial removal:
  ffn.norm change: effectively zero
  max final-logit change: 31.40
  greedy token: Paris -> Franc

tangent removal:
  ffn.norm relative change: 0.364
  max final-logit change: 19.55
  greedy token remains: Paris
```

Thus, relative to the current `resid.time`, the mean component is
normalization-null, the radial component is nearly invisible to `ffn.norm`,
and the tangent component changes the normalized FFN input substantially.

### Small-scale geometry replication

The first dedicated geometry-only run used target-layer-60 `time.out`
directions from the 20 other positions in the same prompt and greedy
continuation, excluding the probe token ` is`. It captured only production
`time.out`, `resid.time`, and `ffn.norm`. It did not perform a vocabulary
lens, a feature analysis, or a logit measurement.

Artifact:

```text
/tmp/opencode/rwkv-tmix-norm-geometry-small-scale-layer60-g1h-13b.json
```

For signed scales `-0.25`, `-0.125`, `0.125`, and `0.25`:

```text
component   applied time.out norm at |scale|=0.125   ||delta ffn.norm||
mean        0.444                                      0.000017 to 0.000019
radial      9.547                                      0.000907 to 0.000970
tangent    27.816                                      9.724 to 9.756
```

At `|scale| = 0.25`:

```text
component   applied time.out norm   ||delta ffn.norm||
mean        0.887                   0.000014 to 0.000021
radial     19.094                   0.001792 to 0.002010
tangent    55.632                  19.342 to 19.469
```

For the tangent component, opposite signed interventions produced
`ffn.norm` changes with cosine `-0.994` at magnitude `0.125` and `-0.978` at
magnitude `0.25`. Doubling the scale had relative error about `0.053` from
the doubled small-scale response for either sign. The radial response also
had near-opposite signed direction, but was four orders of magnitude smaller
in norm.

This is one prompt/trajectory run. The direction samples are not an
independent corpus: generated samples occur after the probe token in the same
trajectory. It establishes the local production-graph response for this
probe, not context-independent behavior or direction specificity.

### Layer-local WKV state necessity test

The layer-60 recurrent `s` state is passed directly to `ggml_rwkv_wkv7` as
`wkv_state`. A paired production-graph run held the input token, every token
shift (`r`) state, every other layer's `s` state, and all weights fixed. It
replaced only layer 60's native `s` vector with zeros.

Artifact:

```text
/tmp/opencode/rwkv-tmix-memory-state-layer60-g1h-13b.json
```

This is an off-manifold zero-state necessity baseline. It measures dependence
on that stored state relative to zero; it does not recover semantic memory
content or define an additive memory decomposition.

For the ` is` probe:

```text
tap                         ||native - zero-state||   native/zero cosine
time.wkv                    261.045                   0.89710
time.rkv                     30.231                   0.86365
time.pre_output              63.328                   0.96473
time.out                     35.533                   0.98892
resid.time                   35.533                   0.99561
ffn.norm                     10.962                   0.99846
```

The shared layer-60 `time.out` direction was largely retained under this
baseline:

```text
                             native       zero-state
shared-direction projection  235.304      227.594
tangent norm                 222.535      216.638
```

Native and zero-state tangents had cosine `0.99935`; their difference norm was
`9.886`.

For this probe and baseline, the stored WKV state strongly affects the raw WKV
output but is not necessary for most of the measured coherent final `time.out`
direction. This does not establish that current-token computation is the sole
source: group normalization, the direct `v * rk` branch, gating, and `W_O`
remain on the native path and can interact with the state-dependent change.

### Normalized WKV local logit lens

`time.wkv_norm` is the production tap after WKV group normalization and before
the direct `v * rk` branch. A symmetric finite-difference JVP was measured for
the unit direction of the native `time.wkv_norm` activation:

```text
d(final logits) / d(time.wkv_norm) @ unit(time.wkv_norm)
```

Each evaluation perturbed the named `time.wkv_norm` tap in the production
graph, ran the remaining gate, output projection, FFN, and final output path,
and read logits from the resulting actual final residual. This is a local,
context-conditioned logit lens, not a standalone-vector final-head lens.

Artifact:

```text
/tmp/opencode/rwkv-tmix-wkv-jvp-lens-layer60-g1h-13b.json
```

The native normalized-WKV norm was `54.7265`. Symmetric perturbations at
`0.5%`, `1%`, and `2%` of that norm applied `0.2736`, `0.5472`, and `1.0945`
activation norms, respectively. Adjacent estimated logit-derivative vectors
had cosine `0.9852` and `0.9895`.

At the middle scale, the most negative local derivatives were:

```text
Focus  -1.0098    Forms  -0.9995    Force  -0.9985
Floor  -0.9860    Float  -0.9817    Found  -0.9765
Forum  -0.9699    Flush  -0.9480
```

` Paris` was a positive derivative (`+0.1229` per unit) but was not among the
largest positive values. The positive list was otherwise heterogeneous.

This does not say that the normalized WKV branch has the meaning of the listed
tokens, nor can the derivative be multiplied by the full branch norm to infer
a finite contribution. It establishes only the local token-logit direction of
this branch at this exact context. The branch also depends on both the prior
WKV state and current token projections; it is not a pure prior-memory vector.

### Normalized WKV local logit lens corpus

The local lens was extended to 16 independently prefixed, hand-audited prompts
in `pocs/interp/tmix_wkv_jvp_corpus.txt`. For each final prompt token, it used
a symmetric `1%` perturbation in the native `time.wkv_norm` direction. The
single-context scale sweep above remains the only scale-convergence check; this
corpus run checks cross-context agreement at that calibrated middle scale.

Artifact:

```text
/tmp/opencode/rwkv-tmix-wkv-jvp-corpus-layer60-g1h-13b.json
```

The mean pairwise cosine between the 16 full-vocabulary JVP vectors was
`0.76875`. The corpus-mean most negative derivatives were:

```text
Focus  -0.5262    Forms  -0.5142    Force  -0.5142
Found  -0.5084    Floor  -0.5079    Float  -0.5043
HTTPS  -0.5024    Hello  -0.4981    Forum  -0.4895
```

This negative family recurred in most contexts. Some context-specific expected
tokens also appeared among positive derivatives: ` Tokyo` for the Japan prompt
and ` Jupiter` for the planet prompt. The aggregate positive list was otherwise
heterogeneous and multilingual.

The result is not uniform. The Italy-capital context had JVP cosine `0.1536`
to the corpus mean and placed `Flush`, `Forum`, and related tokens among its
strongest positive derivatives. The story-opening context also placed that
family among its strongest positive derivatives despite having cosine `0.7403`
to the corpus mean. These exceptions rule out a claim that the normalized WKV
branch always suppresses the family.

The narrow observation is that the native normalized-WKV branch has a broadly
recurrent, but context-modulated, local logit direction. The current experiment
does not identify whether the recurrence originates in prior WKV state, current
projections, or their interaction.

### All-layer normalized WKV local logit lens

The normalized-WKV JVP was evaluated at every layer on one 165-token
calibration prompt containing capitals, simple facts, ordinary sentences, and a
final `The capital of France is` probe. For each layer, a `1%` symmetric
perturbation of that layer's native `time.wkv_norm` direction propagated through
all remaining production layers to the actual final residual and logits.

Full-vocabulary JVPs were written one layer at a time to a ROOT `TTree` rather
than retained in memory. The output has `metadata` and `rwkv_wkv_jvp` trees;
each latter row contains one layer's `logit_jvp`, top token IDs, and tracked
token derivatives.

Artifacts:

```text
build/rwkv-tmix-wkv-jvp-layers-calibration-g1h-13b.root
/tmp/opencode/rwkv-tmix-wkv-jvp-layers-calibration-summary-g1h-13b.json
```

Layer 0's normalized-WKV native direction had strong positive local derivatives
for all tracked factual/capital tokens and strong negative derivatives for the
recurring family:

```text
Paris   +7.481    France  +7.842    Berlin  +8.483
Rome    +9.835    Tokyo   +9.661    Jupiter +7.839

Focus  -22.856    Forms  -25.256    Force  -25.173
Forum  -23.577    Flush  -24.973    Floor  -25.201
```

The pattern changes substantially by layer. At layer 8, the recurring family
has positive derivatives around `+13.5` to `+14.3`, while tracked factual tokens
are negative; at layer 10 the recurring family is again negative around `-6` to
`-7`. The layer-60 derivatives are small by comparison:

```text
Paris +0.166    France +0.152    Berlin +0.169
Rome  +0.159    Tokyo  +0.153    Jupiter +0.120
```

At layer 60, each tracked recurring-family derivative is within about `0.05` of
zero for this prompt.

This is one prompt and one local scale. It shows that related factual tokens do
appear in this logit-lens construction, especially at early layers, and that
the recurring token family has alternating local signs across depth. It does
not establish that a layer stores those facts, that the full finite activation
has these effects, or that the lens isolates prior WKV state from current
projections.

### FFN feature response

The tangent removal changes production channel-mix key features. The channel
path is:

```cpp
channel.xk
  -> channel.key_preact = W_channel_key @ channel.xk
  -> channel.key_relu_sq = relu(channel.key_preact)^2
  -> channel.out
```

Examples of large tangent-induced ReLU-squared changes for the ` is` probe:

```text
feature 5571: delta -44.73, native 26.77
feature 9428: delta -39.23, native 34.09
feature 3797: delta +39.10, native 52.24
feature 13422: delta +24.11, native 34.67
```

The tangent is therefore not a null direction. It changes the feature state
presented to the channel-mix output projection.

## Not Established

- The semantic or functional meaning of the layer-60 direction.
- Whether `Forum/Flush/Force/...` labels identify model content rather than an
  isolated-readout artifact.
- Whether the direction is a bias, calibration feature, formatting feature,
  memory feature, or another control feature.
- Whether the layer-60 mechanism generalizes to other layers.
- Whether the changed channel-mix feature IDs form an interpretable circuit.
- Which individual head or feature is causally necessary or sufficient for a
  downstream behavior.
- Whether memory-derived TMix updates form a coherent rotor/tangent content code.
- Whether a rotor/tangent representation can be transported reproducibly into the
  final normalized residual and output-embedding coordinates.
- Whether rotor coordinates generalize better than raw additive-update coordinates
  across controlled native writes, contexts, layers, or token positions.

## Current Geometry And Interpretation Boundary

The production graph executes:

```python
resid_time = resid_in + time_out
ffn_input = layer_norm(resid_time)
```

After centering and before LayerNorm's learned gain and bias, the incoming and
resulting directions define a canonical minimal rotor. Its oriented two-plane and
angle are the high-dimensional axis-like rotation object. A tangent vector is the
local logarithmic representation of that rotor.

This does not create two competing computational hypotheses. `ADD + LayerNorm` and
the rotor are two descriptions of the same composite operation. The scientific
alternatives concern representation:

```text
rotor-code hypothesis:
  memory content is organized and consumed coherently in the rotor plane, angle,
  or tangent coordinates induced by the native update.

incidental-geometry hypothesis:
  the rotor is only the unavoidable geometry of normalizing a contextual additive
  update and has no stable content map beyond the local execution.
```

Observing a rotor, angle, tangent, or normalized response cannot distinguish these
alternatives because those observations are guaranteed by the operation. A valid
test needs independently known content from native live-quantity writes, matched
mean/radius and random controls, causal use or recovery, and held-out contexts.

The exact LayerNorm geometry is projection into the mean-zero hyperplane,
normalization toward a sphere, and learned affine gain/bias. Define the rotor on
the centered pre-affine sphere. Learned gain maps this sphere to an ellipsoid. The
raw residual skip also remains downstream, so radius invariance is an empirical
question for the whole model rather than an architectural fact.

Relevant prior work:

- Paul M. Riechers, `Geometry and Dynamics of LayerNorm`, arXiv:2405.04134:
  exact projection, normalization, and affine hyperellipsoid geometry.
- Huadong Liao, `Transformer as an Euler Discretization of Score-based
  Variational Flow`, arXiv:2604.23740: residual plus RMSNorm as a tangent
  projection and relaxed retraction.
- Peter Racioppo, `The Transformer as a Polar State Estimator`,
  arXiv:2605.11007: radial/tangential updates and hyperspherical retraction.
- Kobayashi et al., EMNLP 2021, and Ferrando et al., EMNLP 2022 ALTI:
  LayerNorm-aware whole-block attribution, but not rotor-to-output transport.

No reviewed work has yet supplied the complete method needed here: exact
LayerNorm-aware native rotor extraction, validated tangent transport through an
ordinary pretrained model, and mapping into output-embedding coordinates.

The eventual decoding target must be upstream of or validated through the MLP.
Once a contribution passes through the FFN, its output includes the model's learned
response to that contribution rather than a policy-free readout of what memory
added. Directly applying the final head to `time.out` has produced incoherent output
and has no established interpretation. This failure motivates searching for the
representation actually presented to and consumed by the MLP; it does not by
itself prove that the representation is a rotor or destination code.

## Do Not Do

- Do not infer semantics from a standalone final-head lens of a non-final
  residual contribution.
- Do not call an inverse basis conversion a causal inverse or a recovered
  original activation.
- Do not call a raw norm ratio a percentage of information surviving.
- Do not generalize from one layer, prompt, or continuation.
- Do not treat top changed feature IDs as an explanation.
- Do not manually reconstruct RWKV computations. Use production graph captures
  and named-tap perturbations.
- Do not start a new experiment without writing its null hypothesis, control,
  and discriminating outcome first.
- Do not describe a state swap, state delta, rank approximation, or zero recurrent
  state as an isolated memory-item edit. The recurrent state is a tangled
  contextual superposition.
- Do not treat logits from a memory ablation as the content of memory. They include
  the trained model's policy response to the counterfactual.
- Do not claim that high-dimensional rotation lacks an axis-like object. Use the
  canonical minimal rotor's oriented two-plane and angle.
- Do not treat the existence of a rotor after `ADD + LayerNorm` as evidence that
  RWKV uses a rotor code; its existence is guaranteed mathematically.
- Do not run another broad layer or corpus sweep before a controlled,
  single-layer discriminating test passes.

## Current Question

```text
Does the held-out-stable destination produced by structured RWKV TMix additions
reflect learned use of `ADD + LayerNorm` as a rotation that marks information
contributed by memory for the following MLP, or is it only the incidental geometry
of normalizing a structured additive update?

What observation can discriminate those explanations without assuming that
`time.out` is already a directly lensable memory representation or treating the
MLP's learned response as the content that memory added?
```

## Experiment Status

The first distributional gate is specified in
`../experiments/TMIX_ROTOR_SUBSPACE_EXPERIMENT.md`, with completed preliminary runs summarized in
`TMIX_ROTOR_SUBSPACE_RESULTS.md`. No tested layer supported residual-relative
tangent organization. Layer 30 was consistent with stable ambient additive
organization; layer 45 was ambiguous; layer 60 had no detected rank-32 structure;
and exploratory layer 15 was ambiguous. These are not a pre-registered depth trend.

The destination gate specified in
`../experiments/TMIX_DESTINATION_SUBSPACE_EXPERIMENT.md` has been run at confirmatory
layer 30 and replicated across expanded text8 and non-Wikipedia Pile corpora. The
results are recorded in `TMIX_DESTINATION_SUBSPACE_RESULTS.md`. They support a
held-out-stable destination associated with structured empirical TMix additions,
but native token-local pairing specificity did not replicate at layer 30.

An explicitly exploratory Pile scan at layers 15, 30, 45, and 60 is recorded in
`TMIX_DESTINATION_SUBSPACE_PILE_EXPLORATORY_DEPTH_RESULTS.md`. It found stable
destination-like structure above isotropic controls at all sampled layers. Only
layer 15 exceeded the registered global pair-shuffle control; this does not
establish a depth trend or a memory interpretation.

The active hypothesis is that RWKV may use `ADD + LayerNorm` to rotate the residual
into an unused or less-used direction that marks information contributed by TMix,
with the following MLP resolving that information. The competing explanation is
that the geometry is only ordinary normalization of a structured additive update.
The immediate scientific task is to discriminate those explanations before trying
to decode memory content or construct an output-coordinate lens.

The admitted next necessary-condition test is specified in
`../experiments/tmix_destination_base_recovery/EXPERIMENT.md`. At fixed layer 30
and rank 32, it asks whether removing the native train-fitted destination from
held-out post-TMix directions recovers their paired incoming directions better and
more efficiently than complete shuffled empirical and isotropic control pipelines.
It uses the existing Pile ROOT capture only and does not run the model, inspect the
FFN, produce logits, or claim semantic recovery.

A post-hoc proposal to compare native and shuffled destination-coordinate moment
distributions was rejected before implementation. Better moment matching could
still arise from structured addition and would not establish that the model uses
the rotation. Do not revive that metric without first identifying a discriminating
rotor-use claim and matched alternative representation.

## Artifacts

- `/tmp/opencode/rwkv-tmix-clusters-g1h-13b.json`
- `/tmp/opencode/rwkv-tmix-head-geometry-layer60-g1h-13b.json`
- `/tmp/opencode/rwkv-tmix-norm-geometry-layer60-g1h-13b.json`
- `/tmp/opencode/rwkv-tmix-tangent-features-layer60-g1h-13b.json`
- `/tmp/opencode/rwkv-tmix-norm-geometry-small-scale-layer60-g1h-13b.json`
- `/tmp/opencode/rwkv-tmix-memory-state-layer60-g1h-13b.json`
- `/tmp/opencode/rwkv-tmix-wkv-jvp-lens-layer60-g1h-13b.json`
- `/tmp/opencode/rwkv-tmix-wkv-jvp-corpus-layer60-g1h-13b.json`
- `build/rwkv-tmix-wkv-jvp-layers-calibration-g1h-13b.root`
- `/tmp/opencode/rwkv-tmix-wkv-jvp-layers-calibration-summary-g1h-13b.json`

## Audited Destination Base-Recovery Reanalysis (Layer 30 Pile)

The admitted post-hoc destination base-recovery reanalysis has completed on the
fixed existing layer-30 Pile capture: rank 32, seed 45678, 99 repeats, and the
frozen 26/22 prompt split. The corrected run-7 artifact passed its synthetic
self-tests, source-metric reproduction, ROOT/JSON schema checks, exact control
row-shape checks, distinct sub-seed checks, and post-write control-derived
p-value recomputation.

Removing the native train-fitted destination did **not** recover the paired
held-out incoming direction: prompt-balanced angular-recovery fraction was
`-0.290191003138` and recovery efficiency was `-0.262182409117`. The registered
mechanical outcome is `exploratory_no_detected_base_recovery` because the native
fraction is negative. Although it was numerically above the negative control
95th percentiles, that does not override the required positive native recovery.

Permitted conclusion: no positive held-out recovery of incoming direction was
detected after removing this native destination at this layer, rank, and corpus.
This fails the simple separable-destination necessary condition; it does not
negate structured TMix addition, a holistic residual representation, memory
content, rotor coordinates, FFN use, or any causal mechanism.

Artifacts and detailed audit:

- `build/tmix-destination-base-recovery-layer30-pile10k-run7.root`
- `build/tmix-destination-base-recovery-layer30-pile10k-run7.json`
- `pocs/interp/experiments/tmix_destination_base_recovery/RUN7_AUDIT.md`

## Admitted Relative-Geometry Null Test

The next admitted gate is specified in
`../experiments/tmix_relative_geometry_null/EXPERIMENT.md`. It does not attempt to
prove or decode a memory rotor. At fixed exploratory-candidate layer 15 and rank 32,
on a newly frozen corpus, it asks whether native residual/write pairings produce
held-out-stable canonical tangent directions in both ambient and parallel-transported
coordinates beyond complete global, within-prompt, and isotropic finite-addition
pipelines.

The null is structured ambient addition with incidental `ADD + LayerNorm` geometry.
Rejecting it would support continuing to investigate relative coordinates. Failure
would permit deprioritizing only the pre-registered class of held-out-stable,
 low-rank tangent-direction fields in those two coordinate descriptions; it would
 not rule out nonlinear, high-rank, content-conditioned, token-specific, or
 cross-layer geometric codes.

## Classified Relative-Geometry Null Result

The admitted layer-15 relative-geometry null test completed on its newly frozen
Project Gutenberg ebook 73544 corpus. The production capture used the
`rwkv7-g1h_preview4743-13.3b-20260703-ctx8192-Q4_K_M.gguf` model on Vulkan
with `-ngl 99`; all three required F32 taps passed backend verification before
collection. The frozen registration, corpus, and manifest are in
`../experiments/tmix_relative_geometry_null/frozen/`.

The classified artifacts are:

```text
build/tmix-relative-geometry-null-classified-capture.root
build/tmix-relative-geometry-null-classified.root
build/tmix-relative-geometry-null-classified.json
```

The artifact status is `valid`; the mechanical classification is
`relative_geometry_null_rejected`. Native held-out energies were
`0.393052773619` in ambient coordinates and `0.389749780036` in transported
coordinates. Both co-primary representations exceeded all global,
within-prompt, and isotropic self and fixed-native-basis 95th-percentile
controls with empirical p-values at most `0.05`. The closest registered
comparison was transported within-prompt self, with p-value `0.03`.

The ROOT post-write audit verified every required RNTuple, 99 finite control
rows in every one of the twelve control groups, and independently recomputed
the serialized p95 thresholds and p-values. The bounded permitted conclusion
is that this corpus and layer reject the tested incidental-relative-geometry
null and justify continued investigation of relative coordinates. It does not
establish semantic content, pure memory representation, rotor coding, FFN
causal use, or an output-coordinate map.

## Exploratory Relative-Geometry Layer Follow-Up

After the classified layer-15 result, an explicitly exploratory same-corpus
follow-up was run at layers 30, 45, and 60. It reused the fixed rank-32
metric, seed, split, and complete controls; it did not select or validate a
depth trend. The combined F32 production capture and three exploratory result
artifacts are recorded in
`../experiments/tmix_relative_geometry_null/LAYER_FOLLOWUP_RESULTS.md`.

All three selected layers passed the mechanical relative-geometry null
comparison in both coordinate descriptions after serialized-row audit:

```text
layer 30: ambient 0.376565113915, transported 0.360181704631
layer 45: ambient 0.591554697303, transported 0.526803662012
layer 60: ambient 0.953604572558, transported 0.934751748162
```

Each of the twelve control comparisons per layer had empirical p-value `0.01`.
This is an exploratory observation on the frozen corpus and selected layers,
not evidence of a network-wide or monotonic depth pattern. It does not extend
the interpretation boundary beyond the classified layer-15 result.

## Admitted Controlled-Value Coordinate Comparison

The next admitted experiment is specified in
`../experiments/tmix_controlled_value_decode/EXPERIMENT.md`. It avoids activation
transplants and deep state interventions. A frozen full factorial crosses 16
controlled value tokens with 48 carrier tokens while keeping the final read token,
token position, prompt length, and template fixed. The split holds out carriers,
not values.

At layers 15, 30, 45, and 60, equal-capacity carrier-held-out linear decoders compare
the incremental value information available from incoming-plus-raw-update,
incoming-plus-ambient-log-tangent, and incoming-plus-transported-log-tangent
coordinates. Labels come from the controlled prompt rather than model logits, and
native behavior is diagnostic only. This tests whether relative coordinates are a
more stable candidate decoding representation than the raw TMix update; it does not
isolate a single memory state, prove FFN use, or establish a causal rotor code.

## Admitted One-Shot Native-Unrotation Generation Gate

The next causal falsification gate is specified in
`../experiments/tmix_causal_rotation_ablation/EXPERIMENT.md`. It reuses the frozen
controlled-value factorial rather than creating a new corpus. At layer 45, for the
mechanically fixed first value and 16 held-out carriers, it reconstructs a production
`time.out` correction that preserves post-TMix mean and radius while returning the
centered endpoint direction to `resid.in`.

The intervention occurs once at the final prompt token. Each native or perturbed
branch then advances its own recurrent state through up to 64 greedily generated
tokens with no further intervention. The primary response is generated-token
divergence from the native continuation; logits and activation response norms do not
classify the experiment. Native unrotation is compared with three same-value
train-carrier donor rotations and three random tangent rotations matched for endpoint
angle, mean, and radius.

If all native-unrotated continuations remain token-identical to native, the protocol
requires stopping the rotor-channel program at its strongest current candidate. A
positive matched-control result says only that the native direction is behaviorally
special for this controlled completion; it does not distinguish a rotor code from
destruction of an ordinarily useful context-dependent additive update. This edits a
one-token residual contribution, not a disentangled recurrent memory item.

An earlier logit-based Run 1 is retained in
`../experiments/tmix_causal_rotation_ablation/RUN1_INVALID_REPORT.md` and has no
scientific classification. Thirty of 48 random controls failed the registered mean
matching gate because an unconstrained Gaussian was projected off the endpoint
without first being centered. The revised generation protocol centers the Gaussian
before tangent projection and fixes fresh seed `315972`; no Run 1 behavioral
diagnostic is promoted to evidence or reused as a classified result.

## Classified Controlled-Value Coordinate Result

The registered controlled-value coordinate comparison completed on a fresh Vulkan
capture of the frozen 48-carrier by 16-value factorial. Component self-tests and
the production verifier passed; the verifier reported zero final-readout,
state-handoff, and capture-handoff error. The classified analysis passed its
fail-closed ROOT post-write audit, which recomputed held-out metrics, carrier
deltas, exact sign-flip p-values, Holm adjustments, predicate operands, and layer
labels from serialized rows.

Artifacts:

```text
build/tmix-controlled-value-classified-capture-rerun1.root
build/tmix-controlled-value-classified-result-rerun2.root
build/tmix-controlled-value-classified-result-rerun2.json
```

The mechanical overall classification is `relative_decoding_supported`. Layer 45
is the only `relative_coordinate_advantage` layer: relative ambient and transported
coordinates improve on additive by `0.0000291269` and `0.0000307835` held-out
carrier-balanced CE, respectively, with Holm-adjusted p-value `0.0065917969` for
each. Layers 15 and 60 are `additive_coordinates_match_or_win`; layer 30 is
`no_detected_layer_local_increment`.

All decoder models reached carrier-balanced held-out accuracy `1.0` at every
layer, so this is a small, near-ceiling cross-entropy/confidence result rather than
improved class recovery. The bounded conclusion is that, at layer 45 and on this
frozen factorial, incoming-plus-relative coordinates were more linearly decodable
under the registered cross-entropy comparison than incoming-plus-raw-update
coordinates. It does not establish pure memory content, a rotor code, FFN causal
use, semantic generalization, or an output-embedding map.
