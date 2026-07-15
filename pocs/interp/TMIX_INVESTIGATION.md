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

## Current Question

```text
What stable, context-dependent transformation does layer-60 TMix induce in
ffn.norm, and which downstream behavior changes specifically because of it?
```

## Next Experiment: Not Yet Run

Compare the measured tangent-induced `ffn.norm` change with equal-norm random
tangent controls across many contexts.

Pre-register the alternatives:

```text
specific learned direction:
  the measured tangent produces reproducible channel-feature and output effects
  that matched random tangent controls do not.

generic large residual perturbation:
  matched random tangent controls produce comparable feature and output effects.
```

Only after this comparison should individual channel features or head coalitions
be treated as candidates for a functional circuit.

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
