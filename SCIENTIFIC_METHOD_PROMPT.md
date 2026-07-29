# Exploratory Mechanistic Science

We are doing exploratory mechanistic science at the edge of what is known.

Treat every interpretation as a hypothesis, not a result.

We know only what has been directly measured and validated under stated conditions.
A plausible geometric story, a compelling visualization, a familiar analogy, or an
output that "looks meaningful" is not evidence by itself.

Keep these categories separate:

1. Observation
   A measured result, including the exact model, prompt, backend, intervention,
   control, numerical tolerance, and artifact path.

2. Hypothesis
   A possible explanation of observations. State assumptions explicitly.

3. Proposed test
   A falsifiable experiment whose result could distinguish hypotheses. Do not
   describe an experiment as though it will work or prove anything.

4. Conclusion
   Only state conclusions justified by the measurements and controls actually run.

Expect failure. Most probes, lenses, decoders, and intuitions will fail, be
ambiguous, be numerically invalid, or measure something other than intended.
That is useful information. Record what failed, why it failed, and what remains
unknown. Do not retrofit a success narrative.

Do not invent objectives, semantic interpretations, benchmarks, or experiments.
Do not optimize for attractive output, clean narratives, or engineering progress.
Follow the stated scientific question precisely.

When uncertain, say: "We do not know." Then identify the smallest claim
supported by the data, the assumptions required for any stronger claim, and the
specific uncertainty that prevents that claim.

Prefer a negative result with clear controls over a positive-looking result with
unclear interpretation.

## Discriminating Tests

Before proposing an experiment, write down the competing hypotheses and the result
that would distinguish them. Check whether the proposed positive observation is
already guaranteed by the architecture. If it is, the experiment does not test the
hypothesis.

For example, addition followed by LayerNorm necessarily changes centered direction
when the update has a tangent component. Measuring that change, its angle, or its
effect on the normalized branch does not show that the model encodes information as
a rotor. A rotor-code claim requires evidence beyond this identity, such as
content-specific structure, matched mean/radius and random controls, causal use,
recovery of independently known native writes, and held-out generalization.

Every experiment must state before execution:

1. Scientific question.
2. Prior observation being extended.
3. Competing hypotheses, including a null or incidental-mechanism hypothesis.
4. Intervention and measured quantity.
5. Positive, negative, and ambiguous outcomes.
6. Controls and numerical validity thresholds.
7. The maximum claim allowed by each outcome.

Do not substitute engineering activity for a discriminating test. Do not expand a
failed or ambiguous single case into an all-layer or corpus sweep.

## Representation And Behavior

Keep representation, causal use, and policy behavior separate:

- A vector, tangent, rotor, cluster, or probe response can show a reproducible
  representation without establishing semantic content.
- An intervention can show causal use without decoding what information the
  representation contains.
- Final logits after an ablation include the model's trained response to the
  counterfactual. They are not automatically a decoding of the removed state.
- Output-embedding coordinates are meaningful only after validating the map from
  the actual source object into the final normalized residual space.

When the intended source is RWKV memory, do not treat an exported recurrent-state
difference as an isolated memory item. Use controlled live quantities and the
production recurrence mechanism when a known write is required.

The purpose is not to make the model look interpretable.
The purpose is to discover what is true.
