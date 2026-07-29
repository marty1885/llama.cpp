# TMix Controlled-Value Coordinate Decoding Results

## Status

**Classified artifact valid.** A fresh Vulkan capture and `rerun2` analysis
passed component self-tests, the production verifier, complete schema and row-count
reopening, and fail-closed recomputation of held-out metrics, carrier deltas, exact
sign-flip p-values, Holm corrections, predicate operands, layer labels, and overall
classification from serialized ROOT rows.

This is a bounded matched-linear-decoding result. It is not evidence that
`time.out` is pure memory, that RWKV uses a rotor code, that the FFN causally
uses these coordinates, or that there is a validated output-embedding map.

## Frozen Run

```text
model: rwkv7-g1h_preview4743-13.3b-20260703-ctx8192-Q4_K_M.gguf
backend: Vulkan, -ngl 99
layers: 15, 30, 45, 60
rank: 32
seed: 941731
factorial: 48 carriers x 16 controlled values
split: 32 train carriers, 16 held-out carriers
```

The required production verifier passed on the exact model/backend:

```text
final_readout_max_abs_error=0
state_handoff_logit_max_abs_error=0
capture_handoff_max_abs_error=0
```

Artifacts:

```text
build/tmix-controlled-value-classified-prompts.txt
build/tmix-controlled-value-classified-manifest.json
build/tmix-controlled-value-classified-registration.json
build/tmix-controlled-value-classified-capture-rerun1.root
build/tmix-controlled-value-classified-result-rerun2.root
build/tmix-controlled-value-classified-result-rerun2.json
```

The capture has 768 activation rows, 768 behavior rows, and 3,072 sample rows.
The result ROOT has 20 converged decoder fits, 5,120 prediction rows, 1,024
per-carrier metric rows, 24 statistical-test rows, four predicate rows, and 65
behavior-summary rows. All five decoder models had carrier-balanced held-out
accuracy `1.0` at every layer; cross-entropy is therefore the registered primary
metric and measures confidence rather than class-recovery differences here.

## Held-Out Cross-Entropy

Carrier-balanced held-out cross-entropy, where lower is better:

| Layer | Incoming | Additive | Relative Ambient | Relative Transported | Endpoint | Layer Outcome |
|---|---:|---:|---:|---:|---:|---|
| 15 | 0.000184592 | 0.000120870 | 0.000122505 | 0.000122528 | 0.000258900 | additive_coordinates_match_or_win |
| 30 | 0.003723323 | 0.003696567 | 0.002761220 | 0.002760097 | 0.004122416 | no_detected_layer_local_increment |
| 45 | 0.000239450 | 0.000129862 | 0.000100736 | 0.000099079 | 0.000195595 | relative_coordinate_advantage |
| 60 | 0.000770942 | 0.000230571 | 0.000655695 | 0.000614919 | 0.000593607 | additive_coordinates_match_or_win |

## Carrier-Level Tests

Each test is an exact one-sided sign-flip test over the 16 held-out carriers.
The adjusted p-values use the pre-registered Holm families.

### Layer 15

```text
additive gain:                 +0.0000637224, adjusted p = 0.0000610352
ambient relative gain:         +0.0000620875, adjusted p = 0.0001220703
transported relative gain:     +0.0000620647, adjusted p = 0.0001220703
ambient advantage over add:    -0.0000016349, adjusted p = 1.0
transport advantage over add:  -0.0000016577, adjusted p = 1.0
```

### Layer 30

```text
additive gain:                 +0.0000267566, adjusted p = 0.4724121094
ambient relative gain:         +0.0009621031, adjusted p = 0.3672485352
transported relative gain:     +0.0009632260, adjusted p = 0.3672485352
ambient advantage over add:    +0.0009353465, adjusted p = 0.3009338379
transport advantage over add:  +0.0009364694, adjusted p = 0.3009338379
```

### Layer 45

```text
additive gain:                 +0.0001095880, adjusted p = 0.0000610352
ambient relative gain:         +0.0001387149, adjusted p = 0.0001220703
transported relative gain:     +0.0001403714, adjusted p = 0.0001220703
ambient advantage over add:    +0.0000291269, raw p = 0.0008239746, adjusted p = 0.0065917969
transport advantage over add:  +0.0000307835, raw p = 0.0008544922, adjusted p = 0.0065917969
```

Both relative coordinate descriptions satisfy the registered positive-gain and
positive-additive-advantage predicates at layer 45.

### Layer 60

```text
additive gain:                 +0.0005403707, adjusted p = 0.0000610352
ambient relative gain:         +0.0001152474, adjusted p = 0.0001220703
transported relative gain:     +0.0001560226, adjusted p = 0.0009613037
ambient advantage over add:    -0.0004251233, adjusted p = 1.0
transport advantage over add:  -0.0003843481, adjusted p = 1.0
```

## Classified Observation

At layer 45 on this frozen controlled-value factorial, incoming-plus-relative TMix
coordinates produced slightly lower held-out decoder cross-entropy than the
equal-capacity incoming-plus-raw-update representation.

## Fragility And Open Question

The layer-45 relative advantage is small, approximately `3e-5` CE per
held-out carrier, and incoming residual coordinates already decode the value
near ceiling. The result occurs at one of four registered layers. The two
relative coordinate descriptions are related, so their agreement is not two
independent replications.

The exact p-values quantify the registered carrier sign-flip null; they are
not probabilities that the representational interpretation is true or that the
result is not due to a broader chance or selection process. A new frozen,
disjoint replication is needed before treating this layer-specific observation
as stable.
