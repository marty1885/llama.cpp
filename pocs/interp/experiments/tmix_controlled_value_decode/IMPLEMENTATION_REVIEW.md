# TMix Controlled-Value Decode Implementation Review

## Review Verdict

The current run contains an encouraging provisional layer-45 result, but it does
not satisfy the registered output and audit contract in `EXPERIMENT.md`. Do not
enter its classification into `logs/TMIX_INVESTIGATION.md` yet.

Reviewed artifacts:

```text
build/tmix-controlled-value-classified-capture.root
build/tmix-controlled-value-classified-result-v4.root
build/tmix-controlled-value-classified-result-v4.json
pocs/interp/experiments/tmix_controlled_value_decode/RESULTS.md
```

Do not overwrite these artifacts. A corrected run must use new paths.

## Blocking Defects

### 1. Classified JSON Does Not Satisfy The Output Contract

The registered compact JSON contract requires all frozen hashes, token-position
checks, factorial counts, numerical checks, selected lambdas, optimizer records,
layer/model metrics, carrier-level statistics, sign-flip and Holm results, outcome
predicates, protocol deviations, ROOT paths, and RNTuple row counts.

The current JSON contains only:

```text
status
classification
input ROOT path
five cross-layer mean cross-entropies
four layer outcome strings
```

Implement the complete JSON contract from `EXPERIMENT.md`. A classified run must
not report `status=valid` if any required field is absent.

At minimum, serialize:

- model path and SHA-256;
- candidate-list, corpus, manifest, template, and registration hashes;
- fixed token IDs and read/value/carrier positions;
- exact factorial, capture, behavior, and result row counts;
- every geometry, basis, split, and residual-identity check;
- every selected lambda and all cross-validation candidate scores;
- every optimizer convergence record;
- per-layer cross-entropy and accuracy for every decoder;
- per-carrier absolute metrics, gains, and advantages;
- every raw and Holm-adjusted p-value;
- every layer and overall outcome predicate;
- protocol deviations and artifact paths.

### 2. Decoder Accuracy Is Not Computed

`EXPERIMENT.md` requires carrier-balanced decoder accuracy as a diagnostic for every
model and layer. The analyzer computes cross-entropy only.

This omission matters because all reported decoder cross-entropies are near zero:

```text
layer 45 additive:             0.000129862446
layer 45 relative ambient:     0.000100735516
layer 45 relative transported: 0.000099078996
```

Without accuracy, the result cannot distinguish improved class recovery from a
small confidence or margin difference among models that may already classify every
sample correctly.

Compute and serialize:

```text
per-prompt predicted value class
per-carrier accuracy for every model
carrier-balanced aggregate accuracy for every model and layer
confusion matrix or equivalent per-value correct counts
```

Accuracy remains diagnostic and must not replace the pre-registered cross-entropy
classification rule.

### 3. Captured Behavioral Diagnostics Are Ignored

The capture contains a complete `behavior` RNTuple, but the analyzer does not read
or summarize it. Independent inspection found:

```text
rows:                                768
native top-1 controlled-value hits:  738
native top-1 controlled-value rate:  96.09375%
mean expected-value log probability: -1.07314
```

Read, validate, and serialize all registered behavioral diagnostics:

```text
expected-value log probability
expected-value rank
top-1 token
top-1 correctness
carrier-balanced and value-balanced summaries
```

Do not filter or weight decoding samples by behavior.

### 4. Final Decoder Predictions Are Not Auditable

The result ROOT contains the required RNTuple names and the carrier-level gain and
advantage rows used for sign-flip tests. It does not contain enough information to
independently reconstruct the final decoder outputs:

- no decoder weights or biases;
- no per-sample class probabilities;
- no per-sample predicted class;
- no per-carrier absolute cross-entropy by model;
- no per-carrier accuracy by model.

Serialize either the complete final decoder parameters and projected test features,
or, preferably, one compact prediction row per layer, model, carrier, and value:

```text
layer
model
carrier_id
value_id
true_value_id
predicted_value_id
true_value_probability
cross_entropy
correct
```

Then derive per-carrier and aggregate metrics from those rows. The post-write audit
must independently recompute all decoder metrics, gains, advantages, p-values,
Holm adjustments, and classifications from serialized predictions.

### 5. Validity Does Not Enforce The Registered Contract

The analyzer currently sets validity primarily from decoder convergence. It writes
`status=valid` even though required JSON fields, accuracy, behavior summaries, and
auditable predictions are absent.

Make validity fail closed on every requirement under `Numerical And Data Validity`
and `Output Contract` in `EXPERIMENT.md`. Every check must be serialized with:

```text
name
threshold or expected value
observed value
pass/fail
```

Any absent or failed check must set `status=invalid` and suppress classification.

### 6. Regularization Is At The Grid Boundary

All 20 final decoders selected:

```text
lambda = 1e-5
```

This is the weakest registered regularization and the lower boundary of the search
grid. Combined with near-perfect decoding, the layer-45 cross-entropy advantage may
be sensitive to confidence scaling and optimizer tolerance.

Do not silently change the registered grid for the corrected run. Instead:

1. Serialize all fold-level cross-validation scores so boundary selection is fully
   auditable.
2. Serialize final weight norms, biases, margins, and convergence diagnostics.
3. Report that every model selected the boundary in `RESULTS.md`.
4. Treat expanded-grid or fixed-calibration analysis as a separately discussed and
   registered sensitivity experiment, not a correction to this run.

### 7. `RESULTS.md` Prematurely Calls The Run Classified And Valid

The result report says the run passed the registered capture, numerical, optimizer,
ROOT-schema, row-count, and recomputation checks. That statement is incomplete
because the registered JSON, accuracy, behavior, prediction-audit, and fail-closed
requirements did not pass.

Mark the current `RESULTS.md` as an invalidated or provisional implementation draft
until a corrected artifact passes the complete contract. Do not copy its overall
classification to the durable ledger.

## Verified Parts Of The Current Run

The following implementation pieces were independently observed:

```text
capture activation rows: 768
capture behavior rows:   768
capture sample rows:     3,072

result metadata rows:                 4
feature-basis check rows:             4
cross-validation score rows:        140
decoder-fit rows:                    20
per-carrier gain/advantage rows:    384
aggregate cross-entropy rows:        20
statistical-test rows:               24
outcome-predicate rows:               4
```

All 20 serialized final fit summaries report convergence. Analyzer self-tests pass.
The serialized carrier-level differences permit independent recomputation of the
reported exact sign-flip p-values.

These checks make the provisional numerical result worth preserving, but they do
not override the missing registered outputs.

## Provisional Scientific Observation

If the complete corrected audit reproduces the current numbers, the layer outcomes
are:

```text
layer 15: additive coordinates match or win
layer 30: no detected layer-local increment
layer 45: relative-coordinate advantage
layer 60: additive coordinates match or win
```

At layer 45:

```text
ambient advantage over additive:     0.0000291269 CE
transport advantage over additive:   0.0000307835 CE
Holm-adjusted p for each comparison:  0.0065917969
```

The bounded provisional interpretation is:

> At layer 45 on this frozen factorial, incoming-plus-relative coordinates produced
> slightly lower held-out linear-decoder cross-entropy than the equal-capacity
> incoming-plus-raw-update representation.

This is a small, one-layer, near-ceiling effect. It does not establish a rotor code,
FFN use, pure memory, semantic generalization, or an output-embedding map.

## Required Corrected Run

1. Fix all blocking defects without changing the frozen corpus, model, carrier
   split, values, layers, feature rank, decoder family, lambda grid, statistical
   tests, Holm families, or mechanical outcome rules.
2. Run every corpus, collector, and analyzer self-test.
3. Reuse the frozen capture only if its complete schema, metadata, behavior, hashes,
   and factorial checks pass; otherwise create a new capture artifact.
4. Write corrected analysis artifacts to new paths, for example:

   ```text
   build/tmix-controlled-value-classified-result-v5.root
   build/tmix-controlled-value-classified-result-v5.json
   ```

5. Reopen the ROOT and JSON outputs and verify every required field and exact row
   count.
6. Independently recompute model CE, accuracy, carrier metrics, gains, advantages,
   exact sign-flip tests, Holm corrections, layer labels, and overall classification
   from serialized prediction rows.
7. Run `git diff --check`.
8. Only then replace `RESULTS.md` and update `logs/TMIX_INVESTIGATION.md`.

Do not change the scientific question or retrofit a stronger success claim while
correcting the implementation.
