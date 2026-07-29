# TMix Destination Base-Recovery Implementation Review

## Review Status

The current run must not be treated as valid or entered into the scientific ledger.
The native scalar result is provisionally interesting, but the implementation and
artifacts do not satisfy the experiment contract in `EXPERIMENT.md`.

Existing outputs:

```text
build/tmix-destination-base-recovery-layer30-pile10k.root
build/tmix-destination-base-recovery-layer30-pile10k.json
```

Do not overwrite these artifacts. A corrected run must use new paths.

## Blocking Defects

### 1. Required ROOT Data Are Missing

`rootls` reports only:

```text
metadata
```

The required RNTuples from `EXPERIMENT.md` are:

```text
metadata
per_prompt_metrics
aggregate_metrics
control_values
```

In `rwkv_tmix_destination_base_recovery.cpp`, the `control_values` writer created
around lines 353-360 remains alive when `TFile::Close()` is called around line 370.
Its destructor therefore runs after the file has been closed. Scope and destroy
every appended RNTuple writer before closing the `TFile`.

Implement and serialize the missing `per_prompt_metrics` and `aggregate_metrics`
RNTuples as required by `EXPERIMENT.md`. After the corrected run, validate all
dataset names and row counts programmatically, not only with console output.

This defect alone invalidates the current run because the complete control values
cannot be independently audited or used to recompute the classification.

### 2. Global-Shuffle P-Values Use Incorrect Map Keys

Control values are inserted using keys:

```text
global_shuffle_fraction
```

At approximately lines 466-470, the p-value loop instead reads:

```text
global_fraction
```

`std::map::operator[]` silently creates empty distributions for those missing keys.
The resulting empirical p-value is mechanically `1.0`, regardless of the actual
global-shuffle controls.

Use exact existing keys and avoid `operator[]` for required scientific results.
Use `.at()` or an explicit checked lookup so a missing key fails closed.

Add a self-test that stores a control distribution under the complete family name
and verifies that summary and p-value lookup use that same name.

### 3. Required Control Evaluations Are Incomplete

The implementation currently evaluates only matched endpoints for:

```text
global_shuffle
within_prompt_shuffle
isotropic_write
```

The protocol additionally requires:

- applying every shuffled-fitted destination basis to native test endpoints;
- applying every isotropic-fitted destination basis to native test endpoints;
- the random rank-32 input-complement basis diagnostic on native endpoints;
- serialization of `evaluation_endpoint_family` for every control row;
- removed energy and required diagnostic metrics, not only fraction and efficiency.

`random_complement()` is implemented but never called.

Implement every required control or set status `invalid`. Do not silently narrow the
protocol after observing the native result.

### 4. JSON Does Not Meet The Output Contract

The JSON currently contains only:

```text
schema version
experiment
status
classification
post-hoc flag
native metrics
four p-values
ROOT path
```

It omits required fields including:

- input paths and hashes;
- fixed layer, rank, seed, and repeat count;
- token and prompt counts;
- metric exclusion counts;
- numerical checks and their thresholds;
- complete control summaries;
- all empirical p-values;
- every outcome-rule operand and Boolean predicate;
- protocol deviations;
- hashes or row counts permitting control serialization to be audited.

Implement the complete compact-review contract from `EXPERIMENT.md`. A run may not
set status `valid` unless every required JSON and ROOT field is present and finite.

### 5. Numerical Validation Is Incomplete And Not Serialized

The implementation checks some basis and source metrics internally, but it does not
serialize the checks. It also reproduces `E_input`, `E_time`, and `D` but not the
required source `G_dest` metric.

Serialize each numerical criterion with:

```text
threshold
observed value
pass/fail
```

Include at least all criteria listed under `Numerical Validity And Fail-Closed
Criteria` in `EXPERIMENT.md`. Any absent or failed check must set status `invalid`
and suppress classification.

### 6. Synthetic Self-Tests Are Incomplete

The current self-tests cover only:

- one factorized recovery case;
- exact zero-angle exclusion;
- one empirical p-value calculation.

They omit required tests for:

- a holistic/random-direction negative case;
- near-zero angles;
- recovery efficiency toward, orthogonal to, and away from the input;
- global and within-prompt permutation isolation;
- prompt-balanced aggregation with unequal token counts;
- the full classification truth table;
- required control-name lookup, which would have caught the global-key bug.

Implement all registered self-tests and run them before reading the real artifact.

### 7. The Analysis Target Links The Model Runtime

`EXPERIMENT.md` requires a capture-only analysis executable that does not link the
model runtime. The current source includes the complete destination collector after
renaming its `main`, and the CMake target links `llama` and `llama-common`.

Extract or locally implement only the required geometry and basis helpers. Link the
analysis target to OpenBLAS and the required ROOT components, not the model runtime.
The corrected executable must not initialize or contain a callable model-collection
path.

### 8. The Result Report Was Written Before Artifact Validation

`EXPERIMENT.md` explicitly states that no result report should be written until:

```text
both output schemas and row counts are validated
classification is independently recomputed from serialized controls
all numerical and serialization checks pass
```

Those conditions were not met. Keep `RESULTS.md` as an invalidated draft or clearly
mark it invalid until a corrected run succeeds. Do not copy its classification into
`logs/TMIX_INVESTIGATION.md`.

## Provisional Scientific Observation

The native calculation is separate from the broken global p-value lookup and was:

```text
prompt-balanced angular-recovery fraction: -0.29018995
prompt-balanced recovery efficiency:       -0.26218189
absolute angle recovery:                   -0.02966210 radians
cosine gain:                               -0.00381948
```

If these values survive a compliant rerun, removing the native rank-32 destination
makes the held-out endpoint farther from its paired incoming direction. That would
fail the tested necessary condition for the simple factorization:

```text
post-TMix direction = preserved incoming direction + separable destination addition
```

It would not establish that the destination is absent, that TMix lacks structured
addition, or that all rotational representations are false. Do not interpret the
isotropic p-values as support for base recovery: native recovery itself is negative.

## Required Corrected Run

1. Fix every blocking defect above without changing the registered metrics or
   outcome rules.
2. Build and run all synthetic self-tests.
3. Build only the corrected analysis target.
4. Run against the same fixed input ROOT, source JSON, manifest, and seed.
5. Write new artifacts, for example:

   ```text
   build/tmix-destination-base-recovery-layer30-pile10k-run2.root
   build/tmix-destination-base-recovery-layer30-pile10k-run2.json
   ```

6. Programmatically verify all required RNTuples, fields, row counts, finite values,
   distinct sub-seeds, and control-family counts.
7. Independently recompute all summaries, p-values, and classification from the
   serialized ROOT rows and compare them with JSON.
8. Run `git diff --check`.
9. Only then replace or revise `RESULTS.md` and update the durable investigation
   ledger.

Do not change the scientific question, layer, rank, corpus, split, seed, control
count, metrics, or mechanical classification rules while fixing the implementation.
