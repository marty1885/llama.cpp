# TMix Destination Base-Recovery: Run4 Audit

## Verdict

Run 4 substantially improves the implementation, and its negative numerical
classification independently recomputes from the ROOT rows. It is not yet a valid
scientific artifact under `EXPERIMENT.md` because registered fail-closed checks and
output-contract requirements remain absent while the executable reports
`status=valid`.

Do not update `logs/TMIX_INVESTIGATION.md` or replace the invalidated `RESULTS.md`
yet.

## Independently Verified

Artifacts audited:

```text
build/tmix-destination-base-recovery-layer30-pile10k-run4.root
build/tmix-destination-base-recovery-layer30-pile10k-run4.json
build/tmix-destination-base-recovery-layer30-pile10k-run4.log
```

### ROOT Datasets And Rows

The ROOT artifact contains all four required RNTuples:

```text
metadata:           1 row
per_prompt_metrics: 22 rows
aggregate_metrics:  5 rows
control_values:     3,465 rows
```

The control count is exact:

```text
global shuffle:          99 replicates * 2 endpoint families * 5 metrics = 990
within-prompt shuffle:   99 replicates * 2 endpoint families * 5 metrics = 990
isotropic write:         99 replicates * 2 endpoint families * 5 metrics = 990
random input complement:99 replicates * 1 endpoint family  * 5 metrics = 495
total:                                                               3,465
```

Every family contains replicate IDs for all 99 replicates and 99 distinct
sub-seeds.

### Native Aggregation

Independently averaging the 22 serialized per-prompt rows reproduces the native
aggregate RNTuple and JSON exactly at displayed precision:

```text
angular-recovery fraction: -0.290191070987
recovery efficiency:       -0.262182449437
absolute angle recovery:   -0.029662186534
cosine gain:               -0.00381949172624
removed energy:             0.0124816241989
```

### Control Statistics And P-Values

Independent recomputation from `control_values` gives:

```text
global shuffle angular-recovery fraction:
  count 99
  median -0.340392
  p95    -0.335778
  native -0.290191
  empirical upper-tail p = 0.01

global shuffle recovery efficiency:
  count 99
  median -0.279383
  p95    -0.278637
  native -0.262182
  empirical upper-tail p = 0.01

isotropic angular-recovery fraction:
  count 99
  median -0.291575
  p95    -0.290768
  native -0.290191
  empirical upper-tail p = 0.02

isotropic recovery efficiency:
  count 99
  median -0.336623
  p95    -0.336011
  native -0.262182
  empirical upper-tail p = 0.01
```

The corrected global-shuffle lookup is functioning. The registered classification
recomputes as:

```text
exploratory_no_detected_base_recovery
```

because the native angular-recovery fraction is negative. Favorable upper-tail
comparisons against controls do not override the required positive native recovery.

### Runtime Isolation

The analysis binary no longer links `llama` or `llama-common`. Its direct
dependencies are ROOT, OpenBLAS, OpenMP, and their system dependencies.

## Remaining Blocking Defects

### 1. JSON Contract Remains Incomplete

The run-4 JSON lacks required:

- hashes for all input artifacts;
- token, prompt, and metric-exclusion counts;
- complete summaries for every control and endpoint family;
- machine-readable numerical checks with thresholds, observed values, and pass/fail;
- every outcome-rule operand and Boolean predicate;
- protocol-deviation records;
- ROOT dataset and row-count audit information.

The executable still sets `status=valid` without checking this contract. Complete
the JSON and make validity depend on successful serialization and post-write schema
validation.

### 2. Source `G_dest` Is Not Reproduced

`EXPERIMENT.md` requires reproduction of native rank-32:

```text
G_dest
E_input
E_time
D
```

Run 4 checks only `E_input`, `E_time`, and `D`. Implement `G_dest` reproduction,
serialize the observed value and error, and fail closed above the registered
`1e-6` tolerance.

### 3. Numerical Checks Exist Only Partly In The Log

The log records several successful values, but the ROOT/JSON artifacts do not
contain the complete registered validity record. A console log is not a substitute
for machine-readable validation.

Serialize every criterion listed in `EXPERIMENT.md`, including vector dimensions,
unit and stored-versus-recomputed direction errors, prompt split isolation,
permutation isolation, finite aggregate checks, control counts, and distinct
sub-seeds.

Any absent criterion must make the run invalid and suppress classification.

### 4. Registered Self-Tests Remain Incomplete

The implementation still tests only:

- one factorized positive case;
- exact zero-angle exclusion;
- one empirical p-value calculation.

It still lacks:

- holistic/random-direction negative case;
- near-zero angle cases;
- recovery efficiency toward, orthogonal to, and away from the input;
- global and within-prompt permutation isolation;
- prompt-balanced aggregation with unequal token counts;
- full classification truth table;
- checked control-name lookup.

Implement all registered tests. The real-data path must run them before opening the
input artifact, or serialize evidence that the exact binary passed them immediately
before the real run.

### 5. Post-Write Validation Is Still Missing

The review required programmatic validation of output schemas and row counts plus
independent classification recomputation before writing a result report. This audit
performed the independent recomputation externally, but the executable still does
not fail closed on its own malformed or incomplete outputs.

Add a post-write audit that reopens the ROOT and JSON artifacts and verifies:

```text
all required datasets and fields
exact row counts
all finite values
99 replicates and distinct sub-seeds per family
control summaries and p-values
mechanical classification
```

## Required Next Artifact

Fix the remaining blockers without changing the registered scientific question,
layer, rank, corpus, split, seed, controls, metrics, or classification rules. Run
the corrected exact analysis to new paths, for example:

```text
build/tmix-destination-base-recovery-layer30-pile10k-run5.root
build/tmix-destination-base-recovery-layer30-pile10k-run5.json
build/tmix-destination-base-recovery-layer30-pile10k-run5.log
```

The numerical result is expected to remain negative, but scientific acceptance
depends on the validation contract, not the attractiveness or stability of the
number.

Only after the run-5 artifacts pass an independent audit should `RESULTS.md` be
replaced and the durable investigation ledger be updated.
