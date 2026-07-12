# RWKV7 Time-Mix Experiment Plan

## Goal

Determine whether RWKV7 internal time-mix coordinates have stable, prompt-relevant
causal effects that can be viewed only after the model transforms them into residual
coordinates. This is not a claim that a direct readout exposes private chain of
thought.

## Coordinate Rules

There are two different kinds of captured activation:

```text
time.out, channel.out
    Final branch contributions added to the residual stream.
    Direct final-norm/output-head readout is coordinate-valid, but can be trivial.

time.r, time.w, time.k, time.v, time.a, time.g, time.wkv, time.rkv
    Internal time-mix coordinates before the branch's learned output transform.
    Direct output-head projection is exploratory only, not a valid semantic decoder.
```

`time.wkv` is the current token's query-dependent read from RWKV recurrent memory.
`time.rkv` is the post-retrieval interaction before the branch output projection.

## Results So Far

### Absolute Direct Readout

`--all-time-mix` reads all eight internal sources from all 61 layers in one 489-row
output-head GEMM, including a final-residual round-trip control. A five-topic,
top-5 trace showed a common lexical sector across processor, geography,
photosynthesis, violin, and hurricane prompts:

```text
Focus, Force, Forms, Found, Hours, Flush, Forum, Hover, House, Floor
```

No prompt-conditioned conceptual cluster was visible in these absolute raw-vector
readouts. This does not imply that the raw vectors lack information; their geometry
is not that of the final residual/output head.

Artifact: `/tmp/rwkv-topic-all-time-mix-top5.json`.

### France Versus Germany Output-Logit Contrast

The ordinary `--contrast` mode separately reads each source through final
normalization and the output head, then subtracts the two output-logit vectors.
It is an output contrast, not an activation-space delta.

The final-residual control behaved as expected:

```text
France positive:  Paris, Paris, pari, Paris in Cyrillic, Par
Germany positive: Bundes, Federal, Federal, Berlin, Ber
```

Two weak raw-source observations appeared in the top 5:

```text
rwkv.layer.27.time.rkv, France-positive: " Europe" at rank 5
rwkv.layer.58.time.v, Germany-positive: " euro" at rank 2
```

These are evidence that source contrasts can move geography-related output-head
directions. They are not yet discriminative evidence: Europe and euro apply to both
countries, and each observation occurred only once.

Artifact: `/home/marty/.local/share/opencode/tool-output/tool_f522ac03b001q1w6re06qhR2wg`.

### Matched-Pair Follow-Up

The same two source/layer combinations were tested on three further capital pairs:

```text
Italy versus Spain
Japan versus China
Canada versus Australia
```

Neither `rwkv.layer.27.time.rkv` nor `rwkv.layer.58.time.v` repeated a
geography-related top-5 token. They returned to the shared lexical background or
unrelated fragments on every pair. The initial `Europe` and `euro` observations
therefore do not pass a repeatability check and must be treated as chance-level
exploratory hints, not evidence for a readable source/layer.

The final-residual output-contrast control remained geographically structured:

```text
Italy positive: Rome-family tokens; Spain positive: Madrid-family tokens
Japan positive: Tokyo-family tokens; China positive: Beijing-family tokens
Canada positive: Toronto, Quebec, Ottawa, Ontario; Australia positive: Australian,
                   Sydney, Melbourne
```

### All-Layer Follow-Up

The three follow-up pairs were then run through every raw time-mix source at every
layer: 488 sources, two signed top-5 lists per pair, or 14,640 raw readout entries.
The extracted table is:

```text
/tmp/opencode/rwkv-capital-all-time-mix-contrast-top5.tsv
```

Two isolated country lexical hits appeared outside the final-residual control:

```text
Italy-positive,   rwkv.layer.23.time.r: "Italian" at rank 5
Spain-positive,   rwkv.layer.51.time.r: "Rome" at rank 2
```

The second hit has the counterintuitive sign, and neither source/layer repeats on
Japan/China or Canada/Australia.

The full table also contains two coherent, prompt-tangent clusters that a search for
country names alone would miss:

```text
Italy-positive, rwkv.layer.31.time.r:
  Places, places, IPV, places, realise

Japan-positive, rwkv.layer.9.time.g:
  Country, country, one non-Latin piece, Open, Country
```

Their opposing prompt sides were lexical-background tokens rather than another
geography cluster. These are still one-pair observations and require paraphrase and
matched-pair repetition, but they are stronger evidence than the isolated `Europe`
and `euro` hits that some internal source contrasts can expose prompt-tangent output
directions.

### Broad Topic Sweep

Eight held-out pairs were run through all 488 raw sources with signed top-5 output
contrasts:

```text
currency:   Japan versus United Kingdom
phones:     iPhone versus Android phone
retail:     Costco versus Walmart
hardware:   CPU versus RAM
biology:    photosynthesis versus cellular respiration
music:      violin versus piano
geography:  Brazil versus India
technology: Itanium processor versus Eiffel Tower
```

The complete log and extracted table are:

```text
/tmp/opencode/rwkv-wide-all-time-mix-contrast-top5.log
/tmp/opencode/rwkv-wide-all-time-mix-contrast-top5.tsv
```

The wide, preregistered tangent vocabulary found the following raw-source matches:

```text
currency, Japan-positive:
  L34 time.v:   coin, rank 3
  L46 time.wkv: dollar, rank 4

phones:
  L19 time.g, Android-positive: iPhone, rank 2 (counterintuitive sign)
  L48 time.rkv, iPhone-positive: Mobile, rank 4

hardware, CPU-positive:
  L36 time.wkv: ram, rank 3
  L42 time.rkv: Memory, rank 5

geography, Brazil-positive:
  L26 time.w: South, rank 5
  L34 time.r: Asia, ranks 3 and 5 (wrong continent)
  L57 time.g: Capital, rank 4
```

Retail, biology, music, and technology/landmark pairs produced no raw top-5 match
to their preregistered tangent or answer vocabularies. No pair produced a reliable
answer token from a raw time-mix source. The hardware pair is the strongest new
candidate because two different time-mix sources surfaced hardware terms on the
same prompt side, but it is still one pair and not a validated feature.

### France Versus Germany Activation-Space Delta

`--contrast-activation-delta` computes each source activation difference before
the final norm/output head. Its final-residual control produced Paris-family tokens
at ranks 1 through 5. None of the 488 raw time-mix source deltas produced a
France/Germany/Paris/Berlin-related top-5 token; the shared lexical sector remained
dominant.

This rejects direct final-head readout of these raw source deltas for this prompt
pair. It does not reject the presence of information in the sources.

Artifact: `/home/marty/.local/share/opencode/tool-output/tool_f522d0b31001GlRMa8GZXarKvy`.

## Do Not Do

- Do not treat exact emitted-token overlap as the success criterion. `channel.out`
  can align with emitted tokens for a generic continuation or "nothing to add"
  direction.
- Do not use a corpus-wide mean as a claimed geometric center. Every source/layer
  has context-dependent computation and unknown geometry; centering would be a
  visualization heuristic, not causal evidence.
- Do not spend the next compute budget on a rank-512 global residual J-lens. The
  rank-64 residual operator was weak and the rank-128 numerical fit worsened.

### Activation-Center Pilot Note

A first all-source activation-corpus collector was started on 2026-07-12 using
one context-bearing Penn Treebank position per line. The pre-FP16-capture version
measured about `0.20` sampled positions per second: 48 positions took 230 seconds,
or an estimated two hours for 1,500 positions. Disk writes were only about 0.8 MiB/s,
so the bottleneck is 489 individual GPU-to-host activation captures, not SSD I/O.

The collector now captures native FP16 values and writes them directly to an on-disk
ROOT RNTuple. This halves transfer volume and avoids a CPU conversion, but measured
single-position throughput remained `0.206` samples/s. Four independent carrier
states can now share one capture graph: each source is copied once as a
`64 x 64 x 4` tensor and split into four on-disk rows. A 16-position validation run
measured `0.272` samples/s, a 32% improvement, but still projects to about 92 minutes
for 1,500 positions. A proper large run should first test wider capture batches and
batch prefix-state preparation; increasing ROOT write batching alone will not
materially improve throughput.

### Activation-Center Geometry Pilot

The 16-position batch-4 ROOT pilot was analyzed source by source with one-row
bounded reads. `K=8` is degenerate at this sample size: its 3,912 source-clusters
contained 1,852 singleton clusters, with occupancy from 1 through 9. It is not used
for centered readout.

The usable pilot condition is a trimmed `K=1` center per source. A radial
median-plus-three-MAD cutoff retained 11 through 16 of the 16 rows per source,
removing 667 source-row observations in total. Those rows remain in the activation
RNTuple; the cutoff only changes the fitted center.

The raw-source residual-to-raw norm ratio ranges from `0.049` for
`rwkv.layer.32.time.w` to `0.878` for `rwkv.layer.57.time.rkv`, so centering is a
substantial transform for some sources and negligible for others.

The centered static CPU prompt (`A computer's central processing unit is the`) did
not yield `ram`, `Memory`, `CPU`, `processor`, or `computer` in any raw-source top-5
list. This includes the earlier contrast-only hardware candidates
`rwkv.layer.36.time.wkv` and `rwkv.layer.42.time.rkv`. Centering therefore does not
currently turn absolute direct raw-source readout into a reliable semantic decoder.
It remains an empirical visualization transform to test on larger, held-out data.

### Q4 150-Position Centering Check

The Q4_K_M model enabled a 150-position, batch-4 Penn Treebank collection at `0.661`
samples/s, versus `0.272` samples/s for the BF16 batch-4 pilot. A trimmed `K=1`
three-MAD refit retained 120 through 150 rows per source and excluded 4,848 source-row
observations.

The same held-out CPU prompt still produced no `ram`, `Memory`, `CPU`, `processor`, or
`computer` top-5 token in any raw or centered source. In particular, centered
`rwkv.layer.36.time.wkv` produced `族`, `co`, `tribe`, `incidental`, and `tribal`; centered
`rwkv.layer.42.time.rkv` remained unrelated fragments. The larger global center therefore
does not rescue direct raw-source output-head decoding.

The Q4 final-residual round-trip max-logit error was `1.987282`, materially larger than
the BF16 output-only check. Use Q4 for collection throughput, but retain BF16 for any
final numerical claim about output-head readout.

## Next Experiment: Same-Token Native Causal Transport

The zero point must be the unperturbed model computation, not a guessed default
activation. For a fixed RWKV state and current token:

```text
internal source +/- epsilon direction
    -> remaining native time-mix operations
    -> time.out delta and resid.out delta on the same token
    -> production final norm/output head readout
```

### Initial Sources

1. `time.wkv`: the addressed recurrent-memory read is the most architecture-specific
   candidate.
2. `time.rkv`: test next because the France/Germany output contrast weakly exposed
   `Europe` at layer 27.
3. Only then test `time.v`, whose output contrast weakly exposed `euro` at layer 58.

### Per-Source Protocol

1. Capture the source, `time.out`, and final `resid.out` on a fixed current token.
2. Inject matched positive and negative perturbations at the source.
3. Measure central differences at `time.out` and final `resid.out`.
4. Verify deterministic replay, nonzero response, plus/minus symmetry, and agreement
   between two epsilon values.
5. Read out the residual-coordinate deltas only after those numerical checks pass.
6. Repeat across prompt paraphrases and matched topic pairs. A readable result must
   recur in the same source/layer band and be conceptually related to the prompt.

### Initial Native-Transport Checks

`llama-interp-jlens-operator` now supports every raw time-mix source with
`--source-activation time-r|time-w|time-k|time-v|time-a|time-g|time-wkv|time-rkv`.
It captures all diagnostics in FP32, injects a deterministic unit-norm Rademacher
direction as `x +/- epsilon*d`, and reports source injection, same-token `time.out`,
final-residual finite differences, replay, two-epsilon agreement, and a production
output-head finite-difference readout of the final residual. The latter is
coordinate-valid; the Rademacher direction is only a numerical control and its top
tokens have no semantic interpretation.

The following BF16/HIP runs used epsilon 0.20 and comparison epsilon 0.10:

| Source | Prompt | Source error L2 | `time.out` derivative L2 | Final-residual derivative L2 | Final derivative cosine |
| --- | --- | ---: | ---: | ---: | ---: |
| L36 `time.wkv` | CPU prompt | `8.19e-08` | `0.344663` | `1.152170` | `0.9999998` |
| L42 `time.rkv` | CPU prompt | `2.38e-08` | `1.850432` | `2.809951` | `1.0000000` |
| L27 `time.rkv` | `France is a country in` | `2.23e-08` | `0.563979` | `2.126571` | `0.9999999` |
| L58 `time.v` | `The currency used in France is the` | `2.55e-08` | `1.714519` | `1.799514` | `1.0000000` |

Every run had exact clean and repeat-plus replay. Final-residual symmetry errors were
between `0.00235` and `0.01166`; two-epsilon final derivative relative-L2 differences
were at most `5.89e-04`. This validates native causal transport for these sources, but
does not support a semantic claim: the L58 random-direction logit derivative was headed
by unrelated tokens such as `Blo`, `blo`, and `sources`, despite the clean next token
being `Euro`.

The initial centered-direction checks below use `x - c` around the unperturbed source
activation. They are not sufficient without paraphrase and matched-control replication.

### Centered-Direction Checks

The runner now accepts `--center-root CENTROIDS.root`. For a raw source it reads that
source's K=1 centroid and uses the normalized live direction `x - c`; it does not replace
the live activation `x` with the centroid. The following BF16/HIP runs used the trimmed
150-position Q4 calibration center, epsilon 0.20, and comparison epsilon 0.10:

| Source | Prompt | Source error L2 | `time.out` derivative L2 | Final-residual derivative L2 | Final derivative cosine |
| --- | --- | ---: | ---: | ---: | ---: |
| L36 `time.wkv` | CPU prompt | `2.10e-07` | `0.162389` | `0.330419` | `0.9999982` |
| L42 `time.rkv` | CPU prompt | `1.52e-07` | `0.689112` | `1.206834` | `0.9999999` |
| L58 `time.v` | France currency prompt | `1.90e-07` | `0.837385` | `0.862247` | `0.9999998` |

Each centered-direction run had exact clean/repeat-plus replay and final-residual
symmetry error at most `0.00664`. None produced a prompt-relevant logit derivative:
the L36 CPU direction produced unrelated fragments, L42 returned `liest`, `ties`, and
`lies` versus the default-sector `Flush`, and L58 returned `Z` fragments rather than
Euro-related tokens. Thus the current global K=1 centroid is a numerically valid
directional baseline but does not, by itself, reveal a semantic causal direction.

The remaining L36 time-mix sources were scanned with the same centered CPU protocol:

| Source | Final-residual derivative L2 | Top positive derivative tokens |
| --- | ---: | --- |
| `time.r` | `0.650974` | `Getty`, `")`, `Floor`, `Flags`, `Franc` |
| `time.w` | `0.098666` | `ainer`, `WISE`, `VIDIA`, `stores`, `handled` |
| `time.k` | `0.572290` | `Getty`, `Group`, `GROUP`, `Final`, `Floor` |
| `time.a` | `0.099987` | `Frank`, `Found`, `Files`, `Heart`, `Grand` |
| `time.g` | `0.122783` | `Flash`, `Hello`, `Hover`, `Green`, `Georg` |

`VIDIA` and `stores` in the single L36 `time.w` CPU run are not evidence of a hardware
feature: a CPU paraphrase instead produced `ainer`, `Hover`, `agrid`, `Fatal`, and
`Flash`; the unrelated France-capital control produced `Found`, `Affero`, and unrelated
fragments. No CPU/RAM-related token recurred. All five runs had exact replay, injection
error at most `1.49e-06`, and final two-epsilon derivative cosine at least `0.999867`.

The next step is not a larger cluster count. Repeat these centered directions across
preregistered paraphrases and matched unrelated-prompt/random-direction controls, then
consider a larger calibration set or conditional centers only if an effect is stable.

### Matched-Prompt Direction Checks

`llama-interp-jlens-operator --direction-prompt TEXT` captures the selected source at
the final token of `TEXT` and uses normalized `source(primary) - source(TEXT)` as the
perturbation direction. It never imports the contrast prompt's recurrent state; the
primary prompt remains the sole native execution. This tests a task-relevant source
direction without assuming that a global centroid is semantic.

Three BF16/HIP matched-prompt checks passed all numerical controls but did not expose the
paired concepts in the final-logit derivative:

| Primary versus direction prompt | Source | Final-residual derivative L2 | Top positive derivative tokens |
| --- | --- | ---: | --- |
| CPU versus GPU | L36 `time.wkv` | `0.365176` | `CLUDING`, `Statement`, `Flat`, `statement`, `States` |
| CPU versus GPU | L42 `time.rkv` | `1.263692` | `beast`, `wave`, `liest`, `wash`, `waves` |
| France currency versus Germany currency | L58 `time.v` | `0.961238` | `AT`, `ATS`, `SSL`, `at`, `Family` |

The corresponding clean next tokens were `brain`, `brain`, and `Euro`; none of CPU,
GPU, France, Germany, or Euro appeared in the signed top-10 derivative lists. This
rejects these three layer/source/prompt-pair configurations as readable semantic
directions. It does not establish that all prompt-pair directions or source layers fail.

### Promotion Gate

A source/layer is worth further causal work only if it has all of:

- Stable finite-difference behavior across epsilon and replay.
- A nontrivial same-token residual effect.
- Prompt-conditioned, paraphrase-stable readout patterns beyond the shared lexical
  background.
- A matched random-direction or unrelated-prompt control that does not show the
  same pattern.

Until this gate passes, the only coordinate-valid single-prompt views are `time.out`
and `channel.out`.
