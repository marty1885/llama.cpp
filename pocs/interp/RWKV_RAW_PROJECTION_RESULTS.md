# RWKV Raw Projection Results

## K/V Versus Random Baseline

Date: 2026-07-12

Question: do raw RWKV7 `time.k` and `time.v` vectors align with the model's
actual next token more than random vectors in the same 4096-dimensional space?

Command:

```bash
build/bin/llama-rwkv7-time-lens \
  -m rwkv7-g1h_preview4743-13.3b-20260703-ctx8192-BF16.gguf \
  -ngl 999 \
  -p "A company raises money by selling" \
  --project-sources k,v \
  --random-baseline \
  --steps 12 \
  --top 5 \
  --json build/rwkv7-kv-vs-random-finance-12tok.json
```

Continuation:

```text
shares of stock. The company can be a public company, which
```

At every generated token, each of 61 layers contributed one `k` and one `v`
vector. Each vector was independently passed through the native output
normalization and output matrix. Its rank for that step's native greedy next
token was compared with the rank from a deterministic random-sign vector in
the same batched projection graph.

| Source | Observations | Mean actual rank | Mean random rank | Actual beats random | Actual top 100 | Random top 100 | Actual top 1,000 | Random top 1,000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `time.k` | 732 | 3,948 | 21,161 | 91.8% | 34.4% | 0.0% | 67.2% | 0.0% |
| `time.v` | 732 | 4,952 | 21,161 | 89.3% | 31.7% | 0.0% | 62.7% | 0.0% |

All 12 generated-token blocks had a lower mean actual rank than their random
baseline for both `k` and `v`.

## Significance

The paired one-sided win counts are 672/732 for `k` and 654/732 for `v`.
Under an invalid but useful naive null where all 732 comparisons are
independent fair coin flips, exact binomial tails are:

| Source | Naive one-sided p-value |
| --- | ---: |
| `time.k` | 3.6e-132 |
| `time.v` | 1.7e-114 |

Those p-values must not be treated as formal evidence because layers and
adjacent decode steps are correlated, this is one continuation, and the run
uses one deterministic random baseline per source. Treating only the 12 token
blocks as independent gives a conservative one-sided sign-test value of
`2^-12 = 2.44e-4` for each source type: every block favored the real vectors.

The result is promising evidence of non-random next-token alignment, not yet a
general claim about semantic interpretation. Replicate over independent prompt
families and random seeds before making that claim.
