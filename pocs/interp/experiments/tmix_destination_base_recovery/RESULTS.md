# TMix Destination Base-Recovery Results

## Status

Audited exploratory result from the fixed layer-30 Pile capture. The full audit,
including artifact paths, numerical checks, control summaries, and permitted
claim, is in [RUN7_AUDIT.md](RUN7_AUDIT.md).

## Configuration

```text
layer: 30; rank: 32; repeats: 99; seed: 45678
train/test tokens: 1,664 / 1,408
train/test prompts: 26 / 22
```

The analysis read existing ROOT captures only. It did not execute the model,
intervene on recurrence, inspect the FFN, generate logits, or make semantic
claims.

## Result

The mechanical classification is:

```text
exploratory_no_detected_base_recovery
```

| Prompt-balanced held-out metric | Value |
| --- | ---: |
| angular-recovery fraction | `-0.290191003138` |
| recovery efficiency | `-0.262182409117` |
| absolute angle recovery | `-0.029662188836` |
| cosine gain | `-0.003819492382` |
| removed energy | `0.012481627553` |

Removing the native rank-32 destination moved held-out post-TMix directions away
from their paired incoming directions on average. This fails the tested necessary
condition for a simple separable added destination at this layer, rank, and
corpus.

## Permitted Conclusion

No positive held-out recovery of incoming direction was detected after removing
the native destination at this layer, rank, and corpus.

This does not establish that TMix is unstructured, that the destination is absent,
that it contains no memory-related information, that RWKV lacks a rotor-like
description, or that the FFN does not use any related representation.

## Artifacts

```text
build/tmix-destination-base-recovery-layer30-pile10k-run7.root
build/tmix-destination-base-recovery-layer30-pile10k-run7.json
build/tmix-destination-base-recovery-layer30-pile10k-run7.log
```
