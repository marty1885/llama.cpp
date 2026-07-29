# TMix Relative-Geometry Null Results

## Classified Run

The classified run used the frozen Project Gutenberg ebook 73544 corpus and the
registered prompt-only split in `frozen/registration.json`. The model was
`rwkv7-g1h_preview4743-13.3b-20260703-ctx8192-Q4_K_M.gguf` on Vulkan with
`-ngl 99`. All three layer-15 F32 production snapshots passed the backend
verifier before collection.

Artifacts:

```text
build/tmix-relative-geometry-null-classified-capture.root
build/tmix-relative-geometry-null-classified.root
build/tmix-relative-geometry-null-classified.json
```

The classified artifact status is `valid` and its mechanical classification is
`relative_geometry_null_rejected`.

```text
G_ambient_native:   0.393052773619
G_transport_native: 0.389749780036
```

All twelve registered comparisons exceeded their corresponding control 95th
percentile and had empirical p-values at or below 0.05. The least-separated
comparison was transported within-prompt self: native `0.389749780036`,
control p95 `0.389153978474`, and p-value `0.03`.

The post-write audit reopened the ROOT artifact, verified all required
RNTuples, verified 99 finite rows in each of the twelve control groups, and
independently recomputed their p95 thresholds and p-values from serialized
control rows. `git diff --check` passed.

## Permitted Claim

At layer 15 on the frozen held-out corpus, native residual/write pairing
produced rank-32 canonical tangent organization in both ambient and
common-reference coordinates that exceeded complete global, within-prompt,
and isotropic finite-add controls. This rejects the tested
incidental-relative-geometry null and supports continued investigation of
relative coordinates.

This result does not establish semantic content, a memory rotor, FFN causal
use, or an output-coordinate decoding map.
