# TMix Relative-Geometry Exploratory Layer Follow-Up Results

This is an exploratory follow-up on the same frozen corpus as the classified
layer-15 run. It is not a confirmation study and does not establish a depth
trend.

## Artifacts

```text
build/tmix-relative-geometry-null-exploratory-layers-capture.root
build/tmix-relative-geometry-null-exploratory-layer30.{root,json}
build/tmix-relative-geometry-null-exploratory-layer45.{root,json}
build/tmix-relative-geometry-null-exploratory-layer60.{root,json}
```

Each artifact has status `exploratory`, confirmation disabled, and a complete
serialized-row audit: all twelve control groups contain 99 finite values, and
all p95 thresholds and p-values were independently recomputed from ROOT rows.

## Descriptive Results

```text
layer    G_ambient_native    G_transport_native    exploratory label
30       0.376565113915      0.360181704631        relative_geometry_null_rejected
45       0.591554697303      0.526803662012        relative_geometry_null_rejected
60       0.953604572558      0.934751748162        relative_geometry_null_rejected
```

At each selected layer, both coordinate descriptions exceeded all global,
within-prompt, and isotropic self and fixed-native-basis controls. All twelve
empirical p-values per layer were `0.01`.

## Interpretation Boundary

These results describe this one frozen corpus and the selected layers only.
They neither confirm a network-wide pattern nor establish a monotonic depth
trend. They do not establish semantic content, a memory rotor, FFN causal use,
or an output-coordinate decoding map.
