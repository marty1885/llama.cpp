# RWKV Direct-Unembedding Trace

`llama-interp-rwkv-unembed` can greedily generate while directly unembedding named
intermediate activations through the model's production final norm and output head.

## All-Layer Channel Trace

`--all-channel-out` captures `rwkv.layer.L.channel.out` for every RWKV layer at every
generation step. `--json` writes a self-contained schema-versioned trace for the static
viewer. `--until-eog` stops at end-of-generation, with a default 128-token safety cap.
`--all-time-out` similarly captures the attention branch's residual-space contribution,
`rwkv.layer.L.time.out`. `--all-time-mix` captures the eight 4,096-dimensional
time-mix vectors (`r`, `w`, `k`, `v`, `a`, `g`, `wkv`, and `rkv`) from every layer.
The all-layer options are mutually exclusive.

With `--contrast PROMPT`, `--contrast-activation-delta` additionally reads the
source activation difference (prompt minus contrast) through the final norm and
output head. This differs from the default contrast report, which subtracts the
two separately normalized output-logit vectors.

`--center-root CENTROIDS.root` is a static `--all-time-mix` diagnostic. It loads one
`K=1` ROOT centroid per raw source, subtracts it before final-norm/output-head
readout, and leaves the final residual uncentered as a control. It cannot be combined
with contrast, generation, or JSON traces.

Generation readouts are packed into one F32 residual matrix and passed through a single
GPU final-norm/output-head readout. This turns per-source output-head GEMVs into one batched
GEMM and does not evaluate RWKV recurrent blocks for the readout rows. The all-time-mix
mode reserves a 512-row batch, enough for its 488 RWKV sources plus the final-residual control
in this 61-layer model.

```bash
build/bin/llama-interp-rwkv-unembed \
  -m MODEL.gguf -ngl 999 \
  -p "The Itanium processor" \
  --until-eog --all-channel-out --top 5 \
  --json /tmp/itanium-channel-trace.json
```

Open `pocs/interp/rwkv_unembed_viewer.html` in a browser, then select or drop the JSON
file. When the trace is served by the same static web server, it can load it directly:

```text
http://localhost:8000/pocs/interp/rwkv_unembed_viewer.html?trace=relative-url-to-trace.json
```

The viewer presents a selectable, chat-like generation transcript on the left and the
full top-k direct readout for every layer at the selected emission on the right. It filters
to `channel.out` when those sources are present, but accepts smaller `--source` traces too.
It also accepts `llama-interp-jlens-operator --json` perturbation traces, presenting
separate positive and negative final-logit derivative cards for the selected raw source.

## JSON Schema

The root contains `schema_version`, model/run metadata, requested `sources`, and `runs`.
Each run records the prompt, stop reason, and ordered generation `steps`. A step contains
the carrier input token, native generated next token, final-residual round-trip error, and
each source's direct-unembedding top-k tokens (`rank`, token `id`, `piece`, and `logit`).
This preserves token pieces exactly, including whitespace and EOG, so consumers should not
split generated text on spaces.

## Interpretation

`channel.out` is the channel-mix/FFN contribution added to RWKV's residual stream. A
late-layer readout that predicts next-token-relevant concepts is therefore expected. Direct
unembedding is a convenient visualization of that contribution through the production final
head, not evidence of a separate hidden-language representation. Compare layer, timing, and
residual/output-head controls before assigning stronger meaning to a readout.
