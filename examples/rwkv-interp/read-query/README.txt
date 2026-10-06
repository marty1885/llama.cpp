RWKV relational retrieval experiments

These are private research tools. They use full-layer activations and existing callback/checkpoint APIs without changing the core inference implementation. Generated traces, results, model files and binaries stay outside this source directory.

Build from the repository root, after building llama.cpp with Vulkan:

    cmake -S examples/rwkv-interp -B build/rwkv-interp -DLLAMA_BUILD_DIR="$PWD/build"
    cmake --build build/rwkv-interp -j 4

Python analysis requires numpy, pandas and matplotlib. Dataset and recurrence helpers live in ../relations. Commands below run from the repository root. Set MODEL to the RWKV7 GGUF path; the tested model was rwkv7-g1k-13.3b-Q4_K_M.gguf. Every output directory must be fresh.

Capture the crossed story/query baselines:

    build/rwkv-interp/rwkv-interp-collect "$MODEL" examples/rwkv-interp/read-query/dot-plan.tsv build/read-query-baseline --vulkan
    OPENBLAS_NUM_THREADS=4 python3 examples/rwkv-interp/read-query/key_alignment.py build/read-query-baseline examples/rwkv-interp/read-query/dot-prompts.jsonl --full-layer

The plan contains eight stories crossing color assignment, city assignment and sentence order, with two city-qualified questions per story. The collector loads the model once and records fused recurrence operands, reads and downstream activations. Each question is independently restored from its story checkpoint. Questions end at the colon; answer scoring is forced choice between single-token yellow and blue. Original-key alignment is descriptive and does not account for key transport or establish a retrieval mechanism.

Exact write-to-read bookkeeping:

    OPENBLAS_NUM_THREADS=4 python3 examples/rwkv-interp/read-query/trace_reads.py build/read-query-baseline build/read-query-transport
    OPENBLAS_NUM_THREADS=4 python3 examples/rwkv-interp/read-query/summarize_trace.py build/read-query-baseline build/read-query-transport

Each historical value stays fixed while its key is transported by the observed recurrence. Summed write-origin contributions reconstruct the full read, checked at every recorded token/layer and against explicit matrix transport. Block structure implements the native recurrence; reported effects combine all 4096 coordinates. This is conditional bookkeeping, not causal attribution or a semantic label for a write origin.

Broad residual swap control:

    build/rwkv-interp/rwkv-interp-layer-swap "$MODEL" examples/rwkv-interp/read-query/dot-plan.tsv build/read-query-swaps
    python3 examples/rwkv-interp/read-query/summarize_swaps.py build/read-query-swaps

This swaps an entire layer output between opposed questions at city or colon positions, across every layer, with exact self controls. It localizes sensitivity but can transfer question identity directly; it does not explain higher-order retrieval.

Historical-read to later-query mediation:

    build/rwkv-interp/rwkv-interp-read-mediation "$MODEL" examples/rwkv-interp/read-query/dot-plan.tsv build/read-query-mediation
    python3 examples/rwkv-interp/read-query/summarize_mediation.py build/read-query-mediation

The source patch adds a donor-minus-recipient historical-read difference, preserving the immediate write/read term and packed recurrent state at that operation. Later interventions restore either full r or only r's effect on the historical-state read. The latter adds (S_new - v*k^T)*(r_reference-r_actual) to the read and preserves the packed state and other r paths. Source/mediator selection uses original-color cases; reversed-color cases are excluded from selection. Exact self and replay controls check the patch mechanics. Read dependence does not identify a person-valued intermediate.

Natural query reuse across contexts:

    build/rwkv-interp/rwkv-interp-reusable-query "$MODEL" examples/rwkv-interp/read-query/dot-plan.tsv build/read-query-reusable
    python3 examples/rwkv-interp/read-query/summarize_reusable.py build/read-query-reusable

Donors preserve the literal question while changing city assignment, color assignment, both, or sentence order. Full r and historical-read-only query substitutions are tested over all layers and individual layers. Recipient state is restored before each run and evolves normally thereafter; donor recurrent state is never copied. Failure rejects interchangeability of the tested natural query trajectories, not nonlinear or context-dependent joint lookup.

Read-to-query linearity:

    build/rwkv-interp/rwkv-interp-read-linearity "$MODEL" examples/rwkv-interp/read-query/dot-plan.tsv build/read-query-linearity
    GGML_VK_DISABLE_MMVQ=1 build/rwkv-interp/rwkv-interp-read-linearity "$MODEL" examples/rwkv-interp/read-query/dot-plan.tsv build/read-query-linearity-control
    OPENBLAS_NUM_THREADS=1 python3 examples/rwkv-interp/read-query/summarize_linearity.py build/read-query-linearity
    OPENBLAS_NUM_THREADS=1 python3 examples/rwkv-interp/read-query/summarize_linearity.py build/read-query-linearity-control

Place a protocol.txt describing the run in each output directory before analysis; the summarizer hashes it alongside source files and the prompt plan. For the numerical repeat, record GGML_VK_DISABLE_MMVQ=1. Binary/backend hashes can be recorded separately with sha256sum. This source-relative provenance differs from the original build-local scripts' hardcoded paths.

The runner adds full-vector deltas to the layer-34 read at city token 9 and measures full queries at layers 36,48,60 over tokens 9,10,11, plus the answer margin. Directions include recipient-specific city/color historical-read differences, fixed case-0 differences, norm-matched seeded random directions, and natural/random sums. Each run has 976 interventions, including 16 exact zero replays, with one resident Vulkan model. The experiment expects the tested 61-layer, 4096-coordinate model and twelve-token questions.

queries.f32 has shape [976,3,3,4096], ordered by scores.tsv row, token [9,10,11], layer [36,48,60], coordinate. directions.f32 has shape [16,8,4096]; config.txt identifies directions. Metrics include central small-step tangent predictions, derivative convergence, superposition, small-step quadratic fits, half-scale secants and shared directional-response transfer. Errors divide by the intervention-induced change, not baseline magnitude. Quadratic results at fitted small scales are not held-out validation.

The default AMD Vulkan matrix-vector path can quantize input activations to Q8_1. Disabling MMVQ retains Vulkan and the same stored quantized weights, but each mode computes its own baseline and directions. It does not recover original full-precision weights or isolate precision with an identical state.

Observed results on the fixed prompt family: with MMVQ disabled, natural-direction tangent errors at 0.1 magnitude were about 6% for the nearby query and 9% for the final query; at natural magnitude they rose to 43% and 95%. Natural-scale superposition errors were about 47% for the final query and 46% for answer-margin change. A shared small-step response did not transfer well across color assignments or sentence order. These support limited context-specific linear prediction on the tested path, not linear semantic representations or a unique retrieval mechanism.

Intervention callbacks check exact readback, unchanged packed source state and upstream operands, finite values and expected patch counts. All runs use one token, one sequence, F32 state and rollback disabled. Later computation and recurrent state may change. Baseline, direct-read, replay and unrelated-donor controls must be interpreted with the same execution settings. No population significance claim is supported by this small crossed set.
