RWKV interpretability API prototype

Build against this fork's existing llama.cpp build:

    cmake -S examples/rwkv-interp -B build/rwkv-interp -DLLAMA_BUILD_DIR="$PWD/build"
    cmake --build build/rwkv-interp
    build/rwkv-interp/rwkv-interp-prove "$MODEL" --vulkan

The core remains on its existing C++ standard. Only this example requires C++20.
The adapter uses existing exported C APIs plus recurrent-memory internals. It is an experimental source API, not a stable ABI. The internal dependency is concentrated in session construction, locate(), and replacement detachment. A future C ABI can expose tensor descriptors and capture/replace operations behind this adapter after their semantics settle.

Usage (ctx is a configured llama_context):

    rwkv::session experiment(ctx);
    rwkv::session::request prompt{0, prompt_tokens};
    auto logits = experiment.prefill(std::span(&prompt, 1)).front().logits;
    const auto prefix = experiment.capture(0);

    experiment.restore(1, prefix);
    auto matrix = experiment.capture(1, 12, rwkv::state_kind::s);
    std::fill(matrix.values.begin(), matrix.values.end(), 0.0f);
    experiment.replace(1, std::span(&matrix, 1));

    auto steps = experiment.run(1, continuation_tokens);
    while (steps.next()) {
        const auto & step = steps.value();
        auto state = experiment.capture(step.seq, 12, rwkv::state_kind::wkv);
        // Capture or replace here before resuming the coroutine.
    }

    rwkv::prefix_cache prefixes(256*1024*1024);
    auto output = experiment.prefill_cached(2, full_prompt_tokens, prefixes);

State vocabulary and layout:

- state_kind::wkv and state_kind::s name the SAME persistent WKV matrix, stored in llama.cpp's s_l. There are not two independent state tensors. Its flat native F32 layout is [key coordinate, value coordinate, head], first coordinate fastest. Dimensions are [head_size, head_size, embedding_width/head_size].
- shift_att is the preceding attention-normalized input vector, embedding_width elements.
- shift_ffn is the preceding FFN-normalized input vector, embedding_width elements.
- Both shifts are packed in llama.cpp's r_l. They are not the transient receptance r.
- wkv_trace captures the transient WKV activation BEFORE group normalization, gating, and output projection. It is distinct from persistent WKV state. Records contain layer, microbatch ordinal, token count, sequence count, and owned F32 activation values, embedding coordinate fastest. Within each microbatch the native operation groups tokens by sequence.

Transient tracing:

    rwkv::wkv_trace trace(llama_model_n_layer(model));
    trace.selected_layers = {12, 13}; // empty selects all layers
    context_params.cb_eval = rwkv::wkv_trace::callback;
    context_params.cb_eval_user_data = &trace;
    // Create ctx with those parameters, then create session(ctx).
    trace.clear();
    experiment.prefill(requests);
    if (trace.error) std::rethrow_exception(trace.error);
    // trace.records now owns the selected activations.

Keep trace alive for the context lifetime. Clear it before each top-level operation. Layer indices are derived from RWKV operation order for pure RWKV6/7 graphs. This adapter rejects hybrid and other recurrent architectures. Sequence-ID/position mapping for ragged microbatches is not exposed yet; records report native dimensions and order, so use single-sequence traces when token identity is required. Another callback cannot be installed at the same time without composing callbacks.

Execution contracts:

- prefill accepts multiple independent sequence requests and automatically chunks prompts beyond n_batch. It returns owned final-token logits in request order. Duplicate sequence IDs and empty prompts are rejected.
- run owns its continuation token list and yields after each one-token decode. It is cooperative and synchronous: next() computes one step on the caller's thread. It does not promise asynchronous GPU work or parallel execution. Destroying a suspended stream cancels future steps. The context and session must outlive streams; do not execute one context concurrently from multiple threads.
- capture(seq) is an owning sequence checkpoint including position metadata. restore can target another sequence using the same live model and compatible context configuration. It does not include sampler/RNG state or logits. Checkpoints are trusted in-process objects, not a validated interchange format.
- capture(seq, layer, kind) copies only that component. capture_into reuses a caller-owned buffer. Values survive subsequent evaluation and context destruction.
- replace validates all edit sizes and layer/kind selectors before writing. It preserves position and all unselected components. A shared row or pending source/rollback row is materialized through existing sequence serialization before replacement. An already private row is written directly. Duplicate edits apply in order; the last wins. Backend failures or corrupt checkpoints are not transactional.
- Use n_rs_seq = 0 for intervention sessions. Old rollback planes describe an unedited trajectory and must not be used after intervention. Explicit checkpoints provide research branch/restore semantics.
- Full-hit prefix reuse restores both checkpoint and saved logits without decoding. Partial hits restore the longest matching token prefix and decode the suffix. Cache keys include live model identity and exact tokens. Entries are immutable and evicted FIFO under a payload byte budget (allocator overhead excluded). Cache scope must be one model/configuration/adapter setup and one unmodified baseline trajectory; create separate caches for interventions or changed adapters. Cache entries are not persistent across model reloads.
- Persistent captures are after the last token of a completed decode. The fused WKV kernel exposes only terminal persistent state for a microbatch. For per-token persistent trajectories or interventions use run(); adding kernel-side selected-token snapshots is a separate optimization.

Validation target and evidence:

prove.cpp runs on the local trained RWKV7 g1k 13.3B Q4_K_M GGUF, CPU, F32 recurrent state, eight threads, four sequences, rollback disabled. It checks checkpoint replay, no-op replacement, each component's zero intervention and readback, invalid-edit rejection, native batch equivalence, branch isolation including shared-row detachment, coroutine suspension/cancellation, prefix hits/suffix reuse/budget, long prefill chunking, and selected transient layers.

Identical batch branches and replay match exactly. Single-sequence vs two-sequence execution has a measurable numerical difference in this model/build: the native API control matches the wrapper batch exactly. Do not assume batching is numerically transparent for causal experiments. Fix batch shape and execution settings when comparing interventions. This prototype does not establish a cause for that difference.

The synthetic-model generator currently skips RWKV6/7. RWKV6 layout is accepted but unvalidated. GPU transfers use backend tensor APIs. The same validation runner accepts --vulkan to test the shared adapter with Vulkan; collector and intervention examples have also been exercised on Vulkan. This does not validate other GPU backends. Selective activation callbacks can add synchronization overhead. capture_into avoids host allocation, but still synchronizes and transfers the selected state.

These are numerical API checks, not evidence that an object relation has been localized. For object-relation work, compare matched donor/recipient prompts, transfer selected state at the same token boundary, and measure relation-specific continuation logits against no-op and unrelated-donor controls using the same batch shape.

The read-query/ directory contains full-layer capture, intervention and analysis examples. Its README.txt provides reproduction commands. Use these as examples of the shared adapter rather than as established explanations of relational retrieval. Model weights stay resident across cases within each process.
