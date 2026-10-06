RWKV7 relation trajectory dataset, version 1

Purpose

Study how existing memory changes the encoding of a later fact, and how a location-qualified query retrieves one of two people with the same name. No fixed binding direction or additive semantic decomposition is assumed. Collection applies no steering or donor-state transplantation.

Design

32 crossed histories: repeated name (Alice/Alice) versus distinct names (Alice/Bob), two location pairs (Japan/Canada and France/Brazil), two color pairs (yellow/blue and green/purple), both assignments, and both sentence orders. Another 16 histories use shared yellow or shared blue attributes, balanced over identity, location, and order.

Each history has nine segments: the story, two questions about the original story, an update to person 0 followed by both questions, and an independent update to person 1 followed by both questions. Both updates change a hat to red. This yields 48 histories, 96 update branches, 288 retrieval questions, and 432 saved segments. Update branches begin at the original story; questions begin at their named parent, not after another question.

The same update text is reused across matched histories with different prior assignments. Within a history, either query begins from the same parent state. These are the comparisons that test history-dependent writing and query-dependent retrieval.

This is a controlled seed dataset. Names and query templates are fixed; it cannot establish generalization to held-out names, paraphrases, pronouns, late identity revelation, or unrelated relation types. Matching lexical inventories does not guarantee matching latent representations. Raw-state PCA or classifier success alone does not identify a binding mechanism.

Execution

Local trained rwkv7-g1k-13.3b-Q4_K_M.gguf, CPU, eight threads, F32 recurrent states, n_batch = n_ubatch = n_seq_max = 1, rollback disabled. Every saved step is a one-token decode. No sampling, chat template, inserted BOS, or generated reasoning. Each segment is tokenized independently with add_special=false and parse_special=false; concatenating the segment texts and tokenizing once can give different boundary tokens. tokens.tsv is authoritative for the evaluated input. Newlines in plan.tsv are encoded as literal backslash-n and decoded before tokenization.

Candidate scores are teacher-forced log probabilities of a space plus each candidate color after Answer:. All candidate tokens contribute. Scores use the same restored checkpoint and parent logits; candidate completions do not contaminate the next experiment. They measure forced-completion retrieval among the supplied colors, not unrestricted generation or chat accuracy.

Storage

The dataset directory contains config.tsv, prompts.jsonl, tokens.tsv, segments.tsv, scores.tsv, provenance.json, observer-control.txt, and a directory per segment. DONE at the dataset root means collection finished. DONE in a segment means its arrays finished writing. Read only complete segments. Collection rejects an existing output directory to avoid mixing runs. This version does not resume interrupted collection.

All arrays are little-endian IEEE float32, C order, with no binary headers. config.tsv records dimensions and selected layers. The provided reader verifies file sizes and opens arrays read-only using numpy.memmap. On this model the selected layers are 0, 15, 30, 45, 60. All 61 layers are saved at segment boundaries and in vector trajectories.

Per segment, with T input tokens, L model layers, E embedding width, H head count, D head size, and K selected layers:

- wkv.f32: [T+1, K, H, D_value, D_key]. Index 0 is the incoming state; index t+1 is the state after token t. This is persistent S, not a transient activation. Native flat storage has key coordinate fastest.
- shifts.f32: [T+1, K, 2, H, D]. Shift index 0 is attention; 1 is FFN.
- vectors.f32: [T, L, 11, H, D]. Fields in order: r, w, k, v, a, b, wkv_read, group_norm, attn_norm, ffn_norm, layer_out.
- logits.f32: [T, vocabulary_size], next-token logits after each input token. Vocabulary size is inferred from file length by the reader.
- boundary.f32: [L, E*(D+2)]. Per layer: flattened WKV, attention shift, FFN shift. Reader exposes separate shaped views.

r/w/k/v/a/b are the actual fused WKV7 kernel inputs. In particular w is multiplicative decay; a=-normalized_key and b=normalized_key*learned_update_rate. They are not the raw projections or the sigmoid tensor called a in the graph builder. wkv_read is the fused kernel's output before group normalization. group_norm is normalized per-head output before learned affine parameters, the additional current-token term, gating, and output projection. attn_norm/ffn_norm are the named normalized block inputs; layer_out is the residual stream after the complete block. Gates and the separate current-token output term are not independently dumped in this version.

The full initial state for a non-root segment is its parent's boundary.f32. The root story begins from zero state. Thus arbitrary layers can be analyzed at segment boundaries; exact per-token matrix trajectories are directly available at selected layers. For other layers they can be reconstructed from the initial matrix and the saved recurrence operands, subject to accumulated floating-point error. Direct selected-layer trajectories provide a reconstruction check. Capturing all layers at every token is a possible higher-storage follow-up.

Kernel equation, using column vectors and native [value,key] matrix axes:

    S_new = S_old * w[None,:] + v[:,None] * k[None,:] + (S_old @ a)[:,None] * b[None,:]
    y = S_new @ r

Apply this independently for each head and layer. The old-memory transformation and new outer-product write are separately computable without interpreting S_new-S_old as a semantic feature. Across tokens and layers, the operands themselves depend on context. This equation is not a claim that semantic entities have fixed keys.

Use

    python3 examples/rwkv-interp/relations/make_plan.py build/interp/relations
    cmake -S examples/rwkv-interp -B build/rwkv-interp -DLLAMA_BUILD_DIR="$PWD/build"
    cmake --build build/rwkv-interp
    build/rwkv-interp/rwkv-interp-collect MODEL.gguf build/interp/relations/plan.tsv build/interp/relations/data-v2

Copy prompts.jsonl into the new dataset directory after collection, then validate and summarize:

    python3 examples/rwkv-interp/relations/analyze.py build/interp/relations/data-v2

Interactive Python, with examples/rwkv-interp/relations on sys.path:

    from data import Dataset
    ds = Dataset("build/interp/relations/data-v1")
    t = ds.trajectory("h000_u0")
    S_before = t["wkv"][:-1, 0]   # selected layer 0, every update token
    k = t["vectors"]["k"][:, 0]  # all heads at model layer 0
    final = ds.boundary("h000_u0")["wkv"]

Validation

analyze.py checks every saved segment: selected-layer recurrence reconstruction, readout reconstruction, group normalization, exact agreement between final token and boundary dumps, and exact agreement between branch initial states and parent boundaries. It also records query correctness by identity condition and whether the queried person was updated or preserved. query-results.jsonl retains every candidate score and correct/incorrect label so failures can be analyzed rather than silently discarded.

The collector additionally checks that tracing enabled versus disabled gives exactly identical final logits and a full serialized checkpoint for the first story. This is an observer check on one trajectory, not an exhaustive backend guarantee. Only CPU is validated.

Suggested first analysis

Compare the same update tokens across different prior assignments; examine both changes in the write operands and the old-memory transformation. Then compare the two question trajectories from the same parent, including all intermediate query tokens. Condition conclusions on successful retrieval, but retain failed cases as a separate group. Treat whole matched history groups as the statistical unit: branches and questions are not independent observations. Do not random-split tokens or closely matched branches into train and test folds. Fixed names/templates limit what generalization can be claimed from this dataset.

Transported-write analysis

transport.py follows a segment's parent chain and propagates each token's rank-one write through subsequent observed transformations diag(w)+a*b^T. It returns one matrix per original write and that matrix's contribution to the final pre-normalization readout. The matrices sum to the observed terminal state within numerical tolerance, including layers without direct per-token matrix dumps.

    python3 examples/rwkv-interp/relations/transport.py build/interp/relations/data-v1 --segment h000_u0_q0 --layer 30 --head 7 --output build/interp/relations/transport-example.npz

This is bookkeeping with operands held at their observed values. It is not a causal attribution: removing an earlier write could change later operands. Contributions can cancel, and group normalization makes downstream output attribution nonadditive. This decomposition is useful for asking which writes are retained or transformed along the realized trajectory, without assuming fixed semantic directions.

Starting from zero, a head's matrix rank cannot exceed the number of token writes in exact arithmetic. Consequently, low rank on these short prompts is structurally expected and is not evidence for a small number of semantic entities. All saved matrices retain F32 precision; analysis can promote to float64 as the validation and transport scripts do.

    tokens = ds.tokens("h000_u0")
    # Each item contains input token ID, absolute position, and exact token-piece bytes.

Matched-history update profiles

    python3 examples/rwkv-interp/relations/matched_updates.py build/interp/relations/data-v1

This checks identical update token IDs and positions across the two assignment histories, then saves relative RMS differences for r/w/k/v/a/b at every layer. matched-update-profiles.npz retains each pair separately; matched-update-summary.json contains the pair mean. The normalization is the symmetric RMS operand magnitude. Differences establish observed history dependence, not a semantic explanation. Shared-color controls are excluded from this particular paired comparison.
