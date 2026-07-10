#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    printf 'usage: %s MODEL OUTPUT_DIR [CORPUS_FILE]\n' "$0" >&2
    exit 1
fi

model=$1
output_dir=$2
corpus=${3:-}
binary=${JLEN_SWEEP_BINARY:-build/bin/llama-interp-jlens-readout}
samples=${JLEN_SWEEP_SAMPLES:-8}
max_corpus_tokens=${JLEN_SWEEP_MAX_CORPUS_TOKENS:-64}
vmem_kib=${JLEN_SWEEP_VMEM_KIB:-28000000}
layers=${JLEN_SWEEP_LAYERS:-"20 30 40 50"}
offsets=${JLEN_SWEEP_OFFSETS:-"0 1 2"}

if [[ $layers == "20 30 40 50" && $offsets == "0 1 2" && ${JLEN_SWEEP_ALLOW_FULL:-0} != 1 ]]; then
    printf 'refusing the full sweep without JLEN_SWEEP_ALLOW_FULL=1; run a bounded gate first\n' >&2
    exit 1
fi

if [[ ! -x $binary ]]; then
    printf 'missing readout executable: %s\n' "$binary" >&2
    exit 1
fi
if [[ -n $corpus && ! -r $corpus ]]; then
    printf 'unreadable corpus: %s\n' "$corpus" >&2
    exit 1
fi

mkdir -p "$output_dir"

run_readout() {
    local layer=$1
    local offset=$2
    local prefix="$output_dir/layer-${layer}-offset-${offset}"
    local min_future=${JLEN_SWEEP_MIN_FUTURE:-$offset}
    local future_window=${JLEN_SWEEP_FUTURE_WINDOW:-$offset}
    local -a args=(
        -m "$model"
        -ngl 0
        --layer "$layer"
        --samples "$samples"
        --min-future "$min_future"
        --future-window "$future_window"
        --epsilon 0.20
        --compare-epsilon 0.10
        --output "$prefix"
    )

    if [[ -n $corpus ]]; then
        args+=(--corpus "$corpus" --max-corpus-tokens "$max_corpus_tokens")
    fi

    # This includes mapped model pages, so a too-large model fails safely here.
    ulimit -v "$vmem_kib"
    # Prevent the scheduler from registering Vulkan compute backends as well as model offload.
    export GGML_VK_VISIBLE_DEVICES=
    exec "$binary" "${args[@]}"
}

for layer in $layers; do
    for offset in $offsets; do
        (
            run_readout "$layer" "$offset"
        )
    done
done
