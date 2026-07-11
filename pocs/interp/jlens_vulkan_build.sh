#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    printf 'usage: %s MODEL CORPUS OUTPUT_PREFIX\n' "$0" >&2
    exit 1
fi

model=$1
corpus=$2
output_prefix=$3
binary=${JLEN_VK_BUILD_BINARY:-build/bin/llama-interp-jlens-build}
vmem_kib=${JLEN_VK_BUILD_VMEM_KIB:-30000000}
n_gpu_layers=${JLEN_VK_BUILD_N_GPU_LAYERS:-2}
layer=${JLEN_VK_BUILD_LAYER:-59}
rank=${JLEN_VK_BUILD_RANK:-4}
samples_per_direction=${JLEN_VK_BUILD_SAMPLES_PER_DIRECTION:-2}
validation_samples=${JLEN_VK_BUILD_VALIDATION_SAMPLES:-2}
min_future=${JLEN_VK_BUILD_MIN_FUTURE:-0}
future_window=${JLEN_VK_BUILD_FUTURE_WINDOW:-63}
visible_devices=${JLEN_VK_BUILD_VISIBLE_DEVICES:-}

if [[ ! -x $binary ]]; then
    printf 'missing J-lens builder executable: %s\n' "$binary" >&2
    exit 1
fi
if [[ ! -r $corpus ]]; then
    printf 'unreadable corpus: %s\n' "$corpus" >&2
    exit 1
fi

# One process owns Vulkan allocations; retained data is only rank-sized FP32 operator rows.
ulimit -v "$vmem_kib"
if [[ -n $visible_devices ]]; then
    export GGML_VK_VISIBLE_DEVICES=$visible_devices
fi
exec "$binary" \
    -m "$model" \
    -ngl "$n_gpu_layers" \
    --layer "$layer" \
    --corpus "$corpus" \
    --output "$output_prefix" \
    --rank "$rank" \
    --samples-per-direction "$samples_per_direction" \
    --validation-samples "$validation_samples" \
    --validation-modulo 5 \
    --max-corpus-tokens 64 \
    --min-future "$min_future" \
    --future-window "$future_window" \
    --epsilon 0.20 \
    --compare-epsilon 0.10 \
    --repeat-plus
