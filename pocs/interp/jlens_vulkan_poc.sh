#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    printf 'usage: %s MODEL OUTPUT_PREFIX\n' "$0" >&2
    exit 1
fi

model=$1
output_prefix=$2
binary=${JLEN_VK_POC_BINARY:-build/bin/llama-interp-jlens-readout}
vmem_kib=${JLEN_VK_POC_VMEM_KIB:-30000000}
n_gpu_layers=${JLEN_VK_POC_N_GPU_LAYERS:-2}
layer=${JLEN_VK_POC_LAYER:-59}
samples=${JLEN_VK_POC_SAMPLES:-1}
min_future=${JLEN_VK_POC_MIN_FUTURE:-0}
future_window=${JLEN_VK_POC_FUTURE_WINDOW:-63}
visible_devices=${JLEN_VK_POC_VISIBLE_DEVICES:-}

if [[ ! -x $binary ]]; then
    printf 'missing readout executable: %s\n' "$binary" >&2
    exit 1
fi

# One process owns all Vulkan allocations. The readout retains only endpoint captures and
# one reusable FP16 RWKV state; it never retains per-sample activations or states.
ulimit -v "$vmem_kib"
if [[ -n $visible_devices ]]; then
    export GGML_VK_VISIBLE_DEVICES=$visible_devices
fi
exec "$binary" \
    -m "$model" \
    -ngl "$n_gpu_layers" \
    --layer "$layer" \
    --samples "$samples" \
    --min-future "$min_future" \
    --future-window "$future_window" \
    --epsilon 0.20 \
    --compare-epsilon 0.10 \
    --repeat-plus \
    --output "$output_prefix"
