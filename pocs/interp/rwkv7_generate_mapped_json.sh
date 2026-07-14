#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf '%s\n' \
        'usage: rwkv7_generate_mapped_json.sh --model MODEL --activations ACTIVATIONS.root --map-prefix PREFIX --json OUTPUT [-p PROMPT] [--steps N] [--top N] [-ngl N]' \
        '' \
        'Fits any missing all-layer K-to-residual maps from ACTIVATIONS.root, then writes one' \
        'live generation trace containing mapped time.k and captured resid.out for every layer.' \
        'Map files are PREFIX0.rwkv-lrm through PREFIX60.rwkv-lrm.'
}

model=''
activations=''
map_prefix=''
json=''
prompt='The Eiffel Tower is located in'
steps=100
top=5
gpu_layers=0

while (($#)); do
    case "$1" in
        -m|--model) model="$2"; shift 2 ;;
        --activations) activations="$2"; shift 2 ;;
        --map-prefix) map_prefix="$2"; shift 2 ;;
        --json) json="$2"; shift 2 ;;
        -p|--prompt) prompt="$2"; shift 2 ;;
        --steps) steps="$2"; shift 2 ;;
        --top) top="$2"; shift 2 ;;
        -ngl) gpu_layers="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 1 ;;
    esac
done

if [[ -z "$model" || -z "$activations" || -z "$map_prefix" || -z "$json" || "$steps" -le 0 || "$top" -le 0 ]]; then
    usage >&2
    exit 1
fi

repo_root="$(cd -- "$(dirname -- "$0")/../.." && pwd)"
readout="$repo_root/build/bin/llama-rwkv-activation-linear-readout"
lens="$repo_root/build/bin/llama-rwkv7-time-lens"
if [[ ! -x "$readout" || ! -x "$lens" ]]; then
    printf 'Build llama-rwkv-activation-linear-readout and llama-rwkv7-time-lens first.\n' >&2
    exit 1
fi

missing_maps=0
for layer in {0..60}; do
    [[ -f "${map_prefix}${layer}.rwkv-lrm" ]] || missing_maps=1
done

if ((missing_maps)); then
    for layer in {0..60}; do
        "$readout" -m "$model" --input "$activations" --compact-layer-pairs \
            --source "$layer:time.k" --target "$layer:resid.out" \
            --output "${map_prefix}${layer}.fit.tsv" --holdout-bucket 0 \
            --write-map "${map_prefix}${layer}.rwkv-lrm" --fit-only
    done
fi

"$lens" -m "$model" -p "$prompt" --steps "$steps" --top "$top" -ngl "$gpu_layers" \
    --project-sources k,resid.out --linear-readout-map-prefix "$map_prefix" --json "$json"
