# RWKV J-Lens

`llama-interp-jlens-build` estimates a low-rank, strict-future Jacobian operator from
`rwkv.layer.L.resid.out` to the final residual. It writes FP32 Rademacher probe
directions, their averaged FP32 responses, a manifest, and the exact corpus line and
position samples used for the build and held-out validation.

## Build

Configure and build the tool:

```bash
cmake -S . -B build
cmake --build build --target llama-interp-jlens-build -j 2
```

Run a bounded pilot on a line-delimited corpus:

```bash
JLEN_VK_BUILD_VMEM_KIB=40000000 \
JLEN_VK_BUILD_N_GPU_LAYERS=24 \
JLEN_VK_BUILD_LAYER=37 \
JLEN_VK_BUILD_RANK=64 \
JLEN_VK_BUILD_SAMPLES_PER_DIRECTION=16 \
JLEN_VK_BUILD_VALIDATION_SAMPLES=64 \
bash pocs/interp/jlens_vulkan_build.sh MODEL CORPUS OUTPUT_PREFIX
```

The corpus must have enough independent lines for both partitions. The builder assigns
one in every five lines to validation using its recorded split seed; pass a larger
corpus for a scientific result. `jlens_smoke_corpus.txt` is only an end-to-end fixture.

## Hardware

Use one Vulkan process at a time. The BF16 model is about 25.5 GiB, so it cannot be
fully offloaded to the current 11.4 GiB free discrete Vulkan heap. The previous 30 GiB
virtual-memory limit also prevented deeper suffixes because CPU-mapped weights and
Vulkan mappings share process address space.

On the current machine, `-ngl 24` runs layers 38--60 on Vulkan with a 9.74 GiB model
buffer and supports source layer 37 when `JLEN_VK_BUILD_VMEM_KIB=40000000`. Lower
layers require a larger Vulkan heap. On a Strix Halo machine, first select the intended
device with `JLEN_VK_BUILD_VISIBLE_DEVICES`, then increase `JLEN_VK_BUILD_N_GPU_LAYERS`
gradually. Use `999` only after confirming that the Vulkan heap exceeds model weights,
recurrent state, and compute buffers.

## Acceptance

Treat a build as a numerical artifact only when its manifest shows low symmetry error,
exact repeat-plus replay, and strong epsilon agreement. Then increase rank and samples
until held-out `validation_mean_cosine` and `validation_mean_relative_l2` stabilize.
The rank-1 smoke result is expected to have poor held-out operator accuracy and must
not be interpreted as a usable lens.

The stored operator is applied as:

```text
Jhat(x) = (input_dimension / rank) * sum_k response[k] * dot(direction[k], x)
```

The remaining reader work is to apply `Jhat` to captured activations and route the
final residual through RWKV's production final normalization and output head.
