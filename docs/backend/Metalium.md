# llama.cpp for Metalium

## Table of Contents

- [Background](#background)
- [llama.cpp + Metalium](#llamacpp--metalium)
- [Note on current limitations](#note-on-current-limitations)
- [Hardware](#hardware)
- [DataType Supports](#datatype-supports)
- [Environment Variable](#environment-variable)

> [!IMPORTANT]
> Although the author is a Tenstorrent employee, this backend is not an official Tenstorrent product. It started by the author pre-joining Tenstorrent and is developed in the author's free time. It is not supported by Tenstorrent, but the author is happy to help with issues and questions on the [Tenstorrent Discord server](https://discord.gg/tenstorrent). Just don't expect official support or bug fixes from Tenstorrent.

## Background

Tenstorrent produces a range of ASICs with very scalable design that enables efficient inference of AI models.

**Metalium and TTNN** is the default low level and operator library developed by Tenstorrent (analogous to CUDA and cuDNN + ATen). They are the critical part of executing neural network computation on Tenstorrent devices and provides high level primitives for scaling to multiple connected Tenstorrent processors.

This backend tries to call operators in TTNN when possible. If an operator is not supported by TTNN, custom kernels in Metalium are implemented to keep operations and data on-device.

> [!NOTE]
> In Tenstorrent's documentation and codebase, Metalium (officially TT-Metalium) is sometimes abbreviated as Metal. This should not be confused with Apple's Metal API, as they are entirely distinct technologies developed by different companies, designed for different hardware, and built on completely different architectures. To avoid confusion, this backend consistently uses the name Metalium. If you encounter any instance where "Metal" is mentioned in the context of this backend, it should be understood as referring to Metalium.

### llama.cpp + Metalium

The llama.cpp Metalium backend is designed to enable inference on Tenstorrent's Wormhole processors and newer. As experimental software in its early stages, it uses TTNN for both tensor management and its operator library. When TTNN does not support a specific operation, the backend implements custom kernels that run directly on Metalium to maintain on-device computation.

### Note on current limitations

As mentioned, the Metalium backend is experimental software. Thus features will be developed and enabled over time. As of writing the documentation, the following limitations applies:

* Only one device is exposed at a time
    * Multi device scaling is handled using TTNN's native scaling (to be implemented)
    * See the `GGML_METALIUM_DEVICE_ID` and `GGML_METALIUM_MESH_SHAPE` environment variables below
* KV Cache has to be stored on the CPU (via the `-nkvo` flag)
* FP32 tenssors is emulated by internally using BFP16 (The matrix unit does not support FP32 natively, which is the bulk of compute, though the vector unit does support FP32)

### Dependencies

TBD. I don't have a formal list of what is needed for now. But you need to get TTNN/Metalium built and installed. And make the dependencies available to the llama.cpp build system.

### Building and using the backend

There is no "supported" TTNN versions Metalium and TTNN is still a moving target. Instead, need and support for newer versions of TTNN is constantly updated in order to utilize new features and take in bug fixes. However, generally build the latest Metalium and TTNN from the [official repostory](https://github.com/tenstorrent/tt-metal) by following the steps

1. Setup you environment/driver following the [official guide](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
2. Build Metalium (and TTNN) with GCC and install to the build directory.
   * Officially GCC >= 12 and Clang >= 17 is supported.


```bash
cd /path/to/your/tt-metal
export TT_METAL_HOME=`pwd`
mkdir build
cd build
cmake .. -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DCMAKE_INSTALL_PREFIX=`pwd` -DCMAKE_BUILD_TYPE=Release -G Ninja
ninja

# Installs to the build directory
ninja install
```

3. Build llama.cpp with `GGML_METALIUM=ON`. It will read the `TT_METAL_RUNTIME_ROOT` variable above and error if not  detected. Likewise, the backend needs the environmental variables to function.

```bash
cd /path/to/your/llama.cpp
export TT_METAL_RUNTIME_ROOT="${TT_METAL_HOME}"
mkdir build
cd build
cmake .. -DGGML_METALIUM=ON -DCMAKE_BUILD_TYPE=Release
make -j16
```

4. Tenstorrent devices are treated as a GPU in llama.cpp. Use `-ngl` to set number of layers offloaded.

**NOTE:** add the `-nkvo` flag to stop the KV cache being offloaded

```bash
bin/llama-cli -ngl 23 -m tinyllama-1.1b-chat-v1.0.Q4_0.gguf -p "The solution to Riemann hypothesis is" -nkvo
```

> [!IMPORTANT]
> TTNN compiles and sometimes JITs kernel on the fly. Leading to longer initialization time and first token generation time when the kernel cache is cold.

## Hardware

### Hardware support

The following hardware are tested.

| Tenstorrent Device            | Status  |
|:-----------------------------:|:-------:|
| Wormhole N300                 | Tested  |
| Wormhole QuietBox             | Tested  |

## Data Type Supports

Besides the standard FP32 and BFP16 floating point support. Tenstorrent processors support their own native quantized data types (BFLOAT8_B, BFLOAT4_B, etc.). Thus, weights and activations are automatically converted to supported native types. The conversion is as follows:

| GGML Type             | Metalium DataType          |
|-----------------------|----------------------------|
| GGML_TYPE_F32         | BFLOAT16                   |
| GGML_TYPE_F16         | BFLOAT16                   |
| GGML_TYPE_Q4_0        | BFLOAT8_B*                 |
| GGML_TYPE_Q4_1        | BFLOAT8_B*                 |
| GGML_TYPE_Q5_0        | BFLOAT8_B                  |
| GGML_TYPE_Q5_1        | BFLOAT8_B                  |
| GGML_TYPE_Q8_0        | BFLOAT8_B                  |
| GGML_TYPE_Q8_1        | BFLOAT8_B                  |
| GGML_TYPE_Q2_K        | Unsupported                |
| GGML_TYPE_Q3_K        | BFLOAT4_B                  |
| GGML_TYPE_Q4_K        | BFLOAT4_B                  |
| GGML_TYPE_Q5_K        | BFLOAT8_B                  |
| GGML_TYPE_Q6_K        | BFLOAT8_B                  |
| GGML_TYPE_Q8_K        | BFLOAT8_B                  |
| GGML_TYPE_IQ2_XXS     | Unsupported                |
| GGML_TYPE_IQ2_XS      | Unsupported                |
| GGML_TYPE_IQ3_XXS     | Unsupported                |
| GGML_TYPE_IQ1_S       | Unsupported                |
| GGML_TYPE_IQ4_NL      | Unsupported                |
| GGML_TYPE_IQ3_S       | Unsupported                |
| GGML_TYPE_IQ2_S       | Unsupported                |
| GGML_TYPE_IQ4_XS      | Unsupported                |
| GGML_TYPE_I8          | Unsupported                |
| GGML_TYPE_I16         | Unsupported                |
| GGML_TYPE_I32         | UINT32                     |
| GGML_TYPE_I64         | Unsupported                |
| GGML_TYPE_F64         | Unsupported                |
| GGML_TYPE_IQ1_M       | Unsupported                |
| GGML_TYPE_BF16        | BFLOAT16                   |
| GGML_TYPE_Q4_0_4_4    | Unsupported                |
| GGML_TYPE_Q4_0_4_8    | Unsupported                |
| GGML_TYPE_Q4_0_8_8    | Unsupported                |
| GGML_TYPE_TQ1_0       | Unsupported                |
| GGML_TYPE_TQ2_0       | Unsupported                |
| GGML_TYPE_IQ4_NL_4_4  | Unsupported                |
| GGML_TYPE_IQ4_NL_4_8  | Unsupported                |
| GGML_TYPE_IQ4_NL_8_8  | Unsupported                |
| GGML_TYPE_MXFP4       | BFLOAT4_B                  |

`Q4_0` and `Q4_1` are set to BFLOAT8 due to numerical precision issues. It is adviced to use Q4_K_* for better performance.

## Environment Variable

### (CMake) Build flags

| Variable Name               | Default Value                        | Description                                                          |
|-----------------------------|--------------------------------------|----------------------------------------------------------------------|
| GGML_METALIUM               | OFF                                  | Enable building the Metalium backend                                 |
| GGML_METALIUM_EMBED_KERNELS | ON                                   | If compute kernels should be embeded into the executable             |


### Runtime variables

| Variable Name             | Value                                | Description                                                                                                |
|---------------------------|--------------------------------------|------------------------------------------------------------------------------------------------------------|
| TT_METAL_HOME             | string  (mandatory)                  | Path to the root of the tt-metal repository                                                                |
| TT_METAL_RUNTIME_ROOT     | string  (mandatory)                  | Path to the root of the root of the runtime directory (or, the repo dir)                                   |
| GGML_METALIUM_DEVICE_ID   | integer                              | ID of the device to use (single device). 0 is assumed if not set                                           |
| GGML_METALIUM_MESH_SHAPE  | string                               | Shape of the device mesh for clustering (ex: 2x4 for 2x4 mesh)                                             |
| GGML_METALIUM_KERNEL_ROOT | string                               | Root of the Metalium kernel library in case running from weird places and you don't have embedded kernels  |

> [!NOTE]
> `GGML_METALIUM_DEVICE_ID` and `GGML_METALIUM_MESH_SHAPE` cannot be set at the same time.

> [!NOTE]
> Clustering is in early stage development. The option exists for development purpose.

### Debug flags

There are several debug flags available to assist with debugging/performance of the backend. These flags are triggered by setting environment variables and will be removed eventually.

| Variable Name                     | Value           | Description                                                                                                                                                              |
|-----------------------------------|-----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| GGML_METALIUM_PRINT_REJECTED_OPS  | 0(default) or 1 | Print operators GGML asked if the Metalium backend can run, and Metalium reported false                                                                                  |
| GGML_METALIUM_PRINT_VIEW          | 0(default) or 1 | Print all view operations (VIEW, TRANSPOSE, RESHAPE, PERMUTE) that the backend's lazy view system sees                                                                   |
|GGML_METALIUM_DISABLE_PROGRAM_CACHE| 0(default) or 1 | Disables TTNN program cacheing                                                                                                                                           |
| GGML_METALIUM_EXPERIMENTAL_OPS    | 0(default) or 1 | Enables experimental ops that is known to cause trouble                                                                                                                  |

## Known issues

- The backend cannot exit without leaking some memory (gets handled by OS) due to destruction order issues.

## For developers

The following is a very brief and evolving deisgn doc as things develop. Generally:

Most tensors are tiled in the backend. Unless they are indices, KV cache (yet to implement) or embed weights (yet to implement). Which are row major to enable efficient scatter/gather within a single chip.

Though llama.cpp encourages a c-with-classes coding style. TTNN by it's nature is written in very modern C++. The backend also reflects this:

1. GGML interfacing code is written in a c-with-classes style
2. Infrastructure interacting with TTNN, new operators, Metalium utilties are written in modern C++ (up to C++20 which is what TTNN/Metalium uses)

Due to hardware design, most operations are pratically limited to an accuracy BFP16. Which seems to be enough for most models. And so for now FP32 support is emulated with using BFP16 underneath.

### Note on `GGML_MUL_MAT`

`GGML_MUL_MAT` is a weird operation, `mul_mat(a, b)` is in fact evaulating `tranpose(matmul(a, b)) = matmul(tranpose(b), tranpose(a))` since `b` is pre-tranposed on GGML. It effectivly evaulates `matmul(bT, tranpose(a))`

The backend evaluates this through native TTNN matmul as `ttnn.matmul(b, a, transpose_b=true)`.
