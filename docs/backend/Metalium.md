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
    * Multi device scaling is handled using TTNN's native scaling
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

## DataType Supports

Besides the standard FP32 and BFP16 floating point support. Tenstorrent processors support their own native quantized data types (BFLOAT8_B, BFLOAT4_B, etc.). Thus, weights and activations are automatically converted to supported native types. The conversion is as follows:

| GGML Type             | Metalium DataType          |
|-----------------------|----------------------------|
| GGML_TYPE_F32         | BFLOAT16                   |
| GGML_TYPE_F16         | BFLOAT16                   |
| GGML_TYPE_Q4_0        | BFLOAT4_B*                 |
| GGML_TYPE_Q4_1        | BFLOAT4_B*                 |
| GGML_TYPE_Q5_0        | BFLOAT8_B                  |
| GGML_TYPE_Q5_1        | BFLOAT8_B                  |
| GGML_TYPE_Q8_0        | BFLOAT8_B                  |
| GGML_TYPE_Q8_1        | BFLOAT8_B                  |
| GGML_TYPE_Q2_K        | Unsupported                |
| GGML_TYPE_Q3_K        | BFLOAT4_B*                 |
| GGML_TYPE_Q4_K        | BFLOAT4_B*                 |
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
| GGML_TYPE_I32         | INT32                      |
| GGML_TYPE_I64         | Unsupported                |
| GGML_TYPE_F64         | Unsupported                |
| GGML_TYPE_IQ1_M       | Unsupported                |
| GGML_TYPE_BF16        | BFLOAT16                   |
| GGML_TYPE_Q4_0_4_4    | Unsupported                |
| GGML_TYPE_Q4_0_4_8    | Unsupported                |
| GGML_TYPE_Q4_0_8_8    | Unsupported                |
| GGML_TYPE_TQ1_0       | Unsupported                |
| GGML_TYPE_TQ2_0       | Unsupported                |

* All BFLOAT4_B types used to work but is emulated with BFLOAT8_B until upstream bug is fixed

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
| GGML_METALIUM_CACHE_MM_TRANSPOSE  | 0(default) or 1 | TTNN has limited support for pre-transposed matmul that GGML needs and does most on the fly. This options cache the transpose. Trades lot of memory for some performance |
|GGML_METALIUM_DISABLE_PROGRAM_CACHE| 0(default) or 1 | Disables TTNN program cacheing                                                                                                                                           |
| GGML_METALIUM_EXPERIMENTAL_OPS    | 0(default) or 1 | Enables experimental ops that is known to cause trouble                                                                                                                  |

## Know issues

- The backend cannot exit without leaking some memory (gets handled by OS) due to destruction order issues.

## For developers

The following is a very brief and evolving deisgn doc as things develop. Generally:

Most tensors are tiled in the backend. Unless they are indices, KV cache (yet to implement) or embed weights (yet to implement). Which are row major to enable efficient scatter/gather within a single chip.

Though llama.cpp encourages a c-with-classes coding style. TTNN by it's nature is written in very modern C++. The backend also reflects this:

1. GGML interfacing code is written in a c-with-classes style
2. Infrastructure interacting with TTNN, new operators, Metalium utilties are written in modern C++ (up to C++20 which is what TTNN/Metalium uses)

Due to hardware design, most operations are pratically limited to an accuracy BFP16. Which seems to be enough for most models. And so for now FP32 support is emulated with using BFP16 underneath.

### Note on `GGML_METALIUM_CACHE_MM_TRANSPOSE`

`GGML_MUL_MAT` is a weird operation, `mul_mat(a, b)` is in fact evaulating `tranpose(matmul(a, b)) = matmul(tranpose(b), tranpose(a))` since `b` is pre-tranposed on GGML. It effectivly evaulates `matmul(bT, tranpose(a))`

There's is 2 code paths that executed the `GGML_MUL_MAT` operation on device.

* By using TTNN `ttnn.matmul(b, ttnn.transpose(a))`
* By using a custom MUL_MAT written in Metalium kernels

The current custom MUL_MAT kernel is very barebones. But still faster then actually performing a transpose then matmul using TNN. However, if you are willing to sacrifice a lot of DRAM space - since `a` is the weight matrix, the backend can tranpose the weight matrix once and cache the result. This can be done by using the `GGML_METALIUM_CACHE_MM_TRANSPOSE` flag. It transposes the weight matrix once and caches the result for future use.

The eventual goal is to get rid of this flag since there's no reason custom kernels can't reach near the same performance (it's just pre-transpose, but we add a transpose stem to the matrix engine). But until then, it is recommended to use it for better performance.

### On (part of thw) unit test failures

I did some quick debug printing and it shows the follwing. Itt feels like the problem exists in the CPU backend instead of Metalium. The CPU backend is the one getting NaNs while Metalium producing good values. And the printed values are at an offset. Also diffed against upstream - (as of writing) the fork does not touch any part of the CPU backend. But I am more skepitcal that it's my skill issue instead of GGML itself being the problem.

```plaintext
test CPY of FP32 to BFP16 (CPY): nodes: 3

Content of tensor dst (copy of src):
backend: Metalium, CPU    0  0.202148  0.416016, diff = -0.213867
    1 -0.714844  0.302734, diff = -1.017578
    2  0.416016  0.941406, diff = -0.525391
    3  0.302734  0.443359, diff = -0.140625
    4 -0.957031 -0.574219, diff = -0.382812
    5 -0.886719 -1.000000, diff =  0.113281
    6  0.941406 -0.632812, diff =  1.574219
    7  0.443359  0.235352, diff =  0.208008
    8  0.664062  0.049561, diff =  0.614502
    9  0.878906 -0.984375, diff =  1.863281
   10 -0.574219 -0.417969, diff = -0.156250
   11 -1.000000  0.049561, diff = -1.049561
   12 -0.636719 -0.722656, diff =  0.085938
   13  0.984375 -0.906250, diff =  1.890625
   14 -0.632812 -0.267578, diff = -0.365234
   15  0.235352 -0.535156, diff =  0.770508
   16 -0.390625  0.570312, diff = -0.960938
   17  0.223633  0.236328, diff = -0.012695
   18  0.049561  0.028442, diff =  0.021118
   19 -0.984375  0.964844, diff = -1.949219
   20 -0.135742 -0.906250, diff =  0.770508
   21 -0.953125  0.718750, diff = -1.671875
   22 -0.417969 -0.660156, diff =  0.242188
   23  0.049561 -0.099121, diff =  0.148682
   24  0.223633      -inf, diff =       inf
   25 -0.200195      -inf, diff =       inf
   26 -0.722656       inf, diff =      -inf
   27 -0.906250  0.000000, diff = -0.906250
   28 -0.416016       inf, diff =      -inf
   29  0.949219      -inf, diff =       inf
   30 -0.267578       inf, diff =      -inf
   31 -0.535156  0.000000, diff = -0.535156
   32 -0.087891       inf, diff =      -inf
   33 -0.820312      -inf, diff =       inf
   34  0.570312       inf, diff =      -inf
   35  0.236328  0.000000, diff =  0.236328
   36 -0.601562       inf, diff =      -inf
   37 -0.235352      -inf, diff =       inf
   38  0.028442       inf, diff =      -inf
   39  0.964844  0.000000, diff =  0.964844
   40  0.184570  0.000000, diff =  0.184570
   41 -0.066406  0.000000, diff = -0.066406
   42 -0.906250  0.000000, diff = -0.906250
   43  0.718750  0.000000, diff =  0.718750
   44  0.214844  0.000000, diff =  0.214844
   45  0.361328  0.000000, diff =  0.361328
   46 -0.660156  0.000000, diff = -0.660156
   47 -0.099121  0.000000, diff = -0.099121

[CPY] inf mismatch: Metalium=0.223633 CPU=-inf
Content of tensor sent_0:
backend: Metalium, CPU    0 -0.250000 -0.250000, diff =  0.000000
    1  0.593750  0.593750, diff =  0.000000
    2  0.902344  0.902344, diff =  0.000000
    3 -0.632812 -0.632812, diff =  0.000000
    4  0.464844  0.464844, diff =  0.000000
    5  0.558594  0.558594, diff =  0.000000
    6  0.197266  0.197266, diff =  0.000000
    7  0.193359  0.193359, diff =  0.000000
    8 -0.687500 -0.687500, diff =  0.000000
    9 -0.108398 -0.108398, diff =  0.000000
   10 -0.687500 -0.687500, diff =  0.000000
   11 -0.800781 -0.800781, diff =  0.000000
   12 -0.882812 -0.882812, diff =  0.000000
   13 -0.081543 -0.081543, diff =  0.000000
   14  0.730469  0.730469, diff =  0.000000
   15 -0.332031 -0.332031, diff =  0.000000


Content of tensor sent_1:
backend: Metalium, CPU    0  0.726562  0.726562, diff =  0.000000
    1  0.699219  0.699219, diff =  0.000000
    2  0.247070  0.247070, diff =  0.000000
    3 -0.101074 -0.101074, diff =  0.000000
    4 -0.337891 -0.337891, diff =  0.000000
    5 -0.808594 -0.808594, diff =  0.000000
    6 -0.871094 -0.871094, diff =  0.000000
    7 -0.257812 -0.257812, diff =  0.000000
    8 -0.378906 -0.378906, diff =  0.000000
    9  0.337891  0.337891, diff =  0.000000
   10 -0.349609 -0.349609, diff =  0.000000
   11  0.332031  0.332031, diff =  0.000000
   12  0.458984  0.458984, diff =  0.000000
   13  0.182617  0.182617, diff =  0.000000
   14  0.275391  0.275391, diff =  0.000000
   15 -0.451172 -0.451172, diff =  0.000000
```
