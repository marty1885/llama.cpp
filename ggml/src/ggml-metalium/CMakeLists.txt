if("$ENV{TT_METAL_HOME}" STREQUAL "")
    message(FATAL_ERROR "TT_METAL_HOME is not set")
endif()

message(STATUS "Metalium found.")

ggml_add_backend_library(ggml-metalium
    ggml-metalium.cpp
    utils.cpp
    rope.cpp
    mul_mat.cpp
    soft_max.cpp
)

target_include_directories(ggml-metalium PRIVATE
    $ENV{TT_METAL_HOME}/build/include
    # Metalium
    $ENV{TT_METAL_HOME}
    $ENV{TT_METAL_HOME}/tt_metal
    $ENV{TT_METAL_HOME}/tt_metal/api
    $ENV{TT_METAL_HOME}/tt_stl
    $ENV{TT_METAL_HOME}/tt_metal/third_party/umd
    $ENV{TT_METAL_HOME}/tt_metal/third_party/umd/device/
    $ENV{TT_METAL_HOME}/tt_metal/third_party/umd/device/api
    $ENV{TT_METAL_HOME}/tt_metal/third_party/umd/device/api/umd
    $ENV{TT_METAL_HOME}/tt_metal/third_party/umd/device/api/umd/device
    $ENV{TT_METAL_HOME}/tt_metal/hw/inc/
    $ENV{TT_METAL_HOME}/tt_metal/third_party/tracy/public/tracy
    $ENV{TT_METAL_HOME}/tt_metal/third_party/tracy/public/
    $ENV{TT_METAL_HOME}/tt_metal/hostdevcommon/api/
    $ENV{TT_METAL_HOME}/tt_metal/include

    # TTNN
    $ENV{TT_METAL_HOME}/ttnn
    $ENV{TT_METAL_HOME}/ttnn/api
    $ENV{TT_METAL_HOME}/ttnn/cpp

    # Parent GGML stuff
    ..
)

target_link_directories(ggml-metalium PRIVATE
    $ENV{TT_METAL_HOME}/build/ttnn
    $ENV{TT_METAL_HOME}/build/tt_metal
)

target_link_libraries(ggml-metalium PRIVATE
    # Metalium
    tt_metal
    yaml-cpp

    # TTNN
    $ENV{TT_METAL_HOME}/build/ttnn/_ttnncpp.so

    ggml-base
)

if(GGML_METALIUM_EMBED_KERNELS)
    add_library(ggml-metalium-kernels STATIC)
    set(GGML_METALIUM_EMBED_KERNELS
            rope_neox_reader
            rope_neox_compute
            rope_neox_writer
            rope_normal_reader
            rope_normal_compute
            rope_normal_writer
            mul_mat_reader
            mul_mat_compute
            mul_mat_writer
            soft_max_reader
            soft_max_writer
            soft_max_compute
    )
    include(cmake/add_kernel.cmake)
    foreach(kernel IN ITEMS ${GGML_METALIUM_EMBED_KERNELS})
        add_kernel(ggml-metalium-kernels ${kernel})
    endforeach()
    register_kernels(ggml-metalium "${GGML_METALIUM_EMBED_KERNELS}")
    target_link_libraries(ggml-metalium PRIVATE ggml-metalium-kernels)
    target_compile_definitions(ggml-metalium PRIVATE GGML_METALIUM_EMBED_KERNELS)
    set_target_properties(ggml-metalium-kernels PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_options(ggml-metalium-kernels PRIVATE -Wno-missing-declarations)
endif()


target_compile_definitions(
    ggml-metalium PRIVATE
    FMT_HEADER_ONLY
    SPDLOG_FMT_EXTERNAL
)

option(GGML_METALIUM_HACK_DISABLE_CUSTOM_KERNELS "Disable Metalium custom kernels while they are being ported to the new TT interface" OFF)
if(GGML_METALIUM_HACK_DISABLE_CUSTOM_KERNELS)
    target_compile_definitions(ggml-metalium PRIVATE GGML_METALIUM_HACK_CUSTOM_KERNEL_BECAUSE_TT_UPDATED_THE_INTERFACE_AND_I_NEED_TO_FIGURE_OUT_HOW_TO_FIX_TEHM)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsized-deallocation")
endif()

target_precompile_headers(ggml-metalium PRIVATE metalium-pch.hpp)
target_compile_features(ggml-metalium PRIVATE cxx_std_20)
