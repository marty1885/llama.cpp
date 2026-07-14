# dependencies

find_package(Threads REQUIRED)

# third-party

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if (NOT EMSCRIPTEN)
    option(LLAMA_BUILD_RWKV_INTERP "Build RWKV GGML instrumentation tools" OFF)
    if (LLAMA_BUILD_RWKV_INTERP)
        add_subdirectory(interp)
    endif()
    if (NOT GGML_BACKEND_DL)
        add_subdirectory(vdot)
    endif()
endif()
