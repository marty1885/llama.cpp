find_package(LAPACK REQUIRED)
find_package(ROOT QUIET COMPONENTS Core Tree RIO)

if(ROOT_FOUND)
    # ── librwkv_probe (static library of composable primitives) ────────────────
    # Links LAPACK for SVD (numeric.cpp). ROOT is only needed for root_io.cpp.
    set(LIB_TARGET rwkv_probe)
    add_library(${LIB_TARGET} STATIC
        src/model.cpp
        src/state.cpp
        src/capture.cpp
        src/lens.cpp
        src/numeric.cpp
        src/generate.cpp
        src/root_io.cpp
        src/experiment.cpp
        src/pipeline.cpp
        src/head_decomp.cpp
        src/edit.cpp
    )
    target_include_directories(${LIB_TARGET} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_include_directories(${LIB_TARGET} PRIVATE
        ${ROOT_INCLUDE_DIRS}
    )
    target_link_libraries(${LIB_TARGET} PUBLIC
        common llama ${LAPACK_LIBRARIES} ROOT::Core ROOT::Tree ROOT::RIO
    )
    target_compile_features(${LIB_TARGET} PUBLIC cxx_std_17)

    # ── llama-rwkv-probe (default experiment, back-compat CLI) ────────────────
    set(TARGET llama-rwkv-probe)
    add_executable(${TARGET} rwkv-probe.cpp)
    target_link_libraries(${TARGET} PRIVATE
        ${LIB_TARGET} common llama ${CMAKE_THREAD_LIBS_INIT}
        ROOT::Core ROOT::Tree ROOT::RIO
    )
    target_compile_features(${TARGET} PRIVATE cxx_std_17)
    target_include_directories(${TARGET} PRIVATE ${ROOT_INCLUDE_DIRS})
    if(LLAMA_TOOLS_INSTALL)
        install(TARGETS ${TARGET} RUNTIME)
    endif()

    # ── experiments/ — one .cpp per experiment, one binary per .cpp ──────────
    # Adding a new experiment is a single function call below.
    function(add_rwkv_experiment NAME)
        set(EXP_TARGET llama-rwkv-${NAME})
        add_executable(${EXP_TARGET} experiments/${NAME}.cpp)
        target_link_libraries(${EXP_TARGET} PRIVATE
            rwkv_probe common llama ${CMAKE_THREAD_LIBS_INIT}
            ROOT::Core ROOT::Tree ROOT::RIO
        )
        target_compile_features(${EXP_TARGET} PRIVATE cxx_std_17)
        target_include_directories(${EXP_TARGET} PRIVATE
            ${ROOT_INCLUDE_DIRS}
            ${CMAKE_SOURCE_DIR}/src
        )
        if(LLAMA_TOOLS_INSTALL)
            install(TARGETS ${EXP_TARGET} RUNTIME)
        endif()
    endfunction()

    add_rwkv_experiment(residual-patch)
    add_rwkv_experiment(state-geometry)
    add_rwkv_experiment(timemix-decompose)
    add_rwkv_experiment(residual-steer)
    add_rwkv_experiment(state-causal-trace)
    add_rwkv_experiment(state-head-trace)
    add_rwkv_experiment(state-svd)
    add_rwkv_experiment(rank1-edit)
    add_rwkv_experiment(rank1-validate)
    add_rwkv_experiment(decompose-epv)
    add_rwkv_experiment(entity-edit)
    add_rwkv_experiment(cross-context)
    add_rwkv_experiment(cross-entity)
    add_rwkv_experiment(token-delta)
    add_rwkv_experiment(r-gated-edit)
    add_rwkv_experiment(distance-test)
    add_rwkv_experiment(multi-entity)
    add_rwkv_experiment(ambiguous-entity)
    add_rwkv_experiment(state-edit)
    add_rwkv_experiment(edit-decompose)
    add_rwkv_experiment(donor-free)
    add_rwkv_experiment(slerp-edit)
    add_rwkv_experiment(kv-contrib)
    add_rwkv_experiment(calibrated-edit)
    add_rwkv_experiment(calibrated-multi)
    add_rwkv_experiment(adaptive-alpha)
    add_rwkv_experiment(projection-alpha)
    add_rwkv_experiment(order-rotation)
    add_rwkv_experiment(hybrid-edit)
else()
    message(STATUS "ROOT not found — skipping llama-rwkv-probe")
endif()
