# ROOT's packaged config imports VDT but some distro VDT packages do not ship a
# CMake package config. Resolve its conventional header/library pair first.
find_path(VDT_INCLUDE_DIR NAMES vdt/vdtMath.h REQUIRED)
find_library(VDT_LIBRARY NAMES vdt REQUIRED)
find_package(ROOT 6.40 REQUIRED COMPONENTS Matrix ROOTNTuple Tree)
find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENBLAS REQUIRED IMPORTED_TARGET openblas)

add_library(llama-rwkv-activation-store rwkv_activation_store.cpp)
target_compile_features(llama-rwkv-activation-store PRIVATE cxx_std_20)
target_include_directories(llama-rwkv-activation-store PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ../../src ../../common)
target_link_libraries(llama-rwkv-activation-store PRIVATE ROOT::ROOTNTuple PUBLIC ggml)

add_executable(llama-rwkv-activation-collect rwkv_activation_collect.cpp)
target_compile_features(llama-rwkv-activation-collect PRIVATE cxx_std_20)
target_include_directories(llama-rwkv-activation-collect PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ../../src ../../common)
target_link_libraries(llama-rwkv-activation-collect PRIVATE llama-rwkv-activation-store llama llama-common)

add_executable(llama-rwkv-interp-verify rwkv_interp_verify.cpp)
target_compile_features(llama-rwkv-interp-verify PRIVATE cxx_std_20)
target_include_directories(llama-rwkv-interp-verify PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ../../src ../../common)
target_link_libraries(llama-rwkv-interp-verify PRIVATE llama llama-common)

add_executable(llama-rwkv-jacobian-replay-verify rwkv_jacobian_replay_verify.cpp)
target_compile_features(llama-rwkv-jacobian-replay-verify PRIVATE cxx_std_20)
target_include_directories(llama-rwkv-jacobian-replay-verify PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ../../src ../../common)
target_link_libraries(llama-rwkv-jacobian-replay-verify PRIVATE llama llama-common)

add_executable(llama-rwkv-jacobian-linearity-corpus rwkv_jacobian_linearity_corpus.cpp)
target_compile_features(llama-rwkv-jacobian-linearity-corpus PRIVATE cxx_std_20)
target_include_directories(llama-rwkv-jacobian-linearity-corpus PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ../../src ../../common)
target_link_libraries(llama-rwkv-jacobian-linearity-corpus PRIVATE ROOT::Tree llama llama-common)

add_executable(llama-rwkv-key-lens rwkv_key_lens.cpp)
target_compile_features(llama-rwkv-key-lens PRIVATE cxx_std_20)
target_include_directories(llama-rwkv-key-lens PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ../../src ../../common)
target_link_libraries(llama-rwkv-key-lens PRIVATE ROOT::Tree llama llama-common)

add_executable(llama-rwkv-key-input-lens rwkv_key_input_lens.cpp)
target_compile_features(llama-rwkv-key-input-lens PRIVATE cxx_std_20)
target_include_directories(llama-rwkv-key-input-lens PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ../../src ../../common)
target_link_libraries(llama-rwkv-key-input-lens PRIVATE llama llama-common)

add_executable(llama-rwkv-memory-readout-lens rwkv_memory_readout_lens.cpp)
target_compile_features(llama-rwkv-memory-readout-lens PRIVATE cxx_std_20)
target_include_directories(llama-rwkv-memory-readout-lens PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ../../src ../../common)
target_link_libraries(llama-rwkv-memory-readout-lens PRIVATE llama llama-common)

add_executable(llama-rwkv-tmix-filter-verify rwkv_tmix_filter_verify.cpp)
target_compile_features(llama-rwkv-tmix-filter-verify PRIVATE cxx_std_20)
target_include_directories(llama-rwkv-tmix-filter-verify PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ../../src ../../common)
target_link_libraries(llama-rwkv-tmix-filter-verify PRIVATE llama llama-common)

add_executable(llama-rwkv-tmix-norm-geometry rwkv_tmix_norm_geometry.cpp)
target_compile_features(llama-rwkv-tmix-norm-geometry PRIVATE cxx_std_20)
target_include_directories(llama-rwkv-tmix-norm-geometry PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ../../src ../../common)
target_link_libraries(llama-rwkv-tmix-norm-geometry PRIVATE llama llama-common)

add_executable(llama-rwkv-tmix-memory-state rwkv_tmix_memory_state.cpp)
target_compile_features(llama-rwkv-tmix-memory-state PRIVATE cxx_std_20)
target_include_directories(llama-rwkv-tmix-memory-state PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ../../src ../../common)
target_link_libraries(llama-rwkv-tmix-memory-state PRIVATE llama llama-common)

add_executable(llama-rwkv-tmix-wkv-jvp-lens rwkv_tmix_wkv_jvp_lens.cpp)
target_compile_features(llama-rwkv-tmix-wkv-jvp-lens PRIVATE cxx_std_20)
target_include_directories(llama-rwkv-tmix-wkv-jvp-lens PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ../../src ../../common)
target_link_libraries(llama-rwkv-tmix-wkv-jvp-lens PRIVATE llama llama-common)

add_executable(llama-rwkv-tmix-wkv-jvp-corpus rwkv_tmix_wkv_jvp_corpus.cpp)
target_compile_features(llama-rwkv-tmix-wkv-jvp-corpus PRIVATE cxx_std_20)
target_include_directories(llama-rwkv-tmix-wkv-jvp-corpus PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ../../src ../../common)
target_link_libraries(llama-rwkv-tmix-wkv-jvp-corpus PRIVATE llama llama-common)

add_executable(llama-rwkv-tmix-wkv-jvp-layers rwkv_tmix_wkv_jvp_layers.cpp)
target_compile_features(llama-rwkv-tmix-wkv-jvp-layers PRIVATE cxx_std_20)
target_include_directories(llama-rwkv-tmix-wkv-jvp-layers PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ../../src ../../common)
target_link_libraries(llama-rwkv-tmix-wkv-jvp-layers PRIVATE ROOT::Tree llama llama-common)

add_executable(llama-rwkv-tmix-wkv-jvp-layers-summary rwkv_tmix_wkv_jvp_layers_summary.cpp)
target_compile_features(llama-rwkv-tmix-wkv-jvp-layers-summary PRIVATE cxx_std_20)
target_link_libraries(llama-rwkv-tmix-wkv-jvp-layers-summary PRIVATE ROOT::Tree)

add_executable(llama-rwkv-greedy-generate rwkv_greedy_generate.cpp)
target_compile_features(llama-rwkv-greedy-generate PRIVATE cxx_std_20)
target_include_directories(llama-rwkv-greedy-generate PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ../../src ../../common)
target_link_libraries(llama-rwkv-greedy-generate PRIVATE llama llama-common)

add_executable(llama-rwkv-tmix-cluster rwkv_tmix_cluster.cpp)
target_compile_features(llama-rwkv-tmix-cluster PRIVATE cxx_std_20)
target_include_directories(llama-rwkv-tmix-cluster PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ../../src ../../common)
target_link_libraries(llama-rwkv-tmix-cluster PRIVATE llama llama-common)

add_executable(llama-rwkv-tmix-head-geometry rwkv_tmix_head_geometry.cpp)
target_compile_features(llama-rwkv-tmix-head-geometry PRIVATE cxx_std_20)
target_include_directories(llama-rwkv-tmix-head-geometry PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ../../src ../../common)
target_link_libraries(llama-rwkv-tmix-head-geometry PRIVATE llama llama-common)

add_executable(llama-rwkv-inverse-cache-build rwkv_inverse_cache_build.cpp)
target_compile_features(llama-rwkv-inverse-cache-build PRIVATE cxx_std_20)
target_include_directories(llama-rwkv-inverse-cache-build PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ../../src ../../common)
target_link_libraries(llama-rwkv-inverse-cache-build PRIVATE PkgConfig::OPENBLAS ROOT::Tree llama llama-common)
