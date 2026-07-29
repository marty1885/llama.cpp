# ROOT's packaged config imports VDT but some distro VDT packages do not ship a
# CMake package config. Resolve its conventional header/library pair first.
find_path(VDT_INCLUDE_DIR NAMES vdt/vdtMath.h REQUIRED)
find_library(VDT_LIBRARY NAMES vdt REQUIRED)
find_package(ROOT 6.40 REQUIRED COMPONENTS Matrix ROOTNTuple Tree)
find_package(OpenSSL REQUIRED COMPONENTS Crypto)
find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENBLAS REQUIRED IMPORTED_TARGET openblas)
find_package(OpenMP REQUIRED)

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

add_executable(llama-rwkv-rotor-layers rwkv_rotor_layers.cpp)
target_compile_features(llama-rwkv-rotor-layers PRIVATE cxx_std_20)
target_include_directories(llama-rwkv-rotor-layers PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ../../src ../../common)
target_link_libraries(llama-rwkv-rotor-layers PRIVATE llama llama-common)

add_executable(llama-rwkv-rotor-response-curve rwkv_rotor_response_curve.cpp)
target_compile_features(llama-rwkv-rotor-response-curve PRIVATE cxx_std_20)
target_include_directories(llama-rwkv-rotor-response-curve PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ../../src ../../common)
target_link_libraries(llama-rwkv-rotor-response-curve PRIVATE llama llama-common)

add_executable(llama-rwkv-rotor-response-layers rwkv_rotor_response_layers.cpp)
target_compile_features(llama-rwkv-rotor-response-layers PRIVATE cxx_std_20)
target_include_directories(llama-rwkv-rotor-response-layers PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ../../src ../../common)
target_link_libraries(llama-rwkv-rotor-response-layers PRIVATE llama llama-common)

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

add_executable(llama-rwkv-tmix-residual-rotation rwkv_tmix_residual_rotation.cpp)
target_compile_features(llama-rwkv-tmix-residual-rotation PRIVATE cxx_std_20)
target_include_directories(llama-rwkv-tmix-residual-rotation PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ../../src ../../common)
target_link_libraries(llama-rwkv-tmix-residual-rotation PRIVATE llama llama-common)

add_executable(llama-rwkv-tmix-memory-state rwkv_tmix_memory_state.cpp)
target_compile_features(llama-rwkv-tmix-memory-state PRIVATE cxx_std_20)
target_include_directories(llama-rwkv-tmix-memory-state PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ../../src ../../common)
target_link_libraries(llama-rwkv-tmix-memory-state PRIVATE llama llama-common)

add_executable(llama-rwkv-tmix-rotor-subspace rwkv_tmix_rotor_subspace.cpp)
target_compile_features(llama-rwkv-tmix-rotor-subspace PRIVATE cxx_std_20)
target_include_directories(llama-rwkv-tmix-rotor-subspace PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ../../src ../../common)
target_link_libraries(llama-rwkv-tmix-rotor-subspace PRIVATE ROOT::Matrix ROOT::ROOTNTuple llama llama-common)

add_executable(llama-rwkv-tmix-destination-subspace rwkv_tmix_destination_subspace.cpp)
target_compile_features(llama-rwkv-tmix-destination-subspace PRIVATE cxx_std_20)
target_include_directories(llama-rwkv-tmix-destination-subspace PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ../../src ../../common)
target_link_libraries(llama-rwkv-tmix-destination-subspace PRIVATE PkgConfig::OPENBLAS ROOT::Matrix ROOT::ROOTNTuple llama llama-common)

add_executable(llama-rwkv-tmix-destination-base-recovery experiments/tmix_destination_base_recovery/rwkv_tmix_destination_base_recovery.cpp)
target_compile_features(llama-rwkv-tmix-destination-base-recovery PRIVATE cxx_std_20)
target_link_libraries(llama-rwkv-tmix-destination-base-recovery PRIVATE PkgConfig::OPENBLAS ROOT::Matrix ROOT::ROOTNTuple OpenMP::OpenMP_CXX)

add_executable(llama-rwkv-tmix-relative-geometry-null experiments/tmix_relative_geometry_null/rwkv_tmix_relative_geometry_null.cpp)
target_compile_features(llama-rwkv-tmix-relative-geometry-null PRIVATE cxx_std_20)
target_link_libraries(llama-rwkv-tmix-relative-geometry-null PRIVATE PkgConfig::OPENBLAS ROOT::Matrix ROOT::ROOTNTuple OpenMP::OpenMP_CXX)

add_executable(llama-rwkv-tmix-controlled-value-corpus experiments/tmix_controlled_value_decode/rwkv_tmix_controlled_value_corpus.cpp)
target_compile_features(llama-rwkv-tmix-controlled-value-corpus PRIVATE cxx_std_20)
target_include_directories(llama-rwkv-tmix-controlled-value-corpus PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ../../src ../../common)
target_link_libraries(llama-rwkv-tmix-controlled-value-corpus PRIVATE llama llama-common OpenSSL::Crypto)

add_executable(llama-rwkv-tmix-controlled-value-collect experiments/tmix_controlled_value_decode/rwkv_tmix_controlled_value_collect.cpp)
target_compile_features(llama-rwkv-tmix-controlled-value-collect PRIVATE cxx_std_20)
target_include_directories(llama-rwkv-tmix-controlled-value-collect PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ../../src ../../common)
target_link_libraries(llama-rwkv-tmix-controlled-value-collect PRIVATE llama-rwkv-activation-store ROOT::ROOTNTuple OpenSSL::Crypto llama llama-common)

add_executable(llama-rwkv-tmix-controlled-value-decode experiments/tmix_controlled_value_decode/rwkv_tmix_controlled_value_decode.cpp)
target_compile_features(llama-rwkv-tmix-controlled-value-decode PRIVATE cxx_std_20)
target_include_directories(llama-rwkv-tmix-controlled-value-decode PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ../../src ../../common)
target_link_libraries(llama-rwkv-tmix-controlled-value-decode PRIVATE llama-rwkv-activation-store PkgConfig::OPENBLAS ROOT::ROOTNTuple OpenMP::OpenMP_CXX)

add_executable(llama-rwkv-tmix-causal-rotation-ablation experiments/tmix_causal_rotation_ablation/rwkv_tmix_causal_rotation_ablation.cpp)
target_compile_features(llama-rwkv-tmix-causal-rotation-ablation PRIVATE cxx_std_20)
target_include_directories(llama-rwkv-tmix-causal-rotation-ablation PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ../../src ../../common)
target_link_libraries(llama-rwkv-tmix-causal-rotation-ablation PRIVATE ROOT::ROOTNTuple OpenSSL::Crypto llama llama-common)

add_executable(llama-rwkv-text8-prompt-corpus rwkv_text8_prompt_corpus.cpp)
target_compile_features(llama-rwkv-text8-prompt-corpus PRIVATE cxx_std_20)

add_executable(llama-rwkv-pile-prompt-corpus rwkv_pile_prompt_corpus.cpp)
target_compile_features(llama-rwkv-pile-prompt-corpus PRIVATE cxx_std_20)

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
