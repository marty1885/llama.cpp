add_executable(llama-interp-smoke smoke.cpp)
target_compile_features(llama-interp-smoke PRIVATE cxx_std_20)
target_include_directories(llama-interp-smoke PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ../../src ../../common)
target_link_libraries(llama-interp-smoke PRIVATE llama llama-common)

add_executable(llama-interp-probe probe.cpp)
target_compile_features(llama-interp-probe PRIVATE cxx_std_20)
target_include_directories(llama-interp-probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ../../src ../../common)
target_link_libraries(llama-interp-probe PRIVATE llama llama-common)

add_executable(llama-interp-jlens-operator jlens_operator.cpp)
target_compile_features(llama-interp-jlens-operator PRIVATE cxx_std_20)
target_include_directories(llama-interp-jlens-operator PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ../../src ../../common)
target_link_libraries(llama-interp-jlens-operator PRIVATE llama llama-common)

add_executable(llama-interp-jlens-average jlens_average.cpp)
target_compile_features(llama-interp-jlens-average PRIVATE cxx_std_20)
target_include_directories(llama-interp-jlens-average PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ../../src ../../common)
target_link_libraries(llama-interp-jlens-average PRIVATE llama llama-common)

add_executable(llama-interp-jlens-readout jlens_readout.cpp)
target_compile_features(llama-interp-jlens-readout PRIVATE cxx_std_20)
target_include_directories(llama-interp-jlens-readout PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ../../src ../../common)
target_link_libraries(llama-interp-jlens-readout PRIVATE llama llama-common)

add_executable(llama-interp-jlens-build jlens_build.cpp)
target_compile_features(llama-interp-jlens-build PRIVATE cxx_std_20)
target_include_directories(llama-interp-jlens-build PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ../../src ../../common)
target_link_libraries(llama-interp-jlens-build PRIVATE llama llama-common)

add_executable(llama-interp-jlens-eval jlens_eval.cpp)
target_compile_features(llama-interp-jlens-eval PRIVATE cxx_std_20)
target_include_directories(llama-interp-jlens-eval PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ../../src ../../common)
target_link_libraries(llama-interp-jlens-eval PRIVATE llama llama-common)

add_executable(llama-interp-jlens-intervene jlens_intervene.cpp)
target_compile_features(llama-interp-jlens-intervene PRIVATE cxx_std_20)
target_include_directories(llama-interp-jlens-intervene PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ../../src ../../common)
target_link_libraries(llama-interp-jlens-intervene PRIVATE llama llama-common)

# ROOT's packaged config imports VDT but some distro VDT packages do not ship a
# CMake package config. Resolve its conventional header/library pair first.
find_path(VDT_INCLUDE_DIR NAMES vdt/vdtMath.h REQUIRED)
find_library(VDT_LIBRARY NAMES vdt REQUIRED)
find_package(ROOT 6.40 REQUIRED COMPONENTS ROOTNTuple)

add_library(llama-interp-rwkv-activation-store rwkv_activation_store.cpp)
target_compile_features(llama-interp-rwkv-activation-store PRIVATE cxx_std_20)
target_include_directories(llama-interp-rwkv-activation-store PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ../../src ../../common)
target_link_libraries(llama-interp-rwkv-activation-store PRIVATE ROOT::ROOTNTuple PUBLIC ggml)

target_link_libraries(llama-interp-jlens-operator PRIVATE llama-interp-rwkv-activation-store)

add_executable(llama-interp-rwkv-unembed rwkv_unembed.cpp)
target_compile_features(llama-interp-rwkv-unembed PRIVATE cxx_std_20)
target_include_directories(llama-interp-rwkv-unembed PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ../../src ../../common)
target_link_libraries(llama-interp-rwkv-unembed PRIVATE llama-interp-rwkv-activation-store llama llama-common)

add_executable(llama-interp-rwkv-activation-collect rwkv_activation_collect.cpp)
target_compile_features(llama-interp-rwkv-activation-collect PRIVATE cxx_std_20)
target_include_directories(llama-interp-rwkv-activation-collect PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ../../src ../../common)
target_link_libraries(llama-interp-rwkv-activation-collect PRIVATE llama-interp-rwkv-activation-store llama llama-common)

add_executable(llama-interp-rwkv-activation-cluster rwkv_activation_cluster.cpp)
target_compile_features(llama-interp-rwkv-activation-cluster PRIVATE cxx_std_20)
target_include_directories(llama-interp-rwkv-activation-cluster PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ../../src ../../common)
target_link_libraries(llama-interp-rwkv-activation-cluster PRIVATE llama-interp-rwkv-activation-store)

add_executable(llama-interp-rwkv-activation-analyze rwkv_activation_analyze.cpp)
target_compile_features(llama-interp-rwkv-activation-analyze PRIVATE cxx_std_20)
target_include_directories(llama-interp-rwkv-activation-analyze PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ../../src ../../common)
target_link_libraries(llama-interp-rwkv-activation-analyze PRIVATE llama-interp-rwkv-activation-store)

add_executable(llama-interp-backend-parity backend_parity.cpp)
target_compile_features(llama-interp-backend-parity PRIVATE cxx_std_20)
target_include_directories(llama-interp-backend-parity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ../../src ../../common)
target_link_libraries(llama-interp-backend-parity PRIVATE llama llama-common)
