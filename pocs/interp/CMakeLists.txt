# ROOT's packaged config imports VDT but some distro VDT packages do not ship a
# CMake package config. Resolve its conventional header/library pair first.
find_path(VDT_INCLUDE_DIR NAMES vdt/vdtMath.h REQUIRED)
find_library(VDT_LIBRARY NAMES vdt REQUIRED)
find_package(ROOT 6.40 REQUIRED COMPONENTS ROOTNTuple)

add_library(llama-rwkv-activation-store rwkv_activation_store.cpp)
target_compile_features(llama-rwkv-activation-store PRIVATE cxx_std_20)
target_include_directories(llama-rwkv-activation-store PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ../../src ../../common)
target_link_libraries(llama-rwkv-activation-store PRIVATE ROOT::ROOTNTuple PUBLIC ggml)

add_executable(llama-rwkv-activation-collect rwkv_activation_collect.cpp)
target_compile_features(llama-rwkv-activation-collect PRIVATE cxx_std_20)
target_include_directories(llama-rwkv-activation-collect PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ../../src ../../common)
target_link_libraries(llama-rwkv-activation-collect PRIVATE llama-rwkv-activation-store llama llama-common)

add_executable(llama-rwkv-activation-cluster rwkv_activation_cluster.cpp)
target_compile_features(llama-rwkv-activation-cluster PRIVATE cxx_std_20)
target_include_directories(llama-rwkv-activation-cluster PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ../../src ../../common)
target_link_libraries(llama-rwkv-activation-cluster PRIVATE llama-rwkv-activation-store)

add_executable(llama-rwkv-activation-analyze rwkv_activation_analyze.cpp)
target_compile_features(llama-rwkv-activation-analyze PRIVATE cxx_std_20)
target_include_directories(llama-rwkv-activation-analyze PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ../../src ../../common)
target_link_libraries(llama-rwkv-activation-analyze PRIVATE llama-rwkv-activation-store)

add_executable(llama-rwkv-activation-coordinate-probe rwkv_activation_coordinate_probe.cpp)
target_compile_features(llama-rwkv-activation-coordinate-probe PRIVATE cxx_std_20)
target_include_directories(llama-rwkv-activation-coordinate-probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ../../src ../../common)
target_link_libraries(llama-rwkv-activation-coordinate-probe PRIVATE llama-rwkv-activation-store)

add_executable(llama-rwkv-activation-linear-readout rwkv_activation_linear_readout.cpp)
target_compile_features(llama-rwkv-activation-linear-readout PRIVATE cxx_std_20)
target_include_directories(llama-rwkv-activation-linear-readout PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ../../src ../../common)
target_link_libraries(llama-rwkv-activation-linear-readout PRIVATE llama-rwkv-activation-store llama llama-common)

add_executable(llama-rwkv7-time-lens rwkv7_time_lens.cpp)
target_compile_features(llama-rwkv7-time-lens PRIVATE cxx_std_20)
target_include_directories(llama-rwkv7-time-lens PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ../../src ../../common)
target_link_libraries(llama-rwkv7-time-lens PRIVATE llama llama-common)
