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

add_executable(llama-rwkv-interp-verify rwkv_interp_verify.cpp)
target_compile_features(llama-rwkv-interp-verify PRIVATE cxx_std_20)
target_include_directories(llama-rwkv-interp-verify PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ../../src ../../common)
target_link_libraries(llama-rwkv-interp-verify PRIVATE llama llama-common)
