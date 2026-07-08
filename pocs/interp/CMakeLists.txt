add_executable(llama-interp-smoke smoke.cpp)
target_compile_features(llama-interp-smoke PRIVATE cxx_std_20)
target_include_directories(llama-interp-smoke PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ../../src ../../common)
target_link_libraries(llama-interp-smoke PRIVATE llama llama-common)

add_executable(llama-interp-probe probe.cpp)
target_compile_features(llama-interp-probe PRIVATE cxx_std_20)
target_include_directories(llama-interp-probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ../../src ../../common)
target_link_libraries(llama-interp-probe PRIVATE llama llama-common)
