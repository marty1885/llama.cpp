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

add_executable(llama-interp-backend-parity backend_parity.cpp)
target_compile_features(llama-interp-backend-parity PRIVATE cxx_std_20)
target_include_directories(llama-interp-backend-parity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ../../src ../../common)
target_link_libraries(llama-interp-backend-parity PRIVATE llama llama-common)
