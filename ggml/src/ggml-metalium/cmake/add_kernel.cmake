# Usage: add_kernel(target kernel_name)
# Looks for ${CMAKE_CURRENT_SOURCE_DIR}/kernels/${kernel_name}.cpp
# Generates ${CMAKE_BINARY_DIR}/generated/metalium/${kernel_name}.cpp

function(add_kernel target kernel_name)
    set(KERNEL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/kernels/${kernel_name}.cpp")
    set(GEN_DIR "${CMAKE_BINARY_DIR}/generated/metalium")
    set(GEN_CPP "${GEN_DIR}/__embed_${kernel_name}.cpp")

    file(MAKE_DIRECTORY "${GEN_DIR}")

    message(STATUS "Generating ${GEN_CPP}")

    add_custom_command(
        OUTPUT "${GEN_CPP}"
        COMMAND ${CMAKE_COMMAND} -E echo "Embedding kernel: ${kernel_name}"
        COMMAND ${CMAKE_COMMAND}
                -DINPUT=${KERNEL_SRC}
                -DOUTPUT=${GEN_CPP}
                -DNAME=${kernel_name}
                -P ${CMAKE_CURRENT_LIST_DIR}/cmake/embed_kernel.cmake

        DEPENDS "${KERNEL_SRC}" ${CMAKE_CURRENT_LIST_DIR}/cmake/embed_kernel.cmake
        COMMENT "Generating embedded kernel for ${kernel_name}"
        VERBATIM
    )

    target_sources(${target} PRIVATE "${GEN_CPP}")

    # Track kernel name for registration
    set_property(TARGET ${target} APPEND PROPERTY KERNEL_LIST "${kernel_name}")
endfunction()

function(register_kernels target kernel_names)
    set(OUTPUT "${CMAKE_BINARY_DIR}/generated/metalium/__register_all_kernels.cpp")

    # Generate the registration source file
    set(KERNEL_NAMES "${kernel_names}")
    add_custom_command(
        OUTPUT "${OUTPUT}"
        COMMAND ${CMAKE_COMMAND} -E echo "Generating Metalium kernel registration source"
        COMMAND ${CMAKE_COMMAND}
                -DOUTPUT=${OUTPUT}
                "-DKERNEL_NAMES=${KERNEL_NAMES}"
                -P ${CMAKE_CURRENT_LIST_DIR}/cmake/kernel_register.cmake

        DEPENDS ${CMAKE_CURRENT_LIST_DIR}/cmake/kernel_register.cmake
        COMMENT "Generating kernel registration source"
        VERBATIM
    )

    target_sources(${target} PRIVATE "${OUTPUT}")
endfunction()
