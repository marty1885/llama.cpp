# ggml-et: Device kernels (cross-compiled within the main build)
#
# The RISC-V toolchain is set up in-scope so these targets use the
# cross-compiler while the rest of the build uses the host compiler.
# This keeps kernels in compile_commands.json for full IDE support.

# --- RISC-V toolchain setup (scoped to this directory) ---
set(TOOLCHAIN_DIR ${ET_PLATFORM_PATH})
include(${ET_PLATFORM_PATH}/lib/cmake/riscv64-ec-toolchain.cmake)
set(CMAKE_ADDR2LINE "${TOOLCHAIN_DIR}/bin/riscv64-unknown-elf-addr2line")
set(CMAKE_LINKER_TYPE LLD)

# Ensure kernels are built in this directory even if a global output directory is set
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

message(STATUS "ET kernels using RISC-V toolchain at: ${TOOLCHAIN_DIR}")

# DeviceUtils provides the add_riscv_executable macro
list(APPEND CMAKE_MODULE_PATH "${ET_PLATFORM_PATH}/lib/cmake/cmake-modules")
list(APPEND CMAKE_PREFIX_PATH "${ET_PLATFORM_PATH}/lib/cmake")
include(DeviceUtils)

find_package(et-common-libs REQUIRED)
find_package(esperantoTrace REQUIRED)

# --- Kernel configuration ---
if(NOT DEFINED ADDRESS)
    set(ADDRESS "0x8005801000")
    message(STATUS "ADDRESS not specified, using default: ${ADDRESS}")
endif()

set(LINKER_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/src/linker.ld)
set(CHECK_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/scripts/check_unimplemented_instructions.sh)

# Track address changes to trigger relinking
set(ADDRESS_FILE ${CMAKE_CURRENT_BINARY_DIR}/et_address.txt)
file(CONFIGURE OUTPUT ${ADDRESS_FILE} CONTENT "${ADDRESS}" @ONLY)

# KERNELS defined in upper CMakeLists.txt
foreach(KERNEL ${KERNELS})
    add_riscv_executable(${KERNEL})

    # Find the kernel (could be C or C++)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/${KERNEL}.cpp)
        set(KERNEL_SRC src/${KERNEL}.cpp)
    elseif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/${KERNEL}.cc)
        set(KERNEL_SRC src/${KERNEL}.cc)
    elseif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/${KERNEL}.c)
        set(KERNEL_SRC src/${KERNEL}.c)
    else()
        message(FATAL_ERROR "Source for kernel ${KERNEL} not found under src/. Supported extensions c, cpp")
    endif()

    target_sources(${KERNEL}.elf PRIVATE
        ${KERNEL_SRC}
        src/crt.S
    )
    target_include_directories(${KERNEL}.elf PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_libraries(${KERNEL}.elf PRIVATE et-common-libs::cm-umode)
    # C-only flags — must not apply to .S files
    target_compile_options(${KERNEL}.elf PRIVATE
        $<$<COMPILE_LANGUAGE:C,CXX>:-fno-zero-initialized-in-bss>
        $<$<COMPILE_LANGUAGE:C,CXX>:-ffreestanding>
        $<$<COMPILE_LANGUAGE:C,CXX>:-std=c99>
        $<$<COMPILE_LANGUAGE:C,CXX>:-ffat-lto-objects>
        $<$<COMPILE_LANGUAGE:C,CXX>:-mcmodel=medany>
        $<$<COMPILE_LANGUAGE:C,CXX>:-mabi=lp64f>
        $<$<COMPILE_LANGUAGE:C,CXX>:-march=rv64imf>
        $<$<COMPILE_LANGUAGE:C,CXX>:-ffunction-sections>
        $<$<COMPILE_LANGUAGE:C,CXX>:-fdata-sections>
        $<$<COMPILE_LANGUAGE:C,CXX>:-O3>
        $<$<COMPILE_LANGUAGE:C,CXX>:-g0>
        $<$<COMPILE_LANGUAGE:C,CXX>:-nostdlib>
        $<$<COMPILE_LANGUAGE:C,CXX>:-ffreestanding>
    )
    target_link_options(${KERNEL}.elf PRIVATE
        -Wl,--defsym=BASE_ADDRESS=${ADDRESS}
        -Wl,--entry=_start
    )
    # Append to LINK_DEPENDS (macro already sets it for the linker script)
    set_property(TARGET ${KERNEL}.elf APPEND PROPERTY
        LINK_DEPENDS "${ADDRESS_FILE}"
    )

    # Post-build: strip and check (fails build if check script fails)
    add_custom_command(TARGET ${KERNEL}.elf POST_BUILD
        COMMAND ${CMAKE_STRIP} --strip-debug $<TARGET_FILE:${KERNEL}.elf>
        COMMAND ${CHECK_SCRIPT}
            ${CMAKE_OBJDUMP} ${CMAKE_ADDR2LINE} $<TARGET_FILE:${KERNEL}.elf>
        # DEPENDS ${CHECK_SCRIPT} # FIXME: CMake says this file is invalid...?
        VERBATIM
    )
    message(${CHECK_SCRIPT})
endforeach()

# Print summary
message(STATUS "GGML ET Kernels configured:")
foreach(KERNEL ${KERNELS})
    message(STATUS "  - ${KERNEL}")
endforeach()
message(STATUS "Base address: ${ADDRESS}")
