if(NOT DEFINED OUTPUT)
    message(FATAL_ERROR "OUTPUT variable not set")
endif()

if(NOT DEFINED KERNEL_NAMES)
    message(FATAL_ERROR "KERNEL_NAMES variable not set")
endif()

file(WRITE "${OUTPUT}" "#include <mutex>\n")
foreach(NAME IN LISTS KERNEL_NAMES)
    file(APPEND "${OUTPUT}" "extern void metalium_kernel_register_${NAME}();\n")
endforeach()

file(APPEND "${OUTPUT}" "\nvoid metalium_register_all_kernel() {\n")
file(APPEND "${OUTPUT}" "    static std::once_flag once_flag;\n")
file(APPEND "${OUTPUT}" "    std::call_once(once_flag, []() {\n")
foreach(NAME IN LISTS KERNEL_NAMES)
    file(APPEND "${OUTPUT}" "        metalium_kernel_register_${NAME}();\n")
endforeach()
file(APPEND "${OUTPUT}" "    });\n")
file(APPEND "${OUTPUT}" "}\n")

message(STATUS "Generated kernel registration for kernels: ${KERNEL_NAMES} into ${OUTPUT}")