# Called by add_custom_command
file(READ "${INPUT}" KERNEL_CONTENT)

file(WRITE "${OUTPUT}" "
#include <string>
#include <unordered_map>
#include <cstdio>

extern std::unordered_map<std::string, std::string>& ggml_metalium_get_kernel_map();

static const std::string ${NAME}_src = R\"(${KERNEL_CONTENT})\";
void metalium_kernel_register_${NAME}() {
    ggml_metalium_get_kernel_map().emplace(\"${NAME}\", ${NAME}_src);
}
")
