#include <unordered_map>

#include "tt-metalium/host_api.hpp"
#include "tt-metalium/kernel_types.hpp"
#include "tt-metalium/tt_backend_api_types.hpp"

using CoreSpec = std::variant<tt::tt_metal::CoreCoord, tt::tt_metal::CoreRange, tt::tt_metal::CoreRangeSet>;
tt::tt_metal::CBHandle MakeCircularBuffer(
    tt::tt_metal::Program& program, const CoreSpec& core, tt::CBIndex cb, uint32_t size, uint32_t page_size, tt::DataFormat format);
tt::tt_metal::CBHandle MakeCircularBuffer(tt::tt_metal::Program& program, const CoreSpec& core, tt::CBIndex cb, uint32_t n_tiles, tt::tt_metal::DataType dtype);

tt::tt_metal::KernelHandle CreateMetaliumKernel(
    tt::tt_metal::Program& program,
    const std::string& str, // could be path or actual kenrel
    const CoreSpec& core_spec,
    const std::variant<tt::tt_metal::DataMovementConfig, tt::tt_metal::ComputeConfig>& config);

std::string to_string_precise(float value);

template <typename T>
std::optional<T> at_index(const std::vector<T>& vec, size_t index) {
    if (index < vec.size()) {
        return vec[index];
    }
    return std::nullopt;
}
