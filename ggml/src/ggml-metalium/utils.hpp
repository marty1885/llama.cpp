#include <unordered_map>

#include "tt-metalium/host_api.hpp"
#include "tt-metalium/kernel_types.hpp"
#include "tt-metalium/tt_backend_api_types.hpp"

std::unordered_map<std::string, std::string>& ggml_metalium_get_kernel_map();

using CoreSpec = std::variant<CoreCoord, CoreRange, CoreRangeSet>;
tt::tt_metal::CBHandle MakeCircularBuffer(
    tt::tt_metal::Program& program, const CoreSpec& core, tt::CBIndex cb, uint32_t size, uint32_t page_size, tt::DataFormat format);
tt::tt_metal::CBHandle MakeCircularBuffer(tt::tt_metal::Program& program, const CoreSpec& core, tt::CBIndex cb, uint32_t n_tiles, tt::tt_metal::DataType dtype);

tt::tt_metal::KernelHandle CreateMetaliumKernel(
    tt::tt_metal::Program& program,
    const std::string& str, // could be path or actual kenrel
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const std::variant<tt::tt_metal::DataMovementConfig, tt::tt_metal::ComputeConfig, tt::tt_metal::EthernetConfig>& config);
