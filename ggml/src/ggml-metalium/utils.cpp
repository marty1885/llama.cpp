#include <unordered_map>
#include <filesystem>
#include "utils.hpp"

#ifdef GGML_METALIUM_EMBED_KERNELS
std::unordered_map<std::string, std::string>& ggml_metalium_get_kernel_map();
#endif

using namespace tt::tt_metal;

CBHandle MakeCircularBuffer(
    Program& program, const CoreSpec& core, tt::CBIndex cb, uint32_t size, uint32_t page_size, tt::DataFormat format) {
    CircularBufferConfig cb_src0_config = CircularBufferConfig(size, {{cb, format}}).set_page_size(cb, page_size);
    return CreateCircularBuffer(program, core, cb_src0_config);
}

CBHandle MakeCircularBuffer(Program& program, const CoreSpec& core, tt::CBIndex cb, uint32_t n_tiles, tt::tt_metal::DataType dtype)
{
    auto df2dt = [](tt::tt_metal::DataType dt) {
        switch(dt) {
            case tt::tt_metal::DataType::FLOAT32: return tt::DataFormat::Float32;
            case tt::tt_metal::DataType::BFLOAT16: return tt::DataFormat::Float16_b;
            case tt::tt_metal::DataType::INT32: return tt::DataFormat::Int32;
            case tt::tt_metal::DataType::BFLOAT8_B: return tt::DataFormat::Bfp8_b;
            case tt::tt_metal::DataType::BFLOAT4_B: return tt::DataFormat::Bfp4_b;
            case tt::tt_metal::DataType::UINT8: return tt::DataFormat::UInt8;
            case tt::tt_metal::DataType::UINT16: return tt::DataFormat::UInt16;
            case tt::tt_metal::DataType::UINT32: return tt::DataFormat::UInt32;
            default:
                TT_FATAL(false, "Unsupported data type: {}", static_cast<int>(dt));
        }
    };

    auto tile_size = tt::tile_size(df2dt(dtype));
    return MakeCircularBuffer(program, core, cb, n_tiles*tile_size, tile_size, df2dt(dtype));
}

KernelHandle CreateMetaliumKernel(
    Program& program,
    const std::string& str, // could be path or actual kenrel
    const CoreSpec& core_spec,
    const std::variant<DataMovementConfig, ComputeConfig>& config) {

    if(str.find_first_of(" \n\t") != std::string::npos) {
        return tt::tt_metal::CreateKernelFromString(program, str, core_spec, config);
    }
#ifdef GGML_METALIUM_EMBED_KERNELS
    // if it looks like a name
    if(str.find_first_of(" \n\t") == std::string::npos) {
        auto& kernel_map = ggml_metalium_get_kernel_map();
        auto it = kernel_map.find(str);
        if(it != kernel_map.end()) {
            return tt::tt_metal::CreateKernelFromString(program, it->second, core_spec, config);
        }
    }
#endif

    namespace fs = std::filesystem;
    if(fs::exists(str)) {
        return tt::tt_metal::CreateKernel(program, str, core_spec, config);
    }

    if(!fs::path(str).is_absolute()) {
        // We are at root of GGML dir
        fs::path p = fs::current_path() / "ggml/src/ggml-metalium/kernels/" / str;
        if(fs::exists(p)) {
            return tt::tt_metal::CreateKernel(program, p.string(), core_spec, config);
        }
        if(p.extension() != ".cpp") {
            p = (fs::current_path() / "ggml/src/ggml-metalium/kernels/" / str).generic_string() + ".cpp";
            if(fs::exists(p)) {
                return tt::tt_metal::CreateKernel(program, p.string(), core_spec, config);
            }
        }

        // we are at the build folder of llama.cpp
        p = fs::current_path() / "../ggml/src/ggml-metalium/kernels/" / str;
        if(fs::exists(p)) {
            return tt::tt_metal::CreateKernel(program, p.string(), core_spec, config);
        }
        if(p.extension() != ".cpp") {
            p = (fs::current_path() / "../ggml/src/ggml-metalium/kernels/" / str).generic_string() + ".cpp";
            if(fs::exists(p)) {
                return tt::tt_metal::CreateKernel(program, p.string(), core_spec, config);
            }
        }

        const char* metalium_kernel_root = getenv("GGML_METALIUM_KERNEL_ROOT");
        if(metalium_kernel_root != nullptr) {
            p = fs::path(metalium_kernel_root) / str;
            if(fs::exists(p)) {
                return tt::tt_metal::CreateKernel(program, p.string(), core_spec, config);
            }
        }
    }

    throw std::runtime_error("Kernel " + str + " not found in any search path nor itself looks like a kernel");
}

std::string to_string_precise(float value)
{
    std::stringstream ss;
    ss << std::hexfloat << value;
    return ss.str();
}
