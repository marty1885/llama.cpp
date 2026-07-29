#pragma once

#include "rwkv_experiment.hpp"

#include <algorithm>
#include <cstdint>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace rwkv_jacobian_lens {

inline constexpr int32_t lens_spec_version = 1;

// This contract is recorded with every response. Positions index input tokens;
// offset zero is the source token's final target activation.
struct lens_spec {
    std::string source_tap;
    std::string target_tap;
    std::vector<int32_t> offsets;
    uint64_t direction_seed = 1;
    float relative_epsilon = 0.01f;
    std::string direction = "seeded_unit_l2_rademacher";
};

inline void validate(const lens_spec & spec) {
    if (spec.source_tap.empty() || spec.target_tap.empty()) {
        throw std::runtime_error("LensSpec requires exact source and target tap names");
    }
    if (spec.offsets.empty()) {
        throw std::runtime_error("LensSpec requires at least one future offset");
    }
    if (spec.relative_epsilon <= 0.0f || spec.direction.empty()) {
        throw std::runtime_error("LensSpec relative_epsilon must be positive");
    }
    for (const int32_t offset : spec.offsets) {
        if (offset < 0) {
            throw std::runtime_error("LensSpec offsets must be non-negative");
        }
    }
}

inline void normalize_offsets(lens_spec & spec) {
    std::sort(spec.offsets.begin(), spec.offsets.end());
    spec.offsets.erase(std::unique(spec.offsets.begin(), spec.offsets.end()), spec.offsets.end());
}

inline std::string exact_regex(const std::string & name) {
    static constexpr char special[] = R"(\.^$|()[]{}*+?)";
    std::string result = "^";
    for (const char ch : name) {
        if (std::char_traits<char>::find(special, sizeof(special) - 1, ch)) {
            result += '\\';
        }
        result += ch;
    }
    return result + "$";
}

inline void write_json(std::ostream & output, const lens_spec & spec) {
    output << "{\n    \"version\": " << lens_spec_version
           << ",\n    \"source_tap\": ";
    rwkv_experiment::write_json_string(output, spec.source_tap);
    output << ",\n    \"target_tap\": ";
    rwkv_experiment::write_json_string(output, spec.target_tap);
    output << ",\n    \"position_semantics\": \"input-token index; offset zero is the source token\""
           << ",\n    \"suffix_semantics\": \"exact teacher-forced input tokens in both branches\""
           << ",\n    \"derivative\": \"central finite difference\""
           << ",\n    \"direction\": ";
    rwkv_experiment::write_json_string(output, spec.direction);
    output
           << ",\n    \"relative_epsilon\": " << spec.relative_epsilon
           << ",\n    \"direction_seed\": " << spec.direction_seed
           << ",\n    \"offsets\": [";
    for (size_t i = 0; i < spec.offsets.size(); ++i) {
        if (i) output << ", ";
        output << spec.offsets[i];
    }
    output << "]\n  }";
}

} // namespace rwkv_jacobian_lens
