#pragma once

#include "interp.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace rwkv_experiment {

// A captured production-graph step. Experiments can turn any captured activation into
// one or more named logit records without changing the runtime or capture machinery.
struct token_capture {
    int32_t position;
    llama_token input_token;
    llama_interp::activation_set activations;
};

struct logit_record {
    std::string name;
    int32_t layer = -1;
    int32_t position;
    llama_token input_token;
    std::vector<float> logits;
};

struct logit_document {
    std::string experiment;
    std::string model;
    std::string prompt;
    std::vector<std::pair<std::string, std::string>> metadata;
    std::vector<logit_record> outputs;
};

inline llama_interp::task<> capture_each_token(
        llama_interp::runtime & runtime,
        const llama_interp::rwkv_state & initial,
        const std::vector<llama_token> & tokens,
        const std::vector<std::string> & taps,
        std::vector<token_capture> & output) {
    llama_interp::rwkv_state state = initial;
    for (size_t position = 0; position < tokens.size(); ++position) {
        llama_interp::activation_set activations;
        auto call = runtime.prefill_tokens(state, { tokens[position] });
        for (const std::string & tap : taps) {
            call.capture_f32("^" + tap + "$", activations);
        }
        state = co_await call;
        output.push_back({ (int32_t) position, tokens[position], std::move(activations) });
    }
}

inline const llama_interp_activation & require_capture(
        const llama_interp::activation_set & captures,
        const std::string & name) {
    const llama_interp_activation * result = nullptr;
    for (const auto & capture : captures) {
        if (capture.name != name || capture.data_f32.empty()) {
            continue;
        }
        if (result != nullptr) {
            throw std::runtime_error("duplicate capture: " + name);
        }
        result = &capture;
    }
    if (result == nullptr) {
        throw std::runtime_error("missing FP32 capture: " + name);
    }
    return *result;
}

inline void write_json_string(std::ostream & output, const std::string & value) {
    output.put('"');
    for (const unsigned char ch : value) {
        switch (ch) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\b': output << "\\b"; break;
            case '\f': output << "\\f"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (ch < 0x20) {
                    static constexpr char hex[] = "0123456789abcdef";
                    output << "\\u00" << hex[ch >> 4] << hex[ch & 0x0f];
                } else {
                    output.put((char) ch);
                }
        }
    }
    output.put('"');
}

inline void write_top_logits_json(
        const std::string & path,
        const llama_context * ctx,
        const logit_document & document,
        int32_t top_n) {
    if (top_n <= 0) {
        throw std::runtime_error("top_n must be positive");
    }

    const std::filesystem::path output_path(path);
    if (!output_path.parent_path().empty()) {
        std::filesystem::create_directories(output_path.parent_path());
    }
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("failed to open JSON output: " + path);
    }
    output << std::setprecision(9);
    output << "{\n  \"schema_version\": 2,\n  \"experiment\": ";
    write_json_string(output, document.experiment);
    output << ",\n  \"model\": ";
    write_json_string(output, document.model);
    output << ",\n  \"prompt\": ";
    write_json_string(output, document.prompt);
    output << ",\n  \"top_n\": " << top_n << ",\n  \"metadata\": {";
    for (size_t i = 0; i < document.metadata.size(); ++i) {
        if (i != 0) {
            output << ',';
        }
        output << "\n    ";
        write_json_string(output, document.metadata[i].first);
        output << ": ";
        write_json_string(output, document.metadata[i].second);
    }
    if (!document.metadata.empty()) {
        output << '\n';
    }
    output << "  },\n  \"inputs\": [";
    std::map<int32_t, std::map<int32_t, std::vector<const logit_record *>>> hierarchy;
    for (const auto & record : document.outputs) {
        hierarchy[record.position][record.layer].push_back(&record);
    }

    size_t input_index = 0;
    for (const auto & [position, layers] : hierarchy) {
        const logit_record * input_record = layers.begin()->second.front();
        if (input_index++ != 0) output << ',';
        output << "\n    {\n      \"position\": " << position << ",\n      \"input_token\": {\"id\": "
               << input_record->input_token << ", \"text\": ";
        write_json_string(output, common_token_to_piece(ctx, input_record->input_token, true));
        output << "},\n      \"layers\": [";
        size_t layer_index = 0;
        for (const auto & [layer, records] : layers) {
            if (layer_index++ != 0) output << ',';
            output << "\n        {\n          \"layer\": " << layer << ",\n          \"lenses\": [";
            for (size_t record_index = 0; record_index < records.size(); ++record_index) {
                const auto & record = *records[record_index];
                if (record.logits.empty()) {
                    throw std::runtime_error("empty logits for: " + record.name);
                }
                if (record_index != 0) output << ',';
                std::vector<int32_t> indices(record.logits.size());
                std::iota(indices.begin(), indices.end(), 0);
                const size_t count = std::min<size_t>(top_n, indices.size());
                std::partial_sort(indices.begin(), indices.begin() + count, indices.end(),
                    [&record](int32_t left, int32_t right) { return record.logits[left] > record.logits[right]; });

                output << "\n            {\n              \"lens\": ";
                write_json_string(output, record.name);
                output << ",\n              \"result\": {\n                \"top_logits\": [";
                for (size_t i = 0; i < count; ++i) {
                    const int32_t token = indices[i];
                    if (i != 0) output << ',';
                    output << "\n                  {\"token_id\": " << token << ", \"token\": ";
                    write_json_string(output, common_token_to_piece(ctx, token, true));
                    output << ", \"logit\": " << record.logits[token] << '}';
                }
                output << "\n                ]\n              }\n            }";
            }
            output << "\n          ]\n        }";
        }
        output << "\n      ]\n    }";
    }
    output << "\n  ]\n}\n";
    if (!output) {
        throw std::runtime_error("failed to write JSON output: " + path);
    }
}

} // namespace rwkv_experiment
