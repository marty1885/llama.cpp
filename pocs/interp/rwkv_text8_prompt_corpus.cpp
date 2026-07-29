#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

uint64_t fnv1a(const std::string & bytes) {
    uint64_t hash = 1469598103934665603ULL;
    for (unsigned char byte : bytes) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    return hash;
}

uint64_t split_hash(uint64_t value, uint64_t seed) {
    uint64_t hash = 1469598103934665603ULL ^ seed;
    for (unsigned shift = 0; shift < 64; shift += 8) {
        hash ^= (value >> shift) & 0xff;
        hash *= 1099511628211ULL;
    }
    return hash;
}

uint64_t prompt_identifier(uint64_t line_index, const std::string & prompt) {
    uint64_t hash = 1469598103934665603ULL;
    for (unsigned shift = 0; shift < 64; shift += 8) {
        hash ^= (line_index >> shift) & 0xff;
        hash *= 1099511628211ULL;
    }
    for (unsigned char byte : prompt) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    return hash;
}

void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s --input TEXT8 --output PROMPTS.txt --manifest MANIFEST.json --seed N [--prompt-count N] [--words-per-prompt N] [--start-word N]\n",
        argv0);
}

} // namespace

int main(int argc, char ** argv) try {
    std::string input_path, output_path, manifest_path;
    int prompt_count = 64, words_per_prompt = 256;
    size_t start_word = 0;
    uint64_t seed = 0;
    bool have_seed = false;
    uint64_t inspect_split_seed = 0;
    bool inspect_split = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--input") && i + 1 < argc) input_path = argv[++i];
        else if (!std::strcmp(argv[i], "--output") && i + 1 < argc) output_path = argv[++i];
        else if (!std::strcmp(argv[i], "--manifest") && i + 1 < argc) manifest_path = argv[++i];
        else if (!std::strcmp(argv[i], "--prompt-count") && i + 1 < argc) prompt_count = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--words-per-prompt") && i + 1 < argc) words_per_prompt = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--start-word") && i + 1 < argc) start_word = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "--seed") && i + 1 < argc) { seed = std::strtoull(argv[++i], nullptr, 10); have_seed = true; }
        else if (!std::strcmp(argv[i], "--inspect-split") && i + 1 < argc) { inspect_split_seed = std::strtoull(argv[++i], nullptr, 10); inspect_split = true; }
        else { usage(argv[0]); return 1; }
    }
    if (input_path.empty() || (!inspect_split && (output_path.empty() || manifest_path.empty() || !have_seed)) || prompt_count < 8 || words_per_prompt < 1 ||
        (!inspect_split && (std::filesystem::exists(output_path) || std::filesystem::exists(manifest_path)))) {
        throw std::runtime_error("missing argument, invalid extraction size, or output already exists");
    }
    std::ifstream input(input_path, std::ios::binary);
    if (!input) throw std::runtime_error("failed to open source corpus");
    const std::string source((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    std::vector<std::string> words;
    std::string word;
    for (unsigned char byte : source) {
        if (byte == ' ' || byte == '\n' || byte == '\r' || byte == '\t') {
            if (!word.empty()) { words.push_back(std::move(word)); word.clear(); }
        } else {
            word.push_back((char) byte);
        }
    }
    if (!word.empty()) words.push_back(std::move(word));
    if (start_word >= words.size() || words.size() - start_word < (size_t) prompt_count * words_per_prompt) throw std::runtime_error("source corpus is too short");
    if (inspect_split) {
        int train = 0;
        for (int prompt_id = 0; prompt_id < prompt_count; ++prompt_id) train += (split_hash(prompt_id, inspect_split_seed) >> 63) == 0;
        std::printf("seed=%llu train_prompts=%d test_prompts=%d\n", (unsigned long long) inspect_split_seed, train, prompt_count - train);
        return 0;
    }
    const size_t stride = (words.size() - start_word - words_per_prompt) / (prompt_count - 1);
    if (stride < (size_t) words_per_prompt) throw std::runtime_error("requested spans would overlap");
    const std::filesystem::path output_parent = std::filesystem::path(output_path).parent_path();
    const std::filesystem::path manifest_parent = std::filesystem::path(manifest_path).parent_path();
    if (!output_parent.empty()) std::filesystem::create_directories(output_parent);
    if (!manifest_parent.empty()) std::filesystem::create_directories(manifest_parent);
    std::ofstream output(output_path, std::ios::binary);
    if (!output) throw std::runtime_error("failed to create prompt corpus");
    std::vector<std::string> prompts;
    prompts.reserve(prompt_count);
    for (int prompt_id = 0; prompt_id < prompt_count; ++prompt_id) {
        const size_t begin = start_word + prompt_id * stride;
        std::string prompt;
        for (int word_index = 0; word_index < words_per_prompt; ++word_index) {
            if (word_index) prompt += ' ';
            prompt += words[begin + word_index];
        }
        output << prompt << '\n';
        prompts.push_back(std::move(prompt));
    }
    if (!output) throw std::runtime_error("failed to write prompt corpus");
    std::ifstream completed(output_path, std::ios::binary);
    const std::string corpus((std::istreambuf_iterator<char>(completed)), std::istreambuf_iterator<char>());
    std::ofstream manifest(manifest_path);
    if (!manifest) throw std::runtime_error("failed to create manifest");
    manifest << "{\n  \"schema_version\": 1,\n  \"source_path\": \"" << input_path
             << "\",\n  \"source_fnv1a64\": \"" << fnv1a(source)
             << "\",\n  \"extraction\": \"evenly spaced non-overlapping whitespace-word spans; all parameters fixed before activation collection\""
              << ",\n  \"prompt_count\": " << prompt_count << ",\n  \"words_per_prompt\": " << words_per_prompt
              << ",\n  \"start_word\": " << start_word << ",\n  \"word_stride\": " << stride << ",\n  \"seed\": " << seed
              << ",\n  \"corpus_fnv1a64\": \"" << fnv1a(corpus) << "\",\n  \"prompts\": [\n";
    for (int prompt_id = 0; prompt_id < prompt_count; ++prompt_id) {
        if (prompt_id) manifest << ",\n";
        const uint64_t id = prompt_identifier((uint64_t) prompt_id, prompts[prompt_id]);
        const bool train = (split_hash(id, seed) >> 63) == 0;
        manifest << "    {\"line_index\": " << prompt_id << ", \"prompt_id\": \"" << id << "\", \"split\": \"" << (train ? "train" : "test")
                 << "\", \"start_word\": " << start_word + prompt_id * stride << ", \"fnv1a64\": \"" << fnv1a(prompts[prompt_id]) << "\"}";
    }
    manifest << "\n  ]\n}\n";
    if (!manifest) throw std::runtime_error("failed to write manifest");
    std::printf("wrote=%s prompts=%d words_per_prompt=%d manifest=%s\n", output_path.c_str(), prompt_count, words_per_prompt, manifest_path.c_str());
    return 0;
} catch (const std::exception & error) {
    std::fprintf(stderr, "error: %s\n", error.what());
    return 1;
}
