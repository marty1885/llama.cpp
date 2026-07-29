#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
uint64_t fnv(const std::string & value, uint64_t hash = 1469598103934665603ULL) { for (unsigned char c : value) { hash ^= c; hash *= 1099511628211ULL; } return hash; }
uint64_t prompt_id(uint64_t index, const std::string & value) { std::string prefix(8, '\0'); for (int i = 0; i < 8; ++i) prefix[i] = index >> (8 * i); return fnv(value, fnv(prefix)); }
uint64_t split_hash(uint64_t value, uint64_t seed) { std::string bytes(8, '\0'); for (int i = 0; i < 8; ++i) bytes[i] = value >> (8 * i); return fnv(bytes, 1469598103934665603ULL ^ seed); }
std::string decode_base64(const std::string & value) {
    static const std::string alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string output; int buffer = 0, bits = -8;
    for (unsigned char c : value) { if (c == '=') break; const int x = alphabet.find(c); if (x < 0) throw std::runtime_error("invalid base64 input"); buffer = (buffer << 6) | x; bits += 6; if (bits >= 0) { output.push_back((char) ((buffer >> bits) & 0xff)); bits -= 8; } }
    return output;
}
std::string single_line(std::string text) { for (char & c : text) if (std::isspace((unsigned char) c)) c = ' '; std::string out; bool previous_space = true; for (char c : text) { if (c != ' ' || !previous_space) out += c; previous_space = c == ' '; } if (!out.empty() && out.back() == ' ') out.pop_back(); return out; }
struct row { uint64_t source_row; std::string source, text; };
}

int main(int argc, char ** argv) try {
    std::string input, output, manifest; uint64_t seed = 0; bool have_seed = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--input") && i + 1 < argc) input = argv[++i];
        else if (!std::strcmp(argv[i], "--output") && i + 1 < argc) output = argv[++i];
        else if (!std::strcmp(argv[i], "--manifest") && i + 1 < argc) manifest = argv[++i];
        else if (!std::strcmp(argv[i], "--seed") && i + 1 < argc) { seed = std::strtoull(argv[++i], nullptr, 10); have_seed = true; }
        else throw std::runtime_error("usage: --input ROWS.tsv --output PROMPTS.txt --manifest MANIFEST.json --seed N");
    }
    if (input.empty() || output.empty() || manifest.empty() || !have_seed || std::filesystem::exists(output) || std::filesystem::exists(manifest)) throw std::runtime_error("missing argument or existing output");
    const std::vector<std::string> sources = { "Pile-CC", "OpenWebText2", "StackExchange", "PubMed Abstracts", "Github", "USPTO Backgrounds" };
    std::map<std::string, std::vector<row>> selected; std::ifstream file(input); std::string line;
    while (std::getline(file, line)) { const size_t a = line.find('\t'), b = line.find('\t', a + 1); if (a == std::string::npos || b == std::string::npos) continue; const std::string source = line.substr(a + 1, b - a - 1); if (std::find(sources.begin(), sources.end(), source) == sources.end() || selected[source].size() == 8) continue; const std::string text = single_line(decode_base64(line.substr(b + 1))); if (text.size() >= 512) selected[source].push_back({ std::strtoull(line.substr(0, a).c_str(), nullptr, 10), source, text }); }
    std::vector<row> rows; for (const auto & source : sources) { if (selected[source].size() != 8) throw std::runtime_error("insufficient eligible rows for " + source); rows.insert(rows.end(), selected[source].begin(), selected[source].end()); }
    std::filesystem::create_directories(std::filesystem::path(output).parent_path()); std::ofstream prompts(output), meta(manifest); if (!prompts || !meta) throw std::runtime_error("failed to create output");
    for (const auto & value : rows) prompts << value.text << '\n'; prompts.close(); std::ifstream completed(output, std::ios::binary); const std::string corpus((std::istreambuf_iterator<char>(completed)), {});
    meta << "{\n  \"schema_version\": 1,\n  \"dataset\": \"NeelNanda/pile-10k\",\n  \"dataset_revision\": \"main\",\n  \"source_artifact_sha256\": \"a1a9475a8684ac8f1b17a36eccb2ec49c127edd7aae9beb2f240726972d93f31\",\n  \"seed\": " << seed << ",\n  \"corpus_fnv1a64\": \"" << fnv(corpus) << "\",\n  \"prompts\": [\n";
    for (size_t i = 0; i < rows.size(); ++i) { const uint64_t id = prompt_id(i, rows[i].text); if (i) meta << ",\n"; meta << "    {\"line_index\": " << i << ", \"prompt_id\": \"" << id << "\", \"split\": \"" << ((split_hash(id, seed) >> 63) == 0 ? "train" : "test") << "\", \"pile_set_name\": \"" << rows[i].source << "\", \"source_row\": " << rows[i].source_row << ", \"prompt_fnv1a64\": \"" << fnv(rows[i].text) << "\"}"; }
    meta << "\n  ]\n}\n"; std::printf("wrote=%s prompts=%zu manifest=%s\n", output.c_str(), rows.size(), manifest.c_str());
    return 0;
} catch (const std::exception & e) { std::fprintf(stderr, "error: %s\n", e.what()); return 1; }
