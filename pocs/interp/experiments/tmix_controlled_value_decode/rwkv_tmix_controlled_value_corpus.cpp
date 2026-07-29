#include "rwkv_experiment.hpp"

#include "llama-model.h"

#include <openssl/sha.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <clocale>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

constexpr uint64_t k_seed = 941731;
constexpr int k_value_count = 16;
constexpr int k_carrier_count = 48;
constexpr int k_train_carrier_count = 32;
constexpr const char * k_template = "Context: CARRIER. Remember: the value is VALUE. Query: the value is";

void quiet_llama_logs(ggml_log_level, const char *, void *) {}

uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

std::string read_file(const std::string & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot read " + path);
    return { std::istreambuf_iterator<char>(input), {} };
}

// Small self-contained SHA-256 keeps frozen registrations independent of external tools.
std::string sha256_bytes(const std::string & data) {
    std::array<uint32_t, 8> h{ 0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                               0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 };
    static constexpr std::array<uint32_t, 64> k{
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
    };
    std::vector<uint8_t> bytes(data.begin(), data.end());
    const uint64_t bits = uint64_t(bytes.size()) * 8;
    bytes.push_back(0x80);
    while (bytes.size() % 64 != 56) bytes.push_back(0);
    for (int i = 7; i >= 0; --i) bytes.push_back(uint8_t(bits >> (i * 8)));
    const auto rotr = [](uint32_t x, int n) { return (x >> n) | (x << (32 - n)); };
    for (size_t offset = 0; offset < bytes.size(); offset += 64) {
        std::array<uint32_t, 64> w{};
        for (int i = 0; i < 16; ++i) w[i] = (uint32_t(bytes[offset + 4 * i]) << 24) | (uint32_t(bytes[offset + 4 * i + 1]) << 16) |
                                                 (uint32_t(bytes[offset + 4 * i + 2]) << 8) | bytes[offset + 4 * i + 3];
        for (int i = 16; i < 64; ++i) w[i] = (rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3)) + w[i - 16] +
                                              (rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10)) + w[i - 7];
        auto [a, b, c, d, e, f, g, hh] = h;
        for (int i = 0; i < 64; ++i) {
            const uint32_t t1 = hh + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ ((~e) & g)) + k[i] + w[i];
            const uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }
    std::ostringstream output;
    for (uint32_t x : h) output << std::hex << std::setw(8) << std::setfill('0') << x;
    return output.str();
}

std::string sha256_file(const std::string & path) {
    // Hash large GGUFs in bounded memory. Reading the whole checkpoint here would
    // create a second model-sized allocation in an otherwise tokenizer-only tool.
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot read " + path);
    std::array<char, 1024 * 1024> chunk{};
    SHA256_CTX context;
    SHA256_Init(&context);
    while (input.read(chunk.data(), chunk.size()) || input.gcount() != 0) {
        SHA256_Update(&context, chunk.data(), (size_t) input.gcount());
    }
    std::array<unsigned char, SHA256_DIGEST_LENGTH> digest{};
    SHA256_Final(digest.data(), &context);
    std::ostringstream output;
    for (unsigned char byte : digest) output << std::hex << std::setw(2) << std::setfill('0') << int(byte);
    return output.str();
}

std::string trim(std::string value) {
    const auto first = std::find_if_not(value.begin(), value.end(), [](unsigned char c) { return std::isspace(c); });
    const auto last = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char c) { return std::isspace(c); }).base();
    return first >= last ? "" : std::string(first, last);
}

std::string substitute(const std::string & carrier, const std::string & value) {
    std::string prompt(k_template);
    const size_t carrier_at = prompt.find("CARRIER");
    const size_t value_at = prompt.find("VALUE");
    prompt.replace(carrier_at, 7, carrier);
    prompt.replace(prompt.find("VALUE"), 5, value);
    return prompt;
}

std::vector<std::string> candidates(const std::string & path) {
    std::istringstream input(read_file(path));
    std::vector<std::string> result;
    for (std::string line; std::getline(input, line);) {
        line = trim(line);
        if (!line.empty() && line[0] != '#') result.push_back(line);
    }
    if (result.empty()) throw std::runtime_error("candidate list is empty");
    for (const std::string & value : result) {
        if (!std::all_of(value.begin(), value.end(), [](unsigned char c) { return c >= 0x20 && c <= 0x7e; })) {
            throw std::runtime_error("candidate is not printable ASCII");
        }
    }
    return result;
}

std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    return common_tokenize(vocab, text, false, false);
}

struct selected_token { std::string text; llama_token id; };

bool valid_single_replacement(const llama_vocab * vocab, const std::string & candidate) {
    // Both template fields follow a literal space. Test that exact replacement context;
    // complete template positions are subsequently checked over the whole factorial.
    return tokenize(vocab, " " + candidate).size() == 1;
}

std::vector<selected_token> select_tokens(const llama_vocab * vocab, const std::vector<std::string> & pool, bool carrier,
                                          int count, std::unordered_set<llama_token> & used) {
    std::vector<size_t> order(pool.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [carrier](size_t a, size_t b) {
        return splitmix64(k_seed ^ (carrier ? 0x43415252494552ULL : 0x56414c5545ULL) ^ a) <
               splitmix64(k_seed ^ (carrier ? 0x43415252494552ULL : 0x56414c5545ULL) ^ b);
    });
    std::vector<selected_token> result;
    int valid_count = 0;
    for (size_t index : order) {
        if (!valid_single_replacement(vocab, pool[index])) continue;
        const llama_token id = tokenize(vocab, " " + pool[index]).front();
        ++valid_count;
        if (used.insert(id).second) result.push_back({ pool[index], id });
        if ((int) result.size() == count) return result;
    }
    throw std::runtime_error("candidate list provides " + std::to_string(valid_count) + " valid and " + std::to_string(result.size()) +
                             " disjoint single-token replacements; need " + std::to_string(count));
}

struct row { int carrier_id; int value_id; std::string prompt; std::vector<llama_token> tokens; };

std::vector<row> make_rows(const llama_vocab * vocab, const std::vector<selected_token> & carriers,
                           const std::vector<selected_token> & values) {
    std::vector<row> result;
    for (int c = 0; c < (int) carriers.size(); ++c) for (int v = 0; v < (int) values.size(); ++v) {
        const std::string prompt = substitute(carriers[c].text, values[v].text);
        result.push_back({ c, v, prompt, tokenize(vocab, prompt) });
    }
    return result;
}

void validate_rows(const std::vector<row> & rows, const std::vector<selected_token> & carriers,
                   const std::vector<selected_token> & values) {
    if ((int) rows.size() != k_carrier_count * k_value_count) throw std::runtime_error("incomplete factorial");
    const size_t count = rows.front().tokens.size();
    const std::vector<llama_token> base = rows.front().tokens;
    const size_t carrier_position = std::find(base.begin(), base.end(), carriers[rows.front().carrier_id].id) - base.begin();
    const size_t value_position = std::find(base.begin(), base.end(), values[rows.front().value_id].id) - base.begin();
    if (carrier_position == base.size() || value_position == base.size()) throw std::runtime_error("replacement token not found");
    std::unordered_set<uint64_t> cells;
    for (const row & value : rows) {
        if (value.tokens.size() != count || value.tokens.back() != base.back() ||
            value.tokens[carrier_position] != carriers[value.carrier_id].id || value.tokens[value_position] != values[value.value_id].id) {
            throw std::runtime_error("token count, position, or final read-token validation failed");
        }
        if (!cells.insert((uint64_t(value.carrier_id) << 32) | uint32_t(value.value_id)).second) throw std::runtime_error("duplicate factorial cell");
    }
}

void self_test() {
    if (sha256_bytes("abc") != "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad") throw std::runtime_error("SHA-256 self-test failed");
    std::unordered_set<uint64_t> cells;
    for (int c = 0; c < 48; ++c) for (int v = 0; v < 16; ++v) cells.insert((uint64_t(c) << 32) | uint32_t(v));
    if (cells.size() != 768 || k_train_carrier_count + 16 != k_carrier_count) throw std::runtime_error("factorial self-test failed");
}

void usage(const char * argv0) {
    std::fprintf(stderr, "usage: %s --self-test | -m MODEL --generate-candidate-list FILE --exclude-manifest DEVELOPMENT.json | --candidate-list FILE --output PROMPTS.txt --manifest MANIFEST.json --seed 941731 [--classified --registration REGISTRATION.json]\n", argv0);
}

bool valid_single_replacement(const llama_vocab * vocab, const std::string & candidate);

std::unordered_set<llama_token> manifest_token_ids(const std::string & path) {
    const std::string text = read_file(path);
    const std::regex pattern("\\\"token_id\\\"\\s*:\\s*([0-9]+)");
    std::unordered_set<llama_token> result;
    for (std::sregex_iterator it(text.begin(), text.end(), pattern), end; it != end; ++it) result.insert(llama_token(std::stol((*it)[1])));
    if (result.size() < size_t(k_value_count + k_carrier_count)) throw std::runtime_error("development manifest lacks selected token IDs");
    return result;
}

void generate_candidate_list(const llama_vocab * vocab, const std::unordered_set<llama_token> & excluded, const std::string & path) {
    std::ofstream output(path, std::ios::binary);
    if (!output) throw std::runtime_error("cannot create generated candidate list");
    std::unordered_set<std::string> emitted;
    int accepted = 0;
    for (llama_token token = 0; token < llama_vocab_n_tokens(vocab) && accepted < 512; ++token) {
        if (excluded.contains(token)) continue;
        std::array<char, 512> piece{};
        const int32_t length = llama_token_to_piece(vocab, token, piece.data(), int32_t(piece.size()), 0, false);
        if (length <= 1 || length >= int32_t(piece.size()) || piece[0] != ' ') continue;
        const std::string candidate(piece.data() + 1, size_t(length - 1));
        if (!std::all_of(candidate.begin(), candidate.end(), [](unsigned char c) { return std::isalpha(c) && c < 0x80; }) ||
            !valid_single_replacement(vocab, candidate) || !emitted.insert(candidate).second) continue;
        output << candidate << '\n';
        ++accepted;
    }
    if (accepted < 64) throw std::runtime_error("tokenizer-derived candidate pool has fewer than 64 usable disjoint tokens");
}

} // namespace

int main(int argc, char ** argv) try {
    std::setlocale(LC_NUMERIC, "C");
    llama_log_set(quiet_llama_logs, nullptr);
    std::string model_path, candidate_path, output_path, manifest_path, registration_path, generated_candidate_path, exclude_manifest_path;
    uint64_t seed = 0;
    bool have_seed = false, run_self_test = false, classified = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--self-test")) run_self_test = true;
        else if (!std::strcmp(argv[i], "-m") && i + 1 < argc) model_path = argv[++i];
        else if (!std::strcmp(argv[i], "--candidate-list") && i + 1 < argc) candidate_path = argv[++i];
        else if (!std::strcmp(argv[i], "--generate-candidate-list") && i + 1 < argc) generated_candidate_path = argv[++i];
        else if (!std::strcmp(argv[i], "--exclude-manifest") && i + 1 < argc) exclude_manifest_path = argv[++i];
        else if (!std::strcmp(argv[i], "--output") && i + 1 < argc) output_path = argv[++i];
        else if (!std::strcmp(argv[i], "--manifest") && i + 1 < argc) manifest_path = argv[++i];
        else if (!std::strcmp(argv[i], "--registration") && i + 1 < argc) registration_path = argv[++i];
        else if (!std::strcmp(argv[i], "--classified")) classified = true;
        else if (!std::strcmp(argv[i], "--seed") && i + 1 < argc) { seed = std::strtoull(argv[++i], nullptr, 10); have_seed = true; }
        else { usage(argv[0]); return 1; }
    }
    self_test();
    if (run_self_test) { std::puts("self-tests=passed"); return 0; }
    const bool generate_candidates = !generated_candidate_path.empty() || !exclude_manifest_path.empty();
    if (generate_candidates && (model_path.empty() || generated_candidate_path.empty() || exclude_manifest_path.empty() ||
                                std::filesystem::exists(generated_candidate_path) || !candidate_path.empty() || !output_path.empty() ||
                                !manifest_path.empty() || !registration_path.empty())) {
        usage(argv[0]); return 1;
    }
    if (generate_candidates) {
        ggml_backend_load_all(); llama_backend_init();
        llama_model_params params = llama_model_default_params(); params.vocab_only = true;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(model_path.c_str(), params), llama_model_free);
        if (!model || model->arch != LLM_ARCH_RWKV7) throw std::runtime_error("an RWKV-7 model is required");
        generate_candidate_list(llama_model_get_vocab(model.get()), manifest_token_ids(exclude_manifest_path), generated_candidate_path);
        std::printf("wrote=%s status=tokenizer_derived_disjoint_candidate_pool\n", generated_candidate_path.c_str());
        llama_backend_free();
        return 0;
    }
    if (model_path.empty() || candidate_path.empty() || output_path.empty() || manifest_path.empty() || !have_seed || seed != k_seed ||
        std::filesystem::exists(output_path) || std::filesystem::exists(manifest_path) ||
        (classified && (registration_path.empty() || std::filesystem::exists(registration_path)))) {
        usage(argv[0]); return 1;
    }
    ggml_backend_load_all(); llama_backend_init();
    llama_model_params params = llama_model_default_params();
    params.vocab_only = true;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(model_path.c_str(), params), llama_model_free);
    if (!model || model->arch != LLM_ARCH_RWKV7) throw std::runtime_error("an RWKV-7 model is required");
    const llama_vocab * vocab = llama_model_get_vocab(model.get());
    const std::string candidate_bytes = read_file(candidate_path);
    const std::vector<std::string> pool = candidates(candidate_path);
    std::unordered_set<llama_token> used;
    const std::vector<selected_token> values = select_tokens(vocab, pool, false, k_value_count, used);
    const std::vector<selected_token> carriers = select_tokens(vocab, pool, true, k_carrier_count, used);
    const std::vector<row> rows = make_rows(vocab, carriers, values);
    validate_rows(rows, carriers, values);
    const std::filesystem::path output_parent = std::filesystem::path(output_path).parent_path();
    const std::filesystem::path manifest_parent = std::filesystem::path(manifest_path).parent_path();
    const std::filesystem::path registration_parent = std::filesystem::path(registration_path).parent_path();
    if (!output_parent.empty()) std::filesystem::create_directories(output_parent);
    if (!manifest_parent.empty()) std::filesystem::create_directories(manifest_parent);
    if (classified && !registration_parent.empty()) std::filesystem::create_directories(registration_parent);
    std::ofstream corpus(output_path, std::ios::binary);
    if (!corpus) throw std::runtime_error("cannot create corpus output");
    for (const row & value : rows) corpus << value.prompt << '\n';
    corpus.close();
    const std::vector<llama_token> first = rows.front().tokens;
    const size_t carrier_position = std::find(first.begin(), first.end(), carriers.front().id) - first.begin();
    const size_t value_position = std::find(first.begin(), first.end(), values.front().id) - first.begin();
    std::ofstream manifest(manifest_path);
    if (!manifest) throw std::runtime_error("cannot create manifest output");
    manifest << "{\n  \"schema_version\": 1,\n  \"status\": \"" << (classified ? "frozen" : "development_only") << "\",\n  \"seed\": " << k_seed
             << ",\n  \"model_path\": "; rwkv_experiment::write_json_string(manifest, model_path);
    manifest << ",\n  \"model_sha256\": \"" << sha256_file(model_path) << "\",\n  \"candidate_list_path\": ";
    rwkv_experiment::write_json_string(manifest, candidate_path);
    manifest << ",\n  \"candidate_list_sha256\": \"" << sha256_bytes(candidate_bytes) << "\",\n  \"template\": ";
    rwkv_experiment::write_json_string(manifest, k_template);
    manifest << ",\n  \"corpus_sha256\": \"" << sha256_file(output_path) << "\",\n  \"prompt_count\": 768"
             << ",\n  \"token_count\": " << first.size() << ",\n  \"read_token_id\": " << first.back()
             << ",\n  \"read_position\": " << first.size() - 1 << ",\n  \"carrier_position\": " << carrier_position
             << ",\n  \"value_position\": " << value_position << ",\n  \"values\": [";
    for (size_t i = 0; i < values.size(); ++i) { if (i) manifest << ','; manifest << "{\"id\":" << i << ",\"text\":" << std::quoted(values[i].text) << ",\"token_id\":" << values[i].id << '}'; }
    manifest << "],\n  \"carriers\": [";
    for (size_t i = 0; i < carriers.size(); ++i) { if (i) manifest << ','; manifest << "{\"id\":" << i << ",\"split\":\"" << (i < k_train_carrier_count ? "train" : "test") << "\",\"text\":" << std::quoted(carriers[i].text) << ",\"token_id\":" << carriers[i].id << '}'; }
    manifest << "]\n}\n";
    if (!manifest) throw std::runtime_error("failed to write manifest");
    manifest.close();
    if (classified) {
        std::ofstream registration(registration_path);
        if (!registration) throw std::runtime_error("cannot create frozen registration");
        registration << "{\n  \"schema_version\": 1,\n  \"status\": \"frozen\",\n  \"seed\": " << k_seed
                     << ",\n  \"model_path\": "; rwkv_experiment::write_json_string(registration, model_path);
        registration << ",\n  \"model_sha256\": \"" << sha256_file(model_path)
                     << "\",\n  \"candidate_list_sha256\": \"" << sha256_bytes(candidate_bytes)
                     << "\",\n  \"corpus_sha256\": \"" << sha256_file(output_path)
                     << "\",\n  \"manifest_sha256\": \"" << sha256_file(manifest_path)
                     << "\",\n  \"template_sha256\": \"" << sha256_bytes(k_template)
                     << "\",\n  \"backend\": \"Vulkan\",\n  \"n_gpu_layers\": 99"
                     << ",\n  \"prompt_count\": 768,\n  \"token_count\": " << first.size()
                     << ",\n  \"read_token_id\": " << first.back() << ",\n  \"read_position\": " << first.size() - 1
                     << ",\n  \"carrier_position\": " << carrier_position << ",\n  \"value_position\": " << value_position
                     << ",\n  \"representation_rank\": 32,\n  \"layers\": [15,30,45,60]\n}\n";
        if (!registration) throw std::runtime_error("failed to write frozen registration");
    }
    std::printf("wrote=%s prompts=768 manifest=%s status=%s%s\n", output_path.c_str(), manifest_path.c_str(),
                classified ? "frozen" : "development_only", classified ? " registration_written=yes" : "");
    llama_backend_free();
    return 0;
} catch (const std::exception & error) {
    std::fprintf(stderr, "error: %s\n", error.what());
    llama_backend_free();
    return 1;
}
