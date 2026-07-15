#include <TFile.h>
#include <TTree.h>

#include <cstdio>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void usage(const char * argv0) {
    std::fprintf(stderr, "usage: %s --input FILE.root --output FILE.json\n", argv0);
}

} // namespace

int main(int argc, char ** argv) {
    std::string input_path, output_path;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--input" && i + 1 < argc) input_path = argv[++i];
        else if (std::string(argv[i]) == "--output" && i + 1 < argc) output_path = argv[++i];
        else { usage(argv[0]); return 1; }
    }
    if (input_path.empty() || output_path.empty()) { usage(argv[0]); return 1; }
    TFile input(input_path.c_str(), "READ");
    if (input.IsZombie()) throw std::runtime_error("failed to open ROOT input");
    auto * tree = dynamic_cast<TTree *>(input.Get("rwkv_wkv_jvp"));
    if (!tree) throw std::runtime_error("missing rwkv_wkv_jvp TTree");
    int32_t layer = 0;
    std::vector<int32_t> * token_ids = nullptr;
    std::vector<float> * derivatives = nullptr;
    tree->SetBranchAddress("layer", &layer);
    tree->SetBranchAddress("tracked_token_ids", &token_ids);
    tree->SetBranchAddress("tracked_d_logit_per_unit", &derivatives);
    const std::vector<std::string> pieces = {
        " Paris", " France", " Berlin", " Rome", " Tokyo", " Jupiter",
        "Focus", "Forms", "Force", "Forum", "Flush", "Floor", "Float", "Found",
    };
    std::vector<int32_t> saved_ids;
    std::vector<std::vector<float>> values(pieces.size());
    for (Long64_t entry = 0; entry < tree->GetEntries(); ++entry) {
        tree->GetEntry(entry);
        if (!token_ids || !derivatives || token_ids->size() != pieces.size() || derivatives->size() != pieces.size()) {
            throw std::runtime_error("unexpected tracked-token branches");
        }
        if (entry == 0) saved_ids = *token_ids;
        for (size_t i = 0; i < pieces.size(); ++i) values[i].push_back((*derivatives)[i]);
    }
    std::ofstream output(output_path);
    if (!output) throw std::runtime_error("failed to open summary output");
    output << std::setprecision(9) << "{\n  \"input\": \"" << input_path << "\",\n  \"tracked_tokens\": [";
    for (size_t i = 0; i < pieces.size(); ++i) {
        float minimum = std::numeric_limits<float>::infinity();
        float maximum = -std::numeric_limits<float>::infinity();
        int32_t minimum_layer = -1, maximum_layer = -1;
        for (size_t j = 0; j < values[i].size(); ++j) {
            if (values[i][j] < minimum) { minimum = values[i][j]; minimum_layer = (int32_t) j; }
            if (values[i][j] > maximum) { maximum = values[i][j]; maximum_layer = (int32_t) j; }
        }
        if (i) output << ',';
        output << "\n    {\"piece\": \"" << pieces[i] << "\", \"token_id\": " << saved_ids[i]
               << ", \"minimum\": {\"layer\": " << minimum_layer << ", \"value\": " << minimum << "}"
               << ", \"maximum\": {\"layer\": " << maximum_layer << ", \"value\": " << maximum << "}"
               << ", \"per_layer\": [";
        for (size_t j = 0; j < values[i].size(); ++j) {
            if (j) output << ',';
            output << values[i][j];
        }
        output << "]}";
    }
    output << "\n  ]\n}\n";
    if (!output) throw std::runtime_error("failed to write summary output");
    std::printf("wrote=%s layers=%lld\n", output_path.c_str(), tree->GetEntries());
    return 0;
}
