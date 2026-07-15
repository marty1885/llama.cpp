#pragma once

#include <TFile.h>
#include <TTree.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rwkv_inverse_cache {

inline constexpr const char * k_tree_name = "rwkv_inverse_cache";

inline std::string canonical_model_path(const std::string & path) {
    return std::filesystem::canonical(path).string();
}

class writer {
public:
    writer(const std::string & path, const std::string & model_path, int32_t n_layer, int32_t width) :
            model_path_(canonical_model_path(model_path)), n_layer_(n_layer), width_(width) {
        file_.reset(TFile::Open(path.c_str(), "RECREATE"));
        if (!file_ || file_->IsZombie()) throw std::runtime_error("failed to create inverse cache: " + path);
        tree_ = new TTree(k_tree_name, "RWKV square projection inverses");
        tree_->Branch("model_path", &model_path_);
        tree_->Branch("n_layer", &n_layer_);
        tree_->Branch("width", &width_);
        tree_->Branch("layer", &layer_);
        tree_->Branch("projection", &projection_);
        tree_->Branch("values", &values_);
    }

    void append(int32_t layer, const std::string & projection, std::vector<float> values) {
        if (layer < 0 || layer >= n_layer_ || values.size() != (size_t) width_ * width_) {
            throw std::runtime_error("invalid inverse cache record");
        }
        layer_ = layer;
        projection_ = projection;
        values_ = std::move(values);
        tree_->Fill();
    }

    void close() {
        if (!file_) return;
        file_->cd();
        tree_->Write();
        file_->Write();
        file_->Close();
        tree_ = nullptr;
        file_.reset();
    }

    ~writer() { close(); }

private:
    std::unique_ptr<TFile> file_;
    TTree * tree_ = nullptr;
    std::string model_path_;
    int32_t n_layer_ = 0;
    int32_t width_ = 0;
    int32_t layer_ = 0;
    std::string projection_;
    std::vector<float> values_;
};

class reader {
public:
    reader(const std::string & path, const std::string & model_path, int32_t n_layer, int32_t width) {
        file_.reset(TFile::Open(path.c_str(), "READ"));
        if (!file_ || file_->IsZombie()) throw std::runtime_error("failed to open inverse cache: " + path);
        tree_ = dynamic_cast<TTree *>(file_->Get(k_tree_name));
        if (!tree_ || tree_->GetEntries() == 0) throw std::runtime_error("inverse cache has no records: " + path);
        tree_->SetBranchAddress("model_path", &model_path_);
        tree_->SetBranchAddress("n_layer", &n_layer_);
        tree_->SetBranchAddress("width", &width_);
        tree_->SetBranchAddress("layer", &layer_);
        tree_->SetBranchAddress("projection", &projection_);
        tree_->SetBranchAddress("values", &values_);
        for (Long64_t entry = 0; entry < tree_->GetEntries(); ++entry) {
            tree_->GetEntry(entry);
            if (!model_path_ || *model_path_ != canonical_model_path(model_path) || n_layer_ != n_layer || width_ != width ||
                layer_ < 0 || layer_ >= n_layer || !values_ || values_->size() != (size_t) width * width) {
                throw std::runtime_error("inverse cache does not match this model");
            }
            const auto inserted = entries_.emplace(key(layer_, *projection_), entry);
            if (!inserted.second) throw std::runtime_error("duplicate inverse cache record");
        }
    }

    std::vector<float> load(int32_t layer, const std::string & projection) {
        const auto entry = entries_.find(key(layer, projection));
        if (entry == entries_.end()) throw std::runtime_error("missing inverse cache record: " + projection);
        tree_->GetEntry(entry->second);
        return *values_;
    }

private:
    static std::string key(int32_t layer, const std::string & projection) {
        return std::to_string(layer) + ':' + projection;
    }

    std::unique_ptr<TFile> file_;
    TTree * tree_ = nullptr;
    std::unordered_map<std::string, Long64_t> entries_;
    std::string * model_path_ = nullptr;
    int32_t n_layer_ = 0;
    int32_t width_ = 0;
    int32_t layer_ = 0;
    std::string * projection_ = nullptr;
    std::vector<float> * values_ = nullptr;
};

} // namespace rwkv_inverse_cache
