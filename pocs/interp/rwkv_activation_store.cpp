#include "rwkv_activation_store.h"

#include "ggml.h"

#include <cstdio>
#include <memory>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleReader.hxx>
#include <ROOT/RNTupleWriter.hxx>
#include <stdexcept>
#include <utility>

namespace {

using ROOT::RNTupleModel;
using ROOT::RNTupleReader;
using ROOT::RNTupleWriter;

constexpr const char * k_dataset_name = "rwkv_activations";

std::string source_field(size_t source_index) {
    char name[32];
    std::snprintf(name, sizeof(name), "activation_%03zu", source_index);
    return name;
}

}  // namespace

namespace rwkv_activation_store {

struct activation_dataset_writer::impl {
    std::unique_ptr<RNTupleWriter>                   writer;
    std::shared_ptr<uint64_t>                        sample_id;
    std::shared_ptr<uint64_t>                        corpus_line;
    std::shared_ptr<int32_t>                         token_position;
    std::shared_ptr<int32_t>                         token_id;
    std::shared_ptr<int32_t>                         prompt_tokens;
    std::shared_ptr<std::vector<std::string>>        source_names;
    std::vector<std::shared_ptr<std::vector<float>>> activations;
    size_t                                           dimension;
};

activation_dataset_writer::activation_dataset_writer(std::unique_ptr<impl> impl) : impl_(std::move(impl)) {}

activation_dataset_writer::activation_dataset_writer(activation_dataset_writer &&) noexcept             = default;
activation_dataset_writer & activation_dataset_writer::operator=(activation_dataset_writer &&) noexcept = default;
activation_dataset_writer::~activation_dataset_writer()                                                 = default;

activation_dataset_writer activation_dataset_writer::create(const std::string &      path,
                                                            std::vector<std::string> sources,
                                                            size_t                   dimension) {
    if (sources.empty() || dimension == 0) {
        throw std::runtime_error("activation dataset needs sources and a nonzero dimension");
    }
    auto model             = RNTupleModel::Create();
    auto result            = std::make_unique<impl>();
    result->sample_id      = model->MakeField<uint64_t>("sample_id");
    result->corpus_line    = model->MakeField<uint64_t>("corpus_line");
    result->token_position = model->MakeField<int32_t>("token_position");
    result->token_id       = model->MakeField<int32_t>("token_id");
    result->prompt_tokens  = model->MakeField<int32_t>("prompt_tokens");
    result->source_names   = model->MakeField<std::vector<std::string>>("source_names");
    *result->source_names  = sources;
    result->activations.reserve(sources.size());
    for (size_t source = 0; source < sources.size(); ++source) {
        result->activations.push_back(model->MakeField<std::vector<float>>(source_field(source)));
    }
    result->dimension = dimension;
    result->writer    = RNTupleWriter::Recreate(std::move(model), k_dataset_name, path);
    return activation_dataset_writer(std::move(result));
}

void activation_dataset_writer::append(const sample_metadata &                     metadata,
                                       const std::vector<std::span<const float>> & activations) {
    if (!impl_ || activations.size() != impl_->activations.size()) {
        throw std::runtime_error("activation source count differs from dataset schema");
    }
    *impl_->sample_id      = metadata.sample_id;
    *impl_->corpus_line    = metadata.corpus_line;
    *impl_->token_position = metadata.token_position;
    *impl_->token_id       = metadata.token_id;
    *impl_->prompt_tokens  = metadata.prompt_tokens;
    for (size_t source = 0; source < activations.size(); ++source) {
        const auto activation = activations[source];
        if (activation.size() != impl_->dimension) {
            throw std::runtime_error("activation dimension differs from dataset schema");
        }
        auto & encoded = *impl_->activations[source];
        encoded.assign(activation.begin(), activation.end());
    }
    impl_->writer->Fill();
}

void activation_dataset_writer::commit() {
    if (impl_ && impl_->writer) {
        impl_->writer->CommitCluster();
    }
}

struct activation_dataset_reader::impl {
    std::unique_ptr<RNTupleReader> reader;
    std::vector<std::string>       sources;
    std::vector<size_t>            source_indices;
    size_t                         dimension = 0;
};

activation_dataset_reader::activation_dataset_reader(std::unique_ptr<impl> impl) : impl_(std::move(impl)) {}

activation_dataset_reader::activation_dataset_reader(activation_dataset_reader &&) noexcept             = default;
activation_dataset_reader & activation_dataset_reader::operator=(activation_dataset_reader &&) noexcept = default;
activation_dataset_reader::~activation_dataset_reader()                                                 = default;

activation_dataset_reader activation_dataset_reader::open(const std::string & path) {
    auto result    = std::make_unique<impl>();
    result->reader = RNTupleReader::Open(k_dataset_name, path);
    if (result->reader->GetNEntries() == 0) {
        throw std::runtime_error("activation dataset is empty");
    }
    auto stored_names = result->reader->GetView<std::vector<std::string>>("source_names");
    result->sources   = stored_names(0);
    if (result->sources.empty()) {
        throw std::runtime_error("activation dataset has no source names");
    }
    result->source_indices.resize(result->sources.size());
    for (size_t index = 0; index < result->source_indices.size(); ++index) {
        result->source_indices[index] = index;
    }
    auto first        = result->reader->GetView<std::vector<float>>(source_field(0));
    result->dimension = first(0).size();
    if (result->dimension == 0) {
        throw std::runtime_error("activation dataset has zero-dimensional rows");
    }
    return activation_dataset_reader(std::move(result));
}

uint64_t activation_dataset_reader::entries() const {
    return impl_->reader->GetNEntries();
}

size_t activation_dataset_reader::dimension() const {
    return impl_->dimension;
}

const std::vector<std::string> & activation_dataset_reader::sources() const {
    return impl_->sources;
}

void activation_dataset_reader::for_each_source_batch(size_t                                       source_index,
                                                      size_t                                       max_rows,
                                                      const std::function<void(source_batch &&)> & callback) const {
    if (!impl_ || source_index >= impl_->sources.size() || max_rows == 0) {
        throw std::runtime_error("invalid source batch request");
    }
    auto sample_id      = impl_->reader->GetView<uint64_t>("sample_id");
    auto corpus_line    = impl_->reader->GetView<uint64_t>("corpus_line");
    auto token_position = impl_->reader->GetView<int32_t>("token_position");
    auto token_id       = impl_->reader->GetView<int32_t>("token_id");
    auto prompt_tokens  = impl_->reader->GetView<int32_t>("prompt_tokens");
    auto activation     = impl_->reader->GetView<std::vector<float>>(source_field(impl_->source_indices[source_index]));

    source_batch batch;
    batch.dimension = impl_->dimension;
    batch.metadata.reserve(max_rows);
    batch.values.reserve(max_rows * impl_->dimension);
    for (auto entry : impl_->reader->GetEntryRange()) {
        const auto & encoded = activation(entry);
        if (encoded.size() != impl_->dimension) {
            throw std::runtime_error("activation row does not match dataset dimension");
        }
        batch.metadata.push_back(
            { sample_id(entry), corpus_line(entry), token_position(entry), token_id(entry), prompt_tokens(entry) });
        batch.values.insert(batch.values.end(), encoded.begin(), encoded.end());
        if (batch.metadata.size() == max_rows) {
            callback(std::move(batch));
            batch           = {};
            batch.dimension = impl_->dimension;
            batch.metadata.reserve(max_rows);
            batch.values.reserve(max_rows * impl_->dimension);
        }
    }
    if (!batch.metadata.empty()) {
        callback(std::move(batch));
    }
}

}  // namespace rwkv_activation_store
