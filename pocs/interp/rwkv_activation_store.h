#pragma once

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace rwkv_activation_store {

struct sample_metadata {
    uint64_t sample_id;
    uint64_t corpus_line;
    int32_t  token_position;
    int32_t  token_id;
    int32_t  prompt_tokens;
};

struct source_batch {
    std::vector<sample_metadata> metadata;
    std::vector<float>           values;
    size_t                       dimension;
};

class activation_dataset_writer {
  public:
    static activation_dataset_writer create(const std::string &      path,
                                            std::vector<std::string> sources,
                                            size_t                   dimension);

    activation_dataset_writer(activation_dataset_writer &&) noexcept;
    activation_dataset_writer & operator=(activation_dataset_writer &&) noexcept;
    ~activation_dataset_writer();

    activation_dataset_writer(const activation_dataset_writer &)             = delete;
    activation_dataset_writer & operator=(const activation_dataset_writer &) = delete;

    void append(const sample_metadata & metadata, const std::vector<std::span<const float>> & activations);
    void commit();

  private:
    struct impl;
    explicit activation_dataset_writer(std::unique_ptr<impl> impl);
    std::unique_ptr<impl> impl_;
};

class activation_dataset_reader {
  public:
    static activation_dataset_reader open(const std::string & path);

    activation_dataset_reader(activation_dataset_reader &&) noexcept;
    activation_dataset_reader & operator=(activation_dataset_reader &&) noexcept;
    ~activation_dataset_reader();

    activation_dataset_reader(const activation_dataset_reader &)             = delete;
    activation_dataset_reader & operator=(const activation_dataset_reader &) = delete;

    uint64_t                         entries() const;
    size_t                           dimension() const;
    const std::vector<std::string> & sources() const;
    void                             for_each_source_batch(size_t                                       source_index,
                                                           size_t                                       max_rows,
                                                           const std::function<void(source_batch &&)> & callback) const;

  private:
    struct impl;
    explicit activation_dataset_reader(std::unique_ptr<impl> impl);
    std::unique_ptr<impl> impl_;
};

}  // namespace rwkv_activation_store
