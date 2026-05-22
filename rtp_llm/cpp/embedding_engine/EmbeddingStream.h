#pragma once
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <sstream>
#include <optional>
#include <queue>
#include <condition_variable>
#include "autil/TimeUtility.h"
#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/embedding_engine/EmbeddingQuery.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/engine_base/stream/ResourceContext.h"
#include "rtp_llm/cpp/config/ModelConfig.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace rtp_llm {

class EmbeddingStream {
public:
    EmbeddingStream(const std::shared_ptr<EmbeddingInput>& query);
    ~EmbeddingStream() {}

public:
    // Exported to python world.
    std::shared_ptr<EmbeddingInput>  embeddingInput() const;
    std::shared_ptr<EmbeddingOutput> embeddingOutput() const;

    const std::optional<MultimodalFeature>& multimodalFeature() const;

    int64_t inputLength() const;

    int64_t streamId() const;

    int64_t batchSize() const;

    void updateTensorOutput(torch::Tensor t);
    void updateMapOutput(std::vector<std::map<std::string, torch::Tensor>>& map);

    void setMetricReporter(const kmonitor::MetricsReporterPtr& metric_reporter);

    void waitFinish();

    void setStart();

    void setError(const std::string& error_info);

    absl::Status initKVCache(const ResourceContext& resource_context, const ModelConfig& model_config);
    void         releaseKVCache(bool insert_to_cache);

    bool hasKVCache() const {
        return kv_cache_enabled_;
    }

    int64_t prefixLength() const {
        return prefix_length_;
    }

    int64_t contextLength() const {
        return inputLength() - prefix_length_;
    }

    int64_t localReuseLength() const {
        return local_reuse_length_;
    }

    BatchKVCacheResourcePtr kvCacheResource() const {
        return batch_kv_cache_resource_;
    }

    CompleteTokenIdsPtr completeTokenIds() const {
        return complete_token_ids_;
    }

    std::string debugString() const {
        std::stringstream debug_string;
        debug_string << "EmbeddingStream {"
                     //  << "generate_input:" << generate_input_->debugString()
                     //  << ", max_seq_len:" << max_seq_len_
                     //  << ", input_length:" << inputLength()
                     //  << ", seq_length:" << seq_length_
                     //  << ", reuse_length:" << reuse_length_
                     //  << ", batch_size:" << batch_size_
                     << "}";
        return debug_string.str();
    }

protected:
    std::shared_ptr<EmbeddingInput>  embedding_input_;
    std::shared_ptr<EmbeddingOutput> embedding_output_;
    int64_t                          begin_time_;
    std::condition_variable          cond_;
    std::mutex                       lock_;
    StreamState                      stream_state_;
    size_t                           begin_time_us_    = 0;
    size_t                           wait_time_us_     = 0;
    kmonitor::MetricsReporterPtr     metrics_reporter_ = nullptr;
    std::optional<torch::Tensor>     context_position_ids_;
    std::shared_ptr<KVCacheManager>   cache_manager_;
    BatchKVCacheResourcePtr           batch_kv_cache_resource_;
    CompleteTokenIdsPtr               complete_token_ids_;
    bool                              kv_cache_enabled_    = false;
    bool                              kv_cache_released_   = true;
    bool                              reuse_cache_         = false;
    bool                              enable_device_cache_ = true;
    int64_t                           prefix_length_       = 0;
    int64_t                           local_reuse_length_  = 0;

    void reportMetrics();
    bool supportKVCache() const;
};

typedef std::shared_ptr<EmbeddingStream> EmbeddingStreamPtr;
}  // namespace rtp_llm
