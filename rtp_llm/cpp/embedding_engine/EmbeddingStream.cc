#include "rtp_llm/cpp/embedding_engine/EmbeddingStream.h"
#include "autil/TimeUtility.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateConfig.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include <memory>

using namespace std;

namespace rtp_llm {

EmbeddingStream::EmbeddingStream(const shared_ptr<rtp_llm::EmbeddingInput>& query): embedding_input_(query) {
    if (!query.get()) {
        return;
    }
    begin_time_       = autil::TimeUtility::currentTimeInMilliSeconds();
    embedding_output_ = make_shared<EmbeddingOutput>();
    stream_state_     = StreamState::WAITING;
    begin_time_us_    = autil::TimeUtility::currentTimeInMicroSeconds();
}

bool EmbeddingStream::supportKVCache() const {
    if (!embedding_input_) {
        return false;
    }
    if (batchSize() != 1) {
        return false;
    }
    if (embedding_input_->multimodal_features.has_value() || embedding_input_->input_embeddings.has_value()) {
        return false;
    }
    return inputLength() > 0;
}

int64_t EmbeddingStream::streamId() const {
    return embedding_input_->request_id;
}

const std::optional<MultimodalFeature>& EmbeddingStream::multimodalFeature() const {
    return embedding_input_->multimodal_features;
}

int64_t EmbeddingStream::batchSize() const {
    return embedding_input_->input_lengths.size(0);
}

void EmbeddingStream::setMetricReporter(const kmonitor::MetricsReporterPtr& metric_reporter) {
    metrics_reporter_ = metric_reporter;
}

std::shared_ptr<EmbeddingInput> EmbeddingStream::embeddingInput() const {
    return embedding_input_;
}

std::shared_ptr<EmbeddingOutput> EmbeddingStream::embeddingOutput() const {
    return embedding_output_;
}

int64_t EmbeddingStream::inputLength() const {
    return embedding_input_->total_length;
}

absl::Status EmbeddingStream::initKVCache(const ResourceContext& resource_context, const ModelConfig& model_config) {
    RTP_LLM_PROFILE_FUNCTION();
    if (kv_cache_enabled_ || !resource_context.cache_manager || !resource_context.reuse_cache) {
        return absl::OkStatus();
    }
    if (!supportKVCache()) {
        return absl::OkStatus();
    }

    cache_manager_           = resource_context.cache_manager;
    reuse_cache_             = resource_context.reuse_cache;
    enable_device_cache_     = resource_context.enable_device_cache;
    batch_kv_cache_resource_ = std::make_shared<BatchKVCacheResource>();

    const auto& cache_config = cache_manager_->cacheConfig();
    size_t      kernel_blocks_per_kv_block = cache_config.kernelBlocksPerKvBlock();
    batch_kv_cache_resource_->resetBatchSize(1);
    batch_kv_cache_resource_->initGroups(cache_config.groupNums(),
                                         static_cast<int>(cache_config.layer_all_num),
                                         cache_config.layer_to_group_id,
                                         kernel_blocks_per_kv_block,
                                         cache_config.group_types);

    auto generate_input             = std::make_shared<GenerateInput>();
    generate_input->request_id      = streamId();
    generate_input->generate_config = std::make_shared<GenerateConfig>();
    generate_input->input_ids       = embedding_input_->token_ids;
    complete_token_ids_ = std::make_shared<CompleteTokenIds>(1,
                                                             1,
                                                             static_cast<int>(model_config.max_seq_len),
                                                             static_cast<int>(cache_config.seq_size_per_block));
    complete_token_ids_->init(generate_input);

    MallocInfo malloc_info;
    malloc_info.batch_kv_cache_resource      = batch_kv_cache_resource_;
    malloc_info.complete_token_ids           = complete_token_ids_;
    malloc_info.request_id                   = streamId();
    malloc_info.reuse_cache                  = reuse_cache_;
    malloc_info.enable_device_cache          = reuse_cache_ && enable_device_cache_;
    malloc_info.enable_remove_skipped_blocks = false;

    const auto result = cache_manager_->malloc(malloc_info);
    if (!result.success) {
        batch_kv_cache_resource_->clearBlocks();
        complete_token_ids_.reset();
        return absl::InternalError("embedding kv cache malloc failed");
    }

    prefix_length_      = result.reuse_len;
    local_reuse_length_ = result.reuse_len;
    kv_cache_enabled_   = true;
    kv_cache_released_  = false;
    RTP_LLM_LOG_DEBUG("embedding stream [%ld] init kv cache, prefix=%ld, blocks=%d",
                      streamId(),
                      prefix_length_,
                      batch_kv_cache_resource_->curBlocksNum());
    return absl::OkStatus();
}

void EmbeddingStream::releaseKVCache(bool insert_to_cache) {
    RTP_LLM_PROFILE_FUNCTION();
    if (!kv_cache_enabled_ || kv_cache_released_ || !cache_manager_ || !batch_kv_cache_resource_
        || !complete_token_ids_) {
        return;
    }

    if (batch_kv_cache_resource_->curBlocksNum() > 0) {
        if (insert_to_cache && reuse_cache_ && enable_device_cache_) {
            InsertInfo insert_info{batch_kv_cache_resource_, complete_token_ids_, false};
            cache_manager_->insertIntoCache(insert_info);
        }
        FreeInfo free_info{batch_kv_cache_resource_, complete_token_ids_};
        free_info.request_id = streamId();
        cache_manager_->free(free_info);
    }
    batch_kv_cache_resource_->clearBlocks();
    kv_cache_released_  = true;
    kv_cache_enabled_   = false;
    prefix_length_      = 0;
    if (!insert_to_cache) {
        local_reuse_length_ = 0;
    }
    complete_token_ids_.reset();
}

void EmbeddingStream::waitFinish() {
    unique_lock<mutex> lock(lock_);
    while (stream_state_ != StreamState::FINISHED) {
        cond_.wait_for(lock, std::chrono::milliseconds(5));
    }
    if (!embedding_output_->error_info.ok()) {
        throw std::runtime_error("run stream failed: " + embedding_output_->error_info.ToString());
    }
}

void EmbeddingStream::reportMetrics() {
    if (metrics_reporter_) {
        RtpEmbeddingStreamMetricsCollector collector;
        collector.input_token_length = inputLength();
        collector.wait_latency_us    = wait_time_us_;
        collector.total_latency_us   = autil::TimeUtility::currentTimeInMicroSeconds() - begin_time_us_;
        collector.reuse_length       = localReuseLength();
        metrics_reporter_->report<RtpEmbeddingStreamMetrics, RtpEmbeddingStreamMetricsCollector>(nullptr, &collector);
    }
}

void EmbeddingStream::setError(const std::string& error_info) {
    releaseKVCache(false);
    lock_guard<mutex> lock(lock_);
    embedding_output_->setError(ErrorCode::UNKNOWN_ERROR, error_info);
    stream_state_ = StreamState::FINISHED;
    reportMetrics();
    cond_.notify_all();
}

void EmbeddingStream::setStart() {
    wait_time_us_ = autil::TimeUtility::currentTimeInMicroSeconds() - begin_time_us_;
    stream_state_ = StreamState::RUNNING;
}

void EmbeddingStream::updateTensorOutput(torch::Tensor t) {
    releaseKVCache(true);
    lock_guard<mutex> lock(lock_);
    embedding_output_->setTensorOutput(t);
    stream_state_ = StreamState::FINISHED;
    reportMetrics();
    cond_.notify_all();
}

void EmbeddingStream::updateMapOutput(std::vector<std::map<std::string, torch::Tensor>>& map) {
    releaseKVCache(true);
    lock_guard<mutex> lock(lock_);
    embedding_output_->setMapOutput(map);
    stream_state_ = StreamState::FINISHED;
    reportMetrics();
    cond_.notify_all();
}

}  // namespace rtp_llm
