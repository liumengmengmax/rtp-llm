#include "rtp_llm/cpp/embedding_engine/EmbeddingEngine.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/NoBlockCopy.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "autil/EnvUtil.h"
#include <c10/core/InferenceMode.h>
#include <algorithm>
#include <cctype>
#include <exception>
#include <limits>
#include <list>
#include <sstream>
#include <string>
#include <vector>

using namespace std;
namespace rtp_llm {

EmbeddingEngine::EmbeddingEngine(const EngineInitParams& params, py::object handler):
    model_config_(params.model_config_),
    parallelism_config(params.parallelism_config),
    concurrency_config(params.concurrency_config),
    profiling_debug_logging_config_(params.profiling_debug_logging_config),
    metrics_reporter_(params.metrics_reporter),
    step_profiler_(params.profiling_debug_logging_config.torch_cuda_profiler_dir,
                   params.parallelism_config.dp_rank * params.parallelism_config.tp_size
                       + params.parallelism_config.tp_rank) {
    {
        size_t device_id = params.parallelism_config.world_rank % params.parallelism_config.local_world_size;
        rtp_llm::initRuntime(device_id,
                             params.profiling_debug_logging_config.trace_memory,
                             params.device_resource_config.enable_comm_overlap,
                             params.model_config_.mla_ops_type);
    }
    warmupNoBlockCopy();
    resource_context_.initCacheConfig(
        params.kv_cache_config, params.runtime_config.fifo_scheduler_config, params.model_config_.max_seq_len);
    initEmbeddingPrefixCacheConfig();
    initCacheManager(params);
    executor_.reset(
        new EmbeddingExecutor(params, handler, resource_context_, kv_cache_group_num_, kv_cache_layer_to_group_));
    THROW_IF_STATUS_ERROR(buildEmbeddingPrefixCache());
    scheduler_.reset(
        new EmbeddingScheduler(model_config_, concurrency_config, params.runtime_config, metrics_reporter_));

    (void)startLoop();
}

namespace {

std::vector<int32_t> parseEmbeddingPrefixTokens(const std::string& raw_tokens) {
    std::string normalized;
    normalized.reserve(raw_tokens.size());
    for (char ch : raw_tokens) {
        normalized.push_back((ch == ',' || ch == '[' || ch == ']') ? ' ' : ch);
    }

    std::vector<int32_t> tokens;
    std::stringstream    ss(normalized);
    long long            value = 0;
    while (ss >> value) {
        RTP_LLM_CHECK_WITH_INFO(value >= 0 && value <= std::numeric_limits<int32_t>::max(),
                                "invalid embedding prefix token id: %lld",
                                value);
        tokens.push_back(static_cast<int32_t>(value));
    }
    return tokens;
}

}  // namespace

void EmbeddingEngine::initEmbeddingPrefixCacheConfig() {
    const bool enable_prefix_cache = autil::EnvUtil::getEnv("IDLE_FISH_ENABLE_EMBEDDING_PREFIX_CACHE", false);
    if (!enable_prefix_cache) {
        return;
    }
    const bool supports_embedding_cache = model_config_.model_type == "qwen_3_idle_fish_embedding"
                                          && model_config_.task_type == TaskType::DENSE_EMBEDDING
                                          && model_config_.attn_config.is_causal;
    if (!supports_embedding_cache) {
        RTP_LLM_LOG_WARNING("embedding resident prefix cache ignored for model_type=%s task_type=%d is_causal=%d",
                            model_config_.model_type.c_str(),
                            static_cast<int>(model_config_.task_type),
                            static_cast<int>(model_config_.attn_config.is_causal));
        return;
    }

    const std::string raw_tokens = autil::EnvUtil::getEnv("IDLE_FISH_EMBEDDING_PREFIX_CACHE_TOKENS", std::string(""));
    auto              tokens     = parseEmbeddingPrefixTokens(raw_tokens);
    RTP_LLM_CHECK_WITH_INFO(!tokens.empty(),
                            "IDLE_FISH_ENABLE_EMBEDDING_PREFIX_CACHE=1 requires "
                            "IDLE_FISH_EMBEDDING_PREFIX_CACHE_TOKENS");

    resource_context_.embedding_prefix_cache          = std::make_shared<EmbeddingPrefixCache>();
    resource_context_.embedding_prefix_cache->enabled = true;
    resource_context_.embedding_prefix_cache->tokens  = std::move(tokens);
    resource_context_.reuse_cache                     = true;
    RTP_LLM_LOG_INFO("embedding resident prefix cache configured with %zu tokens",
                     resource_context_.embedding_prefix_cache->tokens.size());
}

absl::Status EmbeddingEngine::buildEmbeddingPrefixCache() {
    auto& prefix_cache = resource_context_.embedding_prefix_cache;
    if (!prefix_cache || !prefix_cache->enabled || prefix_cache->ready) {
        return absl::OkStatus();
    }
    if (!resource_context_.cache_manager) {
        return absl::InternalError("embedding resident prefix cache requires cache manager");
    }

    const int64_t prefix_len = static_cast<int64_t>(prefix_cache->tokens.size());
    auto          token_ids =
        torch::from_blob(prefix_cache->tokens.data(), {prefix_len}, torch::TensorOptions(torch::kInt32)).clone();
    auto token_type_ids = torch::zeros({prefix_len}, torch::TensorOptions(torch::kInt32));
    auto input_lengths  = torch::tensor({static_cast<int32_t>(prefix_len)}, torch::TensorOptions(torch::kInt32));
    auto input          = std::make_shared<EmbeddingInput>(token_ids, token_type_ids, input_lengths, /*request_id=*/-1);
    auto stream         = std::make_shared<EmbeddingStream>(input);
    stream->setMetricReporter(metrics_reporter_);
    stream->setKeepKVCacheOnFinish(true);

    std::list<EmbeddingStreamPtr> streams = {stream};
    RETURN_IF_STATUS_ERROR(executor_->process(streams));
    stream->waitFinish();
    if (!stream->hasKVCache() || !stream->kvCacheResource() || !stream->completeTokenIds()) {
        return absl::InternalError("embedding resident prefix cache build did not keep kv cache");
    }

    prefix_cache->kv_cache_resource  = stream->kvCacheResource();
    prefix_cache->complete_token_ids = stream->completeTokenIds();
    prefix_cache->ready              = true;
    RTP_LLM_LOG_INFO("embedding resident prefix cache built, prefix_tokens=%ld blocks=%d",
                     prefix_len,
                     prefix_cache->kv_cache_resource->curBlocksNum());
    return absl::OkStatus();
}

void EmbeddingEngine::releaseEmbeddingPrefixCache() {
    auto& prefix_cache = resource_context_.embedding_prefix_cache;
    if (!prefix_cache || !prefix_cache->ready || !prefix_cache->kv_cache_resource || !prefix_cache->complete_token_ids
        || !resource_context_.cache_manager) {
        return;
    }
    if (prefix_cache->kv_cache_resource->curBlocksNum() > 0) {
        FreeInfo free_info{prefix_cache->kv_cache_resource, prefix_cache->complete_token_ids};
        free_info.request_id = -1;
        resource_context_.cache_manager->free(free_info);
    }
    prefix_cache->kv_cache_resource->clearBlocks();
    prefix_cache->kv_cache_resource.reset();
    prefix_cache->complete_token_ids.reset();
    prefix_cache->ready = false;
}

void EmbeddingEngine::initCacheManager(const EngineInitParams& params) {
    if (!resource_context_.reuse_cache) {
        RTP_LLM_LOG_INFO("embedding kv cache reuse disabled");
        return;
    }
    const bool supports_embedding_cache = model_config_.model_type == "qwen_3_idle_fish_embedding"
                                          && model_config_.task_type == TaskType::DENSE_EMBEDDING
                                          && model_config_.attn_config.is_causal;
    if (!supports_embedding_cache) {
        resource_context_.reuse_cache = false;
        RTP_LLM_LOG_INFO("embedding kv cache reuse disabled for model_type=%s task_type=%d is_causal=%d",
                         model_config_.model_type.c_str(),
                         static_cast<int>(model_config_.task_type),
                         static_cast<int>(model_config_.attn_config.is_causal));
        return;
    }

    auto cache_config = CacheConfigCreator::createConfig(
        model_config_, parallelism_config, params.runtime_config, params.kv_cache_config, std::nullopt);
    RTP_LLM_LOG_INFO("create embedding cache manager with config %s", cache_config.debugString().c_str());
    resource_context_.cache_manager = make_shared<KVCacheManager>(cache_config,
                                                                  false,
                                                                  metrics_reporter_,
                                                                  params.kv_cache_config,
                                                                  parallelism_config,
                                                                  params.runtime_config,
                                                                  params.sp_config,
                                                                  params.pd_sep_config,
                                                                  params.cache_store_config);
    resource_context_.role_type     = params.pd_sep_config.role_type;
    if (!resource_context_.cache_manager->init()) {
        RTP_LLM_FAIL("init embedding kv cache manager failed");
    }
    const auto& cache_cfg    = resource_context_.cache_manager->cacheConfig();
    kv_cache_group_num_      = cache_cfg.groupNums();
    kv_cache_layer_to_group_ = cache_cfg.layer_to_group_id;
}

EmbeddingEngine::~EmbeddingEngine() {
    RTP_LLM_LOG_INFO("destory embedding engine");
    (void)stop();
    releaseEmbeddingPrefixCache();
}

absl::Status EmbeddingEngine::startLoop() {
    RTP_LLM_LOG_INFO("start embedding engine");
    running_     = true;
    loop_thread_ = std::thread(&EmbeddingEngine::loop, this);
    return absl::OkStatus();
}

absl::Status EmbeddingEngine::stop() {
    RTP_LLM_LOG_INFO("stop embedding engine");
    running_ = false;
    RETURN_IF_STATUS_ERROR(scheduler_->stop());
    if (loop_thread_.joinable()) {
        loop_thread_.join();
    }
    return absl::OkStatus();
}

void EmbeddingEngine::loop() {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_LOG_INFO("loop begin");
    c10::InferenceMode inference_guard(true);
    while (running_) {
        auto status = step();
        if (!status.ok()) {
            RTP_LLM_LOG_ERROR("step running error: %s", status.ToString().c_str());
            THROW_IF_STATUS_ERROR(trySaveStepError());
        }
    }
}

std::shared_ptr<EmbeddingOutput> EmbeddingEngine::decode(th::Tensor                       token_ids,
                                                         th::Tensor                       token_type_ids,
                                                         th::Tensor                       input_lengths,
                                                         int64_t                          request_id,
                                                         std::optional<MultimodalFeature> multimodal_features,
                                                         std::optional<th::Tensor>        input_embeddings) {
    auto input = std::make_shared<EmbeddingInput>(
        token_ids, token_type_ids, input_lengths, request_id, multimodal_features, input_embeddings);
    return decode(input);
}

std::shared_ptr<EmbeddingOutput> EmbeddingEngine::decode(std::shared_ptr<EmbeddingInput> input) {
    auto embedding_stream = std::make_shared<EmbeddingStream>(input);
    embedding_stream->setMetricReporter(metrics_reporter_);
    THROW_IF_STATUS_ERROR(enqueue(embedding_stream));
    embedding_stream->waitFinish();
    return embedding_stream->embeddingOutput();
}

absl::Status EmbeddingEngine::trySaveStepError() const {
    return absl::UnimplementedError("can not save yet!");
}

absl::Status EmbeddingEngine::enqueue(EmbeddingStreamPtr streams) {
    return scheduler_->enqueue(streams);
}

absl::Status EmbeddingEngine::step() {
    cudaSyncAndCheck();
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    CHECK_AND_RETURN_REF(streams, scheduler_->scheduleNew());
    if (streams.empty()) {
        RTP_LLM_LOG_INFO("no query run and sleep");
        return absl::OkStatus();
    }
    // If gen_timeline_sync is enabled and no profiling session is currently active,
    // configure + tick BEFORE process() to start the profiler before the actual work runs.
    // After process(), tick() again to count this step. When the step count reaches num_steps,
    // the profiler auto-stops and saves the JSON file. The next step() will start a new window.
    //
    // Window parameters mirror the NormalEngine (LLM) control surface, but since
    // EmbeddingEngine has no per-stream `generateConfig` to source them from, we expose them
    // as environment variables (sharing the GEN_TIMELINE_* prefix with the existing SYNC flag):
    //
    //   GEN_TIMELINE_SYNC        -> ProfilingDebugLoggingConfig.gen_timeline_sync (master switch)
    //   GEN_TIMELINE_TRACE_NAME  -> trace prefix for the JSON filename (default: embedding_timeline)
    //   GEN_TIMELINE_START_STEP  -> warm-up steps to skip before starting the profiler  (default: 0)
    //   GEN_TIMELINE_NUM_STEPS   -> how many steps to capture before stop+flush          (default: 1)
    //
    // For embedding models each step() runs one complete forward pass (unlike NormalEngine
    // where a single request spans many decode steps). One step is usually sufficient to
    // capture a representative trace, so num_steps defaults to 1 to ensure the JSON is
    // flushed even if the server is torn down right after a single query (e.g. on a
    // smoke-test failure path).
    if (profiling_debug_logging_config_.gen_timeline_sync && !step_profiler_.enabled()) {
        // Read profiling window parameters via the project-standard EnvUtil helper
        // (same pattern as PERF_TEST in GenerateStream.cc, BIZ_NAME / CHECKPOINT_PATH in
        // RemoteConnector.cc, FT_SERVER_TEST in Logger.cc, etc.). The template overload
        // dispatches on the default-value type: std::string / int / bool.
        const std::string trace_name =
            autil::EnvUtil::getEnv("GEN_TIMELINE_TRACE_NAME", std::string("embedding_timeline"));
        const int start_step = std::max(0, autil::EnvUtil::getEnv("GEN_TIMELINE_START_STEP", 0));
        const int num_steps  = std::max(1, autil::EnvUtil::getEnv("GEN_TIMELINE_NUM_STEPS", 1));
        RTP_LLM_LOG_INFO("EmbeddingEngine timeline profiling configured: trace=%s start_step=%d num_steps=%d",
                         trace_name.c_str(),
                         start_step,
                         num_steps);
        step_profiler_.configure(true, trace_name, start_step, num_steps);
        step_profiler_.tick();  // tick once now so start_step counting begins immediately
    }
    try {
        auto status = executor_->process(streams);
        if (!status.ok()) {
            for (auto& stream : streams) {
                stream->setError(status.ToString());
                RTP_LLM_LOG_WARNING(
                    "error_stream_info: length: %d, exception: %s", stream->inputLength(), status.ToString().c_str());
            }
        }
    } catch (const exception& e) {
        std::string error_msg = e.what();
        RTP_LLM_LOG_WARNING("run engine failed, stream size: %d, error: %s", streams.size(), error_msg.c_str());
        for (auto& stream : streams) {
            stream->setError(error_msg);
            RTP_LLM_LOG_WARNING("error_stream_info: length: %d", stream->inputLength());
        }
        if (error_msg.find("CUDA Driver error") != string::npos || error_msg.find("CUDA error") != string::npos) {
            RTP_LLM_LOG_ERROR("detect CUDA error, do abort");
            abort();
        }
    }
    // tick profiler after process() to count this step (and stop when num_steps reached).
    step_profiler_.tick();
    cudaSyncAndCheck();
    return absl::OkStatus();
}

}  // namespace rtp_llm
