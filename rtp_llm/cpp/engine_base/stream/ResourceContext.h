#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "rtp_llm/cpp/engine_base/system_prompt/SystemPrompt.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/config/RoleTypes.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

namespace rtp_llm {

class GenerateStream;
class BatchKVCacheResource;
class CompleteTokenIds;

struct EmbeddingPrefixCache {
    bool                                  enabled = false;
    bool                                  ready   = false;
    std::vector<int32_t>                  tokens;
    std::shared_ptr<BatchKVCacheResource> kv_cache_resource;
    std::shared_ptr<CompleteTokenIds>     complete_token_ids;
};

struct ResourceContext {
    std::shared_ptr<KVCacheManager>       cache_manager;
    std::shared_ptr<SystemPrompt>         system_prompt;
    std::shared_ptr<EmbeddingPrefixCache> embedding_prefix_cache;

    RoleType role_type{RoleType::PDFUSION};
    bool     decode_entrance{false};  // PD反转模式：Decode侧作为请求入口

    bool    reuse_cache{false};
    bool    enable_memory_cache{false};
    bool    enable_remote_cache{false};
    bool    enable_device_cache{true};
    bool    write_cache_sync{false};
    bool    enable_tiered_memory_cache{false};
    int64_t device_cache_min_free_blocks{0};
    int     load_cache_retry_times{1};

    void initCacheConfig(const KVCacheConfig&       kv_cache_config,
                         const FIFOSchedulerConfig& scheduler_config,
                         int64_t                    max_seq_len);
};

}  // namespace rtp_llm
