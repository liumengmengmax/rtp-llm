#pragma once

#if USING_CUDA
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace rtp_llm {

void invokeFusedQKNormRope(__nv_bfloat16*       qkv,
                           const __nv_bfloat16* q_weight,
                           const __nv_bfloat16* k_weight,
                           const int*           position_ids,
                           float                eps,
                           int                  num_tokens,
                           int                  num_heads_q,
                           int                  num_heads_k,
                           int                  num_heads_v,
                           int                  head_dim,
                           float                rope_base,
                           bool                 is_neox_style,
                           float                rope_scale,
                           int                  rotary_dim,
                           cudaStream_t         stream);

void invokeFusedQKNormRopeWithCache(__nv_bfloat16*       qkv,
                                    const __nv_bfloat16* q_weight,
                                    const __nv_bfloat16* k_weight,
                                    const int*           position_ids,
                                    const float2*        rope_cache,
                                    float                eps,
                                    int                  num_tokens,
                                    int                  num_heads_q,
                                    int                  num_heads_k,
                                    int                  num_heads_v,
                                    int                  head_dim,
                                    bool                 is_neox_style,
                                    int                  rotary_dim,
                                    cudaStream_t         stream);

}  // namespace rtp_llm
#endif
