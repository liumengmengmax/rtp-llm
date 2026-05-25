#pragma once

#include "rtp_llm/models_py/bindings/common/Torch_ext.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/fused_qk_norm_rope.h"

namespace torch_ext {

void fusedQKNormRope(at::Tensor& qkv,
                     at::Tensor& q_weight,
                     at::Tensor& k_weight,
                     at::Tensor& position_ids,
                     double      eps,
                     int64_t     num_heads_q,
                     int64_t     num_heads_k,
                     int64_t     num_heads_v,
                     int64_t     head_dim,
                     double      rope_base,
                     bool        is_neox_style,
                     double      rope_scale,
                     int64_t     rotary_dim);

void fusedQKNormRopeWithCache(at::Tensor& qkv,
                              at::Tensor& q_weight,
                              at::Tensor& k_weight,
                              at::Tensor& position_ids,
                              at::Tensor& rope_cache,
                              double      eps,
                              int64_t     num_heads_q,
                              int64_t     num_heads_k,
                              int64_t     num_heads_v,
                              int64_t     head_dim,
                              bool        is_neox_style,
                              int64_t     rotary_dim);

}  // namespace torch_ext
