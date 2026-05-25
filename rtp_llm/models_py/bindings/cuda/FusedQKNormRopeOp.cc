#include "rtp_llm/models_py/bindings/cuda/FusedQKNormRopeOp.h"

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
                     int64_t     rotary_dim) {
    CHECK_INPUT(qkv);
    CHECK_INPUT(q_weight);
    CHECK_INPUT(k_weight);
    CHECK_INPUT(position_ids);
    CHECK_DIM(2, qkv);
    CHECK_DIM(1, q_weight);
    CHECK_DIM(1, k_weight);
    CHECK_DIM(1, position_ids);

    CHECK_EQ(qkv.scalar_type(), at::ScalarType::BFloat16);
    CHECK_EQ(q_weight.scalar_type(), at::ScalarType::BFloat16);
    CHECK_EQ(k_weight.scalar_type(), at::ScalarType::BFloat16);
    CHECK_EQ(position_ids.scalar_type(), at::ScalarType::Int);
    CHECK_EQ(q_weight.size(0), head_dim);
    CHECK_EQ(k_weight.size(0), head_dim);
    CHECK_EQ(position_ids.size(0), qkv.size(0));
    CHECK_EQ(qkv.size(1), (num_heads_q + num_heads_k + num_heads_v) * head_dim);
    CHECK_GE(rotary_dim, 0);
    TORCH_CHECK(rotary_dim <= head_dim, "rotary_dim must be <= head_dim");
    CHECK_EQ(rotary_dim % (head_dim / 32), 0);
    TORCH_CHECK(rope_scale > 0.0, "rope_scale must be positive");

    const int64_t rotary_lanes = rotary_dim / (head_dim / 32);
    TORCH_CHECK(rotary_lanes > 0, "rotary_lanes must be positive");
    if (is_neox_style) {
        const int64_t half_rotary_lanes = rotary_lanes / 2;
        TORCH_CHECK(half_rotary_lanes > 0, "half_rotary_lanes must be positive");
        CHECK_EQ(half_rotary_lanes & (half_rotary_lanes - 1), 0);
    }

    StreamType stream = GET_CURRENT_STREAM();
    rtp_llm::invokeFusedQKNormRope(reinterpret_cast<__nv_bfloat16*>(qkv.data_ptr()),
                                   reinterpret_cast<const __nv_bfloat16*>(q_weight.data_ptr()),
                                   reinterpret_cast<const __nv_bfloat16*>(k_weight.data_ptr()),
                                   reinterpret_cast<const int*>(position_ids.data_ptr()),
                                   static_cast<float>(eps),
                                   static_cast<int>(qkv.size(0)),
                                   static_cast<int>(num_heads_q),
                                   static_cast<int>(num_heads_k),
                                   static_cast<int>(num_heads_v),
                                   static_cast<int>(head_dim),
                                   static_cast<float>(rope_base),
                                   is_neox_style,
                                   static_cast<float>(rope_scale),
                                   static_cast<int>(rotary_dim),
                                   stream);
}

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
                              int64_t     rotary_dim) {
    CHECK_INPUT(qkv);
    CHECK_INPUT(q_weight);
    CHECK_INPUT(k_weight);
    CHECK_INPUT(position_ids);
    CHECK_INPUT(rope_cache);
    CHECK_DIM(2, qkv);
    CHECK_DIM(1, q_weight);
    CHECK_DIM(1, k_weight);
    CHECK_DIM(1, position_ids);
    CHECK_DIM(2, rope_cache);

    CHECK_EQ(qkv.scalar_type(), at::ScalarType::BFloat16);
    CHECK_EQ(q_weight.scalar_type(), at::ScalarType::BFloat16);
    CHECK_EQ(k_weight.scalar_type(), at::ScalarType::BFloat16);
    CHECK_EQ(position_ids.scalar_type(), at::ScalarType::Int);
    CHECK_EQ(rope_cache.scalar_type(), at::ScalarType::Float);
    CHECK_EQ(q_weight.size(0), head_dim);
    CHECK_EQ(k_weight.size(0), head_dim);
    CHECK_EQ(position_ids.size(0), qkv.size(0));
    CHECK_EQ(qkv.size(1), (num_heads_q + num_heads_k + num_heads_v) * head_dim);
    CHECK_GE(rotary_dim, 0);
    TORCH_CHECK(rotary_dim <= head_dim, "rotary_dim must be <= head_dim");
    CHECK_EQ(rotary_dim % (head_dim / 32), 0);
    CHECK_EQ(rope_cache.size(1), rotary_dim);
    TORCH_CHECK(rotary_dim % 2 == 0, "rotary_dim must be even for rope cache");

    const int64_t rotary_lanes = rotary_dim / (head_dim / 32);
    TORCH_CHECK(rotary_lanes > 0, "rotary_lanes must be positive");
    if (is_neox_style) {
        const int64_t half_rotary_lanes = rotary_lanes / 2;
        TORCH_CHECK(half_rotary_lanes > 0, "half_rotary_lanes must be positive");
        CHECK_EQ(half_rotary_lanes & (half_rotary_lanes - 1), 0);
    }

    StreamType stream = GET_CURRENT_STREAM();
    rtp_llm::invokeFusedQKNormRopeWithCache(reinterpret_cast<__nv_bfloat16*>(qkv.data_ptr()),
                                            reinterpret_cast<const __nv_bfloat16*>(q_weight.data_ptr()),
                                            reinterpret_cast<const __nv_bfloat16*>(k_weight.data_ptr()),
                                            reinterpret_cast<const int*>(position_ids.data_ptr()),
                                            reinterpret_cast<const float2*>(rope_cache.data_ptr()),
                                            static_cast<float>(eps),
                                            static_cast<int>(qkv.size(0)),
                                            static_cast<int>(num_heads_q),
                                            static_cast<int>(num_heads_k),
                                            static_cast<int>(num_heads_v),
                                            static_cast<int>(head_dim),
                                            is_neox_style,
                                            static_cast<int>(rotary_dim),
                                            stream);
}

}  // namespace torch_ext
