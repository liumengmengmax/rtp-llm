#include "rtp_llm/models_py/bindings/cuda/kernels/fused_qk_norm_rope.h"

#if USING_CUDA
#include "rtp_llm/models_py/bindings/cuda/cuda_host_utils.h"

#include <ATen/cuda/Exceptions.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>

namespace rtp_llm {
namespace {

constexpr unsigned int kFullWarpMask = 0xffffffffu;

template<int N>
struct PackedAs;

template<>
struct PackedAs<1> {
    using Type = uint32_t;
};

template<>
struct PackedAs<2> {
    using Type = uint2;
};

template<>
struct PackedAs<4> {
    using Type = uint4;
};

template<typename T>
__inline__ __device__ T warpReduceSum(T val) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(kFullWarpMask, val, mask, 32);
    }
    return val;
}

template<typename T>
__host__ __device__ inline T divUp(T m, T n) {
    return (m + n - 1) / n;
}

__device__ inline float computeBaseFreq(float base, int rotary_dim, int half_dim) {
    return powf(base, -2.0f * static_cast<float>(half_dim) / static_cast<float>(rotary_dim));
}

template<int HeadDim, bool Interleave>
__global__ void fusedQKNormRopeKernel(__nv_bfloat16* __restrict__ qkv,
                                      int   num_heads_q,
                                      int   num_heads_k,
                                      int   num_heads_v,
                                      float eps,
                                      const __nv_bfloat16* __restrict__ q_weight,
                                      const __nv_bfloat16* __restrict__ k_weight,
                                      float rope_base,
                                      const int* __restrict__ position_ids,
                                      const float2* __restrict__ rope_cache,
                                      int   num_tokens,
                                      float rope_scale,
                                      int   rotary_dim) {
    const int warps_per_block = blockDim.x / 32;
    const int warp_id         = threadIdx.x / 32;
    const int lane_id         = threadIdx.x % 32;
    const int global_warp_idx = blockIdx.x * warps_per_block + warp_id;

    const int total_qk_heads = num_heads_q + num_heads_k;
    const int token_idx      = global_warp_idx / total_qk_heads;
    const int local_head_idx = global_warp_idx % total_qk_heads;
    if (token_idx >= num_tokens) {
        return;
    }

    constexpr int elems_per_thread = HeadDim / 32;
    static_assert(HeadDim % 64 == 0, "HeadDim must be divisible by 64");
    constexpr int elem_bytes = elems_per_thread * sizeof(__nv_bfloat16);
    static_assert(elem_bytes % 4 == 0, "packed load width must be a multiple of 4 bytes");
    constexpr int vec_size = elem_bytes / 4;
    using VecT             = typename PackedAs<vec_size>::Type;

    const bool is_q       = local_head_idx < num_heads_q;
    const int  head_idx   = is_q ? local_head_idx : local_head_idx - num_heads_q;
    const int  all_heads  = num_heads_q + num_heads_k + num_heads_v;
    const int  head_start = is_q ? head_idx * HeadDim : (num_heads_q + head_idx) * HeadDim;
    const int  base_off   = token_idx * all_heads * HeadDim + head_start;
    const int  elem_off   = base_off + lane_id * elems_per_thread;

    float elems[elems_per_thread];
    float square_sum = 0.0f;
    {
        VecT vec = *reinterpret_cast<const VecT*>(&qkv[elem_off]);
#pragma unroll
        for (int i = 0; i < vec_size; ++i) {
            auto vals = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<uint32_t*>(&vec) + i));
            elems[2 * i]     = vals.x;
            elems[2 * i + 1] = vals.y;
            square_sum += vals.x * vals.x + vals.y * vals.y;
        }
    }

    square_sum          = warpReduceSum(square_sum);
    const float rms_rcp = rsqrtf(square_sum / static_cast<float>(HeadDim) + eps);

#pragma unroll
    for (int i = 0; i < elems_per_thread; ++i) {
        const int   dim    = lane_id * elems_per_thread + i;
        const float weight = is_q ? __bfloat162float(q_weight[dim]) : __bfloat162float(k_weight[dim]);
        elems[i] *= rms_rcp * weight;
    }

    const int rotary_lanes = rotary_dim / elems_per_thread;
    if (lane_id < rotary_lanes) {
        float       elems_rot[elems_per_thread];
        float       cos_vals[elems_per_thread];
        float       sin_vals[elems_per_thread];
        const int   pos_id = position_ids[token_idx];
        const float pos    = static_cast<float>(pos_id) / rope_scale;

        if constexpr (Interleave) {
#pragma unroll
            for (int i = 0; i < elems_per_thread; ++i) {
                elems_rot[i]       = (i % 2 == 0) ? -elems[i + 1] : elems[i - 1];
                const int dim_idx  = lane_id * elems_per_thread + i;
                const int half_dim = dim_idx / 2;
                if (rope_cache != nullptr) {
                    const float2 coef = rope_cache[pos_id * (rotary_dim / 2) + half_dim];
                    cos_vals[i]       = coef.x;
                    sin_vals[i]       = coef.y;
                } else {
                    const float theta = pos * computeBaseFreq(rope_base, rotary_dim, half_dim);
                    __sincosf(theta, &sin_vals[i], &cos_vals[i]);
                }
            }
        } else {
            __syncwarp();
            const int          half_rotary_lanes = rotary_lanes / 2;
            const unsigned int active_mask       = rotary_lanes >= 32 ? kFullWarpMask : ((1u << rotary_lanes) - 1u);
#pragma unroll
            for (int i = 0; i < elems_per_thread; ++i) {
                elems_rot[i] = __shfl_xor_sync(active_mask, elems[i], half_rotary_lanes);
                if (lane_id < half_rotary_lanes) {
                    elems_rot[i] = -elems_rot[i];
                }
                int dim_idx        = lane_id * elems_per_thread + i;
                dim_idx            = (dim_idx * 2) % rotary_dim;
                const int half_dim = dim_idx / 2;
                if (rope_cache != nullptr) {
                    const float2 coef = rope_cache[pos_id * (rotary_dim / 2) + half_dim];
                    cos_vals[i]       = coef.x;
                    sin_vals[i]       = coef.y;
                } else {
                    const float theta = pos * computeBaseFreq(rope_base, rotary_dim, half_dim);
                    __sincosf(theta, &sin_vals[i], &cos_vals[i]);
                }
            }
            __syncwarp();
        }

#pragma unroll
        for (int i = 0; i < elems_per_thread; ++i) {
            elems[i] = elems[i] * cos_vals[i] + elems_rot[i] * sin_vals[i];
        }
    }

    {
        VecT vec;
#pragma unroll
        for (int i = 0; i < vec_size; ++i) {
            auto vals = __float22bfloat162_rn(make_float2(elems[2 * i], elems[2 * i + 1]));
            reinterpret_cast<__nv_bfloat162&>(*(reinterpret_cast<uint32_t*>(&vec) + i)) = vals;
        }
        *reinterpret_cast<VecT*>(&qkv[elem_off]) = vec;
    }
}

#define DISPATCH_INTERLEAVE(INTERLEAVE_VALUE, INTERLEAVE_CONST, ...)                                                   \
    if (INTERLEAVE_VALUE) {                                                                                            \
        constexpr bool INTERLEAVE_CONST = true;                                                                        \
        __VA_ARGS__;                                                                                                   \
    } else {                                                                                                           \
        constexpr bool INTERLEAVE_CONST = false;                                                                       \
        __VA_ARGS__;                                                                                                   \
    }

template<int HeadDim>
void launchFusedQKNormRope(__nv_bfloat16*       qkv,
                           int                  num_tokens,
                           int                  num_heads_q,
                           int                  num_heads_k,
                           int                  num_heads_v,
                           float                eps,
                           const __nv_bfloat16* q_weight,
                           const __nv_bfloat16* k_weight,
                           float                rope_base,
                           bool                 interleave,
                           const int*           position_ids,
                           const float2*        rope_cache,
                           float                rope_scale,
                           int                  rotary_dim,
                           cudaStream_t         stream) {
    constexpr int block_size      = 256;
    constexpr int warps_per_block = block_size / 32;
    const int     total_warps     = num_tokens * (num_heads_q + num_heads_k);
    const int     grid_size       = divUp(total_warps, warps_per_block);
    DISPATCH_INTERLEAVE(interleave, Interleave, {
        fusedQKNormRopeKernel<HeadDim, Interleave><<<grid_size, block_size, 0, stream>>>(qkv,
                                                                                         num_heads_q,
                                                                                         num_heads_k,
                                                                                         num_heads_v,
                                                                                         eps,
                                                                                         q_weight,
                                                                                         k_weight,
                                                                                         rope_base,
                                                                                         position_ids,
                                                                                         rope_cache,
                                                                                         num_tokens,
                                                                                         rope_scale,
                                                                                         rotary_dim);
    });
}

void invokeFusedQKNormRopeImpl(__nv_bfloat16*       qkv,
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
                               float                rope_base,
                               bool                 is_neox_style,
                               float                rope_scale,
                               int                  rotary_dim,
                               cudaStream_t         stream) {
    const bool interleave = !is_neox_style;
    switch (head_dim) {
        case 64:
            launchFusedQKNormRope<64>(qkv,
                                      num_tokens,
                                      num_heads_q,
                                      num_heads_k,
                                      num_heads_v,
                                      eps,
                                      q_weight,
                                      k_weight,
                                      rope_base,
                                      interleave,
                                      position_ids,
                                      rope_cache,
                                      rope_scale,
                                      rotary_dim,
                                      stream);
            break;
        case 128:
            launchFusedQKNormRope<128>(qkv,
                                       num_tokens,
                                       num_heads_q,
                                       num_heads_k,
                                       num_heads_v,
                                       eps,
                                       q_weight,
                                       k_weight,
                                       rope_base,
                                       interleave,
                                       position_ids,
                                       rope_cache,
                                       rope_scale,
                                       rotary_dim,
                                       stream);
            break;
        case 256:
            launchFusedQKNormRope<256>(qkv,
                                       num_tokens,
                                       num_heads_q,
                                       num_heads_k,
                                       num_heads_v,
                                       eps,
                                       q_weight,
                                       k_weight,
                                       rope_base,
                                       interleave,
                                       position_ids,
                                       rope_cache,
                                       rope_scale,
                                       rotary_dim,
                                       stream);
            break;
        default:
            throw std::invalid_argument("invokeFusedQKNormRope only supports head_dim 64/128/256");
    }
    check_cuda_value(cudaPeekAtLastError());
    check_cuda_error();
}

}  // namespace

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
                           cudaStream_t         stream) {
    invokeFusedQKNormRopeImpl(qkv,
                              q_weight,
                              k_weight,
                              position_ids,
                              nullptr,
                              eps,
                              num_tokens,
                              num_heads_q,
                              num_heads_k,
                              num_heads_v,
                              head_dim,
                              rope_base,
                              is_neox_style,
                              rope_scale,
                              rotary_dim,
                              stream);
}

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
                                    cudaStream_t         stream) {
    invokeFusedQKNormRopeImpl(qkv,
                              q_weight,
                              k_weight,
                              position_ids,
                              rope_cache,
                              eps,
                              num_tokens,
                              num_heads_q,
                              num_heads_k,
                              num_heads_v,
                              head_dim,
                              10000.0f,
                              is_neox_style,
                              1.0f,
                              rotary_dim,
                              stream);
}

}  // namespace rtp_llm
#endif
