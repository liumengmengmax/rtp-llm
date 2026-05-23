#include "rtp_llm/models_py/bindings/cuda/kernels/idlefish_mlp_l2norm.h"

#include "rtp_llm/models_py/bindings/cuda/cuda_type_utils.cuh"
#include "rtp_llm/models_py/bindings/cuda/reduce_kernel_utils.cuh"

#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace rtp_llm {

template<typename T, int ThreadsPerOutput>
__global__ void idleFishMlpL2NormKernel(const T* __restrict__ input,
                                        const T* __restrict__ weight,
                                        const T* __restrict__ bias,
                                        T* __restrict__ output,
                                        int64_t rows,
                                        int64_t input_size,
                                        int64_t output_size,
                                        float   eps) {
    const int64_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    extern __shared__ float smem[];
    float*                  partial = smem;
    float*                  values  = smem + blockDim.x;

    const int out_col = threadIdx.x / ThreadsPerOutput;
    const int lane    = threadIdx.x % ThreadsPerOutput;

    float acc = 0.0f;
    if (out_col < output_size) {
        const int64_t input_offset  = row * input_size;
        const int64_t weight_offset = static_cast<int64_t>(out_col) * input_size;
        for (int64_t k = lane; k < input_size; k += ThreadsPerOutput) {
            acc += cuda_cast<float>(input[input_offset + k]) * cuda_cast<float>(weight[weight_offset + k]);
        }
    }
    partial[threadIdx.x] = acc;
    __syncthreads();

    if (lane == 0 && out_col < output_size) {
        float     value = cuda_cast<float>(bias[out_col]);
        const int base  = out_col * ThreadsPerOutput;
#pragma unroll
        for (int i = 0; i < ThreadsPerOutput; ++i) {
            value += partial[base + i];
        }
        values[out_col] = value;
    }
    __syncthreads();

    float sum = 0.0f;
    for (int64_t col = threadIdx.x; col < output_size; col += blockDim.x) {
        const float value = values[col];
        sum += value * value;
    }
    sum = blockReduceSum<float>(sum);

    __shared__ float inv_norm;
    if (threadIdx.x == 0) {
        inv_norm = rsqrtf(sum + eps);
    }
    __syncthreads();

    for (int64_t col = threadIdx.x; col < output_size; col += blockDim.x) {
        output[row * output_size + col] = cuda_cast<T>(values[col] * inv_norm);
    }
}

void invokeIdleFishMlpL2Norm(at::Tensor&       output,
                             const at::Tensor& input,
                             const at::Tensor& weight,
                             const at::Tensor& bias,
                             float             eps,
                             cudaStream_t      stream) {
    const auto rows        = input.size(0);
    const auto input_size  = input.size(1);
    const auto output_size = weight.size(0);
    if (rows == 0 || input_size == 0 || output_size == 0) {
        return;
    }
    TORCH_CHECK(output_size <= 128, "invokeIdleFishMlpL2Norm only supports output_size <= 128, got ", output_size);

    constexpr int threads_per_output = 8;
    int           threads            = static_cast<int>(output_size) * threads_per_output;
    threads                          = ((threads + 31) / 32) * 32;
    const size_t shared_bytes        = static_cast<size_t>(threads + output_size) * sizeof(float);

    const dim3 grid(static_cast<unsigned int>(rows));
    const dim3 block(threads);

    switch (input.scalar_type()) {
        case at::ScalarType::Half: {
            using KernelType = half;
            idleFishMlpL2NormKernel<KernelType, threads_per_output>
                <<<grid, block, shared_bytes, stream>>>(reinterpret_cast<const KernelType*>(input.data_ptr()),
                                                        reinterpret_cast<const KernelType*>(weight.data_ptr()),
                                                        reinterpret_cast<const KernelType*>(bias.data_ptr()),
                                                        reinterpret_cast<KernelType*>(output.data_ptr()),
                                                        rows,
                                                        input_size,
                                                        output_size,
                                                        eps);
            break;
        }
        case at::ScalarType::BFloat16: {
            using KernelType = __nv_bfloat16;
            idleFishMlpL2NormKernel<KernelType, threads_per_output>
                <<<grid, block, shared_bytes, stream>>>(reinterpret_cast<const KernelType*>(input.data_ptr()),
                                                        reinterpret_cast<const KernelType*>(weight.data_ptr()),
                                                        reinterpret_cast<const KernelType*>(bias.data_ptr()),
                                                        reinterpret_cast<KernelType*>(output.data_ptr()),
                                                        rows,
                                                        input_size,
                                                        output_size,
                                                        eps);
            break;
        }
        default:
            TORCH_CHECK(false, "invokeIdleFishMlpL2Norm only supports fp16/bf16 input, got ", input.scalar_type());
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace rtp_llm
