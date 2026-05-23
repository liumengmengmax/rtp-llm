#include "rtp_llm/models_py/bindings/cuda/kernels/l2norm.h"

#include "rtp_llm/models_py/bindings/cuda/cuda_type_utils.cuh"
#include "rtp_llm/models_py/bindings/cuda/reduce_kernel_utils.cuh"

#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace rtp_llm {

template<typename T>
__global__ void
l2NormKernel(const T* __restrict__ input, T* __restrict__ output, int64_t rows, int64_t hidden_size, float eps) {
    const int64_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const int64_t offset = row * hidden_size;
    float         sum    = 0.0f;
    for (int64_t col = threadIdx.x; col < hidden_size; col += blockDim.x) {
        const float val = cuda_cast<float>(input[offset + col]);
        sum += val * val;
    }

    sum = blockReduceSum<float>(sum);

    __shared__ float inv_norm;
    if (threadIdx.x == 0) {
        inv_norm = rsqrtf(sum + eps);
    }
    __syncthreads();

    for (int64_t col = threadIdx.x; col < hidden_size; col += blockDim.x) {
        const float val      = cuda_cast<float>(input[offset + col]) * inv_norm;
        output[offset + col] = cuda_cast<T>(val);
    }
}

void invokeL2Norm(at::Tensor& output, const at::Tensor& input, float eps, cudaStream_t stream) {
    const auto rows        = input.size(0);
    const auto hidden_size = input.size(1);
    if (rows == 0 || hidden_size == 0) {
        return;
    }

    constexpr int threads = 256;
    const dim3    grid(static_cast<unsigned int>(rows));
    const dim3    block(threads);

    switch (input.scalar_type()) {
        case at::ScalarType::Half: {
            using KernelType = half;
            l2NormKernel<KernelType><<<grid, block, 0, stream>>>(reinterpret_cast<const KernelType*>(input.data_ptr()),
                                                                 reinterpret_cast<KernelType*>(output.data_ptr()),
                                                                 rows,
                                                                 hidden_size,
                                                                 eps);
            break;
        }
        case at::ScalarType::BFloat16: {
            using KernelType = __nv_bfloat16;
            l2NormKernel<KernelType><<<grid, block, 0, stream>>>(reinterpret_cast<const KernelType*>(input.data_ptr()),
                                                                 reinterpret_cast<KernelType*>(output.data_ptr()),
                                                                 rows,
                                                                 hidden_size,
                                                                 eps);
            break;
        }
        default:
            TORCH_CHECK(false, "invokeL2Norm only supports fp16/bf16 input, got ", input.scalar_type());
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace rtp_llm
