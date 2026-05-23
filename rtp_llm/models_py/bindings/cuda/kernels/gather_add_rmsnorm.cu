#include "rtp_llm/models_py/bindings/cuda/kernels/gather_add_rmsnorm.h"

#include "rtp_llm/models_py/bindings/cuda/cuda_type_utils.cuh"
#include "rtp_llm/models_py/bindings/cuda/reduce_kernel_utils.cuh"

#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace rtp_llm {

template<typename T, typename IndexT>
__global__ void gatherAddRMSNormKernel(const T* __restrict__ input,
                                       const T* __restrict__ residual,
                                       const IndexT* __restrict__ indices,
                                       const T* __restrict__ weight,
                                       T* __restrict__ output,
                                       int64_t rows,
                                       int64_t hidden_size,
                                       float   eps) {
    const int64_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const int64_t input_row     = static_cast<int64_t>(indices[row]);
    const int64_t input_offset  = input_row * hidden_size;
    const int64_t output_offset = row * hidden_size;

    float sum = 0.0f;
    for (int64_t col = threadIdx.x; col < hidden_size; col += blockDim.x) {
        const float val = cuda_cast<float>(input[input_offset + col]) + cuda_cast<float>(residual[input_offset + col]);
        sum += val * val;
    }

    sum = blockReduceSum<float>(sum);

    __shared__ float inv_rms;
    if (threadIdx.x == 0) {
        inv_rms = rsqrtf(sum / static_cast<float>(hidden_size) + eps);
    }
    __syncthreads();

    for (int64_t col = threadIdx.x; col < hidden_size; col += blockDim.x) {
        const float val = cuda_cast<float>(input[input_offset + col]) + cuda_cast<float>(residual[input_offset + col]);
        const float out = val * inv_rms * cuda_cast<float>(weight[col]);
        output[output_offset + col] = cuda_cast<T>(out);
    }
}

template<typename T>
void dispatchGatherAddRMSNormIndex(at::Tensor&       output,
                                   const at::Tensor& input,
                                   const at::Tensor& residual,
                                   const at::Tensor& indices,
                                   const at::Tensor& weight,
                                   float             eps,
                                   cudaStream_t      stream) {
    const auto rows        = output.size(0);
    const auto hidden_size = output.size(1);
    if (rows == 0 || hidden_size == 0) {
        return;
    }

    constexpr int threads = 256;
    const dim3    grid(static_cast<unsigned int>(rows));
    const dim3    block(threads);

    switch (indices.scalar_type()) {
        case at::ScalarType::Int: {
            using IndexType = int32_t;
            gatherAddRMSNormKernel<T, IndexType>
                <<<grid, block, 0, stream>>>(reinterpret_cast<const T*>(input.data_ptr()),
                                             reinterpret_cast<const T*>(residual.data_ptr()),
                                             reinterpret_cast<const IndexType*>(indices.data_ptr()),
                                             reinterpret_cast<const T*>(weight.data_ptr()),
                                             reinterpret_cast<T*>(output.data_ptr()),
                                             rows,
                                             hidden_size,
                                             eps);
            break;
        }
        case at::ScalarType::Long: {
            using IndexType = int64_t;
            gatherAddRMSNormKernel<T, IndexType>
                <<<grid, block, 0, stream>>>(reinterpret_cast<const T*>(input.data_ptr()),
                                             reinterpret_cast<const T*>(residual.data_ptr()),
                                             reinterpret_cast<const IndexType*>(indices.data_ptr()),
                                             reinterpret_cast<const T*>(weight.data_ptr()),
                                             reinterpret_cast<T*>(output.data_ptr()),
                                             rows,
                                             hidden_size,
                                             eps);
            break;
        }
        default:
            TORCH_CHECK(false, "invokeGatherAddRMSNorm only supports int32/int64 indices, got ", indices.scalar_type());
    }
}

void invokeGatherAddRMSNorm(at::Tensor&       output,
                            const at::Tensor& input,
                            const at::Tensor& residual,
                            const at::Tensor& indices,
                            const at::Tensor& weight,
                            float             eps,
                            cudaStream_t      stream) {
    switch (input.scalar_type()) {
        case at::ScalarType::Half: {
            using KernelType = half;
            dispatchGatherAddRMSNormIndex<KernelType>(output, input, residual, indices, weight, eps, stream);
            break;
        }
        case at::ScalarType::BFloat16: {
            using KernelType = __nv_bfloat16;
            dispatchGatherAddRMSNormIndex<KernelType>(output, input, residual, indices, weight, eps, stream);
            break;
        }
        default:
            TORCH_CHECK(false, "invokeGatherAddRMSNorm only supports fp16/bf16 input, got ", input.scalar_type());
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace rtp_llm
