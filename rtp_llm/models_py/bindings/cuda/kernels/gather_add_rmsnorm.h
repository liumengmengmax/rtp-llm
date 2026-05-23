#pragma once

#include <ATen/ATen.h>
#include <cuda_runtime_api.h>

namespace rtp_llm {

void invokeGatherAddRMSNorm(at::Tensor&       output,
                            const at::Tensor& input,
                            const at::Tensor& residual,
                            const at::Tensor& indices,
                            const at::Tensor& weight,
                            float             eps,
                            cudaStream_t      stream);

}  // namespace rtp_llm
