#pragma once

#include <ATen/ATen.h>
#include <cuda_runtime_api.h>

namespace rtp_llm {

void invokeIdleFishMlpL2Norm(at::Tensor&       output,
                             const at::Tensor& input,
                             const at::Tensor& weight,
                             const at::Tensor& bias,
                             float             eps,
                             cudaStream_t      stream);

}  // namespace rtp_llm
