#pragma once

#include <ATen/ATen.h>
#include <cuda_runtime_api.h>

namespace rtp_llm {

void invokeL2Norm(at::Tensor& output, const at::Tensor& input, float eps, cudaStream_t stream);

}  // namespace rtp_llm
