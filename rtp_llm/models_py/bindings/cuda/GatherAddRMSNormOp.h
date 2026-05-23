#pragma once

#include "rtp_llm/models_py/bindings/common/Torch_ext.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/gather_add_rmsnorm.h"

namespace torch_ext {

void gatherAddRMSNorm(
    at::Tensor& output, at::Tensor& input, at::Tensor& residual, at::Tensor& indices, at::Tensor& weight, double eps);

}  // namespace torch_ext
