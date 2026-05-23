#pragma once

#include "rtp_llm/models_py/bindings/common/Torch_ext.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/idlefish_mlp_l2norm.h"

namespace torch_ext {

void idleFishMlpL2Norm(at::Tensor& output, at::Tensor& input, at::Tensor& weight, at::Tensor& bias, double eps);

}  // namespace torch_ext
