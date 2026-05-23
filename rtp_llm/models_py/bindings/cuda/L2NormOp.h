#pragma once

#include "rtp_llm/models_py/bindings/common/Torch_ext.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/l2norm.h"

namespace torch_ext {

void l2norm(at::Tensor& output, at::Tensor& input, double eps);

}  // namespace torch_ext
