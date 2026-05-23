#include "rtp_llm/models_py/bindings/cuda/GatherAddRMSNormOp.h"

namespace torch_ext {

void gatherAddRMSNorm(
    at::Tensor& output, at::Tensor& input, at::Tensor& residual, at::Tensor& indices, at::Tensor& weight, double eps) {
    CHECK_INPUT(input);
    CHECK_INPUT(residual);
    CHECK_INPUT(output);
    CHECK_INPUT(indices);
    CHECK_INPUT(weight);
    CHECK_DIM(2, input);
    CHECK_DIM(2, residual);
    CHECK_DIM(2, output);
    CHECK_DIM(1, indices);
    CHECK_DIM(1, weight);
    CHECK_EQ(input.scalar_type(), residual.scalar_type());
    CHECK_EQ(input.scalar_type(), output.scalar_type());
    CHECK_EQ(input.scalar_type(), weight.scalar_type());
    TORCH_CHECK(indices.scalar_type() == at::ScalarType::Int || indices.scalar_type() == at::ScalarType::Long,
                "indices must be int32 or int64, got ",
                indices.scalar_type());

    auto device = input.device();
    CHECK_EQ(residual.device(), device);
    CHECK_EQ(output.device(), device);
    CHECK_EQ(indices.device(), device);
    CHECK_EQ(weight.device(), device);
    CHECK_EQ(input.size(0), residual.size(0));
    CHECK_EQ(input.size(1), residual.size(1));
    CHECK_EQ(input.size(1), output.size(1));
    CHECK_EQ(output.size(0), indices.numel());
    CHECK_EQ(weight.numel(), input.size(1));

    StreamType stream = GET_CURRENT_STREAM();
    rtp_llm::invokeGatherAddRMSNorm(output, input, residual, indices, weight, static_cast<float>(eps), stream);
}

}  // namespace torch_ext
