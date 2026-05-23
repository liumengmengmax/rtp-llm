#include "rtp_llm/models_py/bindings/cuda/IdleFishMlpL2NormOp.h"

namespace torch_ext {

void idleFishMlpL2Norm(at::Tensor& output, at::Tensor& input, at::Tensor& weight, at::Tensor& bias, double eps) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);
    CHECK_INPUT(output);
    CHECK_DIM(2, input);
    CHECK_DIM(2, weight);
    CHECK_DIM(1, bias);
    CHECK_DIM(2, output);
    CHECK_EQ(input.scalar_type(), weight.scalar_type());
    CHECK_EQ(input.scalar_type(), bias.scalar_type());
    CHECK_EQ(input.scalar_type(), output.scalar_type());

    auto device = input.device();
    CHECK_EQ(weight.device(), device);
    CHECK_EQ(bias.device(), device);
    CHECK_EQ(output.device(), device);
    CHECK_EQ(input.size(0), output.size(0));
    CHECK_EQ(input.size(1), weight.size(1));
    CHECK_EQ(weight.size(0), output.size(1));
    CHECK_EQ(weight.size(0), bias.numel());

    StreamType stream = GET_CURRENT_STREAM();
    rtp_llm::invokeIdleFishMlpL2Norm(output, input, weight, bias, static_cast<float>(eps), stream);
}

}  // namespace torch_ext
