#include "rtp_llm/models_py/bindings/cuda/L2NormOp.h"

namespace torch_ext {

void l2norm(at::Tensor& output, at::Tensor& input, double eps) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_DIM(2, input);
    CHECK_DIM(2, output);
    CHECK_EQ(input.scalar_type(), output.scalar_type());
    CHECK_EQ(input.size(0), output.size(0));
    CHECK_EQ(input.size(1), output.size(1));

    auto device = input.device();
    CHECK_EQ(output.device(), device);

    StreamType stream = GET_CURRENT_STREAM();
    rtp_llm::invokeL2Norm(output, input, static_cast<float>(eps), stream);
}

}  // namespace torch_ext
