"""CUDA F16 (non-quantized) Linear implementation"""

import os
from typing import Optional

import torch
from torch.nn import functional as F

from rtp_llm.models_py.modules.factory.linear import LinearBase
from rtp_llm.ops import HWKernelConfig

_IDLE_FISH_LINEAR_MATMUL_NO_BIAS = os.environ.get(
    "IDLE_FISH_ENABLE_LINEAR_MATMUL_NO_BIAS", ""
).strip() in ("1", "true", "True")


class CudaF16Linear(LinearBase):
    """CUDA F16 (non-quantized) Linear"""

    @classmethod
    def can_handle(
        cls,
        quant_config: object,
        weight: torch.Tensor,
        weight_scales: Optional[torch.Tensor],
        hw_kernel_config: Optional["HWKernelConfig"] = None,
        weight_scale_2: Optional[torch.Tensor] = None,
        input_scale: Optional[torch.Tensor] = None,
    ) -> bool:
        """Handle non-FP8 and non-FP4 cases (no weight_scales)"""
        return weight_scales is None

    def __init__(
        self,
        weight: torch.Tensor,
        weight_scales: Optional[torch.Tensor] = None,
        input_scales: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        quant_config: object = None,
        weight_scale_2: Optional[torch.Tensor] = None,
    ):
        super().__init__(
            weight, weight_scales, input_scales, bias, quant_config, weight_scale_2
        )
        self.matmul_weight = (
            weight if _IDLE_FISH_LINEAR_MATMUL_NO_BIAS and bias is None else None
        )
        self.weight = weight.T
        self.bias = bias

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.matmul_weight is not None:
            return torch.matmul(input, self.matmul_weight)
        return F.linear(input, self.weight, self.bias)
