"""CUDA-specific activation function implementations."""

import os
import threading
from typing import Dict, Tuple

import torch

from rtp_llm.models_py.modules.base.common.activation import SiluAndMulBase
from rtp_llm.ops.compute_ops import rtp_llm_ops

_IDLE_FISH_REUSE_SILU_MUL_BUFFER = os.environ.get(
    "IDLE_FISH_ENABLE_REUSE_SILU_MUL_BUFFER", ""
).strip() in ("1", "true", "True")

_SILU_MUL_BUFFERS: Dict[Tuple[int, int, int, torch.dtype], torch.Tensor] = {}


class FusedSiluAndMul(SiluAndMulBase):
    """CUDA implementation of silu_and_mul using rtp_llm_ops."""

    def _make_output(
        self, gate_up: torch.Tensor, output_shape: torch.Size, stream_id: int
    ) -> torch.Tensor:
        if not _IDLE_FISH_REUSE_SILU_MUL_BUFFER:
            return torch.empty(output_shape, dtype=gate_up.dtype, device=gate_up.device)
        if not gate_up.is_cuda or not gate_up.is_contiguous():
            return torch.empty(output_shape, dtype=gate_up.dtype, device=gate_up.device)

        device_index = gate_up.get_device()
        key = (threading.get_ident(), device_index, int(stream_id), gate_up.dtype)
        output = _SILU_MUL_BUFFERS.get(key)
        if output is None or output.shape != output_shape:
            output = torch.empty(
                output_shape, dtype=gate_up.dtype, device=gate_up.device
            )
            _SILU_MUL_BUFFERS[key] = output
        return output

    def forward(self, gate_up: torch.Tensor) -> torch.Tensor:
        """
        Perform SiLU activation and element-wise multiplication using CUDA kernel.

        Args:
            output: Output tensor to write result to
            gate_up: Input tensor with concatenated gate and up projections
        """
        d = gate_up.shape[-1] // 2
        output_shape = gate_up.shape[:-1] + (d,)
        stream_id = torch.cuda.current_stream().cuda_stream
        output = self._make_output(gate_up, output_shape, stream_id)
        rtp_llm_ops.silu_and_mul(output, gate_up, stream_id)
        return output
