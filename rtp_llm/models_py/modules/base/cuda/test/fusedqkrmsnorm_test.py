import itertools
import os
from unittest import SkipTest, TestCase, main
from unittest.mock import patch

import torch
from torch import dtype as _dtype

from rtp_llm.models_py.modules import FusedQKRMSNorm, QKRMSNorm
from rtp_llm.ops.compute_ops import rtp_llm_ops


class FusedQKRMSNormTest(TestCase):
    DTYPES = [torch.half, torch.bfloat16]
    NUM_TOKENS = [7, 83, 4096]
    HEAD_NUM = [40]
    KV_HEAD_NUM = [40, 8, 4]
    SIZE_PER_HEAD = [128]

    # DTYPES = [torch.bfloat16]
    # NUM_TOKENS = [4096]
    # HEAD_NUM = [40]
    # KV_HEAD_NUM =  [40]
    # SIZE_PER_HEAD = [128]

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _run_fused_qk_rmsnorm_test(
        self,
        num_tokens: int,
        head_num: int,
        kv_head_num: int,
        size_per_head: int,
        dtype: _dtype,
    ):
        torch.manual_seed(0)

        hidden_size = head_num * size_per_head + 2 * kv_head_num * size_per_head

        q_weight = torch.randn(size_per_head, dtype=dtype)
        k_weight = torch.randn(size_per_head, dtype=dtype)

        qkrmsnorm = QKRMSNorm(q_weight, k_weight, head_num, kv_head_num, size_per_head)
        fused_qkrmsnorm = FusedQKRMSNorm(
            q_weight, k_weight, head_num, kv_head_num, size_per_head
        )

        x = torch.randn(num_tokens, hidden_size, dtype=dtype)

        ref = qkrmsnorm(x)
        self.assertTrue(
            torch.allclose(ref, fused_qkrmsnorm(x.clone()), atol=1e-2, rtol=1e-2)
        )

        with patch.dict(os.environ, {"IDLE_FISH_ENABLE_RTP_FUSED_QK_RMSNORM": "1"}):
            rtp_fused_qkrmsnorm = FusedQKRMSNorm(
                q_weight, k_weight, head_num, kv_head_num, size_per_head
            )
        self.assertTrue(
            torch.allclose(ref, rtp_fused_qkrmsnorm(x.clone()), atol=1e-2, rtol=1e-2)
        )

    def test_fusedqkrmsnorm(self):
        for params in itertools.product(
            self.NUM_TOKENS,
            self.HEAD_NUM,
            self.KV_HEAD_NUM,
            self.SIZE_PER_HEAD,
            self.DTYPES,
        ):
            with self.subTest(
                num_tokens=params[0],
                head_num=params[1],
                kv_head_num=params[2],
                size_per_head=params[3],
                dtype=params[4],
            ):
                self._run_fused_qk_rmsnorm_test(*params)

    @staticmethod
    def _apply_base_rope_neox(
        x: torch.Tensor, position_ids: torch.Tensor, base: float = 10000.0
    ) -> torch.Tensor:
        rotary_dim = x.shape[-1]
        half_dim = rotary_dim // 2
        freqs = 1.0 / (
            base
            ** (
                torch.arange(0, rotary_dim, 2, device=x.device, dtype=torch.float32)
                / rotary_dim
            )
        )
        angles = position_ids.to(torch.float32)[:, None] * freqs[None, :]
        cos = angles.cos()[:, None, :]
        sin = angles.sin()[:, None, :]
        x_float = x.float()
        x1 = x_float[..., :half_dim]
        x2 = x_float[..., half_dim:rotary_dim]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(x.dtype)

    def test_fused_qk_norm_rope(self):
        torch.manual_seed(1)
        dtype = torch.bfloat16
        num_tokens = 83
        head_num = 40
        kv_head_num = 8
        size_per_head = 128
        hidden_size = (head_num + 2 * kv_head_num) * size_per_head
        q_size = head_num * size_per_head
        kv_size = kv_head_num * size_per_head

        q_weight = torch.randn(size_per_head, dtype=dtype)
        k_weight = torch.randn(size_per_head, dtype=dtype)
        qkv = torch.randn(num_tokens, hidden_size, dtype=dtype)
        position_ids = torch.arange(num_tokens, dtype=torch.int32, device=qkv.device)

        qkrmsnorm = QKRMSNorm(q_weight, k_weight, head_num, kv_head_num, size_per_head)
        ref = qkrmsnorm(qkv.clone())
        q_ref, k_ref, v_ref = ref.split([q_size, kv_size, kv_size], dim=-1)
        q_ref = q_ref.reshape(num_tokens, head_num, size_per_head)
        k_ref = k_ref.reshape(num_tokens, kv_head_num, size_per_head)
        q_ref = self._apply_base_rope_neox(q_ref, position_ids)
        k_ref = self._apply_base_rope_neox(k_ref, position_ids)
        ref = torch.cat(
            [
                q_ref.reshape(num_tokens, q_size),
                k_ref.reshape(num_tokens, kv_size),
                v_ref,
            ],
            dim=-1,
        )

        out = qkv.clone()
        rtp_llm_ops.fused_qk_norm_rope(
            out,
            q_weight,
            k_weight,
            position_ids,
            1e-6,
            head_num,
            kv_head_num,
            kv_head_num,
            size_per_head,
            10000.0,
            True,
            1.0,
            size_per_head,
        )
        self.assertTrue(torch.allclose(ref, out, atol=4e-2, rtol=4e-2))

        freqs = 1.0 / (
            10000.0
            ** (
                torch.arange(
                    0, size_per_head, 2, device=qkv.device, dtype=torch.float32
                )
                / size_per_head
            )
        )
        angles = position_ids.to(torch.float32)[:, None] * freqs[None, :]
        rope_cache = (
            torch.stack((angles.cos(), angles.sin()), dim=-1)
            .reshape(num_tokens, size_per_head)
            .contiguous()
        )
        out_cache = qkv.clone()
        rtp_llm_ops.fused_qk_norm_rope_with_cache(
            out_cache,
            q_weight,
            k_weight,
            position_ids,
            rope_cache,
            1e-6,
            head_num,
            kv_head_num,
            kv_head_num,
            size_per_head,
            True,
            size_per_head,
        )
        self.assertTrue(torch.allclose(ref, out_cache, atol=4e-2, rtol=4e-2))


if __name__ == "__main__":
    main()
