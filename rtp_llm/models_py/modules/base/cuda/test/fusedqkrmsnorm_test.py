import itertools
import os
from unittest import SkipTest, TestCase, main
from unittest.mock import patch

import torch
from torch import dtype as _dtype

from rtp_llm.models_py.modules import FusedQKRMSNorm, QKRMSNorm


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


if __name__ == "__main__":
    main()
