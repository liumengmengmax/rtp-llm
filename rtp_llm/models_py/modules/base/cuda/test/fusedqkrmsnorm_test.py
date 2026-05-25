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
    def _rope_cache(
        position_ids: torch.Tensor, rotary_dim: int, base: float = 10000.0
    ) -> torch.Tensor:
        freqs = 1.0 / (
            base
            ** (
                torch.arange(
                    0, rotary_dim, 2, device=position_ids.device, dtype=torch.float32
                )
                / rotary_dim
            )
        )
        angles = position_ids.to(torch.float32)[:, None] * freqs[None, :]
        return (
            torch.stack((angles.cos(), angles.sin()), dim=-1)
            .reshape(position_ids.numel(), rotary_dim)
            .contiguous()
        )

    @staticmethod
    def _apply_base_rope(
        x: torch.Tensor,
        position_ids: torch.Tensor,
        rotary_dim: int,
        is_neox_style: bool,
        base: float = 10000.0,
    ) -> torch.Tensor:
        rope_cache = FusedQKRMSNormTest._rope_cache(position_ids, rotary_dim, base)
        cos = rope_cache[:, 0::2][:, None, :]
        sin = rope_cache[:, 1::2][:, None, :]
        x_float = x.float()
        out = x_float.clone()
        x_rot = x_float[..., :rotary_dim]

        if is_neox_style:
            half_dim = rotary_dim // 2
            x1 = x_rot[..., :half_dim]
            x2 = x_rot[..., half_dim:rotary_dim]
            out_rot = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        else:
            x_even = x_rot[..., 0::2]
            x_odd = x_rot[..., 1::2]
            out_rot = torch.empty_like(x_rot)
            out_rot[..., 0::2] = x_even * cos - x_odd * sin
            out_rot[..., 1::2] = x_odd * cos + x_even * sin

        out[..., :rotary_dim] = out_rot
        return out.to(x.dtype)

    def _assert_close_with_rope_debug(
        self,
        ref: torch.Tensor,
        out: torch.Tensor,
        label: str,
        head_dim: int,
    ):
        if torch.allclose(ref, out, atol=1e-2, rtol=1e-2):
            return
        diff = (out - ref).abs()
        flat_idx = int(diff.argmax().item())
        hidden_size = ref.shape[-1]
        token_idx = flat_idx // hidden_size
        hidden_idx = flat_idx % hidden_size
        head_idx = hidden_idx // head_dim
        dim_idx = hidden_idx % head_dim
        self.fail(
            f"{label} mismatch: max_abs_diff={diff.flatten()[flat_idx].item()} "
            f"at token={token_idx}, head={head_idx}, dim={dim_idx}, "
            f"out={out.flatten()[flat_idx].item()}, ref={ref.flatten()[flat_idx].item()}"
        )

    def test_fused_qk_norm_rope(self):
        cases = itertools.product(
            (64, 128, 256),
            ("partial", "full"),
            (True, False),
        )
        for size_per_head, rotary_mode, is_neox_style in cases:
            rotary_dim = (
                size_per_head // 2 if rotary_mode == "partial" else size_per_head
            )
            with self.subTest(
                size_per_head=size_per_head,
                rotary_dim=rotary_dim,
                is_neox_style=is_neox_style,
            ):
                torch.manual_seed(1)
                dtype = torch.bfloat16
                num_tokens = 83
                head_num = 40
                kv_head_num = 8
                hidden_size = (head_num + 2 * kv_head_num) * size_per_head
                q_size = head_num * size_per_head
                kv_size = kv_head_num * size_per_head

                q_weight = torch.randn(size_per_head, dtype=dtype) * 0.25
                k_weight = torch.randn(size_per_head, dtype=dtype) * 0.25
                qkv = torch.randn(num_tokens, hidden_size, dtype=dtype)
                position_ids = torch.arange(
                    num_tokens, dtype=torch.int32, device=qkv.device
                )

                qkrmsnorm = QKRMSNorm(
                    q_weight, k_weight, head_num, kv_head_num, size_per_head
                )
                ref = qkrmsnorm(qkv.clone())
                q_ref, k_ref, v_ref = ref.split([q_size, kv_size, kv_size], dim=-1)
                q_ref = q_ref.reshape(num_tokens, head_num, size_per_head)
                k_ref = k_ref.reshape(num_tokens, kv_head_num, size_per_head)
                q_ref = self._apply_base_rope(
                    q_ref, position_ids, rotary_dim, is_neox_style
                )
                k_ref = self._apply_base_rope(
                    k_ref, position_ids, rotary_dim, is_neox_style
                )
                ref = torch.cat(
                    [
                        q_ref.reshape(num_tokens, q_size),
                        k_ref.reshape(num_tokens, kv_size),
                        v_ref,
                    ],
                    dim=-1,
                )

                case_label = (
                    f"head_dim={size_per_head}, rotary_dim={rotary_dim}, "
                    f"is_neox_style={is_neox_style}"
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
                    is_neox_style,
                    1.0,
                    rotary_dim,
                )
                self._assert_close_with_rope_debug(
                    ref, out, f"fused_qk_norm_rope {case_label}", size_per_head
                )

                rope_cache = self._rope_cache(position_ids, rotary_dim)
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
                    is_neox_style,
                    rotary_dim,
                )
                self._assert_close_with_rope_debug(
                    ref,
                    out_cache,
                    f"fused_qk_norm_rope_with_cache {case_label}",
                    size_per_head,
                )

    def test_fused_qk_norm_rope_rejects_empty_rotary_lanes(self):
        torch.manual_seed(2)
        dtype = torch.bfloat16
        num_tokens = 4
        head_num = 2
        kv_head_num = 1
        size_per_head = 64
        hidden_size = (head_num + 2 * kv_head_num) * size_per_head
        qkv = torch.randn(num_tokens, hidden_size, dtype=dtype)
        q_weight = torch.randn(size_per_head, dtype=dtype)
        k_weight = torch.randn(size_per_head, dtype=dtype)
        position_ids = torch.arange(num_tokens, dtype=torch.int32, device=qkv.device)

        with self.assertRaisesRegex(RuntimeError, "rotary_lanes must be positive"):
            rtp_llm_ops.fused_qk_norm_rope(
                qkv.clone(),
                q_weight,
                k_weight,
                position_ids,
                1e-6,
                head_num,
                kv_head_num,
                kv_head_num,
                size_per_head,
                10000.0,
                False,
                1.0,
                0,
            )

        rope_cache = torch.empty(num_tokens, 0, dtype=torch.float32, device=qkv.device)
        with self.assertRaisesRegex(RuntimeError, "rotary_lanes must be positive"):
            rtp_llm_ops.fused_qk_norm_rope_with_cache(
                qkv.clone(),
                q_weight,
                k_weight,
                position_ids,
                rope_cache,
                1e-6,
                head_num,
                kv_head_num,
                kv_head_num,
                size_per_head,
                False,
                0,
            )


if __name__ == "__main__":
    main()
