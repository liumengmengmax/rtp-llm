"""
Triton fused MLP + L2Norm kernel for idle-fish embedding output head.

"""

import torch
import triton
import triton.language as tl


@triton.jit
def _mlp_l2norm_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    B,
    K,
    N,
    stride_ib,
    stride_ik,
    stride_wn,
    stride_wk,
    stride_ob,
    stride_on,
    eps: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    OUTPUT_BF16: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_M

    m_offsets = row_start + tl.arange(0, BLOCK_M)
    n_offsets = tl.arange(0, BLOCK_N)
    m_mask = m_offsets < B
    n_mask = n_offsets < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        a = tl.load(
            input_ptr + m_offsets[:, None] * stride_ib + k_offsets[None, :] * stride_ik,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        b = tl.load(
            weight_ptr
            + n_offsets[None, :] * stride_wn
            + k_offsets[:, None] * stride_wk,
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        )
        acc += tl.dot(a, b)

    bias = tl.load(bias_ptr + n_offsets, mask=n_mask, other=0.0)
    acc += bias[None, :]

    sq = tl.where(n_mask[None, :], acc * acc, 0.0)
    sq_sum = tl.sum(sq, axis=1)
    inv_norm = tl.rsqrt(sq_sum + eps)
    result = acc * inv_norm[:, None]

    out_dtype = tl.bfloat16 if OUTPUT_BF16 else tl.float16
    out_ptrs = (
        output_ptr + m_offsets[:, None] * stride_ob + n_offsets[None, :] * stride_on
    )
    tl.store(out_ptrs, result.to(out_dtype), mask=m_mask[:, None] & n_mask[None, :])


def _fallback_linear_l2norm(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Fallback path: F.linear + L2 normalize (no fusion, works for any N)."""
    out = torch.nn.functional.linear(input, weight, bias)
    return torch.nn.functional.normalize(out, p=2, dim=-1, eps=eps)


def triton_idlefish_mlp_l2norm(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    assert bias is not None, "triton_idlefish_mlp_l2norm requires bias (got None)"
    B, K = input.shape
    N = weight.shape[0]
    assert weight.shape[1] == K, f"weight shape {weight.shape} vs input K={K}"
    assert bias.shape[0] == N, f"bias shape {bias.shape} vs N={N}"

    # N > 256 causes register spill / OutOfResources on most GPUs;
    # fallback to unfused path which has no dimension constraint.
    if N > 256:
        return _fallback_linear_l2norm(input, weight, bias, eps)

    output = torch.empty((B, N), dtype=input.dtype, device=input.device)

    BLOCK_N = triton.next_power_of_2(N)
    BLOCK_K = 64
    BLOCK_M = 16

    grid = (triton.cdiv(B, BLOCK_M),)
    _mlp_l2norm_kernel[grid](
        input,
        weight,
        bias,
        output,
        B,
        K,
        N,
        input.stride(0),
        input.stride(1),
        weight.stride(0),
        weight.stride(1),
        output.stride(0),
        output.stride(1),
        eps=eps,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        OUTPUT_BF16=(input.dtype == torch.bfloat16),
    )
    return output
