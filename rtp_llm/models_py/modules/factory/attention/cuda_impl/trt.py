import os
from typing import Optional

import torch

from rtp_llm.models_py.modules.factory.attention import common
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.ops import (
    AttentionConfigs,
    FMHAType,
    ParallelismConfig,
    RopeStyle,
    check_rope_cache,
    get_rope_cache_once,
)
from rtp_llm.ops.compute_ops import (
    FusedRopeKVCachePrefillOpQKVOut,
    FusedRopeKVCachePrefillOpQOut,
    LayerKVCache,
    PyAttentionInputs,
    TRTAttnOp,
    TRTPagedAttnOp,
    cuda_graph_copy_large2small,
    cuda_graph_copy_small2large,
    rtp_llm_ops,
)


class TRTMHAImpl(FMHAImplBase):

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        # Create implementations
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.fmha_impl = TRTAttnOp(attn_configs)
        self.rope_kvcache_impl = FusedRopeKVCachePrefillOpQKVOut(attn_configs)
        self.attn_configs = attn_configs

        # Store input info
        self.attn_inputs = attn_inputs
        self.input_lengths = attn_inputs.input_lengths
        self.cu_seq_lens = attn_inputs.cu_seqlens

        # Only TRTMHAImpl uses prefill_cuda_graph_copy_params
        self.prefill_cuda_graph_copy_params = attn_inputs.prefill_cuda_graph_copy_params

        # Create params
        self.fmha_params = self.fmha_impl.prepare(attn_inputs)
        self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)
        self._skip_rope_once = False

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        # Create temporary instance to check support
        fmha_impl = TRTAttnOp(attn_configs)
        return fmha_impl.support(attn_inputs)

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_idx: Optional[int] = 0,
    ) -> torch.Tensor:
        # Apply RoPE and KV Cache processing
        if self.need_rope_kv_cache:
            skip_rope = self._skip_rope_once
            self._skip_rope_once = False
            padding_offset = getattr(self.rope_params, "padding_offset", None)
            bypass_noop_rope = (
                skip_rope
                and kv_cache is None
                and (padding_offset is None or padding_offset.numel() == 0)
                and os.environ.get("IDLE_FISH_ENABLE_FUSED_QK_NORM_ROPE_BYPASS", "0")
                == "1"
            )
            if bypass_noop_rope:
                fmha_input = qkv
            else:
                fmha_input = self.rope_kvcache_impl.forward(
                    qkv, kv_cache, self.rope_params, skip_rope=skip_rope
                )
        else:
            fmha_input = qkv

        # Apply write cache store if needed
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        # CUDA graph copy logic specific to TRTMHAImpl
        if self.prefill_cuda_graph_copy_params:
            # Infer qkv_dim from fmha_input tensor shape
            qkv_dim = fmha_input.shape[1]
            total_len = (
                self.prefill_cuda_graph_copy_params.max_seq_len
                * self.prefill_cuda_graph_copy_params.max_batch_size
            )
            aligned_attn_buf = torch.zeros(
                (total_len, qkv_dim),
                dtype=fmha_input.dtype,
                device=fmha_input.device,
            )

            cuda_graph_copy_small2large(
                fmha_input,
                aligned_attn_buf,
                self.prefill_cuda_graph_copy_params.cuda_graph_prefill_batch_size,
                self.prefill_cuda_graph_copy_params.max_batch_size,
                self.prefill_cuda_graph_copy_params.max_seq_len,
                self.input_lengths,
                qkv_dim,
                self.cu_seq_lens,
            )
            fmha_input = aligned_attn_buf

        # Execute FMHA forward
        res = self.fmha_impl.forward(fmha_input, kv_cache, self.fmha_params)
        if self.prefill_cuda_graph_copy_params:
            # Infer hidden_size from res tensor shape
            hidden_size = res.shape[1]
            compact_attn_buf = torch.zeros(
                (qkv.shape[0], hidden_size), dtype=res.dtype, device=res.device
            )
            cuda_graph_copy_large2small(
                res,
                compact_attn_buf,
                self.prefill_cuda_graph_copy_params.cuda_graph_prefill_batch_size,
                self.prefill_cuda_graph_copy_params.max_batch_size,
                self.prefill_cuda_graph_copy_params.max_seq_len,
                self.input_lengths,
                hidden_size,
                self.cu_seq_lens,
            )

            res = compact_attn_buf
        return res

    def apply_fused_qk_norm_rope(
        self,
        qkv: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        eps: float,
    ) -> bool:
        if not self.need_rope_kv_cache:
            return False
        if self.prefill_cuda_graph_copy_params:
            return False
        position_ids = getattr(self.rope_params, "position_ids", None)
        if position_ids is None:
            return False

        rope_config = self.attn_configs.rope_config
        if rope_config.style != RopeStyle.Base:
            return False
        rotary_dim = rope_config.dim or self.attn_configs.size_per_head
        if rotary_dim != self.attn_configs.size_per_head:
            return False
        if (
            qkv.dtype != torch.bfloat16
            or q_weight.dtype != torch.bfloat16
            or k_weight.dtype != torch.bfloat16
        ):
            return False

        if os.environ.get("IDLE_FISH_ENABLE_FUSED_QK_NORM_ROPE_CACHE", "0") == "1":
            rope_cache = get_rope_cache_once(rope_config, self.attn_configs.max_seq_len)
            if (
                check_rope_cache(rope_config, rope_cache)
                and rope_cache.data.is_cuda
                and rope_cache.data.dtype == torch.float32
                and rope_cache.data.dim() == 2
                and rope_cache.data.shape[1] == rotary_dim
                and rope_cache.data.is_contiguous()
            ):
                rtp_llm_ops.fused_qk_norm_rope_with_cache(
                    qkv,
                    q_weight,
                    k_weight,
                    position_ids,
                    rope_cache.data,
                    eps,
                    self.attn_configs.head_num,
                    self.attn_configs.kv_head_num,
                    self.attn_configs.kv_head_num,
                    self.attn_configs.size_per_head,
                    bool(rope_config.is_neox_style),
                    int(rotary_dim),
                )
                self._skip_rope_once = True
                return True

        rtp_llm_ops.fused_qk_norm_rope(
            qkv,
            q_weight,
            k_weight,
            position_ids,
            eps,
            self.attn_configs.head_num,
            self.attn_configs.kv_head_num,
            self.attn_configs.kv_head_num,
            self.attn_configs.size_per_head,
            float(rope_config.base),
            bool(rope_config.is_neox_style),
            float(rope_config.scale),
            int(rotary_dim),
        )
        self._skip_rope_once = True
        return True

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs):
        pass


class TRTPagedMHAImpl(FMHAImplBase):

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        # Create implementations
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.fmha_impl = TRTPagedAttnOp(attn_configs)
        self.rope_kvcache_impl = FusedRopeKVCachePrefillOpQOut(attn_configs)
        self.attn_configs = attn_configs

        # Store input info
        self.attn_inputs = attn_inputs
        self.input_lengths = attn_inputs.input_lengths
        self.cu_seq_lens = attn_inputs.cu_seqlens

        # Create params
        self.fmha_params = self.fmha_impl.prepare(attn_inputs)
        self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        # Create temporary instance to check support
        fmha_impl = TRTPagedAttnOp(attn_configs)
        return fmha_impl.support(attn_inputs)

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_idx: int,
    ) -> torch.Tensor:
        # Apply RoPE and KV Cache processing
        if self.need_rope_kv_cache:
            fmha_input = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)
        else:
            fmha_input = qkv

        # Apply write cache store if needed
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        # Execute FMHA forward
        return self.fmha_impl.forward(fmha_input, kv_cache, self.fmha_params)

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs):
        if not attn_inputs.is_prefill and (
            attn_inputs.prefix_lengths is None
            or attn_inputs.prefix_lengths.numel() == 0
        ):
            attn_inputs.prefix_lengths = torch.zeros_like(
                attn_inputs.input_lengths, device=attn_inputs.input_lengths.device
            )
        common.update_trt_params(
            self.fmha_impl,
            self.rope_kvcache_impl,
            self.fmha_params,
            self.rope_params,
            attn_inputs,
        )
