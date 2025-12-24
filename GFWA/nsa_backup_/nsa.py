import torch
from typing import Optional, Tuple, Union
from torch.nn.attention.flex_attention import create_block_mask
# from flash_attn import flash_attn_func
from .. import flash_attention as gfwa_flash_attention
import os
import time
from pathlib import Path

from fla.ops.utils.pooling import mean_pooling
from fla.ops.nsa.parallel import parallel_nsa_topk

from .compression import compression_attention
from .selection import selection_attention


def _prepare_log_fgate_for_gfwa(
    log_fgate: Optional[torch.Tensor],
    *,
    batch_size: int,
    seq_len: int,
    num_heads: int,
) -> Optional[torch.Tensor]:
    """Format log-domain forgetting gate for GFWA attention.

    Accepts either `[B, T, H]` or `[B, H, T]` layouts and returns `[B, T, H]`
    in float32 as expected by the high-level GFWA helper (with `head_first=False`).
    """
    if log_fgate is None:
        return None
    if log_fgate.ndim != 3:
        raise ValueError(f"log_fgate must be 3D, got shape {tuple(log_fgate.shape)}")

    if log_fgate.shape == (batch_size, seq_len, num_heads):
        formatted = log_fgate
    elif log_fgate.shape == (batch_size, num_heads, seq_len):
        formatted = log_fgate.transpose(1, 2).contiguous()
    else:
        raise ValueError(
            f"Unexpected log_fgate shape {tuple(log_fgate.shape)}; expected "
            f"[B, T, H]={batch_size, seq_len, num_heads} or "
            f"[B, H, T]={batch_size, num_heads, seq_len}"
        )
    return formatted.to(torch.float32)


def _sliding_window_attention_with_gfwa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    window_size: int,
    scale: float,
    log_fgate: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute causal sliding-window attention using GFWA FlashAttention."""
    B, M, H, _ = q.shape
    log_fgate_formatted = _prepare_log_fgate_for_gfwa(
        log_fgate,
        batch_size=B,
        seq_len=M,
        num_heads=H,
    )
    return gfwa_flash_attention(
        q,
        k,
        v,
        log_fgate=log_fgate_formatted,
        head_first=False,
        sm_scale=scale,
        window_size=window_size,
    )


def nsa_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_cmp: Optional[torch.Tensor] = None,
    g_slc: Optional[torch.Tensor] = None,
    g_swa: Optional[torch.Tensor] = None,
    block_count: int = 16,
    block_size: int = 64,
    window_size: int = 0,
    scale: Optional[float] = None,
    return_attn_weights: bool = False,
    layer_idx: Optional[int] = None,
    log_fgate: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
    # Declare global variables at the beginning of function
    global _layer_timings, _timing_stats

    B, M, H, D = q.shape
    _, N, G, _ = k.shape

    assert g_cmp is not None and g_slc is not None and g_swa is not None, "g_cmp, g_slc, and g_swa are required"
    assert k.shape == (B, N, G, D), f"k shape: {k.shape} must be ({B}, {N}, {G}, {D})"
    assert v.shape == (B, N, G, D), f"v shape: {v.shape} must be ({B}, {N}, {G}, {D})"
    assert g_cmp.shape == (B, M, H), f"g_cmp shape: {g_cmp.shape} must be ({B}, {M}, {H})"
    assert g_slc.shape == (B, M, H), f"g_slc shape: {g_slc.shape} must be ({B}, {M}, {H})"
    assert g_swa.shape == (B, M, H), f"g_swa shape: {g_swa.shape} must be ({B}, {M}, {H})"

    if scale is None:
        scale = D ** -0.5

    k_cmp, v_cmp = mean_pooling(k, block_size), mean_pooling(v, block_size)

    # Ensure compressed tensors are on the same device as q
    k_cmp = k_cmp.to(q.device)
    v_cmp = v_cmp.to(q.device)

    N_block = k_cmp.shape[1]


    # Causal compression mask: only allow fully past blocks.
    def cmp_mask(b, h, q_idx, kv_idx):
        return q_idx >= (kv_idx + 1) * block_size

    block_mask = create_block_mask(cmp_mask, B, H, M, N_block)
    # Ensure block_mask is on the same device as the input tensors
    if hasattr(block_mask, 'to'):
        block_mask = block_mask.to(q.device)

    o_cmp, lse_cmp = compression_attention(q, k_cmp, v_cmp, block_mask)

    # Selection branch timing
    block_indices = parallel_nsa_topk(
        q=q,
        k=k_cmp,
        lse=lse_cmp,
        block_counts=block_count,
        block_size=block_size,
        scale=scale,
        # cu_seqlens=None
    )

    if block_indices.dtype.is_floating_point:
        raise RuntimeError("parallel_nsa_topk returned floating point indices")
    block_indices = block_indices.clamp_min_(0)

    if return_attn_weights:
        o_slc, lse_slc = selection_attention(
            q, k, v, block_indices, block_count, block_size, scale,
            return_attn_probs=True
        )
    else:
        o_slc = selection_attention(
            q, k, v, block_indices, block_count, block_size, scale
        )

    # Sliding window branch timing
    # o_swd = flash_attn_func(
    #     q, k, v,
    #     causal=True,
    #     window_size=(window_size-1, 0)
    # )
    o = o_cmp * g_cmp.unsqueeze(-1) + o_slc * g_slc.unsqueeze(-1)
    if window_size > 0:
        n_head = H
        n_kv_head = G
        num_key_value_groups = n_head // n_kv_head
        # o_swd = _sliding_window_attention_with_gfwa(
        #     q,
        #     k,
        #     v,
        #     window_size=window_size,
        #     scale=scale,
        #     log_fgate=log_fgate,
        # )
        o_swd = _sliding_window_attention_with_gfwa(
            q,
            k.repeat_interleave(num_key_value_groups, dim=2).contiguous(),
            v.repeat_interleave(num_key_value_groups, dim=2).contiguous(),
            window_size=window_size,
            scale=scale,
            log_fgate=log_fgate,
        )
        o = o + o_swd * g_swa.unsqueeze(-1)


    if return_attn_weights:
        attn_weights = {
            'compression': {
                'lse': lse_cmp,  # [B, H, M] - log-sum-exp values
                'block_indices': block_indices,  # [B, M, G, T] - selected block indices
            },
            'selection': {
                'lse': lse_slc,  # [B, H, M] - log-sum-exp values
                'block_indices': block_indices,  # [B, M, G, T] - selected block indices
            },
            'sliding_window': {
                'note': 'Flash attention does not return attention weights directly'
            },
            'gating_weights': {
                'g_cmp': g_cmp,  # [B, M, H] - compression gating weights
                'g_slc': g_slc,  # [B, M, H] - selection gating weights
                'g_swa': g_swa,  # [B, M, H] - sliding window gating weights
            },
            # 'timing_info': timing_info  # Add timing information to weights dict
        }
        return o, attn_weights
    else:
        return o
