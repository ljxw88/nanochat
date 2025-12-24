import math
import torch
import triton
import triton.language as tl
from einops import rearrange
from typing import Optional
try:
    from .forward import _fwd_kernel, get_fwd_config
    from .backward import (
        _bwd_preprocess, _bwd_kv_kernel, _bwd_q_kernel, get_bwd_config,
        get_bwd_kv_config, get_bwd_q_config
    )
except ImportError:
    # Fallback for standalone usage
    from forward import _fwd_kernel, get_fwd_config
    from backward import (
        _bwd_preprocess, _bwd_kv_kernel, _bwd_q_kernel, get_bwd_config,
        get_bwd_kv_config, get_bwd_q_config
    )

__all__ = ["flash_attention"]


# File flash.py
def maybe_contiguous(x):
    # only when the inner most dimension is contiguous can LDGSTS be used
    # so inner-dimension contiguity is enforced.
    return x.contiguous() if x.stride(-1) != 1 else x

def rounded_multiple(a, b):
    return (a + b - 1) // b * b

# --------------------------- public API ---------------------------
class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, log_fgate, seq_start, causal, sm_scale, return_log_normalizer, window_size=None):
        assert causal, "Only causal attention is supported"
        Dq, Dk, Dv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Dq == Dk == Dv, "feature size of q, k, v should be equal"
        assert Dk in {16, 32, 64, 128}, "We only support head dims in {16, 32, 64, 128}"

        B, H, M, D = q.shape
        if seq_start is not None:
            has_seq_start = True
            assert seq_start.shape == (B,)
        else:
            has_seq_start = False
            seq_start = torch.zeros((B,), device=q.device, dtype=torch.long)
        N = k.shape[2]
        
        # Handle optional log_fgate for pure causal mode
        if log_fgate is not None:
            assert log_fgate.shape == (B, H, N)
            log_fgate = log_fgate.float()
            if has_seq_start:
                log_fgate = log_fgate.clone()
                # We absolutely don't want masked value to affect result. If we
                # don't do this then it could via affecting numerical precision of
                # cumsum
                mask_index = (torch.arange(N, device=q.device)[None, None, :] < seq_start[:, None, None])
                mask_index = torch.broadcast_to(mask_index, log_fgate.size())
                log_fgate[mask_index] = 0.0
            log_lambda = torch.cumsum(log_fgate, dim=-1, dtype=log_fgate.dtype).float()
        else:
            # Pure causal mode: no forgetting, so log_lambda is all zeros
            log_lambda = torch.zeros((B, H, N), device=q.device, dtype=torch.float32)

        Hk, Hv = k.shape[1], v.shape[1]
        assert Hk == Hv, "num of heads in k and v should be equal"
        assert H == Hk, "groupped query attention has not been tested. You can uncomment this if you know what you are doing."
        assert H % Hk == 0, "number of heads in q must be a multiple of that in k & v"
        num_groups = H // Hk

        P_SEQ = N - M
        larger_m = M > N
        assert (not larger_m), "The key/value tensors must be longer than the query tensor"

        if sm_scale is None:
            sm_scale = 1. / math.sqrt(D)

        # contiguity
        q, k, v = maybe_contiguous(q), maybe_contiguous(k), maybe_contiguous(v)

        # to work around https://github.com/openai/triton/issues/2441
        device = torch.cuda.device_of(q)

        with torch.cuda.device(device):

            config = get_fwd_config(B, H, M, N, D, causal)
            BLOCK_M, BLOCK_N, num_stages, num_warps = config
            
            # Sliding window configuration
            has_sliding_window = window_size is not None
            if has_sliding_window:
                assert window_size > 0, "Window size must be positive"
            else:
                window_size = 0

            divisible_m = M % BLOCK_M == 0
            divisible_n = N % BLOCK_N == 0
            # consider using 3d grid to avoid div & rem
            grid = (triton.cdiv(M, BLOCK_M), H, B)
            o = torch.empty_like(q)
            L = torch.empty((B, H, M), device=q.device, dtype=torch.float32)
            _fwd_kernel[grid](
                q, k, v, log_lambda, seq_start, sm_scale,
                L, o,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                log_lambda.stride(0), log_lambda.stride(1), log_lambda.stride(2),
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                B, H, M, N, P_SEQ, num_groups,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=D,
                IS_CAUSAL=causal, LARGER_M=larger_m, HAS_SEQ_START=has_seq_start,
                HAS_SLIDING_WINDOW=has_sliding_window, WINDOW_SIZE=window_size,
                DIVISIBLE_M=divisible_m, DIVISIBLE_N=divisible_n,
                num_warps=num_warps, num_stages=num_stages,
            )

        # autograd context maintenance
        ctx.save_for_backward(q, k, v, o, L, log_lambda, seq_start)
        ctx.sm_scale = sm_scale
        ctx.causal = causal
        ctx.has_seq_start = has_seq_start
        ctx.has_log_fgate = log_fgate is not None
        ctx.has_sliding_window = has_sliding_window
        ctx.window_size = window_size if has_sliding_window else 0

        has_extra_return = return_log_normalizer
        if has_extra_return:
            outs = (
                o,
                L if return_log_normalizer else None,
            )
            return outs
        return o

    @staticmethod
    def backward(ctx, do, *ignored):
        q, k, v, o, L, log_lambda, seq_start = ctx.saved_tensors
        sm_scale = ctx.sm_scale
        causal = ctx.causal
        has_seq_start = ctx.has_seq_start
        has_log_fgate = ctx.has_log_fgate
        has_sliding_window = ctx.has_sliding_window
        window_size = ctx.window_size

        B, H, M, D = q.shape
        N = k.shape[2]
        Hk = k.shape[1]
        num_groups = H // Hk
        P_SEQ = N - M
        larger_m = M > N

        if sm_scale is None:
            sm_scale = 1. / math.sqrt(D)

        # to work around https://github.com/openai/triton/issues/2441
        device = torch.cuda.device_of(q)
        with torch.cuda.device(device):
            config = get_bwd_config(B, H, M, N, D, causal)
            BLOCK_M, BLOCK_N, num_stages, num_warps = config

            divisible_m = M % BLOCK_M == 0
            divisible_n = N % BLOCK_N == 0

            delta = torch.empty_like(L)
            grid = (triton.cdiv(M, BLOCK_M), H, B)
            _bwd_preprocess[grid](
                o, do,
                delta,
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                do.stride(0), do.stride(1), do.stride(2), do.stride(3),
                delta.stride(0), delta.stride(1), delta.stride(2),
                M,
                BLOCK_M=BLOCK_M, D_HEAD=D,
                DIVISIBLE_M=divisible_m,
            )

            # NOTE that dk & dv always have the same number of heads as q, instead of q.
            BLOCK_M, BLOCK_N, num_stages, num_warps = get_bwd_kv_config(B, H, M, N, D, causal)
            divisible_m = M % BLOCK_M == 0
            divisible_n = N % BLOCK_N == 0

            dk = torch.empty((B, H, N, D), dtype=k.dtype, device=q.device)
            dv = torch.empty((B, H, N, D), dtype=v.dtype, device=q.device)
            # Initialize to zeros to avoid any chance of reading uninitialized memory
            dlog_lambda = torch.zeros((B, H, N), dtype=log_lambda.dtype, device=q.device)
            grid = (triton.cdiv(N, BLOCK_N), H, B)
            _bwd_kv_kernel[grid](
                q, k, v, log_lambda, seq_start, sm_scale, do,
                dk, dv, dlog_lambda,
                L, delta,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                log_lambda.stride(0), log_lambda.stride(1), log_lambda.stride(2),
                do.stride(0), do.stride(1), do.stride(2), do.stride(3),
                dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
                dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
                dlog_lambda.stride(0), dlog_lambda.stride(1), dlog_lambda.stride(2),
                B, H, M, N, P_SEQ,
                num_groups,
                BLOCK_M=BLOCK_M, BLOCK_DMODEL=D, BLOCK_N=BLOCK_N, CAUSAL=causal,
                HAS_SLIDING_WINDOW=has_sliding_window, WINDOW_SIZE=window_size,
                DIVISIBLE_M=divisible_m, DIVISIBLE_N=divisible_n, HAS_SEQ_START=has_seq_start,
                num_stages=num_stages, num_warps=num_warps,
            )

            BLOCK_M, BLOCK_N, num_stages, num_warps = get_bwd_q_config(B, H, M, N, D, causal)
            divisible_m = M % BLOCK_M == 0
            divisible_n = N % BLOCK_N == 0
            dq = torch.zeros_like(q)
            grid = (triton.cdiv(M, BLOCK_M), H, B)
            _bwd_q_kernel[grid](
                q, k, v, log_lambda, seq_start, sm_scale, do,
                dq, dlog_lambda,
                L, delta,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                log_lambda.stride(0), log_lambda.stride(1), log_lambda.stride(2),
                do.stride(0), do.stride(1), do.stride(2), do.stride(3),
                dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
                dlog_lambda.stride(0), dlog_lambda.stride(1), dlog_lambda.stride(2),
                B, H, M, N, P_SEQ,
                num_groups,
                BLOCK_M=BLOCK_M, BLOCK_DMODEL=D, BLOCK_N=BLOCK_N,
                CAUSAL=causal, LARGER_M=larger_m, HAS_SEQ_START=has_seq_start,
                HAS_SLIDING_WINDOW=has_sliding_window, WINDOW_SIZE=window_size,
                DIVISIBLE_M=divisible_m, DIVISIBLE_N=divisible_n,
                num_stages=num_stages, num_warps = num_warps,
            )
            dk = dk.reshape((B, Hk, num_groups, N, D)).sum(2)
            dv = dv.reshape((B, Hk, num_groups, N, D)).sum(2)
        
        # Compute gradient for log_fgate only if it was provided
        if has_log_fgate:
            dcumsum = torch.cumsum(dlog_lambda, dim=-1, dtype=log_lambda.dtype)
            dlog_fgate = dlog_lambda + dcumsum[..., -1:] - dcumsum
            dlog_fgate = dlog_fgate.float()
        else:
            dlog_fgate = None
            
        return dq, dk, dv, dlog_fgate, None, None, None, None, None


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_fgate: Optional[torch.Tensor] = None,
    *,
    head_first: bool = True,
    seq_start: Optional[torch.Tensor] = None,
    sm_scale: Optional[float] = None,
    window_size: Optional[int] = None,
):
    """
    A FlashAttention-based implementation of Forgetting Attention. 

    Note:
    - We recommand bfloat16/float16 for q, k, v and float32 for log_fgate. float32 for 
      q, k, v is also supported, but the kernel will not use tensor cores if q, k, v are
      in float32 (which would be slow).
    - We only support seqlen_q <= seqlen_k
    - We only support causal attention
    - Head dimension must be in one of {16, 32, 64, 128}

    Arguments:
        - q: (batch_size, seqlen_q, num_heads, head_dim) unless head_first=True.
        - k: (batch_size, seqlen_k, num_heads, head_dim) unless head_first=True.
        - v: (batch_size, seqlen_k, num_heads, head_dim) unless head_first=True.
        - log_fgate: Optional (batch_size, seqlen_k, num_heads) unless head_first=True. 
              This should be the **log** of the forget gates. This is typically the 
              output of torch.nn.functional.logsigmoid. If None, pure causal attention
              is performed (equivalent to standard Flash Attention).
        - head_first: if True, the order the num_heads and seqlen_* axis of the all 
              FloatTensor inputs and outputs should be (num_heads, seq_len_*) instead of
              (seq_len_*, num_heads)
        - seq_start: If not None, should be LongTensor with shape (batch_size,) 
              and range in [0, seq_len_k). For each batch index batch_id, no attention 
              will be allocated to tokens before the token index seq_start[batch_id]. 
              This is useful for left-padded inputs.
        - sm_scale: The scaling of attention scores before applying softmax. If
              None, it defaults to (1.0 / math.sqrt(head_dim))
        - window_size: Optional sliding window size. If provided, attention is
              limited to a sliding window around each query position. Compatible
              with both pure-causal and causal-forgetting modes.

    Returns:
        out (torch.Tensor): (batch_size, seqlen_q, num_heads, head_dim) unless head_first=True.
    """
    if not head_first:
        q, k, v = [rearrange(item, "b t h d -> b h t d") for item in (q, k, v)]
        if log_fgate is not None:
            log_fgate = rearrange(log_fgate, "b t h -> b h t")
    out = FlashAttention.apply(q, k, v, log_fgate, seq_start, True, sm_scale, False, window_size)
    if not head_first:
        out = rearrange(out, "b h t d -> b t h d")
    return out
