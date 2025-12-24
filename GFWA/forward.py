import torch
import triton
import triton.language as tl

# NOTE: this function can be overwritten at runtime to use your custom config
def get_fwd_config(B, H, M, N, D, causal):
    assert causal
    if torch.cuda.get_device_capability() == (8, 0):
        if D <= 64:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 32, 3, 4
        else:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 4, 4
    elif torch.cuda.get_device_capability() == (9, 0):
        # H100
        if D <= 64:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 3, 8
        else:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 128, 2, 8
    elif torch.cuda.get_device_capability() == (8, 6):
        if not causal:
            if D <= 64:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 3, 4
            else:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 2, 4
        else: # causal
            if D <= 64:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 3, 4
            else:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 2, 4
    elif torch.cuda.get_device_capability() == (8, 9):
        # L40S
        if D <= 64:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 2, 4
        else:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 2, 4
    else:
        BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 2, 4
    return (BLOCK_M, BLOCK_N, num_stages, num_warps)


@triton.jit
def _fwd_kernel(
    Q, K, V, LOG_LAMBDA, SEQ_START, sm_scale,
    L, O,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_log_lambda_z, stride_log_lambda_h, stride_log_lambda_n,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, M, N, P_SEQ,
    num_groups,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr, LARGER_M: tl.constexpr, HAS_SEQ_START: tl.constexpr,
    HAS_SLIDING_WINDOW: tl.constexpr, WINDOW_SIZE: tl.constexpr,
    DIVISIBLE_M: tl.constexpr, DIVISIBLE_N: tl.constexpr,
):
    input_dtype = Q.dtype.element_ty
    # -- grid id --
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)

    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    log2e: tl.constexpr = 1.4426950408889634
    loge2: tl.constexpr = 0.6931471805599453
    qk_scale = sm_scale * log2e

    # offset pointers for (batch, head)
    off_hk = off_h // num_groups
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_hk * stride_kh
    V += off_z * stride_vz + off_hk * stride_vh
    LOG_LAMBDA += off_z * stride_log_lambda_z + off_h * stride_log_lambda_h
    O += off_z * stride_oz + off_h * stride_oh
    L += (off_z * H + off_h) * M # l's shape is (B, H, M)

    offs_m_base = tl.arange(0, BLOCK_M)
    offs_m = start_m * BLOCK_M + offs_m_base
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_DMODEL)


    # initialize pointers to value-like data
    q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk) # (BLOCK_M, BLOCK_DMODEL)
    log_lambda_out_ptrs = LOG_LAMBDA + (P_SEQ + offs_m) * stride_log_lambda_n
    o_ptrs = O + (offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok) # (BLOCK_M, BLOCK_DMODEL)
    l_ptrs = L + offs_m

    # initialize pointer to m and l, fp32 for accumulators
    m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # load q
    if DIVISIBLE_M:
        q = tl.load(q_ptrs, cache_modifier=".cg")
        log_lambda_out = tl.load(log_lambda_out_ptrs, cache_modifier=".cg")
    else:
        mask_m = offs_m < M
        q = tl.load(q_ptrs, mask=mask_m[:, None], cache_modifier=".cg")
        log_lambda_out = tl.load(log_lambda_out_ptrs, mask=mask_m, cache_modifier=".cg")

    # NOTE: Loop-Bound-For-N
    # The indices in m-dimension that this block may access is in `[start_m * BLOCK_M, (start_m + 1) * BLOCK_M)`.
    # According to the rule of causal masking, the max index in n-dimension that this block may access
    # is `P_SEQ + (start_m + 1) * BLOCK_M`, but it should never exceed `N` (the K/V seqlen).
    # See also https://github.com/FlagOpen/FlagAttention/pull/8
    if IS_CAUSAL:
        # Use conservative upper bound to prevent warp divergence
        hi_causal = P_SEQ + (start_m + 1) * BLOCK_M
        if LARGER_M:
            hi_causal = tl.maximum(0, hi_causal)
        hi = tl.minimum(N, hi_causal)
    else:
        hi = N

    # Base lower bound
    lo = 0
    seq_start = 0
    if HAS_SEQ_START:
        SEQ_START += off_z
        seq_start = tl.load(SEQ_START)
        # Use conservative lower bound to prevent thread divergence
        lo = tl.minimum(seq_start, hi)

    # Tighten bounds for sliding window by shrinking the K/V range for this query block.
    # For the query block covering q in [q_block_start, q_block_end], the union of valid keys is
    # [max(0, q_block_start - WINDOW_SIZE + 1), q_block_end]. We also restrict hi to q_block_end + 1
    # to avoid iterating tiles that will be fully masked by the window.
    if HAS_SLIDING_WINDOW:
        q_block_start = P_SEQ + start_m * BLOCK_M
        # Use conservative block end to ensure uniform bounds across threads in the warp
        q_block_end = P_SEQ + (start_m + 1) * BLOCK_M - 1
        lo_sw = tl.maximum(0, q_block_start - WINDOW_SIZE + 1)
        lo = tl.maximum(lo, lo_sw)
        hi = tl.minimum(hi, q_block_end + 1)

    # Align to BLOCK_N for tiled loads
    lo = (lo // BLOCK_N) * BLOCK_N
    offs_n_init = offs_n_base + lo

    # loop over k, v and update accumulators
    k_ptrs = K + (offs_k[:, None] * stride_kk + offs_n_init[None, :] * stride_kn) # (BLOCK_DMODEL, BLOCK_N)
    v_ptrs = V + (offs_n_init[:, None] * stride_vn + offs_k[None, :] * stride_vk) # (BLOCK_N, BLOCK_DMODEL)
    log_lambda_in_ptrs = LOG_LAMBDA + (offs_n_init * stride_log_lambda_n) # (BLOCK_N, BLOCK_DMODEL)
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + offs_n_base

        # -- load k, v --
        if DIVISIBLE_N:
            k = tl.load(k_ptrs, cache_modifier=".cg")
            v = tl.load(v_ptrs, cache_modifier=".cg")
            log_lambda_in = tl.load(log_lambda_in_ptrs, cache_modifier=".cg")
        else:
            mask_n = offs_n < N
            k = tl.load(k_ptrs, mask=mask_n[None, :], cache_modifier=".cg")
            v = tl.load(v_ptrs, mask=mask_n[:, None], cache_modifier=".cg")
            log_lambda_in = tl.load(log_lambda_in_ptrs, mask=mask_n, cache_modifier=".cg")

        # -- compute qk ---
        # s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s = tl.dot(q, k, input_precision="ieee") * qk_scale

        # Prepare masks and gate the forgetting bias inside the valid region
        if not DIVISIBLE_N:
            valid_mask = (offs_n[None, :] < N)
        else:
            valid_mask = tl.full([BLOCK_M, BLOCK_N], True, tl.int1)

        if IS_CAUSAL:
            causal_mask = (P_SEQ + offs_m[:, None]) >= offs_n[None, :]
            valid_mask = valid_mask & causal_mask
        if HAS_SEQ_START:
            seq_mask = offs_n[None, :] >= seq_start
            valid_mask = valid_mask & seq_mask

        decay_bias = (log_lambda_out[:, None] - log_lambda_in[None, :]) * log2e
        if HAS_SLIDING_WINDOW:
            # Apply sliding window mask: query at position q attends to keys [q - WINDOW_SIZE + 1, q]
            query_pos = P_SEQ + offs_m[:, None]  # [BLOCK_M, 1]
            key_pos = offs_n[None, :]            # [1, BLOCK_N]
            window_mask = (key_pos >= (query_pos - WINDOW_SIZE + 1)) & (key_pos <= query_pos)
            valid_mask = valid_mask & window_mask

            # Add forgetting gate ONLY inside the window (and other masks); -inf elsewhere
            s = tl.where(valid_mask, s + decay_bias, float("-inf"))
        else:
            # No sliding window: standard forgetting bias everywhere, then mask
            s = s + decay_bias
            if not DIVISIBLE_N:
                s = tl.where(offs_n[None, :] < N, s, float("-inf"))
            if IS_CAUSAL:
                s = tl.where(causal_mask, s, float("-inf"))
            if HAS_SEQ_START:
                s = tl.where(seq_mask, s, float("-inf"))


        # -- compute scaling constant ---
        s_max = tl.max(s, 1)
        m_i_new = tl.maximum(m_i, s_max)

        if HAS_SLIDING_WINDOW:
            # Special handling when s_max is -inf (all attention scores masked)
            # Only needed for sliding-window where entire tiles can be invalid
            # Use uniform conditional logic to prevent warp divergence
            all_masked = s_max == -float("inf")
            # Ensure all threads follow the same execution path
            alpha = tl.math.exp2((m_i - m_i_new))
            p_raw = tl.math.exp2(s - m_i_new[:, None])
            p = tl.where(all_masked[:, None], 0.0, p_raw)
            p_sum = tl.sum(p, 1)
            # Use conditional assignments that don't cause divergence
            acc_scaled = acc * alpha[:, None]
            acc = tl.where(all_masked[:, None], acc, acc_scaled)
            acc += tl.dot(p.to(input_dtype), v, input_precision="ieee")
            l_i = tl.where(all_masked, l_i, l_i * alpha + p_sum)
            m_i = tl.where(all_masked, m_i, m_i_new)
        else:
            # Match forgetting_attention.py exactly for non-window case
            alpha = tl.math.exp2(m_i - m_i_new)
            p = tl.math.exp2(s - m_i_new[:, None])
            p_sum = tl.sum(p, 1)
            acc = acc * alpha[:, None]
            acc += tl.dot(p.to(input_dtype), v, input_precision="ieee")
            l_i = l_i * alpha + p_sum
            m_i = m_i_new
        # update pointers
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn
        log_lambda_in_ptrs += BLOCK_N * stride_log_lambda_n

    # write back l & o
    if IS_CAUSAL and (LARGER_M or HAS_SEQ_START):
        is_empty_line = (offs_m + P_SEQ) < seq_start
        acc = tl.where(is_empty_line[:, None], 0.0, acc * (1.0 / l_i[:, None]))
        l = tl.where(is_empty_line, float("-inf"), m_i * loge2 + tl.log(l_i))
    else:
        # Match forgetting_attention.py exactly when no sliding window
        if HAS_SLIDING_WINDOW:
            # For sliding window, some queries might have l_i = 0 if they had no valid attention
            empty_attention = l_i == 0.0
            l_i_safe = tl.where(empty_attention, 1.0, l_i)
            acc = tl.where(empty_attention[:, None], 0.0, acc * (1.0 / l_i_safe[:, None]))
            l = tl.where(empty_attention, float("-inf"), m_i * loge2 + tl.log(l_i))
        else:
            acc = acc * (1.0 / l_i[:, None])
            l = m_i * loge2 + tl.log(l_i)


    if DIVISIBLE_M:
        tl.store(l_ptrs, l, cache_modifier=".cg")
        tl.store(o_ptrs, acc.to(input_dtype), cache_modifier=".cg")
    else:
        mask_m = offs_m < M
        tl.store(l_ptrs, l, mask=mask_m, cache_modifier=".cg")
        tl.store(o_ptrs, acc.to(input_dtype), mask=mask_m[:, None], cache_modifier=".cg")
        
