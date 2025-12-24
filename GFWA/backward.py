import torch
import triton
import triton.language as tl

# NOTE: this function can be overwritten at runtime to use your custom config
def get_bwd_config(B, H, M, N, D, causal):
    if torch.cuda.get_device_capability() == (9, 0):
        if not causal:
            BLOCK_M = 128 if D <= 64 else 64
            BLOCK_N = 64
            num_stages = 2
            num_warps = 4
        else:
            BLOCK_M = 64
            BLOCK_N = 64
            num_stages = 3 if D <= 64 else 2
            num_warps = 4
    elif torch.cuda.get_device_capability() == (8, 0):
        if not causal:
            BLOCK_M = 128 if D <= 64 else 64
            BLOCK_N = 64
            num_stages = 2
            num_warps = 4
        else:
            BLOCK_M = 64
            BLOCK_N = 64
            num_stages = 3 if D <= 64 else 2
            num_warps = 4
    elif torch.cuda.get_device_capability() == (8, 6): # tune for RTX-3090, device_capability(8, 6)
        if not causal:
            if D <= 64:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 2, 4
            else:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 2, 8
        else:
            if D <= 64:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 2, 4
            else:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 32, 32, 2, 4
    else:
        BLOCK_M, BLOCK_N, num_stages, num_warps = 32, 32, 1, 4
    return (BLOCK_M, BLOCK_N, num_stages, num_warps)

def get_bwd_kv_config(B, H, M, N, D, causal):
    assert causal
    if torch.cuda.get_device_capability() == (8, 0): # A100
        if D <= 64:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 4, 4
        else:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 32, 128, 4, 8
    elif torch.cuda.get_device_capability() == (8, 6): # tune for RTX-3090, device_capability(8, 6)
        if D <= 64:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 2, 4
        else:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 32, 32, 2, 4
    elif torch.cuda.get_device_capability() == (8, 9): # L40S
        if D <= 64:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 128, 4, 8
        else:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 32, 128, 2, 8
    elif torch.cuda.get_device_capability() == (9, 0): # H100
        if D <= 64:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 3, 4
        else:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 2, 4
    else:
        BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 2, 4
    return (BLOCK_M, BLOCK_N, num_stages, num_warps)

def get_bwd_q_config(B, H, M, N, D, causal):
    assert causal
    if torch.cuda.get_device_capability() == (8, 0): # A100
        if D <= 64:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 3, 4
        else:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 4, 8
    elif torch.cuda.get_device_capability() == (8, 6): # tune for RTX-3090, device_capability(8, 6)
        if D <= 64:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 2, 4
        else:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 32, 32, 2, 4
    elif torch.cuda.get_device_capability() == (8, 9): # L40S
        if D <= 64:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 4, 4
        else:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 3, 4
    elif torch.cuda.get_device_capability() == (9, 0): # H100
        if D <= 64:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 128, 4, 8
        else:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 128, 2, 8
    else:
        BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 2, 4
    return (BLOCK_M, BLOCK_N, num_stages, num_warps)


@triton.jit
def _bwd_preprocess(
    Out, DO,
    Delta,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_doz, stride_doh, stride_dom, stride_dok,
    stride_dz, stride_dh, stride_dm,
    M,
    BLOCK_M: tl.constexpr, D_HEAD: tl.constexpr,
    DIVISIBLE_M: tl.constexpr,
):
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    Out += off_z * stride_oz + off_h * stride_oh
    DO += off_z * stride_doz + off_h * stride_doh
    Delta += off_z * stride_dz + off_h * stride_dh

    # compute (Out * Dout).sum() for vector interpretation
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, D_HEAD)

    # load
    o_ptrs = Out + off_m[:, None] * stride_om + off_n[None, :] * stride_ok
    do_ptrs = DO + off_m[:, None] * stride_dom + off_n[None, :] * stride_dok

    if DIVISIBLE_M:
        o = tl.load(o_ptrs).to(tl.float32)
        do = tl.load(do_ptrs).to(tl.float32)
    else:
        mask_m = off_m < M
        o = tl.load(o_ptrs, mask=mask_m[:, None]).to(tl.float32)
        do = tl.load(do_ptrs, mask=mask_m[:, None]).to(tl.float32)

    # compute
    delta = tl.sum(o * do, axis=1)

    # write-back
    d_ptrs = Delta + off_m * stride_dm
    if DIVISIBLE_M:
        tl.store(d_ptrs, delta)
    else:
        tl.store(d_ptrs, delta, mask=mask_m)


@triton.jit
def _bwd_kv_kernel(
    Q, K, V, LOG_LAMBDA, SEQ_START, sm_scale, DO,
    DK, DV, DLOG_LAMBDA,
    L,
    D,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_log_lambda_z, stride_log_lambda_h, stride_log_lambda_n,
    stride_doz, stride_doh, stride_dom, stride_dok,
    stride_dkz, stride_dkh, stride_dkn, stride_dkk,
    stride_dvz, stride_dvh, stride_dvn, stride_dvk,
    stride_dlog_lambda_z, stride_dlog_lambda_h, stride_dlog_lambda_n,
    Z, H, M, N, P_SEQ,
    num_groups,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    HAS_SLIDING_WINDOW: tl.constexpr, WINDOW_SIZE: tl.constexpr,
    DIVISIBLE_M: tl.constexpr, DIVISIBLE_N: tl.constexpr, HAS_SEQ_START: tl.constexpr,
):
    input_dtype = Q.dtype.element_ty
    # -- grid id --
    start_n = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    log2e: tl.constexpr = 1.4426950408889634
    qk_scale = sm_scale * log2e

    # offset pointers for (batch, head)
    off_hk = off_h // num_groups
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_hk * stride_kh
    V += off_z * stride_vz + off_hk * stride_vh
    LOG_LAMBDA += off_z * stride_log_lambda_z + off_h * stride_log_lambda_h
    DO += off_z * stride_doz + off_h * stride_doh

    # offset pointers for batch/head
    DK += off_z * stride_dkz + off_h * stride_dkh
    DV += off_z * stride_dvz + off_h * stride_dvh
    DLOG_LAMBDA += off_z * stride_dlog_lambda_z + off_h * stride_dlog_lambda_h

    # offset pointers for batch/head
    D += (off_z * H + off_h) * M
    L += (off_z * H + off_h) * M

    if CAUSAL:
        lo = tl.maximum(start_n * BLOCK_N - P_SEQ, 0)
        lo = (lo // BLOCK_M) * BLOCK_M
    else:
        lo = 0
        
    # --- Sliding-window lower bound for KV tile (use gk_min) ---
    if HAS_SLIDING_WINDOW:
        gk_min = start_n * BLOCK_N
        sw_lo = tl.maximum(0, gk_min - P_SEQ)              # inclusive
        lo = tl.maximum(lo, (sw_lo // BLOCK_M) * BLOCK_M)  # align down

    offs_m_init = lo + tl.arange(0, BLOCK_M)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m_base = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_DMODEL)

    # initialize pointers to value-like data
    q_ptrs = Q + (offs_m_init[:, None] * stride_qm + offs_k[None, :] * stride_qk) # (BLOCK_M, BLOCK_DMODEL)
    log_lambda_out_ptrs = LOG_LAMBDA + (P_SEQ + offs_m_init) * stride_log_lambda_n # (BLOCK_N, BLOCK_DMODEL)
    k_ptrs = K + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk) # (BLOCK_N, BLOCK_DMODEL)
    v_ptrs = V + (offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk) # (BLOCK_N, BLOCK_DMODEL)
    log_lambda_in_ptrs = LOG_LAMBDA + (offs_n * stride_log_lambda_n) # (BLOCK_N, BLOCK_DMODEL)
    do_ptrs = DO + (offs_m_init[:, None] * stride_dom + offs_k[None, :] * stride_dok) # (BLOCK_M, BLOCK_DMODEL)

    dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_k[None, :] * stride_dvk) # (BLOCK_N, BLOCK_DMODEL)
    dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_k[None, :] * stride_dkk) # (BLOCK_N, BLOCK_DMODEL)
    dlog_lambda_in_ptrs = DLOG_LAMBDA + (offs_n * stride_dlog_lambda_n) # (BLOCK_N, BLOCK_DMODEL)

    # k and v stay in SRAM throughout
    if DIVISIBLE_N:
        v = tl.load(v_ptrs)
        k = tl.load(k_ptrs)
        log_lambda_in = tl.load(log_lambda_in_ptrs)
    else:
        mask_n = offs_n < N
        v = tl.load(v_ptrs, mask=mask_n[:, None])
        k = tl.load(k_ptrs, mask=mask_n[:, None])
        log_lambda_in = tl.load(log_lambda_in_ptrs, mask=mask_n)

    # If the N block doesn't contain seq_start, no need to loop
    if HAS_SEQ_START:
        SEQ_START += off_z
        seq_start = tl.load(SEQ_START)
        # Use uniform bounds to prevent warp divergence - check if this tile overlaps with valid range
        tile_end = start_n * BLOCK_N + BLOCK_N - 1
        has_valid_seq = tile_end >= seq_start - 1
        hi = tl.where(has_valid_seq, M, lo)
    else:
        hi = M
        
    # --- Sliding-window upper bound for KV tile (use gk_max) ---
    if HAS_SLIDING_WINDOW:
        gk_min = start_n * BLOCK_N
        # Use conservative bound to ensure uniform execution across threads
        gk_max = gk_min + BLOCK_N - 1
        # queries that can attend any key in this tile end at gk_max - P_SEQ + W  (exclusive)
        sw_hi = tl.minimum(M, gk_max - P_SEQ + WINDOW_SIZE)  # exclusive
        hi = tl.minimum(hi, sw_hi)

    # initialize dk amd dv
    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dlog_lambda_in = tl.zeros([BLOCK_N], dtype=tl.float32)

    # loop over a col
    for start_m in range(lo, hi, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m = start_m + offs_m_base
        causal_mask = (P_SEQ + offs_m[None, :]) >= (offs_n[:, None]) # (BLOCK_M, BLOCK_N)

        # load q1, k1, q2, k2, v, do on-chip
        if DIVISIBLE_M:
            q = tl.load(q_ptrs)
            log_lambda_out = tl.load(log_lambda_out_ptrs)
        else:
            mask_m = offs_m < M
            valid_mask = mask_m[None, :] # & mask_n
            q = tl.load(q_ptrs, mask=mask_m[:, None])
            log_lambda_out = tl.load(log_lambda_out_ptrs, mask=mask_m)

        sT = tl.dot(k, tl.trans(q), input_precision="ieee") * qk_scale
        decay_bias = (log_lambda_out[None, :] - log_lambda_in[:, None]) * log2e

        if HAS_SLIDING_WINDOW:
            # Apply forgetting bias inside window; -inf elsewhere
            if not DIVISIBLE_M:
                base_mask = valid_mask
            else:
                base_mask = tl.full([BLOCK_N, BLOCK_M], True, tl.int1)
            if CAUSAL:
                causal_mask = (P_SEQ + offs_m[None, :]) >= (offs_n[:, None])
                base_mask = base_mask & causal_mask
            if HAS_SEQ_START:
                seq_mask = offs_n[:, None] >= seq_start
                base_mask = base_mask & seq_mask
            query_pos = P_SEQ + offs_m[None, :]
            key_pos = offs_n[:, None]
            window_mask = (key_pos >= (query_pos - WINDOW_SIZE + 1)) & (key_pos <= query_pos)
            base_mask = base_mask & window_mask
            # IMPORTANT: -inf outside valid region to avoid exp2 on invalid entries
            sT = tl.where(base_mask, sT + decay_bias, float("-inf"))
        else:
            # Match forgetting_attention.py: add bias, mask later
            sT = sT + decay_bias

        # -- recompute p ---
        if DIVISIBLE_M:
            l = tl.load(L + offs_m)
        else:
            l = tl.load(L + offs_m, mask=mask_m)
        # Guard against L == -inf to prevent NaN in exp2
        empty = (l == -float("inf"))           # l is [BLOCK_M]
        l_safe = tl.where(empty, 0.0, l)
        pT = tl.math.exp2(sT - l_safe[None, :] * log2e) # (BLOCK_N, BLOCK_M)
        pT = tl.where(empty[None, :], 0.0, pT)

        if not DIVISIBLE_M:
            pT = tl.where(valid_mask, pT, 0.0)
        if CAUSAL:
            pT = tl.where(causal_mask, pT, 0.0)
        if HAS_SEQ_START:
            seq_mask = offs_n[:, None] >= seq_start
            pT = tl.where(seq_mask, pT, 0.0)
        if HAS_SLIDING_WINDOW:
            # Apply sliding window mask additionally in windowed mode
            query_pos = P_SEQ + offs_m[None, :]
            key_pos = offs_n[:, None]
            window_mask = (key_pos >= (query_pos - WINDOW_SIZE + 1)) & (key_pos <= query_pos)
            pT = tl.where(window_mask, pT, 0.0)

        # compute dv = dot(p, do)
        if DIVISIBLE_M:
            do = tl.load(do_ptrs)
        else:
            do = tl.load(do_ptrs, mask=mask_m[:, None]) # (BLOCK_M, BLOCK_DMODEL)


        dv += tl.dot(pT.to(input_dtype), do, input_precision="ieee") # (BLOCK_N, BLOCK_DMODEL)  # still correct

        # compute dp = dot(v, do)
        if DIVISIBLE_M:
            delta = tl.load(D + offs_m)
        else:
            delta = tl.load(D + offs_m, mask=mask_m)
        # dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dpT = tl.dot(v, tl.trans(do), input_precision="ieee")


        # compute ds = p * (dp - delta[:, None])
        dsT = pT * (dpT - delta[None, :]) # (BLOCK_M, BLOCK_N)

        if not DIVISIBLE_M:
            dsT = tl.where(valid_mask, dsT, 0.0)
        if CAUSAL:
            dsT = tl.where(causal_mask, dsT, 0.0)
        if HAS_SEQ_START:
            seq_mask = offs_n[:, None] >= seq_start
            dsT = tl.where(seq_mask, dsT, 0.0)
        if HAS_SLIDING_WINDOW:
            # Apply sliding window mask for gradient computation
            query_pos = P_SEQ + offs_m[None, :]
            key_pos = offs_n[:, None]
            window_mask = (key_pos >= (query_pos - WINDOW_SIZE + 1)) & (key_pos <= query_pos)
            dsT = tl.where(window_mask, dsT, 0.0)

        # compute dk = dot(ds.T, q) masking
        dk += tl.dot(dsT.to(input_dtype), q, input_precision="ieee")
        dlog_lambda_in += -tl.sum(dsT, axis=1)

        # increment pointers
        q_ptrs += BLOCK_M * stride_qm
        log_lambda_out_ptrs += BLOCK_M * stride_log_lambda_n
        do_ptrs += BLOCK_M * stride_dom

    dk *= sm_scale
    if HAS_SEQ_START:
        # Mask out 
        seq_mask = (offs_n >= seq_start)
        dk = tl.where(seq_mask[:, None], dk, 0.0)
        dv = tl.where(seq_mask[:, None], dv, 0.0)
        dlog_lambda_in = tl.where(seq_mask, dlog_lambda_in, 0.0)
    if DIVISIBLE_N:
        tl.store(dk_ptrs, dk.to(input_dtype)) # (BLOCK_N, BLOCK_DMODEL)
        tl.store(dv_ptrs, dv.to(input_dtype)) # (BLOCK_N, BLOCK_DMODEL,)
        tl.store(dlog_lambda_in_ptrs, dlog_lambda_in.to(tl.float32)) # (BLOCK_N, BLOCK_DMODEL,)
    else:
        tl.store(dk_ptrs, dk.to(input_dtype), mask=mask_n[:, None]) # (BLOCK_N, BLOCK_DMODEL)
        tl.store(dv_ptrs, dv.to(input_dtype), mask=mask_n[:, None]) # (BLOCK_N, BLOCK_DMODEL)
        tl.store(dlog_lambda_in_ptrs, dlog_lambda_in.to(tl.float32), mask=mask_n) # (BLOCK_N, BLOCK_DMODEL,)


@triton.jit
def _bwd_q_kernel(
    Q, K, V, LOG_LAMBDA, SEQ_START, sm_scale, DO,
    DQ, DLOG_LAMBDA,
    L,
    D,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_log_lambda_z, stride_log_lambda_h, stride_log_lambda_n,
    stride_doz, stride_doh, stride_dom, stride_dok,
    stride_dqz, stride_dqh, stride_dqm, stride_dqk,
    stride_dlog_lambda_z, stride_dlog_lambda_h, stride_dlog_lambda_n,
    Z, H, M, N, P_SEQ,
    num_groups,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr, LARGER_M: tl.constexpr, HAS_SEQ_START: tl.constexpr,
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
    qk_scale = sm_scale * log2e

    # offset pointers for (batch, head)
    off_hk = off_h // num_groups
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_hk * stride_kh
    V += off_z * stride_vz + off_hk * stride_vh
    LOG_LAMBDA += off_z * stride_log_lambda_z + off_h * stride_log_lambda_h
    DO += off_z * stride_doz + off_h * stride_doh
    D += (off_z * H + off_h) * M
    L += (off_z * H + off_h) * M

    # offset pointers for batch/head
    DQ += off_z * stride_dqz + off_h * stride_dqh
    DLOG_LAMBDA += off_z * stride_dlog_lambda_z + off_h * stride_dlog_lambda_h

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_DMODEL)

    # initialize pointers to value-like data
    q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk) # (BLOCK_M, BLOCK_DMODEL)
    log_lambda_out_ptrs = LOG_LAMBDA + (P_SEQ + offs_m) * stride_log_lambda_n

    dq_ptrs = DQ + (offs_m[:, None] * stride_dqm + offs_k[None, :] * stride_dqk) # (BLOCK_M, BLOCK_DMODEL)
    dlog_lambda_out_ptrs = DLOG_LAMBDA + (P_SEQ + offs_m) * stride_dlog_lambda_n
    do_ptrs = DO + (offs_m[:, None] * stride_dom + offs_k[None, :] * stride_dok) # (BLOCK_M, BLOCK_DMODEL)

    # pointer to row-wise quantities in value-like data
    d_ptrs = D + offs_m
    l_ptrs = L + offs_m

    # load q: it will stay in SRAM throughout
    if DIVISIBLE_M:
        q = tl.load(q_ptrs)
        do = tl.load(do_ptrs)
        delta = tl.load(d_ptrs)
        l = tl.load(l_ptrs)
        log_lambda_out = tl.load(log_lambda_out_ptrs)
    else:
        mask_m = offs_m < M
        q = tl.load(q_ptrs, mask=mask_m[:, None])
        do = tl.load(do_ptrs, mask=mask_m[:, None])
        delta = tl.load(d_ptrs, mask=mask_m)
        l = tl.load(l_ptrs, mask=mask_m)
        log_lambda_out = tl.load(log_lambda_out_ptrs, mask=mask_m)

    # initialize dq
    dq = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    dlog_lambda_out = tl.zeros([BLOCK_M], dtype=tl.float32)

    # loop over k, v and update accumulator
    # see note "Loop-Bound-For-N"
    if CAUSAL:
        # Use conservative upper bound to prevent warp divergence
        hi_causal = P_SEQ + (start_m + 1) * BLOCK_M
        if LARGER_M:
            hi_causal = tl.maximum(0, hi_causal)
        hi = tl.minimum(N, hi_causal)
    else:
        hi = N
        
    # Apply sliding window constraints for backward Q kernel
    if HAS_SLIDING_WINDOW:
        q_min = P_SEQ + start_m * BLOCK_M
        q_max = P_SEQ + (start_m + 1) * BLOCK_M - 1
        # Use conservative block end to ensure uniform bounds
        sw_hi = q_max + 1
        if CAUSAL:
            hi = tl.minimum(hi, sw_hi)
        else:
            hi = tl.minimum(N, sw_hi)

    offs_n_base = tl.arange(0, BLOCK_N)
    offs_n_init = offs_n_base
    if HAS_SEQ_START:
        SEQ_START += off_z
        seq_start = tl.load(SEQ_START)
        # Use conservative lower bound to prevent thread divergence
        lo = tl.minimum(seq_start, hi)
        lo = (lo // BLOCK_N) * BLOCK_N
        offs_n_init += lo
    else:
        lo = 0
        
    # Apply sliding window lower bound for backward Q kernel
    if HAS_SLIDING_WINDOW:
        q_min = P_SEQ + start_m * BLOCK_M
        sw_lo = tl.maximum(0, q_min - WINDOW_SIZE + 1)
        sw_lo = (sw_lo // BLOCK_N) * BLOCK_N
        lo = tl.maximum(lo, sw_lo)
        offs_n_init = offs_n_base + lo
    k_ptrs = K + (offs_n_init[:, None] * stride_kn + offs_k[None, :] * stride_kk) # (BLOCK_N, BLOCK_DMODEL)
    v_ptrs = V + (offs_n_init[:, None] * stride_vn + offs_k[None, :] * stride_vk) # (BLOCK_N, BLOCK_DMODEL)
    log_lambda_in_ptrs = LOG_LAMBDA + (offs_n_init * stride_log_lambda_n)

    # loop over a row
    for start_n in range(lo, hi, BLOCK_N):
        offs_n = start_n + offs_n_base

        # load k1, k2, v on chip
        if DIVISIBLE_N:
            v = tl.load(v_ptrs)
            k = tl.load(k_ptrs)
            log_lambda_in = tl.load(log_lambda_in_ptrs)
        else:
            mask_n = offs_n < N
            v = tl.load(v_ptrs, mask=mask_n[:, None])
            k = tl.load(k_ptrs, mask=mask_n[:, None])
            log_lambda_in = tl.load(log_lambda_in_ptrs, mask=mask_n)


        # recompute p = softmax(qk * sm_scale, dim=-1)
        if not DIVISIBLE_N:
            valid_mask = mask_n[None, :]
        if CAUSAL:
            causal_mask = (P_SEQ + offs_m[:, None]) >= (offs_n[None, :])
        s = tl.dot(q, tl.trans(k), input_precision="ieee") * qk_scale
        decay_bias = (log_lambda_out[:, None] - log_lambda_in[None, :]) * log2e

        if HAS_SLIDING_WINDOW:
            # Build masks and gate forgetting bias inside the window/valid region
            if DIVISIBLE_N:
                valid_mask_full = tl.full([BLOCK_M, BLOCK_N], True, tl.int1)
            else:
                valid_mask_full = valid_mask
            if CAUSAL:
                valid_mask_full = valid_mask_full & causal_mask
            if HAS_SEQ_START:
                seq_mask = offs_n[None, :] >= seq_start
                valid_mask_full = valid_mask_full & seq_mask
            query_pos = P_SEQ + offs_m[:, None]
            key_pos = offs_n[None, :]
            window_mask = (key_pos >= (query_pos - WINDOW_SIZE + 1)) & (key_pos <= query_pos)
            valid_mask_full = valid_mask_full & window_mask
            # IMPORTANT: -inf outside valid region
            s = tl.where(valid_mask_full, s + decay_bias, float("-inf"))
        else:
            # Match forgetting_attention.py: add bias without extra masking at s
            s = s + decay_bias

        # NOTE: since softmax in backward is pointwise, the normalizer has been saved in fwd
        # Guard against L == -inf to prevent NaN/Inf in exp2
        empty = (l == -float("inf"))
        l_safe = tl.where(empty, 0.0, l)
        p = tl.math.exp2(s - l_safe[:, None] * log2e) # (BLOCK_M, BLOCK_N)
        p = tl.where(empty[:, None], 0.0, p)

        # compute dp = dot(v, do)
        dp = tl.dot(do.to(input_dtype), tl.trans(v), input_precision="ieee")


        # no need to mask dp
        ds = p * (dp - delta[:, None]) # (BLOCK_M, BLOCK_N)

        # mask ds to ensure no small values
        if not DIVISIBLE_N:
            ds = tl.where(valid_mask, ds, 0.0)
        if CAUSAL:
            ds = tl.where(causal_mask, ds, 0.0)
        if HAS_SEQ_START:
            ds = tl.where(offs_n[None, :] >= seq_start, ds, 0.0)
        if HAS_SLIDING_WINDOW:
            # Apply sliding window mask for backward Q kernel
            query_pos = P_SEQ + offs_m[:, None]
            key_pos = offs_n[None, :]
            window_mask = (key_pos >= (query_pos - WINDOW_SIZE + 1)) & (key_pos <= query_pos)
            ds = tl.where(window_mask, ds, 0.0)

        dq += tl.dot(ds.to(input_dtype), k, input_precision="ieee")
        dlog_lambda_out += tl.sum(ds, axis=1)

        # increment pointers
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn
        log_lambda_in_ptrs += BLOCK_N * stride_log_lambda_n

    dq *= sm_scale
    if DIVISIBLE_M:
        tmp = tl.load(dlog_lambda_out_ptrs)
    else:
        tmp = tl.load(dlog_lambda_out_ptrs, mask=mask_m)
    dlog_lambda_out += tmp
    if DIVISIBLE_M:
        tl.store(dq_ptrs, dq.to(input_dtype))
        tl.store(dlog_lambda_out_ptrs, dlog_lambda_out)
    else:
        tl.store(dq_ptrs, dq.to(input_dtype), mask=mask_m[:, None])
        tl.store(dlog_lambda_out_ptrs, dlog_lambda_out, mask=mask_m)
