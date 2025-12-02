import torch
import triton
import triton.language as tl
import os


AUTOTUNE = os.getenv("TRITON_AUTOTUNE", "0").lower() == "attention_pair_bias"

if AUTOTUNE:
    configs = []
    for BM in [16, 32, 64, 128]:
        for BN in [16, 32, 64, 128]:
            for s in [1, 2, 3, 4]:
                for w in [4, 8, 16]:
                    configs.append(
                        triton.Config(
                            {"BLOCK_M": BM, "BLOCK_N": BN},
                            num_stages=s,
                            num_warps=w,
                        )
                    )
    pre_configs = []
    for BM in [16, 32, 64, 128, 256]:
        for s in [1, 2, 3]:
            for w in [4, 8]:
                pre_configs.append(
                    triton.Config(
                        {"BLOCK_M": BM},
                        num_stages=s,
                        num_warps=w,
                    )
                )
    bwd_configs = []
    for BM in [16, 32, 64]:
        for BN in [16, 32, 64]:
            for BD in [16, 32, 64]:
                for s in [1, 2, 3]:
                    for w in [4, 8]:
                        bwd_configs.append(
                            triton.Config(
                                {"BLOCK_M": BM, "BLOCK_N": BN, "BLOCK_D": BD},
                                num_stages=s,
                                num_warps=w,
                            )
                        )
else:
    configs = [
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_stages=1, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_stages=1, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128}, num_stages=1, num_warps=8),
    ]
    pre_configs = [
        triton.Config({"BLOCK_M": 32}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_M": 16}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_M": 32}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 16}, num_stages=1, num_warps=4),
        triton.Config({"BLOCK_M": 64}, num_stages=2, num_warps=8),
    ]
    bwd_configs = [
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 16, "BLOCK_D": 32}, num_stages=3, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 16, "BLOCK_D": 32}, num_stages=3, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 16, "BLOCK_D": 32}, num_stages=2, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 16, "BLOCK_D": 32}, num_stages=3, num_warps=8
        ),
    ]


def get_seq_group(L):
    GROUP_LENGTHS = [32, 64, 128, 256, 512]
    for group_l in GROUP_LENGTHS:
        if L <= group_l:
            return group_l
    return GROUP_LENGTHS[-1]


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    k_ptr,
    v_ptr,
    start_m,
    start_d,
    qk_scale,
    b_ptr,
    stride_kn,
    stride_vn,
    stride_bm,
    stride_bn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    N_CTX: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_D: tl.constexpr,
):
    lo, hi = 0, N_CTX
    # loop over k, v and update accumulator
    offset_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offset_k = start_d * BLOCK_D + tl.arange(0, BLOCK_D)

    for start_n in range(lo, hi, BLOCK_N):
        # start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        if EVEN_N:
            k = tl.load(k_ptr, mask=offset_k[:, None] < HEAD_DIM, other=0.0)
            bias_val = tl.load(b_ptr)
            v = tl.load(v_ptr, mask=offset_k[None, :] < HEAD_DIM, other=0.0)
        else:
            offset_n = start_n + tl.arange(0, BLOCK_N)
            bias_mask = (offset_m[:, None] < N_CTX) & (offset_n[None, :] < N_CTX)
            bias_val = tl.load(b_ptr, mask=bias_mask, other=float("-inf"))
            if EVEN_D:
                k = tl.load(k_ptr, mask=offset_n[None, :] < N_CTX, other=0.0)
                v = tl.load(v_ptr, mask=offset_n[:, None] < N_CTX, other=0.0)
            else:
                k_mask = (offset_n[None, :] < N_CTX) & (offset_k[:, None] < HEAD_DIM)
                v_mask = (offset_n[:, None] < N_CTX) & (offset_k[None, :] < HEAD_DIM)
                k = tl.load(k_ptr, mask=k_mask, other=0.0)
                v = tl.load(v_ptr, mask=v_mask, other=0.0)

        qk = tl.dot(q, k, allow_tf32=False)
        qk += bias_val / (qk_scale / 1.44269504)
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]

        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        acc = tl.dot(p, v, acc, allow_tf32=False)
        # update m_i and l_i
        m_i = m_ij
        b_ptr += BLOCK_N * stride_bn
        k_ptr += BLOCK_N * stride_kn
        v_ptr += BLOCK_N * stride_vn
    return acc, l_i, m_i


# fmt: off
@triton.autotune(configs=configs, key=["GROUP_N", "H", "HEAD_DIM"])
@triton.jit
def _attn_fwd(
    Q, K, V, Bias, sm_scale,
    M, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_on,
    stride_bz, stride_bh, stride_bm, stride_bn,
    Z, H: tl.constexpr, N_CTX, HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    GROUP_N: tl.constexpr,
):
    tl.static_assert(BLOCK_D >= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    start_d = tl.program_id(2)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    bias_offset = off_z * stride_bz + off_h * stride_bh

    offset_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offset_n = tl.arange(0, BLOCK_N)
    offset_k = start_d * BLOCK_D + tl.arange(0, BLOCK_D)

    q_ptr = (
        Q + qvk_offset + offset_m[:, None] * stride_qm + offset_k[None, :] * stride_qk
    )
    k_ptr = (
        K + qvk_offset + offset_n[None, :] * stride_kn + offset_k[:, None] * stride_kk
    )
    v_ptr = (
        V + qvk_offset + offset_n[:, None] * stride_vn + offset_k[None, :] * stride_vk
    )
    o_ptr = (
        Out + qvk_offset + offset_m[:, None] * stride_om + offset_k[None, :] * stride_on
    )
    bias_ptr = (
        Bias
        + bias_offset
        + offset_m[:, None] * stride_bm
        + offset_n[None, :] * stride_bn
    )

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)

    EVEN_N = (N_CTX % BLOCK_M == 0) & (N_CTX % BLOCK_N == 0)
    EVEN_D = HEAD_DIM % BLOCK_D == 0

    # load q: it will stay in SRAM throughout
    if EVEN_N and EVEN_D:
        q = tl.load(q_ptr)
    elif not EVEN_N and EVEN_D:
        q = tl.load(q_ptr, mask=offset_m[:, None] < N_CTX, other=0.0)
    elif EVEN_N and not EVEN_D:
        q = tl.load(q_ptr, mask=offset_k[None, :] < HEAD_DIM, other=0.0)
    else:
        q_mask = (offset_m[:, None] < N_CTX) & (offset_k[None, :] < HEAD_DIM)
        q = tl.load(q_ptr, mask=q_mask, other=0.0)

    acc, l_i, m_i = _attn_fwd_inner(
        acc,
        l_i,
        m_i,
        q,
        k_ptr,
        v_ptr,
        start_m,
        start_d,
        qk_scale,
        bias_ptr,
        stride_kn,
        stride_vn,
        stride_bm,
        stride_bn,
        BLOCK_M,
        BLOCK_N,
        BLOCK_D,
        HEAD_DIM,
        N_CTX,
        EVEN_N,
        EVEN_D,
    )
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + start_m * BLOCK_M + tl.arange(0, BLOCK_M)

    if EVEN_N and EVEN_D:
        tl.store(m_ptrs, m_i)
        tl.store(o_ptr, acc.to(Out.type.element_ty))
    elif not EVEN_N and EVEN_D:
        tl.store(m_ptrs, m_i, mask=offset_m < N_CTX)
        tl.store(o_ptr, acc.to(Out.type.element_ty), mask=offset_m[:, None] < N_CTX)
    elif EVEN_N and not EVEN_D:
        tl.store(m_ptrs, m_i)
        tl.store(o_ptr, acc.to(Out.type.element_ty), mask=offset_k[None, :] < HEAD_DIM)
    else:
        tl.store(m_ptrs, m_i, mask=offset_m < N_CTX)
        out_mask = (offset_m[:, None] < N_CTX) & (offset_k[None, :] < HEAD_DIM)
        tl.store(o_ptr, acc.to(Out.type.element_ty), mask=out_mask)
# fmt: on


# fmt: off
@triton.autotune(configs=pre_configs, key=["GROUP_N", "H", "HEAD_DIM"])
@triton.jit
def _attn_bwd_preprocess(
    O, DO, Delta,
    Z, H: tl.constexpr, N_CTX, HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
    GROUP_N: tl.constexpr,
):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_n = tl.arange(0, BLOCK_D)

    o_ptr = O + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]
    do_ptr = DO + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]

    mask_m = (off_m[:, None] < N_CTX) & (off_n[None, :] < HEAD_DIM)
    mask_delta = off_m < N_CTX

    o = tl.load(o_ptr, mask=mask_m, other=0.0)
    # do = tl.load(do_ptr, mask=mask_m, other=0.0).to(tl.float32)
    do = tl.load(do_ptr, mask=mask_m, other=0.0)

    delta = tl.sum(o * do, axis=1)

    delta_ptr = Delta + off_hz * N_CTX + off_m
    tl.store(delta_ptr, delta, mask=mask_delta)
# fmt: on


# The main inner-loop logic for computing dK and dV.
@triton.jit
def _attn_bwd_dkdv(
    dk,
    dv,
    DBias,
    Q,
    k,
    v,
    Bias,
    sm_scale,  #
    DO,  #
    M,
    D,  #
    EVEN_N,
    EVEN_D,  #
    # shared by Q/K/V/DO.
    stride_tok,
    stride_d,  #
    H,
    N_CTX,
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    BLOCK_D: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,  #
    # Filled in by the wrapper.
    start_n,
    start_m,
    stride_bm,
    stride_bn,
):
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_D)
    qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    biasT_ptrs = Bias + offs_m[None, :] * stride_bm + offs_n[:, None] * stride_bn
    dbiasT_ptrs = DBias + offs_m[None, :] * stride_bm + offs_n[:, None] * stride_bn
    m_ptrs = M + offs_m
    d_ptrs = D + offs_m

    lo, hi = 0, N_CTX
    for start_m in range(lo, hi, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        if EVEN_N and EVEN_D:
            qT = tl.load(qT_ptrs)
            m = tl.load(m_ptrs)
            biasT = tl.load(biasT_ptrs)
            do = tl.load(do_ptrs)
            Di = tl.load(d_ptrs)
        elif not EVEN_N and EVEN_D:
            qT = tl.load(qT_ptrs, mask=offs_m[None, :] < N_CTX, other=0.0)
            bias_mask = (offs_m[None, :] < N_CTX) & (offs_n[:, None] < N_CTX)
            biasT = tl.load(biasT_ptrs, mask=bias_mask, other=0.0)
            do = tl.load(do_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
            m = tl.load(m_ptrs, mask=offs_m < N_CTX, other=0.0)
            Di = tl.load(d_ptrs, mask=offs_m < N_CTX, other=0.0)
        elif EVEN_N and not EVEN_D:
            qT = tl.load(qT_ptrs, mask=offs_k[:, None] < HEAD_DIM)
            biasT = tl.load(biasT_ptrs)
            do = tl.load(do_ptrs, mask=offs_k[None, :] < HEAD_DIM)
            m = tl.load(m_ptrs)
            Di = tl.load(d_ptrs)
        else:
            qT_mask = (offs_m[None, :] < N_CTX) & (offs_k[:, None] < HEAD_DIM)
            do_mask = (offs_m[:, None] < N_CTX) & (offs_k[None, :] < HEAD_DIM)
            qT = tl.load(qT_ptrs, mask=qT_mask, other=0.0)

            bias_mask = (offs_m[None, :] < N_CTX) & (offs_n[:, None] < N_CTX)
            biasT = tl.load(biasT_ptrs, mask=bias_mask, other=0.0)
            do = tl.load(do_ptrs, mask=do_mask, other=0.0)
            m = tl.load(m_ptrs, mask=offs_m < N_CTX, other=0.0)
            Di = tl.load(d_ptrs, mask=offs_m < N_CTX, other=0.0)

        qkT = tl.dot(k, qT, allow_tf32=False)
        pT = tl.math.exp2(qkT + biasT - m[None, :])
        # Compute dV.
        ppT = pT
        dv += tl.dot(ppT, do, allow_tf32=False)

        # D (= delta) is pre-divided by ds_scale.
        # Compute dP and dS.
        # dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
        dpT = tl.dot(v, tl.trans(do), allow_tf32=False)
        dsT = pT * (dpT - Di[None, :])
        # dsT = dsT
        dk += tl.dot(dsT, tl.trans(qT), allow_tf32=False)

        if DBias is not None:  # or just skip the check if always passing DBias
            dsT_fp32 = dsT.to(tl.float32)
            if EVEN_N:
                tl.store(dbiasT_ptrs, dsT_fp32)
            else:
                mask = (offs_m[None, :] < N_CTX) & (offs_n[:, None] < N_CTX)
                tl.store(dbiasT_ptrs, dsT_fp32, mask=mask)
        # Increment pointers.
        qT_ptrs += BLOCK_M * stride_tok
        do_ptrs += BLOCK_M * stride_tok
        biasT_ptrs += BLOCK_M * stride_bm
        dbiasT_ptrs += BLOCK_M * stride_bm
        m_ptrs += BLOCK_M
        d_ptrs += BLOCK_M
    return dk, dv


# the main inner-loop logic for computing dQ
@triton.jit
def _attn_bwd_dq(
    dq,
    q,
    K,
    V,
    Bias,  #
    do,
    m,
    D,
    EVEN_N,
    EVEN_D,
    # shared by Q/K/V/DO.
    stride_tok,
    stride_d,  #
    H,
    N_CTX,  #
    BLOCK_N: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_D: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,
    # Filled in by the wrapper.
    start_m,
    start_n,  #
    MASK: tl.constexpr,
    stride_bm,
    stride_bn,
):
    offs_m = start_m + tl.arange(0, BLOCK_N)
    offs_n = start_n + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_D)
    kT_ptrs = K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    vT_ptrs = V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    biasT_ptrs = Bias + offs_n[None, :] * stride_bn + offs_m[:, None] * stride_bm

    # D (= delta) is pre-divided by ds_scale.
    if EVEN_N:
        Di = tl.load(D + offs_m)
    else:
        Di = tl.load(D + offs_m, mask=offs_m < N_CTX, other=0.0)
    lo, hi = 0, N_CTX

    for start_n in range(lo, hi, BLOCK_M):
        offs_n = start_n + tl.arange(0, BLOCK_M)
        if EVEN_N and EVEN_D:
            kT = tl.load(kT_ptrs)
            vT = tl.load(vT_ptrs)
            bias = tl.load(biasT_ptrs)
        elif not EVEN_N and EVEN_D:
            kT = tl.load(kT_ptrs, mask=offs_n[None, :] < N_CTX, other=0.0)
            vT = tl.load(vT_ptrs, mask=offs_n[None, :] < N_CTX, other=0.0)
            bias = tl.load(
                biasT_ptrs,
                mask=(offs_m[:, None] < N_CTX) & (offs_n[None, :] < N_CTX),
                other=float("-inf"),
            )
        elif EVEN_N and not EVEN_D:
            kT = tl.load(kT_ptrs, mask=offs_k[:, None] < HEAD_DIM, other=0.0)
            vT = tl.load(vT_ptrs, mask=offs_k[:, None] < HEAD_DIM, other=0.0)
            bias = tl.load(biasT_ptrs)
        else:
            kT = tl.load(
                kT_ptrs,
                mask=(offs_n[None, :] < N_CTX) & (offs_k[:, None] < HEAD_DIM),
                other=0.0,
            )
            vT = tl.load(
                vT_ptrs,
                mask=(offs_n[None, :] < N_CTX) & (offs_k[:, None] < HEAD_DIM),
                other=0.0,
            )
            bias = tl.load(
                biasT_ptrs,
                mask=(offs_m[:, None] < N_CTX) & (offs_n[None, :] < N_CTX),
                other=float("-inf"),
            )

        qk = tl.dot(q, kT, allow_tf32=False)

        p = tl.math.exp2(qk + bias - m)
        # Compute dP and dS.
        dp = tl.dot(do, vT, allow_tf32=False)
        ds = p * (dp - Di[:, None])
        # Compute dQ.
        dq += tl.dot(ds, tl.trans(kT), allow_tf32=False)

        # Increment pointers.
        kT_ptrs += BLOCK_M * stride_tok
        vT_ptrs += BLOCK_M * stride_tok
        biasT_ptrs += BLOCK_M * stride_bn
    return dq


# fmt: off
@triton.autotune(configs=bwd_configs, key=["GROUP_N", "H", "HEAD_DIM"])
@triton.jit
def _attn_bwd(
    Q, K, V, Bias, sm_scale,
    DO, DQ, DK, DV, DBias,
    M, D,
    stride_z, stride_h, stride_tok, stride_d,
    bias_stride_z, bias_stride_h, bias_stride_m, bias_stride_n,
    H: tl.constexpr, N_CTX, HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    GROUP_N: tl.constexpr,
):
    tl.static_assert(BLOCK_D >= HEAD_DIM)
    LN2: tl.constexpr = 0.6931471824645996  # = ln(2)

    bhid = tl.program_id(2)
    off_chz = (bhid * N_CTX).to(tl.int64)
    adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    pid = tl.program_id(0)

    # offset pointers for batch/head
    Q += adj
    K += adj
    V += adj
    DO += adj
    DQ += adj
    DK += adj
    DV += adj
    M += off_chz
    D += off_chz

    # Also offset DBias so it points to start of [bhid, :, :].
    # If DBias has shape [B*H, N_CTX, N_CTX], do:
    # offset_bias = (bhid * N_CTX * N_CTX).to(tl.int64)
    offset_bias = (bias_stride_h * (bhid % H) + bias_stride_z * (bhid // H)).to(tl.int64)
    Bias += offset_bias
    DBias += offset_bias

    # load scales
    offs_k = tl.arange(0, BLOCK_D)

    start_n = pid * BLOCK_N
    offs_n = start_n + tl.arange(0, BLOCK_N)

    dv = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)

    EVEN_N = (N_CTX % BLOCK_N == 0) & (N_CTX % BLOCK_M == 0)
    EVEN_D = HEAD_DIM % BLOCK_D == 0

    # load K and V: they stay in SRAM throughout the inner loop.
    k_ptr = K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    v_ptr = V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d

    if EVEN_N and EVEN_D:
        k = tl.load(k_ptr)
        v = tl.load(v_ptr)
    elif not EVEN_N and EVEN_D:
        k = tl.load(k_ptr, mask=offs_n[:, None] < N_CTX, other=0.0)
        v = tl.load(v_ptr, mask=offs_n[:, None] < N_CTX, other=0.0)
    elif EVEN_N and not EVEN_D:
        k = tl.load(k_ptr, mask=offs_k[None, :] < HEAD_DIM, other=0.0)
        v = tl.load(v_ptr, mask=offs_k[None, :] < HEAD_DIM, other=0.0)
    else:
        k = tl.load(
            k_ptr,
            mask=(offs_n[:, None] < N_CTX) & (offs_k[None, :] < HEAD_DIM),
            other=0.0,
        )
        v = tl.load(
            v_ptr,
            mask=(offs_n[:, None] < N_CTX) & (offs_k[None, :] < HEAD_DIM),
            other=0.0,
        )

    # load

    # Compute dK and dV for non-masked blocks.
    dk, dv = _attn_bwd_dkdv(
        dk,
        dv,
        DBias,
        Q,
        k,
        v,
        Bias,
        sm_scale,
        DO,
        M,
        D,
        EVEN_N,
        EVEN_D,
        stride_tok,
        stride_d,
        H,
        N_CTX,
        BLOCK_M,
        BLOCK_N,
        BLOCK_D,
        HEAD_DIM,
        start_n,
        start_m=0,
        stride_bm=bias_stride_m,
        stride_bn=bias_stride_n,
    )

    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    dk *= sm_scale
    dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d

    if EVEN_N and EVEN_D:
        tl.store(dv_ptrs, dv)
        tl.store(dk_ptrs, dk)
    elif not EVEN_N and EVEN_D:
        tl.store(dv_ptrs, dv, mask=offs_n[:, None] < N_CTX)
        tl.store(dk_ptrs, dk, mask=offs_n[:, None] < N_CTX)
    elif EVEN_N and not EVEN_D:
        tl.store(dv_ptrs, dv, mask=offs_k[None, :] < HEAD_DIM)
        tl.store(dk_ptrs, dk, mask=offs_k[None, :] < HEAD_DIM)
    else:
        tl.store(
            dv_ptrs, dv, mask=(offs_n[:, None] < N_CTX) & (offs_k[None, :] < HEAD_DIM)
        )
        tl.store(
            dk_ptrs, dk, mask=(offs_n[:, None] < N_CTX) & (offs_k[None, :] < HEAD_DIM)
        )

    # THIS BLOCK DOES DQ:
    start_m = pid * BLOCK_N
    offs_m = start_m + tl.arange(0, BLOCK_N)

    dq = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    q_do_offset = offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    if EVEN_N and EVEN_D:
        q = tl.load(Q + q_do_offset)
        do = tl.load(DO + q_do_offset)
        m = tl.load(M + offs_m)
    elif not EVEN_N and EVEN_D:
        q = tl.load(Q + q_do_offset, mask=offs_m[:, None] < N_CTX, other=0.0)
        do = tl.load(DO + q_do_offset, mask=offs_m[:, None] < N_CTX, other=0.0)
        m = tl.load(M + offs_m, mask=offs_m < N_CTX, other=0.0)
    elif EVEN_N and not EVEN_D:
        q = tl.load(Q + q_do_offset, mask=offs_k[None, :] < HEAD_DIM, other=0.0)
        do = tl.load(DO + q_do_offset, mask=offs_k[None, :] < HEAD_DIM, other=0.0)
        m = tl.load(M + offs_m)
    else:
        q = tl.load(
            Q + q_do_offset,
            mask=(offs_m[:, None] < N_CTX) & (offs_k[None, :] < HEAD_DIM),
            other=0.0,
        )
        do = tl.load(
            DO + q_do_offset,
            mask=(offs_m[:, None] < N_CTX) & (offs_k[None, :] < HEAD_DIM),
            other=0.0,
        )
        m = tl.load(M + offs_m, mask=offs_m < N_CTX, other=0.0)
    m = m[:, None]

    dq = _attn_bwd_dq(
        dq,
        q,
        K,
        V,
        Bias,  #
        do,
        m,
        D,  #
        EVEN_N,
        EVEN_D,
        stride_tok,
        stride_d,  #
        H,
        N_CTX,  #
        BLOCK_N,
        BLOCK_M,
        BLOCK_D,
        HEAD_DIM,  #
        start_m,
        start_n=0,
        MASK=False,
        stride_bm=bias_stride_m,
        stride_bn=bias_stride_n,
    )
    # Write back dQ.
    dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    dq *= LN2
    if EVEN_N and EVEN_D:
        tl.store(dq_ptrs, dq)
    elif not EVEN_N and EVEN_D:
        tl.store(dq_ptrs, dq, mask=offs_m[:, None] < N_CTX)
    elif EVEN_N and not EVEN_D:
        tl.store(dq_ptrs, dq, mask=offs_k[None, :] < HEAD_DIM)
    else:
        tl.store(
            dq_ptrs, dq, mask=(offs_m[:, None] < N_CTX) & (offs_k[None, :] < HEAD_DIM)
        )
# fmt: on


class TritonAttentionPairBiasFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bias: torch.Tensor,
    ):
        if q.dtype != torch.float32:
            raise ValueError(f"Only support float32, but got {q.dtype=}. ")
        if not q.dtype == k.dtype == v.dtype == bias.dtype:
            raise ValueError(
                "q, k, v, bias must have the same dtype, "
                f"but got {q.dtype=}, {k.dtype=}, {v.dtype=}, {bias.dtype=}"
            )
        if not q.shape == k.shape == v.shape:
            raise ValueError(
                "q, k, v must have the same shape, "
                f"but got {q.shape=}, {k.shape=}, {v.shape=}"
            )
        if q.ndim != 4:
            raise ValueError(
                f"q, k, v must have 4D, but got {q.ndim=}D, {k.ndim=}D, {v.ndim=}D"
            )

        q, k, v, bias = [x.contiguous() for x in (q, k, v, bias)]
        B, H, N, D = q.shape
        if D > 64:
            raise ValueError(
                f"Only support HEAD_DIM <= 64, but got {D=}. Recommend to use 32."
            )
        if bias.shape != (B, H, N, N):
            raise ValueError(f"bias must have shape {B, H, N, N}, but got {bias.shape=}")

        sm_scale = D**-0.5
        o = torch.empty_like(q)
        M = torch.empty(B, H, N, device=q.device, dtype=torch.float32)

        # fmt: off
        grid = lambda META: (
            triton.cdiv(N, META["BLOCK_M"]),
            B * H,
            triton.cdiv(D, META["BLOCK_D"]),
        )
        _attn_fwd[grid](
            q, k, v, bias, sm_scale,
            M, o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            bias.stride(0), bias.stride(1), bias.stride(2), bias.stride(3),
            B, H, N, D,
            BLOCK_D=triton.next_power_of_2(D),
            GROUP_N=get_seq_group(N),
        )
        # fmt: on

        o = o.contiguous()
        ctx.save_for_backward(
            q.to(torch.bfloat16),
            k.to(torch.bfloat16),
            v.to(torch.bfloat16),
            bias.to(torch.bfloat16),
            o.to(torch.bfloat16),
            M.to(torch.bfloat16),
        )
        return o

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        q, k, v, bias, o, M = (x.to(torch.float32) for x in ctx.saved_tensors)
        grad_output = grad_output.contiguous()

        B, H, N, D = q.shape
        RCP_LN2 = 1.4426950408889634
        sm_scale = D**-0.5
        delta = torch.empty_like(M)

        # fmt: off
        grid = lambda META: (triton.cdiv(N, META["BLOCK_M"]), B * H, 1)
        _attn_bwd_preprocess[grid](
            o, grad_output, delta,
            B, H, N, D,
            BLOCK_D=64,
            GROUP_N=get_seq_group(N),
        )
        # fmt: on

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        dbias = torch.empty_like(bias)

        # fmt: off
        grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]), 1, B * H)
        _attn_bwd[grid](
            q, k * sm_scale * RCP_LN2, v, bias * RCP_LN2, sm_scale,
            grad_output, dq, dk, dv, dbias,
            M, delta,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            bias.stride(0), bias.stride(1), bias.stride(2), bias.stride(3),
            H, N, D,
            GROUP_N=get_seq_group(N),
        )
        # fmt: on

        return dq, dk, dv, dbias


triton_attention_pair_bias = TritonAttentionPairBiasFunction.apply
