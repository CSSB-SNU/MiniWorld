import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from einops import rearrange

AUTOTUNE = os.getenv("TRITON_AUTOTUNE", "0").lower() == "attention_pair_bias"

if AUTOTUNE:
    configs = [
        triton.Config({"BLOCK_M": m, "BLOCK_K": k}, w, s)
        for m in [16, 32, 64, 128]
        for k in [16, 32, 64, 128]
        for w in [4, 8, 16]
        for s in [1, 2, 3, 4]
        if not m * k > 32 * 64
    ]
    pre_configs = [
        triton.Config({"BLOCK_M": BM}, num_stages=s, num_warps=w)
        for BM in [16, 32, 64, 128, 256]
        for s in [1, 2, 3]
        for w in [4, 8]
    ]
    bwd_configs = [
        triton.Config(
            {"BLOCK_M": BM, "BLOCK_N": BN, "BLOCK_D": BD}, num_stages=s, num_warps=w,
        )
        for BM in [16, 32, 64]
        for BN in [16, 32, 64]
        for BD in [16, 32, 64]
        for s in [1, 2, 3]
        for w in [4, 8]
    ]
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
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_D": 32}, num_stages=2, num_warps=4,
        ),
    ]


def get_seq_group(L):
    GROUP_LENGTHS = [32, 64, 128, 256, 512]
    for group_l in GROUP_LENGTHS:
        if group_l >= L:
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
    dtype = k_ptr.dtype.element_ty

    for start_n in range(lo, hi, BLOCK_N):
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

        qk = tl.dot(q, k, allow_tf32=True) + bias_val / (qk_scale / 1.44269504)
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]

        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        p = p.to(dtype)
        acc = tl.dot(p, v, acc, allow_tf32=True)
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
    do = tl.load(do_ptr, mask=mask_m, other=0.0)

    delta = tl.sum(o * do, axis=1)

    delta_ptr = Delta + off_hz * N_CTX + off_m
    tl.store(delta_ptr, delta, mask=mask_delta)
# fmt: on


# The main inner-loop logic for computing dK and dV.
@triton.jit
def _attn_bwd_dqdkdv(
    dk,
    dv,
    DBias,
    Q,
    k,
    v,
    Bias,
    qk_scale,
    DO,
    DQ,
    M,
    D,
    EVEN_N,
    EVEN_D,
    # shared by Q/K/V/DO.
    stride_tok,
    stride_d,
    H,
    N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
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
    dqT_ptrs = DQ + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    biasT_ptrs = Bias + offs_m[None, :] * stride_bm + offs_n[:, None] * stride_bn
    dbiasT_ptrs = DBias + offs_m[None, :] * stride_bm + offs_n[:, None] * stride_bn
    m_ptrs = M + offs_m
    d_ptrs = D + offs_m

    lo, hi = 0, N_CTX
    dtype = qT_ptrs.dtype.element_ty
    for start_m in range(lo, hi, BLOCK_M):  # noqa: PLR1704
        offs_m = start_m + tl.arange(0, BLOCK_M)
        qT_mask = (offs_m[None, :] < N_CTX) & (offs_k[:, None] < HEAD_DIM)
        do_mask = (offs_m[:, None] < N_CTX) & (offs_k[None, :] < HEAD_DIM)
        qT = tl.load(qT_ptrs, mask=qT_mask, other=0.0)

        bias_mask = (offs_m[None, :] < N_CTX) & (offs_n[:, None] < N_CTX)
        biasT = tl.load(biasT_ptrs, mask=bias_mask, other=float("-inf"))
        do = tl.load(do_ptrs, mask=do_mask, other=0.0)
        m = tl.load(m_ptrs, mask=offs_m < N_CTX, other=0.0)
        Di = tl.load(d_ptrs, mask=offs_m < N_CTX, other=0.0)

        qT = qT.to(dtype)
        qkT = tl.dot(k, qT, allow_tf32=True) + biasT / (qk_scale / 1.44269504)
        qkT = qkT * qk_scale - m[None, :]

        pT = tl.math.exp2(qkT)
        # Compute dV.
        pT = pT.to(dtype)
        dv = tl.dot(pT, do, dv, allow_tf32=True)

        # D (= delta) is pre-divided by ds_scale.
        # Compute dP and dS.
        dpT = tl.dot(v, tl.trans(do), allow_tf32=True)
        dsT = pT * (dpT - Di[None, :])

        if DBias is not None:  # or just skip the check if always passing DBias
            if EVEN_N:
                tl.atomic_add(dbiasT_ptrs, dsT)
            else:
                tl.atomic_add(dbiasT_ptrs, dsT, mask=bias_mask)

        dsT = dsT.to(dtype)
        dk = tl.dot(dsT, tl.trans(qT), dk, allow_tf32=True)
        dqT_to_add = tl.dot(tl.trans(k), dsT, allow_tf32=True) * (qk_scale / 1.44269504)
        tl.atomic_add(dqT_ptrs, dqT_to_add, qT_mask)

        # Increment pointers.
        qT_ptrs += BLOCK_M * stride_tok
        do_ptrs += BLOCK_M * stride_tok
        dqT_ptrs += BLOCK_M * stride_tok
        biasT_ptrs += BLOCK_M * stride_bm
        dbiasT_ptrs += BLOCK_M * stride_bm
        m_ptrs += BLOCK_M
        d_ptrs += BLOCK_M
    return dk, dv


# fmt: off
# @triton.autotune(configs=bwd_configs, key=["GROUP_N", "H", "HEAD_DIM"])
@triton.jit
# fmt: on
def _attn_bwd(
    Q, K, V, Bias, sm_scale,
    DO, DQ, DK, DV, DBias,
    M, D,
    stride_a, stride_z, stride_tok, stride_d,
    bias_stride_z, bias_stride_m, bias_stride_n,
    B, H: tl.constexpr, N_CTX, N_AUG, HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    GROUP_N: tl.constexpr,
):
    # q : (A, B, H, N_CTX, HEAD_DIM), M : (A, B, H, N_CTX)
    # dbias= (B, H, N_CTX, N_CTX)
    # bias= (B, H, N_CTX, N_CTX)
    tl.static_assert(BLOCK_D >= HEAD_DIM)

    pid = tl.program_id(0)
    aid = tl.program_id(1)  # aug id
    bhid = tl.program_id(2) # batch * head id
    qkv_adj = bhid * stride_z + aid * stride_a
    M_adj = bhid * N_CTX + aid * N_CTX * H * B

    # offset pointers for batch/head
    Q += qkv_adj
    K += qkv_adj
    V += qkv_adj
    DO += qkv_adj
    DQ += qkv_adj
    DK += qkv_adj
    DV += qkv_adj
    M += M_adj
    D += M_adj

    # Also offset DBias so it points to start of [bhid, :, :].
    # If DBias has shape [B*H, N_CTX, N_CTX], do:
    offset_bias = bhid * N_CTX * N_CTX
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
    qk_scale = sm_scale * 1.44269504  # 1/log(2)
    dk, dv = _attn_bwd_dqdkdv(
        dk,
        dv,
        DBias,
        Q,
        k,
        v,
        Bias,
        qk_scale,
        DO,
        DQ,
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
            dv_ptrs, dv, mask=(offs_n[:, None] < N_CTX) & (offs_k[None, :] < HEAD_DIM),
        )
        tl.store(
            dk_ptrs, dk, mask=(offs_n[:, None] < N_CTX) & (offs_k[None, :] < HEAD_DIM),
        )





class TritonAtomAugmentedAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        W_bias: torch.Tensor,
        pair: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        # q,k,v: (A, B, H, N, D)
        # W_bias: (d_pair, H)
        # pair: (B, N, N, d_pair)
        # mask: (B, N)
        op_dtype = q.dtype
        _W_bias = W_bias.to(op_dtype)
        if not q.dtype == k.dtype == v.dtype:
            msg = f"q, k, v must have the same dtype, but got {q.dtype=}, {k.dtype=}, {v.dtype=}"
            raise ValueError(msg)
        if not q.shape == k.shape == v.shape:
            msg = f"q, k, v must have the same shape, but got {q.shape=}, {k.shape=}, {v.shape=}"
            raise ValueError(msg)
        if q.ndim != 5:  # noqa: PLR2004
            msg = f"q, k, v must have 5D, but got {q.ndim=}D, {k.ndim=}D, {v.ndim=}D"
            raise ValueError(msg)

        q, k, v, pair = [x.contiguous() for x in (q, k, v, pair)]
        A, B, H, N, D = q.shape
        if D > 64:  # noqa: PLR2004
            msg = f"Only support HEAD_DIM <= 64, but got {D=}."
            raise ValueError(msg)
        q,k,v = [
            rearrange(x, "a b h n d -> (a b) h n d", a=A, b=B, h=H)
            for x in (q, k, v)
        ]

        bias = F.linear(pair, _W_bias, bias=None) # (B, N, N, H)
        if mask is not None:
            key_mask   = mask[:, None, :, None]
            both = ~key_mask
            bias.masked_fill_(both, -65535.0)  # to avoid nan in triton kernel
        bias = rearrange(bias, "b n m h -> b h n m").contiguous()  # (B, H, N, N)
        bias = bias.expand(A*B, H, N, N)  # (AB, H, N, N)

        sm_scale = D**-0.5
        o = torch.empty_like(q)
        M = torch.empty(A, B, H, N, device=q.device, dtype=torch.float32)

        # fmt: off
        grid = lambda META: (
            triton.cdiv(N, META["BLOCK_M"]),
            A * B * H,
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
            A*B, H, N, D,
            BLOCK_D=triton.next_power_of_2(D),
            GROUP_N=get_seq_group(N),
        )
        # fmt: on

        o = o.contiguous()
        q,k,v,o = [
            rearrange(x, "(a b) h n d -> a b h n d", a=A, b=B, h=H)
            for x in (q, k, v, o)
        ]

        ctx.save_for_backward(q, k, v, W_bias, pair, mask, o, M)
        ctx.op_dtype = op_dtype
        ctx.d_pair = pair.shape[-1]
        ctx.return_dmask = mask is not None
        return o

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        op_dtype = ctx.op_dtype
        # q, k, v, W_bias, pair, mask, o, M = ctx.saved_tensors
        q, k, v, W_bias, pair, mask, o, M = ctx.saved_tensors
        q, k, v, W_bias, pair, o, M = [
            x.to(op_dtype).contiguous() for x in (q, k, v, W_bias, pair, o, M)
        ]


        # saved_tensors = torch.load("./to_rm/saved_tensors.pt")
        # qq, kk, vv, bias, W_bias, pair, mask, o, M = saved_tensors
        # breakpoint()
        grad_output = grad_output.contiguous()
        A, B, H, N, D = q.shape
        sm_scale = D**-0.5
        delta = torch.empty_like(M)
        d_pair = ctx.d_pair

        # fmt: off
        grid = lambda META: (triton.cdiv(N, META["BLOCK_M"]), A * B * H, 1)
        _attn_bwd_preprocess[grid](
            o, grad_output, delta,
            B*A, H, N, D,
            BLOCK_D=32,
            GROUP_N=get_seq_group(N),
        )
        # fmt: on

        # bias recalculation
        bias = F.linear(pair, W_bias, bias=None)  # (B, N, N, H)
        if mask is not None:
            key_mask   = mask[:, None, :, None]   # (B, 1, N, 1) → j축
            both = ~key_mask
            bias.masked_fill_(both, -65535.0)  # to avoid nan in triton kernel
        bias = rearrange(bias, "b n m h -> b h n m").contiguous()  # (BH, N, N)

        dq = torch.zeros_like(q).float()
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        dbias = torch.zeros(B, H, N, N, device=q.device, dtype=torch.float32)

        # fmt: off
        grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]), A, B*H)
        _attn_bwd[grid](
            q, k, v, bias, sm_scale,
            grad_output, dq, dk, dv, dbias,
            M, delta,
            q.stride(1), q.stride(2), q.stride(3), q.stride(4),
            bias.stride(0), bias.stride(2), bias.stride(3),
            B, H, N, A, D,
            GROUP_N=get_seq_group(N),
            BLOCK_M=32, BLOCK_N=64, BLOCK_D=triton.next_power_of_2(D),
            num_warps=4, num_stages=1,
        )

        # # naive backward
        # attn = torch.einsum("abhid,abhjd->abhij", q, k)* sm_scale + bias.unsqueeze(0)
        # attn = torch.softmax(attn, dim=-1)  # (A, B, H, N, N)
        # grad_attn = torch.einsum("abhid,abhjd->abhij", grad_output, v)
        # tmp = grad_attn * attn
        # grad_logits  = tmp - attn * tmp.sum(dim=-1, keepdim=True)
        # dq_naive = torch.einsum("abhij,abhjd->abhid", grad_logits, k) * sm_scale
        # diff = dq - dq_naive

        # if diff.abs().mean() > 1e-3:
        #     breakpoint()
        # if diff.abs().mean() > 1e-4:
        #     print(
        #         f"Warning: dq diff is {diff.abs().mean():.4f}, "
        #         "this may indicate a bug in the backward pass."
        #     )

        dq = dq.to(q.dtype)
        # dbias = dbias.to(bias.dtype)

        # fmt: on
        # calculate W_bias and pair bias gradients
        db_flat = rearrange(dbias, "b h n m -> (b n m) h", b=B, h=H).contiguous()

        pair_flat = pair.view(B * N * N, d_pair)      # (B*N*N, d_pair)

        dW_bias = db_flat.t().matmul(pair_flat.float())   # (H, d_pair)
        dpair_flat = db_flat.to(W_bias.dtype).matmul(W_bias)
        dpair = dpair_flat.view(B, N, N, d_pair)

        dW_bias = dW_bias.float()
        dpair = dpair.to(pair.dtype)

        return dq, dk, dv, dW_bias, dpair, None


triton_atom_augmented_attention = TritonAtomAugmentedAttentionFunction.apply
