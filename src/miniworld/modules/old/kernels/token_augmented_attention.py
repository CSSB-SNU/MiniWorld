import torch
import triton
import triton.language as tl
import os
from einops import rearrange, reduce


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
            {"BLOCK_M": 64, "BLOCK_N": 16, "BLOCK_D": 32}, num_stages=4, num_warps=8
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
    N_CTX,
    EVEN_N: tl.constexpr,
    EVEN_D: tl.constexpr,
):
    lo, hi = 0, N_CTX
    # loop over k, v and update accumulator
    offset_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offset_k = tl.arange(0, BLOCK_D)
    q_dtype = k_ptr.dtype.element_ty

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
            k = tl.load(k_ptr, mask=offset_n[None, :] < N_CTX, other=0.0)
            v = tl.load(v_ptr, mask=offset_n[:, None] < N_CTX, other=0.0)

        qk = tl.dot(q, k) + bias_val / (qk_scale / 1.44269504)
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]

        p = tl.math.exp2(qk)
        p = tl.maximum(p, 0.0)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        alpha = tl.maximum(alpha, 0.0)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        p = p.to(q_dtype)
        acc = tl.dot(p, v, acc)
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
    off_z = off_hz // H
    off_h = off_hz % H
    qkv_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    bias_offset = off_z * stride_bz + off_h * stride_bh

    offset_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offset_n = tl.arange(0, BLOCK_N)
    offset_k = tl.arange(0, BLOCK_D)

    q_ptr = (
        Q + qkv_offset + offset_m[:, None] * stride_qm + offset_k[None, :] * stride_qk
    )
    k_ptr = (
        K + qkv_offset + offset_n[None, :] * stride_kn + offset_k[:, None] * stride_kk
    )
    v_ptr = (
        V + qkv_offset + offset_n[:, None] * stride_vn + offset_k[None, :] * stride_vk
    )
    o_ptr = (
        Out + qkv_offset + offset_m[:, None] * stride_om + offset_k[None, :] * stride_on
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
    # m_ptrs = (
    #     M
    #     + off_z * stride_mz
    #     + off_h * stride_mh
    #     + off_t * stride_mt
    #     + start_m * BLOCK_M
    #     + tl.arange(0, BLOCK_M)
    # )
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
def _attn_bwd_dqdkdv(
    dk,
    dv,
    DBias,
    Q,
    k,
    v,
    Bias,
    qk_scale,  #
    DO,
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

    q_dtype = Q.dtype.element_ty
    bias_dtype = Bias.dtype.element_ty

    lo, hi = 0, N_CTX
    for start_m in range(lo, hi, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        qT_mask = (offs_m[None, :] < N_CTX) & (offs_k[:, None] < HEAD_DIM)
        do_mask = (offs_m[:, None] < N_CTX) & (offs_k[None, :] < HEAD_DIM)
        qT = tl.load(qT_ptrs, mask=qT_mask, other=0.0)

        bias_mask = (offs_m[None, :] < N_CTX) & (offs_n[:, None] < N_CTX)
        biasT = tl.load(biasT_ptrs, mask=bias_mask, other=float("-inf"))
        do = tl.load(do_ptrs, mask=do_mask, other=0.0)
        m = tl.load(m_ptrs, mask=offs_m < N_CTX, other=0.0)
        Di = tl.load(d_ptrs, mask=offs_m < N_CTX, other=0.0)

        qkT = tl.dot(k, qT) + biasT / (qk_scale / 1.44269504)
        qkT = qkT * qk_scale - m[None, :]

        pT = tl.math.exp2(qkT)
        # Compute dV.

        pT = pT.to(q_dtype)
        dv = tl.dot(pT, do, dv)

        # D (= delta) is pre-divided by ds_scale.
        # Compute dP and dS.
        dpT = tl.dot(v, tl.trans(do))
        dsT = pT * (dpT - Di[None, :])

        if DBias is not None:  # or just skip the check if always passing DBias
            if EVEN_N:
                tl.store(dbiasT_ptrs, dsT)
            else:
                mask = (offs_m[None, :] < N_CTX) & (offs_n[:, None] < N_CTX)
                tl.store(dbiasT_ptrs, dsT, mask=mask)

        dsT = dsT.to(bias_dtype)
        qT = qT.to(bias_dtype)
        dk = tl.dot(dsT, tl.trans(qT), dk)

        # Increment pointers.
        qT_ptrs += BLOCK_M * stride_tok
        do_ptrs += BLOCK_M * stride_tok
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
    DO, DK, DV, DBias,
    M, D,
    stride_z, stride_h, stride_tok, stride_d,
    bias_stride_z, bias_stride_h, bias_stride_m, bias_stride_n,
    H: tl.constexpr, N_CTX, HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    GROUP_N: tl.constexpr,
):
    tl.static_assert(BLOCK_D >= HEAD_DIM)

    bhid = tl.program_id(2)
    off_chz = (bhid * N_CTX).to(tl.int64)
    adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    pid = tl.program_id(0)

    # offset pointers for batch/head
    Q += adj
    K += adj
    V += adj
    DO += adj
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
    qk_scale = sm_scale * 1.44269504  # 1/log(2)
    dk, dv = _attn_bwd_dqdkdv(  #
        dk,
        dv,
        DBias,
        Q,
        k,
        v,
        Bias,
        qk_scale,  #
        DO,
        M,
        D,  #
        EVEN_N,
        EVEN_D,  #
        stride_tok,
        stride_d,  #
        H,
        N_CTX,  #
        BLOCK_M,
        BLOCK_N,
        BLOCK_D,
        HEAD_DIM,  #
        start_n,
        start_m=0,
        stride_bm=bias_stride_m,
        stride_bn=bias_stride_n,
    )

    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    dk = dk * sm_scale
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


class TritonAugmentedAttentionPairBiasFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bias: torch.Tensor,
    ):
        op_dtype = q.dtype
        if not q.dtype == k.dtype == v.dtype:
            raise ValueError(
                "q, k, v must have the same dtype, "
                f"but got {q.dtype=}, {k.dtype=}, {v.dtype=}"
            )
        if not q.shape == k.shape == v.shape:
            raise ValueError(
                "q, k, v must have the same shape, "
                f"but got {q.shape=}, {k.shape=}, {v.shape=}"
            )

        A, B, H, N, D = q.shape
        q, k, v = [
            rearrange(x, "A B H N D -> (A B) H N D") for x in (q, k, v)
        ]
        bias = rearrange(bias, "B N1 N2 H-> B H N1 N2").contiguous()
        bias = bias.expand(A*B, H, N, N)

        q, k, v = [x.contiguous() for x in (q, k, v)]

        sm_scale = D**-0.5
        o = torch.empty_like(q)
        M = torch.empty(A*B, H, N, device=q.device, dtype=torch.float32)

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
        ctx.save_for_backward(q, k, v, bias, o, M)
        ctx.op_dtype = op_dtype
        o = rearrange(o, "(A B) H N D -> A B H N D", B=B, A=A)
        ctx.shape = (A, B, H, N, D)
        return o

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        q, k, v, bias, o, M = ctx.saved_tensors
        op_dtype = ctx.op_dtype
        q, k, v, bias, o, M = [
            x.to(op_dtype).contiguous() for x in (q, k, v, bias, o, M)
        ]
        grad_output = grad_output.contiguous()
        grad_output = rearrange(
            grad_output, "A B H N D -> (A B) H N D", B=ctx.shape[1], A=ctx.shape[0]
        )

        AB, H, N, D = q.shape
        sm_scale = D**-0.5
        delta = torch.empty_like(M)

        # fmt: off
        grid = lambda META: (triton.cdiv(N, META["BLOCK_M"]), AB * H, 1)
        _attn_bwd_preprocess[grid](
            o, grad_output, delta,
            AB, H, N, D,
            BLOCK_D=64,
            GROUP_N=get_seq_group(N),
        )
        # fmt: on

        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        bias = bias.contiguous() # <- potential issue (we can edit grid to speed up)
        dbias = torch.empty_like(bias).contiguous()

        # fmt: off
        grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]), 1, AB * H)
        _attn_bwd[grid](
            q, k, v, bias, sm_scale,
            grad_output, dk, dv, dbias,
            M, delta,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            bias.stride(0), bias.stride(1), bias.stride(2), bias.stride(3),
            H, N, D,
            GROUP_N=get_seq_group(N),
            BLOCK_M=64, BLOCK_N=128, BLOCK_D=triton.next_power_of_2(D),
            num_warps=8, num_stages=1,
        )
        # without atomic_add
        dq = torch.einsum('ahjd,ahij->ahid', k, dbias.to(k.dtype)) * sm_scale

        # fmt: on
        dbias = reduce(
            dbias, "(A B) H N1 N2 -> B N1 N2 H", "sum", B=ctx.shape[1], A=ctx.shape[0]
        ).contiguous()
        dq, dk, dv = [
            rearrange(x, "(A B) H N D -> A B H N D", B=ctx.shape[1], A=ctx.shape[0])
            for x in (dq, dk, dv)
        ]

        # if torch.isnan(dq).any() or torch.isnan(dk).any() or torch.isnan(dv).any() or torch.isnan(dbias).any():
        #     to_save = [
        #         q, k, v, bias, o, M, grad_output, dq, dk, dv, dbias
        #     ]
        #     torch.save(
        #         to_save, "debug_attention_pair_bias.pt"
        #     )
        #     raise RuntimeError("NaN detected in dq, dk, or dv gradients.")

        return dq, dk, dv, dbias


triton_token_augmented_attention = TritonAugmentedAttentionPairBiasFunction.apply

if __name__ == "__main__":
    # This is just for testing the function.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    to_debug = torch.load("repaired.pt", map_location=device, weights_only=False)
    
    q, k, v, bias, o, M, grad_output, dq, dk, dv, dbias = to_debug
    q, k, v, bias, grad_output = [t.to(device) for t in (q, k, v, bias, grad_output)]

    # make each tensor a *fresh leaf* with requires_grad=True
    q, k, v = [t.unsqueeze(1).detach().requires_grad_() for t in (q, k, v)]
    grad_output = grad_output.unsqueeze(1)
    bias = bias[0:1]
    bias = rearrange(bias, "B H N1 N2 -> B N1 N2 H").to(device).contiguous()
    bias = bias.detach().requires_grad_()
    out = TritonAugmentedAttentionPairBiasFunction.apply(q, k, v, bias)
    out.backward(grad_output)

    dq_, dk_, dv_, dbias_ = q.grad, k.grad, v.grad, bias.grad
    dq, dk, dv, dbias = [x.to(device) for x in (dq, dk, dv, dbias)]
    diff_dq = torch.abs(dq - dq_).max()
    diff_dk = torch.abs(dk - dk_).max()
    diff_dv = torch.abs(dv - dv_).max()
    diff_dbias = torch.abs(dbias - dbias_).max()

    print(f"Max diff in dq: {diff_dq.item()}")
    print(f"Max diff in dk: {diff_dk.item()}")
    print(f"Max diff in dv: {diff_dv.item()}")
    print(f"Max diff in dbias: {diff_dbias.item()}")
    breakpoint()
