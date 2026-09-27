"""Fused ESMFold2 SWA atom DiT block (team_gm SWAAtomBlock, block_style "esmfold2"), bf16, Triton.

Block (per row r of the flattened [N = A*B, S] atom sequence; the adaLN modulation depends only on (b, atom) and is hoisted):
    mod = silu(c) @ Wmod^T -> shift_a | scale_a | gate_a | shift_f | scale_f | gate_f           (per (b, atom), once)
    x   = rmsnorm(q) * (1 + scale_a) + shift_a                                                    (bf16)
    q_h, k_h, v_h = split_heads(x @ Wqkv^T);  q_h, k_h = rope(rmsnorm_D(q_h)), rope(rmsnorm_D(k_h))
    o   = sliding-window attention (|i - j| <= 64, keys / queries >= seqused masked, padding rows 0)
    a   = (sigmoid(x @ Wg^T) * o) @ Wo^T ;  q = q + gate_a * a
    y   = rmsnorm(q) * (1 + scale_f) + shift_f ;  q = q + gate_f * ((silu(y Wu1^T) * (y Wu2^T)) @ Wd^T)
Kernels: `_qkvg_fwd` (norm + modulate + 4 projections + qk-norm + RoPE), `_attn_fwd` (window attention over all heads + gate +
out-proj + gated residual), `_ffn_fwd` (norm + modulate + SwiGLU + gated residual)."""
import math
import torch
import triton
import triton.language as tl

LAST = {}
FP32_EPS = float(torch.finfo(torch.float32).eps)


def _bucket(n):
    return int(math.log2(max(int(n), 1)))


@triton.jit
def _rope(x, cs, sn, BR: tl.constexpr, H: tl.constexpr, D: tl.constexpr):
    """x [BR, H*D] fp32 (per head: halves x1 | x2) -> x*cos + rotate_half(x)*sin, cos / sin [BR, D/2] shared by the heads."""
    HALF: tl.constexpr = D // 2
    x4 = tl.reshape(x, (BR, H, 2, HALF))
    x1, x2 = tl.split(tl.permute(x4, (0, 1, 3, 2)))                                  # [BR, H, HALF] each
    c = cs[:, None, :]; s = sn[:, None, :]
    y1 = x1 * c - x2 * s
    y2 = x2 * c + x1 * s
    return tl.reshape(tl.permute(tl.join(y1, y2), (0, 1, 3, 2)), (BR, H * D))


@triton.jit
def _head_rms(x, eps, BR: tl.constexpr, H: tl.constexpr, D: tl.constexpr):
    x3 = tl.reshape(x, (BR, H, D))
    r = 1.0 / tl.sqrt(tl.sum(x3 * x3, axis=2) / D + eps)
    return tl.reshape(x3 * r[:, :, None], (BR, H * D))


# ================================================================ ① qkvg ===============================================================
@triton.autotune(configs=[triton.Config({"BR": br}, num_warps=w, num_stages=st) for br in (32, 64, 128) for w in (4, 8) for st in (1, 2)],
                 key=["SB"])
@triton.jit(do_not_specialize=["NROWS", "S", "B"])
def _qkvg_fwd(Q, MOD, COS, SIN, WQKV, WG, QO, KO, VO, GO, RSTD, NROWS, S, B, eps, qk_eps, SB,
              C: tl.constexpr, H: tl.constexpr, D: tl.constexpr, MODW: tl.constexpr, BR: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    rows = pid * BR + tl.arange(0, BR).to(tl.int64); ok = rows < NROWS
    cc = tl.arange(0, C).to(tl.int64)
    n = rows // S; s = rows - n * S; mrow = (n % B) * S + s                          # (b, atom) row of the hoisted modulation
    m2 = ok[:, None]
    q = tl.load(Q + rows[:, None] * C + cc[None, :], mask=m2, other=0.0).to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(q * q, axis=1) / C + eps)
    tl.store(RSTD + rows, rstd, mask=ok)
    shift = tl.load(MOD + mrow[:, None] * MODW + cc[None, :], mask=m2, other=0.0).to(tl.float32)
    scale = tl.load(MOD + mrow[:, None] * MODW + C + cc[None, :], mask=m2, other=0.0).to(tl.float32)
    x = (q * rstd[:, None] * (1.0 + scale) + shift).to(tl.bfloat16)
    HALF: tl.constexpr = D // 2
    hc = tl.arange(0, HALF).to(tl.int64)
    cs = tl.load(COS + mrow[:, None] * HALF + hc[None, :], mask=m2, other=1.0)
    sn = tl.load(SIN + mrow[:, None] * HALF + hc[None, :], mask=m2, other=0.0)
    wofs = cc[None, :] * C + cc[:, None]                                              # W^T tile [in][out]
    # q, k: projection -> bf16 -> per-head RMSNorm (fp32 eps) -> bf16 -> RoPE -> bf16
    pq = tl.dot(x, tl.load(WQKV + wofs)).to(tl.bfloat16).to(tl.float32)
    pq = _head_rms(pq, qk_eps, BR, H, D).to(tl.bfloat16).to(tl.float32)
    tl.store(QO + rows[:, None] * C + cc[None, :], _rope(pq, cs, sn, BR, H, D).to(tl.bfloat16), mask=m2)
    pk = tl.dot(x, tl.load(WQKV + C * C + wofs)).to(tl.bfloat16).to(tl.float32)
    pk = _head_rms(pk, qk_eps, BR, H, D).to(tl.bfloat16).to(tl.float32)
    tl.store(KO + rows[:, None] * C + cc[None, :], _rope(pk, cs, sn, BR, H, D).to(tl.bfloat16), mask=m2)
    pv = tl.dot(x, tl.load(WQKV + 2 * C * C + wofs))
    tl.store(VO + rows[:, None] * C + cc[None, :], pv.to(tl.bfloat16), mask=m2)
    pg = tl.dot(x, tl.load(WG + wofs))
    tl.store(GO + rows[:, None] * C + cc[None, :], pg.to(tl.bfloat16), mask=m2)


# ================================================================ ② window attention + gate + out-proj + residual =====================
@triton.autotune(configs=[triton.Config({"BM": bm, "BN": bn}, num_warps=w, num_stages=st) for bm, bn in ((64, 64), (64, 32), (32, 32), (128, 64))
                          for w in (4, 8) for st in (1, 2)], key=["SB", "HW"])
@triton.jit(do_not_specialize=["S", "B"])
def _attn_fwd(QI, QO_, KO, VO, GO, MOD, WO, SEQU, OUT, O, LSE, S, B, scale, SB,
              C: tl.constexpr, H: tl.constexpr, D: tl.constexpr, HW: tl.constexpr, MODW: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    """Program = (query tile of BM rows, sequence n); all heads. Keys j with |i - j| <= HW and j < seqused[n]."""
    t = tl.program_id(0); n = tl.program_id(1).to(tl.int64)
    su = tl.load(SEQU + n)
    i0 = t * BM
    qi = i0 + tl.arange(0, BM)
    qok = qi < su
    base = n * S
    cc = tl.arange(0, C).to(tl.int64); dc = tl.arange(0, D).to(tl.int64)
    o_all = tl.zeros((BM, C), dtype=tl.float32)
    lo = tl.maximum(i0 - HW, 0) // BN * BN
    hi = tl.minimum(i0 + BM + HW, su)
    for h in tl.static_range(H):
        qh = tl.load(QO_ + (base + qi)[:, None] * C + h * D + dc[None, :], mask=qok[:, None], other=0.0)
        m = tl.full((BM,), -float("inf"), tl.float32); l = tl.zeros((BM,), tl.float32); acc = tl.zeros((BM, D), tl.float32)
        for j0 in range(lo, hi, BN):
            kj = j0 + tl.arange(0, BN)
            kok = kj < su
            kh = tl.load(KO + (base + kj)[:, None] * C + h * D + dc[None, :], mask=kok[:, None], other=0.0)
            vh = tl.load(VO + (base + kj)[:, None] * C + h * D + dc[None, :], mask=kok[:, None], other=0.0)
            sc = tl.dot(qh, tl.trans(kh)) * scale
            dij = qi[:, None] - kj[None, :]
            allow = (dij <= HW) & (dij >= -HW) & kok[None, :]
            sc = tl.where(allow, sc, -float("inf"))
            mn = tl.maximum(m, tl.max(sc, axis=1))
            mn_s = tl.where(mn == -float("inf"), 0.0, mn)
            p = tl.exp(sc - mn_s[:, None])
            alpha = tl.exp(m - mn_s)
            l = l * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), vh)
            m = mn
        l_s = tl.where(l == 0.0, 1.0, l)
        oh = acc / l_s[:, None]
        tl.store(LSE + (n * H + h) * S + qi, tl.where(l > 0.0, m + tl.log(l_s), 0.0), mask=qi < S)
        o_all = tl.where((cc[None, :] // D) == h, tl.reshape(tl.broadcast_to(oh[:, None, :], (BM, H, D)), (BM, C)), o_all)
    o_all = tl.where(qok[:, None], o_all, 0.0).to(tl.bfloat16)                          # FA output (bf16), padding rows 0
    qs = qi < S
    rows = base + qi
    tl.store(O + rows[:, None] * C + cc[None, :], o_all, mask=qs[:, None])
    g = tl.load(GO + rows[:, None] * C + cc[None, :], mask=qs[:, None], other=0.0).to(tl.float32)
    gated = (tl.sigmoid(g) * o_all.to(tl.float32)).to(tl.bfloat16)
    att = tl.dot(gated, tl.load(WO + cc[None, :] * C + cc[:, None])).to(tl.bfloat16)   # out_proj (bf16 out)
    mrow = (n % B) * S + qi
    ga = tl.load(MOD + mrow[:, None] * MODW + 2 * C + cc[None, :], mask=qs[:, None], other=0.0).to(tl.bfloat16)
    qin = tl.load(QI + rows[:, None] * C + cc[None, :], mask=qs[:, None], other=0.0)
    tl.store(OUT + rows[:, None] * C + cc[None, :], (qin + (ga * att)), mask=qs[:, None])


# ================================================================ ③ FFN ================================================================
@triton.autotune(configs=[triton.Config({"BR": br, "NH": nh}, num_warps=w, num_stages=st) for br in (32, 64, 128) for nh in (64, 128)
                          for w in (4, 8) for st in (1, 2)], key=["SB"])
@triton.jit(do_not_specialize=["NROWS", "S", "B"])
def _ffn_fwd(Q, MOD, WU, WD, OUT, RSTD, NROWS, S, B, eps, SB,
             C: tl.constexpr, NHID: tl.constexpr, MODW: tl.constexpr, BR: tl.constexpr, NH: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    rows = pid * BR + tl.arange(0, BR).to(tl.int64); ok = rows < NROWS
    cc = tl.arange(0, C).to(tl.int64)
    n = rows // S; s = rows - n * S; mrow = (n % B) * S + s
    m2 = ok[:, None]
    qb = tl.load(Q + rows[:, None] * C + cc[None, :], mask=m2, other=0.0)
    q = qb.to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(q * q, axis=1) / C + eps)
    tl.store(RSTD + rows, rstd, mask=ok)
    shift = tl.load(MOD + mrow[:, None] * MODW + 3 * C + cc[None, :], mask=m2, other=0.0).to(tl.float32)
    scale = tl.load(MOD + mrow[:, None] * MODW + 4 * C + cc[None, :], mask=m2, other=0.0).to(tl.float32)
    y = (q * rstd[:, None] * (1.0 + scale) + shift).to(tl.bfloat16)
    nh = tl.arange(0, NH).to(tl.int64)
    acc = tl.zeros((BR, C), dtype=tl.float32)
    for j0 in range(0, NHID, NH):
        hid = j0 + nh
        a = tl.dot(y, tl.load(WU + hid[None, :] * C + cc[:, None]))                     # [BR, NH] = y Wu1[hid]^T
        b = tl.dot(y, tl.load(WU + (NHID + hid)[None, :] * C + cc[:, None]))
        hh = (a * tl.sigmoid(a) * b).to(tl.bfloat16)
        acc += tl.dot(hh, tl.load(WD + cc[None, :] * NHID + hid[:, None]))            # Wd [C][NHID] -> [NH][C]
    gf = tl.load(MOD + mrow[:, None] * MODW + 5 * C + cc[None, :], mask=m2, other=0.0).to(tl.bfloat16)
    tl.store(OUT + rows[:, None] * C + cc[None, :], qb + gf * acc.to(tl.bfloat16), mask=m2)


# ================================================================ host =================================================================
def block_fwd(q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, B, half_window=64, eps=FP32_EPS, save=False):
    """q [N, S, C] bf16; mod [B*S, 6C] fp32 (hoisted adaLN modulation of this block); cos / sin [B*S, D/2] fp32; seqused [N] int32."""
    N, S, C = q.shape; H = 4; D = C // H; M = N * S
    qf = q.reshape(M, C)
    Qh = torch.empty_like(qf); Kh = torch.empty_like(qf); Vh = torch.empty_like(qf); G = torch.empty_like(qf)
    r1 = torch.empty(M, device=q.device); sb = _bucket(M)
    LAST["qkvg"] = _qkvg_fwd[lambda m: (triton.cdiv(M, m["BR"]),)](qf, mod, cos, sin, wqkv, wg, Qh, Kh, Vh, G, r1, M, S, B, eps, FP32_EPS, sb,
                                                                   C=C, H=H, D=D, MODW=6 * C)
    q1 = torch.empty_like(qf); O = torch.empty_like(qf); lse = torch.empty(N, H, S, device=q.device)
    LAST["attn"] = _attn_fwd[lambda m: (triton.cdiv(S, m["BM"]), N)](qf, Qh, Kh, Vh, G, mod, wo, seqused, q1, O, lse, S, B, D ** -0.5, sb,
                                                                     C=C, H=H, D=D, HW=half_window, MODW=6 * C)
    q2 = torch.empty_like(qf); r2 = torch.empty(M, device=q.device)
    NHID = wd.shape[1]
    LAST["ffn"] = _ffn_fwd[lambda m: (triton.cdiv(M, m["BR"]),)](q1, mod, wu, wd, q2, r2, M, S, B, eps, sb, C=C, NHID=NHID, MODW=6 * C)
    out = q2.view(N, S, C)
    if save:
        return out, dict(Qh=Qh, Kh=Kh, Vh=Vh, G=G, O=O, lse=lse, q1=q1, r1=r1, r2=r2)
    return out


def hoist_mod(c1, wmod):
    """c1 [B, S, d_cond] (augment-invariant conditioning), wmod [6C, d_cond] -> [B*S, 6C] fp32 = silu(c1) (in c1's dtype) @ wmod^T,
    fp32 accumulate and kept fp32 (as the engine's rmsnorm_adamod keeps its projections in registers)."""
    a = torch.nn.functional.silu(c1).float()
    return (a.reshape(-1, a.shape[-1]) @ wmod.float().t()).contiguous()
