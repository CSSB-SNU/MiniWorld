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
    # head-major destinations [N, H, S, D]: element (row, h*D + d) -> ((n*H + h)*S + s)*D + d
    hm = ((n[:, None] * H + (cc[None, :] // D)) * S + s[:, None]) * D + (cc[None, :] % D)
    pq = tl.dot(x, tl.load(WQKV + wofs)).to(tl.bfloat16).to(tl.float32)
    pq = _head_rms(pq, qk_eps, BR, H, D).to(tl.bfloat16).to(tl.float32)
    tl.store(QO + hm, _rope(pq, cs, sn, BR, H, D).to(tl.bfloat16), mask=m2)
    pk = tl.dot(x, tl.load(WQKV + C * C + wofs)).to(tl.bfloat16).to(tl.float32)
    pk = _head_rms(pk, qk_eps, BR, H, D).to(tl.bfloat16).to(tl.float32)
    tl.store(KO + hm, _rope(pk, cs, sn, BR, H, D).to(tl.bfloat16), mask=m2)
    pv = tl.dot(x, tl.load(WQKV + 2 * C * C + wofs))
    tl.store(VO + hm, pv.to(tl.bfloat16), mask=m2)
    pg = tl.dot(x, tl.load(WG + wofs))
    tl.store(GO + rows[:, None] * C + cc[None, :], pg.to(tl.bfloat16), mask=m2)


# ================================================================ ② window attention (one head per program) ==========================
@triton.autotune(configs=[triton.Config({"BM": bm, "BN": bn}, num_warps=w, num_stages=st) for bm, bn in ((64, 64), (128, 64), (64, 32), (128, 32))
                          for w in (4, 8) for st in (1, 2, 3)], key=["SB", "HW"])
@triton.jit(do_not_specialize=["S"])
def _attn_fwd(QH, KH, VH, SEQU, O, LSE, S, scale, SB,
              C: tl.constexpr, H: tl.constexpr, D: tl.constexpr, HW: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    """Program = (query tile, n*H + h).  Q / K / V head-major [N, H, S, D]; keys j with |i - j| <= HW and j < seqused[n].
    O [N, S, C] (bf16, head h in columns h*D..), padding query rows 0; LSE [N, H, S]."""
    t = tl.program_id(0); nh = tl.program_id(1).to(tl.int64)
    n = nh // H; h = nh - n * H
    su = tl.load(SEQU + n)
    i0 = t * BM
    qi = i0 + tl.arange(0, BM); qok = qi < su
    dc = tl.arange(0, D).to(tl.int64)
    hb = nh * S * D
    qh = tl.load(QH + hb + qi[:, None] * D + dc[None, :], mask=qok[:, None], other=0.0)
    m = tl.full((BM,), -float("inf"), tl.float32); l = tl.zeros((BM,), tl.float32); acc = tl.zeros((BM, D), tl.float32)
    lo = tl.maximum(i0 - HW, 0) // BN * BN
    hi = tl.minimum(i0 + BM + HW, su)
    for j0 in range(lo, hi, BN):
        kj = j0 + tl.arange(0, BN); kok = kj < su
        kh = tl.load(KH + hb + kj[:, None] * D + dc[None, :], mask=kok[:, None], other=0.0)
        vh = tl.load(VH + hb + kj[:, None] * D + dc[None, :], mask=kok[:, None], other=0.0)
        sc = tl.dot(qh, tl.trans(kh)) * scale
        dij = qi[:, None] - kj[None, :]
        sc = tl.where((dij <= HW) & (dij >= -HW) & kok[None, :], sc, -float("inf"))
        mn = tl.maximum(m, tl.max(sc, axis=1))
        mn_s = tl.where(mn == -float("inf"), 0.0, mn)
        p = tl.exp(sc - mn_s[:, None])
        alpha = tl.exp(m - mn_s)
        l = l * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), vh)
        m = mn
    l_s = tl.where(l == 0.0, 1.0, l)
    o = tl.where(qok[:, None], acc / l_s[:, None], 0.0)
    qs = qi < S
    tl.store(LSE + nh * S + qi, tl.where(l > 0.0, m + tl.log(l_s), 0.0), mask=qs)
    tl.store(O + (n * S + qi)[:, None] * C + h * D + dc[None, :], o.to(tl.bfloat16), mask=qs[:, None])


# ================================================================ ③ out-proj + gated residual + FFN ===================================
@triton.autotune(configs=[triton.Config({"BR": br, "NH": nh}, num_warps=w, num_stages=st) for br in (32, 64, 128) for nh in (64, 128)
                          for w in (4, 8) for st in (1, 2)], key=["SB", "SAVE"])
@triton.jit(do_not_specialize=["NROWS", "S", "B"])
def _oproj_ffn_fwd(QI, O, G, MOD, WO, WU, WD, Q1, OUT, RSTD, NROWS, S, B, eps, SB,
                   C: tl.constexpr, NHID: tl.constexpr, MODW: tl.constexpr, SAVE: tl.constexpr, BR: tl.constexpr, NH: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    rows = pid * BR + tl.arange(0, BR).to(tl.int64); ok = rows < NROWS
    cc = tl.arange(0, C).to(tl.int64)
    n = rows // S; s = rows - n * S; mrow = (n % B) * S + s
    m2 = ok[:, None]
    ofs = rows[:, None] * C + cc[None, :]
    g = tl.load(G + ofs, mask=m2, other=0.0).to(tl.float32)
    gated = (tl.sigmoid(g) * tl.load(O + ofs, mask=m2, other=0.0).to(tl.float32)).to(tl.bfloat16)
    att = tl.dot(gated, tl.load(WO + cc[None, :] * C + cc[:, None])).to(tl.bfloat16)
    ga = tl.load(MOD + mrow[:, None] * MODW + 2 * C + cc[None, :], mask=m2, other=0.0).to(tl.bfloat16)
    q1 = tl.load(QI + ofs, mask=m2, other=0.0) + ga * att                             # bf16
    if SAVE:
        tl.store(Q1 + ofs, q1, mask=m2)
    q = q1.to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(q * q, axis=1) / C + eps)
    if SAVE:
        tl.store(RSTD + rows, rstd, mask=ok)
    shift = tl.load(MOD + mrow[:, None] * MODW + 3 * C + cc[None, :], mask=m2, other=0.0).to(tl.float32)
    scale = tl.load(MOD + mrow[:, None] * MODW + 4 * C + cc[None, :], mask=m2, other=0.0).to(tl.float32)
    y = (q * rstd[:, None] * (1.0 + scale) + shift).to(tl.bfloat16)
    nh = tl.arange(0, NH).to(tl.int64)
    acc = tl.zeros((BR, C), dtype=tl.float32)
    for j0 in range(0, NHID, NH):
        hid = j0 + nh
        a = tl.dot(y, tl.load(WU + hid[None, :] * C + cc[:, None]))
        b = tl.dot(y, tl.load(WU + (NHID + hid)[None, :] * C + cc[:, None]))
        hh = (a * tl.sigmoid(a) * b).to(tl.bfloat16)
        acc += tl.dot(hh, tl.load(WD + cc[None, :] * NHID + hid[:, None]))
    gf = tl.load(MOD + mrow[:, None] * MODW + 5 * C + cc[None, :], mask=m2, other=0.0).to(tl.bfloat16)
    tl.store(OUT + ofs, q1 + gf * acc.to(tl.bfloat16), mask=m2)


# ================================================================ host =================================================================
def block_fwd(q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, B, half_window=64, eps=FP32_EPS, save=False):
    """q [N, S, C] bf16; mod [B*S, 6C] fp32 (hoisted adaLN modulation of this block); cos / sin [B*S, D/2] fp32; seqused [N] int32."""
    N, S, C = q.shape; H = 4; D = C // H; M = N * S
    qf = q.reshape(M, C)
    Qh = torch.empty(N, H, S, D, device=q.device, dtype=q.dtype); Kh = torch.empty_like(Qh); Vh = torch.empty_like(Qh); G = torch.empty_like(qf)
    r1 = torch.empty(M, device=q.device); sb = _bucket(M)
    LAST["qkvg"] = _qkvg_fwd[lambda m: (triton.cdiv(M, m["BR"]),)](qf, mod, cos, sin, wqkv, wg, Qh, Kh, Vh, G, r1, M, S, B, eps, FP32_EPS, sb,
                                                                   C=C, H=H, D=D, MODW=6 * C)
    O = torch.empty_like(qf); lse = torch.empty(N, H, S, device=q.device)
    LAST["attn"] = _attn_fwd[lambda m: (triton.cdiv(S, m["BM"]), N * H)](Qh, Kh, Vh, seqused, O, lse, S, D ** -0.5, sb, C=C, H=H, D=D, HW=half_window)
    q2 = torch.empty_like(qf)
    q1 = torch.empty_like(qf) if save else q2; r2 = torch.empty(M, device=q.device) if save else r1
    NHID = wd.shape[1]
    LAST["ffn"] = _oproj_ffn_fwd[lambda m: (triton.cdiv(M, m["BR"]),)](qf, O, G, mod, wo, wu, wd, q1, q2, r2, M, S, B, eps, sb,
                                                                       C=C, NHID=NHID, MODW=6 * C, SAVE=save)
    out = q2.view(N, S, C)
    if save:
        return out, dict(Qh=Qh, Kh=Kh, Vh=Vh, G=G, O=O, lse=lse, q1=q1, r1=r1, r2=r2)
    return out


def hoist_mod(c1, wmod):
    """c1 [B, S, d_cond] (augment-invariant conditioning), wmod [6C, d_cond] -> [B*S, 6C] fp32 = silu(c1) (in c1's dtype) @ wmod^T,
    fp32 accumulate and kept fp32 (as the engine's rmsnorm_adamod keeps its projections in registers)."""
    a = torch.nn.functional.silu(c1).float()
    return (a.reshape(-1, a.shape[-1]) @ wmod.float().t()).contiguous()


# ================================================================ backward ============================================================
# Tiles for the kernels that reduce the (augment-invariant) modulation gradient: SP augments x AT atoms of one batch element b;
# rows r = ((a*B + b)*S + s).  Pre-summed over the tile's augments, then one atomic add per (atom, channel).
@triton.jit
def _tile_rows(S, A, B, SP: tl.constexpr, AT: tl.constexpr):
    ab = tl.program_id(0).to(tl.int64); ag = tl.program_id(1).to(tl.int64); b = tl.program_id(2).to(tl.int64)
    R: tl.constexpr = SP * AT
    r = tl.arange(0, R).to(tl.int64)
    a = ag * SP + r // AT; s = ab * AT + r % AT
    ok = (a < A) & (s < S)
    rows = (a * B + b) * S + s
    return rows, ok, b * S + s, s


@triton.jit
def _presum_add(DMOD, x, col0, ab_idx, b, S, ok_at, C: tl.constexpr, MODW: tl.constexpr, SP: tl.constexpr, AT: tl.constexpr):
    """x [SP*AT, C] (zero on invalid rows) -> sum over the SP augments -> atomic add into DMOD[(b*S + atom), col0 + c]."""
    cc = tl.arange(0, C).to(tl.int64)
    ar = ab_idx * AT + tl.arange(0, AT).to(tl.int64)
    tl.atomic_add(DMOD + (b * S + ar)[:, None] * MODW + col0 + cc[None, :], tl.sum(tl.reshape(x, (SP, AT, C)), axis=0),
                  mask=ok_at[:, None], sem="relaxed")


@triton.autotune(configs=[triton.Config({"SP": sp, "AT": at, "NH": nh}, num_warps=w, num_stages=1)
                          for sp, at in ((2, 16), (4, 8), (2, 32), (4, 16)) for nh in (64, 128) for w in (4, 8)],
                 key=["SB"], restore_value=["DMOD"])
@triton.jit(do_not_specialize=["S", "A", "B"])
def _post_bwd(DQ2, Q1, O, G, MOD, WO, WU, WD, DQ1, DOUT_, DG, DVv, DAB, HH, DFFN, YN, DATT, GATED, DMOD, S, A, B, eps, SB,
              C: tl.constexpr, H: tl.constexpr, D: tl.constexpr, NHID: tl.constexpr, MODW: tl.constexpr,
              SP: tl.constexpr, AT: tl.constexpr, NH: tl.constexpr):
    """FFN backward (hidden recomputed per chunk) + out-proj / sigmoid-gate backward.  Writes dq1 (fp32, the residual stream grad
    before the attention path), dO (bf16), dG (bf16), D = per-head rowsum(dO*O), the dW operands, and the modulation grads."""
    rows, ok, mrow, s_ = _tile_rows(S, A, B, SP, AT)
    b = tl.program_id(2).to(tl.int64); ab_idx = tl.program_id(0).to(tl.int64)
    ok_at = (ab_idx * AT + tl.arange(0, AT)) < S
    cc = tl.arange(0, C).to(tl.int64)
    m2 = ok[:, None]
    ofs = rows[:, None] * C + cc[None, :]
    dq2 = tl.load(DQ2 + ofs, mask=m2, other=0.0).to(tl.float32)
    q1b = tl.load(Q1 + ofs, mask=m2, other=0.0)
    q1 = q1b.to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(q1 * q1, axis=1) / C + eps)
    xh = q1 * rstd[:, None]
    shift = tl.load(MOD + mrow[:, None] * MODW + 3 * C + cc[None, :], mask=m2, other=0.0)
    scale = tl.load(MOD + mrow[:, None] * MODW + 4 * C + cc[None, :], mask=m2, other=0.0)
    gf = tl.load(MOD + mrow[:, None] * MODW + 5 * C + cc[None, :], mask=m2, other=0.0).to(tl.bfloat16).to(tl.float32)
    y = (xh * (1.0 + scale) + shift).to(tl.bfloat16)
    tl.store(YN + ofs, y, mask=m2)
    dffn = (dq2 * gf).to(tl.bfloat16)
    tl.store(DFFN + ofs, dffn, mask=m2)
    nh = tl.arange(0, NH).to(tl.int64)
    ffn = tl.zeros((SP * AT, C), dtype=tl.float32)
    dy = tl.zeros((SP * AT, C), dtype=tl.float32)
    for j0 in range(0, NHID, NH):
        hid = j0 + nh
        wu1T = tl.load(WU + hid[None, :] * C + cc[:, None]); wu2T = tl.load(WU + (NHID + hid)[None, :] * C + cc[:, None])   # [C][NH]
        a = tl.dot(y, wu1T); bb = tl.dot(y, wu2T)
        sa = tl.sigmoid(a)
        hb = (a * sa * bb).to(tl.bfloat16)
        wdT = tl.load(WD + cc[None, :] * NHID + hid[:, None])                                        # [NH][C] = Wd[:, hid]^T
        ffn += tl.dot(hb, wdT)
        dh = tl.dot(dffn, tl.trans(wdT))                                                             # [R][NH]
        da = (dh * bb * sa * (1.0 + a * (1.0 - sa))).to(tl.bfloat16)
        db = (dh * a * sa).to(tl.bfloat16)
        dy += tl.dot(da, tl.trans(wu1T)) + tl.dot(db, tl.trans(wu2T))
        hm = ok[:, None]
        tl.store(DAB + rows[:, None] * (2 * NHID) + hid[None, :], da, mask=hm)
        tl.store(DAB + rows[:, None] * (2 * NHID) + NHID + hid[None, :], db, mask=hm)
        tl.store(HH + rows[:, None] * NHID + hid[None, :], hb, mask=hm)
    ffn = ffn.to(tl.bfloat16).to(tl.float32)
    _presum_add(DMOD, tl.where(m2, dq2 * ffn, 0.0), 5 * C, ab_idx, b, S, ok_at, C, MODW, SP, AT)      # d gate_f
    _presum_add(DMOD, tl.where(m2, dy * xh, 0.0), 4 * C, ab_idx, b, S, ok_at, C, MODW, SP, AT)        # d scale_f
    _presum_add(DMOD, tl.where(m2, dy, 0.0), 3 * C, ab_idx, b, S, ok_at, C, MODW, SP, AT)             # d shift_f
    dxh = dy * (1.0 + scale)
    dq1 = dq2 + rstd[:, None] * (dxh - xh * (tl.sum(dxh * xh, axis=1) / C)[:, None])
    # ---- out-proj + gate: q1 = q + ga * ((sigmoid(g) * O) Wo^T) ----
    g = tl.load(G + ofs, mask=m2, other=0.0).to(tl.float32); so = tl.sigmoid(g)
    o = tl.load(O + ofs, mask=m2, other=0.0).to(tl.float32)
    gated = (so * o).to(tl.bfloat16)
    tl.store(GATED + ofs, gated, mask=m2)
    woT = tl.load(WO + cc[None, :] * C + cc[:, None])                                               # [in][out]
    att = tl.dot(gated, woT).to(tl.bfloat16).to(tl.float32)
    ga = tl.load(MOD + mrow[:, None] * MODW + 2 * C + cc[None, :], mask=m2, other=0.0).to(tl.bfloat16).to(tl.float32)
    _presum_add(DMOD, tl.where(m2, dq1 * att, 0.0), 2 * C, ab_idx, b, S, ok_at, C, MODW, SP, AT)       # d gate_a
    datt = (dq1 * ga).to(tl.bfloat16)
    tl.store(DATT + ofs, datt, mask=m2)
    dgated = tl.dot(datt, tl.trans(woT))                                                            # datt Wo
    do_ = dgated * so
    tl.store(DOUT_ + ofs, do_.to(tl.bfloat16), mask=m2)
    tl.store(DG + ofs, (dgated * o * so * (1.0 - so)).to(tl.bfloat16), mask=m2)
    dvv = tl.sum(tl.reshape(do_.to(tl.bfloat16).to(tl.float32) * o, (SP * AT, H, D)), axis=2)          # D = rowsum(dO * O) per head
    hh_ = tl.arange(0, H).to(tl.int64)
    n = rows // S
    tl.store(DVv + (n[:, None] * H + hh_[None, :]) * S + s_[:, None], dvv, mask=m2)
    tl.store(DQ1 + ofs, dq1, mask=m2)


@triton.autotune(configs=[triton.Config({"BM": bm, "BN": bn}, num_warps=w, num_stages=st) for bm, bn in ((64, 64), (64, 32), (128, 64), (32, 32))
                          for w in (4, 8) for st in (1, 2)], key=["SB", "HW"])
@triton.jit(do_not_specialize=["S"])
def _attn_bwd_dq(QH, KH, VH, DOUT_, LSE, DVv, SEQU, DQH, S, scale, SB,
                 C: tl.constexpr, H: tl.constexpr, D: tl.constexpr, HW: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    t = tl.program_id(0); nh = tl.program_id(1).to(tl.int64)
    n = nh // H; h = nh - n * H
    su = tl.load(SEQU + n)
    i0 = t * BM
    qi = i0 + tl.arange(0, BM); qok = qi < su
    dc = tl.arange(0, D).to(tl.int64)
    hb = nh * S * D
    q = tl.load(QH + hb + qi[:, None] * D + dc[None, :], mask=qok[:, None], other=0.0)
    do_ = tl.load(DOUT_ + (n * S + qi)[:, None] * C + h * D + dc[None, :], mask=qok[:, None], other=0.0)
    lse = tl.load(LSE + nh * S + qi, mask=qok, other=0.0)
    dv = tl.load(DVv + nh * S + qi, mask=qok, other=0.0)
    dq = tl.zeros((BM, D), tl.float32)
    lo = tl.maximum(i0 - HW, 0) // BN * BN
    hi = tl.minimum(i0 + BM + HW, su)
    for j0 in range(lo, hi, BN):
        kj = j0 + tl.arange(0, BN); kok = kj < su
        k = tl.load(KH + hb + kj[:, None] * D + dc[None, :], mask=kok[:, None], other=0.0)
        v = tl.load(VH + hb + kj[:, None] * D + dc[None, :], mask=kok[:, None], other=0.0)
        dij = qi[:, None] - kj[None, :]
        allow = (dij <= HW) & (dij >= -HW) & kok[None, :] & qok[:, None]
        p = tl.where(allow, tl.exp(tl.dot(q, tl.trans(k)) * scale - lse[:, None]), 0.0)
        dp = tl.dot(do_, tl.trans(v))
        ds = (p * (dp - dv[:, None])).to(tl.bfloat16)
        dq += tl.dot(ds, k)
    tl.store(DQH + hb + qi[:, None] * D + dc[None, :], dq * scale, mask=(qi < S)[:, None])


@triton.autotune(configs=[triton.Config({"BM": bm, "BN": bn}, num_warps=w, num_stages=st) for bm, bn in ((64, 64), (32, 64), (64, 128), (32, 32))
                          for w in (4, 8) for st in (1, 2)], key=["SB", "HW"])
@triton.jit(do_not_specialize=["S"])
def _attn_bwd_dkv(QH, KH, VH, DOUT_, LSE, DVv, SEQU, DKH, DVH, S, scale, SB,
                  C: tl.constexpr, H: tl.constexpr, D: tl.constexpr, HW: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    """Program = (key tile of BN keys, n*H + h); loops the query tiles (BM) whose windows reach the tile."""
    t = tl.program_id(0); nh = tl.program_id(1).to(tl.int64)
    n = nh // H; h = nh - n * H
    su = tl.load(SEQU + n)
    j0 = t * BN
    kj = j0 + tl.arange(0, BN); kok = kj < su
    dc = tl.arange(0, D).to(tl.int64)
    hb = nh * S * D
    k = tl.load(KH + hb + kj[:, None] * D + dc[None, :], mask=kok[:, None], other=0.0)
    v = tl.load(VH + hb + kj[:, None] * D + dc[None, :], mask=kok[:, None], other=0.0)
    dk = tl.zeros((BN, D), tl.float32); dv = tl.zeros((BN, D), tl.float32)
    lo = tl.maximum(j0 - HW, 0) // BM * BM
    hi = tl.minimum(j0 + BN + HW, su)
    for i0 in range(lo, hi, BM):
        qi = i0 + tl.arange(0, BM); qok = qi < su
        q = tl.load(QH + hb + qi[:, None] * D + dc[None, :], mask=qok[:, None], other=0.0)
        do_ = tl.load(DOUT_ + (n * S + qi)[:, None] * C + h * D + dc[None, :], mask=qok[:, None], other=0.0)
        lse = tl.load(LSE + nh * S + qi, mask=qok, other=0.0)
        dvv = tl.load(DVv + nh * S + qi, mask=qok, other=0.0)
        dij = qi[None, :] - kj[:, None]                                                  # [BN keys][BM queries]
        allow = (dij <= HW) & (dij >= -HW) & kok[:, None] & qok[None, :]
        pT = tl.where(allow, tl.exp(tl.dot(k, tl.trans(q)) * scale - lse[None, :]), 0.0)
        dv += tl.dot(pT.to(tl.bfloat16), do_)
        dpT = tl.dot(v, tl.trans(do_))
        dsT = (pT * (dpT - dvv[None, :])).to(tl.bfloat16)
        dk += tl.dot(dsT, q)
    km = (kj < S)[:, None]
    tl.store(DKH + hb + kj[:, None] * D + dc[None, :], dk * scale, mask=km)
    tl.store(DVH + hb + kj[:, None] * D + dc[None, :], dv, mask=km)


@triton.jit
def _unrope(dy, cs, sn, BR: tl.constexpr, H: tl.constexpr, D: tl.constexpr):
    """Transpose of _rope: dx1 = dy1*c + dy2*s ; dx2 = dy2*c - dy1*s."""
    HALF: tl.constexpr = D // 2
    y4 = tl.reshape(dy, (BR, H, 2, HALF))
    y1, y2 = tl.split(tl.permute(y4, (0, 1, 3, 2)))
    c = cs[:, None, :]; s = sn[:, None, :]
    return tl.reshape(tl.permute(tl.join(y1 * c + y2 * s, y2 * c - y1 * s), (0, 1, 3, 2)), (BR, H * D))


@triton.jit
def _head_rms_bwd(p, dy, eps, BR: tl.constexpr, H: tl.constexpr, D: tl.constexpr):
    """y = p * r (r = rsqrt(mean_D p^2 + eps)) -> dp = r * (dy - yhat * mean_D(dy * yhat)), yhat = p * r."""
    p3 = tl.reshape(p, (BR, H, D)); d3 = tl.reshape(dy, (BR, H, D))
    r = 1.0 / tl.sqrt(tl.sum(p3 * p3, axis=2) / D + eps)
    yh = p3 * r[:, :, None]
    return tl.reshape(r[:, :, None] * (d3 - yh * (tl.sum(d3 * yh, axis=2) / D)[:, :, None]), (BR, H * D))


@triton.autotune(configs=[triton.Config({"SP": sp, "AT": at}, num_warps=w, num_stages=1)
                          for sp, at in ((2, 16), (4, 8), (2, 32), (4, 16)) for w in (4, 8)],
                 key=["SB"], restore_value=["DMOD"])
@triton.jit(do_not_specialize=["S", "A", "B"])
def _qkvg_bwd(QI, MOD, COS, SIN, WQKV, WG, DQH, DKH, DVH, DG, DQ1, DQ, DP, XN, DMOD, S, A, B, eps, qk_eps, SB,
              C: tl.constexpr, H: tl.constexpr, D: tl.constexpr, MODW: tl.constexpr, SP: tl.constexpr, AT: tl.constexpr):
    rows, ok, mrow, s_ = _tile_rows(S, A, B, SP, AT)
    b = tl.program_id(2).to(tl.int64); ab_idx = tl.program_id(0).to(tl.int64)
    ok_at = (ab_idx * AT + tl.arange(0, AT)) < S
    R: tl.constexpr = SP * AT
    cc = tl.arange(0, C).to(tl.int64)
    m2 = ok[:, None]
    ofs = rows[:, None] * C + cc[None, :]
    q = tl.load(QI + ofs, mask=m2, other=0.0).to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(q * q, axis=1) / C + eps)
    xh = q * rstd[:, None]
    shift = tl.load(MOD + mrow[:, None] * MODW + cc[None, :], mask=m2, other=0.0)
    scale = tl.load(MOD + mrow[:, None] * MODW + C + cc[None, :], mask=m2, other=0.0)
    x = (xh * (1.0 + scale) + shift).to(tl.bfloat16)
    tl.store(XN + ofs, x, mask=m2)
    HALF: tl.constexpr = D // 2
    hc = tl.arange(0, HALF).to(tl.int64)
    cs = tl.load(COS + mrow[:, None] * HALF + hc[None, :], mask=m2, other=1.0)
    sn = tl.load(SIN + mrow[:, None] * HALF + hc[None, :], mask=m2, other=0.0)
    n = rows // S
    hm = ((n[:, None] * H + (cc[None, :] // D)) * S + s_[:, None]) * D + (cc[None, :] % D)
    wofs = cc[None, :] * C + cc[:, None]
    # q: recompute the pre-norm projection, back through RoPE and the per-head RMSNorm
    wqT = tl.load(WQKV + wofs)
    pq = tl.dot(x, wqT).to(tl.bfloat16).to(tl.float32)
    dpq = _head_rms_bwd(pq, _unrope(tl.load(DQH + hm, mask=m2, other=0.0), cs, sn, R, H, D), qk_eps, R, H, D)
    dpq_b = dpq.to(tl.bfloat16)
    tl.store(DP + rows[:, None] * (4 * C) + cc[None, :], dpq_b, mask=m2)
    dx = tl.dot(dpq_b, tl.trans(wqT))
    wkT = tl.load(WQKV + C * C + wofs)
    pk = tl.dot(x, wkT).to(tl.bfloat16).to(tl.float32)
    dpk = _head_rms_bwd(pk, _unrope(tl.load(DKH + hm, mask=m2, other=0.0), cs, sn, R, H, D), qk_eps, R, H, D)
    dpk_b = dpk.to(tl.bfloat16)
    tl.store(DP + rows[:, None] * (4 * C) + C + cc[None, :], dpk_b, mask=m2)
    dx += tl.dot(dpk_b, tl.trans(wkT))
    dpv_b = tl.load(DVH + hm, mask=m2, other=0.0).to(tl.bfloat16)
    tl.store(DP + rows[:, None] * (4 * C) + 2 * C + cc[None, :], dpv_b, mask=m2)
    dx += tl.dot(dpv_b, tl.trans(tl.load(WQKV + 2 * C * C + wofs)))
    dpg_b = tl.load(DG + ofs, mask=m2, other=0.0)
    tl.store(DP + rows[:, None] * (4 * C) + 3 * C + cc[None, :], dpg_b, mask=m2)
    dx += tl.dot(dpg_b, tl.trans(tl.load(WG + wofs)))
    _presum_add(DMOD, tl.where(m2, dx * xh, 0.0), C, ab_idx, b, S, ok_at, C, MODW, SP, AT)          # d scale_a
    _presum_add(DMOD, tl.where(m2, dx, 0.0), 0, ab_idx, b, S, ok_at, C, MODW, SP, AT)               # d shift_a
    dxh = dx * (1.0 + scale)
    dq = tl.load(DQ1 + ofs, mask=m2, other=0.0) + rstd[:, None] * (dxh - xh * (tl.sum(dxh * xh, axis=1) / C)[:, None])
    tl.store(DQ + ofs, dq.to(tl.bfloat16), mask=m2)


def block_bwd(dy, q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, B, sv, half_window=64, eps=FP32_EPS):
    """Backward of block_fwd.  dy [N, S, C] bf16.  Returns dq [N,S,C] bf16, dmod [B*S, 6C] fp32, dWqkv, dWg, dWo, dWu, dWd (bf16)."""
    N, S, C = q.shape; H = 4; D = C // H; M = N * S; A = N // B; NHID = wd.shape[1]
    dev = q.device; sb = _bucket(M)
    dmod = torch.zeros(B * S, 6 * C, device=dev)
    dq1 = torch.empty(M, C, device=dev); dO = torch.empty(M, C, device=dev, dtype=torch.bfloat16); dG = torch.empty_like(dO)
    Dv = torch.empty(N, H, S, device=dev)
    dab = torch.empty(M, 2 * NHID, device=dev, dtype=torch.bfloat16); hh = torch.empty(M, NHID, device=dev, dtype=torch.bfloat16)
    dffn = torch.empty_like(dO); yn = torch.empty_like(dO); datt = torch.empty_like(dO); gated = torch.empty_like(dO)
    grid_t = lambda m: (triton.cdiv(S, m["AT"]), triton.cdiv(A, m["SP"]), B)
    LAST["post"] = _post_bwd[grid_t](dy.reshape(M, C), sv["q1"], sv["O"], sv["G"], mod, wo, wu, wd, dq1, dO, dG, Dv, dab, hh, dffn, yn,
                                     datt, gated, dmod, S, A, B, eps, sb, C=C, H=H, D=D, NHID=NHID, MODW=6 * C)
    dQh = torch.empty(N, H, S, D, device=dev); dKh = torch.empty_like(dQh); dVh = torch.empty_like(dQh)
    LAST["dq"] = _attn_bwd_dq[lambda m: (triton.cdiv(S, m["BM"]), N * H)](sv["Qh"], sv["Kh"], sv["Vh"], dO, sv["lse"], Dv, seqused, dQh, S, D ** -0.5, sb,
                                                                          C=C, H=H, D=D, HW=half_window)
    LAST["dkv"] = _attn_bwd_dkv[lambda m: (triton.cdiv(S, m["BN"]), N * H)](sv["Qh"], sv["Kh"], sv["Vh"], dO, sv["lse"], Dv, seqused, dKh, dVh, S,
                                                                            D ** -0.5, sb, C=C, H=H, D=D, HW=half_window)
    dq = torch.empty(M, C, device=dev, dtype=torch.bfloat16); dP = torch.empty(M, 4 * C, device=dev, dtype=torch.bfloat16); xn = torch.empty_like(dO)
    LAST["qkvg_bwd"] = _qkvg_bwd[grid_t](q.reshape(M, C), mod, cos, sin, wqkv, wg, dQh, dKh, dVh, dG, dq1, dq, dP, xn, dmod, S, A, B, eps, FP32_EPS, sb,
                                         C=C, H=H, D=D, MODW=6 * C)
    dWqkv = dP[:, :3 * C].t() @ xn; dWg = dP[:, 3 * C:].t() @ xn
    dWo = datt.t() @ gated; dWu = dab.t() @ yn; dWd = dffn.t() @ hh
    return dq.view(N, S, C), dmod, dWqkv, dWg, dWo, dWu, dWd


class SWABlockFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, B, half_window):
        train = any(ctx.needs_input_grad)
        q = q.contiguous()
        if not train:
            return block_fwd(q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, B, half_window)
        out, sv = block_fwd(q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, B, half_window, save=True)
        ctx.sv_keys = list(sv.keys())
        ctx.save_for_backward(q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, *sv.values())
        ctx.B, ctx.hw = B, half_window
        return out

    @staticmethod
    def backward(ctx, dy):
        t = ctx.saved_tensors
        q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd = t[:10]
        sv = dict(zip(ctx.sv_keys, t[10:]))
        dq, dmod, dWqkv, dWg, dWo, dWu, dWd = block_bwd(dy.contiguous(), q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, ctx.B, sv, ctx.hw)
        return dq, dmod, None, None, None, dWqkv, dWg, dWo, dWu, dWd, None, None


def swa_block(q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, B, half_window=64):
    """Differentiable fused ESMFold2 SWA atom block.  q [N, S, C] bf16, mod [B*S, 6C] fp32 (differentiable; hoisted modulation)."""
    return SWABlockFn.apply(q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, B, half_window)
