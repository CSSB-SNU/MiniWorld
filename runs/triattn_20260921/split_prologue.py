"""fpf_triatt_pro/prologue.py — triangle-attention PROLOGUE producer kernel ('triatt.qkvb layout v0', FPF_SPEC_v0)

    triatt_prologue(module, z, ending=False, ln_mode='fused'|'welford'|'stock', x_ln=None) -> (q, k, v, g, bias_f32)

For a stock TriangleAttention `module` (c_in=C, H heads, D per head) and the pair tensor z [N(I), N(J), C] bf16:
    x    = z  (starting node)  |  z^T (token axes swapped in the address math; ending node in BLOCK mode -- z^T never materialized)
    x_ln = bf16(LN_fp32(x) * bf16(gamma) + bf16(beta))       (registers only; NOT written to HBM unless write_x)
    q,k,v= bf16(fp32-acc x_ln @ Wq/k/v^T) -> [I, H, J, D] contiguous  (== stock linear_q(x).view(..,H,D).transpose(-2,-3).contiguous(), what cuEq consumes)
    g    = bf16(fp32-acc x_ln @ Wg^T)     -> [I, J, H*D] contiguous, PRE-sigmoid (== stock linear_g(x))
    bias = fp32(bf16(fp32-acc x_ln @ Wb^T))-> [H, I, J] fp32 contiguous (== permute_final_dims(linear(x),(2,0,1)).float(): bf16 GEMM output, then .float())
LN statistics:
    'fused'   = two-pass fp32 (mean, then centered sum of squares), IEEE sqrt/div        -> Tier-2 a priori vs fast_layernorm (different reduction tree)
    'welford' = emulation of the stock fast_layernorm LayerNormForwardV2<bf16,float4> for C == 256: per-lane sequential fp32 Welford over 8 contiguous
                elements + xor-butterfly merge (16,8,4,2,1) with per-lane results, fast-math division / rsqrt.approx  -> bit-exact candidate (MEASURED)
    'stock'   = x_ln given (materialized by the stock module.layer_norm call) -> kernel does only the 5 projections + layouts. EXACT candidate:
                tl.dot over the full K=C in ONE MMA chain, fp32 acc, one rounding to bf16 (== cuBLAS bf16 GEMM bits on H100 for K<=256, measured).
Fixed tile configs (PINNED_CONFIG; no autotune at run time, no atomics, no split-K); config = pure function of (C, H*D, GPU class);
env FPF_TRIATTPRO_CFG="BI,BJ,W,S" exists for OFFLINE sweeps only.  Shape-generic: C power of 2 in {64,128,256}, H*D multiple of 16, D power of 2.
"""
from __future__ import annotations
import os, json, hashlib
import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------------------------------------------------------------- pinned config table
# key: (C, H*D, gpu_class) -> dict(BI, BJ, num_warps, num_stages).  A CTA handles a BI x BJ tile of (i, j) pairs = BM = BI*BJ rows of the flattened
# [I*J, C] problem.  BI >= 8 keeps the ending-node (transposed) z read in >= 8-row (8 x 512 B) contiguous runs; q/k/v stores are BJ x (D*2 B) runs.
PINNED_CONFIG = {
    (256, 256, "H100"): dict(BI=8, BJ=16, num_warps=8, num_stages=2),     # the stock pairformer / MSA pair stack (c_z=256, H=8, D=32)     checked bit-exact (the set below)
    (64, 64, "H100"): dict(BI=8, BJ=16, num_warps=4, num_stages=2),       # the stock template-embedder pair stack (c=64, H=2, D=32)       no VERIFIED record -> not in the set below -> measured off (stock forward)
    (128, 128, "H100"): dict(BI=8, BJ=16, num_warps=8, num_stages=2),     # the c_z=128 class (H=4, D=32)                                    implemented, UNTESTED -> not in the set below -> falls back
    (256, 256, "A100"): dict(BI=8, BJ=8, num_warps=4, num_stages=2),      # sm_80 port, UNTESTED -> falls back
}
# A key enters VERIFIED only after a bit-exact (ln_mode='stock') / Tier-2 (ln_mode='fused') test record on real dumps exists for it; the registry op runs the Triton path
# for VERIFIED keys.  A PINNED key without a VERIFIED record (the template stack (64,64,'H100'), the c_z=128 class (128,128,'H100'), the sm_80 port (256,256,'A100'),
# a key merged into PINNED_CONFIG at run time) is MEASURED OFF for the registry op: the STOCK forward BY NAME ('cell:c<C>_hd<HD>_<gpu>+off(not-measured)').
# Any OTHER key is UNKNOWN and is served the SAFE prologue settings of its capability (opt_core.kernels.safe_settings 'pair_fused:prologue': 8x8 tiles), engaged and
# named ONCE ('[opt_core/fpf_triatt_pro] safe settings served (no_cell:<cell>, cc .., triton ..)'); a key no safe row admits stays on the stock forward.
VERIFIED = {(256, 256, "H100")}
if os.environ.get("FPF_TRIATTPRO_VERIFIED_EXTRA"):      # test jobs only: e.g. "64,64,H100" to exercise a candidate key before it is promoted
    for item in os.environ["FPF_TRIATTPRO_VERIFIED_EXTRA"].split(";"):
        c_, hd_, g_ = item.split(","); VERIFIED.add((int(c_), int(hd_), g_))
SAFE_LEVER = "pair_fused:prologue"                       # the safe-settings rows whose settings this kernel's cfg reads (BI, BJ, num_warps, num_stages)
from opt_core.kernels import safe_settings as _safe, cell_words as _cw       # noqa: E402 — the core's one cell vocabulary + safe-settings rows (stdlib at import)
_NET = _safe.SafeNet("fpf_triatt_pro")                   # names the SAFE engagement once per process (settings_word())


def _cc(device) -> str:
    return "%d.%d" % tuple(torch.cuda.get_device_capability(torch.device(device).index or 0))


def cell_word(C: int, HD: int, gpu: str) -> str:
    return "cell:c%d_hd%d_%s" % (C, HD, gpu)


def cell_verdict(C: int, HD: int, device=None, *, admit: str = "pinned", gpu: str = None, cc: str = None):
    """The ONE cell decision -> (kind, cfg, word) (opt_core.kernels.cell_words.decide): 'pinned' (PINNED_CONFIG; admit='verified': only keys with a VERIFIED record —
    a pinned key without one is 'off'), 'off' ('cell:...+off(not-measured)'), 'safe' (an UNKNOWN key: the capability's SAFE settings), 'none' (nothing admits it).
    ``gpu`` / ``cc`` name the card for a decision off the device (tests)."""
    gpu = gpu if gpu is not None else gpu_class(device)
    cc = cc if cc is not None else _cc(device)
    key = (int(C), int(HD), gpu)
    pinned = PINNED_CONFIG.get(key) if (admit == "pinned" or key in VERIFIED) else None
    return _cw.decide(pinned=pinned, measured_off=key in PINNED_CONFIG, miss_word=cell_word(*key), safe_lever=SAFE_LEVER, cc=cc, dims={"c_z": int(C)},
                      shape_word="c%d_hd%d_%s" % key)


def eligible(C: int, HD: int, device, **kw) -> bool:
    """True when the registry op runs the Triton path: a VERIFIED key, or an UNKNOWN key the SAFE settings serve."""
    return cell_verdict(C, HD, device, admit="verified", **kw)[0] in ("pinned", "safe")


def stock_word(C: int, HD: int, device, **kw) -> str:
    """The census word of a key the registry op does not run ('' when it runs)."""
    kind, _cfg, word = cell_verdict(C, HD, device, admit="verified", **kw)
    return "" if kind in ("pinned", "safe") else word


def settings_word():
    """``safe:no_cell:<cell>`` once the SAFE settings serve an UNKNOWN key in this process, else None."""
    return _NET.word()


def config_table_sha256() -> str:
    return hashlib.sha256(json.dumps({str(k): v for k, v in PINNED_CONFIG.items()}, sort_keys=True).encode()).hexdigest()


_GPU_CLASS = {}


def gpu_class(device) -> str:
    idx = torch.device(device).index or 0
    if idx not in _GPU_CLASS:
        name = torch.cuda.get_device_name(idx)
        mj, mn = torch.cuda.get_device_capability(idx)
        if any(s in name for s in ("H100", "H200", "H800")):
            _GPU_CLASS[idx] = "H100"                                   # shipped checked keys are ('...', 'H100')
        elif "A100" in name:
            _GPU_CLASS[idx] = "A100"
        else:
            _GPU_CLASS[idx] = f"sm{mj}{mn}"                            # exact arch key (sm100 = B200, sm120 = RTX PRO 6000 Blackwell); no key of these classes is pinned here: their cells arrive as a caller-supplied cfg= or are served the SAFE settings
    return _GPU_CLASS[idx]


def pick_config(C: int, HD: int, device, **kw) -> dict:
    env = os.environ.get("FPF_TRIATTPRO_CFG", "").strip()          # offline sweeps only; never set in the path
    if env:
        BI, BJ, W, S = (int(v) for v in env.split(","))
        return dict(BI=BI, BJ=BJ, num_warps=W, num_stages=S)
    kind, cfg, word = cell_verdict(C, HD, device, **kw)
    if kind == "safe":                                              # an UNKNOWN key: the capability's SAFE settings, engaged and named ONCE
        _NET.engage(word, _safe.where_word(kw.get("cc") or _cc(device), _safe.triton_mm()))
    if cfg is None:
        raise KeyError(f"fpf_triatt_pro: no launch config for (C, H*D, gpu_class) = {(C, HD, kw.get('gpu') or gpu_class(device))} ({word}); registry op falls back to stock for such shapes")
    return cfg


# ----------------------------------------------------------------------------------------------------------------------------- device helpers (PTX-exact fp32 ops)
@triton.jit
def _xor_partner(x, m: tl.constexpr, BM: tl.constexpr, LANES: tl.constexpr):
    # x: [BM, LANES] -> y[:, l] = x[:, l ^ m]   (m = power of two < LANES); emulates __shfl_xor_sync(0xffffffff, x, m)
    if m == 1:
        x3 = tl.reshape(x, (BM, LANES // 2, 2))
        y3 = tl.flip(x3, 2)
        return tl.reshape(y3, (BM, LANES))
    else:
        x4 = tl.reshape(x, (BM, LANES // (2 * m), 2, m))
        x4p = tl.permute(x4, (0, 1, 3, 2))            # [BM, G, m, 2]  pair axis last
        y4p = tl.flip(x4p, 3)
        y4 = tl.permute(y4p, (0, 1, 3, 2))
        return tl.reshape(y4, (BM, LANES))


_ADDR_SPLIT = os.environ.get("FPF_TRIATTPRO_ADDR", "chain") == "split"   # "chain" restores the [BM, BN] int64 address chain
_BN_CAP = int(os.environ.get("FPF_TRIATTPRO_BN", "128"))            # offline override of the cell row's BN (projection output-chunk width; the fp32 acc is [BM, BN])
# q/k/v physical layout.  MEASURED, and the pair-natural order LOSES: prologue 358 -> 408 us (L768) / 620 -> 723 (L1024),
# core +1.0..1.2 %, and at L384 the Triton core route refuses the strided view and a slower CUDA route takes over.
# The reason the head layout ([I, H, J, D]) is already good: a CTA tile spans BJ = 16 CONSECUTIVE j, and in that layout
# those 16 j x 32 d sit adjacent, so each (i, h) stores one 1024 B run -- not the 64 B run a per-element reading suggests.
_QKV_PAIR = os.environ.get("FPF_TRIATTPRO_QKV", "head") == "pair"   # offline only; the cell row's qkv_pair wins when set
_MAXNREG = int(os.environ.get("FPF_TRIATTPRO_MAXNREG", "0"))        # offline override of the cell row's maxnreg (ptxas .maxnreg; 128 at 256 threads = 2 CTA/SM)
_ASM = tl.constexpr(os.environ.get("TRITON_INTERPRET", "0") != "1" and os.environ.get("FPF_TRIATTPRO_NOASM", "0") != "1")
# ^ inline PTX on GPU; plain ops under the CPU interpreter (debug only).  FPF_TRIATTPRO_NOASM=1 is an experiment: tl.inline_asm_elementwise
# requires every operand to have the SAME SHAPE, so the per-column gamma/beta must be broadcast_to([BM, C]) before the affine -- two full
# fp32 tiles (64 registers per thread each at BM=C=128) that exist only to satisfy the asm signature.  Plain tl.* ops fold the broadcast,
# which is worth testing against 190 registers -> 1 CTA/SM -> 12.5 % occupancy.  Same op order; the asm form additionally pins .ftz.


@triton.jit
def _fdiv_fast(a, b):
    # nvcc --use_fast_math => -prec-div=false -ftz=true : fp32 '/' -> div.full.ftz.f32 (checked against the compiled extension's PTX)
    if _ASM:
        return tl.inline_asm_elementwise("div.full.ftz.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32, is_pure=True, pack=1)
    else:
        return a / b


@triton.jit
def _rsqrt_fast(a):
    if _ASM:
        return tl.inline_asm_elementwise("rsqrt.approx.ftz.f32 $0, $1;", "=f,f", [a], dtype=tl.float32, is_pure=True, pack=1)
    else:
        return 1.0 / tl.sqrt_rn(a)


@triton.jit
def _rcp_fast(a):
    # MUFU.RCP: what nvcc --use_fast_math emits for fp32 '/' (x / y -> x * rcp.approx(y), usually contracted into an FFMA)
    if _ASM:
        return tl.inline_asm_elementwise("rcp.approx.ftz.f32 $0, $1;", "=f,f", [a], dtype=tl.float32, is_pure=True, pack=1)
    else:
        return 1.0 / a


@triton.jit
def _mul_rn(a, b):
    if _ASM:
        return tl.inline_asm_elementwise("mul.rn.ftz.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32, is_pure=True, pack=1)
    else:
        return a * b


@triton.jit
def _add_rn(a, b):
    if _ASM:
        return tl.inline_asm_elementwise("add.rn.ftz.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32, is_pure=True, pack=1)
    else:
        return a + b


@triton.jit
def _sub_rn(a, b):
    if _ASM:
        return tl.inline_asm_elementwise("sub.rn.ftz.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32, is_pure=True, pack=1)
    else:
        return a - b


@triton.jit
def _fma_rn(a, b, c):
    if _ASM:
        return tl.inline_asm_elementwise("fma.rn.ftz.f32 $0, $1, $2, $3;", "=f,f,f,f", [a, b, c], dtype=tl.float32, is_pure=True, pack=1)
    else:
        return tl.fma(a, b, c)


@triton.jit
def _welford_step(xv, t_mean, t_m2, cnt, FMA_M2: tl.constexpr):
    # LayerNormForwardV2<bf16,float4> (protenix/model/layer_norm/kernel/layer_norm_cuda_kernel.cu, built -O3 --use_fast_math for sm_90), op order per element:  r = MUFU.RCP(count) ; d1 = x - mean ; mean = FFMA(d1, r, mean) ;
    # d2 = x - mean ; m2 = FFMA(d1, d2, m2).   (FMA_M2 flag kept for A/B: False -> separate mul+add)
    r = _rcp_fast(tl.zeros_like(xv) + cnt)
    d1 = _sub_rn(xv, t_mean)
    t_mean = _fma_rn(d1, r, t_mean)
    d2 = _sub_rn(xv, t_mean)
    if FMA_M2:
        t_m2 = _fma_rn(d1, d2, t_m2)
    else:
        t_m2 = _add_rn(t_m2, _mul_rn(d1, d2))
    return t_mean, t_m2


@triton.jit
def _welford_lane_chunk8(xc, BM: tl.constexpr, LANES: tl.constexpr, FMA_M2: tl.constexpr):
    """sequential WelfordOnline over the 8 elements (index order) of xc [BM, LANES, 8], starting from zero state. Returns (mean, m2) [BM, LANES]."""
    a = tl.reshape(xc, (BM, LANES, 4, 2))
    ev, od = tl.split(a)                                   # ev = elements 0,2,4,6 ; od = 1,3,5,7            [BM, LANES, 4]
    e04, e26 = tl.split(tl.reshape(ev, (BM, LANES, 2, 2)))  # e04 = (0,4), e26 = (2,6)                        [BM, LANES, 2]
    e15, e37 = tl.split(tl.reshape(od, (BM, LANES, 2, 2)))
    x0, x4 = tl.split(e04); x2, x6 = tl.split(e26); x1, x5 = tl.split(e15); x3, x7 = tl.split(e37)      # [BM, LANES] each
    t_mean = tl.zeros((BM, LANES), dtype=tl.float32); t_m2 = tl.zeros((BM, LANES), dtype=tl.float32)
    t_mean, t_m2 = _welford_step(x0, t_mean, t_m2, 1.0, FMA_M2)
    t_mean, t_m2 = _welford_step(x1, t_mean, t_m2, 2.0, FMA_M2)
    t_mean, t_m2 = _welford_step(x2, t_mean, t_m2, 3.0, FMA_M2)
    t_mean, t_m2 = _welford_step(x3, t_mean, t_m2, 4.0, FMA_M2)
    t_mean, t_m2 = _welford_step(x4, t_mean, t_m2, 5.0, FMA_M2)
    t_mean, t_m2 = _welford_step(x5, t_mean, t_m2, 6.0, FMA_M2)
    t_mean, t_m2 = _welford_step(x6, t_mean, t_m2, 7.0, FMA_M2)
    t_mean, t_m2 = _welford_step(x7, t_mean, t_m2, 8.0, FMA_M2)
    return t_mean, t_m2


@triton.jit
def _welford_merge_level(mean, m2, ca, mm: tl.constexpr, BM: tl.constexpr, FMA_MERGE: tl.constexpr):
    LANES: tl.constexpr = 32
    b_mean = _xor_partner(mean, mm, BM, LANES)
    b_m2 = _xor_partner(m2, mm, BM, LANES)
    zf = tl.zeros_like(mean)
    n = ca + ca
    r = _rcp_fast(zf + n)
    delta = _sub_rn(b_mean, mean)
    t = _mul_rn(delta, delta)
    t = _mul_rn(zf + ca, t)
    nb_n = _mul_rn(zf + ca, r)
    if FMA_MERGE:
        s = _fma_rn(nb_n, t, b_m2)
        mean = _fma_rn(nb_n, delta, mean)
    else:
        s = _add_rn(b_m2, _mul_rn(nb_n, t))
        mean = _add_rn(mean, _mul_rn(nb_n, delta))
    m2 = _add_rn(m2, s)
    return mean, m2


@triton.jit
def _welford_butterfly32(mean, m2, BM: tl.constexpr, FMA_MERGE: tl.constexpr):
    """xor-butterfly all-reduce over 32 lanes (masks 16,8,4,2,1); every lane starts with count 8 (C=256). Per-lane results kept (like the CUDA kernel).
    NOTE: bit-exact emulation of fast_layernorm is not achieved by this scheme; the path is kept for reference only and is never engaged."""
    mean, m2 = _welford_merge_level(mean, m2, 8.0, 16, BM, FMA_MERGE)
    mean, m2 = _welford_merge_level(mean, m2, 16.0, 8, BM, FMA_MERGE)
    mean, m2 = _welford_merge_level(mean, m2, 32.0, 4, BM, FMA_MERGE)
    mean, m2 = _welford_merge_level(mean, m2, 64.0, 2, BM, FMA_MERGE)
    mean, m2 = _welford_merge_level(mean, m2, 128.0, 1, BM, FMA_MERGE)
    return mean, m2


# ----------------------------------------------------------------------------------------------------------------------------- the prologue kernel
@triton.jit
def _triatt_prologue_kernel(
    Z, XLN, LNW, LNB, WQKVG, WB,
    Q, K, V, G, BIAS, XOUT,
    NI, NJ,
    s_zi, s_zj,                       # element strides of z in the x-frame: x[i, j, :] = Z + i*s_zi + j*s_zj   (ending node: swapped strides)
    eps,
    C: tl.constexpr, H: tl.constexpr, D: tl.constexpr, HB: tl.constexpr,
    BI: tl.constexpr, BJ: tl.constexpr, BN: tl.constexpr,
    LN_MODE: tl.constexpr,            # 0 = x_ln given (XLN [NI*NJ, C] contiguous, x-frame) ; 1 = two-pass fp32 LN ; 2 = fast_layernorm Welford emulation (C==256)
    WRITE_X: tl.constexpr, FMA_M2: tl.constexpr, FMA_MERGE: tl.constexpr, FMA_AFF: tl.constexpr,
    NOASM_AFF: tl.constexpr,
    ADDR_SPLIT: tl.constexpr,         # q/k/v and bias store offsets as a [BM] row base + a [BN] column delta (one broadcast add) instead of a [BM, BN] int64 chain
    QKV_PAIR: tl.constexpr,           # q/k/v written in pair-natural [I, J, H, D] (contiguous, handed to the core as the [I, H, J, D] view) instead of [I, H, J, D] storage
    DOT_F32: tl.constexpr,            # debug only (TRITON_INTERPRET=1 on CPU mis-executes bf16 tl.dot): upcast dot operands to fp32. NEVER set on GPU runs.
    J_FAST: tl.constexpr = False,    # accepted for compatibility; this experiment fixes its own launch order
):
    HD: tl.constexpr = H * D
    BM: tl.constexpr = BI * BJ
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    r = tl.arange(0, BM)
    ii = pid_i * BI + r // BJ                      # [BM] x-frame row index i
    jj = pid_j * BJ + r % BJ                       # [BM] x-frame col index j
    rmask = (ii < NI) & (jj < NJ)
    rc = tl.arange(0, C)
    ii64 = ii.to(tl.int64)
    jj64 = jj.to(tl.int64)
    xrow = (ii64 * NJ + jj64) * C                  # row offset of (i, j) in a contiguous x-frame [NI, NJ, C] tensor

    if LN_MODE == 0:
        x = tl.load(XLN + xrow[:, None] + rc[None, :], mask=rmask[:, None], other=0.0)          # bf16 [BM, C]
    else:
        zoff = ii64 * s_zi + jj64 * s_zj
        zt = tl.load(Z + zoff[:, None] + rc[None, :], mask=rmask[:, None], other=0.0)         # bf16 [BM, C]
        xf = zt.to(tl.float32)
        wln = tl.load(LNW + rc).to(tl.float32)                                                  # bf16 params (weight.to(bf16) done on host) -> fp32
        bln = tl.load(LNB + rc).to(tl.float32)
        if LN_MODE == 1:
            mean = tl.sum(xf, axis=1) / C
            xc = xf - mean[:, None]
            var = tl.sum(xc * xc, axis=1) / C
            inv = 1.0 / tl.sqrt_rn(var + eps)
            y = xc * inv[:, None] * wln[None, :] + bln[None, :]
        else:
            LANES: tl.constexpr = 32
            xl = tl.reshape(xf, (BM, LANES, 8))                                                 # lane t <-> elements [8t, 8t+8)
            t_mean, t_m2 = _welford_lane_chunk8(xl, BM, LANES, FMA_M2)
            mean_l, m2_l = _welford_butterfly32(t_mean, t_m2, BM, FMA_MERGE)                    # per-lane [BM, 32]
            var_l = _mul_rn(_rcp_fast(tl.zeros_like(m2_l) + C), m2_l)                          # op order: r = RCP(count); var = r * m2
            var_l = tl.maximum(var_l, 0.0)
            inv_l = _rsqrt_fast(_add_rn(var_l, tl.zeros_like(var_l) + eps))                    # + eps ; MUFU.RSQ
            mean_e = tl.reshape(tl.broadcast_to(mean_l[:, :, None], (BM, LANES, 8)), (BM, C))
            inv_e = tl.reshape(tl.broadcast_to(inv_l[:, :, None], (BM, LANES, 8)), (BM, C))
            nrm = _mul_rn(inv_e, _sub_rn(xf, mean_e))                                          # op order: FADD (x - mean) ; FMUL inv * .
            if NOASM_AFF:                                   # plain ops fold a broadcast; the asm form cannot, so it needs the full tiles
                wb2 = wln[None, :]
                bb2 = bln[None, :]
            else:
                wb2 = tl.broadcast_to(wln[None, :], (BM, C))
                bb2 = tl.broadcast_to(bln[None, :], (BM, C))
            if FMA_AFF:
                y = _fma_rn(nrm, wb2, bb2)                                                      # op order: FFMA(nrm, gamma, beta)
            else:
                y = _add_rn(_mul_rn(nrm, wb2), bb2)
        x = tl.where(rmask[:, None], y, 0.0).to(tl.bfloat16)
        if WRITE_X:
            tl.store(XOUT + xrow[:, None] + rc[None, :], x, mask=rmask[:, None])

    # ---- projections q | k | v | g : WQKVG = cat([Wq, Wk, Wv, Wg]) [4*HD, C] bf16 row-major; BN-wide output-column chunks; full K=C per tl.dot (one MMA chain)
    rn = tl.arange(0, BN)
    dd64 = (rn % D).to(tl.int64)
    for _chunk in tl.static_range(1):
        n0 = tl.program_id(2) * BN
        wT = tl.load(WQKVG + (n0 + rn)[None, :].to(tl.int64) * C + rc[:, None])                # [C, BN] = W[n0:n0+BN, :]^T
        if DOT_F32:
            acc = tl.dot(x.to(tl.float32), wT.to(tl.float32))
        else:
            acc = tl.dot(x, wT)                                                                 # bf16 x bf16, fp32 acc over the FULL K=C in one chain
        o16 = acc.to(tl.bfloat16)                                                               # [BM, BN]
        if n0 // HD == 3:                                                                       # g  [I, J, HD]
            goff = (ii64 * NJ + jj64) * HD + (n0 % HD)
            tl.store(G + goff[:, None] + rn[None, :], o16, mask=rmask[:, None])
        else:                                                                                   # q/k/v [I, H, J, D]
            if QKV_PAIR:                                                                    # [I, J, H, D]: one H*D-wide run per (i, j) -- the g store's own expression
                off = ((ii64 * NJ + jj64) * HD + (n0 % HD))[:, None] + rn[None, :]
            else:
                hh64 = ((n0 % HD + rn) // D).to(tl.int64)
                if ADDR_SPLIT:                                                                  # ((i H + h) NJ + j) D + d  =  [i H NJ D + j D] (per row) + [h NJ D + d] (per column)
                    off = (ii64 * (H * NJ * D) + jj64 * D)[:, None] + (hh64 * (NJ * D) + dd64)[None, :]
                else:
                    off = ((ii64[:, None] * H + hh64[None, :]) * NJ + jj64[:, None]) * D + dd64[None, :]
            if n0 // HD == 0:
                tl.store(Q + off, o16, mask=rmask[:, None])
            elif n0 // HD == 1:
                tl.store(K + off, o16, mask=rmask[:, None])
            else:
                tl.store(V + off, o16, mask=rmask[:, None])
    if tl.program_id(2) == 0:
        # ---- triangle bias: WB [HB, C] (rows >= H zero-padded), output fp32 [H, I, J] = float(bf16(acc))
        rb = tl.arange(0, HB)
        wbT = tl.load(WB + rb[None, :].to(tl.int64) * C + rc[:, None])                             # [C, HB]
        if DOT_F32:
            accb = tl.dot(x.to(tl.float32), wbT.to(tl.float32))
        else:
            accb = tl.dot(x, wbT)
        b32 = accb.to(tl.bfloat16).to(tl.float32)                                                   # [BM, HB] stock rounding point: bf16 GEMM out, then .float()
        if ADDR_SPLIT:                                                                             # (h NI + i) NJ + j  =  [i NJ + j] (per row) + [h NI NJ] (per column)
            boff = (ii64 * NJ + jj64)[:, None] + (rb.to(tl.int64) * (NI * NJ))[None, :]
        else:
            boff = (rb[None, :].to(tl.int64) * NI + ii64[:, None]) * NJ + jj64[:, None]
        tl.store(BIAS + boff, b32, mask=rmask[:, None] & (rb[None, :] < H))


