"""fpf_triatt_epi/epilogue.py — FPF 'TriAttEpi': triangle-attention EPILOGUE kernel for the engine's pair stack (FPF_SPEC_v0 registry op `triatt`).

Engine stock class reproduced:
  "the engine's TriangleAttention output path (layers.Attention._wrap_up: sigmoid gate, linear_o OpenfoldLinear bf16; + PairformerBlock residual add / transpose)"

Stock (the engine's pinned release, eval, bf16 autocast, triangle_attention='cuequivariance'):
    o  = cuequivariance_triangular_attn(q,k,v,bias_f32,mask_bool,scale)[0]      # [(B,) I, H, J, D] bf16 contiguous (cuEq layout)
    o  = o.transpose(-2,-3)                                                      # view [.., I, J, H, D]
    g  = sigmoid(linear_g(x))               # linear_g: OpenfoldLinear no-bias -> bf16 GEMM out; sigmoid: ATen opmath fp32 1/(1+exp(-g)) -> bf16
    o  = o * g.view(.., H, D)               # fp32 multiply of two bf16 -> bf16
    o  = flatten_final_dims(o, 2)           # [.., I, J, H*D]  (reshape -> copy)
    u  = linear_o(o)                        # OpenfoldLinear(H*D -> c, bias=False): F.linear(bf16, W.to(bf16)) -> cuBLAS bf16 GEMM, fp32 acc -> bf16
    (caller PairformerBlock, inplace path)  z += u     # ATen add: fp32(z)+fp32(u) -> bf16
    ending node: the whole TriangleAttention runs on x = z^T (view) and returns u^T (view); caller: z(=z^T contiguous) += u ; z = z.transpose(-2,-3).contiguous()

This module provides ONE Triton kernel `_triatt_epilogue_kernel` doing, per CTA tile of BI i-rows x BJ j-cols (BM = BI*BJ rows of (i,j)) and all c output channels:
    for h in 0..H-1 (ascending):   o_h = O[i, h, j, :]  ([BM, D] gathered from H contiguous [BJ, D] slabs)      g_h = G[i, j, h*D:(h+1)*D]
                                   gated_h = bf16( fp32(o_h) * fp32( bf16( 1/(1+exp(-fp32(g_h))) ) ) )        (stock rounding points)
                                   acc    += gated_h @ WoT[h*D:(h+1)*D, :]     (bf16 MMA, fp32 accumulate; K visited ascending 0..c-1 in ONE accumulator chain
                                                                                 == a single tl.dot over K=c == cuBLAS bf16 GEMM k-order on H100 for K<=256 (measured; confirmed at op level))
    u = bf16(acc)
    op mode    (RESIDUAL=False): OUT[i, j, :] = u                                (registry `triatt` contract: return the update, caller adds)
    block mode (RESIDUAL=True) : Z[zi, zj, :] = bf16( fp32(Z[zi,zj,:]) + fp32(u) )   in place, with (zi,zj) = (i,j) for the starting node and (j,i) for the ENDING node
                                 (ENDING=True: the epilogue of the transposed-frame attention is scattered straight into the UNtransposed z -> the two
                                  `z.transpose(-2,-3).contiguous()` passes and the separate residual-add pass of PairformerBlock disappear).
No atomics, no autotune at run time, no split-K: fixed tile config from PINNED_CONFIG (pure function of (c, H, D)); grid = (cdiv(I,BI), cdiv(J,BJ), B).

z-sized HBM passes (1 pass = one full read or write of an [N,N,c] bf16 tensor):
    stock epilogue: sigmoid (r+w=2) + o*g (2r+w=3) + flatten copy (0: ATen writes the mul output contiguous already? NO: o*g output takes o's strided layout
                    -> reshape copies: r+w=2) + linear_o (r+w=2) + residual z+=u (2r+w=3) [+ ending transpose.contiguous r+w=2] = 12 (start) / 14 (ending)  (+ g GEMM r+w, unchanged, lives in the prologue)
    this kernel   : read o (1) + read g (1) + [op mode: write u (1)] or [block mode: read z (1) + write z (1)] = 3 (op) / 4 (block), start and ending alike.

Public API
----------
    triatt_epilogue(o, g, wo16, z=None, *, ending=False, residual=False, out=None, cfg=None) -> update | None
    fn(module, x, mask=None, chunk_size=None, triangle_attention='cuequivariance', inplace_safe=False)      # registry `triatt` (TriangleAttention.forward contract)
    fn_block_residual(module, z, ending)                                                                    # composition helper: z += triatt(z) [start] or the
                                                                                                            # ending-node equivalent incl. both transposes, IN PLACE on z
    torch_reference_epilogue(o, g, wo16, z=None, ending=False, residual=False)                              # exact torch emulation of the stock math (testing)
    PINNED_CONFIG, config_sha256()
    Weight cache: module._fpf_cache['triatt_epi'] = {'dev', 'wo16' (= mha.linear_o.weight.to(bf16)), 'woT16' (its [K,c] transpose)} — namespaced; the root dict is shared with sibling packages and never replaced.

Block-level composition — reproducing the stock inplace sequence
        z += tri_att_start(z); z = z.transpose(-2,-3).contiguous(); z += tri_att_end(z); z = z.transpose(-2,-3).contiguous()
    with z NEVER transposed or copied (the engine builds BOTH modules with starting=True; the ending role comes only from the caller's transposes):
        o, g = prologue_attention(block.tri_att_start, x=z)                      # stock LN + bias linear + q/k/v(+g) GEMMs + cuEq (or sibling prologue/K2 kernels);
        triatt_epilogue(o, g, Wo_start_bf16, z, ending=False, residual=True)      #   o [I,H,J,D] cuEq layout, g [I,J,H*D] pre-sigmoid  ->  z[i,j,:] += u[i,j,:]
        o, g = prologue_attention(block.tri_att_end, x=z.transpose(-2,-3))      # prologue in the TRANSPOSED frame: x[i,j] = z[j,i] (stock LN .contiguous() makes the
        triatt_epilogue(o, g, Wo_end_bf16, z, ending=True, residual=True)         #   same bits as stock's zT)  ->  z[j,i,:] += u[i,j,:]  (transposed scatter, coalesced per BI rows)
    `fn_block_residual(module, z, ending)` implements exactly this with the STOCK prologue + cuEq and is tested bit-exact against the stock statements (selftest 'block').
"""
from __future__ import annotations
import os, math, json, hashlib
import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    try:
        from triton.language.extra import libdevice as _ld
    except Exception:  # pragma: no cover
        try:
            from triton.language.extra.cuda import libdevice as _ld
        except Exception:
            _ld = None
    _HAS_TRITON = True
except Exception:  # pragma: no cover
    triton = None; tl = None; _ld = None; _HAS_TRITON = False

from opt_core.kernels import safe_settings as _safe, cell_words as _cw      # the core's one cell vocabulary + safe-settings rows (stdlib at import)
_STATS = {}          # call counters: 'kernel' / 'fallback' (a caller can print fpf_triatt_epi.epilogue._STATS)
ENGINE_STOCK_CLASS = ("the engine's TriangleAttention output path (layers.Attention._wrap_up: sigmoid gate, linear_o OpenfoldLinear bf16; "
                      "+ PairformerBlock residual add / transpose)")

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# PINNED tile-config table: key = (c, H, D, gpu_class) -> dict(BI, BJ, num_warps, num_stages, EXP)
#   BM = BI*BJ rows per CTA.  BI, BJ multiples of 8 so that both the o slab reads ([BJ, D] contiguous = BJ*64 B per (i,h)) and the transposed z writes
#   (z[j, i0:i0+BI, :] = BI*512 B contiguous) are full-sector.  Same table entry for every N (no N-bucketing -> ORDER/FIXEDCFG trivially satisfied).
#   EXP: 'libdevice' -> libdevice expf (== CUDA expf used by ATen sigmoid) ; 'tl' -> tl.exp.  Only 'libdevice' is the config.
# Env override FPF_TRIATT_EPI_CFG="BI,BJ,W,S" is for offline sweeps ONLY (the launcher rows are produced with the variable unset; recorded in results).
_NREG = ({"maxnreg": int(os.environ["FPF_TRIATT_EPI_MAXNREG"])}
         if os.environ.get("FPF_TRIATT_EPI_MAXNREG", "0") not in ("", "0") else {})   # offline override of the cell row's maxnreg (unconstrained = 255 regs x 128 thr = 2 CTA/SM)

PINNED_CONFIG = {
    # ONLY cells checked bit-exact on real dumps. key (c, H, D, gpu_class) -> config.
    # KVER: 2 = _triatt_epilogue_kernel_v2 (one gathered [BM,K] tile, one dot); 1 = per-head dots (tested variant, same bits).
    (256, 8, 32, "sm90"): dict(KVER=2, BI=16, BJ=8, num_warps=8, num_stages=1, EXP="libdevice"),
}
# NOT a dispatch table: candidate configs for cells that are implemented (constexpr-generic kernel) but NOT checked. They are reachable ONLY when the engineering
# flag FPF_TRIATT_EPI_ALLOW_UNVERIFIED=1 is set (checking jobs); the registry fn never uses them otherwise -> stock forward fallback.
CANDIDATE_CONFIG_UNVERIFIED = {
    (256, 8, 32, "sm80"): dict(KVER=2, BI=16, BJ=8, num_warps=8, num_stages=1, EXP="libdevice"),   # A100 class (code portable, untested)
    (128, 4, 32, "sm90"): dict(KVER=2, BI=8, BJ=8, num_warps=4, num_stages=1, EXP="libdevice"),    # the c=128, H=4, D=32 pair stack: 8x8 measured -14.4 % (L768) / -16.1 % (L384) vs 16x8 = 95 % of the 4-pass byte floor
    (64, 2, 32, "sm90"): dict(KVER=2, BI=16, BJ=8, num_warps=4, num_stages=1, EXP="libdevice"),    # the template pair stack (c=64, H=2, D=32)
    (64, 4, 16, "sm90"): dict(KVER=2, BI=16, BJ=8, num_warps=4, num_stages=1, EXP="libdevice"),    # the OpenFold3 template-stack TriAtt (c=64, H=4, D=16)
}

# Cells TESTED bit-exact vs stock on H100 (real dumps, N in {356,546,705,813}, pf_c1_b0 + pf_c10_b47) == exactly the keys of PINNED_CONFIG at import. The registry fn
# runs the kernel for these (c, H, D, gpu_class) cells; a cell the measured compositions meet with this kernel OFF (MEASURED_OFF: the candidate cells, and any cell merged
# into PINNED_CONFIG after import without a VERIFIED record) keeps the STOCK forward BY NAME ('cell:c<c>_h<H>_d<D>_<gpu>+off(not-measured)'); any OTHER cell is UNKNOWN and is
# served the SAFE epilogue settings of its capability (opt_core.kernels.safe_settings 'pair_fused:epilogue': single-stage tiles), engaged and named ONCE
# ('[opt_core/fpf_triatt_epi] safe settings served (no_cell:<cell>, cc .., triton ..)'); a cell no safe row admits stays on the stock forward ('cell:...[+no_safe(<why>)]').
VERIFIED = set(PINNED_CONFIG.keys())
MEASURED_OFF = set(CANDIDATE_CONFIG_UNVERIFIED)        # (c, H, D, gpu_class) cells the measured compositions run with this kernel OFF: the stock forward by name, never the SAFE settings
SAFE_LEVER = "pair_fused:epilogue"                      # the safe-settings rows whose settings this kernel's cfg reads (KVER, BI, BJ, num_warps, num_stages, EXP)
_NET = _safe.SafeNet("fpf_triatt_epi")                  # names the SAFE engagement once per process (settings_word() for a caller's status line)


def _cc(dev) -> str:
    return "%d.%d" % tuple(torch.cuda.get_device_capability(dev))


def cell_word(c: int, H: int, D: int, gpu: str) -> str:
    return "cell:c%d_h%d_d%d_%s" % (c, H, D, gpu)


def cell_verdict(c: int, H: int, D: int, dev=None, *, admit: str = "pinned", gpu: str = None, cc: str = None):
    """The ONE cell decision -> (kind, cfg, word) (opt_core.kernels.cell_words.decide): 'pinned' (PINNED_CONFIG; admit='verified': only cells with a VERIFIED record — a
    cell merged into the table at run time without one is 'off'), 'candidate' (FPF_TRIATT_EPI_ALLOW_UNVERIFIED=1), 'off' (MEASURED_OFF: word 'cell:...+off(not-measured)'),
    'safe' (an UNKNOWN cell: the capability's SAFE settings), 'none' (nothing admits it).  ``gpu`` / ``cc`` name the card for a decision off the device (tests)."""
    gpu = gpu if gpu is not None else _gpu_class(dev)
    cc = cc if cc is not None else _cc(dev)
    key = (int(c), int(H), int(D), gpu)
    pinned = PINNED_CONFIG.get(key) if (admit == "pinned" or key in VERIFIED) else None
    return _cw.decide(pinned=pinned, candidate=CANDIDATE_CONFIG_UNVERIFIED.get(key), allow_candidate=bool(os.environ.get("FPF_TRIATT_EPI_ALLOW_UNVERIFIED", "")),
                      measured_off=key in MEASURED_OFF or key in PINNED_CONFIG, miss_word=cell_word(*key), safe_lever=SAFE_LEVER, cc=cc,
                      dims={"c_z": int(c), "H": int(H), "D": int(D)}, shape_word="c%d_h%d_d%d_%s" % key)


def kernel_eligible(c: int, H: int, D: int, dev, **kw) -> bool:
    """True when the registry fn runs the kernel for this cell: a VERIFIED cell, an admitted candidate, or an UNKNOWN cell the SAFE settings serve."""
    return cell_verdict(c, H, D, dev, admit="verified", **kw)[0] in ("pinned", "candidate", "safe")


def stock_word(c: int, H: int, D: int, dev, **kw) -> str:
    """The census word of a cell the kernel does not run ('' when it runs): 'cell:...+off(not-measured)' / 'cell:...[+no_safe(...)]'."""
    kind, _cfg, word = cell_verdict(c, H, D, dev, admit="verified", **kw)
    return "" if kind in ("pinned", "candidate", "safe") else word


def config_sha256() -> str:
    return hashlib.sha256(json.dumps({"pinned": {repr(k): v for k, v in sorted(PINNED_CONFIG.items(), key=repr)}, "verified": sorted(repr(k) for k in VERIFIED)}, sort_keys=True).encode()).hexdigest()


def settings_word():
    """``safe:no_cell:<cell>`` once the SAFE settings serve an UNKNOWN cell in this process (opt_core.kernels.safe_settings), else None."""
    return _NET.word()


def _gpu_class(dev) -> str:
    cap = torch.cuda.get_device_capability(dev)
    if cap == (9, 0):
        return "sm90"
    if cap[0] == 8:
        return "sm80"                                                  # shipped A100-class key (code portable, untested)
    return f"sm{cap[0]}{cap[1]}"                                       # exact arch key; no sm100/sm120 cell is pinned here: such cells arrive as a caller-supplied cfg= or are served the SAFE settings


def pick_config(c: int, H: int, D: int, dev, **kw) -> dict:
    env = os.environ.get("FPF_TRIATT_EPI_CFG", "").strip()
    if env:                                     # offline sweeps only (never set in runs)
        vals = [int(v) for v in env.split(",")]
        BI, BJ, W, S = vals[:4]; KV = vals[4] if len(vals) > 4 else 2
        return dict(KVER=KV, BI=BI, BJ=BJ, num_warps=W, num_stages=S, EXP=os.environ.get("FPF_TRIATT_EPI_EXP", "libdevice"), _env=True)
    kind, cfg, word = cell_verdict(c, H, D, dev, **kw)
    if kind == "safe":                          # an UNKNOWN cell: the capability's SAFE settings, engaged and named ONCE
        _NET.engage(word, _safe.where_word(kw.get("cc") or _cc(dev), _safe.triton_mm()))
    if cfg is None:
        raise NotImplementedError(f"triatt_epilogue: no launch config for cell {(c, H, D, kw.get('gpu') or _gpu_class(dev))} ({word}) — caller must use the stock path")
    return cfg


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
if _HAS_TRITON:
    _HAS_LD = tl.constexpr(_ld is not None)

    @triton.jit
    def _triatt_epilogue_kernel(O, G, W, Z, OUT,
                                I, J,
                                so_b, so_i, so_h, so_j,          # o strides (elements): [B, I, H, J, D], stride_d == 1
                                sg_b, sg_i, sg_j,                # g strides: [B, I, J, H*D], last == 1
                                sw_n,                            # Wo [c_out, K] row stride (K contiguous) -> WoT[k, n] = W + n*sw_n + k
                                sz_b, sz_r, sz_c,                # z strides: [B, R, Cc, c] (last == 1); start: (zi,zj)=(i,j) ; ending: (zi,zj)=(j,i)
                                su_b, su_i, su_j,                # out strides [B, I, J, c] (op mode)
                                H: tl.constexpr, D: tl.constexpr, C: tl.constexpr,
                                BI: tl.constexpr, BJ: tl.constexpr,
                                ENDING: tl.constexpr, RESIDUAL: tl.constexpr, USE_LD: tl.constexpr, RESIDUAL_OUT: tl.constexpr = False):
        BM: tl.constexpr = BI * BJ
        pid_i = tl.program_id(0); pid_j = tl.program_id(1); pid_b = tl.program_id(2).to(tl.int32)
        r = tl.arange(0, BM)
        ii = pid_i * BI + r // BJ                     # [BM] i index of row r  (i-major inside the tile: rows r = bi*BJ + bj)
        jj = pid_j * BJ + r % BJ                      # [BM] j index
        rmask = (ii < I) & (jj < J)
        ii64 = ii.to(tl.int32); jj64 = jj.to(tl.int32)
        d = tl.arange(0, D)                           # [D]
        n = tl.arange(0, C)                           # [C] output channels
        o_row = O + pid_b * so_b + ii64 * so_i + jj64 * so_j            # [BM] pointers to O[b, i, 0, j, 0]
        g_row = G + pid_b * sg_b + ii64 * sg_i + jj64 * sg_j            # [BM] pointers to G[b, i, j, 0]
        acc = tl.zeros((BM, C), dtype=tl.float32)
        for h in tl.static_range(H):
            o_h = tl.load(o_row[:, None] + h * so_h + d[None, :], mask=rmask[:, None], other=0.0)          # [BM, D] bf16
            g_h = tl.load(g_row[:, None] + h * D + d[None, :], mask=rmask[:, None], other=0.0)             # [BM, D] bf16
            gf = g_h.to(tl.float32)
            if USE_LD:
                e = _ld.exp(-gf)
            else:
                e = tl.exp(-gf)
            sg = (1.0 / (1.0 + e)).to(tl.bfloat16).to(tl.float32)                                      # ATen sigmoid on bf16: opmath fp32, rounded to bf16
            gated = (o_h.to(tl.float32) * sg).to(tl.bfloat16)                                            # o * g : fp32 mul of two bf16 -> bf16
            w_h = tl.load(W + n[None, :] * sw_n + (h * D + d)[:, None])                                  # [D, C] = WoT rows hD..hD+D  (WoT[k,n] = Wo[n,k])
            acc = tl.dot(gated, w_h, acc)                                                                # fp32 accumulate, K ascending across heads (one chain)
        u16 = acc.to(tl.bfloat16)
        if RESIDUAL:
            if ENDING:
                z_ptr = Z + pid_b * sz_b + jj64[:, None] * sz_r + ii64[:, None] * sz_c + n[None, :]
            else:
                z_ptr = Z + pid_b * sz_b + ii64[:, None] * sz_r + jj64[:, None] * sz_c + n[None, :]
            zt = tl.load(z_ptr, mask=rmask[:, None], other=0.0)
            znew = (zt.to(tl.float32) + u16.to(tl.float32)).to(tl.bfloat16)                              # z += u : fp32 add of two bf16 -> bf16
            if RESIDUAL_OUT:                                                                             # out-of-place block mode: same value, a different destination
                if ENDING:
                    r_ptr = OUT + pid_b * su_b + jj64[:, None] * su_i + ii64[:, None] * su_j + n[None, :]
                else:
                    r_ptr = OUT + pid_b * su_b + ii64[:, None] * su_i + jj64[:, None] * su_j + n[None, :]
                tl.store(r_ptr, znew, mask=rmask[:, None])
            else:
                tl.store(z_ptr, znew, mask=rmask[:, None])
        else:
            u_ptr = OUT + pid_b * su_b + ii64[:, None] * su_i + jj64[:, None] * su_j + n[None, :]
            tl.store(u_ptr, u16, mask=rmask[:, None])

    @triton.jit
    def _triatt_epilogue_kernel_v2(O, G, WT, Z, OUT,
                                   I, J,
                                   so_b, so_i, so_h, so_j,
                                   sg_b, sg_i, sg_j,
                                   swt_k,                        # WoT [K, C] contiguous (row stride swt_k == C): WoT[k, n] = Wo[n, k]  (cached transposed copy of the bf16 weight)
                                   sz_b, sz_r, sz_c,
                                   su_b, su_i, su_j,
                                   H: tl.constexpr, D: tl.constexpr, C: tl.constexpr,
                                   BI: tl.constexpr, BJ: tl.constexpr,
                                   ENDING: tl.constexpr, RESIDUAL: tl.constexpr, USE_LD: tl.constexpr, RESIDUAL_OUT: tl.constexpr = False):
        # v2: gather the full gated tile [BM, H*D] (column k = h*D + d  <-  O[i, h, j, d]) and do ONE tl.dot over K = H*D against the coalesced WoT tile.
        # Same arithmetic as v1 (single fp32 accumulator chain over ascending k) -> same bits; fewer, larger, coalesced loads.
        BM: tl.constexpr = BI * BJ
        K: tl.constexpr = H * D
        pid_i = tl.program_id(0); pid_j = tl.program_id(1); pid_b = tl.program_id(2).to(tl.int32)
        r = tl.arange(0, BM)
        ii = pid_i * BI + r // BJ
        jj = pid_j * BJ + r % BJ
        rmask = (ii < I) & (jj < J)
        ii64 = ii.to(tl.int32); jj64 = jj.to(tl.int32)
        k = tl.arange(0, K)                            # [K] gathered column index
        hh = k // D; dd = k - hh * D
        n = tl.arange(0, C)
        o_ptr = O + pid_b * so_b + ii64[:, None] * so_i + jj64[:, None] * so_j + (hh * so_h + dd)[None, :]      # [BM, K]
        g_ptr = G + pid_b * sg_b + ii64[:, None] * sg_i + jj64[:, None] * sg_j + k[None, :]                    # [BM, K] (g column = h*D+d = k)
        o_t = tl.load(o_ptr, mask=rmask[:, None], other=0.0)
        g_t = tl.load(g_ptr, mask=rmask[:, None], other=0.0)
        gf = g_t.to(tl.float32)
        if USE_LD:
            e = _ld.exp(-gf)
        else:
            e = tl.exp(-gf)
        sg = (1.0 / (1.0 + e)).to(tl.bfloat16).to(tl.float32)
        gated = (o_t.to(tl.float32) * sg).to(tl.bfloat16)                                                     # [BM, K] bf16
        w = tl.load(WT + k[:, None].to(tl.int32) * swt_k + n[None, :])                                          # [K, C] bf16, rows contiguous (coalesced)
        acc = tl.dot(gated, w)                                                                                  # fp32 acc, single chain over K
        u16 = acc.to(tl.bfloat16)
        if RESIDUAL:
            if ENDING:
                z_ptr = Z + pid_b * sz_b + jj64[:, None] * sz_r + ii64[:, None] * sz_c + n[None, :]
            else:
                z_ptr = Z + pid_b * sz_b + ii64[:, None] * sz_r + jj64[:, None] * sz_c + n[None, :]
            zt = tl.load(z_ptr, mask=rmask[:, None], other=0.0)
            znew = (zt.to(tl.float32) + u16.to(tl.float32)).to(tl.bfloat16)
            if RESIDUAL_OUT:                                                                             # out-of-place block mode: same value, a different destination
                if ENDING:
                    r_ptr = OUT + pid_b * su_b + jj64[:, None] * su_i + ii64[:, None] * su_j + n[None, :]
                else:
                    r_ptr = OUT + pid_b * su_b + ii64[:, None] * su_i + jj64[:, None] * su_j + n[None, :]
                tl.store(r_ptr, znew, mask=rmask[:, None])
            else:
                tl.store(z_ptr, znew, mask=rmask[:, None])
        else:
            u_ptr = OUT + pid_b * su_b + ii64[:, None] * su_i + jj64[:, None] * su_j + n[None, :]
            tl.store(u_ptr, u16, mask=rmask[:, None])


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------