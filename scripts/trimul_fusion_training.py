"""Experimental forward substitutions; backward is the unchanged production function.
Generated from the installed Triton class; F1--F3 and the complete save order are
retained. Imported only by the isolated comparison harness.
"""
import torch
from miniworld_engine.kernels.trimul_inproj.triton.bidirectional import (
    bidir_front_triton, packed_forward, _BidirBackHalfTriton,
)
from trimul_output_fusion import output_forward

class BidirFusionA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_n, WL, WLg, WR, WRg, Wg, Wp, ln_out_w, ln_out_b, eps, h, mask,
                residual, dropscale=None):
        B, L, _, D = x_n.shape
        M = B * L * L
        H = 2 * h                                                 # = WL.shape[1]
        left, right, preact = bidir_front_triton(x_n, WL, WLg, WR, WRg, pair_mask=mask)
        lf = left.reshape(H, L, L)
        rf = right.reshape(H, L, L)
        # Mask applies to the contraction inputs (left/right) ONLY — NOT to x_n, so the
        # output gate sigmoid(x_n@Wg) stays unmasked (matches the pytorch reference).
        mm = mask  # Applied inside the front stores; x_n/output gate stay unmasked.
        tri = packed_forward(lf, rf, h)
        view = tri.reshape(H, M).t()                              # (M, H) m-major
        y, proj, te_xn, mean_out, rstd_out, gate = output_forward(
            view, x_n.reshape(M, D), ln_out_w, ln_out_b, Wp, Wg,
            residual, dropscale, eps, L, 'A')
        ctx.save_for_backward(x_n, WL, WLg, WR, WRg, Wg, Wp, ln_out_w,
                              preact, lf, rf, tri, te_xn, mean_out, rstd_out, gate, proj)
        ctx.eps, ctx.h, ctx.mm = eps, h, mm
        ctx.dropscale, ctx.seq_len = dropscale, L
        return y.reshape(B, L, L, D)

    backward = staticmethod(_BidirBackHalfTriton.backward)

class BidirFusionB(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_n, WL, WLg, WR, WRg, Wg, Wp, ln_out_w, ln_out_b, eps, h, mask,
                residual, dropscale=None):
        B, L, _, D = x_n.shape
        M = B * L * L
        H = 2 * h                                                 # = WL.shape[1]
        left, right, preact = bidir_front_triton(x_n, WL, WLg, WR, WRg, pair_mask=mask)
        lf = left.reshape(H, L, L)
        rf = right.reshape(H, L, L)
        # Mask applies to the contraction inputs (left/right) ONLY — NOT to x_n, so the
        # output gate sigmoid(x_n@Wg) stays unmasked (matches the pytorch reference).
        mm = mask  # Applied inside the front stores; x_n/output gate stay unmasked.
        tri = packed_forward(lf, rf, h)
        view = tri.reshape(H, M).t()                              # (M, H) m-major
        y, proj, te_xn, mean_out, rstd_out, gate = output_forward(
            view, x_n.reshape(M, D), ln_out_w, ln_out_b, Wp, Wg,
            residual, dropscale, eps, L, 'B')
        ctx.save_for_backward(x_n, WL, WLg, WR, WRg, Wg, Wp, ln_out_w,
                              preact, lf, rf, tri, te_xn, mean_out, rstd_out, gate, proj)
        ctx.eps, ctx.h, ctx.mm = eps, h, mm
        ctx.dropscale, ctx.seq_len = dropscale, L
        return y.reshape(B, L, L, D)

    backward = staticmethod(_BidirBackHalfTriton.backward)
