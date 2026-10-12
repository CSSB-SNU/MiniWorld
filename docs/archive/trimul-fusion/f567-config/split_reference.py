import torch
from miniworld_engine.kernels.trimul_inproj.triton.bidirectional import (bidir_front_triton,packed_forward,packed_backward,front_bwd_dW,_te_forward,_te_backward,gate_elem_train,gate_elem_bwd_ew)
class SplitReference(torch.autograd.Function):
    """front → 2 contractions (outgoing [:h] / incoming [h:]) → LN_out+@Wp → gate,
    as ONE Function so the backward matches cute's fused structure (gate dx_n add
    folded into the front dxn GEMM). Weights x@W form; Wp is nn.Linear (N,K) form."""

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
        proj, te_xn, mean_out, rstd_out = _te_forward(
            view, ln_out_w, ln_out_b, Wp, None, eps)              # (M, D)
        # fuse the pairformer residual (== module input pair [M,D]) + row-broadcast dropout
        # into the gate store epilogue (same path the cute dispatch uses).
        y, gate = gate_elem_train(x_n.reshape(M, D), proj, Wg, residual, dropscale, seq_len=L)
        ctx.save_for_backward(x_n, WL, WLg, WR, WRg, Wg, Wp, ln_out_w,
                              preact, lf, rf, tri, te_xn, mean_out, rstd_out, gate, proj)
        ctx.eps, ctx.h, ctx.mm = eps, h, mm
        ctx.dropscale, ctx.seq_len = dropscale, L
        return y.reshape(B, L, L, D)

    @staticmethod
    def backward(ctx, gy):
        (x_n, WL, WLg, WR, WRg, Wg, Wp, ln_out_w,
         preact, lf, rf, tri, te_xn, mean_out, rstd_out, gate, proj) = ctx.saved_tensors
        B, L, _, D = x_n.shape
        M = B * L * L
        h = ctx.h
        H = 2 * h
        gy = gy.reshape(M, D).contiguous()  # guard: autograd may hand a non-contiguous / broadcast (.sum) grad
        # residual grad passes straight through; op-branch grad is drop_row-scaled in gate_elem_bwd_ew
        d_residual = gy.reshape(M, D)

        # ② gate bwd (elementwise; dx_gate folded into the dxn GEMM below); dropout-scale dy
        d_proj, d_glogit = gate_elem_bwd_ew(gy, proj, gate, ctx.dropscale, ctx.seq_len)
        dWg = torch.mm(x_n.reshape(M, D).t(), d_glogit)           # (D, D) cuBLAS

        # ① LN_out + @Wp bwd (te_style)
        view = tri.reshape(H, M).t()
        d_view, dLNo_w, dLNo_b, dWp, _ = _te_backward(
            d_proj, te_xn, view, mean_out, rstd_out, ln_out_w, Wp, has_bias=False)
        # `_te_backward` writes dx at x's strides and x here is `view` ((1, M)), so d_view is
        # m-major, `.t()` is contiguous and this reshape is a FREE VIEW -- d_tri ALIASES
        # d_view. Deleting the name frees nothing on its own; the storage goes at the `del`
        # below, which names d_tri and both of its slices. (bidir_training_sm100.py's header
        # states the same aliasing; an earlier comment here claimed a copy and was wrong.)
        d_tri = d_view.t().reshape(H, L, L)
        del d_view, d_proj, view

        # contraction bwd (split outgoing/incoming), cuBLAS bmm
        d_left, d_right = packed_backward(d_tri, lf, rf, h)
        d_left = d_left.reshape(B, H, L, L)
        d_right = d_right.reshape(B, H, L, L)
        del d_tri
        # front bwd: d_concat (triton) + dW (cuBLAS) + W_stack; dxn fuses the gate add
        dconc, dWL, dWLg, dWR, dWRg, W_stack = front_bwd_dW(
            d_left, d_right, preact, x_n, WL, WLg, WR, WRg, pair_mask=ctx.mm)
        dx = torch.mm(d_glogit, Wg.t())                          # dx_gate  (M, D)
        dx.addmm_(dconc.t(), W_stack)                            # + dconcᵀ@W_stack (in-place)
        dx_n = dx.reshape(B, L, L, D)
        # trailing Nones: eps, h, mask; then d_residual (fused residual input), dropscale
        return (dx_n, dWL, dWLg, dWR, dWRg, dWg, dWp, dLNo_w, dLNo_b, None, None, None,
                d_residual, None)
