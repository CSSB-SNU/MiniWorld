"""Graph-safe Protenix training loss: ProtenixLoss(mode="train") with the sparse smooth-LDDT / bond path, same terms and weights.

Everything that depends only on the sample (coordinate mask, representative atoms, frames, the sparse LDDT / bond pair lists,
true distances and bins, resolution validity) is computed once, eagerly, at construction. The per-step part uses fixed-size
index_select / gather only -- no nonzero, boolean indexing, .item() or NaN checks -- so it can be captured in a CUDA graph
together with the model. The rigid alignment of the MSE term uses Horn's quaternion method (top eigenvector of the 4x4 matrix
by repeated squaring) instead of torch.linalg.svd, whose error check syncs; it gives the same proper rotation as the SVD with
the reflection fix. Cross entropies against one-hot labels are gathers of log_softmax (same value and gradient).
"""
import os
import torch
import torch.nn.functional as F
from protenix.model.modules.frames import expressCoordinatesInFrame, gather_frame_atom_by_indices
from protenix.utils.torch_utils import cdist


def _ce(logits, idx):
    """softmax_cross_entropy(logits, one_hot(idx)) = -log_softmax(logits)[..., idx]"""
    return -torch.gather(F.log_softmax(logits, dim=-1), -1, idx.unsqueeze(-1)).squeeze(-1)


def horn_rotation(H, n_sq=40):
    """Proper rotation R maximising sum_i w_i b_i . (R a_i), given H = sum_i w_i a_i b_i^T  [..., 3, 3] (Horn 1987).
    Top eigenvector of Horn's symmetric 4x4 matrix N: (N + |N|_F I) is PSD; its normalised repeated square converges to the
    rank-one projector on that eigenvector (fp64, fixed iteration count, no host sync)."""
    H = H.double()
    Sxx, Sxy, Sxz = H[..., 0, 0], H[..., 0, 1], H[..., 0, 2]
    Syx, Syy, Syz = H[..., 1, 0], H[..., 1, 1], H[..., 1, 2]
    Szx, Szy, Szz = H[..., 2, 0], H[..., 2, 1], H[..., 2, 2]
    N = torch.stack([
        torch.stack([Sxx + Syy + Szz, Syz - Szy, Szx - Sxz, Sxy - Syx], -1),
        torch.stack([Syz - Szy, Sxx - Syy - Szz, Sxy + Syx, Szx + Sxz], -1),
        torch.stack([Szx - Sxz, Sxy + Syx, -Sxx + Syy - Szz, Syz + Szy], -1),
        torch.stack([Sxy - Syx, Szx + Sxz, Syz + Szy, -Sxx - Syy + Szz], -1)], -2)
    P = N + torch.linalg.matrix_norm(N, keepdim=True) * torch.eye(4, dtype=N.dtype, device=N.device)
    for _ in range(n_sq):
        P = P / torch.linalg.matrix_norm(P, keepdim=True).clamp_min(1e-300)
        P = P @ P
    j = torch.diagonal(P, dim1=-2, dim2=-1).argmax(-1)          # the column with the largest entry of the projector
    q = torch.gather(P, -1, j[..., None, None].expand(*P.shape[:-1], 1)).squeeze(-1)
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-300)
    q0, q1, q2, q3 = q.unbind(-1)
    R = torch.stack([
        torch.stack([q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3, 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)], -1),
        torch.stack([2 * (q1 * q2 + q0 * q3), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3, 2 * (q2 * q3 - q0 * q1)], -1),
        torch.stack([2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3], -1)], -2)
    return R


class GraphSafeLoss(torch.nn.Module):
    CONF = ("plddt_loss", "pde_loss", "resolved_loss", "pae_loss")

    def __init__(self, loss_mod, cfg, feat, label):
        super().__init__()
        assert not cfg.train_confidence_only
        assert cfg.loss_metrics_sparse_enable and cfg.loss.diffusion_sparse_loss_enable and not cfg.loss.diffusion_lddt_loss_dense, \
            "implements the sparse smooth-LDDT / sparse bond path"
        L = loss_mod
        self.wt = dict(L.loss_weight)
        self.sd = float(cfg.sigma_data)
        self.mse_cfg = L.mse_loss
        self.eps = dict(dg=L.distogram_loss.eps, pde=L.pde_loss.eps, pae=L.pae_loss.eps, plddt=L.plddt_loss.eps)
        self.no_bins = dict(dg=L.distogram_loss.no_bins, pde=L.pde_loss.no_bins, pae=L.pae_loss.no_bins, plddt=L.plddt_loss.no_bins)
        assert L.plddt_loss.normalize
        res = float(feat["resolution"].reshape(-1)[0])
        self.valid_res = bool(cfg.loss.resolution.min <= res <= cfg.loss.resolution.max)
        reg = lambda k, v: self.register_buffer(k, v, persistent=False)
        xt, cm = label["coordinate"], label["coordinate_mask"]
        with torch.no_grad():
            lab = L.calculate_label(feat, dict(label))
            # smooth LDDT (sparse): pair list and true distances
            li, mi = torch.nonzero(lab["lddt_mask"], as_tuple=True)
            reg("lddt_l", li); reg("lddt_m", mi)
            reg("lddt_true", torch.linalg.vector_norm(xt.index_select(-2, li) - xt.index_select(-2, mi), ord=2, dim=-1))
            self.csr = None
            if os.environ.get("PFX_LOSS_SLDDT_CUDA", "1") == "1":     # one-kernel forward + gradient (pfx_slddt_cuda)
                import pfx_slddt_cuda
                self.csr = pfx_slddt_cuda.build_csr(li, mi, self.lddt_true, xt.size(-2))
            # bond (sparse)
            bi, bj = torch.nonzero(feat["bond_mask"] * lab["distance_mask"], as_tuple=True)
            self.n_bond = int(bi.numel())
            reg("bond_i", bi); reg("bond_j", bj)
            reg("bond_true", torch.linalg.vector_norm(xt.index_select(-2, bi) - xt.index_select(-2, bj), ord=2, dim=-1))
            del lab
            # MSE: per-atom weight, masked true coordinates and their weighted centroid
            M = self.mse_cfg
            w = (1 + M.weight_dna * feat["is_dna"] + M.weight_rna * feat["is_rna"] + M.weight_ligand * feat["is_ligand"]) * cm
            w = w.float()
            reg("mse_w", w); reg("cm", cm)
            a = (xt * cm.unsqueeze(-1)).float()
            wn = w.sum(-1, keepdim=True).unsqueeze(-1)
            reg("mse_wn", wn)
            mu_a = torch.sum(a * w.unsqueeze(-1), dim=-2, keepdim=True) / wn
            reg("mse_ac", a - mu_a)
            reg("mse_den", cm.sum(dim=-1, keepdim=True) + M.eps)
            # distogram: true bins and pair mask
            tb, pm = L.distogram_loss.calculate_label(true_coordinate=xt, coordinate_mask=cm, rep_atom_mask=feat["distogram_rep_atom_mask"])
            reg("dg_bins", tb.argmax(-1)); reg("dg_pm", pm)
            # plddt: atoms with coordinates, m-atoms, true l-m distances, locality pair mask
            cidx = cm.bool().nonzero().squeeze(-1)
            reg("pl_cidx", cidx)
            xc = xt.index_select(-2, cidx)
            is_nuc = (feat["is_rna"] + feat["is_dna"]).index_select(0, cidx).bool()
            is_poly = (1 - feat["is_ligand"]).index_select(0, cidx)
            rep = feat["plddt_m_rep_atom_mask"].index_select(0, cidx).bool()
            mloc = (rep * is_poly).bool().nonzero().squeeze(-1)
            reg("pl_m", mloc)
            td = torch.cdist(xc, xc.index_select(-2, mloc))
            nm = is_nuc.index_select(0, mloc)
            loc = (td < L.plddt_loss.is_nucleotide_threshold) * nm.unsqueeze(-2) + (td < L.plddt_loss.is_not_nucleotide_threshold) * (~nm.unsqueeze(-2))
            diag = (1 - torch.eye(xc.size(-2), device=xc.device, dtype=td.dtype)).bool().index_select(-1, mloc)
            pair = loc * diag
            reg("pl_true", td); reg("pl_pair", pair)
            reg("pl_wsum", torch.sum(pair.to(td.dtype), dim=-1, keepdim=True))
            reg("pl_bound", torch.linspace(L.plddt_loss.min_bin, L.plddt_loss.max_bin, L.plddt_loss.no_bins + 1, device=xt.device))
            # pde: representative atoms, true distances, pair mask
            rep = feat["distogram_rep_atom_mask"].bool().nonzero().squeeze(-1)
            reg("pde_rep", rep)
            xr = xt.index_select(-2, rep)
            reg("pde_gt", cdist(xr, xr))
            tm = cm.index_select(-1, rep)
            pm = tm[..., None] * tm[..., None, :]
            reg("pde_pm", pm); reg("pde_den", self.eps["pde"] + torch.sum(pm, dim=(-1, -2)))
            reg("pde_bound", torch.linspace(L.pde_loss.min_bin, L.pde_loss.max_bin, L.pde_loss.no_bins + 1, device=xt.device))
            # pae: frames that exist, their atoms, true frame-local coordinates, frame-token pair mask
            hf = feat["has_frame"].bool().nonzero().squeeze(-1)
            fai = feat["frame_atom_index"].index_select(0, hf)
            reg("pae_hf", hf); reg("pae_fai", fai)
            tf = gather_frame_atom_by_indices(coordinate=xt, frame_atom_index=fai, dim=-2)
            tfm = gather_frame_atom_by_indices(coordinate=cm.bool(), frame_atom_index=fai, dim=-1).sum(dim=-1) >= 3
            prep = feat["pae_rep_atom_mask"].bool().nonzero().squeeze(-1)
            reg("pae_rep", prep)
            pm = tfm[..., None] * cm.bool().index_select(-1, prep)[..., None, :]
            reg("pae_pm", pm); reg("pae_den", self.eps["pae"] + torch.sum(pm, dim=(-1, -2)))
            reg("pae_xt", expressCoordinatesInFrame(coordinate=xt.index_select(-2, prep), frames=tf))
            reg("pae_bound", torch.linspace(L.pae_loss.min_bin, L.pae_loss.max_bin, L.pae_loss.no_bins + 1, device=xt.device) ** 2)
            reg("res_lab", cm.long())

    # ---- terms ----
    def smooth_lddt(self, x):
        if self.csr is not None and x.dim() == 3:
            import pfx_slddt_cuda
            return pfx_slddt_cuda.smooth_lddt_loss(x, self.csr)
        return self.smooth_lddt_torch(x)

    def smooth_lddt_torch(self, x):
        pd = torch.linalg.vector_norm(x.index_select(-2, self.lddt_l) - x.index_select(-2, self.lddt_m), ord=2, dim=-1)
        dd = torch.abs(pd - self.lddt_true)
        e = 0
        for t in (0.5, 1, 2, 4):
            e += 0.25 * torch.sigmoid(t - dd)
        return 1 - torch.mean(torch.mean(e, dim=-1))

    def bond(self, x, scale):
        if self.n_bond == 0:
            return x.new_zeros(())
        pd = torch.linalg.vector_norm(x.index_select(-2, self.bond_i) - x.index_select(-2, self.bond_j), ord=2, dim=-1)
        b = torch.mean((pd - self.bond_true) ** 2, dim=-1)
        if scale is not None:
            b = b * scale
        return b.mean(dim=-1)

    def align(self, x):
        """MSELoss.weighted_rigid_align: the true coordinates rigidly moved onto each sample (no grad)."""
        with torch.no_grad():
            b = (x * self.cm[..., None, :, None]).float()
            w = self.mse_w
            mu_b = torch.sum(b * w.unsqueeze(-1), dim=-2, keepdim=True) / self.mse_wn
            H = torch.matmul((self.mse_ac * w.unsqueeze(-1)).transpose(-2, -1), b - mu_b)
            R = horn_rotation(H).to(b.dtype)
            return (torch.matmul(self.mse_ac, R.transpose(-1, -2)) + mu_b).to(x.dtype)

    def mse(self, x, scale):
        xa = self.align(x)
        se = ((x - xa) ** 2).sum(dim=-1)
        s = (self.mse_w * se).sum(dim=-1) / self.mse_den
        if scale is not None:
            s = s * scale
        return torch.mean(self.mse_cfg.weight_mse * s.mean(dim=-1))

    def distogram(self, logits):
        err = _ce(logits, self.dg_bins.expand(logits.shape[:-1]))
        return torch.mean(torch.sum(err * self.dg_pm, dim=(-1, -2)) / (self.eps["dg"] + torch.sum(self.dg_pm, dim=(-1, -2))))

    # the label (bin) computations stay out of torch.compile: compiled cdist rounds differently and flips a few edge bins
    @torch.compiler.disable
    @torch.no_grad()
    def plddt_bins(self, xm):
        pc = xm.index_select(-2, self.pl_cidx)
        dd = torch.abs(torch.cdist(pc, pc.index_select(-2, self.pl_m)) - self.pl_true.unsqueeze(-3))
        lddt_lm = ((dd < 0.5).to(dtype=dd.dtype) + (dd < 1.0) + (dd < 2.0) + (dd < 4.0)) * 0.25
        pal = torch.sum(lddt_lm * self.pl_pair.unsqueeze(-3), dim=-1, keepdim=True) / (self.pl_wsum + self.eps["plddt"])
        return torch.clamp(torch.sum(pal > self.pl_bound, dim=-1), min=1, max=self.no_bins["plddt"]) - 1

    @torch.compiler.disable
    @torch.no_grad()
    def pde_bins(self, xm):
        pr = xm.index_select(-2, self.pde_rep)
        de = torch.abs(cdist(pr, pr) - self.pde_gt.unsqueeze(-3))
        return torch.clamp(torch.sum(de.unsqueeze(-1) > self.pde_bound, dim=-1), min=1, max=self.no_bins["pde"]) - 1

    @torch.compiler.disable
    @torch.no_grad()
    def pae_bins(self, xm):
        pf = gather_frame_atom_by_indices(coordinate=xm, frame_atom_index=self.pae_fai, dim=-2)
        xp = expressCoordinatesInFrame(coordinate=xm.index_select(-2, self.pae_rep), frames=pf)
        sq = torch.sum((xp - self.pae_xt.unsqueeze(-4)) ** 2, dim=-1) * self.pae_pm
        tb = torch.sum(sq.unsqueeze(-1) > self.pae_bound, dim=-1)
        tb = torch.where(self.pae_pm, tb, torch.ones_like(tb) * self.no_bins["pae"])
        return torch.clamp(tb, min=1, max=self.no_bins["pae"]) - 1

    def plddt(self, logits, xm):
        bins = self.plddt_bins(xm)
        err = _ce(logits.index_select(-2, self.pl_cidx), bins)
        return torch.mean(err.mean(dim=-1).mean(dim=-1))

    def pde(self, logits, xm):
        bins = self.pde_bins(xm)
        err = _ce(logits, bins)
        l = torch.sum(err * self.pde_pm.unsqueeze(-3), dim=(-1, -2)) / self.pde_den.unsqueeze(-1)
        return torch.mean(l.mean(dim=-1))

    def resolved(self, logits):
        err = _ce(logits, self.res_lab.expand(logits.shape[:-1]))
        return torch.mean(err.mean(dim=-1).mean(dim=-1))

    def pae(self, logits, xm):
        bins = self.pae_bins(xm)
        err = _ce(logits.index_select(-3, self.pae_hf), bins)
        l = torch.sum(err * self.pae_pm.unsqueeze(-3), dim=(-1, -2)) / self.pae_den.unsqueeze(-1)
        return torch.mean(l.mean(dim=-1))

    def forward(self, pred, which="all"):
        """pred: Protenix's pred_dict keys (coordinate, noise_level, distogram, coordinate_mini, plddt, pde, pae, resolved).
        which: "all" | "diff" (diffusion + distogram terms) | "conf" (confidence terms). Returns (weighted sum, {term: loss})."""
        terms = {}
        if which in ("all", "diff"):
            x, sig = pred["coordinate"], pred["noise_level"]
            scale = (sig ** 2 + self.sd ** 2) / (self.sd * sig) ** 2
            terms.update({"smooth_lddt_loss": self.smooth_lddt(x), "bond_loss": self.bond(x, scale), "mse_loss": self.mse(x, scale)})
            if "distogram" in pred:
                terms["distogram_loss"] = self.distogram(pred["distogram"])
        if which in ("all", "conf") and all(k in pred for k in ("plddt", "pde", "pae", "resolved")):
            xm = pred["coordinate_mini"].detach()
            terms["plddt_loss"] = self.plddt(pred["plddt"], xm)
            terms["pde_loss"] = self.pde(pred["pde"], xm)
            terms["resolved_loss"] = self.resolved(pred["resolved"])
            terms["pae_loss"] = self.pae(pred["pae"], xm)
        cum = 0.0
        for k, v in terms.items():
            if k in self.CONF and not self.valid_res:
                v = 0.0 * v
            cum = cum + self.wt[k] * v
        return cum, terms
