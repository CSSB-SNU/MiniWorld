"""Capture-safe stand-ins for the host-synchronising pieces of the phase-2 diffusion training step.

Why: the step as the trainer runs it cannot be recorded into a CUDA graph. ``torch.linalg.svd``, ``torch.tensor(..., device=cuda)``,
``.item()`` and ``isnan`` host reads fail during capture, and scipy ``Rotation.random`` / CPU ``torch.rand`` would be frozen into
constants. These functions compute the same quantities with GPU tensor ops only:

  * ``weighted_align_gs``    Kabsch alignment through Horn's quaternion, equal to ``team_gm.utils.align.weighted_align``
                             (``tests/test_phase2_graph_safe.py``). Plain fp32 elementwise / reduction ops, NO matmul or einsum:
                             under ``torch.set_float32_matmul_precision("medium")`` (the training scripts set it) a matmul would run in
                             bf16 and lose ~3 digits on coordinates with an extent of hundreds of Angstrom.
  * ``rot_trans_gs`` / ``sample_noise_gs``   the augmentation rotation and the EDM noise level drawn on the GPU
  * ``cal_loss_gs``          ``EuclideanDiffuser.cal_loss`` without the host-side NaN branch
  * ``bond_loss_gs``         the AF3 Eq.5 bond loss over a fixed-size, padded bond-pair list (phase 2b)
  * ``graph_safe_sampling``  swaps the two sampling functions into a diffuser for the duration of a ``with`` block
  * ``clear_pack_caches``    the engine's weight-pack caches (keyed on data_ptr / _version) must be empty when a graph is captured, or
                             the replays would keep using the packed weights of the capture-time step
"""

import contextlib
import sys
import types

import torch


def quat_to_rotmat(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        -1,
    ).reshape(*q.shape[:-1], 3, 3)


def horn_rot(H, squarings=24):
    """R_col (acts on column vectors) maximising tr(R_col H), H_ab = sum_i w x_a y_b, so R_col x ~ y: the top eigenvector of
    Horn's symmetric 4x4 matrix by repeated squaring of (N + ||N|| I) -- a fixed number of tensor ops, no host sync.
    ``squarings`` = 24 raises M to 2**24: converged for eigenvalue gaps down to ~1e-6 of ||N|| (an extended, almost collinear
    1149 A chain needs > 12; compact protein-sized crops converge with a handful)."""
    Sxx, Sxy, Sxz = H[:, 0, 0], H[:, 0, 1], H[:, 0, 2]
    Syx, Syy, Syz = H[:, 1, 0], H[:, 1, 1], H[:, 1, 2]
    Szx, Szy, Szz = H[:, 2, 0], H[:, 2, 1], H[:, 2, 2]
    N = torch.stack(
        [
            torch.stack([Sxx + Syy + Szz, Syz - Szy, Szx - Sxz, Sxy - Syx], -1),
            torch.stack([Syz - Szy, Sxx - Syy - Szz, Sxy + Syx, Szx + Sxz], -1),
            torch.stack([Szx - Sxz, Sxy + Syx, -Sxx + Syy - Szz, Syz + Szy], -1),
            torch.stack([Sxy - Syx, Szx + Sxz, Syz + Szy, -Sxx - Syy + Szz], -1),
        ],
        -2,
    )
    nrm = N.flatten(-2).norm(dim=-1)[:, None, None].clamp_min(1e-12)
    M = (
        N + nrm * torch.eye(4, device=H.device, dtype=H.dtype)
    ) / nrm  # eigenvalues in [0, 2]
    for _ in range(squarings):
        M = (M.unsqueeze(-1) * M.unsqueeze(-3)).sum(
            -2
        )  # M @ M, exact fp32 (no tensor cores)
        M = M / M.flatten(-2).norm(dim=-1)[:, None, None].clamp_min(1e-30)
    u0 = torch.arange(
        1, 5, device=H.device, dtype=H.dtype
    )  # generic start vector (a kernel, not an H2D copy)
    q = (M * u0).sum(-1)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    return quat_to_rotmat(q)


@torch.no_grad()
def weighted_align_gs(x, y, weight):
    """team_gm.utils.align.weighted_align with the SVD replaced by horn_rot; same conventions (aligned = x_c @ R + y_c)."""
    shape, L = x.shape, x.shape[-2]
    x, y, w = x.reshape(-1, L, 3), y.reshape(-1, L, 3), weight.reshape(-1, L)
    w_sum = w.sum(dim=-1, keepdim=True)
    w = w.unsqueeze(-1)
    xc = ((x * w).sum(dim=-2) / w_sum).unsqueeze(-2)
    yc = ((y * w).sum(dim=-2) / w_sum).unsqueeze(-2)
    x_cen, y_cen = x - xc, y - yc
    H = ((x_cen * w).unsqueeze(-1) * y_cen.unsqueeze(-2)).sum(
        dim=-3
    )  # [n,3,3], exact fp32
    R = horn_rot(H).transpose(-1, -2)  # row-vector convention
    aligned = (x_cen.unsqueeze(-1) * R.unsqueeze(-3)).sum(dim=-2) + yc
    return aligned.reshape(shape)


def rot_trans_gs(self, x):
    """``Diffuser.random_rotation_and_translation`` with a GPU-side uniform rotation (normalised Gaussian quaternion)."""
    shp = x.shape
    x = x.reshape(-1, shp[-2], 3)
    n = x.shape[0]
    q = torch.randn(n, 4, device=x.device, dtype=x.dtype)
    R = quat_to_rotmat(q / q.norm(dim=-1, keepdim=True))
    t = (
        torch.randn(n, 1, 3, device=x.device, dtype=x.dtype)
        * self.config.translation_noise
    )
    return (torch.bmm(x, R.transpose(-1, -2)) + t).reshape(shp)


def sample_noise_gs(self, batch_size, uniform=False):
    """``EDMScheduler.sample_noise`` with ``torch.rand`` on the GPU (ln(sigma / sigma_data) ~ N(P_mean, P_std^2), clamped)."""
    c = self.config
    u = torch.rand(batch_size, device=torch.device("cuda", torch.cuda.current_device()))
    sigma = c.sigma_data * torch.exp(c.P_mean + c.P_std * torch.special.ndtri(u))
    return torch.clamp(
        sigma, min=c.sigma_min * c.sigma_data, max=c.sigma_max * c.sigma_data
    )


def cal_loss_gs(diffuser, x0, x_input, x_update, sigma, mask, atom_weight, align_atom_weight=False):
    """``EuclideanDiffuser.cal_loss`` minus the host-side isnan/save branch, alignment through ``weighted_align_gs``.

    ``align_atom_weight=False`` aligns the ground truth with the resolved-atom mask as the weight, as ``cal_loss`` does. ``True`` is
    AF3 SI Eq.2 + Algorithm 28 (``weighted_rigid_align``): the alignment uses the same per-atom weights ``w_l = 1 + a_dna*is_dna + a_rna*is_rna +
    a_lig*is_ligand`` (Eq.4) as the MSE, times the resolved mask, so DNA / RNA / ligand atoms pull the superposition harder.
    The alignment carries no gradient either way."""
    sch, dtype = diffuser.scheduler, x_update.dtype
    noisy_x = x_input / sch.input_scale(sigma).to(device=x0.device, dtype=diffuser.dtype)
    x0, noisy_x = x0.to(dtype=dtype), noisy_x.to(dtype=dtype)
    c_skip, c_out = (
        sch.skip_scale(sigma).to(dtype=dtype),
        sch.output_scale(sigma).to(dtype=dtype),
    )
    weight = sch.loss_weight(sigma).to(dtype=dtype)
    weight = weight * mask.unsqueeze(-1)
    weight = weight * atom_weight.to(dtype=dtype).unsqueeze(-1)
    x_pred = c_skip * noisy_x + c_out * x_update
    mask = mask.expand(x_pred.shape[:-1])
    align_weight = torch.where(mask.any(dim=-1, keepdim=True), mask, True).to(dtype)
    if align_atom_weight:
        align_weight = align_weight * atom_weight.to(dtype)  # w_l >= 1: an empty example keeps a positive total
    x0_safe = torch.where(mask.unsqueeze(-1), x0, 0.0)
    pred_safe = torch.where(mask.unsqueeze(-1), x_pred, 0.0)
    x0_aligned = weighted_align_gs(x0_safe, pred_safe, align_weight)
    sq = torch.where(mask.unsqueeze(-1), (pred_safe - x0_aligned).pow(2), 0.0)
    numerator = (sq * weight).sum(dim=(-2, -1))
    denominator = mask.sum(dim=-1).clamp_min(1).to(dtype) * x_pred.shape[-1]
    return (numerator / denominator).mean()


def bond_loss_gs(diffuser, x0, x_input, x_update, sigma, bond_i, bond_j, bond_valid):
    """AF3 Eq.5 bond loss of ``DiffusionClient.loss_fn`` for a graph: the same MSE of bonded-atom-pair distances against the ground
    truth on the same ``get_x_pred`` prediction, weighted per augment by w(sigma) -- over a FIXED-size, padded pair list.

    ``bond_i`` / ``bond_j`` [P] atom indices and ``bond_valid`` [P] bool replace the sparse ``bond_atom_pairs`` (its length changes
    from sample to sample and is read on the host). Padded pairs are masked out of the mean; they use a finite non-zero vector so
    the backward of the norm never sees a zero distance. No bond in the batch gives 0, as the eager loss does.
    """
    x_pred = diffuser.get_x_pred(x_input=x_input, x_update=x_update, sigma=sigma)
    x0 = x0.to(dtype=x_pred.dtype)
    valid = bond_valid.to(dtype=x_pred.dtype)

    def distance(x):
        delta = x[..., bond_i, :] - x[..., bond_j, :]
        return torch.where(bond_valid[:, None], delta, 1.0).norm(dim=-1)

    per_augment = ((distance(x_pred) - distance(x0)).pow(2) * valid).sum(dim=-1) / valid.sum().clamp_min(1.0)  # [A, B]
    w_sigma = diffuser.scheduler.loss_weight(sigma).to(dtype=per_augment.dtype, device=per_augment.device)
    return (w_sigma.reshape(per_augment.shape) * per_augment).mean()


@contextlib.contextmanager
def graph_safe_sampling(diffuser):
    """Inside the block ``diffuser.sample`` draws its rotation and noise level on the GPU."""
    sch = diffuser.scheduler
    diffuser.random_rotation_and_translation = types.MethodType(rot_trans_gs, diffuser)
    sch.sample_noise = types.MethodType(sample_noise_gs, sch)
    try:
        yield
    finally:
        del diffuser.random_rotation_and_translation
        del sch.sample_noise


_PACK_CACHES = (
    "miniworld_engine.integrations.bias_only_dit_train",
    "miniworld_engine.kernels.swa_dit.cuda.sm100",
    "miniworld_engine.integrations.atom_dit",
    "miniworld_engine.integrations.token_dit_train",
)


def clear_pack_caches():
    """Empty the engine's python weight-pack caches (call right before and right after a capture)."""
    for name in _PACK_CACHES:
        mod = sys.modules.get(name)
        if mod is not None and hasattr(mod, "_PACKS"):
            mod._PACKS.clear()
