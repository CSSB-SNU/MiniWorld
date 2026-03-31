from __future__ import annotations

import math
from math import pi

import numpy as np
import torch
from torch import Tensor


def hat_so3(omega: Tensor) -> Tensor:
    """(B,3) -> (B,3,3) skew matrix."""
    wx, wy, wz = omega.unbind(-1)
    O = torch.zeros(*omega.shape[:-1], 3, 3, device=omega.device, dtype=omega.dtype)
    O[..., 0, 1] = -wz
    O[..., 0, 2] = wy
    O[..., 1, 0] = wz
    O[..., 1, 2] = -wx
    O[..., 2, 0] = -wy
    O[..., 2, 1] = wx
    return O


def rodrigues(omega: Tensor) -> Tensor:
    """Axis–angle to rotation matrix using Rodrigues. omega: (B,3)."""
    theta = omega.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    k = omega / theta
    K = hat_so3(k)
    I = torch.eye(3, device=omega.device, dtype=omega.dtype)
    I = I.expand(*omega.shape[:-1], 3, 3)
    s = torch.sin(theta)[..., None]
    c = torch.cos(theta)[..., None]
    R = I + s * K + (1 - c) * (K @ K)
    # For very small angles, use 1st order approx
    small = (theta.squeeze(-1) < 1e-4)[..., None, None]
    R_small = I + hat_so3(omega)
    return torch.where(small, R_small, R)


def exp_se3(omega: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
    """Exponential map from (omega,v) in se(3) to (R,T).
    omega, v: (B,3). Returns R: (B,3,3), T: (B,3).
    """
    B = omega.shape[0]
    R = rodrigues(omega)
    theta = omega.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    K = hat_so3(omega / theta)
    I = torch.eye(3, device=omega.device, dtype=omega.dtype).expand(B, 3, 3)
    s = torch.sin(theta)
    c = torch.cos(theta)
    A = s / theta
    Bcoef = (1 - c) / (theta**2)
    V = I + A[..., None] * K + Bcoef[..., None] * (K @ K)
    T = (V @ v.unsqueeze(-1)).squeeze(-1)
    return R, T


def apply_rt(R: Tensor, T: Tensor, pts: Tensor) -> Tensor:
    """Apply (R,T) to points. R:(B,3,3) or (3,3), T:(B,3) or (3,), pts:(...,3)."""
    if R.dim() == 2:
        return pts @ R.mT + T
    # batch over first dim of pts assumed to match R,T or be broadcastable
    return (pts @ R.transpose(-1, -2)) + T.unsqueeze(-2)


def apply_chain_rt(
    x: Tensor,
    R: Tensor,
    T: Tensor,
    atom_chain_break: dict,
    inverse: bool = False,
) -> Tensor:
    output = []
    for cc, chain_ID in enumerate(atom_chain_break.keys()):
        atom_start, atom_end = atom_chain_break[chain_ID]
        x_rt = x[:, atom_start : atom_end + 1]
        if inverse:
            _R = R[:, cc].mT
            _T = torch.einsum("bij,bj->bi", R[:, cc].mT, -T[:, cc])
            x_rt = apply_rt(_R, _T, x_rt)
        else:
            x_rt = apply_rt(R[:, cc], T[:, cc], x_rt)
        output.append(x_rt)
    output = torch.cat(output, dim=1)
    return output


def log_so3(R: Tensor) -> Tensor:
    """Log map for SO(3) (stable for near-identity). Returns axis-angle (3,)."""
    # Using formula with clamped trace
    tr = R.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos_theta = ((tr - 1) / 2).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)
    # For small angles, use approximation
    small = theta < 1e-4
    if small.any():
        # skew = (R - R^T)/2, vee(skew) is approx omega
        skew = 0.5 * (R - R.transpose(-1, -2))
        omega_approx = torch.stack(
            [skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]],
            dim=-1,
        )
        omega = omega_approx
    else:
        denom = 2 * torch.sin(theta)
        skew = (R - R.transpose(-1, -2)) / denom[..., None, None]
        omega = theta.unsqueeze(-1) * torch.stack(
            [skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]],
            dim=-1,
        )
    return omega


def compose(R1: Tensor, T1: Tensor, R2: Tensor, T2: Tensor) -> tuple[Tensor, Tensor]:
    """(R1,T1)∘(R2,T2)."""
    R = R1 @ R2
    T = (R1 @ T2.unsqueeze(-1)).squeeze(-1) + T1
    return R, T


def exp_so3(omega: torch.Tensor) -> torch.Tensor:
    """omega: (B,C,3)  ->  R: (B,C,3,3)"""
    B, C = omega.shape[:2]
    omega = omega.view(-1, 3)
    theta = omega.norm(dim=-1, keepdim=True)  # (C,1)
    u = torch.where(theta > 0, omega / theta, torch.zeros_like(omega))
    ux, uy, uz = u.unbind(-1)

    K = torch.zeros(B * C, 3, 3, device=omega.device, dtype=omega.dtype)
    K[:, 0, 1] = -uz
    K[:, 0, 2] = uy
    K[:, 1, 0] = uz
    K[:, 1, 2] = -ux
    K[:, 2, 0] = -uy
    K[:, 2, 1] = ux

    I = torch.eye(3, device=omega.device, dtype=omega.dtype).expand(B * C, 3, 3)
    th = theta.squeeze(-1)

    # small-angle safe sin/cos
    s = torch.where(th > 0, torch.sin(th), th - th**3 / 6 + th**5 / 120)
    c = torch.where(th > 0, torch.cos(th), 1 - th**2 / 2 + th**4 / 24)

    a = s.view(B * C, 1, 1)
    b = (1 - c).view(B * C, 1, 1)
    R = I + a * K + b * (K @ K)
    R = R.view(B, C, 3, 3)
    return R


# def sample_rotation_heat(
#     sigma_R,                   # float or (B,): 'heat time' for rotation
#     C: int,
#     steps: int = 64,
#     device=None,
#     dtype=torch.float32,
# ) -> torch.Tensor:
#     B = sigma_R.shape[0]
#     sigma_R = sigma_R[:,None].expand(-1,C).unsqueeze(-1)


#     R = torch.eye(3, device=device, dtype=dtype).expand(B,C,3,3).clone()
#     for _ in range(steps):
#         d_omega = torch.randn(B, C, 3, device=device, dtype=dtype) * sigma_R
#         R = R @ exp_so3(d_omega)
#     return R  # (C,3,3)


def sample_rotation_heat(
    sigma_R,
    C: int,
    steps: int = 64,
    device=None,
    dtype=torch.float32,
):
    """sigma_R: (B,) where sigma_R^2 is the 'heat time' t.
    Returns R: (B,C,3,3)
    """
    B = sigma_R.shape[0]
    # per-step std = sqrt( t / steps ) = sigma_R / sqrt(steps)
    step_std = (sigma_R / math.sqrt(steps)).view(B, 1, 1)  # (B,1,1)

    R = torch.eye(3, device=device, dtype=dtype).expand(B, C, 3, 3).clone()
    for _ in range(steps):
        d_omega = torch.randn(B, C, 3, device=device, dtype=dtype) * step_std
        R = R @ exp_so3(d_omega)  # right-invariant increments
    return R


def sample_translation(sigma_T, C):
    B = sigma_T.shape[0]
    sigma_T = sigma_T[:, None, None]  # (B,1,1)
    return torch.randn(B, C, 3, device=sigma_T.device, dtype=sigma_T.dtype) * sigma_T


def sample_rigid(
    sigma_R,
    sigma_T,
    C: int,
    steps_R: int = 64,
    device=None,
    dtype=torch.float32,
):
    R = sample_rotation_heat(sigma_R, C, steps=steps_R, device=device, dtype=dtype)
    T = sample_translation(sigma_T, C)
    return R, T


def se3_heat_step_sigma(
    R_t: Tensor,
    T_t: Tensor,
    sigma_R_t: Tensor,
    sigma_T_t: Tensor,  # (B,) or scalar
    sigma_R_s: Tensor,
    sigma_T_s: Tensor,  # target sigmas (B,) or scalar
    stochastic: bool = False,
    mode: str = "group",  # "group" or "algebra"
    eps: float = 1e-12,
) -> tuple[Tensor, Tensor]:
    B = R_t.shape[0]
    device, dtype = R_t.device, R_t.dtype
    # # cast/broadcast
    # def _to(x): return torch.as_tensor(x, device=device, dtype=dtype)
    # sigma_R_t = _to(sigma_R_t).reshape(-1)
    # sigma_T_t = _to(sigma_T_t).reshape(-1)
    # sigma_R_s = _to(sigma_R_s).reshape(-1)
    # sigma_T_s = _to(sigma_T_s).reshape(-1)

    # # ensure shapes broadcast to batch
    # B = R_t.shape[0]
    # if sigma_R_t.numel() == 1: sigma_R_t = sigma_R_t.expand(B)
    # if sigma_T_t.numel() == 1: sigma_T_t = sigma_T_t.expand(B)
    # if sigma_R_s.numel() == 1: sigma_R_s = sigma_R_s.expand(B)
    # if sigma_T_s.numel() == 1: sigma_T_s = sigma_T_s.expand(B)

    # # ---- Rotation step ----
    # dec_R = sigma_R_s <= sigma_R_t  # boolean mask
    # alpha_R = torch.where(
    #     dec_R, (sigma_R_s**2) / (sigma_R_t**2 + eps), torch.ones_like(sigma_R_t)
    # )
    # var_R = torch.where(
    #     dec_R,
    #     (sigma_R_s**2 - (sigma_R_s**4)/(sigma_R_t**2 + eps)),
    #     (sigma_R_s**2 - sigma_R_t**2)
    # ).clamp_min(0.0)
    # std_R = torch.sqrt(var_R + 0.0).unsqueeze(-1)  # (B,1)

    # omega_t = log_so3(R_t)                         # (B,3)

    # if mode == "group":
    #     R_mean = exp_so3(alpha_R.unsqueeze(-1) * omega_t)
    #     if stochastic:
    #         d_omega = std_R * torch.randn_like(omega_t)
    #         R_s = exp_so3(d_omega) @ R_mean
    #     else:
    #         R_s = R_mean
    # elif mode == "algebra":
    #     if stochastic:
    #         omega_s = alpha_R.unsqueeze(-1)*omega_t + std_R*torch.randn_like(omega_t)
    #     else:
    #         omega_s = alpha_R.unsqueeze(-1)*omega_t
    #     R_s = exp_so3(omega_s)
    # else:
    #     raise ValueError("mode must be 'group' or 'algebra'.")

    # # ---- Translation step (exact in ℝ^3) ----
    # dec_T = sigma_T_s <= sigma_T_t
    # alpha_T = torch.where(
    #     dec_T, (sigma_T_s**2) / (sigma_T_t**2 + eps), torch.ones_like(sigma_T_t)
    # )
    # var_T = torch.where(
    #     dec_T,
    #     (sigma_T_s**2 - (sigma_T_s**4)/(sigma_T_t**2 + eps)),
    #     (sigma_T_s**2 - sigma_T_t**2)
    # ).clamp_min(0.0)
    # std_T = torch.sqrt(var_T + 0.0).unsqueeze(-1)  # (B,1)

    # if stochastic:
    #     T_s = alpha_T.unsqueeze(-1)*T_t + std_T*torch.randn_like(T_t)
    # else:
    #     T_s = alpha_T.unsqueeze(-1)*T_t

    inc_R = sigma_R_s >= sigma_R_t  # forward (add variance)
    var_R = (sigma_R_s**2 - sigma_R_t**2).clamp_min(0.0)
    std_R = torch.sqrt(var_R + 0.0).view(B, 1, 1)  # (B,1,1)
    if inc_R.all():
        # forward: Brownian increment
        d_omega = torch.randn(B, R_t.shape[1], 3, device=device, dtype=dtype) * std_R
        R_s = R_t @ exp_so3(d_omega)  # right-invariant
    elif (~inc_R).all():
        # decreasing noise: deterministic shrink in small-angle algebra
        omega_t = log_so3(R_t)  # (B,C,3)
        scale = ((sigma_R_s + eps) / (sigma_R_t + eps)).view(B, 1, 1)
        R_s = exp_so3(scale * omega_t)  # around identity
    else:
        raise ValueError("Mixed inc/dec batch not supported in this helper.")
    # ---- Translation step (ℝ^3 heat semigroup) ----
    inc_T = sigma_T_s >= sigma_T_t
    var_T = (sigma_T_s**2 - sigma_T_t**2).clamp_min(0.0)
    std_T = torch.sqrt(var_T + 0.0).view(B, 1, 1)  # (B,1,1)
    if inc_T.all():
        T_s = T_t + torch.randn_like(T_t) * std_T
    elif (~inc_T).all():
        scale_T = ((sigma_T_s + eps) / (sigma_T_t + eps)).view(B, 1, 1)
        T_s = scale_T * T_t
    else:
        raise ValueError("Mixed inc/dec batch not supported in this helper.")

    return R_s, T_s


def se3_heat_step_delta_sigma(
    R_t: Tensor,
    T_t: Tensor,
    sigma_R_t: Tensor,
    sigma_T_t: Tensor,  # current σ (B,) or scalar
    d_sigma_R: Tensor,
    d_sigma_T: Tensor,  # Δσ can be + or -
    stochastic: bool = False,
    mode: str = "group",
    sigma_floor: float = 0.0,  # e.g., 1e-6
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    device, dtype = R_t.device, R_t.dtype

    def _to(x):
        return torch.as_tensor(x, device=device, dtype=dtype).reshape(-1)

    sigma_R_t = _to(sigma_R_t)
    sigma_T_t = _to(sigma_T_t)
    d_sigma_R = _to(d_sigma_R)
    d_sigma_T = _to(d_sigma_T)

    B = R_t.shape[0]
    if sigma_R_t.numel() == 1:
        sigma_R_t = sigma_R_t.expand(B)
    if sigma_T_t.numel() == 1:
        sigma_T_t = sigma_T_t.expand(B)
    if d_sigma_R.numel() == 1:
        d_sigma_R = d_sigma_R.expand(B)
    if d_sigma_T.numel() == 1:
        d_sigma_T = d_sigma_T.expand(B)

    sigma_R_s = (sigma_R_t + d_sigma_R).clamp_min(sigma_floor)
    sigma_T_s = (sigma_T_t + d_sigma_T).clamp_min(sigma_floor)

    R_s, T_s = se3_heat_step_sigma(
        R_t,
        T_t,
        sigma_R_t,
        sigma_T_t,
        sigma_R_s,
        sigma_T_s,
        stochastic=stochastic,
        mode=mode,
    )
    return R_s, T_s


def SE3_oper(
    length: int,
    noise_level: float = 5.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate random SE(3) rotation matrices (length,3,3) and Gaussian noise (length,3)."""
    if rng is None:
        rng = np.random.default_rng()

    u1 = rng.random(size=length)
    u2 = rng.random(size=length)
    u3 = rng.random(size=length)

    r1 = np.sqrt(1 - u1)
    r2 = np.sqrt(u1)
    th1 = 2 * pi * u2
    th2 = 2 * pi * u3

    qw = r2 * np.cos(th2)
    qx = r1 * np.sin(th1)
    qy = r1 * np.cos(th1)
    qz = r2 * np.sin(th2)

    ww, xx, yy, zz = qw * qw, qx * qx, qy * qy, qz * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz

    R = np.empty((length, 3, 3), dtype=np.float32)

    R[:, 0, 0] = ww + xx - yy - zz
    R[:, 0, 1] = 2 * (xy - wz)
    R[:, 0, 2] = 2 * (xz + wy)
    R[:, 1, 0] = 2 * (xy + wz)
    R[:, 1, 1] = ww - xx + yy - zz
    R[:, 1, 2] = 2 * (yz - wx)
    R[:, 2, 0] = 2 * (xz - wy)
    R[:, 2, 1] = 2 * (yz + wx)
    R[:, 2, 2] = ww - xx - yy + zz

    noise = rng.standard_normal(size=(length, 3)).astype(np.float32) * noise_level

    return R, noise
