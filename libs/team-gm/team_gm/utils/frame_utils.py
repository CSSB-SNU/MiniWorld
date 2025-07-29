import torch

from scipy.spatial.transform import Rotation as R
from jaxtyping import Float, Int, Bool

from team_gm import typecheck
from .rigid_utils import Rotation, Rigid


@typecheck
def create_rigid(
    rot_mats: Float[torch.Tensor, "* 3 3"],
    trans: Float[torch.Tensor, "* 3"],
) -> Rigid:
    rots = Rotation(rot_mats=rot_mats)
    return Rigid(rots=rots, trans=trans)


def rigid_to_device(rigid: Rigid, device: torch.device | str) -> Rigid:
    return create_rigid(
        rigid.get_rots().get_rot_mats().to(device=device),
        rigid.get_trans().to(device=device),
    )


@typecheck
def local_to_global(
    rigid: Rigid,
    local_atom_pos: Float[torch.Tensor, "* L N 3"],
) -> Float[torch.Tensor, "* L N 3"]:
    """Transform local atoms to global with rigids transformation.

    Parameters
    ----------
    rigid: Rigid, (..., L)
        Global transformation rigid.
    local_atom_pos: torch.Tensor, (..., L, N, 3)
        Local atom coordinates.

    Returns
    -------
    global_atom_pos: torch.Tensor, (..., L, N, 3)
        Tensor of transformed global atoms position.
    """
    if local_atom_pos.dtype == rigid.get_trans().dtype:
        return rigid[..., None].apply(local_atom_pos)
    return (
        rigid[..., None]
        .apply(local_atom_pos.to(rigid.get_trans().dtype))
        .to(dtype=local_atom_pos.dtype)
    )


@typecheck
def global_to_local(
    rigid: Rigid,
    global_atom_pos: Float[torch.Tensor, "* L N 3"],
) -> Float[torch.Tensor, "* L N 3"]:
    """Transform global atoms to local with rigids transformation.

    Parameters
    ----------
    rigid: Rigid, (..., L)
        Global transformation rigid.
    global_atom_pos: torch.Tensor, (..., L, N, 3)
        Global atom coordinates.

    Returns
    -------
    local_atom_pos: torch.Tensor, (..., L, N, 3)
        Tensor of transformed local atoms position.
    """
    if global_atom_pos.dtype == rigid.get_trans().dtype:
        return rigid[..., None].invert_apply(global_atom_pos)
    return (
        rigid[..., None]
        .invert_apply(global_atom_pos.to(rigid.get_trans().dtype))
        .to(dtype=global_atom_pos.dtype)
    )


@typecheck
def cal_unit_vector(
    rigids: Rigid,  # (..., L)
    eps=1e-20,
) -> Float[torch.Tensor, "... L L 3"]:
    points = rigids.get_trans()[..., None, :, :]
    rigid_vec = rigids[..., None].invert_apply(points)

    inv_distance_scalar = torch.rsqrt(eps + torch.sum(rigid_vec**2, dim=-1))
    unit_vector = rigid_vec * inv_distance_scalar[..., None]
    unit_vector = torch.unbind(unit_vector[..., None, :], dim=-1)
    unit_vector = torch.cat(unit_vector, dim=-1)
    return unit_vector


@typecheck
def cal_distogram(
    positions: Float[torch.Tensor, "* L 3"],
    min_bin: float = 1e-3,
    max_bin: float = 20.0,
    num_bins: int = 22,
) -> Int[torch.Tensor, "* L L num_bins"]:
    """Calculate frame distogram.

    Parameters
    ----------
    positions: FloatTensor, [B, L, 3]
        Tensor of each frame positions.
    min_bin: float, default = 1e-3
        Minimum distance for get bin.
    max_bin: flaot, default = 20.0
        Maximum distance for get bin.
    num_bins: int, default = 22
        Number of bins for distogram.

    Returns
    -------
    dgram: LongTensor, [B, L, L, num_binds]
        Distogram for frame positions.
    """
    device = positions.device

    dists_2d = torch.linalg.norm(
        positions.unsqueeze(-2) - positions.unsqueeze(-3), axis=-1
    )[..., None]
    lower = torch.linspace(min_bin, max_bin, num_bins).to(device)
    upper = torch.cat([lower[1:], lower.new_tensor([1e8])], dim=-1)
    dgram = ((dists_2d > lower) * (dists_2d < upper)).long()
    return dgram


@typecheck
def atom_to_frame(
    p_x_axis: Float[torch.Tensor, "* L 3"],
    origin: Float[torch.Tensor, "* L 3"],
    p_xy_plane: Float[torch.Tensor, "* L 3"],
) -> Rigid:  # (..., L)
    """Transforms all atom to frames.

    Parameters
    ----------
    p_x_axis: FloatTensor, (..., L, 3)
        Negative x-axis of each frame.
    origin: FloatTensor, (..., L, 3)
        Origin of each frame.
    p_xy_plane: FloatTensor, (..., L, 3)
        Point on the xy-plane of each frame.

    Returns
    -------
    gt_frames: Rigid, (..., L)
        Rigid tensor of each frame.
    """
    rigid = Rigid.from_3_points(
        p_neg_x_axis=p_x_axis,
        origin=origin,
        p_xy_plane=p_xy_plane,
    )

    # Because the Rigid.from_3_points function assumes the first point is the negative
    # x-axis, we need to flip the x and z axes of the rotation matrix.
    rots = torch.eye(3)
    rots[0, 0] = -1
    rots[2, 2] = -1
    rots = Rotation(rot_mats=rots)
    rigid = rigid.compose(Rigid(rots, None))
    return rigid


@typecheck
def centered_gaussian(
    res_mask: Bool[torch.Tensor, "* L"],
) -> Float[torch.Tensor, "* L 3"]:
    """Generate centered 3d Gaussian noise with mask.
    Noise will be centered at the origin except for the residues
    that are masked out and will be filled with NaN.

    Parameters
    ----------
    res_mask : BoolTensor, (..., L)
        Residue mask tensor.

    Returns
    -------
    noise: FloatTensor, (..., L, 3)
        Centered 3d Gaussian noise tensor.
    """
    noise = torch.randn(*res_mask.shape, 3, device=res_mask.device)
    noise[~res_mask] = float("NaN")  # TODO: make without NaN
    noise = noise - torch.nanmean(noise, dim=-2, keepdim=True)
    noise = noise.masked_fill(~res_mask[..., None], 0)
    return noise


@typecheck
def uniform_so3(
    res_mask: Bool[torch.Tensor, "* L"],
) -> Float[torch.Tensor, "* L 3 3"]:
    """Generate random SO(3) rotation matrices with mask.
    Rotation matrices will be generated randomly except for the residues
    that are masked out and will be filled with NaN.

    Parameters
    ----------
    res_mask : BoolTensor, (..., L)
        Residue mask tensor.

    Returns
    -------
    rot_mats: torch.Tensor, (..., L, 3, 3)
        Random SO(3) rotation matrices
    """
    rot_mats = R.random(res_mask.numel()).as_matrix()
    rot_mats = torch.Tensor(rot_mats).to(res_mask.device)
    rot_mats = rot_mats.reshape(*res_mask.shape, 3, 3)
    rot_mats[~res_mask] = torch.eye(3).to(res_mask.device)
    return rot_mats
