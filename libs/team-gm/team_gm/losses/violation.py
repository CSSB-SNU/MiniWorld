import torch

from jaxtyping import Float, Bool, Int

from team_gm import typecheck
from team_gm.data import chemical


@typecheck
def cal_clash_loss(
    atom_pos: Float[torch.Tensor, "* L N 3"],
    atom_mask: Bool[torch.Tensor, "* L N"],
    seq_idx: Int[torch.Tensor, "* L"],
    restype: Int[torch.Tensor, "* L"],
    loss_mask: Bool[torch.Tensor, "* L"],
    asym_id: Int[torch.Tensor, "* L"] | None = None,
    tolerance: float = 1.8,
    debug: bool = False,
) -> Float[torch.Tensor, "*"]:
    """Calculate all atom structure clash loss faster

    Parameters (N = 23으로 가정)
    ----------
    atom_pos: FloatTensor, [B, L, N, 3]
        Tensor of atom positions.
    atom_mask: BoolTensor, [B, L, N]
        Tensor of atom mask.
    seq_idx: LongTensor, [B, L]
        Tensor of seqeunce index.
    restype: LongTensor, [B, L]
        Tensor of one-hot vector residue type.
    loss_mask: BoolTensor, [B, L]
        Tensor of masking certain residue loss.
    asym_id: LongTensor, [B, L] (optional)
        Tensor of asymmetric unit ID for each residue.
    tolerance: float
        Tolerance about how much long distance is allowed between atom
        pair distance and atom pair van der Waals radius.
    """
    if atom_pos.ndim != 4:
        raise ValueError(
            f"atom_pos should be 4D tensor, but got {atom_pos.ndim}D tensor."
        )
    _, L, N = atom_mask.shape
    device = atom_mask.device
    restype = restype.to("cpu")

    loss_mask_2d = loss_mask[:, None] * loss_mask[:, :, None]
    clash_atom_mask = torch.tril(loss_mask_2d)
    clash_atom_mask = clash_atom_mask[..., None, None]
    clash_atom_mask = clash_atom_mask * (
        atom_mask[:, :, None, :, None] * atom_mask[:, None, :, None, :]
    )

    # Determine if residues are nucleic acids (NA) or proteins (PROT)
    is_NA = (restype[-1] if restype.ndim == 2 else restype) > 21
    is_PROT = ~is_NA

    # Create one-hot encodings for specific atom types
    def create_one_hot(index, num_classes=23):
        one_hot = torch.nn.functional.one_hot(
            seq_idx.new_tensor(index), num_classes=num_classes
        )
        return one_hot.reshape(*((1,) * len(seq_idx.shape[:-1])), *one_hot.shape)

    c_one_hot = create_one_hot(2)  # C atom
    n_one_hot = create_one_hot(0)  # N atom
    p_one_hot = create_one_hot(1)  # P atom
    o3_one_hot = create_one_hot(8)  # O3' atom

    # Create neighbor mask for adjacent residues
    neighbour_mask = (seq_idx[..., :, None] + 1) == seq_idx[..., None, :]
    if asym_id is not None:
        neighbour_mask &= asym_id[..., :, None] == asym_id[..., None, :]
    neighbour_mask = neighbour_mask[..., None, None]

    # Define C-N bonds for proteins and P-O3' bonds for nucleic acids
    c_n_bonds = (
        neighbour_mask
        * c_one_hot[..., None, None, :, None]
        * n_one_hot[..., None, None, None, :]
    )
    p_o3_bonds = (
        neighbour_mask
        * p_one_hot[..., None, None, :, None]
        * o3_one_hot[..., None, None, None, :]
    )

    # Apply masks for proteins and nucleic acids
    Prot_2d = is_PROT[None, :, None, None, None] * is_PROT[None, None, :, None, None]
    NA_2d = is_NA[None, :, None, None, None] * is_NA[None, None, :, None, None]

    c_n_bonds *= Prot_2d
    p_o3_bonds *= NA_2d

    # Exclude C-N and P-O3' bonds from clash atom mask
    clash_atom_mask *= (1.0 - c_n_bonds) * (1.0 - p_o3_bonds)

    # Exclude bond in same residue
    self_bond_mask = chemical.RES_TYPE_BOND_MATRIX[restype]
    batch_idx, res_idx, row_idx, col_idx = torch.where(self_bond_mask)
    clash_atom_mask[batch_idx, res_idx, res_idx, row_idx, col_idx] = False

    # Exclude bond with self atom
    self_atom_mask = torch.tril(torch.ones(L, N, N), -1).bool()
    res_idx, row_idx, col_idx = torch.where(~self_atom_mask)
    clash_atom_mask[:, res_idx, res_idx, row_idx, col_idx] = False

    # Make lower distance between to atoms of near by residue
    atom_radius = chemical.RES_TYPE_ATOM_RADIUS[restype][:, :, :23].to(device)
    dists_lower_bound = clash_atom_mask * (
        atom_radius[..., :, None, :, None] + atom_radius[..., None, :, None, :]
    )

    pair_dists = atom_pos[..., None, :, None, :] - atom_pos[..., None, :, None, :, :]
    pair_dists = torch.norm(pair_dists, dim=-1)

    clash_atom_loss = torch.relu((dists_lower_bound - pair_dists) - tolerance)
    clash_atom_loss *= clash_atom_mask
    if debug:
        return clash_atom_loss
    clash_atom_loss = clash_atom_loss.sum(dim=(-1, -2, -3, -4))
    clash_atom_loss /= clash_atom_mask.sum(dim=(-1, -2, -3, -4))
    return clash_atom_loss
