import time
import torch
import torch.nn.functional as F
import numpy as np

from pathlib import Path, PosixPath
from typing import TextIO
from dataclasses import is_dataclass

from team_gm.data import chemical


NM_TO_ANG_SCALE = 10.0
ANG_TO_NM_SCALE = 1 / NM_TO_ANG_SCALE


def to_numpy(x: torch.Tensor) -> np.ndarray:
    if x.dtype == torch.bfloat16:
        x = x.float()
    return x.detach().cpu().numpy()


def auto_tensor_collate(
    data_list: list[torch.Tensor], batch_dim: int | None = None
) -> torch.Tensor:
    """Automatic collate tensors with padding.
    Tensors will be padded to match the maximum size along their dimensions.
    Please note that all tensors in `data_list` must have the same number of dimensions.

    Parameters
    ----------
    data_list: list[torch.Tensor]
        A list of Tensor data.
    batch_dim: int or None, default = None
        If specified, the tensors will be concatenated along this dimension.
        If None, the tensors will be stacked into a new dimension.

    Returns
    -------
    padded_data: torch.Tensor
        Padded tensor with dimensions extended to each dimension's maximum size.
    """
    if not data_list:
        raise ValueError("data_list cannot be empty.")

    if not all(data.ndim == data_list[0].ndim for data in data_list):
        raise ValueError(
            "All tensors in data_list must have the same number of dimensions. "
            f"Got {[data.shape for data in data_list]}."
        )

    shapes = torch.tensor([data.shape for data in data_list])
    max_shape = shapes.max(dim=0).values

    padding_needed = max_shape - shapes
    if batch_dim is not None:
        padding_needed[:, batch_dim] = 0

    if not torch.all(padding_needed == 0):
        padded_list = []
        for i, data in enumerate(data_list):
            current_padding = padding_needed[i]
            padding_spec = [
                p for pad in reversed(current_padding.tolist()) for p in (0, pad)
            ]
            padded_list.append(F.pad(data, padding_spec, value=0))
    else:
        padded_list = data_list

    if batch_dim is None:
        return torch.stack(padded_list, dim=0)
    else:
        return torch.cat(padded_list, dim=batch_dim)


def move_data_to_device(data, device: str | torch.device):
    if not isinstance(device, torch.device):
        device = torch.device(device)

    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_data_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_data_to_device(v, device) for v in data]
    elif isinstance(data, tuple):
        return tuple(move_data_to_device(v, device) for v in data)
    elif is_dataclass(data):
        try:
            return data.to(device=device)
        except AttributeError as err:
            raise AttributeError(
                f"Cannot move `{type(data)}` to `{device}`. "
                "If you use custom dataclass, please implement a method to move it to a "
                "device (e.g., a .to(device) method)"
            ) from err
    else:
        return data


# TODO: make multi chain function
def write_pdb(
    atom_pos: torch.Tensor | np.ndarray,
    atom_mask: torch.Tensor | np.ndarray | None = None,
    atom_names: np.ndarray | None = None,
    b_factors: torch.Tensor | np.ndarray | None = None,
    seq_idx: torch.Tensor | np.ndarray | None = None,
    res_names: np.ndarray | None = None,
    file_path: str | PosixPath | None = None,
) -> PosixPath:
    """Write single chain atom positions as `.pdb` file.

    Parameters
    ----------
    atom_pos: FloatTensor or np.ndarray, [L, N, 3] or [M, L, N, 3]
        Atom xyz positions, shape [L, N, 3] (single model) or [M, L, N, 3]
        (multi-model).
    atom_mask: BoolTensor or np.ndarray, [L, N] or [M, L, N], (optional)
        Boolean mask for valid atoms. Defulat: inferred from non-zero
        positions.
    atom_names: np.ndarray, [L, N] or [M, L, N], (optional)
        Object type ndarray of atom names. Default: `C`(carbon) for all atoms.
    b_factors: FloatTensor or np.ndarray, [L, N] or [M, L, N], (optional)
        B factors of each atoms.
    seq_idx: LongTensor or np.ndarray, [L] or [M, L], (optional)
        Tensor of seqeunce index.
    res_names: np.ndarray, [L] or [M, L], (optional)
        Object type ndarray of residue names. Default: 'X' for all residues.
    file_path: str or PosixPath, (optional)
        Output `.pdb` file path. Default: current timestamp.

    Returns
    -------
    file_path: PosixPath
        Path to the saved `.pdb` file.
    """

    def _write_model(
        f: TextIO,
        atom_pos: np.ndarray,
        atom_mask: np.ndarray,
        atom_names: np.ndarray,
        b_factors: np.ndarray,
        seq_idx: np.ndarray,
        res_names: np.ndarray,
    ):
        """Write a single model."""
        for i, (res_idx, atom_idx) in enumerate(zip(*np.where(atom_mask))):
            x, y, z = atom_pos[res_idx, atom_idx]
            f.write(
                f"ATOM  {i + 1:5d} {atom_names[res_idx, atom_idx]:>4} "
                f"{res_names[res_idx]:>3} A{seq_idx[res_idx]:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}"
                f"{1:>6.2f}{b_factors[res_idx, atom_idx]:>6.2f}"
                f"{atom_names[res_idx, atom_idx][0]:>12}\n"
            )

    # convert tensors to numpy arrays
    if isinstance(atom_pos, torch.Tensor):
        atom_pos = to_numpy(atom_pos)
    if isinstance(atom_mask, torch.Tensor):
        atom_mask = to_numpy(atom_mask)
    if isinstance(b_factors, torch.Tensor):
        b_factors = to_numpy(b_factors)
    if isinstance(seq_idx, torch.Tensor):
        seq_idx = to_numpy(seq_idx)

    # validate inputs
    if atom_mask is None:
        atom_mask = ~np.isnan(atom_pos).any(-1)
    if atom_names is None:
        atom_names = np.full(atom_mask.shape, "C", dtype="U")
    if b_factors is None:
        b_factors = np.zeros(atom_mask.shape)
    if seq_idx is None:
        seq_idx = np.empty(atom_mask.shape[:-1], dtype=np.int64)
        seq_idx[..., :] = np.arange(atom_mask.shape[-2]) + 1
    if res_names is None:
        res_names = np.full(atom_mask.shape[:-1], "X", dtype="U")
    if file_path is None:
        file_path = str(int(time.time())) + ".pdb"

    # validate dimensions
    assert atom_pos.ndim in [3, 4]
    assert atom_pos.shape[-1] == 3
    if atom_pos.ndim == 4:
        N = atom_pos.shape[0]
        expand_model = lambda a: np.tile(a, (N,) + (1,) * a.ndim)
        if atom_mask.ndim == 2:
            atom_mask = expand_model(atom_mask)
        if atom_names.ndim == 2:
            atom_names = expand_model(atom_names)
        if b_factors.ndim == 2:
            b_factors = expand_model(b_factors)
        if res_names.ndim == 1:
            res_names = expand_model(res_names)
        if seq_idx.ndim == 1:
            seq_idx = expand_model(seq_idx)
    assert atom_mask.shape == atom_pos.shape[:-1]
    assert atom_names.shape == atom_pos.shape[:-1]
    assert b_factors.shape == atom_pos.shape[:-1]
    assert res_names.shape == atom_pos.shape[:-2]
    assert seq_idx.shape == atom_pos.shape[:-2]
    atom_mask[atom_names is None] = False
    atom_mask[np.isnan(atom_pos).any(-1)] = False

    file_path = Path(file_path)
    with file_path.open("w") as f:
        # write multi model
        if atom_pos.ndim == 4:
            for model_idx in range(len(atom_pos)):
                f.write(f"MODEL     {model_idx + 1:4d}\n")
                _write_model(
                    f,
                    atom_pos[model_idx],
                    atom_mask[model_idx],
                    atom_names[model_idx],
                    b_factors[model_idx],
                    seq_idx[model_idx],
                    res_names[model_idx],
                )
                f.write("ENDMDL\n")
        # write single model
        elif atom_pos.ndim == 3:
            _write_model(
                f, atom_pos, atom_mask, atom_names, b_factors, seq_idx, res_names
            )
        f.write("END")

    return file_path


def write_na_pdb(
    atom_pos: torch.Tensor | np.ndarray,
    atom_mask: torch.Tensor | np.ndarray | None = None,
    b_factors: torch.Tensor | np.ndarray | None = None,
    seq_idx: torch.Tensor | np.ndarray | None = None,
    res_names: np.ndarray | None = None,
    file_path: str | PosixPath | None = None,
    backbone_only: bool = False,
) -> PosixPath:
    """Write single chain nucleic acid atom positions as `.pdb` file.

    Parameters
    ----------
    atom_pos: FloatTensor or np.ndarray, [L, N, 3] or [M, L, N, 3]
        Atom xyz positions, shape [L, N, 3] (single model) or [M, L, N, 3]
        (multi-model).
    atom_mask: BoolTensor or np.ndarray, [L, N] or [M, L, N], (optional)
        Boolean mask for valid atoms. Defulat: inferred from non-zero
        positions.
    b_factors: FloatTensor or np.ndarray, [L, N] or [M, L, N], (optional)
        B factors of each atoms.
    seq_idx: LongTensor or np.ndarray, [L] or [M, L], (optional)
        Tensor of seqeunce index.
    res_names: np.ndarray, [L] or [M, L], (optional)
        Array of residue order type.
    file_path: str or PosixPath, (optional)
        Output `.pdb` file path. Default: current timestamp.
    backbone_only: bool, default = False
        If `True`, only backbone atom were written.

    Returns
    -------
    file_path: PosixPath
        Path to the saved `.pdb` file.
    """
    if backbone_only:
        res_names = np.full(atom_pos.shape[:-2], chemical.RNA_UNK)
    elif res_names is None:
        return write_pdb(atom_pos, atom_mask, b_factors=b_factors, file_path=file_path)

    if not np.isin(res_names, chemical.RNA_RES_NAMES + chemical.DNA_RES_NAMES).all():
        condition = ~np.isin(res_names, chemical.RNA_RES_NAMES + chemical.DNA_RES_NAMES)
        raise ValueError(res_names[condition])

    mapping = lambda x: chemical.RES_NAME_TO_ATOM_NAMES[x]
    atom_names = np.vectorize(mapping, otypes=["O"])(res_names)
    atom_names = np.array(atom_names.tolist())

    if atom_mask is None:
        mapping = lambda x: False if x is None else True
        atom_mask = np.vectorize(mapping)(atom_names)

    return write_pdb(
        atom_pos, atom_mask, atom_names, b_factors, seq_idx, res_names, file_path
    )


def write_prot_pdb(
    atom_pos: torch.Tensor | np.ndarray,
    atom_mask: torch.Tensor | np.ndarray | None = None,
    b_factors: torch.Tensor | np.ndarray | None = None,
    seq_idx: torch.Tensor | np.ndarray | None = None,
    res_names: np.ndarray | None = None,
    file_path: str | PosixPath | None = None,
    backbone_only: bool = False,
) -> PosixPath:
    """Write single chain protein atom positions as `.pdb` file.

    Parameters
    ----------
    atom_pos: FloatTensor or np.ndarray, [L, N, 3] or [M, L, N, 3]
        Atom xyz positions, shape [L, N, 3] (single model) or [M, L, N, 3]
        (multi-model).
    atom_mask: BoolTensor or np.ndarray, [L, N] or [M, L, N], (optional)
        Boolean mask for valid atoms. Defulat: inferred from non-zero
        positions.
    b_factors: FloatTensor or np.ndarray, [L, N] or [M, L, N], (optional)
        B factors of each atoms.
    seq_idx: LongTensor or np.ndarray, [L] or [M, L], (optional)
        Tensor of seqeunce index.
    res_names: np.ndarray, [L] or [M, L], (optional)
        Array of residue order type.
    file_path: str or PosixPath, (optional)
        Output `.pdb` file path. Default: current timestamp.
    backbone_only: bool, default = False
        If `True`, only backbone atom were written.

    Returns
    -------
    file_path: PosixPath
        Path to the saved `.pdb` file.
    """
    if backbone_only:
        res_names = np.full(atom_pos.shape[:-2], chemical.PROT_UNK)
    elif res_names is None:
        return write_pdb(atom_pos, atom_mask, b_factors=b_factors, file_path=file_path)

    if not np.isin(res_names, chemical.PROT_RES_NAMES).all():
        condition = ~np.isin(res_names, chemical.PROT_RES_NAMES)
        raise ValueError(res_names[condition])

    mapping = lambda x: chemical.RES_NAME_TO_ATOM_NAMES[x]
    atom_names = np.vectorize(mapping, otypes=["O"])(res_names)
    atom_names = np.array(atom_names.tolist())

    if atom_mask is None:
        mapping = lambda x: False if x is None else True
        atom_mask = np.vectorize(mapping)(atom_names)

    return write_pdb(
        atom_pos, atom_mask, atom_names, b_factors, seq_idx, res_names, file_path
    )
