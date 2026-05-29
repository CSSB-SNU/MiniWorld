"""Build warm-start ``init_x0`` tensors for the decoupled-diffusion solver.

The solver's standard init is ``y = randn * sigma_y_max`` (full noise).
Two warm-start modes plug into the same ``init_x0`` / ``start_sigma_y``
solver hook (see :mod:`miniworld.diffusion.decoupled_xpred.solver`):

* **flexible-docking** — :func:`build_flexible_docking_init_x0`. One CIF
  per combine-group provides each group's known internal coords. The
  default ``start_sigma_y`` is the scheduler's phase-1 boundary, so the
  first solver step samples max R/T per group and the *inter-group* pose
  is randomized while *intra-group* structure is preserved.
* **refinement** — :func:`build_refinement_init_x0`. A single CIF covers
  every query chain (rough full structure). ``start_sigma_y`` is small
  (deep phase 2), so the per-step R/T noise is tiny and the input's
  inter-chain geometry is essentially preserved while atom coords are
  cleaned up.

Atoms are written to the global ``(N_atom, 3)`` tensor in the exact
order produced by :func:`build_inference_batch`: chains in
``sorted(spec.chain_indices())`` order, residues in fasta order, atoms in
CCD canonical order with terminal OXT/OP3 stripped on polymer residues.

Hard rules — any violation raises immediately:
  * Per-mode validation in :mod:`.spec` enforces full chain coverage.
  * Each mapped CIF chain must have the same residue count as the
    corresponding query chain, identical 3-letter codes by position,
    and every required atom name in each residue.
"""

from __future__ import annotations

import gzip
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from miniworld.data.features import Batch

    from .spec import InferenceSpec


@dataclass
class _CifResidue:
    """A single standard residue parsed from a warm-start CIF."""

    chem_comp_id: str  # 3-letter, uppercased
    atoms: dict[str, np.ndarray]  # atom_name -> (3,) float32 xyz


def _open_cif(cif_path: Path) -> tuple[str, Path | None]:
    """Return ``(path_to_use, tmp_to_delete)`` for ``Bio.PDB.MMCIFParser``."""
    if cif_path.suffix.lower() == ".gz":
        with gzip.open(cif_path, "rb") as src:
            data = src.read()
        tmp = Path(tempfile.mkstemp(suffix=".cif")[1])
        tmp.write_bytes(data)
        return str(tmp), tmp
    return str(cif_path), None


def _load_cif_chains(cif_path: Path) -> dict[str, list[_CifResidue]]:
    """Parse a CIF and return ``{label_asym_id: [_CifResidue, ...]}``.

    Heteroatoms (residues whose hetflag != ' ') are skipped — same
    convention as ``complex_template._load_chain_backbone_from_cif``.
    ``auth_chains=False`` makes ``chain.id`` expose ``label_asym_id``.
    """
    from Bio.PDB.MMCIFParser import MMCIFParser

    parser = MMCIFParser(QUIET=True, auth_chains=False)
    parse_path, tmp_path = _open_cif(cif_path)
    try:
        structure = parser.get_structure("warmstart", parse_path)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
    model = next(iter(structure))

    chains: dict[str, list[_CifResidue]] = {}
    for chain in model:
        residues: list[_CifResidue] = []
        for res in chain:
            if res.id[0] != " ":  # skip heteroatoms / waters
                continue
            atoms: dict[str, np.ndarray] = {}
            for atom in res:
                atoms[atom.get_name()] = atom.get_coord().astype(np.float32)
            residues.append(_CifResidue(
                chem_comp_id=res.resname.upper(),
                atoms=atoms,
            ))
        if residues:
            chains[chain.id] = residues
    return chains


@dataclass
class _BatchAtomLayout:
    """Pre-computed views into ``Batch`` used by the per-chain filler."""

    atom_names: np.ndarray             # (N_atom,) str
    chem_comp_ids: np.ndarray          # (N_res_global,) str
    atom_chain_local: np.ndarray       # (N_atom,) chain_local_idx
    atom_residue: np.ndarray           # (N_atom,) global residue idx
    chain_idx_to_local: dict[int, int]
    n_atom: int


def _read_batch_layout(spec: "InferenceSpec", batch: "Batch") -> _BatchAtomLayout:
    atom_names = batch.atom_ids[0]
    chem_comp_ids = batch.chem_comp_ids[0]
    token_asym_id = batch.scheme.token_asym_id[0].cpu().numpy()
    atom_to_token = batch.scheme.atom_to_token_idx_map[0].cpu().numpy()
    atom_chain_local = token_asym_id[atom_to_token]
    # Globally-unique atom -> residue lookup. ``reference.space_uid`` is the
    # canonical atom->global-residue map produced by
    # :func:`build_inference_batch`; the older ``token_residue_idx`` path
    # used to work as a stand-in here because the inference build wrote
    # globally-offset values into it, but ``token_residue_idx`` now mirrors
    # the dataloader's per-chain ``cif_idx`` semantics (resets per chain,
    # may be non-contiguous under spatial-crop overrides), so it would
    # collide across chains.
    atom_residue = batch.reference.space_uid[0].cpu().numpy()
    chain_indices = spec.chain_indices()
    chain_idx_to_local = {ci: li for li, ci in enumerate(chain_indices)}
    return _BatchAtomLayout(
        atom_names=atom_names,
        chem_comp_ids=chem_comp_ids,
        atom_chain_local=atom_chain_local,
        atom_residue=atom_residue,
        chain_idx_to_local=chain_idx_to_local,
        n_atom=int(atom_names.shape[0]),
    )


def _fill_chain_init_x0(
    init_x0: torch.Tensor,
    layout: _BatchAtomLayout,
    cif_path: Path,
    cif_chain_id: str,
    cif_residues: list[_CifResidue],
    query_chain_idx: int,
    *,
    where: str,
) -> None:
    """Fill ``init_x0`` rows for one (query chain, CIF chain) pair in place.

    ``where`` is a label like ``"flexible_docking.groups[0]"`` or
    ``"refinement"`` prefixed onto every error message so the user can
    locate the offending YAML entry.
    """
    chain_local = layout.chain_idx_to_local[query_chain_idx]
    atom_indices_for_chain = np.where(layout.atom_chain_local == chain_local)[0]
    if atom_indices_for_chain.size == 0:
        msg = (
            f"{where}: query chain index {query_chain_idx} has no atoms in "
            f"the batch."
        )
        raise ValueError(msg)

    residue_to_atom_idxs: dict[int, list[int]] = {}
    for a in atom_indices_for_chain:
        residue_to_atom_idxs.setdefault(
            int(layout.atom_residue[a]), [],
        ).append(int(a))
    query_residues = list(residue_to_atom_idxs.keys())

    if len(query_residues) != len(cif_residues):
        msg = (
            f"{where}: query chain {query_chain_idx} has "
            f"{len(query_residues)} residues but CIF chain {cif_chain_id} "
            f"in {cif_path} has {len(cif_residues)}. They must match "
            f"exactly."
        )
        raise ValueError(msg)

    for pos, (q_res_idx, cif_res) in enumerate(
        zip(query_residues, cif_residues, strict=True),
    ):
        expected_chem = str(layout.chem_comp_ids[q_res_idx])
        if cif_res.chem_comp_id != expected_chem:
            msg = (
                f"{where}: residue mismatch on query chain "
                f"{query_chain_idx} at position {pos}: query residue "
                f"{expected_chem} vs CIF chain {cif_chain_id} residue "
                f"{cif_res.chem_comp_id}."
            )
            raise ValueError(msg)
        for a in residue_to_atom_idxs[q_res_idx]:
            name = str(layout.atom_names[a])
            xyz = cif_res.atoms.get(name)
            if xyz is None:
                msg = (
                    f"{where}: atom {name!r} of residue {expected_chem} "
                    f"(query chain {query_chain_idx}, global residue "
                    f"{q_res_idx}) is missing in CIF {cif_path} chain "
                    f"{cif_chain_id}. Available atoms in this residue: "
                    f"{sorted(cif_res.atoms)}"
                )
                raise KeyError(msg)
            init_x0[a] = torch.from_numpy(xyz)


def _center_per_diffusion_group(
    init_x0: torch.Tensor,
    spec: "InferenceSpec",
    batch: "Batch",
) -> None:
    """Subtract each combine-group's centroid in place.

    ``scheme.atom_to_chain_id`` already holds the post-remap combine-group
    id (see build.py:218-220) — i.e. the same group index used by the
    solver's per-step ``apply_chain_rt``. Centering decouples the
    rotation (applied around the origin) from the translation noise so
    they act as orthogonal degrees of freedom.
    """
    atom_to_group = batch.scheme.atom_to_chain_id[0].cpu().numpy()
    n_groups = int(atom_to_group.max()) + 1 if atom_to_group.size else 0
    for group_idx in range(n_groups):
        mask_np = atom_to_group == group_idx
        if not mask_np.any():
            continue
        mask = torch.from_numpy(mask_np)
        centroid = init_x0[mask].mean(dim=0)
        init_x0[mask] = init_x0[mask] - centroid


def _assert_no_nan(init_x0: torch.Tensor, where: str) -> None:
    if init_x0.isnan().any():
        unfilled = init_x0.isnan().any(dim=-1).nonzero(as_tuple=True)[0]
        msg = (
            f"{where}: {int(unfilled.numel())} atoms still have NaN init "
            f"coords; first indices: {unfilled[:10].tolist()}."
        )
        raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------


def build_flexible_docking_init_x0(
    spec: "InferenceSpec",
    batch: "Batch",
) -> torch.Tensor:
    """Build the ``(N_atom, 3)`` warm-start tensor for flexible-docking.

    Assumes ``spec.flexible_docking`` is set and has already passed the
    spec-level validation (1:1 with ``diffusion_groups``, full chain
    coverage). Reads the atom layout directly from ``batch`` so it stays
    consistent with whatever :func:`build_inference_batch` produced.
    """
    fd = spec.flexible_docking
    if fd is None:
        msg = "spec.flexible_docking is None — nothing to build."
        raise ValueError(msg)

    layout = _read_batch_layout(spec, batch)
    init_x0 = torch.full((layout.n_atom, 3), float("nan"), dtype=torch.float32)

    for group_idx, (group_chains, group_spec) in enumerate(
        zip(spec.diffusion_groups, fd.groups, strict=True),
    ):
        where = f"flexible_docking.groups[{group_idx}]"
        cif_path = Path(group_spec.cif)
        if not cif_path.is_file():
            msg = f"{where}.cif not found: {cif_path}"
            raise FileNotFoundError(msg)
        cif_chains = _load_cif_chains(cif_path)

        for query_chain_idx in group_chains:
            cif_chain_id = group_spec.chain_map[str(query_chain_idx)]
            if cif_chain_id not in cif_chains:
                msg = (
                    f"{where}: CIF {cif_path} has no chain "
                    f"{cif_chain_id!r}. Available: {sorted(cif_chains)}"
                )
                raise KeyError(msg)
            _fill_chain_init_x0(
                init_x0=init_x0,
                layout=layout,
                cif_path=cif_path,
                cif_chain_id=cif_chain_id,
                cif_residues=cif_chains[cif_chain_id],
                query_chain_idx=query_chain_idx,
                where=where,
            )

    _assert_no_nan(init_x0, "flexible_docking")

    if fd.center_groups:
        _center_per_diffusion_group(init_x0, spec, batch)

    return init_x0


def build_refinement_init_x0(
    spec: "InferenceSpec",
    batch: "Batch",
) -> torch.Tensor:
    """Build the ``(N_atom, 3)`` warm-start tensor for refinement.

    Single CIF holds the rough full structure; every query chain is
    mapped to one CIF chain via ``spec.refinement.chain_map``. Spec-level
    validation already enforced full coverage.

    ``diffusion_groups`` is not constrained here — when set, the solver
    will use it for per-group ``apply_chain_rt`` at the (small)
    ``start_sigma_y``; otherwise each chain is its own singleton group
    (build.py's default). ``center_groups`` is honored but defaults to
    ``False`` for refinement so the input's inter-chain geometry is
    preserved.
    """
    rs = spec.refinement
    if rs is None:
        msg = "spec.refinement is None — nothing to build."
        raise ValueError(msg)

    layout = _read_batch_layout(spec, batch)
    init_x0 = torch.full((layout.n_atom, 3), float("nan"), dtype=torch.float32)

    cif_path = Path(rs.cif)
    if not cif_path.is_file():
        msg = f"refinement.cif not found: {cif_path}"
        raise FileNotFoundError(msg)
    cif_chains = _load_cif_chains(cif_path)

    for query_chain_idx_str, cif_chain_id in rs.chain_map.items():
        query_chain_idx = int(query_chain_idx_str)
        if cif_chain_id not in cif_chains:
            msg = (
                f"refinement: CIF {cif_path} has no chain "
                f"{cif_chain_id!r}. Available: {sorted(cif_chains)}"
            )
            raise KeyError(msg)
        _fill_chain_init_x0(
            init_x0=init_x0,
            layout=layout,
            cif_path=cif_path,
            cif_chain_id=cif_chain_id,
            cif_residues=cif_chains[cif_chain_id],
            query_chain_idx=query_chain_idx,
            where="refinement",
        )

    _assert_no_nan(init_x0, "refinement")

    if rs.center_groups:
        _center_per_diffusion_group(init_x0, spec, batch)

    return init_x0
