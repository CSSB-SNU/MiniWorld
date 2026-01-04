from pathlib import Path

import numpy as np
import torch

from miniworld.data.features.features_biomol import Batch
from miniworld.data.mapping import AtomMapping


def batch_to_cif(  # noqa: PLR0915
    batch: Batch,
    atom_pos_pred: torch.Tensor | None,
    save_path: Path,
) -> None:
    """Convert a batch to CIF format string."""

    def _to_mmcif_format(array: torch.Tensor | list) -> list:
        array = array.cpu().numpy() if isinstance(array, torch.Tensor) else array
        _list = [str(item) for item in array]
        max_length = max([len(item) for item in _list])
        return [item.ljust(max_length) for item in _list]

    output = f"#\ndata_{batch.name[0]}\n"

    header = [
        "_atom_site.group_PDB",
        "_atom_site.id",
        "_atom_site.type_symbol",
        "_atom_site.label_atom_id",
        "_atom_site.label_alt_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_asym_id",
        "_atom_site.label_entity_id",
        "_atom_site.label_seq_id",
        "_atom_site.pdbx_PDB_ins_code",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.occupancy",
        "_atom_site.B_iso_or_equiv",
        "_atom_site.pdbx_formal_charge",
        "_atom_site.auth_seq_id",
        "_atom_site.auth_comp_id",
        "_atom_site.auth_asym_id",
        "_atom_site.auth_atom_id",
        "_atom_site.pdbx_PDB_model_num",
    ]
    output += "#\n"
    output += "loop_\n"
    output += "\n".join(header) + "\n"
    atom_mapping = AtomMapping()

    xyz = (
        atom_pos_pred if atom_pos_pred is not None else batch.structure.atom_pos
    )  # if no prediction, use input coords
    xyz = xyz[0]
    mask = batch.structure.atom_pos_mask[0].bool()
    length = mask.sum().item()
    atom_to_res = batch.scheme.atom_to_residue_idx_map[0]
    res_to_asym = batch.scheme.residue_asym_id
    res_to_entity = batch.scheme.residue_entity_id
    atom_to_asym = res_to_asym[0, atom_to_res]
    atom_to_entity = res_to_entity[0, atom_to_res]

    hetero = torch.tensor(batch.heteros[0].value).to(device=mask.device)
    atom_ids = batch.atom_ids[0]
    chem_comp_ids = batch.chem_comp_ids[0]
    label_atom_id_list = np.array(atom_ids)[mask.cpu()]
    label_comp_id_list = np.array(chem_comp_ids)[atom_to_res.cpu()][mask.cpu()]
    group_PDB_list = hetero[atom_to_res][mask]
    group_PDB_list = ["HETATM" if g == 1 else "ATOM" for g in group_PDB_list]
    id_list = 1 + np.arange(length)
    type_symbol_list = atom_mapping.index_to_atom(
        batch.reference.element[0].cpu().numpy(),
    )[mask.cpu()]
    label_alt_id_list = ["."] * length
    label_asym_id_list = atom_to_asym[mask]
    label_entity_id_list = atom_to_entity[mask]
    label_seq_id_list = batch.scheme.residue_idx[0, atom_to_res][mask]
    auth_idx_list = label_seq_id_list  # assuming auth seq id == label seq id
    auth_seq_id_list = label_seq_id_list  # assuming auth seq id == label seq id
    ins_code_list = ["?"] * length

    cartn_x_list = xyz[mask, 0]
    cartn_y_list = xyz[mask, 1]
    cartn_z_list = xyz[mask, 2]
    occupancy_list = [1.0] * length
    b_iso_or_equiv_list = [100.0] * length  # TODO replace with plddt.
    pdbx_formal_charge_list = batch.reference.charge[0][atom_to_res][mask]
    pdbx_PDB_model_num_list = [1] * length

    # to mmcif format
    group_PDB_list = _to_mmcif_format(group_PDB_list)
    type_symbol_list = _to_mmcif_format(type_symbol_list)
    label_atom_id_list = _to_mmcif_format(label_atom_id_list)
    label_comp_id_list = _to_mmcif_format(label_comp_id_list)
    label_asym_id_list = _to_mmcif_format(label_asym_id_list)
    label_entity_id_list = _to_mmcif_format(label_entity_id_list)
    label_seq_id_list = _to_mmcif_format(label_seq_id_list)
    ins_code_list = _to_mmcif_format(ins_code_list)
    auth_idx_list = _to_mmcif_format(auth_idx_list)
    auth_seq_id_list = _to_mmcif_format(auth_seq_id_list)
    id_list = _to_mmcif_format(id_list)
    cartn_x_list = [f"{x:.3f}" for x in cartn_x_list]
    cartn_y_list = [f"{y:.3f}" for y in cartn_y_list]
    cartn_z_list = [f"{z:.3f}" for z in cartn_z_list]
    cartn_x_list = _to_mmcif_format(cartn_x_list)
    cartn_y_list = _to_mmcif_format(cartn_y_list)
    cartn_z_list = _to_mmcif_format(cartn_z_list)
    occupancy_list = [f"{o:.2f}" for o in occupancy_list]
    b_iso_or_equiv_list = [f"{b:.2f}" for b in b_iso_or_equiv_list]
    occupancy_list = _to_mmcif_format(occupancy_list)
    b_iso_or_equiv_list = _to_mmcif_format(b_iso_or_equiv_list)
    pdbx_formal_charge_list = _to_mmcif_format(pdbx_formal_charge_list)
    pdbx_PDB_model_num_list = _to_mmcif_format(pdbx_PDB_model_num_list)

    for idx in range(length):
        fields = [
            group_PDB_list[idx],
            id_list[idx],
            type_symbol_list[idx],
            label_atom_id_list[idx],
            label_alt_id_list[idx],
            label_comp_id_list[idx],
            label_asym_id_list[idx],
            label_entity_id_list[idx],
            label_seq_id_list[idx],
            ins_code_list[idx],
            cartn_x_list[idx],
            cartn_y_list[idx],
            cartn_z_list[idx],
            occupancy_list[idx],
            b_iso_or_equiv_list[idx],
            pdbx_formal_charge_list[idx],
            auth_seq_id_list[idx],
            label_comp_id_list[idx],
            label_asym_id_list[idx],
            label_atom_id_list[idx],
            pdbx_PDB_model_num_list[idx],
        ]
        output += " ".join(map(str, fields)) + "\n"

    save_path.write_text(output)
