from pathlib import Path

import numpy as np
import torch

from miniworld.data.constants import AtomMapping
from miniworld.data.features import Batch

# MiniWorld entity_type index → CIF _entity.type
_ENTITY_IDX_TO_CIF_ENTITY_TYPE = {
    0: "polymer",      # ANTIBODY
    1: "polymer",      # PROTEIN
    2: "polymer",      # DPROTEIN
    3: "polymer",      # RNA
    4: "polymer",      # DNA
    5: "polymer",      # NA
    6: "non-polymer",  # LIGAND
    7: "branched",     # BRANCHED
}

# MiniWorld entity_type index → CIF _entity_poly.type
_ENTITY_IDX_TO_POLY_TYPE = {
    0: "polypeptide(L)",            # ANTIBODY
    1: "polypeptide(L)",            # PROTEIN
    2: "polypeptide(D)",            # DPROTEIN
    3: "polyribonucleotide",        # RNA
    4: "polydeoxyribonucleotide",   # DNA
    5: "other",                     # NA
}


def _unwrap_node_features(raw_list) -> list[str]:
    """Extract plain strings from a list that may contain NodeFeature objects."""
    if isinstance(raw_list, list) and len(raw_list) > 0 and isinstance(raw_list[0], list):
        raw_list = raw_list[0]
    return [str(getattr(c, "value", c)) for c in raw_list]


def batch_to_cif(  # noqa: PLR0915
    batch: Batch,
    atom_pos_pred: torch.Tensor | None,
    save_path: Path,
) -> None:
    """Convert a batch to PXMeter-compatible mmCIF format."""

    def _pad(array: torch.Tensor | list) -> list[str]:
        array = array.cpu().numpy() if isinstance(array, torch.Tensor) else array
        items = [str(x) for x in array]
        w = max(len(s) for s in items)
        return [s.ljust(w) for s in items]

    # ── entry name ────────────────────────────────────────────────────
    _name = batch.name[0]
    if isinstance(_name, list):
        _name = _name[0]
    entry_id = str(_name)

    # ── unwrap NodeFeature lists ──────────────────────────────────────
    chem_comp_ids = _unwrap_node_features(batch.chem_comp_ids[0])  # per-residue (crop order)

    # ── mapping tensors (all on CPU) ──────────────────────────────────
    atom_mask = batch.structure.atom_pos_mask[0].bool().cpu()
    token_mask = batch.structure.token_mask[0].bool().cpu()
    atom_to_token = batch.scheme.atom_to_token_idx_map[0].cpu()
    atom_to_chain_idx = batch.scheme.atom_to_chain_id[0].cpu()

    token_cif_res = batch.scheme.token_residue_idx[0].cpu()   # CIF residue idx per token
    atom_cif_res = token_cif_res[atom_to_token]                # CIF residue idx per atom

    # per-token chain / entity
    token_asym = batch.scheme.token_asym_id[0].cpu()
    token_entity = batch.scheme.token_entity_id[0].cpu()
    atom_asym = token_asym[atom_to_token]
    atom_entity = token_entity[atom_to_token]

    # Build crop-order residue index using (asym_id, cif_res_idx) as key.
    # CIF residue indices are NOT unique across chains (e.g., chain 0 res 1 = LKC,
    # chain 3 res 1 = MG), so cif_res alone is insufficient.
    stride = int(token_cif_res.max().item()) + 1
    token_res_key = token_asym * stride + token_cif_res   # unique per (chain, res)
    _, token_crop_res = torch.unique(token_res_key, sorted=True, return_inverse=True)
    atom_crop_res = token_crop_res[atom_to_token]

    chain_entity_types = batch.chain.entity_type[0].cpu()

    # ── entity info ───────────────────────────────────────────────────
    entity_info: dict[int, int] = {}
    entity_asym_ids: dict[int, set[int]] = {}

    for eid, asym, cidx in zip(
        atom_entity[atom_mask], atom_asym[atom_mask], atom_to_chain_idx[atom_mask],
    ):
        eid_val = eid.item()
        if eid_val not in entity_info:
            entity_info[eid_val] = chain_entity_types[cidx.item()].item()
        entity_asym_ids.setdefault(eid_val, set()).add(asym.item())

    sorted_entity_ids = sorted(entity_info.keys())
    poly_entities = [e for e in sorted_entity_ids if entity_info[e] in _ENTITY_IDX_TO_POLY_TYPE]
    poly_entity_set = set(poly_entities)

    # ── write header ──────────────────────────────────────────────────
    output = f"#\ndata_{entry_id}\n"
    output += f"#\n_entry.id   {entry_id}\n"

    # _entity
    if sorted_entity_ids:
        output += "#\nloop_\n_entity.id\n_entity.type\n"
        for eid in sorted_entity_ids:
            cif_type = _ENTITY_IDX_TO_CIF_ENTITY_TYPE.get(entity_info[eid], "non-polymer")
            output += f"{eid} {cif_type}\n"

    # _entity_poly
    if poly_entities:
        output += "#\nloop_\n_entity_poly.entity_id\n_entity_poly.type\n_entity_poly.pdbx_strand_id\n"
        for eid in poly_entities:
            poly_type = _ENTITY_IDX_TO_POLY_TYPE[entity_info[eid]]
            strand_ids = ",".join(str(a) for a in sorted(entity_asym_ids[eid]))
            output += f"{eid} '{poly_type}' {strand_ids}\n"

    # _entity_poly_seq — one entry per residue, using CIF residue idx as num
    seen_res: set[tuple[int, int, int]] = set()
    seq_entries: list[tuple[int, str, int]] = []
    for t in range(token_cif_res.shape[0]):
        if not token_mask[t]:
            continue
        eid = token_entity[t].item()
        if eid not in poly_entity_set:
            continue
        asym = token_asym[t].item()
        cif_res = token_cif_res[t].item()
        key = (eid, asym, cif_res)
        if key in seen_res:
            continue
        seen_res.add(key)
        crop_idx = token_crop_res[t].item()
        mon_id = chem_comp_ids[crop_idx] if 0 <= crop_idx < len(chem_comp_ids) else "UNK"
        seq_entries.append((eid, mon_id, cif_res))

    if seq_entries:
        seq_entries.sort(key=lambda x: (x[0], x[2]))
        output += "#\nloop_\n_entity_poly_seq.entity_id\n_entity_poly_seq.mon_id\n_entity_poly_seq.num\n"
        for eid, mon_id, seq_num in seq_entries:
            output += f"{eid} {mon_id} {seq_num}\n"

    # ── _atom_site ────────────────────────────────────────────────────
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
    output += "#\nloop_\n" + "\n".join(header) + "\n"

    atom_mapping = AtomMapping()
    xyz = (atom_pos_pred if atom_pos_pred is not None else batch.structure.atom_pos)[0]
    length = atom_mask.sum().item()

    hetero = torch.tensor(batch.heteros[0].value)
    atom_ids = batch.atom_ids[0]

    # Per-atom fields (masked to valid atoms)
    label_atom_id_list = np.array(atom_ids)[atom_mask.numpy()]
    label_comp_id_list = np.array(chem_comp_ids)[atom_crop_res.numpy()][atom_mask.numpy()]
    group_PDB_list = hetero[atom_crop_res][atom_mask]
    group_PDB_list = ["HETATM" if g == 1 else "ATOM" for g in group_PDB_list]

    id_list = 1 + np.arange(length)
    type_symbol_list = atom_mapping.index_to_atom(
        batch.reference.element[0].cpu().numpy(),
    )[atom_mask.numpy()]

    label_alt_id_list = ["."] * length
    label_asym_id_list = atom_asym[atom_mask]
    label_entity_id_list = atom_entity[atom_mask]
    # Use CIF residue index — atoms of the same residue share the same seq_id
    label_seq_id_list = atom_cif_res[atom_mask]
    auth_seq_id_list = label_seq_id_list
    ins_code_list = ["?"] * length

    cartn_x_list = xyz[atom_mask, 0]
    cartn_y_list = xyz[atom_mask, 1]
    cartn_z_list = xyz[atom_mask, 2]
    occupancy_list = [1.0] * length
    b_iso_or_equiv_list = [100.0] * length
    pdbx_formal_charge_list = batch.reference.charge[0][atom_mask]
    pdbx_PDB_model_num_list = [1] * length

    # Format all columns
    group_PDB_list = _pad(group_PDB_list)
    type_symbol_list = _pad(type_symbol_list)
    label_atom_id_list = _pad(label_atom_id_list)
    label_comp_id_list = _pad(label_comp_id_list)
    label_asym_id_list = _pad(label_asym_id_list)
    label_entity_id_list = _pad(label_entity_id_list)
    label_seq_id_list = _pad(label_seq_id_list)
    ins_code_list = _pad(ins_code_list)
    auth_seq_id_list = _pad(auth_seq_id_list)
    id_list = _pad(id_list)
    cartn_x_list = _pad([f"{x:.3f}" for x in cartn_x_list])
    cartn_y_list = _pad([f"{y:.3f}" for y in cartn_y_list])
    cartn_z_list = _pad([f"{z:.3f}" for z in cartn_z_list])
    occupancy_list = _pad([f"{o:.2f}" for o in occupancy_list])
    b_iso_or_equiv_list = _pad([f"{b:.2f}" for b in b_iso_or_equiv_list])
    pdbx_formal_charge_list = _pad(pdbx_formal_charge_list)
    pdbx_PDB_model_num_list = _pad(pdbx_PDB_model_num_list)

    for i in range(length):
        output += " ".join([
            group_PDB_list[i],
            id_list[i],
            type_symbol_list[i],
            label_atom_id_list[i],
            label_alt_id_list[i],
            label_comp_id_list[i],
            label_asym_id_list[i],
            label_entity_id_list[i],
            label_seq_id_list[i],
            ins_code_list[i],
            cartn_x_list[i],
            cartn_y_list[i],
            cartn_z_list[i],
            occupancy_list[i],
            b_iso_or_equiv_list[i],
            pdbx_formal_charge_list[i],
            auth_seq_id_list[i],
            label_comp_id_list[i],
            label_asym_id_list[i],
            label_atom_id_list[i],
            pdbx_PDB_model_num_list[i],
        ]) + "\n"

    save_path.write_text(output)
