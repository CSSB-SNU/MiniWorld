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

    # Map each token back to its cifmol-residue index (== index into `chem_comp_ids`).
    # Tokens are emitted residue-by-residue with all tokens of one residue consecutive,
    # so unique_consecutive on a per-(chain, cif_res) key yields the cifmol-residue
    # index directly — independent of any (asym, cif_res) sort order.
    stride = int(token_cif_res.max().item()) + 1
    token_res_key = token_asym * stride + token_cif_res
    _, token_crop_res = torch.unique_consecutive(token_res_key, return_inverse=True)
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

    # _entity_poly_seq — one entry per unique residue per polymer entity.
    # mmCIF spec: `num` must be 1-based and sequential per entity (and must equal
    # _atom_site.label_seq_id for the corresponding atoms). We collect unique
    # (entity, cif_res) pairs in cif_res order, then assign a 1-based num per entity.
    # `entity_cif_res_to_label_seq` is reused below for _atom_site.label_seq_id so
    # both fields stay in lock-step.
    seen_res: set[tuple[int, int]] = set()
    per_entity_res: dict[int, list[tuple[int, str]]] = {}
    for t in range(token_cif_res.shape[0]):
        if not token_mask[t]:
            continue
        eid = token_entity[t].item()
        if eid not in poly_entity_set:
            continue
        cif_res = token_cif_res[t].item()
        key = (eid, cif_res)
        if key in seen_res:
            continue
        seen_res.add(key)
        crop_idx = token_crop_res[t].item()
        mon_id = chem_comp_ids[crop_idx] if 0 <= crop_idx < len(chem_comp_ids) else "UNK"
        per_entity_res.setdefault(eid, []).append((cif_res, mon_id))

    entity_cif_res_to_label_seq: dict[tuple[int, int], int] = {}
    seq_entries: list[tuple[int, str, int]] = []
    for eid in sorted(per_entity_res.keys()):
        residues = sorted(per_entity_res[eid], key=lambda x: x[0])
        for new_num, (cif_res, mon_id) in enumerate(residues, start=1):
            entity_cif_res_to_label_seq[(eid, cif_res)] = new_num
            seq_entries.append((eid, mon_id, new_num))

    if seq_entries:
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
    # group_PDB must be HETATM for anything outside a polymer entity -- ligands,
    # ions, branched sugars, water. `hetero` cannot decide this: in mmCIF
    # `_pdbx_poly_seq_scheme.hetero` flags point MICROheterogeneity (alternate
    # residues at one position), not het records, and it is 0 for every residue
    # in BioMolDB -- so keying group_PDB off it wrote ATOM for every ligand.
    # Downstream tools that classify ligands by the het flag then see none:
    # DockQ (`parse_hetatms`) fails with "no identical corresponding chain was
    # found" on protein-ligand targets. Entity type is the correct source, and
    # `poly_entity_set` is already derived from it above.
    het_res = hetero[atom_crop_res][atom_mask]
    atom_entity_masked = atom_entity[atom_mask]
    group_PDB_list = [
        "HETATM" if (int(e) not in poly_entity_set or int(g) == 1) else "ATOM"
        for e, g in zip(atom_entity_masked.tolist(), het_res.tolist(), strict=True)
    ]

    id_list = 1 + np.arange(length)
    type_symbol_list = atom_mapping.index_to_atom(
        batch.reference.element[0].cpu().numpy(),
    )[atom_mask.numpy()]

    label_alt_id_list = ["."] * length
    label_asym_id_list = atom_asym[atom_mask].long()
    label_entity_id_list = atom_entity[atom_mask].long()
    # auth_seq_id keeps the original PDB residue number (cif_res).
    # label_seq_id must match _entity_poly_seq.num — 1-based per polymer entity —
    # for polymer atoms, and "." for non-polymer atoms (mmCIF spec).
    auth_seq_id_list = atom_cif_res[atom_mask].long()
    masked_eids = atom_entity[atom_mask].tolist()
    masked_cif_res = atom_cif_res[atom_mask].tolist()
    label_seq_id_list = [
        str(entity_cif_res_to_label_seq[(eid, cr)])
        if (eid, cr) in entity_cif_res_to_label_seq
        else "."
        for eid, cr in zip(masked_eids, masked_cif_res)
    ]
    ins_code_list = ["?"] * length

    cartn_x_list = xyz[atom_mask, 0]
    cartn_y_list = xyz[atom_mask, 1]
    cartn_z_list = xyz[atom_mask, 2]
    occupancy_list = [1.0] * length
    b_iso_or_equiv_list = [100.0] * length
    pdbx_formal_charge_list = batch.reference.charge[0][atom_mask].long()
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

    # ── bonds: _struct_conn (inter-residue covalent links) ──
    # mmCIF readers (biotite/PXMeter) auto-derive intra-residue bonds — including
    # double/aromatic bond orders — and standard polymer backbone links from
    # residue names via the CCD, but they cannot derive non-standard covalent
    # links (disulfides, glycans, covalent ligands, modified residues); those
    # must be supplied via _struct_conn, which this writes.
    #
    # We deliberately do NOT emit _chem_comp_bond: the batch's bond data
    # (cifmol.atoms.bond_type) only covers polymer residues, not non-polymer
    # ligands, and biotite treats _chem_comp_bond as an *exclusive* per-component
    # override (no CCD fallback for comps absent from it). Emitting an incomplete
    # table would therefore drop every ligand's intra-residue bonds — the
    # opposite of the goal. See git history / to_cif notes for the analysis.
    output += _bond_records(
        bonds=(batch.bonds[0] if getattr(batch, "bonds", None) else None),
        atom_mask=atom_mask,
        atom_crop_res=atom_crop_res,
        atom_asym=atom_asym,
        atom_entity=atom_entity,
        atom_cif_res=atom_cif_res,
        chem_comp_ids=chem_comp_ids,
        atom_ids=atom_ids,
        entity_cif_res_to_label_seq=entity_cif_res_to_label_seq,
        pad=_pad,
    )

    save_path.write_text(output)


# struct_conn conn_type_id values that denote covalent (i.e. real) connections;
# everything else (hydrog, saltbr, mismat, ...) is dropped.
_COVALENT_CONN_TYPES = {
    "covale",
    "covale_base",
    "covale_phosphate",
    "covale_sugar",
    "disulf",
    "modres",
    "modres_link",
    "metalc",
}


def _bond_records(  # noqa: PLR0913
    bonds: dict | None,
    atom_mask: torch.Tensor,
    atom_crop_res: torch.Tensor,
    atom_asym: torch.Tensor,
    atom_entity: torch.Tensor,
    atom_cif_res: torch.Tensor,
    chem_comp_ids: list[str],
    atom_ids,
    entity_cif_res_to_label_seq: dict[tuple[int, int], int],
    pad,
) -> str:
    """Build the ``_struct_conn`` CIF loop (inter-residue covalent links).

    Returns an empty string when no bond metadata is available (callers that
    construct a Batch without bonds keep the previous bond-free output) or when
    the structure has no non-standard covalent links.

    Note: we intentionally do not write ``_chem_comp_bond``. Intra-residue bonds
    (including double/aromatic orders) are recovered by mmCIF readers from the
    CCD by residue name, and the batch lacks bond data for non-polymer ligands,
    so an emitted ``_chem_comp_bond`` would necessarily be incomplete — and
    biotite uses it as an exclusive override, which would then drop ligand bonds.
    """
    if not bonds:
        return ""

    mask = atom_mask.cpu().numpy().astype(bool)
    n_atom = mask.shape[0]
    res_of = atom_crop_res.cpu().numpy()
    asym_of = atom_asym.cpu().numpy()
    eid_of = atom_entity.cpu().numpy()
    cifres_of = atom_cif_res.cpu().numpy()
    comp_of = np.asarray(chem_comp_ids, dtype=object)[res_of]
    name_of = np.asarray(atom_ids)

    def _present(i: int, j: int) -> bool:
        return 0 <= i < n_atom and 0 <= j < n_atom and mask[i] and mask[j]

    def _seq_of(i: int) -> str:
        key = (int(eid_of[i]), int(cifres_of[i]))
        return str(entity_cif_res_to_label_seq[key]) if key in entity_cif_res_to_label_seq else "."

    # Inter-residue, non-backbone chemical bonds carried in bond_type (e.g. a
    # polymer crosslink): readers won't auto-derive them, so route to struct_conn.
    # ("canonical" marks the standard backbone, which readers connect themselves.)
    b_src = np.asarray(bonds["bond_src"], dtype=np.int64)
    b_dst = np.asarray(bonds["bond_dst"], dtype=np.int64)
    b_order = np.asarray(bonds["bond_order"]).astype(str)
    sc_extra: list[tuple[int, int]] = []
    for i in range(b_src.shape[0]):
        s, d = int(b_src[i]), int(b_dst[i])
        if _present(s, d) and res_of[s] != res_of[d] and str(b_order[i]).lower() != "canonical":
            sc_extra.append((s, d))

    output = ""

    # ── _struct_conn: inter-residue covalent links ──
    sc_src = np.asarray(bonds["sc_src"], dtype=np.int64)
    sc_dst = np.asarray(bonds["sc_dst"], dtype=np.int64)
    sc_type = np.asarray(bonds["sc_type"]).astype(str)
    sc_rows: list[tuple] = []
    seen: set[tuple[int, int]] = set()

    def _add_conn(s: int, d: int, conn_type: str) -> None:
        if not _present(s, d):
            return
        key = (min(s, d), max(s, d))
        if key in seen:
            return
        seen.add(key)
        value_order = "?" if conn_type == "metalc" else "sing"
        sc_rows.append((
            conn_type,
            value_order,
            str(asym_of[s]), str(asym_of[d]),
            str(comp_of[s]), str(comp_of[d]),
            _seq_of(s), _seq_of(d),
            str(name_of[s]), str(name_of[d]),
            "?", "?",
        ))

    for i in range(sc_src.shape[0]):
        conn_type = str(sc_type[i]).lower()
        if conn_type in _COVALENT_CONN_TYPES:
            _add_conn(int(sc_src[i]), int(sc_dst[i]), conn_type)
    for s, d in sc_extra:
        _add_conn(s, d, "covale")

    if sc_rows:
        ids = [str(k + 1) for k in range(len(sc_rows))]
        cols = list(zip(*sc_rows))
        padded = [pad(ids)] + [pad([str(x) for x in col]) for col in cols]
        output += (
            "#\nloop_\n"
            "_struct_conn.id\n"
            "_struct_conn.conn_type_id\n"
            "_struct_conn.pdbx_value_order\n"
            "_struct_conn.ptnr1_label_asym_id\n"
            "_struct_conn.ptnr2_label_asym_id\n"
            "_struct_conn.ptnr1_label_comp_id\n"
            "_struct_conn.ptnr2_label_comp_id\n"
            "_struct_conn.ptnr1_label_seq_id\n"
            "_struct_conn.ptnr2_label_seq_id\n"
            "_struct_conn.ptnr1_label_atom_id\n"
            "_struct_conn.ptnr2_label_atom_id\n"
            "_struct_conn.pdbx_ptnr1_PDB_ins_code\n"
            "_struct_conn.pdbx_ptnr2_PDB_ins_code\n"
        )
        for r in range(len(sc_rows)):
            output += " ".join(col[r] for col in padded) + "\n"

    return output
