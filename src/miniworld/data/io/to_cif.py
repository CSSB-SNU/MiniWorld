from pathlib import Path

import numpy as np
import torch

from miniworld.data.constants import AtomMapping
from miniworld.data.features import Batch

# Not fully implemented yet...


def batch_to_cif(  # noqa: PLR0915
    batch: Batch,
    atom_pos_pred: torch.Tensor | None,
    save_path: Path,
    chain_names: dict[str, str] | None = None,
) -> None:
    """Convert a batch to CIF format string.

    ``chain_names`` maps the numeric ``token_asym_id`` (as a string, e.g. ``"0"``)
    to the desired output chain label (e.g. ``"A"``). FoldBench scores interfaces
    by chain id, so the CIF must carry the real chain letters (A/E/F), not the
    internal 0/1/2. When ``None`` the numeric asym id is written verbatim (legacy).
    """

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
    atom_mapping = AtomMapping()

    xyz = (
        atom_pos_pred if atom_pos_pred is not None else batch.structure.atom_pos
    )  # if no prediction, use input coords
    xyz = xyz[0]
    mask = batch.structure.atom_pos_mask[0].bool()
    length = mask.sum().item()
    atom_to_token = batch.scheme.atom_to_token_idx_map[0]
    atom_to_res = batch.reference.space_uid[0]
    atom_to_asym = batch.scheme.token_asym_id[0][atom_to_token]
    atom_to_entity = batch.scheme.token_entity_id[0][atom_to_token]

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
    if chain_names is not None:
        # Map internal numeric asym id -> real chain label (A/E/F) for FoldBench.
        label_asym_id_list = [
            chain_names.get(str(int(a)), str(int(a)))
            for a in label_asym_id_list.cpu().numpy()
        ]
    label_entity_id_list = atom_to_entity[mask]
    label_seq_id_list = batch.scheme.token_residue_idx[0][atom_to_token][mask]
    auth_idx_list = label_seq_id_list  # assuming auth seq id == label seq id
    auth_seq_id_list = label_seq_id_list  # assuming auth seq id == label seq id
    ins_code_list = ["?"] * length

    cartn_x_list = xyz[mask, 0]
    cartn_y_list = xyz[mask, 1]
    cartn_z_list = xyz[mask, 2]
    occupancy_list = [1.0] * length
    b_iso_or_equiv_list = [100.0] * length  # TODO replace with plddt.
    # int(), not the raw float tensor: _atom_site.pdbx_formal_charge is an mmCIF int
    # column, and str(0.0) -> "0.0" makes the file unreadable by OpenStructure
    # ("Expecting integer value for atom_site.pdbx_formal_charge"), i.e. by every
    # FoldBench metric, which all go through `ost compare-structures`.
    pdbx_formal_charge_list = [int(c) for c in batch.reference.charge[0][mask].tolist()]
    pdbx_PDB_model_num_list = [1] * length

    # ---- entity records -------------------------------------------------
    # _atom_site alone is not enough for the consumers that score these files.
    # OpenStructure falls back to sequence-identity heuristics without
    # _entity/_entity_poly ("mmCIF file does not define _entity.type ...",
    # "SEQRES is missing for polymer chain(s) ..."), and FoldBench's DockQv2
    # parser hard-requires _entity_poly_seq.entity_id -- without it every
    # protein-DNA and protein-RNA target raises KeyError and those two
    # categories score nothing at all. Everything below is derived from the
    # same masked per-atom arrays the _atom_site loop writes, so the two
    # tables cannot disagree.
    #
    # entity_type ints (EntityMapping): 0 ANTIBODY, 1 PROTEIN, 2 DPROTEIN,
    # 3 RNA, 4 DNA, 5 NA, 6 LIGAND, 7 BRANCHED.
    poly_type_by_entity_type = {
        0: "polypeptide(L)",
        1: "polypeptide(L)",
        2: "polypeptide(D)",
        3: "polyribonucleotide",
        4: "polydeoxyribonucleotide",
        5: "polydeoxyribonucleotide/polyribonucleotide hybrid",
    }
    # Index the chain features with token_asym_id, not atom_to_chain_id: the latter
    # doubles as the diffusion solver's rigid-frame grouping and is remapped to group
    # ids when spec.diffusion_groups is set, while token_asym_id is what this writer
    # already uses to assign chain labels.
    atom_entity_type = batch.chain.entity_type[0][atom_to_asym][mask].cpu().numpy()
    raw_entity_ids = [int(e) for e in label_entity_id_list.cpu().numpy()]
    raw_seq_ids = [int(v) for v in label_seq_id_list.cpu().numpy()]
    asym_strs = [
        str(a) for a in (
            label_asym_id_list
            if isinstance(label_asym_id_list, list)
            else label_asym_id_list.cpu().numpy()
        )
    ]

    # First pass, per CHAIN: token_residue_idx runs globally across the complex, so
    # a homomer's second copy carries different residue ids than its first. Number
    # residues 1..N within each chain; copies of one entity then agree, which is
    # what _entity_poly_seq.num means.
    chain_order: list[str] = []
    chain_entity: dict[str, int] = {}
    chain_seq: dict[str, list[tuple[int, str]]] = {}
    entity_order: list[int] = []
    entity_mol_type: dict[int, int] = {}
    entity_chains: dict[int, list[str]] = {}
    for i in range(length):
        e, a = raw_entity_ids[i], asym_strs[i]
        if a not in chain_entity:
            chain_order.append(a)
            chain_entity[a] = e
            chain_seq[a] = []
        if not chain_seq[a] or chain_seq[a][-1][0] != raw_seq_ids[i]:
            chain_seq[a].append((raw_seq_ids[i], str(label_comp_id_list[i])))
        if e not in entity_mol_type:
            entity_order.append(e)
            entity_mol_type[e] = int(atom_entity_type[i])
            entity_chains[e] = []
        if a not in entity_chains[e]:
            entity_chains[e].append(a)

    # The entity's sequence is the first chain that carries it; the others repeat it.
    entity_seq = {e: chain_seq[entity_chains[e][0]] for e in entity_order}
    # mmCIF entity ids are 1-based; label_seq_id is a 1-based index into
    # _entity_poly_seq for polymers and "." for non-polymers (matches the
    # FoldBench ground-truth files).
    entity_id_1based = {e: str(n + 1) for n, e in enumerate(entity_order)}
    is_polymer = {
        e: (t in poly_type_by_entity_type and len(entity_seq[e]) > 1)
        for e, t in entity_mol_type.items()
    }
    seq_num = {
        a: {old: n + 1 for n, (old, _) in enumerate(seq)}
        for a, seq in chain_seq.items()
    }
    label_entity_id_list = [entity_id_1based[e] for e in raw_entity_ids]
    label_seq_id_list = [
        str(seq_num[a][sid]) if is_polymer[e] else "."
        for e, a, sid in zip(raw_entity_ids, asym_strs, raw_seq_ids, strict=True)
    ]

    output += "#\nloop_\n_entity.id\n_entity.type\n"
    for e in entity_order:
        etype = (
            "polymer" if is_polymer[e]
            else ("branched" if entity_mol_type[e] == 7 else "non-polymer")
        )
        output += f"{entity_id_1based[e]} {etype}\n"

    poly_entities = [e for e in entity_order if is_polymer[e]]
    if poly_entities:
        output += (
            "#\nloop_\n_entity_poly.entity_id\n_entity_poly.type\n"
            "_entity_poly.pdbx_strand_id\n"
        )
        for e in poly_entities:
            ptype = poly_type_by_entity_type[entity_mol_type[e]]
            strands = ",".join(entity_chains[e])
            output += f"{entity_id_1based[e]} '{ptype}' {strands}\n"

        output += (
            "#\nloop_\n_entity_poly_seq.entity_id\n_entity_poly_seq.num\n"
            "_entity_poly_seq.mon_id\n_entity_poly_seq.hetero\n"
        )
        for e in poly_entities:
            for n, (_, comp) in enumerate(entity_seq[e]):
                output += f"{entity_id_1based[e]} {n + 1} {comp} n\n"

    output += "#\n"
    output += "loop_\n"
    output += "\n".join(header) + "\n"

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
