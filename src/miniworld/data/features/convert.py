from __future__ import annotations

from typing import TYPE_CHECKING

import biomol
import numpy as np
import torch

from miniworld.data.constants import AtomMapping, EntityMapping
from miniworld.data.pipeline.template import ProteinTemplate
from miniworld.utils.structure import SE3_oper

from .batch import Batch
from .features import (
    ChainFeatures,
    MSAFeatures,
    ReferenceFeatures,
    SchemeFeatures,
    SequenceFeatures,
    StructureFeatures,
    TemplateFeatures,
)

if TYPE_CHECKING:
    from miniworld.data.mols import CIFMolAttached


def atom_bonds_to_token_bonds(
    cifmol: CIFMolAttached,
    atom_to_token_idx_map: np.ndarray,  # (n_atoms,)
) -> np.ndarray:
    """Convert atom-level bonds to token-level bonds, keeping only inter-residue bonds between non-canonical residues."""
    # add covale struct_conn info as token bond
    try:
        sc_value = cifmol.atoms.struct_conn.value[:, 0]
    except biomol.exceptions.FeatureKeyError:
        # to avoid empty bond, add [0,0]
        return np.array([[0, 0]])
    sc_keep = sc_value == "covale"
    sc_src, sc_dst = cifmol.atoms.struct_conn.src, cifmol.atoms.struct_conn.dst
    sc_src = sc_src[sc_keep].astype(np.int64, copy=False)
    sc_dst = sc_dst[sc_keep].astype(np.int64, copy=False)
    sc_t_src = atom_to_token_idx_map[sc_src]
    sc_t_dst = atom_to_token_idx_map[sc_dst]
    sc_a = np.minimum(sc_t_src, sc_t_dst)
    sc_b = np.maximum(sc_t_src, sc_t_dst)
    sc_pairs = np.stack([sc_a, sc_b], axis=1)

    return sc_pairs.astype(np.int64, copy=False)


def to_scheme_features(
    cifmol: CIFMolAttached,
    token_to_residue_idx_map: np.ndarray,
    atom_to_token_idx_map: np.ndarray,
) -> SchemeFeatures:
    """Make scheme features."""
    cropped_token_len = token_to_residue_idx_map.shape[0]
    chain_num = cifmol.chains.chain_id.shape[0]
    chain_asym_id = np.arange(chain_num).astype(np.int64)
    chain_entity_id = cifmol.chains.entity_id.value
    same_entity = chain_entity_id[:, None] == chain_entity_id[None, :]
    chain_sym_id = np.triu(same_entity, k=0).sum(axis=0) - 1

    token_idx = np.arange(cropped_token_len, dtype=np.int64)
    res_to_chain = cifmol.index_table.res_to_chain

    token_residue_idx = np.take(cifmol.residues.cif_idx.value, token_to_residue_idx_map)
    token_to_chain = np.take(res_to_chain, token_to_residue_idx_map)
    token_asym_id = np.take(chain_asym_id, token_to_chain)
    token_entity_id = np.take(chain_entity_id, token_to_chain)
    token_sym_id = np.take(chain_sym_id, token_to_chain)

    atom_to_chain_id = np.take(token_asym_id, atom_to_token_idx_map)

    return SchemeFeatures.from_sample(
        token_residue_idx=torch.from_numpy(token_residue_idx.astype(np.int64)),
        token_idx=torch.from_numpy(token_idx.astype(np.int64)),
        token_asym_id=torch.from_numpy(token_asym_id.astype(np.int64)),
        token_entity_id=torch.from_numpy(token_entity_id.astype(np.int64)),
        token_sym_id=torch.from_numpy(token_sym_id.astype(np.int64)),
        atom_to_token_idx_map=torch.from_numpy(
            atom_to_token_idx_map.astype(np.int64),
        ),
        atom_to_chain_id=torch.from_numpy(atom_to_chain_id.astype(np.int64)),
    )


def to_msa_features(
    msa_features: MSAFeatures,
    token_to_residue: np.ndarray,
) -> MSAFeatures:
    """Map residue-level MSA features to token-level using token_to_residue mapping."""
    return MSAFeatures(
        aligned_sequences=msa_features.aligned_sequences[:, :, token_to_residue],
        mask=msa_features.mask,
        has_deletion=msa_features.has_deletion[:, :, token_to_residue],
        deletion_value=msa_features.deletion_value[:, :, token_to_residue],
        profile=msa_features.profile[:, token_to_residue, :],
        deletion_mean=msa_features.deletion_mean[:, token_to_residue],
    )


def to_template_features(
    templates: ProteinTemplate,
    token_to_residue_idx_map: np.ndarray,
) -> TemplateFeatures:
    """Convert ProteinTemplate to TemplateFeatures, mapping from residues to tokens."""
    return TemplateFeatures.from_sample(
        mask=torch.from_numpy(templates.mask.astype(np.bool)),
        ids=torch.from_numpy(
            templates.ids[:, token_to_residue_idx_map].astype(np.int64),
        ),
        res_type=torch.from_numpy(
            templates.res_type[:, token_to_residue_idx_map].astype(np.int64),
        ),
        cb_xyz=torch.from_numpy(
            templates.cb_xyz[:, token_to_residue_idx_map].astype(np.float32),
        ),
        cb_mask=torch.from_numpy(
            templates.cb_mask[:, token_to_residue_idx_map].astype(np.bool),
        ),
        bb_xyz=torch.from_numpy(
            templates.bb_xyz[:, token_to_residue_idx_map].astype(np.float32),
        ),
        bb_mask=torch.from_numpy(
            templates.bb_mask[:, token_to_residue_idx_map].astype(np.bool),
        ),
    )


def to_reference_features(
    cifmol: CIFMolAttached,
    rng: np.random.Generator | None = None,
) -> ReferenceFeatures:
    """Convert CIFMol to ReferenceFeatures."""
    cropped_residue_len = len(cifmol.residues)
    ref_pos = cifmol.atoms.model_xyz.value
    ref_pos = np.array(ref_pos, dtype=object)

    mask = (ref_pos == "?") | (ref_pos == ".")
    ref_pos[mask] = 0.0
    ref_pos = ref_pos.astype(np.float32, copy=False)
    ref_mask = ~np.isnan(ref_pos).any(axis=1)
    ref_element = cifmol.atoms.element.value
    ref_charge = cifmol.atoms.charge.value
    ref_charge = np.array(
        [float(c) if c not in {"?", "."} else 0.0 for c in ref_charge],
    )
    ref_space_uid = cifmol.index_table.atom_to_res

    N_res = ref_space_uid.max() + 1
    res_to_atoms = [np.where(ref_space_uid == i)[0] for i in range(N_res)]

    Rs, Ts = SE3_oper(cropped_residue_len, rng=rng)
    random_ref_pos = []
    for ii, atom_indices in enumerate(res_to_atoms):
        R, T = Rs[ii], Ts[ii]
        _ref_pos = ref_pos[atom_indices]
        _ref_pos = (_ref_pos - _ref_pos.mean(axis=0)) @ R + T  # random SE(3) operation
        random_ref_pos.append(_ref_pos)
    ref_pos = np.vstack(random_ref_pos)
    ref_element = AtomMapping().atom_to_index(ref_element)  # convert str to int
    return ReferenceFeatures.from_sample(
        pos=torch.from_numpy(ref_pos.astype(np.float32)),
        mask=torch.from_numpy(ref_mask.astype(np.bool)),
        element=torch.from_numpy(ref_element.astype(np.int64)),
        charge=torch.from_numpy(ref_charge.astype(np.float32)),
        space_uid=torch.from_numpy(ref_space_uid.astype(np.int64)),
    )


def to_structure_features(
    cifmol: CIFMolAttached,
    atom_to_token_idx_map: np.ndarray,
    token_to_residue_idx_map: np.ndarray,
) -> StructureFeatures:
    """Convert CIFMol to StructureFeatures."""
    cropped_token_len = len(token_to_residue_idx_map)
    atom_pos = cifmol.atoms.xyz.value
    atom_pos_mask = np.isfinite(atom_pos).all(axis=1)
    atom_mask = np.ones_like(atom_pos_mask, dtype=bool)

    # centering atom_pos
    valid_pos = atom_pos[atom_pos_mask]  # (N_valid, 3)
    mean_vector = valid_pos.mean(axis=0, keepdims=True)
    atom_pos = atom_pos - mean_vector
    atom_pos = np.where(atom_pos_mask.astype(bool)[:, None], atom_pos, 0.0)

    token_bond = atom_bonds_to_token_bonds(
        cifmol=cifmol,
        atom_to_token_idx_map=atom_to_token_idx_map,
    )

    atom_bond_type = cifmol.atoms.bond_type.value  # (n_atom_bond, )
    atom_bond_stereo = cifmol.atoms.bond_stereo.value  # (n_atom_bond, )
    atom_bond_aromatic = cifmol.atoms.bond_aromatic.value  # (n_atom_bond, )
    atom_bond = np.stack(
        [atom_bond_type, atom_bond_stereo, atom_bond_aromatic],
        axis=1,
    )  # (n_atom_bond, 3)
    atom_bond = np.zeros_like(atom_bond, dtype=np.int64)  # placeholder

    return StructureFeatures.from_sample(
        atom_pos=torch.from_numpy(atom_pos.astype(np.float32)),
        atom_pos_mask=torch.from_numpy(atom_pos_mask.astype(np.bool)),
        atom_mask=torch.from_numpy(atom_mask.astype(np.bool)),
        atom_bond=torch.from_numpy(atom_bond.astype(np.int64)),
        token_mask=torch.ones((cropped_token_len,), dtype=torch.bool),  # all ones
        token_bond=torch.from_numpy(token_bond.astype(np.int64)),
    )


def to_chain_features(
    cifmol: CIFMolAttached,
) -> ChainFeatures:
    """Convert CIFMol to ChainFeatures."""
    entity_mapping = EntityMapping()
    seq_id_list = cifmol.chains.seq_id.value.tolist()
    entity_id_list = [seq_id[0] for seq_id in seq_id_list]
    entity_types = entity_mapping.tag_to_idx(entity_id_list)

    return ChainFeatures.from_sample(
        entity_type=torch.from_numpy(entity_types.astype(np.int64)),
    )


def make_batch(
    cifmol: CIFMolAttached,
    msa: MSAFeatures,
    templates: ProteinTemplate,
    atom_to_token_idx_map: np.ndarray,
    token_to_residue_idx_map: np.ndarray,
    rng: np.random.Generator | None = None,
) -> Batch:
    """Make features from cifmol and MSA."""
    if rng is None:
        rng = np.random.default_rng()
    msa_token = to_msa_features(msa, token_to_residue_idx_map)
    template_token = to_template_features(templates, token_to_residue_idx_map)

    scheme = to_scheme_features(cifmol, token_to_residue_idx_map, atom_to_token_idx_map)
    sequence = SequenceFeatures(token_type=msa_token.aligned_sequences[:, 0])
    reference = to_reference_features(cifmol, rng)
    structure = to_structure_features(
        cifmol,
        atom_to_token_idx_map,
        token_to_residue_idx_map,
    )
    chain = to_chain_features(cifmol)

    hetero = cifmol.residues.hetero
    atom_ids = cifmol.atoms.id
    chem_comp_ids = cifmol.residues.chem_comp_id

    pdb_id, assembly_id, model_id, alt_id = (
        cifmol.id,
        cifmol.assembly_id,
        cifmol.model_id,
        cifmol.alt_id,
    )

    return Batch(
        name=[f"{pdb_id}_{assembly_id}_{model_id}_{alt_id}"],
        heteros=[hetero],
        atom_ids=[atom_ids],
        chem_comp_ids=[chem_comp_ids],
        sequence=sequence,
        structure=structure,
        msa=msa_token,
        template=template_token,
        reference=reference,
        scheme=scheme,
        chain=chain,
    )
