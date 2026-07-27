from __future__ import annotations

from typing import TYPE_CHECKING

import biomol
import numpy as np
import torch
from jaxtyping import Int

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

    token_contacts = to_token_contacts(
        x0=torch.from_numpy(atom_pos.astype(np.float32, copy=False)).unsqueeze(0),
        atom_to_token_idx_map=torch.from_numpy(
            atom_to_token_idx_map.astype(np.int64, copy=False),
        ).unsqueeze(0),
        x_mask=torch.from_numpy(
            atom_pos_mask.astype(np.bool_, copy=False),
        ).unsqueeze(0),
        token_length=cropped_token_len,
    ).squeeze(0)

    # centering atom_pos
    valid_pos = atom_pos[atom_pos_mask]  # (N_valid, 3)
    if valid_pos.shape[0] > 0:
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

    # Dense token-bond adjacency [L_token, L_token] (bool), built once here (per
    # sample, host-side) so the captured model forward reads a fixed-shape field
    # instead of scattering the variable-length token_bond inside the CUDA graph.
    tb = token_bond.astype(np.int64).reshape(-1, 2)
    token_bond_feat = torch.zeros(
        (cropped_token_len, cropped_token_len), dtype=torch.bool,
    )
    if tb.shape[0] > 0:
        bi = torch.from_numpy(tb[:, 0])
        bj = torch.from_numpy(tb[:, 1])
        keep = (bi != bj) & (bi >= 0) & (bj >= 0) & (bi < cropped_token_len) & (
            bj < cropped_token_len
        )
        bi, bj = bi[keep], bj[keep]
        token_bond_feat[bi, bj] = True
        token_bond_feat[bj, bi] = True

    # Per-token representative atom (CB, or CA when CB is absent — the pseudo-beta
    # convention) for the CB-based distogram target. Selecting exactly one atom per
    # token here lets the loss reuse the shortest-distance path: with one atom/token
    # the "shortest" token-token distance IS the CB-CB distance. Tokens with neither
    # CB nor CA (ligands / non-standard) fall back to all their atoms (shortest).
    atom_is_rep = _build_atom_is_rep(
        cifmol=cifmol,
        atom_to_token_idx_map=atom_to_token_idx_map,
        atom_pos_mask=atom_pos_mask,
        n_tokens=cropped_token_len,
    )

    return StructureFeatures.from_sample(
        atom_pos=torch.from_numpy(atom_pos.astype(np.float32)),
        atom_pos_mask=torch.from_numpy(atom_pos_mask.astype(np.bool)),
        atom_mask=torch.from_numpy(atom_mask.astype(np.bool)),
        atom_bond=torch.from_numpy(atom_bond.astype(np.int64)),
        token_contacts=token_contacts.to(torch.int64),
        token_mask=torch.ones((cropped_token_len,), dtype=torch.bool),  # all ones
        token_bond=torch.from_numpy(token_bond.astype(np.int64)),
        token_bond_feat=token_bond_feat,
        atom_is_rep=torch.from_numpy(atom_is_rep),
    )


def _build_atom_is_rep(
    cifmol: CIFMolAttached,
    atom_to_token_idx_map: np.ndarray,
    atom_pos_mask: np.ndarray,
    n_tokens: int,
) -> np.ndarray:
    """Per-atom bool: one representative atom (CB, else CA) per token.

    Tokens whose CB/CA are both missing fall back to marking ALL their atoms, so the
    downstream shortest-distance reduction reverts to all-atom for those tokens.
    """
    names = np.asarray(cifmol.atoms.id).astype(str)
    names = np.char.strip(np.char.upper(names))
    tok = np.asarray(atom_to_token_idx_map).astype(np.int64)
    valid = np.asarray(atom_pos_mask).astype(bool)
    n_atom = names.shape[0]

    cb_of = np.full(n_tokens, -1, dtype=np.int64)
    ca_of = np.full(n_tokens, -1, dtype=np.int64)

    def _assign(dst: np.ndarray, sel: np.ndarray) -> None:
        idxs = np.where(sel & valid)[0]
        t = tok[idxs]
        ok = (t >= 0) & (t < n_tokens)
        dst[t[ok]] = idxs[ok]

    _assign(cb_of, names == "CB")
    _assign(ca_of, names == "CA")
    rep_of = np.where(cb_of >= 0, cb_of, ca_of)  # (n_tokens,)

    atom_is_rep = np.zeros(n_atom, dtype=bool)
    has_rep = rep_of >= 0
    atom_is_rep[rep_of[has_rep]] = True
    # tokens with neither CB nor CA -> fall back to all their atoms (shortest distance)
    no_rep_tokens = np.where(~has_rep)[0]
    if no_rep_tokens.size:
        atom_is_rep |= np.isin(tok, no_rep_tokens)
    return atom_is_rep


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


# For steering, we give true contact/non-contact supervision from true structures.
@torch.no_grad()
def to_token_contacts(
    x0: torch.Tensor,
    atom_to_token_idx_map: torch.Tensor,
    x_mask: torch.Tensor | None = None,
    token_mask: torch.Tensor | None = None,
    token_length: int | None = None,
    positive_cutoff: float = 6.0,
    negative_cutoff: float = 12.0,
    lambda_n: int = 20,  # poisson lambda for sampling
    prob: float = 1.0,
) -> Int[torch.Tensor, "B n_token_contact 3"]:
    """Sample sparse token-level contact supervision as ``(i, j, type)`` triples.

    The input coordinates are atom-level. We first pool valid atom positions into a
    single token coordinate by averaging only valid atoms that map to that token,
    then sample contact / non-contact pairs on the token graph. Each output row is:

        ``[i, j, 0]``: definite contact
        ``[i, j, 1]``: definite non-contact

    Unknown, masked, diagonal, and unsampled pairs are omitted entirely.
    """
    if x0.ndim != 3:
        msg = f"Expected x0 to have shape (B, L_atom, 3), got {tuple(x0.shape)}."
        raise ValueError(msg)

    if atom_to_token_idx_map.ndim != 2:
        msg = (
            "Expected atom_to_token_idx_map to have shape (B, L_atom), "
            f"got {tuple(atom_to_token_idx_map.shape)}."
        )
        raise ValueError(msg)

    if atom_to_token_idx_map.shape != x0.shape[:2]:
        msg = (
            "Expected atom_to_token_idx_map to match x0 over (B, L_atom), "
            f"got {tuple(atom_to_token_idx_map.shape)} vs {tuple(x0.shape[:2])}."
        )
        raise ValueError(msg)

    B, _, _ = x0.shape
    device = x0.device

    if x_mask is None:
        x_mask = torch.isfinite(x0).all(dim=-1)
    else:
        if x_mask.shape != x0.shape[:2]:
            msg = (
                "Expected x_mask to match x0 over (B, L_atom), "
                f"got {tuple(x_mask.shape)} vs {tuple(x0.shape[:2])}."
            )
            raise ValueError(msg)
        x_mask = x_mask.bool() & torch.isfinite(x0).all(dim=-1)

    if token_mask is not None:
        token_mask = token_mask.bool()
        if token_mask.shape[0] != B:
            msg = (
                "Expected token_mask to share the batch dimension with x0, "
                f"got {tuple(token_mask.shape)} vs batch size {B}."
            )
            raise ValueError(msg)
        if token_length is None:
            token_length = int(token_mask.shape[1])

    if token_length is None:
        token_length = int(atom_to_token_idx_map.max().item()) + 1

    atom_to_token_idx_map = atom_to_token_idx_map.long().clamp(
        min=0,
        max=token_length - 1,
    )

    valid_x0 = torch.where(x_mask[..., None], x0, 0.0).to(torch.float32)
    token_sum = torch.zeros(
        (B, token_length, 3),
        device=device,
        dtype=valid_x0.dtype,
    )
    token_sum.scatter_add_(
        1,
        atom_to_token_idx_map[..., None].expand(-1, -1, 3),
        valid_x0,
    )

    token_count = torch.zeros((B, token_length), device=device, dtype=valid_x0.dtype)
    token_count.scatter_add_(1, atom_to_token_idx_map, x_mask.to(valid_x0.dtype))

    token_x0 = token_sum / token_count.clamp(min=1.0)[..., None]
    valid_token_mask = token_count > 0
    if token_mask is not None:
        valid_token_mask = valid_token_mask & token_mask
    token_x0 = torch.where(valid_token_mask[..., None], token_x0, 0.0)

    dist = torch.cdist(token_x0, token_x0)
    upper = torch.triu(
        torch.ones((token_length, token_length), dtype=torch.bool, device=device),
        diagonal=1,
    )
    valid_pair = valid_token_mask[:, :, None] & valid_token_mask[:, None, :] & upper

    contact = valid_pair & (dist < positive_cutoff)
    noncontact = valid_pair & (dist > negative_cutoff)

    Nc = contact.sum(dim=(1, 2)).clamp(min=1)
    Nn = noncontact.sum(dim=(1, 2)).clamp(min=1)

    pc = (lambda_n / Nc).clamp(max=1.0)
    pn = (lambda_n / Nn).clamp(max=1.0)

    use_cond = (torch.rand(B, device=device) < prob).float()

    mode = torch.randint(0, 3, (B,), device=device)

    use_contact = ((mode == 0) | (mode == 2)).float() * use_cond
    use_noncontact = ((mode == 1) | (mode == 2)).float() * use_cond

    # Bernoulli approximation
    pc = pc.view(B, 1, 1)
    pn = pn.view(B, 1, 1)
    use_contact = use_contact.view(B, 1, 1)
    use_noncontact = use_noncontact.view(B, 1, 1)

    contact_sample = (
        contact
        & (torch.rand((B, token_length, token_length), device=device) < pc)
        & (use_contact.bool())
    )

    noncontact_sample = (
        noncontact
        & (torch.rand((B, token_length, token_length), device=device) < pn)
        & (use_noncontact.bool())
    )

    contact_idx = torch.nonzero(contact_sample, as_tuple=False)
    noncontact_idx = torch.nonzero(noncontact_sample, as_tuple=False)

    contact_pairs = torch.cat(
        [
            contact_idx,
            torch.zeros((contact_idx.shape[0], 1), device=device, dtype=torch.long),
        ],
        dim=-1,
    )
    noncontact_pairs = torch.cat(
        [
            noncontact_idx,
            torch.ones((noncontact_idx.shape[0], 1), device=device, dtype=torch.long),
        ],
        dim=-1,
    )
    all_pairs = torch.cat([contact_pairs, noncontact_pairs], dim=0)

    if all_pairs.shape[0] == 0:
        return torch.zeros((B, 0, 3), device=device, dtype=torch.long)

    sort_idx = torch.argsort(all_pairs[:, 0])
    all_pairs = all_pairs[sort_idx]
    batch_idx = all_pairs[:, 0]
    counts = torch.bincount(batch_idx, minlength=B)
    max_pair_count = int(counts.max().item())
    batch_offsets = torch.cumsum(counts, dim=0) - counts
    local_idx = torch.arange(all_pairs.shape[0], device=device)
    local_idx = local_idx - torch.repeat_interleave(batch_offsets, counts)

    out = torch.zeros((B, max_pair_count, 3), device=device, dtype=torch.long)
    out[batch_idx, local_idx] = all_pairs[:, 1:]

    return out


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
