"""Construct a ``Batch`` from an ``InferenceSpec`` without going through ``CIFMolAttached``.

The Batch is built directly from per-chain fasta + a3m + CCD lookups. Atom
positions (the GT structure) are filled with zeros; ``atom_pos_mask`` is also
zero (no GT to compute losses against), but the model only reads the *shape*
of ``atom_pos`` plus ``atom_mask`` (which atoms exist) for sampling.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import torch

from miniworld.data.constants import AtomMapping, EntityMapping, ResidueMapping
from miniworld.data.features import Batch
from miniworld.data.features.features import (
    ChainFeatures,
    MSAFeatures,
    ReferenceFeatures,
    SchemeFeatures,
    SequenceFeatures,
    StructureFeatures,
    TemplateFeatures,
)
from miniworld.data.features.convert import to_template_features
from miniworld.data.io.load import load_template
from miniworld.data.pipeline import ComplexMSA, MSA, ProteinTemplate, sample_msa
from miniworld.utils.structure.se3 import SE3_oper

from .a3m import parse_a3m_file
from .ccd import CCDLookup, CCDResidue
from .fasta import ChainSpec, EntityType, parse_fasta_file
from .tokenization import TokenizationPolicy

if TYPE_CHECKING:
    from .spec import InferenceSpec


# Mapping fasta entity type -> EntityMapping tag letter (see constants/mapping.py).
_ENTITY_TYPE_TO_TAG: dict[EntityType, str] = {
    EntityType.PROTEIN: "P",
    EntityType.RNA: "R",
    EntityType.DNA: "D",
    EntityType.NON_POLYMER: "L",
    EntityType.BRANCHED: "B",
}

# Mapping fasta entity type -> a3m polymer view used for residue encoding.
_ENTITY_TYPE_TO_POLYMER: dict[EntityType, str] = {
    EntityType.PROTEIN: "protein",
    EntityType.RNA: "rna",
    EntityType.DNA: "dna",
}

# Per-chain "terminal" atom that the dataloader strips via
# ``miniworld.data.pipeline.utils.remove_terminal_oxygen``. CCD canonical
# templates carry these atoms on every residue, but real cifmols only have
# them on the polymer terminus, so we mirror the dataloader and drop them
# from every residue of the corresponding entity type.
_TERMINAL_ATOM_BY_ENTITY: dict[EntityType, str] = {
    EntityType.PROTEIN: "OXT",
    EntityType.RNA: "OP3",
    EntityType.DNA: "OP3",
}


@dataclass
class _ChainExpansion:
    """Per-chain quantities pre-computed before assembling the Batch."""

    spec: ChainSpec
    residues: list[CCDResidue]            # one CCDResidue per residue
    n_residues: int
    n_atoms: int
    n_tokens: int                          # token count for this chain (>= n_residues)
    atom_offset: int                       # global atom offset (start)
    token_offset: int                      # global token offset (start)
    msa: MSA                               # residue-level MSA for this chain
    atom_to_token_local: np.ndarray        # (n_atoms,) local atom -> local token [0..n_tokens-1]
    token_to_residue_local: np.ndarray     # (n_tokens,) local token -> local residue [0..n_residues-1]
    residue_token_offsets: np.ndarray      # (n_residues+1,) cum token count per residue (within chain)


def build_inference_batch(
    spec: "InferenceSpec",
    max_msa_depth: int = 256,
    missing_policy: str = "query",
    seed: int = 0,
) -> Batch:
    """Build a B=1 ``Batch`` from an ``InferenceSpec``."""
    rng = np.random.default_rng(seed)
    rm = ResidueMapping()

    chain_indices = spec.chain_indices()
    ccd_lookup = CCDLookup(spec.ccd_db)
    policy = (
        TokenizationPolicy.from_file(spec.tokenization)
        if spec.tokenization is not None
        else TokenizationPolicy()
    )

    # fasta is letter-keyed (homo-mer chains share one entry), so parse once
    # per unique letter and clone the result for each chain that uses it.
    # ``chain_letters`` is the authoritative source of per-chain letters; the
    # ``Chain:<X>`` header inside the fasta itself is informational only.
    final_chain_letter: dict[int, str] = {
        ci: spec.chain_letters[str(ci)] for ci in chain_indices
    }
    parsed_per_letter: dict[str, ChainSpec] = {}
    chain_specs: dict[int, ChainSpec] = {}
    for ci in chain_indices:
        letter = final_chain_letter[ci]
        if letter not in parsed_per_letter:
            parsed_per_letter[letter] = parse_fasta_file(
                spec.fasta[letter], chain_index=ci,
            )
        base = parsed_per_letter[letter]
        chain_specs[ci] = dataclasses.replace(
            base, chain_index=ci, chain_letter=letter,
        )
    letter_to_chains = _build_letter_to_chains(final_chain_letter, chain_indices)

    expansions: list[_ChainExpansion] = []
    atom_offset = 0
    token_offset = 0
    for ci in chain_indices:
        cs = chain_specs[ci]
        residues_full = [ccd_lookup[ccd] for ccd in cs.chemcomp_ids]
        n_res = len(residues_full)
        if n_res == 0:
            msg = f"Chain {ci} (Chain:{cs.chain_letter}) has zero residues."
            raise ValueError(msg)
        strip_atom = _TERMINAL_ATOM_BY_ENTITY.get(cs.entity_type)
        # Mirror dataloader.remove_terminal_oxygen: strip OXT (protein) /
        # OP3 (RNA/DNA) from every residue of polymer chains. Returns the
        # stripped CCDResidues plus per-residue boolean keep masks so we
        # can apply the same mask to the fragmentation atom-to-frag map.
        residues, keep_masks = _strip_terminal_atoms(residues_full, strip_atom)
        n_atoms = sum(r.n_atoms for r in residues)
        chain_msa = _load_or_build_chain_msa(spec, ci, cs, rm)
        atom_to_token_local, token_to_residue_local, residue_token_offsets = (
            _tokenize_chain(cs, residues, residues_full, keep_masks, ccd_lookup, policy)
        )
        n_tokens_chain = int(residue_token_offsets[-1])
        expansions.append(
            _ChainExpansion(
                spec=cs,
                residues=residues,
                n_residues=n_res,
                n_atoms=n_atoms,
                n_tokens=n_tokens_chain,
                atom_offset=atom_offset,
                token_offset=token_offset,
                msa=chain_msa,
                atom_to_token_local=atom_to_token_local,
                token_to_residue_local=token_to_residue_local,
                residue_token_offsets=residue_token_offsets,
            ),
        )
        atom_offset += n_atoms
        token_offset += n_tokens_chain

    total_tokens = token_offset
    total_atoms = atom_offset
    n_chains = len(expansions)

    # chain residue offsets (global residue indices)
    chain_residue_offsets = np.zeros(n_chains + 1, dtype=np.int64)
    for i, exp in enumerate(expansions):
        chain_residue_offsets[i + 1] = chain_residue_offsets[i] + exp.n_residues

    # --- Scheme ---
    chain_entity_id = _compute_entity_ids(expansions)
    chain_sym_id = _compute_sym_ids(chain_entity_id)
    token_idx = np.arange(total_tokens, dtype=np.int64)
    token_to_residue_idx_map = np.empty(total_tokens, dtype=np.int64)

    token_asym_id = np.empty(total_tokens, dtype=np.int64)
    token_entity_id_arr = np.empty(total_tokens, dtype=np.int64)
    token_sym_id_arr = np.empty(total_tokens, dtype=np.int64)
    atom_to_token_idx_map = np.empty(total_atoms, dtype=np.int64)
    atom_to_chain_id = np.empty(total_atoms, dtype=np.int64)
    atom_to_residue = np.empty(total_atoms, dtype=np.int64)

    for chain_local_idx, exp in enumerate(expansions):
        chain_res_offset = int(chain_residue_offsets[chain_local_idx])
        token_lo = exp.token_offset
        token_hi = exp.token_offset + exp.n_tokens
        token_asym_id[token_lo:token_hi] = chain_local_idx
        token_entity_id_arr[token_lo:token_hi] = chain_entity_id[chain_local_idx]
        token_sym_id_arr[token_lo:token_hi] = chain_sym_id[chain_local_idx]
        token_to_residue_idx_map[token_lo:token_hi] = (
            exp.token_to_residue_local + chain_res_offset
        )

        atom_lo = exp.atom_offset
        atom_hi = atom_lo + exp.n_atoms
        atom_to_token_idx_map[atom_lo:atom_hi] = exp.atom_to_token_local + exp.token_offset
        atom_to_chain_id[atom_lo:atom_hi] = chain_local_idx

        atom_cursor = atom_lo
        for res_local_idx, res in enumerate(exp.residues):
            global_residue = chain_res_offset + res_local_idx
            atom_to_residue[atom_cursor:atom_cursor + res.n_atoms] = global_residue
            atom_cursor += res.n_atoms

    # ``atom_to_chain_id`` doubles as the solver's ``atom_to_combine`` argument
    # (per-chain SE(3) frame). When the user wires up ``combine_groups``, remap
    # chain ids to group ids so multiple chains share one rigid frame; the
    # per-chain ``token_asym_id`` is left untouched, so chain-aware model
    # embeddings and the to_cif chain assignment are unaffected.
    if spec.combine_groups:
        chain_to_group = _build_chain_to_group(spec.combine_groups, n_chains)
        atom_to_chain_id = chain_to_group[atom_to_chain_id]

    scheme = SchemeFeatures.from_sample(
        token_residue_idx=torch.from_numpy(token_to_residue_idx_map),
        token_idx=torch.from_numpy(token_idx),
        token_asym_id=torch.from_numpy(token_asym_id),
        token_entity_id=torch.from_numpy(token_entity_id_arr),
        token_sym_id=torch.from_numpy(token_sym_id_arr),
        atom_to_token_idx_map=torch.from_numpy(atom_to_token_idx_map),
        atom_to_chain_id=torch.from_numpy(atom_to_chain_id),
    )

    # --- Reference (CCD canonical coords + per-residue random SE(3)) ---
    ref_pos, ref_element, ref_charge = _build_reference_arrays(expansions, rng)
    reference = ReferenceFeatures.from_sample(
        pos=torch.from_numpy(ref_pos.astype(np.float32)),
        mask=torch.ones(total_atoms, dtype=torch.bool),
        element=torch.from_numpy(ref_element.astype(np.int64)),
        charge=torch.from_numpy(ref_charge.astype(np.float32)),
        space_uid=torch.from_numpy(atom_to_residue.astype(np.int64)),
    )

    # --- MSA: residue-level features expanded to per-token via token->residue map ---
    complex_msa = ComplexMSA(
        MSAs=[exp.msa for exp in expansions],
        missing_policy=missing_policy,
    )
    msa_residue = sample_msa(complex_msa, max_msa_depth=max_msa_depth, rng=rng)
    msa_features = MSAFeatures(
        aligned_sequences=msa_residue.aligned_sequences[
            :, :, token_to_residue_idx_map,
        ],
        mask=msa_residue.mask,
        has_deletion=msa_residue.has_deletion[:, :, token_to_residue_idx_map],
        deletion_value=msa_residue.deletion_value[:, :, token_to_residue_idx_map],
        profile=msa_residue.profile[:, token_to_residue_idx_map, :],
        deletion_mean=msa_residue.deletion_mean[:, token_to_residue_idx_map],
    )

    # --- Sequence (token_type from MSA query row) ---
    sequence = SequenceFeatures(token_type=msa_features.aligned_sequences[:, 0])

    # --- Structure ---
    token_bond = _build_token_bonds(expansions)
    contacts = spec.contacts
    if spec.template_as_contact and spec.complex_templates:
        from .complex_template import derive_contacts_from_complex_templates
        from .spec import ContactsSpec

        extra_positive = derive_contacts_from_complex_templates(spec)
        if extra_positive:
            # Dedupe while preserving the user's explicit contacts first.
            merged = list(dict.fromkeys([*contacts.positive, *extra_positive]))
            contacts = ContactsSpec(positive=merged, negative=contacts.negative)
    token_contacts = _build_token_contacts(contacts, letter_to_chains, expansions)
    # ``atom_pos_mask`` marks atoms whose positions should be denoised by the
    # diffusion solver and emitted to the CIF output. For inference we want
    # every atom predicted, so set it to all-True (no GT, but all valid).
    structure = StructureFeatures.from_sample(
        atom_pos=torch.zeros(total_atoms, 3, dtype=torch.float32),
        atom_pos_mask=torch.ones(total_atoms, dtype=torch.bool),
        atom_mask=torch.ones(total_atoms, dtype=torch.bool),
        atom_bond=torch.zeros(0, 6, dtype=torch.long),
        token_contacts=token_contacts,
        token_mask=torch.ones(total_tokens, dtype=torch.bool),
        token_bond=torch.from_numpy(token_bond.astype(np.int64)),
    )

    # --- Templates ---
    template = _build_template_features(
        spec=spec,
        expansions=expansions,
        token_to_residue_idx_map=token_to_residue_idx_map,
        rng=rng,
    )
    if template is None:
        template = TemplateFeatures.from_sample(
            mask=torch.zeros(1, dtype=torch.bool),
            ids=torch.zeros(1, total_tokens, dtype=torch.long),
            res_type=torch.zeros(1, total_tokens, dtype=torch.long),
            cb_xyz=torch.zeros(1, total_tokens, 3, dtype=torch.float32),
            cb_mask=torch.zeros(1, total_tokens, dtype=torch.bool),
            bb_xyz=torch.zeros(1, total_tokens, 3, 3, dtype=torch.float32),
            bb_mask=torch.zeros(1, total_tokens, dtype=torch.bool),
        )

    # --- Chain ---
    entity_mapping = EntityMapping()
    entity_tags = [_ENTITY_TYPE_TO_TAG[exp.spec.entity_type] for exp in expansions]
    entity_type_arr = entity_mapping.tag_to_idx(entity_tags).astype(np.int64)
    chain = ChainFeatures.from_sample(entity_type=torch.from_numpy(entity_type_arr))

    # --- Metadata for CIF output ---
    # ``chem_comp_ids`` and ``heteros`` are stored per-residue (indexed via
    # ``space_uid`` in to_cif.py), so we keep length = total_residues.
    total_residues = int(chain_residue_offsets[-1])
    atom_ids = np.empty(total_atoms, dtype=object)
    chem_comp_ids = np.empty(total_residues, dtype=object)
    hetero_per_residue = np.empty(total_residues, dtype=np.int64)
    for chain_local_idx, exp in enumerate(expansions):
        chain_res_offset = int(chain_residue_offsets[chain_local_idx])
        is_hetero = exp.spec.entity_type in (EntityType.NON_POLYMER, EntityType.BRANCHED)
        atom_cursor = exp.atom_offset
        for res_local_idx, res in enumerate(exp.residues):
            global_residue = chain_res_offset + res_local_idx
            chem_comp_ids[global_residue] = res.chemcomp_id
            hetero_per_residue[global_residue] = 1 if is_hetero else 0
            for atom_id in res.atom_ids:
                atom_ids[atom_cursor] = str(atom_id)
                atom_cursor += 1

    name = spec.name or "inference"
    return Batch(
        name=[name],
        heteros=[SimpleNamespace(value=hetero_per_residue)],
        atom_ids=[atom_ids],
        chem_comp_ids=[chem_comp_ids],
        sequence=sequence,
        structure=structure,
        msa=msa_features,
        template=template,
        reference=reference,
        scheme=scheme,
        chain=chain,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_chain_refs(
    ref: str,
    letter_to_chains: dict[str, list[int]],
    n_chains: int,
    *,
    where: str,
) -> list[int]:
    """Resolve a single chain reference (numeric index or letter) to chain local indices.

    Numeric references map to a single chain (validated against ``n_chains``).
    Letter references can expand to multiple chains when the letter is shared
    by several chains via :attr:`InferenceSpec.chain_letters`.
    """
    s = ref.strip()
    if s.lstrip("-").isdigit():
        ci = int(s)
        if not (0 <= ci < n_chains):
            msg = f"{where}: chain index {ci} out of range [0, {n_chains})."
            raise IndexError(msg)
        return [ci]
    if s not in letter_to_chains:
        msg = (
            f"{where}: unknown chain reference {ref!r}. "
            f"Known letters: {sorted(letter_to_chains)}, "
            f"valid indices: 0..{n_chains - 1}."
        )
        raise KeyError(msg)
    return list(letter_to_chains[s])


def _build_chain_to_group(
    combine_groups: list[list[int]],
    n_chains: int,
) -> np.ndarray:
    """Map per-chain index to a combine-group id.

    Each entry in ``combine_groups`` is a list of **numeric chain indices**
    (0-based). Explicit groups become ids 0..K-1 in order; chains not
    listed each get their own singleton group with ids K, K+1, ... in
    chain-local order.
    """
    chain_to_group = np.full(n_chains, -1, dtype=np.int64)
    for group_idx, indices in enumerate(combine_groups):
        for ci in indices:
            if not (0 <= ci < n_chains):
                msg = (
                    f"combine_groups[{group_idx}]: chain index {ci} out of "
                    f"range [0, {n_chains})."
                )
                raise IndexError(msg)
            if chain_to_group[ci] != -1:
                msg = (
                    f"chain index {ci} appears in multiple combine_groups "
                    f"(in group {int(chain_to_group[ci])} and {group_idx})."
                )
                raise ValueError(msg)
            chain_to_group[ci] = group_idx
    next_id = len(combine_groups)
    for ci in range(n_chains):
        if chain_to_group[ci] == -1:
            chain_to_group[ci] = next_id
            next_id += 1
    return chain_to_group


def _build_letter_to_chains(
    final_chain_letter: dict[int, str],
    chain_indices: list[int],
) -> dict[str, list[int]]:
    """Inverse map ``letter -> [chain_local_idx, ...]`` (1-to-many).

    Letters may repeat across chains (homo-mer copies). The order of
    chain indices in each list follows ``chain_indices`` order.
    """
    letter_to_chains: dict[str, list[int]] = {}
    for local_idx, ci in enumerate(chain_indices):
        letter = final_chain_letter[ci]
        letter_to_chains.setdefault(letter, []).append(local_idx)
    return letter_to_chains


def _load_or_build_chain_msa(
    spec: "InferenceSpec",
    chain_index: int,
    cs: ChainSpec,
    rm: ResidueMapping,
) -> MSA:
    """Load the chain's a3m if provided, otherwise build a query-only MSA.

    a3m is letter-keyed in the spec, so chains sharing a letter (homo-mers)
    resolve to the same a3m file.
    """
    letter = spec.chain_letters[str(chain_index)]
    a3m_path = spec.a3m.get(letter)
    polymer_kind = _ENTITY_TYPE_TO_POLYMER.get(cs.entity_type)
    if a3m_path is not None and polymer_kind is not None:
        return parse_a3m_file(Path(a3m_path), polymer=polymer_kind)
    return MSA.from_query(
        query_sequence=_encode_query_only(cs, rm),
        seq_id=cs.fasta_id,
    )


def _encode_query_only(cs: ChainSpec, rm: ResidueMapping) -> np.ndarray:
    """Encode the chain's query sequence to integer tokens."""
    if cs.entity_type == EntityType.PROTEIN:
        return rm.protein.map(cs.one_letter_seq).astype(np.int32)
    if cs.entity_type == EntityType.RNA:
        return rm.rna.map(cs.one_letter_seq).astype(np.int32)
    if cs.entity_type == EntityType.DNA:
        return rm.dna.map(cs.one_letter_seq).astype(np.int32)
    return np.full(len(cs.chemcomp_ids), rm.LIGAND_INDEX, dtype=np.int32)


def _compute_entity_ids(expansions: list[_ChainExpansion]) -> np.ndarray:
    """Group chains by (entity_type, chemcomp tuple) and assign unique entity ids."""
    sig_to_id: dict[tuple, int] = {}
    out = np.empty(len(expansions), dtype=np.int64)
    next_id = 0
    for i, exp in enumerate(expansions):
        sig = (exp.spec.entity_type.value, tuple(exp.spec.chemcomp_ids))
        if sig not in sig_to_id:
            sig_to_id[sig] = next_id
            next_id += 1
        out[i] = sig_to_id[sig]
    return out


def _compute_sym_ids(chain_entity_id: np.ndarray) -> np.ndarray:
    """For each chain, count of preceding chains with the same entity_id."""
    same = chain_entity_id[:, None] == chain_entity_id[None, :]
    return np.triu(same, k=0).sum(axis=0) - 1


def _build_reference_arrays(
    expansions: list[_ChainExpansion],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate per-residue CCD reference coords, applying random SE(3) per residue.

    Mirrors ``to_reference_features`` (convert.py) for the standard SE(3)
    randomization that the model expects on the reference frame.
    """
    total_atoms = sum(exp.n_atoms for exp in expansions)
    ref_pos = np.zeros((total_atoms, 3), dtype=np.float32)
    ref_element_str = np.empty(total_atoms, dtype=object)
    ref_charge = np.zeros(total_atoms, dtype=np.float32)

    n_residues = sum(exp.n_residues for exp in expansions)
    Rs, Ts = SE3_oper(n_residues, rng=rng)

    res_global = 0
    atom_cursor = 0
    for exp in expansions:
        for res in exp.residues:
            R, T = Rs[res_global], Ts[res_global]
            xyz = res.atom_xyz.astype(np.float32, copy=False)
            xyz_centered = xyz - xyz.mean(axis=0, keepdims=True)
            ref_pos[atom_cursor:atom_cursor + res.n_atoms] = xyz_centered @ R + T
            ref_element_str[atom_cursor:atom_cursor + res.n_atoms] = res.atom_elements
            ref_charge[atom_cursor:atom_cursor + res.n_atoms] = res.atom_charges
            atom_cursor += res.n_atoms
            res_global += 1

    ref_element = AtomMapping().atom_to_index(ref_element_str.tolist())
    return ref_pos, ref_element, ref_charge


# Entity types that the StructCooker template DB indexes; non-polymer / branched
# / antibody-NA chains contribute empty slots since template_mols don't apply.
_TEMPLATE_ELIGIBLE_ENTITIES: set[EntityType] = {
    EntityType.PROTEIN,
    EntityType.RNA,
    EntityType.DNA,
}


def _build_single_chain_template_layers(
    spec: "InferenceSpec",
    expansions: list[_ChainExpansion],
    rng: np.random.Generator,
) -> list[ProteinTemplate]:
    """Per-chain single-chain templates loaded from ``spec.template_db``.

    Always returns ``len(expansions)`` ``ProteinTemplate`` objects (empty
    slots for chains without a configured seq_id). The slot count of each
    output equals the number of valid templates loaded for that chain
    (0..spec.template_n).
    """
    out: list[ProteinTemplate] = []
    if spec.template_db is None or not spec.template:
        return [ProteinTemplate.empty(exp.n_residues) for exp in expansions]

    for chain_local_idx, exp in enumerate(expansions):
        chain_idx_str = str(exp.spec.chain_index)
        seq_id = spec.template.get(chain_idx_str)
        if (
            seq_id is None
            or exp.spec.entity_type not in _TEMPLATE_ELIGIBLE_ENTITIES
        ):
            out.append(ProteinTemplate.empty(exp.n_residues))
            continue
        loaded = load_template(
            seq_id=seq_id,
            template_id=chain_local_idx,
            env_path=spec.template_db,
            crop_indices=None,  # no crop in inference
            n_templates=spec.template_n,
            rng=rng,
        )
        # ``load_template`` returns a template at the cluster's canonical
        # length; on length mismatch fall back to an empty slot so the rest
        # of the batch still builds.
        if loaded.res_num != exp.n_residues:
            out.append(ProteinTemplate.empty(exp.n_residues))
            continue
        out.append(loaded)
    return out


def _build_template_features(
    spec: "InferenceSpec",
    expansions: list[_ChainExpansion],
    token_to_residue_idx_map: np.ndarray,
    rng: np.random.Generator,
) -> TemplateFeatures | None:
    """Load templates (single-chain + complex) and project to tokens.

    Single-chain templates come from ``spec.template_db`` keyed by
    ``spec.template[chain_idx]`` (one slot per template, multiple chains
    contribute independent slots). Complex templates from
    ``spec.complex_templates`` add one shared slot across the chains they
    cover, with NaN coords / mask=False for non-participating chains. The
    two are stacked along the slot axis per chain, then concatenated along
    the residue axis to produce one global ``ProteinTemplate``.
    """
    from .complex_template import load_complex_template_layers, stack_slots

    if (
        (spec.template_db is None or not spec.template)
        and not spec.complex_templates
    ):
        return None

    single_chain_layers = _build_single_chain_template_layers(spec, expansions, rng)
    complex_layers = load_complex_template_layers(spec, expansions)

    per_chain_combined: list[ProteinTemplate] = []
    for ci, exp in enumerate(expansions):
        layers: list[ProteinTemplate] = []
        if single_chain_layers[ci].slot_num > 0:
            layers.append(single_chain_layers[ci])
        if complex_layers[ci].slot_num > 0:
            layers.append(complex_layers[ci])
        if not layers:
            per_chain_combined.append(ProteinTemplate.empty(exp.n_residues))
        elif len(layers) == 1:
            per_chain_combined.append(layers[0])
        else:
            per_chain_combined.append(stack_slots(layers))

    merged = ProteinTemplate.concat(per_chain_combined)
    if merged.slot_num == 0:
        return None
    return to_template_features(merged, token_to_residue_idx_map)


def _strip_terminal_atoms(
    residues_full: list[CCDResidue],
    strip_atom: str | None,
) -> tuple[list[CCDResidue], list[np.ndarray]]:
    """Apply ``remove_terminal_oxygen``-equivalent stripping to each CCDResidue.

    Returns the stripped residues plus per-residue boolean ``keep`` masks
    aligned with the original CCD atom order. When ``strip_atom`` is ``None``
    (ligand / branched chain), the masks are all-True and residues are returned
    unchanged.
    """
    if strip_atom is None:
        masks = [np.ones(r.n_atoms, dtype=bool) for r in residues_full]
        return list(residues_full), masks

    stripped: list[CCDResidue] = []
    masks: list[np.ndarray] = []
    for r in residues_full:
        keep = np.array([str(a) != strip_atom for a in r.atom_ids], dtype=bool)
        masks.append(keep)
        if keep.all():
            stripped.append(r)
            continue
        stripped.append(
            CCDResidue(
                chemcomp_id=r.chemcomp_id,
                atom_ids=r.atom_ids[keep],
                atom_elements=r.atom_elements[keep],
                atom_charges=r.atom_charges[keep],
                atom_xyz=r.atom_xyz[keep],
            ),
        )
    return stripped, masks


def _tokenize_chain(
    cs: ChainSpec,
    residues: list[CCDResidue],
    residues_full: list[CCDResidue],
    keep_masks: list[np.ndarray],
    ccd_lookup: CCDLookup,
    policy: TokenizationPolicy,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (atom_to_token_local, token_to_residue_local, residue_token_offsets).

    Per-residue resolution from ``policy``. Resolution=1 keeps the residue as
    one token; 0 atomizes it; intermediate values pick a fragment merge level
    via ``CCDLookup.fragments`` (mirroring ``dynamic_tokenize``). Fragments are
    looked up using the *full* CCD atom set, then projected through the keep
    mask so the resulting local indices align with the *stripped* atom set.
    """
    n_atoms_chain = sum(r.n_atoms for r in residues)
    atom_to_token_local = np.empty(n_atoms_chain, dtype=np.int64)
    token_to_residue_local: list[int] = []
    residue_token_offsets: list[int] = [0]

    atom_cursor = 0
    for r_local, (residue, residue_full, keep) in enumerate(
        zip(residues, residues_full, keep_masks),
    ):
        res_1based = r_local + 1
        resolution = policy.resolution(cs.chain_letter, res_1based)
        atom_to_frag_local, n_frags = _residue_token_assignment(
            residue=residue,
            residue_full=residue_full,
            keep=keep,
            resolution=resolution,
            ccd_lookup=ccd_lookup,
        )
        atom_to_token_local[atom_cursor:atom_cursor + residue.n_atoms] = (
            atom_to_frag_local + residue_token_offsets[-1]
        )
        token_to_residue_local.extend([r_local] * n_frags)
        residue_token_offsets.append(residue_token_offsets[-1] + n_frags)
        atom_cursor += residue.n_atoms

    return (
        atom_to_token_local,
        np.asarray(token_to_residue_local, dtype=np.int64),
        np.asarray(residue_token_offsets, dtype=np.int64),
    )


def _residue_token_assignment(
    residue: CCDResidue,
    residue_full: CCDResidue,
    keep: np.ndarray,
    resolution: float,
    ccd_lookup: CCDLookup,
) -> tuple[np.ndarray, int]:
    """Return ``(atom_to_frag_local, n_frags)`` for a single residue.

    ``atom_to_frag_local[i]`` is the fragment (= token) index within this
    residue for the i-th *stripped* atom (length == ``residue.n_atoms``).
    """
    if resolution >= 1.0 - 1e-9:
        return np.zeros(residue.n_atoms, dtype=np.int64), 1
    fragments = ccd_lookup.fragments(residue.chemcomp_id)
    available = sorted(fragments.keys())
    idx = max(0, min(round(resolution * (len(available) - 1)), len(available) - 1))
    merge_val = available[idx]
    frag_mol = fragments[merge_val]
    local_frag_full = np.asarray(frag_mol.index_table.atom_to_res, dtype=np.int64)
    n_frags = len(frag_mol.residues)
    if local_frag_full.shape[0] != residue_full.n_atoms:
        # Atom-count mismatch CCD-template vs the residue we built; fall back
        # to one token per residue (mirrors dataloader's safety branch).
        return np.zeros(residue.n_atoms, dtype=np.int64), 1
    # Project the per-CCD-atom fragment indices through the keep mask, then
    # compact any fragments that lost all of their atoms.
    local_frag = local_frag_full[keep]
    unique = np.unique(local_frag)
    if unique.shape[0] != n_frags:
        remap = np.full(n_frags, -1, dtype=np.int64)
        remap[unique] = np.arange(unique.shape[0], dtype=np.int64)
        local_frag = remap[local_frag]
        n_frags = int(unique.shape[0])
    return local_frag, n_frags


def _build_token_bonds(expansions: list[_ChainExpansion]) -> np.ndarray:
    """Token-level inter-residue bonds (branched / explicit covalent edges).

    Each branched bond is between two residues; we anchor the token-level
    edge at each residue's *first* token (``residue_token_offsets[r]``).
    """
    pairs: list[tuple[int, int]] = []
    for exp in expansions:
        for a, b in exp.spec.branched_bonds:
            ga = exp.token_offset + int(exp.residue_token_offsets[a])
            gb = exp.token_offset + int(exp.residue_token_offsets[b])
            pairs.append((min(ga, gb), max(ga, gb)))
    if not pairs:
        # Match convert.py's "to avoid empty bond, add [0,0]" placeholder.
        return np.array([[0, 0]], dtype=np.int64)
    return np.asarray(pairs, dtype=np.int64)


def _build_token_contacts(
    contacts,
    letter_to_chains: dict[str, list[int]],
    expansions: list[_ChainExpansion],
) -> torch.Tensor:
    """Translate user contacts into ``(n, 3)`` global token triples.

    Each pair is ``"<chain_ref_a>:<res_a>-<chain_ref_b>:<res_b>"`` where
    ``chain_ref`` is either a letter (resolved through ``letter_to_chains``;
    duplicates expand the contact via Cartesian product) or a numeric chain
    index. Residue indices are 1-based.

    Output rows: ``[token_i_global, token_j_global, type]`` with type
    0=contact, 1=non-contact. Leading batch dim added by
    ``StructureFeatures.from_sample``.
    """
    triples: set[tuple[int, int, int]] = set()
    n_chains = len(expansions)
    n_residues_per_chain = [exp.n_residues for exp in expansions]
    for kind, pairs in (("positive", contacts.positive), ("negative", contacts.negative)):
        type_id = 0 if kind == "positive" else 1
        for s in pairs:
            chain_a, res_a, chain_b, res_b = contacts.parse_pair(s)
            chains_a = _resolve_chain_refs(
                chain_a, letter_to_chains, n_chains, where=f"contact {s!r}",
            )
            chains_b = _resolve_chain_refs(
                chain_b, letter_to_chains, n_chains, where=f"contact {s!r}",
            )
            for ci_a in chains_a:
                ti = _resolve_global_token(ci_a, res_a, expansions, n_residues_per_chain, s)
                for ci_b in chains_b:
                    tj = _resolve_global_token(ci_b, res_b, expansions, n_residues_per_chain, s)
                    if ti == tj:
                        # Skip rather than raise — Cartesian expansion may
                        # legitimately hit the same token (e.g. chain ref
                        # letter expands to one chain on both sides at the
                        # same residue index).
                        continue
                    a, b = (ti, tj) if ti < tj else (tj, ti)
                    triples.add((a, b, type_id))

    if not triples:
        return torch.zeros(0, 3, dtype=torch.long)
    rows = sorted(triples)
    return torch.tensor(rows, dtype=torch.long)


def _resolve_global_token(
    chain_local: int,
    res_1based: int,
    expansions: list[_ChainExpansion],
    n_residues_per_chain: list[int],
    raw: str,
) -> int:
    """Map ``(chain_local_idx, 1-based residue idx)`` to the global token id.

    ``chain_local`` is already a numeric chain index (letter-to-chain
    resolution happens in the caller).
    """
    if not (1 <= res_1based <= n_residues_per_chain[chain_local]):
        msg = (
            f"Contact {raw!r}: residue index {res_1based} out of range for "
            f"chain index {chain_local} (length {n_residues_per_chain[chain_local]})."
        )
        raise ValueError(msg)
    exp = expansions[chain_local]
    # Anchor the contact at the residue's first token.
    return exp.token_offset + int(exp.residue_token_offsets[res_1based - 1])
