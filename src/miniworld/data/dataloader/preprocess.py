"""Per-item preprocessing pipeline: DataRecord → Batch.

The Preprocessor bundles the tokenizer + CCD cache + per-config knobs that a
single sample needs, so BioMolData.__getitem__ can delegate the CIF load →
crop → MSA/template load → batch build chain in one call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from miniworld.data.features import Batch, make_batch
from miniworld.data.io import load_cifmol
from miniworld.data.pipeline import (
    get_chain_crop_indices,
    sample_msa,
)
from miniworld.data.pipeline.utils import (
    NoInterfaceError,
    find_interface_residues,
    remove_terminal_oxygen,
)
from miniworld.utils.crop import crop_spatial_segment_token

from .loading import load_record_msa, load_record_templates
from .types import DataRecord, ResourceLocator

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from miniworld.configs.data import (
        BioMolDBConfig,
        CropConfig,
        MSAConfig,
        TemplateConfig,
        TokenizerConfig,
    )
    from miniworld.data.mols import CIFMolAttached, FragmentedCCDMol
    from miniworld.data.pipeline import Tokenizer


class WrongCroppingError(ValueError):
    """Raised when the cropping strategy fails to produce a valid crop."""


class Preprocessor:
    """DataRecord → Batch pipeline.

    Groups the tokenizer, CCD cache, and per-config knobs so BioMolData just
    delegates ``__getitem__`` to :meth:`process`. Split out from BioMolData to
    keep the Dataset class focused on catalog/sampler responsibilities.
    """

    def __init__(
        self,
        *,
        tokenizer: Tokenizer,
        fragmented_ccd_mols: Mapping[str, dict[int, FragmentedCCDMol]],
        pdb_config: BioMolDBConfig | None,
        crop_config: CropConfig,
        msa_config: MSAConfig,
        template_config: TemplateConfig,
        tokenizer_config: TokenizerConfig,
        resources: ResourceLocator | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.fragmented_ccd_mols = fragmented_ccd_mols
        self.pdb_config = pdb_config
        self.crop_config = crop_config
        self.msa_config = msa_config
        self.template_config = template_config
        self.tokenizer_config = tokenizer_config
        # Unified location authority (train_item mode); resolves the exact msa
        # shard by seq_id at runtime instead of scanning the shard list.
        self.resources = resources

    # -- top-level entry --------------------------------------------------

    def process(
        self,
        record: DataRecord,
        *,
        crop_indices: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> Batch:
        """Full pipeline: CIF → crop → MSA → templates → batch."""
        if rng is None:
            rng = np.random.default_rng()
        cifmol = load_cifmol(
            db_path=record.cif_db_path,
            pdb_id=record.record_id,
            assembly_id=record.assembly_id,
            model_id=record.model_id,
            alt_id=record.alt_id,
        )
        record = self._resolve_feature_keys(record)

        if crop_indices is None:
            (
                crop_indices,
                chain_id_to_crop_indices,
                atom_to_token_idx_map,
                token_to_residue_idx_map,
                _focus,
            ) = self.get_crop_indices(
                cifmol=cifmol,
                chain_ids=list(record.chain_ids),
                max_tokens=self.crop_config.max_tokens,
                max_atoms=self.crop_config.max_atoms,
                rng=rng,
            )
            if crop_indices.shape[0] == 0:
                msg = (
                    f"Failed to crop {record.record_id}_"
                    f"{record.assembly_id}_{record.model_id}_{record.alt_id}."
                )
                raise WrongCroppingError(msg)
        else:
            chain_id_to_crop_indices = get_chain_crop_indices(
                cifmol=cifmol,
                crop_indices=crop_indices,
            )
            valid_xyz = cifmol.atoms.xyz.value
            valid_mask = np.isfinite(valid_xyz).all(axis=1)
            focus = (
                valid_xyz[valid_mask][rng.integers(0, valid_mask.sum())]
                if valid_mask.any()
                else np.zeros(3)
            )
            atom_to_token_idx_map, token_to_residue_idx_map = self.tokenizer.tokenize(
                cifmol,
                focus=focus,
                fragmented_ccd_mols=self.fragmented_ccd_mols,
                config=self.tokenizer_config.dynamic_config,
            )
        record = self._resolve_feature_keys(
            record,
            tuple(chain_id_to_crop_indices),
        )

        cifmol = cifmol.residues[crop_indices].extract()
        atom_mask = remove_terminal_oxygen(cifmol)
        cifmol = cifmol.atoms[atom_mask].extract()
        atom_to_token_idx_map = atom_to_token_idx_map[atom_mask]

        # Removing atoms above (remove_terminal_oxygen strips OXT/OP3) can orphan a
        # token: modified/non-canonical residues are tokenized per-atom, so the
        # removed OXT had its own token, which would otherwise survive as a valid
        # (token_mask=True) token with zero atoms. Drop atomless tokens and renumber
        # so every token keeps >=1 atom and the token/atom maps stay consistent.
        n_tokens = int(token_to_residue_idx_map.shape[0])
        token_has_atom = np.zeros(n_tokens, dtype=bool)
        token_has_atom[atom_to_token_idx_map] = True
        if not token_has_atom.all():
            remap = np.cumsum(token_has_atom) - 1  # old token idx -> compacted idx
            atom_to_token_idx_map = remap[atom_to_token_idx_map]
            token_to_residue_idx_map = token_to_residue_idx_map[token_has_atom]

        complex_msa = load_record_msa(
            cifmol=cifmol,
            chain_id_to_crop_indices=chain_id_to_crop_indices,
            record=record,
            missing_policy=self.msa_config.missing_policy,
            pairing_mode=self.msa_config.pairing_mode,
            locator=self.resources,
        )
        msa = sample_msa(
            msa=complex_msa,
            max_msa_depth=self.msa_config.max_msa_depth,
            rng=rng,
            sample_depth=self.msa_config.sample_depth,
        )

        templates = load_record_templates(
            cifmol=cifmol,
            chain_id_to_crop_indices=chain_id_to_crop_indices,
            record=record,
            n_templates=self.template_config.n_templates,
            rng=rng,
        )

        return make_batch(
            cifmol=cifmol,
            msa=msa,
            templates=templates,
            atom_to_token_idx_map=atom_to_token_idx_map,
            token_to_residue_idx_map=token_to_residue_idx_map,
            rng=rng,
        )

    # -- feature-key resolution (compat records) --------------------------

    def _resolve_feature_keys(
        self,
        record: DataRecord,
        chain_ids: Sequence[str] | None = None,
    ) -> DataRecord:
        """Align per-chain MSA/template LMDB paths to the resolved chain set.

        The lookup KEYS are read from the CIF at load time (msa←seq_id,
        template←``{RECORD}_{auth}``), so this only realigns the db PATHS when the
        chain set changes — a compat pdb record (empty paths → filled from
        ``pdb_config``) or a crop dropping a chain from an interface item.
        """
        resolved = tuple(chain_ids) if chain_ids is not None else record.chain_ids
        if record.chain_ids == resolved and len(record.msa_db_paths) == len(resolved):
            return record

        if record.source == "pdb" and self.pdb_config is not None:
            pdb = self.pdb_config
            msa_by: tuple[tuple[Path, ...], ...] = ((pdb.a3m_db_path,),) * len(resolved)
            tmpl_by: tuple[Path | None, ...] = (pdb.template_db_path,) * len(resolved)
        else:
            # Map each resolved chain back to its original per-chain dbs; every
            # chain of one distillation/train_item source shares the same shards,
            # while manifest complexes keep per-chain paths.
            index_of = {c: i for i, c in enumerate(record.chain_ids)}
            base_msa = record.msa_db_paths[0] if record.msa_db_paths else ()
            base_tmpl = (
                record.template_db_paths[0] if record.template_db_paths else None
            )

            def msa_of(chain_id: str) -> tuple[Path, ...]:
                i = index_of.get(chain_id)
                if i is not None and i < len(record.msa_db_paths):
                    return record.msa_db_paths[i]
                return base_msa

            def tmpl_of(chain_id: str) -> Path | None:
                i = index_of.get(chain_id)
                if i is not None and i < len(record.template_db_paths):
                    return record.template_db_paths[i]
                return base_tmpl

            msa_by = tuple(msa_of(c) for c in resolved)
            tmpl_by = tuple(tmpl_of(c) for c in resolved)

        return DataRecord(
            item_id=record.item_id,
            source=record.source,
            record_id=record.record_id,
            cif_db_path=record.cif_db_path,
            assembly_id=record.assembly_id,
            model_id=record.model_id,
            alt_id=record.alt_id,
            chain_ids=resolved,
            feature_keys=(),
            seq_ids=(),
            msa_db_paths=msa_by,
            template_db_paths=tmpl_by,
            weight=record.weight,
            item_kind=record.item_kind,
            weight_group=record.weight_group,
        )

    # -- cropping ---------------------------------------------------------

    def get_crop_indices(
        self,
        cifmol: CIFMolAttached,
        chain_ids: list[str],
        max_tokens: int,
        max_atoms: int,
        rng: np.random.Generator,
    ) -> tuple[
        np.ndarray,
        dict[str, np.ndarray],
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Spatial-segment cropping around a focus atom, with tokenization."""
        selected_atoms = self._select_focus_atoms(cifmol, chain_ids, rng)
        valid = np.all(np.isfinite(selected_atoms.xyz), axis=-1)
        selected_atoms = selected_atoms[valid]
        if len(selected_atoms) == 0:
            msg = f"No valid atoms found for chain_ids {chain_ids} in cifmol {cifmol.id}"
            raise WrongCroppingError(msg)
        focus = selected_atoms.xyz[rng.integers(0, len(selected_atoms))].value

        segment_size = int(
            rng.integers(
                self.crop_config.min_segment_size,
                self.crop_config.max_segment_size + 1,
            ),
        )

        atom_to_token_idx_map, token_to_residue_idx_map = self.tokenizer.tokenize(
            cifmol,
            focus=focus,
            fragmented_ccd_mols=self.fragmented_ccd_mols,
            config=self.tokenizer_config.dynamic_config,
        )

        crop_indices = crop_spatial_segment_token(
            cifmol,
            focus,
            tokens_to_res=token_to_residue_idx_map,
            segment_size=segment_size,
            max_tokens=max_tokens,
            max_atoms=max_atoms,
        )

        chain_id_to_crop_indices = get_chain_crop_indices(
            cifmol=cifmol,
            crop_indices=crop_indices,
        )

        crop_indices = cast("np.ndarray", crop_indices)

        if crop_indices.shape[0] == 0:
            msg = f"Failed to crop {cifmol.id} with chain_ids {chain_ids}."
            raise WrongCroppingError(msg)

        cropped_atom, cropped_tok = self._reindex_after_crop(
            crop_indices=crop_indices,
            atom_to_token_idx_map=atom_to_token_idx_map,
            token_to_residue_idx_map=token_to_residue_idx_map,
        )

        return (
            crop_indices,
            chain_id_to_crop_indices,
            cropped_atom,
            cropped_tok,
            focus,
        )  # pyright: ignore[reportPossiblyUnboundVariable]

    def _select_focus_atoms(
        self,
        cifmol: CIFMolAttached,
        chain_ids: list[str],
        rng: np.random.Generator,
    ):
        """Pick candidate atoms for cropping focus (interface for pairs)."""
        match chain_ids:
            case [chain_id]:
                return cifmol.chains.select(chain_id=chain_id).atoms
            case [chain_id1, chain_id2]:
                if rng.random() < self.crop_config.chain_crop_prob:
                    chain_id = rng.choice([chain_id1, chain_id2])
                    return cifmol.chains.select(chain_id=chain_id).atoms
                try:
                    return find_interface_residues(cifmol, chain_id1, chain_id2).atoms
                except NoInterfaceError:
                    chain_id = rng.choice([chain_id1, chain_id2])
                    return cifmol.chains.select(chain_id=chain_id).atoms
            case _:
                msg = f"Unexpected chain_ids: {chain_ids}"
                raise ValueError(msg)

    @staticmethod
    def _reindex_after_crop(
        *,
        crop_indices: np.ndarray,
        atom_to_token_idx_map: np.ndarray,
        token_to_residue_idx_map: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compact atom/token indices to the [0, cropped) range."""
        token_mask = np.isin(token_to_residue_idx_map, crop_indices)
        cropped_token_indices = np.where(token_mask)[0]
        cropped_token_to_residue = token_to_residue_idx_map[token_mask]

        max_res = token_to_residue_idx_map.max() + 1
        residue_lookup = np.full(max_res, -1)
        residue_lookup[crop_indices] = np.arange(len(crop_indices))
        cropped_token_to_residue_reindexed = residue_lookup[cropped_token_to_residue]

        atom_mask = np.isin(atom_to_token_idx_map, cropped_token_indices)
        cropped_atom_to_token = atom_to_token_idx_map[atom_mask]

        max_token = atom_to_token_idx_map.max() + 1
        token_lookup = np.full(max_token, -1)
        token_lookup[cropped_token_indices] = np.arange(len(cropped_token_indices))
        cropped_atom_to_token_reindexed = token_lookup[cropped_atom_to_token]

        return cropped_atom_to_token_reindexed, cropped_token_to_residue_reindexed
