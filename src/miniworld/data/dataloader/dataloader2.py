from __future__ import annotations

import functools
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from pydantic import BaseModel
from torch.utils.data import DataLoader

from miniworld.configs import (
    BioMolDBConfig,
    CropConfig,
    MSAConfig,
    SamplerConfig,
    TemplateConfig,
    TokenizerConfig,
)
from miniworld.data.features import (
    Batch,
    make_batch,
)
from miniworld.data.io import (
    extract_lmdb_keys,
    load_cifmol,
    load_msa,
    load_raw_data,
    load_templates,
)
from miniworld.data.mols import CCDMol, FragmentedCCDMol
from miniworld.data.pipeline import (
    ProteinTemplate,
    Tokenizer,
    fragment_ccdmol_all_merges,
    get_chain_crop_indices,
    sample_msa,
)
from miniworld.data.pipeline.utils import (
    NoInterfaceError,
    find_interface_residues,
    remove_terminal_oxygen,
)
from miniworld.utils.crop import crop_spatial_segment_token

from .sampler import PDBWeightedSampler

if TYPE_CHECKING:
    from pathlib import Path

    from miniworld.data.mols import CIFMolAttached


class FragmentedCCDMolCache(Mapping[str, dict[int, FragmentedCCDMol]]):
    """Lazy cache for CCD fragmentations keyed by chemcomp id."""

    def __init__(self, ccd_preprocessed_path: Path, keys: list[str]) -> None:
        self.ccd_preprocessed_path = ccd_preprocessed_path
        self._keys = set(keys)
        self._cache: dict[str, dict[int, FragmentedCCDMol]] = {}

    def __getitem__(self, key: str) -> dict[int, FragmentedCCDMol]:
        if key in self._cache:
            return self._cache[key]
        if key not in self._keys:
            raise KeyError(key)

        data = load_raw_data(key, self.ccd_preprocessed_path)
        if data is None:
            raise KeyError(key)

        ccdmol = CCDMol.from_bytes(data)
        fragments = fragment_ccdmol_all_merges(ccdmol)
        self._cache[key] = fragments
        return fragments

    def __iter__(self) -> Iterator[str]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and (key in self._cache or key in self._keys)


def _ceil_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _bucketed_collate(
    batch_list: list[Batch],
    bucket_msa_multiple: int | None = None,
    bucket_template_multiple: int | None = 1,
    bucket_token_multiple: int | None = None,
    bucket_atom_multiple: int | None = None,
) -> Batch:
    """Collate with shape bucketing for torch.compile cache efficiency.

    Collates the batch normally, then pads dimensions to bucket boundaries by
    collating with a dummy empty batch of the bucketed size and discarding it.
    """
    batch = Batch.collate_fn(batch_list)

    if bucket_token_multiple is None and bucket_atom_multiple is None:
        return batch

    n_temp = batch.template_number
    msa_depth = batch.msa_depth
    n_tokens = batch.token_length
    n_atoms = batch.atom_length

    bucketed_msa = (
        _ceil_to_multiple(msa_depth, bucket_msa_multiple)
        if bucket_msa_multiple
        else msa_depth
    )

    bucketed_template = (
        _ceil_to_multiple(n_temp, bucket_template_multiple)
        if bucket_template_multiple
        else n_temp
    )

    bucketed_tokens = (
        _ceil_to_multiple(n_tokens, bucket_token_multiple)
        if bucket_token_multiple
        else n_tokens
    )
    bucketed_atoms = (
        _ceil_to_multiple(n_atoms, bucket_atom_multiple)
        if bucket_atom_multiple
        else n_atoms
    )

    if (
        bucket_msa_multiple
        and bucketed_msa == msa_depth
        and bucketed_template == n_temp
        and bucketed_tokens == n_tokens
        and bucketed_atoms == n_atoms
    ):
        return batch

    dummy = Batch.empty(
        n_temp=n_temp,
        msa_depth=bucketed_msa,
        n_tokens=bucketed_tokens,
        n_atoms=bucketed_atoms,
    )
    padded = Batch.collate_fn([batch, dummy])
    return padded[0 : batch.batch_size]


class WrongCroppingError(ValueError):
    """Raised when the cropping strategy fails to produce a valid crop."""


class DataBias:
    """Identifier for a cif entry, consisting of pdb_id, assembly_id, model_id, and alt_id."""

    def __init__(
        self,
        pdb_id: str,
        assembly_id: str,
        model_id: str,
        alt_id: str,
        chain_id1: str,
        chain_id2: str | None = None,
    ) -> None:
        self.pdb_id = pdb_id
        self.assembly_id = assembly_id
        self.model_id = model_id
        self.alt_id = alt_id
        self.chain_id1 = chain_id1
        self.chain_id2 = chain_id2


class BioMolData(torch.utils.data.Dataset):
    """Dataset for biomolecular complexes based on BioMolDB."""

    class BioMolConfig(BaseModel):
        """Configuration for BioMolData."""

        crop_config: CropConfig = CropConfig()
        msa_config: MSAConfig = MSAConfig()
        template_config: TemplateConfig = TemplateConfig()
        DB_config: BioMolDBConfig = BioMolDBConfig()
        tokenizer_config: TokenizerConfig = TokenizerConfig()
        sampler_config: SamplerConfig | None = SamplerConfig()

    def __init__(
        self,
        config: BioMolConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.epoch: int = 0
        self.seed: int = 0
        self.tokenizer = Tokenizer(config=config.tokenizer_config)

        self._load_edge_to_cif_ids()
        self._load_ccd_preprocessed()

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for this dataset, which can be used to change sampling behavior."""
        self.epoch = epoch

    def _make_seed(self, idx: int) -> int:
        return (self.seed * 1000003 + self.epoch * 100003 + idx) & 0xFFFF_FFFF

    def _load_edge_to_cif_ids(self) -> None:
        # load edge_id to cif_ids mapping
        self.edge_id_to_bias: dict[str, list[DataBias]] = {}
        with self.config.DB_config.edge_id_to_bias_path.open("r") as f:
            for ii, _line in enumerate(f):
                if ii == 0:
                    continue  # skip header
                line = _line.strip()
                (
                    cluster1,
                    cluster2,
                    pdb_id,
                    assembly_id,
                    model_id,
                    alt_id,
                    chain_id1,
                    chain_id2,
                ) = line.split("\t")
                pdb_id = pdb_id.lower()  # to match cif_db keys
                if line == "":
                    continue
                if cluster2 == "None":
                    edge_id = cluster1
                    value = DataBias(pdb_id, assembly_id, model_id, alt_id, chain_id1)
                else:
                    edge_id = f"{cluster1}_{cluster2}"
                    value = DataBias(
                        pdb_id,
                        assembly_id,
                        model_id,
                        alt_id,
                        chain_id1,
                        chain_id2,
                    )
                if edge_id not in self.edge_id_to_bias:
                    self.edge_id_to_bias[edge_id] = []
                self.edge_id_to_bias[edge_id].append(value)

        self.edge_id_list = list(self.edge_id_to_bias.keys())

    def _load_ccd_preprocessed(self) -> None:
        if self.config.DB_config.ccd_preprocessed_path is None:
            msg = "CCD preprocessed path is not provided in the config."
            raise ValueError(msg)

        keys = extract_lmdb_keys(self.config.DB_config.ccd_preprocessed_path)
        self.fragmented_ccd_mols: Mapping[str, dict[int, FragmentedCCDMol]] = (
            FragmentedCCDMolCache(self.config.DB_config.ccd_preprocessed_path, keys)
        )

    def __len__(self) -> int:
        """Return the number of edges in the dataset."""
        return len(self.edge_id_list)

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
        """Get crop indices for a given cifmol, either by cropping or using provided indices."""
        match chain_ids:
            case [chain_id]:
                selected_atoms = cifmol.chains.select(chain_id=chain_id).atoms
            case [chain_id1, chain_id2]:
                try:
                    selected_atoms = find_interface_residues(
                        cifmol,
                        chain_id1,
                        chain_id2,
                    ).atoms
                except NoInterfaceError:
                    chain_id = rng.choice([chain_id1, chain_id2])
                    selected_atoms = cifmol.chains.select(chain_id=chain_id).atoms
            case _:
                msg = f"Unexpected chain_ids: {chain_ids}"
                raise ValueError(msg)

        valid = np.all(np.isfinite(selected_atoms.xyz), axis=-1)
        selected_atoms = selected_atoms[valid]
        if len(selected_atoms) == 0:
            msg = f"No valid atoms found for chain_ids {chain_ids} in cifmol {cifmol.id}"
            raise WrongCroppingError(msg)
        focus = selected_atoms.xyz[rng.integers(0, len(selected_atoms))].value

        segment_size = int(
            rng.integers(
                self.config.crop_config.min_segment_size,
                self.config.crop_config.max_segment_size + 1,
            ),
        )

        atom_to_token_idx_map, token_to_residue_idx_map = self.tokenizer.tokenize(
            cifmol,
            focus=focus,
            fragmented_ccd_mols=self.fragmented_ccd_mols,
            config=self.config.tokenizer_config.dynamic_config,
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

        token_mask = np.isin(token_to_residue_idx_map, crop_indices)
        cropped_token_indices = np.where(token_mask)[0]
        cropped_token_to_residue_idx_map = token_to_residue_idx_map[token_mask]

        max_res = token_to_residue_idx_map.max() + 1
        lookup = np.full(max_res, -1)
        lookup[crop_indices] = np.arange(len(crop_indices))

        cropped_token_to_residue_idx_map_reindexed = lookup[
            cropped_token_to_residue_idx_map
        ]

        atom_mask = np.isin(atom_to_token_idx_map, cropped_token_indices)
        cropped_atom_to_token_idx_map = atom_to_token_idx_map[atom_mask]

        max_token = atom_to_token_idx_map.max() + 1
        lookup_token = np.full(max_token, -1)
        lookup_token[cropped_token_indices] = np.arange(len(cropped_token_indices))
        cropped_atom_to_token_idx_map_reindexed = lookup_token[
            cropped_atom_to_token_idx_map
        ]

        return (
            crop_indices,
            chain_id_to_crop_indices,
            cropped_atom_to_token_idx_map_reindexed,
            cropped_token_to_residue_idx_map_reindexed,
            focus,
        )  # pyright: ignore[reportPossiblyUnboundVariable]

    def __getitem__(self, idx: int) -> Batch:
        """Get a data sample by index."""
        seed = self._make_seed(idx)
        rng = np.random.default_rng(seed)
        edge_id = self.edge_id_list[idx]
        biases: list[DataBias] = self.edge_id_to_bias[edge_id]
        bias = biases[rng.integers(0, len(biases))]
        pdb_id, assembly_id, model_id, alt_id = (
            bias.pdb_id,
            bias.assembly_id,
            bias.model_id,
            bias.alt_id,
        )
        chain_ids = [bias.chain_id1] + ([bias.chain_id2] if bias.chain_id2 else [])

        while True:
            try:
                item = self.get_item_by_id(
                    pdb_id=pdb_id,
                    assembly_id=assembly_id,
                    model_id=model_id,
                    alt_id=alt_id,
                    chain_ids=chain_ids,
                    rng=rng,
                )
                break
            except WrongCroppingError:
                idx = int(rng.integers(0, len(self)))
                edge_id = self.edge_id_list[idx]
                biases = self.edge_id_to_bias[edge_id]
                bias = biases[rng.integers(0, len(biases))]
                pdb_id, assembly_id, model_id, alt_id = (
                    bias.pdb_id,
                    bias.assembly_id,
                    bias.model_id,
                    bias.alt_id,
                )
                chain_ids = [bias.chain_id1] + (
                    [bias.chain_id2] if bias.chain_id2 else []
                )

        return item

    def get_item_by_id(
        self,
        pdb_id: str,
        assembly_id: str | None = None,
        model_id: str | None = None,
        alt_id: str | None = None,
        chain_ids: list[str] | None = None,
        crop_indices: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> Batch:
        """Get a data sample by cif_id."""
        if rng is None:
            rng = np.random.default_rng()
        cifmol = load_cifmol(
            db_path=self.config.DB_config.cif_db_path,
            pdb_id=pdb_id,
            assembly_id=assembly_id,
            model_id=model_id,
            alt_id=alt_id,
        )

        if chain_ids is None:  # randoml sample chain_id
            chain_ids = rng.choice(cifmol.chains.chain_id.value)
        if crop_indices is None:
            (
                crop_indices,
                chain_id_to_crop_indices,
                atom_to_token_idx_map,
                token_to_residue_idx_map,
                focus,
            ) = self.get_crop_indices(
                cifmol=cifmol,
                chain_ids=chain_ids,  # pyright: ignore[reportArgumentType]
                max_tokens=self.config.crop_config.max_tokens,
                max_atoms=self.config.crop_config.max_atoms,
                rng=rng,
            )
            if crop_indices.shape[0] == 0:
                msg = f"Failed to crop {pdb_id}_{assembly_id}_{model_id}_{alt_id} with chain_ids {chain_ids}."
                raise WrongCroppingError(msg)
        else:
            chain_id_to_crop_indices = get_chain_crop_indices(
                cifmol=cifmol,
                crop_indices=crop_indices,
            )
            focus = None

            # No spatial focus (pre-provided crop_indices): sample a random valid atom.
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
                config=self.config.tokenizer_config.dynamic_config,
            )
        cifmol: CIFMolAttached = cifmol.residues[crop_indices].extract()
        atom_mask = remove_terminal_oxygen(cifmol)
        cifmol = cifmol.atoms[atom_mask].extract()
        atom_to_token_idx_map = atom_to_token_idx_map[atom_mask]

        # Load MSA
        complex_msa = load_msa(
            cifmol=cifmol,
            chain_id_to_crop_indices=chain_id_to_crop_indices,  # pyright: ignore[reportPossiblyUnboundVariable]
            env_path=self.config.DB_config.a3m_db_path,
        )
        msa = sample_msa(
            msa=complex_msa,
            max_msa_depth=self.config.msa_config.max_msa_depth,
            rng=rng,
        )

        templates: ProteinTemplate = load_templates(
            cifmol=cifmol,
            chain_id_to_crop_indices=chain_id_to_crop_indices,
            env_path=self.config.DB_config.template_db_path,
            n_templates=self.config.template_config.n_templates,
            rng=rng,
        )

        return make_batch(
            cifmol=cifmol,
            msa=msa,
            templates=templates,
            atom_to_token_idx_map=atom_to_token_idx_map,  # pyright: ignore[reportPossiblyUnboundVariable]
            token_to_residue_idx_map=token_to_residue_idx_map,  # pyright: ignore[reportPossiblyUnboundVariable]
            rng=rng,
        )

    def create_ddp_dataloader(
        self,
        rank: int,
        *,
        world_size: int = 1,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        num_workers: int = 0,
        bucket_msa_multiple: int | None = None,
        bucket_token_multiple: int | None = None,
        bucket_atom_multiple: int | None = None,
        **kwargs: object,
    ) -> DataLoader:
        """Create a distributed DataLoader with WeightedSampler."""
        self.seed = int(seed)

        # default distributed sampler
        sampler = PDBWeightedSampler(
            dataset=self,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )

        kwargs.pop("shuffle", None)
        kwargs.pop("world_size", None)
        kwargs.update({"sampler": sampler})
        if num_workers == 0:
            # remove prefetch_factor
            kwargs.pop("prefetch_factor", None)

        worker_seed_rng = torch.Generator()
        worker_seed_rng.manual_seed(int(seed) + int(rank))

        params = {
            "shuffle": False,  # leave False when using a sampler
            "drop_last": False,  # override to True for train
            "num_workers": num_workers,
            "pin_memory": False,
            "generator": worker_seed_rng,
            "multiprocessing_context": ("spawn" if num_workers > 0 else None),
            "collate_fn": functools.partial(
                _bucketed_collate,
                bucket_msa_multiple=bucket_msa_multiple,
                bucket_token_multiple=bucket_token_multiple,
                bucket_atom_multiple=bucket_atom_multiple,
            ),
        }
        params.update(kwargs)
        return DataLoader(self, **params)
