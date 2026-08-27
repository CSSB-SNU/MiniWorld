from __future__ import annotations

import functools
import re
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
# from miniworld.data.mols import CCDMol, FragmentedCCDMol
from miniworld.data.pipeline import (
    ProteinTemplate,
    Tokenizer,
    # fragment_ccdmol_all_merges,
    get_chain_crop_indices,
    sample_msa,
)
from miniworld.data.pipeline.utils import (
    NoInterfaceError,
    find_interface_residues,
    remove_terminal_oxygen,
)
from miniworld.utils.crop import crop_spatial_segment_token

from .sampler import WeightedSampler

if TYPE_CHECKING:
    from pathlib import Path

    from miniworld.data.mols import CIFMolAttached


# class FragmentedCCDMolCache(Mapping[str, dict[int, FragmentedCCDMol]]):
#     """Lazy cache for CCD fragmentations keyed by chemcomp id."""

#     def __init__(self, ccd_preprocessed_path: Path, keys: list[str]) -> None:
#         self.ccd_preprocessed_path = ccd_preprocessed_path
#         self._keys = set(keys)
#         self._cache: dict[str, dict[int, FragmentedCCDMol]] = {}

#     def __getitem__(self, key: str) -> dict[int, FragmentedCCDMol]:
#         if key in self._cache:
#             return self._cache[key]
#         if key not in self._keys:
#             raise KeyError(key)

#         data = load_raw_data(key, self.ccd_preprocessed_path)
#         if data is None:
#             raise KeyError(key)

#         ccdmol = CCDMol.from_bytes(data)
#         fragments = fragment_ccdmol_all_merges(ccdmol)
#         self._cache[key] = fragments
#         return fragments

#     def __iter__(self) -> Iterator[str]:
#         return iter(self._keys)

#     def __len__(self) -> int:
#         return len(self._keys)

#     def __contains__(self, key: object) -> bool:
#         return isinstance(key, str) and (key in self._cache or key in self._keys)


def _ceil_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _bucketed_collate(
    batch_list: list[Batch],
    bucket_msa_multiple: int | None = None,
    # bucket_template_multiple: int | None = 1,
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

    # n_temp = batch.template_number
    msa_depth = batch.msa_depth
    n_tokens = batch.token_length
    n_atoms = batch.atom_length

    bucketed_msa = (
        _ceil_to_multiple(msa_depth, bucket_msa_multiple)
        if bucket_msa_multiple
        else msa_depth
    )

    # bucketed_template = (
    #     _ceil_to_multiple(n_temp, bucket_template_multiple)
    #     if bucket_template_multiple
    #     else n_temp
    # )

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
        # and bucketed_template == n_temp
        and bucketed_tokens == n_tokens
        and bucketed_atoms == n_atoms
    ):
        return batch

    dummy = Batch.empty(
        # n_temp=n_temp,
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

        self.weights: list[float] = []
        self.items: list[DataBias] = []

        self._load_items()
        # self._load_ccd_preprocessed()

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for this dataset, which can be used to change sampling behavior."""
        self.epoch = epoch

    def _make_seed(self, idx: int) -> int:
        return (self.seed * 1000003 + self.epoch * 100003 + idx) & 0xFFFF_FFFF

    def _load_items(self) -> None:
        # load edge_id to cif_ids mapping

        types = {
            "antibody_antibody",
            "antibody_nucleic_acid",
            "antibody_protein",
            "DNA_DNA",
            "RNA_RNA",
            "DNA_RNA",
            "NA_NA",
            "protein_nucleic_acid",
            "protein_protein",
            "protein_ligand",
            "ligand_ligand",
            "sole",
            "etc_interface",
        }

        type_counts = dict.fromkeys(types, 0)

        def _get_type(edge_id: str) -> str:  # noqa: C901, PLR0911
            """Get type for a given edge_id based on its type."""
            if "_" not in edge_id:
                return "sole"
            parse = set(re.findall(r"c([A-Z])", edge_id))

            # Antibody
            if parse == {"A"}:
                return "antibody_antibody"
            if parse <= {"A", "D", "R"} and "A" in parse:
                return "antibody_nucleic_acid"
            if parse <= {"A", "P"} and "A" in parse:
                return "antibody_protein"

            # Nucleic acid only
            if parse == {"D"}:
                return "DNA_DNA"
            if parse == {"R"}:
                return "RNA_RNA"
            if parse == {"D", "R"}:
                return "DNA_RNA"
            if parse <= {"D", "R", "N"} and "N" in parse:
                return "NA_NA"

            # Protein related
            if parse <= {"P", "D", "R", "N"} and "P" in parse and len(parse) > 1:
                return "protein_nucleic_acid"
            if parse == {"P"}:
                return "protein_protein"
            if parse <= {"P", "L"} and "P" in parse:
                return "protein_ligand"

            # Ligand
            if parse == {"L"}:
                return "ligand_ligand"

            # fallback
            return "etc_interface"

        edge_id_to_items = {}
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
                if edge_id not in edge_id_to_items:
                    edge_id_to_items[edge_id] = []
                edge_id_to_items[edge_id].append(value)
                type_name = _get_type(edge_id)
                type_counts[type_name] += 1

        self.items = []
        self.weights = []
        for edge_id, items in edge_id_to_items.items():
            type_name = _get_type(edge_id)
            if self.config.sampler_config is not None:
                weights = (
                    getattr(self.config.sampler_config, type_name)
                    / type_counts[type_name]
                    / len(items)
                )
            else:
                weights = 1.0 / len(items)
            self.weights.extend([weights] * len(items))
            self.items.extend(items)

    # def _load_ccd_preprocessed(self) -> None:
    #     if self.config.DB_config.ccd_preprocessed_path is None:
    #         msg = "CCD preprocessed path is not provided in the config."
    #         raise ValueError(msg)

    #     keys = extract_lmdb_keys(self.config.DB_config.ccd_preprocessed_path)
    #     self.fragmented_ccd_mols: Mapping[str, dict[int, FragmentedCCDMol]] = (
    #         FragmentedCCDMolCache(self.config.DB_config.ccd_preprocessed_path, keys)
    #     )

    def __len__(self) -> int:
        """Return the number of edges in the dataset."""
        return len(self.items)

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
            # focus=focus,
            # fragmented_ccd_mols=self.fragmented_ccd_mols,
            # config=self.config.tokenizer_config.dynamic_config,
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
            cropped_atom_to_token_idx_map_reindexed, # reindex within cropped tokens, not global index
            cropped_token_to_residue_idx_map_reindexed, # reindex within cropped residues, not global index
            focus,
        )  # pyright: ignore[reportPossiblyUnboundVariable]

    def __getitem__(self, idx: int) -> Batch:
        """Get a data sample by index."""
        seed = self._make_seed(idx)
        rng = np.random.default_rng(seed)
        bias = self.items[idx]
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
                bias = self.items[idx]
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
                # focus=focus,
                # fragmented_ccd_mols=self.fragmented_ccd_mols,
                # config=self.config.tokenizer_config.dynamic_config,
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

        # templates: ProteinTemplate = load_templates(
        #     cifmol=cifmol,
        #     chain_id_to_crop_indices=chain_id_to_crop_indices,
        #     env_path=self.config.DB_config.template_db_path,
        #     n_templates=self.config.template_config.n_templates,
        #     rng=rng,
        # )

        return make_batch(
            cifmol=cifmol,
            msa=msa,
            # templates=templates,
            atom_to_token_idx_map=atom_to_token_idx_map,
            token_to_residue_idx_map=token_to_residue_idx_map,
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
        sampler = WeightedSampler(
            dataset=self,
            weights=self.weights,
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

from collections import Counter
from team_gm.constants import ResidueType

def audit_chem_comp_vs_mapped_restype(
    dataset: BioMolData,
    n_samples: int = 2000,
    seed: int = 0,
) -> None:
    rng = random.Random(seed)

    pair_counts: Counter[tuple[str, str]] = Counter()
    mismatch_counts: Counter[tuple[str, str]] = Counter()

    total_residues = 0
    mapped_diff_count = 0
    failed_samples = 0

    for _ in range(n_samples):
        try:
            idx = rng.randrange(len(dataset))
            edge_id = dataset.edge_id_list[idx]
            bias_str = rng.choice(dataset.edge_id_to_bias[edge_id])

            pdb_id, assembly_id, model_id, alt_id = bias_str.split("_")[:4]
            bias = re.findall(r"\(([^)]+)\)", bias_str)

            cifmol = load_cifmol(
                db_path=dataset.config.DB_config.cif_db_path,
                pdb_id=pdb_id.lower(),
                assembly_id=assembly_id,
                model_id=model_id,
                alt_id=alt_id,
            )
            cifmol = remove_terminal_oxygen(cifmol)

            max_tokens = dataset.config.crop_config.residue_crop_length
            max_atoms = dataset.config.crop_config.atom_crop_length
            crop_indices = None

            while True:
                crop_indices, _ = dataset.get_crop_indices(
                    cifmol=cifmol,
                    crop_indices=crop_indices,
                    bias=bias,
                    max_tokens=max_tokens,
                    max_atoms=max_atoms,
                )
                cropped = cifmol.residues[crop_indices].extract()
                _, token_to_residue_idx_map = dataset.tokenizer.tokenize(cropped)

                if token_to_residue_idx_map.shape[0] <= dataset.config.crop_config.residue_crop_length:
                    cifmol = cropped
                    break

                max_tokens = int(max_tokens * 0.9)
                max_atoms = int(max_atoms * 0.9)
                crop_indices = None

            chem_comp_ids = np.asarray(cifmol.residues.chem_comp_id.value, dtype=object)
            canonical_one_letter = np.asarray(cifmol.residues.one_letter_code_can.value, dtype=object)
            res_to_chain = np.asarray(cifmol.index_table.res_to_chain, dtype=np.int64)
            chain_entity_types = np.asarray(cifmol.chains.entity_type.value, dtype=object)

            for r in range(len(cifmol.residues)):
                chem = str(chem_comp_ids[r])
                entity_type = str(chain_entity_types[res_to_chain[r]])
                can1 = str(canonical_one_letter[r])

                mapped = ResidueType.resolve(
                    chem_comp_id=chem,
                    entity_type=entity_type,
                    canonical_one_letter=can1,
                ).name

                pair_counts[(chem, mapped)] += 1
                total_residues += 1

                if chem != mapped:
                    mapped_diff_count += 1
                    mismatch_counts[(chem, mapped)] += 1

        except Exception:
            failed_samples += 1

    print("\n=== chem_comp_id vs mapped canonical residue type ===")
    print(f"samples_checked={n_samples}")
    print(f"samples_failed={failed_samples}")
    print(f"total_cropped_residues={total_residues}")
    if total_residues > 0:
        print(
            f"mapped_different={mapped_diff_count} "
            f"({mapped_diff_count / total_residues:.6f})"
        )

    print("\nTop 30 mismatches: chem_comp_id -> mapped_restype")
    for (chem, mapped), cnt in mismatch_counts.most_common(30):
        print(f"{chem:>8} -> {mapped:<12} : {cnt}")

    print("\nTop 30 overall pairs: (chem_comp_id, mapped_restype)")
    for (chem, mapped), cnt in pair_counts.most_common(30):
        print(f"({chem:>8}, {mapped:<12}) : {cnt}")


if __name__ == "__main__":
    # test dataloader
    from pathlib import Path
    config = BioMolData.BioMolConfig(
        crop_config=CropConfig(
            residue_crop_length=512,
            atom_crop_length=4096,
            remain_invalid_tokens=False,
        ),
        msa_config=MSAConfig(
            n_samples=4,
            max_msa_depth=256,
            missing_policy="gap",
        ),
        DB_config=BioMolDBConfig(
            cif_db_path=Path(
                "/public_data/bsoohyuncd/BioMolDB_20260224/cif_attached_train_20260224_res9_chain300.lmdb",
            ),
            a3m_db_path=Path("/public_data/bsoohyuncd/BioMolDB_20260224/a3m.lmdb"),
            edge_id_to_bias_path=(
                Path(
                    "/public_data/bsoohyuncd/BioMolDB_20260224/metadata/train_20260224_edge_node.tsv",
                )
            ),
            # ccd_preprocessed_path=Path("/public_data/preprocessed_CCD.lmdb"),
        ),
    )
    dataset = BioMolData(config)
    dataloader = dataset.create_ddp_dataloader(
        rank=0,
        world_size=1,
        shuffle=True,
        seed=42,
        drop_last=False,
        num_workers=0,
        bucket_token_multiple=None,
        bucket_atom_multiple=None,
    )
    for i, batch in enumerate(dataloader):
        print(f"Batch {i}:")
        print("  atom pos shape:", batch.structure.atom_pos.shape)
        print("  atom pos mask shape:", batch.structure.atom_pos_mask.shape)
        print("  token length:", batch.token_length)
        print("  atom length:", batch.atom_length)
        if i >= 0:
            break

    # wo dataloader
    # test data 1hcu
    # cif_id = "3ni0"
    # data = dataset.get_item_by_id(pdb_id="3ni0", assembly_id="1", model_id="1", alt_id=".", chain_ids=["A", "C"])
    # print('atom pos', data.structure.atom_pos)
    # print('atom pos mask', data.structure.atom_pos_mask)
    # for _ in range(10):
    #     batch = dataset[174]

    # for idx in range(len(dataset)):
    #     print(f"Testing dataset idx {idx}/{len(dataset)}")
    #     batch = dataset[idx]

    # test dataloader    for i, batch in enumerate(dataloader):

    # for i, batch in enumerate(dataloader):
    #     print(f"Batch {i}:")
    
    # audit_chem_comp_vs_mapped_restype(
    # dataset,
    # n_samples=500,
    # seed=0,
    # )