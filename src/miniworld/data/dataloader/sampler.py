from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
import torch
from torch.utils.data import Dataset, DistributedSampler

from miniworld.configs import (
    SamplerConfig,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


class _HasSamplerConfig(Protocol):
    sampler_config: SamplerConfig | None


class _WeightedSamplerDataset(Protocol):
    edge_id_list: list[str]
    config: _HasSamplerConfig


class PDBWeightedSampler(DistributedSampler):
    """Sampler that samples indices according to given weights."""

    def __init__(self, dataset: Dataset[object], **kwargs: Any) -> None:
        super().__init__(dataset, **kwargs)
        typed_dataset = cast("_WeightedSamplerDataset", dataset)
        self.edge_id_list = typed_dataset.edge_id_list
        self.num_samples = len(typed_dataset.edge_id_list)
        self._load_weights(typed_dataset.config.sampler_config)

    def _load_weights(self, config: SamplerConfig | None) -> None:  # noqa: C901
        """Load weights from config and edge_id_list."""
        if config is None:
            # uniform weights
            weights = np.ones(len(self.edge_id_list), dtype=np.float32)
            self.weights = torch.from_numpy(weights)
            return

        def _get_weight(edge_id: str) -> float:  # noqa: C901, PLR0911
            """Get weight for a given edge_id based on its type."""
            if "_" not in edge_id:
                return config.sole
            parse = set(re.findall(r"c([A-Z])", edge_id))

            # Antibody
            if parse == {"A"}:
                return config.antibody_antibody
            if parse <= {"A", "D", "R"} and "A" in parse:
                return config.antibody_nucleic_acid
            if parse <= {"A", "P"} and "A" in parse:
                return config.antibody_protein

            # Nucleic acid only
            if parse == {"D"}:
                return config.DNA_DNA
            if parse == {"R"}:
                return config.RNA_RNA
            if parse == {"D", "R"}:
                return config.DNA_RNA
            if parse <= {"D", "R", "N"} and "N" in parse:
                return config.NA_NA

            # Protein 관련
            if parse <= {"P", "D", "R", "N"} and "P" in parse and len(parse) > 1:
                return config.protein_nucleic_acid
            if parse == {"P"}:
                return config.protein_protein
            if parse <= {"P", "L"} and "P" in parse:
                return config.protein_ligand

            # Ligand
            if parse == {"L"}:
                return config.ligand_ligand

            # fallback
            return config.etc_interface

        initial_weights = np.array(
            [_get_weight(edge_id) for edge_id in self.edge_id_list],
            dtype=np.float32,
        )
        weights = initial_weights / initial_weights.sum()
        self.weights = torch.from_numpy(weights)

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # weighted sampling으로 교체, 나머지 DDP 로직은 DistributedSampler에 위임
        all_indices = torch.multinomial(
            self.weights,
            self.total_size,
            replacement=False,
            generator=g,
        ).tolist()

        return iter(all_indices[self.rank : self.total_size : self.num_replicas])

    def __len__(self) -> int:
        return self.num_samples
