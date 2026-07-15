"""Test: simulate multi-GPU data loading and check which samples each rank sees.

Usage:
    python tests/test_multigpu_data_distribution.py [--world-size 4] [--steps 10] [--seed 42]

No actual multi-GPU needed — creates one dataloader per rank and iterates.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from miniworld.configs import (
    BioMolDBConfig,
    CropConfig,
    MSAConfig,
    SamplerConfig,
    TokenizerConfig,
)
from miniworld.configs.data import DynamicTokenizationConfig
from miniworld.data.dataloader.dataloader import BioMolData

DB_PATH = Path("/NHNHOME/WORKSPACE/0226010152_A/data")


def build_dataset() -> BioMolData:
    config = BioMolData.BioMolConfig(
        crop_config=CropConfig(
            max_tokens=384,
            max_atoms=4096,
            remain_invalid_tokens=False,
        ),
        msa_config=MSAConfig(
            max_msa_depth=256,
            missing_policy="query",
        ),
        DB_config=BioMolDBConfig(
            cif_db_path=DB_PATH / "cif_attached_train_20260224_res9_chain300.lmdb",
            a3m_db_path=DB_PATH / "a3m.lmdb",
            edge_id_to_bias_path=DB_PATH / "metadata" / "train_20260224_edge_node.tsv",
            template_db_path=DB_PATH / "template.lmdb",
            ccd_preprocessed_path=DB_PATH / "CCD" / "preprocessed_CCD.lmdb",
        ),
        tokenizer_config=TokenizerConfig(
            level="dynamic",
            dynamic_config=DynamicTokenizationConfig(
                minimum_resolution_ratio=[0.2, 0.6, 0.2],
                sigma_flat_prob=0.3,
                sigma_min=4.0,
                sigma_max=8.0,
            ),
        ),
        sampler_config=SamplerConfig(
            antibody_antibody=0.3,
            antibody_nucleic_acid=1.0,
            antibody_protein=15.0,
            DNA_DNA=2.0,
            DNA_RNA=2.5,
            RNA_RNA=2.0,
            NA_NA=1.0,
            protein_nucleic_acid=20.0,
            protein_protein=45.0,
            protein_ligand=10.0,
            ligand_ligand=0.2,
            etc_interface=0.5,
            sole=0.5,
        ),
    )
    return BioMolData(config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check data distribution across ranks")
    parser.add_argument("--world-size", type=int, default=4, help="Simulated GPU count")
    parser.add_argument("--steps", type=int, default=10, help="Number of steps to run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epoch", type=int, default=0, help="Epoch number")
    args = parser.parse_args()

    world_size = args.world_size
    num_steps = args.steps
    seed = args.seed
    epoch = args.epoch

    print(f"=== Multi-GPU Data Distribution Test ===")
    print(f"world_size={world_size}, steps={num_steps}, seed={seed}, epoch={epoch}")
    print()

    dataset = build_dataset()
    dataset.set_epoch(epoch)

    # Create one dataloader per rank
    dataloaders = {}
    for rank in range(world_size):
        dl = dataset.create_ddp_dataloader(
            rank=rank,
            world_size=world_size,
            shuffle=True,
            seed=seed,
            drop_last=True,
            num_workers=0,
            bucket_token_multiple=128,
            bucket_atom_multiple=1024,
        )
        dl.sampler.set_epoch(epoch)
        dataloaders[rank] = dl

    # Iterate and collect names per rank
    rank_names: dict[int, list[str]] = {r: [] for r in range(world_size)}
    rank_iters = {r: iter(dl) for r, dl in dataloaders.items()}

    for step in range(num_steps):
        print(f"--- step {step} ---")
        for rank in range(world_size):
            try:
                batch = next(rank_iters[rank])
            except StopIteration:
                print(f"  rank {rank}: <exhausted>")
                continue

            names = batch.name
            n_tokens = batch.token_length
            n_atoms = batch.atom_length
            rank_names[rank].extend(names)

            for name in names:
                print(f"  rank {rank}: {name}  (tokens={n_tokens}, atoms={n_atoms})")
        print()

    # Summary: overlap check
    print("=== Summary ===")
    for rank in range(world_size):
        print(f"rank {rank}: {len(rank_names[rank])} samples")

    # Check for duplicates across ranks
    all_names = []
    for rank in range(world_size):
        all_names.extend(rank_names[rank])
    unique = set(all_names)
    duplicates = len(all_names) - len(unique)
    print(f"\ntotal samples seen: {len(all_names)}")
    print(f"unique samples: {len(unique)}")
    print(f"duplicates across ranks: {duplicates}")

    if duplicates > 0:
        from collections import Counter

        counts = Counter(all_names)
        dup_names = [name for name, cnt in counts.items() if cnt > 1]
        print(f"\nduplicated sample names:")
        for name in dup_names[:20]:
            ranks_with = [r for r in range(world_size) if name in rank_names[r]]
            print(f"  {name} -> ranks {ranks_with}")


if __name__ == "__main__":
    main()
