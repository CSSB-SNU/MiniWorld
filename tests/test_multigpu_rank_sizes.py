"""Simulate multi-GPU dataloading and measure per-rank batch sizes.

Reproduces the v2.1.1 training setup (grad_accum_steps=32, bucketing on)
and reports n_tokens / n_atoms / msa_depth statistics per simulated rank.

Usage:
    python tests/test_multigpu_rank_sizes.py --world-size 4 --microbatches 32
"""

from __future__ import annotations

import argparse
import statistics
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
            min_segment_size=1,
            max_segment_size=41,
            monomer_only=False,
            remain_invalid_tokens=False,
            bucket_msa_size=128,
            bucket_token_size=128,
            bucket_atom_size=1024,
            chain_crop_prob=0.5,
        ),
        msa_config=MSAConfig(max_msa_depth=384, missing_policy="query"),
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
            protein_protein=45.0,
            protein_ligand=10.0,
            protein_dna=10.0,
            protein_rna=10.0,
            antibody_protein=15.0,
            dna_dna=5.0,
            rna_rna=5.0,
            dna_rna=0.5,
            antibody_antibody=0.3,
            antibody_ligand=1.0,
            na_ligand=1.0,
            etc_interface=0.5,
            sole=0.5,
        ),
    )
    return BioMolData(config)


def _fmt(xs: list[int]) -> str:
    if not xs:
        return "(empty)"
    return (
        f"sum={sum(xs):>7d}  mean={statistics.mean(xs):>6.1f}  "
        f"min={min(xs):>4d}  max={max(xs):>4d}  "
        f"p50={int(statistics.median(xs)):>4d}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--microbatches", type=int, default=32,
                        help="microbatches per rank (= grad_accum_steps)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--bucket-token", type=int, default=128)
    parser.add_argument("--bucket-atom", type=int, default=1024)
    parser.add_argument("--bucket-msa", type=int, default=128)
    args = parser.parse_args()

    print(f"world_size={args.world_size} microbatches/rank={args.microbatches} "
          f"seed={args.seed} epoch={args.epoch}")
    print("(bucketed sizes: tokens↑{}, atoms↑{}, msa↑{})".format(
        args.bucket_token, args.bucket_atom, args.bucket_msa))
    print()

    ds = build_dataset()
    ds.set_epoch(args.epoch)
    print(f"dataset items: {len(ds.items)}  sum(weights)={sum(ds.weights):.4g}")
    print()

    per_rank_tokens: dict[int, list[int]] = {}
    per_rank_atoms: dict[int, list[int]] = {}
    per_rank_msa: dict[int, list[int]] = {}

    for rank in range(args.world_size):
        dl = ds.create_ddp_dataloader(
            rank=rank,
            world_size=args.world_size,
            shuffle=True,
            seed=args.seed,
            drop_last=True,
            num_workers=0,
            bucket_msa_multiple=args.bucket_msa,
            bucket_token_multiple=args.bucket_token,
            bucket_atom_multiple=args.bucket_atom,
        )
        dl.sampler.set_epoch(args.epoch)

        toks, atms, msas = [], [], []
        it = iter(dl)
        for mb in range(args.microbatches):
            batch = next(it)
            toks.append(int(batch.token_length))
            atms.append(int(batch.atom_length))
            msas.append(int(batch.msa_depth))
            print(f"  rank={rank} mb={mb:02d}  "
                  f"tokens={toks[-1]:>4d}  atoms={atms[-1]:>4d}  "
                  f"msa={msas[-1]:>4d}  name={batch.name[0]}")
        per_rank_tokens[rank] = toks
        per_rank_atoms[rank] = atms
        per_rank_msa[rank] = msas
        print()

    print("=" * 72)
    print("Per-rank summary over one accumulation window "
          f"({args.microbatches} microbatches)")
    print("=" * 72)
    for rank in range(args.world_size):
        print(f"\nrank {rank}:")
        print(f"  tokens: {_fmt(per_rank_tokens[rank])}")
        print(f"  atoms : {_fmt(per_rank_atoms[rank])}")
        print(f"  msa   : {_fmt(per_rank_msa[rank])}")

    print("\n" + "=" * 72)
    print("Cross-rank imbalance (sum of sizes per rank)")
    print("=" * 72)
    tok_sums = [sum(per_rank_tokens[r]) for r in range(args.world_size)]
    atm_sums = [sum(per_rank_atoms[r]) for r in range(args.world_size)]
    msa_sums = [sum(per_rank_msa[r]) for r in range(args.world_size)]

    def _imbalance(xs: list[int]) -> str:
        mn, mx, mean = min(xs), max(xs), statistics.mean(xs)
        return (f"min={mn:>7d}  max={mx:>7d}  mean={mean:>8.1f}  "
                f"(max-min)/mean={(mx-mn)/mean*100:5.1f}%  max/min={mx/max(mn,1):.2f}x")

    print(f"  token-sum per rank: {tok_sums}")
    print(f"    {_imbalance(tok_sums)}")
    print(f"  atom-sum  per rank: {atm_sums}")
    print(f"    {_imbalance(atm_sums)}")
    print(f"  msa-sum   per rank: {msa_sums}")
    print(f"    {_imbalance(msa_sums)}")

    # approx compute proxy: pairwise attention ~ tokens^2, atom ops ~ atoms * msa
    comp_proxy = [sum(t*t for t in per_rank_tokens[r]) for r in range(args.world_size)]
    print(f"\n  compute proxy  sum(tokens^2) per rank: {comp_proxy}")
    print(f"    {_imbalance(comp_proxy)}")


if __name__ == "__main__":
    main()
