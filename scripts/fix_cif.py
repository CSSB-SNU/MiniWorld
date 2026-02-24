from __future__ import annotations

import re
from pathlib import Path

import click
from Bio.PDB.MMCIF2Dict import MMCIF2Dict as mmcif2dict  # noqa: N813
from Bio.PDB.mmcifio import MMCIFIO
from omegaconf import OmegaConf
from pydantic import BaseModel

from miniworld.data.dataloader.dataloader_edge_backprop import (
    MSAConfig,
)
from miniworld.data.dataloader.dataloader_foldbench import (
    FoldBenchData,
    FoldBenchDataConfig,
)
from miniworld.data.features.batch_foldbench import (
    Batch,
)
from miniworld.data.io import (
    load_bytes,
    load_raw_data,
)


@click.group()
def cli():
    pass


def load_cifmol(db_path: Path, cif_id: str) -> tuple[dict, dict]:
    """Load CIFMolAttached from LMDB by cif_id."""
    pdb_id, assembly_id, model_id, alt_id = re.findall(
        r"\([^)]*\)|[^_]+",
        cif_id,
    )
    value = load_raw_data(pdb_id, db_path)

    if value is None:
        msg = f"Key '{pdb_id}' not found in LMDB database at '{db_path}'."
        raise KeyError(msg)

    value = load_bytes(value)
    metadata = value["metadata_dict"]
    item = value["assembly_dict"].get(f"{assembly_id}_{model_id}_{alt_id}")
    if item is None:
        msg = f"CIFMol '{cif_id}' not found in LMDB database at '{db_path}'."
        raise KeyError(msg)
    return item, metadata


def process_cif_file(cif_path: Path, output_path: Path, batch: Batch):
    mmcif_dict = mmcif2dict(str(cif_path))

    tag = "_atom_site.pdbx_formal_charge"
    if tag in mmcif_dict:
        mmcif_dict[tag] = ["0"] * len(mmcif_dict[tag])

    """
    loop_
    _entity_poly_seq.entity_id
    _entity_poly_seq.hetero
    _entity_poly_seq.mon_id
    _entity_poly_seq.num
    """
    polymer_types = {0, 1, 2, 3, 4, 5}
    entity_type = batch.chain.entity_type[0].tolist()  # assuming batch size is 1

    entity_polymer_map = {}
    polymer_mask = []
    chain_indices = batch.scheme.residue_asym_id[0].tolist()  # assuming batch size is 1
    entity_id = batch.scheme.residue_entity_id[0].tolist()  # assuming batch size is 1
    entity_id = [str(e) for e in entity_id]
    for chain_idx, entity in zip(chain_indices, entity_id, strict=True):
        if entity_type[chain_idx] in polymer_types:
            polymer_mask.append(True)
            entity_polymer_map[entity] = True
        else:
            polymer_mask.append(False)
            entity_polymer_map[entity] = False

    hetero = list(batch.heteros[0].value)  # assuming batch size is 1
    hetero = ["n" if h == 0 else "y" for h in hetero]
    mon_id = list(batch.chem_comp_ids[0].value)  # assuming batch size is 1
    mon_id = [str(m) for m in mon_id]
    num = batch.scheme.residue_idx[0].tolist()  # assuming batch size is 1
    num = [str(n) for n in num]

    # Filter out duplicate entries
    key_list = [(e, n) for e, n in zip(entity_id, num, strict=True)]
    seen_keys = set()
    duplicate_mask = []
    for key in key_list:
        if key in seen_keys:
            duplicate_mask.append(True)
        else:
            seen_keys.add(key)
            duplicate_mask.append(False)
    mask = [(not d) and p for d, p in zip(duplicate_mask, polymer_mask, strict=True)]

    entity_id = [e for e, p in zip(entity_id, mask, strict=True) if p]
    hetero = [h for h, p in zip(hetero, mask, strict=True) if p]
    mon_id = [m for m, p in zip(mon_id, mask, strict=True) if p]
    num = [n for n, p in zip(num, mask, strict=True) if p]

    mmcif_dict["_entity_poly_seq.entity_id"] = entity_id
    mmcif_dict["_entity_poly_seq.hetero"] = hetero
    mmcif_dict["_entity_poly_seq.mon_id"] = mon_id
    mmcif_dict["_entity_poly_seq.num"] = num

    atom_site_group_pdb = mmcif_dict["_atom_site.group_PDB"]
    atom_site_entity_id = mmcif_dict["_atom_site.label_entity_id"]

    hetero_atom_mask = [
        not entity_polymer_map.get(entity, True) for entity in atom_site_entity_id
    ]
    atom_site_group_pdb = ["HETATM" if hetero else "ATOM" for hetero in hetero_atom_mask]
    # mmcif_dict["_atom_site.label_seq_id"] = [
    #     "." if hetero else seq_id
    #     for hetero, seq_id in zip(
    #         hetero_atom_mask,
    #         mmcif_dict["_atom_site.label_seq_id"],
    #         strict=True,
    #     )
    # ]
    # 0 -> "A", 1 -> "B", etc.
    mmcif_dict["_atom_site.label_asym_id"] = [
        chr(ord("A") + int(i) % 26) for i in mmcif_dict["_atom_site.label_asym_id"]
    ]
    atom_site_occupancy = ["1.00"] * len(atom_site_group_pdb)

    mmcif_dict["_atom_site.occupancy"] = atom_site_occupancy
    mmcif_dict["_atom_site.group_PDB"] = atom_site_group_pdb

    mask = batch.structure.atom_pos_mask[0].bool()
    atom_to_res = batch.scheme.atom_to_residue_idx_map[0]
    mmcif_dict["_atom_site.auth_seq_id"] = [
        str(seq_id.item())
        for seq_id in batch.scheme.residue_idx_mono[0, atom_to_res][mask]
    ]

    """
    _entity.id 
    _entity.type 
    """
    mmcif_dict["_entity.id"] = list(entity_polymer_map.keys())
    mmcif_dict["_entity.type"] = [
        "polymer" if entity_polymer_map[entity] else "non-polymer"
        for entity in entity_polymer_map
    ]

    io = MMCIFIO()
    io.set_dict(mmcif_dict)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    io.save(str(output_path))


@cli.command()
@click.option(
    "--cif-dir",
    type=Path,
    required=True,
    help="Directory containing predicted structures in PDB format.",
)
@click.option(
    "--output-dir",
    type=Path,
    required=True,
    help="Output directory where the generated CSV file will be saved.",
)
@click.option(
    "--config",
    type=Path,
    required=True,
    help="Path to the configuration file for FoldBenchData and MSAConfig.",
)
def fix_cif(
    cif_dir: Path,
    output_dir: Path,
    config: Path,
):
    class ValidateConfig(BaseModel):
        """Overall configuration for validation."""

        data: FoldBenchDataConfig
        msa: MSAConfig

    cfg = OmegaConf.load(config)
    cfg = ValidateConfig.model_validate(cfg)
    cif_paths = list(cif_dir.glob("*.cif"))

    cfg = OmegaConf.load(config)
    cfg = ValidateConfig.model_validate(cfg)

    fold_bench_data_config = FoldBenchData.FoldBenchConfig(
        msa_config=cfg.msa,
        data_config=cfg.data,
    )

    fold_bench_dataset = FoldBenchData(fold_bench_data_config)
    for cif_path in cif_paths:
        cif_id = cif_path.stem.split("_pred")[0]
        batch = fold_bench_dataset.get_item_by_id(cif_id)
        output_path = output_dir / cif_path.name
        if output_path.exists():
            click.echo(f"{output_path} already exists, skipping {cif_path}")
            continue
        process_cif_file(cif_path, output_path, batch)


if __name__ == "__main__":
    cli()
