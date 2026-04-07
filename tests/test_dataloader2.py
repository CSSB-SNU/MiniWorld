from pathlib import Path

import torch

from miniworld.configs import (
    BioMolDBConfig,
    CropConfig,
    MSAConfig,
    TokenizerConfig,
)
from miniworld.configs.data import DynamicTokenizationConfig
from miniworld.data.dataloader.dataloader2 import BioMolData
from miniworld.data.features import Batch


def _get_array(obj):
    """Unwrap biomol Feature objects to numpy arrays."""
    return obj.value if hasattr(obj, "value") else obj


def save_batch_as_pdb(batch: Batch, output_dir: str = ".") -> None:
    """Save each sample in the batch as a PDB file.

    B-factor = number of atoms in the same token (fragment size).
    Color by B-factor in PyMOL/ChimeraX to visualize tokenization granularity.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    batch_size = batch.structure.atom_mask.shape[0]

    for b in range(batch_size):
        name = batch.name[b]
        atom_mask = batch.structure.atom_mask[b]  # (L_atom,)  bool
        atom_pos = batch.structure.atom_pos[b]  # (L_atom, 3)
        atom_to_token = batch.scheme.atom_to_token_idx_map[b]  # (L_atom,)
        token_residue_idx = batch.scheme.token_residue_idx[b]  # (L_token,)
        token_asym_id = batch.scheme.token_asym_id[b]  # (L_token,)

        # B-factor: number of atoms per token → mapped back to each atom
        n_tokens = token_residue_idx.shape[0]
        token_sizes = torch.zeros(n_tokens, dtype=torch.long, device=atom_mask.device)
        valid_tokens = atom_to_token[atom_mask]
        token_sizes.scatter_add_(0, valid_tokens, torch.ones_like(valid_tokens))
        b_factors = token_sizes[atom_to_token].float()  # (L_atom,)

        atom_ids = _get_array(batch.atom_ids[b])  # (L_atom,)  str
        chem_comp_ids = _get_array(batch.chem_comp_ids[b])  # (L_res,)  str
        heteros = _get_array(batch.heteros[b])  # (L_res,)   bool

        # per-atom residue and chain index
        atom_to_res = token_residue_idx[atom_to_token].cpu().numpy()  # (L_atom,)
        atom_to_chain = token_asym_id[atom_to_token].cpu().numpy()  # (L_atom,)

        valid_indices = atom_mask.cpu().numpy().nonzero()[0]
        lines = []

        for serial, i in enumerate(valid_indices, start=1):
            x, y, z = atom_pos[i].cpu().tolist()
            bfac = b_factors[i].item()
            res_idx = int(atom_to_res[i])
            chain_int = int(atom_to_chain[i])
            chain_id = chr(65 + chain_int % 26)  # 0→A, 1→B, …
            res_seq = res_idx % 10000  # PDB res seq: 4 digits max

            atom_name = str(atom_ids[i]) if i < len(atom_ids) else "X"
            res_name = (
                str(chem_comp_ids[res_idx]) if res_idx < len(chem_comp_ids) else "UNK"
            )
            is_hetero = bool(heteros[res_idx]) if res_idx < len(heteros) else False
            record = "HETATM" if is_hetero else "ATOM  "

            # PDB atom name: 4-char field; 1-char elements padded with leading space
            atom_name_fmt = (
                f"{atom_name:<4s}" if len(atom_name) >= 4 else f" {atom_name:<3s}"
            )

            # Standard PDB ATOM record (cols 1-66)
            line = (
                f"{record}{serial:5d} {atom_name_fmt} {res_name:3s} {chain_id:1s}"
                f"{res_seq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00{bfac:6.2f}"
            )
            lines.append(line)

        lines.append("END")
        pdb_path = out / f"{name}.pdb"
        pdb_path.write_text("\n".join(lines) + "\n")
        print(f"Saved: {pdb_path}")


if __name__ == "__main__":
    config = BioMolData.BioMolConfig(
        crop_config=CropConfig(
            max_tokens=512,
            max_atoms=4096,
            remain_invalid_tokens=False,
        ),
        msa_config=MSAConfig(
            max_msa_depth=256,
            missing_policy="gap",
        ),
        DB_config=BioMolDBConfig(
            cif_db_path=Path(
                "/home/psk6950/data//BioMolDB_20260224/cif_attached_train.lmdb",
            ),
            a3m_db_path=Path("/home/psk6950/data/BioMolDB_20260224/a3m.lmdb"),
            edge_id_to_bias_path=Path(
                "/home/psk6950/data/BioMolDB_20260224/metadata/train_edge_node.tsv",
            ),
            template_db_path=Path(
                "/home/psk6950/data/BioMolDB_20260224/template.lmdb",
            ),
            ccd_preprocessed_path=Path(
                "/home/psk6950/data/CCD/preprocessed_CCD.lmdb",
            ),
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
    )
    dataset = BioMolData(config)
    dataloader = dataset.create_ddp_dataloader(
        rank=0,
        world_size=1,
        shuffle=True,
        seed=42,
        drop_last=False,
        num_workers=0,
        bucket_token_multiple=128,
        bucket_atom_multiple=1024,
    )

    for batch in dataloader:
        print(batch.name)
        # save_batch_as_pdb(batch, output_dir="./pdb_out")
        # breakpoint()
