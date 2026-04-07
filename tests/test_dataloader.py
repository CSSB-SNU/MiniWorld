from pathlib import Path

from miniworld.configs import (
    BioMolDBConfig,
    CropConfig,
    MSAConfig,
)
from miniworld.data.dataloader.dataloader import BioMolData

if __name__ == "__main__":
    # test dataloader
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
            edge_id_to_bias_path=(
                Path(
                    "/home/psk6950/data/BioMolDB_20260224/metadata/train_edge_node.tsv",
                )
            ),
            template_db_path=Path(
                "/home/psk6950/data/BioMolDB_20260224/template.lmdb",
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
        breakpoint()
