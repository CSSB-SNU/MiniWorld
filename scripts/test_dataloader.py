import numpy as np
import random
import torch
import click
import time
from pathlib import Path

from omegaconf import OmegaConf

from pydantic import BaseModel

from MiniWorld.data.dataloader.dataloader_edge_backprop import AdaptiveEdgeSampler

from MiniWorld.data.features.features_multistate import Batch
from MiniWorld.data.dataloader.dataloader_edge_backprop_test import (
    BioMolData,
    EdgeWeightConfig,
    BioMolDBConfig,
    CropConfig,
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


class TestClient:
    class DataConfig(BaseModel):
        train_DB: BioMolDBConfig
        valid_DB: BioMolDBConfig
        edge_weight: EdgeWeightConfig
        crop: CropConfig

    class ExperimentsConfig(BaseModel):
        train_item: int = 25600
        valid_item: int = 2560
        num_workers: int = 4
        prefetch_factor: int = 4
        seed: int = 0


    class Config(BaseModel):
        data: "TestClient.DataConfig"
        experiment: "TestClient.ExperimentsConfig"

    def set_seed(self, seed: int):
        """Set the random seed for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)

    def __init__(self, config: Config, name: str = "Test-PSK-2"):
        self.set_seed(config.experiment.seed)

    def training_step(self, batch: Batch) -> float:
        time.sleep(0.5)
        return torch.rand(1)

    def validation_step(self, batch: Batch) -> float:
        return torch.rand(1)

    def training_epoch(self, dataloader: torch.utils.data.DataLoader, num_item: int) -> list:
        outputs = []
        sampler: AdaptiveEdgeSampler = dataloader.sampler
        pair_keys = list(set(sampler.stats.edge_type))
        # remove Q, X
        pair_keys = [key for key in pair_keys if "Q" not in key and "X" not in key]
        statistics = dict.fromkeys(pair_keys, 0)
        dataset_edge_type = sampler.stats.edge_type
        dataset_statistics = {key: sum(1 for et in dataset_edge_type if et == key) for key in pair_keys}

        print(f"Dataset statistics: {dataset_statistics}")
        breakpoint()
        for batch_idx, batch in enumerate(dataloader):
            # print(f"Train Batch {batch_idx}")
            output = self.training_step(batch)
            sampler.stats.update(batch.scheme.edge_index, output)
            edge_type = [sampler.stats.edge_type[i] for i in batch.scheme.edge_index[0].cpu().numpy()]
            for et in edge_type:
                if et not in statistics:
                    continue
                statistics[et] += 1
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}: statistics = {statistics}")
            if batch_idx == num_item - 1:
                break
        return outputs


@click.group()
def cli():
    pass



@cli.command()
@click.option("--config", type=click.Path(exists=True), help="config file")
def train(
    config: str | None = None,
    device: str = "cpu",
    seed: int | None = 1123,
    slurm: bool = False,
    **slurm_kwargs,
):
    """Test Dataloader."""

    # Load client
    config_path = Path(config)
    if not config_path.exists():
        raise FileNotFoundError(f"Cannot found config file: {config_path}")
    config = OmegaConf.load(config)
    config = TestClient.Config(**config)
    client = TestClient(config, name="MiniWorld")

    # Setup wandb
    if seed is not None:
        set_seed(seed)

    train_data_config = BioMolData.BioMolConfig(
        crop_config=config.data.crop.model_dump(),
        DB_config=config.data.train_DB.model_dump(),
        edge_weight_config=config.data.edge_weight.model_dump(),
    )
    if config.experiment.prefetch_factor == 0:
        prefetch_factor = None
    else:
        prefetch_factor = config.experiment.prefetch_factor

    train_loader = BioMolData(train_data_config).create_ddp_dataloader(
        rank=0,
        world_size=1,
        drop_last=True,
        batch_size=1,
        num_workers=config.experiment.num_workers,
        prefetch_factor=prefetch_factor,
    )

    train_num_item = config.experiment.train_item
    for epoch in range(2):
        train_loader.sampler.set_epoch(epoch)
        client.training_epoch(train_loader, train_num_item)

if __name__ == "__main__":
    cli()
