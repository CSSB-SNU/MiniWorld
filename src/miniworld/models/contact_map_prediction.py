import random
from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from matplotlib import pyplot as plt
from pydantic import BaseModel
from team_gm import BaseClient
from team_gm.core.callbacks import ModelEMA
from team_gm.modules import Pairformer
from torch import nn

from miniworld.data.dataloader.dataloader_multistate_contam import (
    CropConfig,
    KmerFastAlignConfig,
    MSAConfig,
    MultistateConfig,
    MultiStatedbConfig,
)
from miniworld.data.features.features_biomol import Batch, NoisyBatch
from miniworld.loss.auxiliary import (
    cal_contact_map_focal_loss, 
    cal_contact_map_weighted_bce_loss, 
    extract_contact_map,
    cal_long_range_precision,
    cal_long_range_recall,
    cal_long_range_f1,
    cal_long_range_auroc,
)
from miniworld.modules.configs import (
    CommonConfig,
    DiffusionConfig,
)
from miniworld.modules.feature_embedder import InputFeatureEmbedder
from miniworld.modules.head import ContactMapHead
from miniworld.modules.msa_module import MSAModule
from miniworld.modules.primitives import (
    LayerNorm,
    Linear,
)
from miniworld.utils.precision_manager import PrecisionConfig, precision_manager


class ContactMapPredictionModel(nn.Module):
    """Structure Contact Map Prediction model."""

    class ConditionConfig(BaseModel):
        """Configuration for condition modules."""

        pairformer: Pairformer.Config
        msa_module: MSAModule.Config
        n_recycle_max: int = 4

    class Config(BaseModel):
        """Configuration for the AF3 model."""

        common: CommonConfig
        trunk: "ContactMapPredictionModel.ConditionConfig"
        diffusion: DiffusionConfig
        precision: PrecisionConfig

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.n_recycle_max = config.trunk.n_recycle_max

        # feature initialization
        self.input_feature_embedder = InputFeatureEmbedder(
            config.common,
            config.diffusion,
        )

        # Recycle layers
        self.add_pair_recycle = nn.Sequential(
            LayerNorm(
                config.common.d_token_pair,
                implementation=config.common.implementation,
            ),
            Linear(
                config.common.d_token_pair,
                config.common.d_token_pair,
                init="zero",
            ),
        )
        # Trunk forward
        self.msa_module = MSAModule(config.common, config.trunk.msa_module)
        self.pairformer_blocks = Pairformer(config.trunk.pairformer)

        # ContactMap prediction
        self.contact_map_head = ContactMapHead(config.common)

    def forward(self, batch: Batch) -> tuple[torch.Tensor, ...]:
        """Forward pass of the condition modules with recycling."""
        if self.training:
            n_recycle = random.randint(1, self.n_recycle_max)
        else:
            n_recycle = self.n_recycle_max
        if batch.msa.aligned_sequences.shape[1] != self.n_recycle_max:
            msg = (
                "The number of MSA sequences should match the number of recycle steps."
            )
            raise ValueError(msg)

        # input feature embedding
        (
            token_single_input,
            token_single_init,
            token_pair_init,
        ) = self.input_feature_embedder(batch)

        token_pair = torch.zeros_like(token_pair_init)
        # backprop cheating
        token_single_input = token_single_input + 0.0 * token_single_init.sum()
        # Trunk forward with recycling
        for i_cycle in range(n_recycle):
            with ExitStack() as stack:
                if i_cycle < n_recycle - 1:
                    stack.enter_context(torch.no_grad())
                    stack.enter_context(torch.inference_mode())
                token_pair = token_pair_init + self.add_pair_recycle(token_pair)
                token_pair = token_pair + self.msa_module(
                    batch,
                    i_cycle,
                    token_pair,
                    token_single_input,
                    batch.structure.residue_mask,
                )

                token_pair, _ = self.pairformer_blocks.forward(
                    token_pair,
                    None,
                    batch.structure.residue_mask,
                )

        return self.contact_map_head(token_pair)


class ContactMapPredictionClient(BaseClient):
    """Client for training and inference of ContactMapPredictor model."""

    class DataConfig(BaseModel):
        """Configuration for data loading."""

        crop: CropConfig
        msa: MSAConfig
        kmer_fast_align: KmerFastAlignConfig
        multistate: MultistateConfig
        train_preprocessing: MultiStatedbConfig
        valid_preprocessing: MultiStatedbConfig

    class ExperimentsConfig(BaseModel):
        """Configuration for experiments."""

        comment: str = "default"
        name: str = "ContactMapPredictor-PSK-2"
        overfitting: bool = False
        overfitting_dir: str | None = None  # Directory for overfitting mode
        train_item: int = 25600
        valid_item: int = 2560
        num_batch: int = 1
        num_epoch: int = 1000
        optimizer: Literal["AdamW", "Muon"] = "AdamW"
        max_lr: float = 1e-4
        min_lr: float = 1e-5
        weight_decay: float = 0.01
        warmup_steps: int = 5e3
        decay_steps: int = 5e6
        decay_factor: float = 0.95
        self_condition: bool = True
        compile: bool = False
        num_augment: int = 8
        eval_freq: int = 10
        eval_sample_num: int = 5
        eval_timesteps: int = 100
        eval_input_num: int = 50
        grad_clip_max_norm: float = 1.0
        grad_accum_steps: int = 256
        num_workers: int = 4
        prefetch_factor: int = 4
        seed: int = 0
        use_ema: bool = True
        ema_decay: float = 0.999
        bce_pos_weight: float = 8.0

    class Config(BaseModel):
        """Configuration for the ContactMapPredictor client."""

        data: "ContactMapPredictionClient.DataConfig"
        model: "ContactMapPredictionModel.Config"
        experiment: "ContactMapPredictionClient.ExperimentsConfig"

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.config = config
        self.set_seed(config.experiment.seed)
        self.register_model(ContactMapPredictionModel(config.model))

        if config.experiment.use_ema:
            self.add_callback(ModelEMA(config.experiment.ema_decay))

    def set_seed(self, seed: int) -> None:
        """Set the random seed for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)

    def loss_fn(self, noisy_batch: NoisyBatch) -> tuple[torch.Tensor, Mapping]:
        """Compute the loss given a noisy batch."""
        contact_map_logit = self.model.forward(noisy_batch)

        focal_loss = cal_contact_map_focal_loss(
            contact_map_logit,
            noisy_batch.structure.atom_pos,
            noisy_batch.structure.atom_pos_mask,
            noisy_batch.scheme.atom_to_residue_idx_map,
        )

        weighted_bce_loss = cal_contact_map_weighted_bce_loss(
            contact_map_logit,
            noisy_batch.structure.atom_pos,
            noisy_batch.structure.atom_pos_mask,
            noisy_batch.scheme.atom_to_residue_idx_map,
            pos_weight=self.config.experiment.bce_pos_weight,
        )

        loss = weighted_bce_loss

        # Long-range metrics (|i-j| >= min_seq_sep)
        min_seq_sep = 16
        long_range_precision = cal_long_range_precision(
            contact_map_logit,
            noisy_batch.structure.atom_pos,
            noisy_batch.structure.atom_pos_mask,
            noisy_batch.scheme.atom_to_residue_idx_map,
            min_seq_sep=min_seq_sep,
        )
        long_range_recall = cal_long_range_recall(
            contact_map_logit,
            noisy_batch.structure.atom_pos,
            noisy_batch.structure.atom_pos_mask,
            noisy_batch.scheme.atom_to_residue_idx_map,
            min_seq_sep=min_seq_sep,
        )
        long_range_f1 = cal_long_range_f1(
            contact_map_logit,
            noisy_batch.structure.atom_pos,
            noisy_batch.structure.atom_pos_mask,
            noisy_batch.scheme.atom_to_residue_idx_map,
            min_seq_sep=min_seq_sep,
        )
        long_range_auroc = cal_long_range_auroc(
            contact_map_logit,
            noisy_batch.structure.atom_pos,
            noisy_batch.structure.atom_pos_mask,
            noisy_batch.scheme.atom_to_residue_idx_map,
            min_seq_sep=min_seq_sep,
        )

        loss = weighted_bce_loss

        return loss, {
            "focal_loss": focal_loss.item(),
            "weighted_bce_loss": weighted_bce_loss.item(),
            "total_loss": loss.item(),
            "lr_precision": long_range_precision.mean().item(),
            "lr_recall": long_range_recall.mean().item(),
            "lr_f1": long_range_f1.mean().item(),
            "lr_auroc": long_range_auroc.item(),
        }

    def training_step(self, batch: Batch) -> dict[str, float]:
        """Train the model on a batch."""
        with precision_manager(self.model, self.config.model.precision):
            loss, loss_dict = self.loss_fn(batch)
            self.backward(loss)
            return loss_dict

    def validation_step(self, batch: Batch) -> dict[str, float]:
        """Valdiate the model on a batch."""
        # Note that when doing validation, we measure inference quality, not a loss.
        # Please keep in mind that batch is duplicated to eval_sample_num, sample quality
        # is measured by the best sample in the batch. Therefore the batch size should be
        # give as 1.
        if batch.shape[0] != 1:
            msg = "Batch size for validation must be 1."
            raise ValueError(msg)
        _, loss_dict = self.loss_fn(batch)
        return loss_dict

    @torch.no_grad()
    def predict_contact_map(
        self,
        batch: Batch,
        save_dir: Path | None = None,
    ) -> torch.Tensor:
        """Predict contact maps for the given batch."""
        self.model.eval()
        batch = batch.to(self.device)
        contact_map_logit = self.model.forward(batch)
        contact_map_prob = torch.softmax(contact_map_logit, dim=-1)[..., 1]

        contact_target, residue_pair_mask = extract_contact_map(
            batch.structure.atom_pos,
            batch.structure.atom_pos_mask,
            batch.scheme.atom_to_residue_idx_map,
        )

        if save_dir is not None:
            # save the contact map as an image
            contact_map_prob = contact_map_prob[0].cpu().numpy()
            contact_target = contact_target[0].cpu().numpy()
            residue_pair_mask = residue_pair_mask[0].cpu().numpy()
            contact_map_prob = contact_map_prob * residue_pair_mask
            save_path = save_dir / f"{batch.name[0]}_contact_map.png"
            if not save_dir.exists():
                save_dir.mkdir(parents=True, exist_ok=True)
            # 2 panels: predicted and target
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(contact_map_prob, cmap="Reds", vmin=0, vmax=1)
            axes[0].set_title("Predicted Contact Map")
            axes[1].imshow(contact_target, cmap="Reds", vmin=0, vmax=1)
            axes[1].set_title("Target Contact Map")
            plt.savefig(save_path)
            plt.close(fig)

        return contact_map_prob
