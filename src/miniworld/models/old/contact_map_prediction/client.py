import random
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from matplotlib import pyplot as plt
from pydantic import BaseModel
from team_gm import BaseClient
from team_gm.core.callbacks import ModelEMA
from team_gm.utils.precision_manager import precision_manager

from miniworld.data.dataloader.configs import (
    CropConfig,
    MSAConfig,
    MultistateConfig,
    MultiStateDBConfig,
)
from miniworld.data.features.batch_multistate import Batch
from miniworld.loss.auxiliary import (
    cal_atom_distogram_loss,
    cal_contact_map_focal_loss,
    cal_contact_map_weighted_bce_loss,
    cal_long_range_auroc,
    cal_long_range_f1,
    cal_long_range_precision,
    cal_long_range_recall,
    extract_contact_map,
)

from .model import ContactMapPredictionModel


class ContactMapPredictionClient(BaseClient):
    """Client for training and inference of ContactMapPredictor model."""

    class DataConfig(BaseModel):
        """Configuration for data loading."""

        crop: CropConfig
        msa: MSAConfig
        multistate: MultistateConfig
        train_preprocessing: MultiStateDBConfig
        valid_preprocessing: MultiStateDBConfig

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
        warmup_steps: int = int(5e3)
        decay_steps: int = int(5e6)
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
        long_range_min_seq_sep: int | None = None
        long_range_sigmoid_k: float | None = None
        long_range_sigmoid_amp: float | None = None

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

    def loss_fn(self, batch: Batch) -> tuple[torch.Tensor, dict]:
        """Compute the loss given a noisy batch."""
        # Long-range weighting hyperparameters (fallback to function defaults)
        lr_min_seq_sep = (
            self.config.experiment.long_range_min_seq_sep
            if self.config.experiment.long_range_min_seq_sep is not None
            else 16
        )
        lr_sigmoid_k = (
            self.config.experiment.long_range_sigmoid_k
            if self.config.experiment.long_range_sigmoid_k is not None
            else 1.0
        )
        lr_sigmoid_amp = (
            self.config.experiment.long_range_sigmoid_amp
            if self.config.experiment.long_range_sigmoid_amp is not None
            else 0.0
        )

        if self.config.model.use_distogram:
            distogram_logit = self.model.forward(batch)
            contact_logit = torch.logsumexp(distogram_logit[..., :13], dim=-1)
            noncontact_logit = torch.logsumexp(distogram_logit[..., 13:], dim=-1)
            contact_map_logit = torch.stack([noncontact_logit, contact_logit], dim=-1)

            loss = cal_atom_distogram_loss(
                distogram_logit,
                batch.structure.atom_pos,
                batch.structure.atom_pos_mask,
                batch.scheme.atom_to_residue_idx_map,
            )
        else:
            contact_map_logit = self.model.forward(batch)
            loss = cal_contact_map_weighted_bce_loss(
                contact_map_logit,
                batch.structure.atom_pos,
                batch.structure.atom_pos_mask,
                batch.scheme.atom_to_residue_idx_map,
                pos_weight=self.config.experiment.bce_pos_weight,
                long_range_min_seq_sep=lr_min_seq_sep,
                long_range_sigmoid_k=lr_sigmoid_k,
                long_range_sigmoid_amp=lr_sigmoid_amp,
            )

        focal_loss = cal_contact_map_focal_loss(
            contact_map_logit,
            batch.structure.atom_pos,
            batch.structure.atom_pos_mask,
            batch.scheme.atom_to_residue_idx_map,
        )

        # Long-range metrics (|i-j| >= min_seq_sep)
        min_seq_sep = lr_min_seq_sep
        long_range_precision = cal_long_range_precision(
            contact_map_logit,
            batch.structure.atom_pos,
            batch.structure.atom_pos_mask,
            batch.scheme.atom_to_residue_idx_map,
            min_seq_sep=min_seq_sep,
        )
        long_range_recall = cal_long_range_recall(
            contact_map_logit,
            batch.structure.atom_pos,
            batch.structure.atom_pos_mask,
            batch.scheme.atom_to_residue_idx_map,
            min_seq_sep=min_seq_sep,
        )
        long_range_f1 = cal_long_range_f1(
            contact_map_logit,
            batch.structure.atom_pos,
            batch.structure.atom_pos_mask,
            batch.scheme.atom_to_residue_idx_map,
            min_seq_sep=min_seq_sep,
        )
        long_range_auroc = cal_long_range_auroc(
            contact_map_logit,
            batch.structure.atom_pos,
            batch.structure.atom_pos_mask,
            batch.scheme.atom_to_residue_idx_map,
            min_seq_sep=min_seq_sep,
        )

        return loss, {
            "focal_loss": focal_loss.item(),
            "main_loss": loss.item(),
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
            contact_map_prob_np = contact_map_prob[0].cpu().numpy()
            contact_target = contact_target[0].cpu().numpy()
            residue_pair_mask = residue_pair_mask[0].cpu().numpy()
            contact_map_prob_np = contact_map_prob_np * residue_pair_mask
            save_path = save_dir / f"{batch.name[0]}_contact_map.png"
            if not save_dir.exists():
                save_dir.mkdir(parents=True, exist_ok=True)
            # 2 panels: predicted and target
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(contact_map_prob_np, cmap="Reds", vmin=0, vmax=1)
            axes[0].set_title("Predicted Contact Map")
            axes[1].imshow(contact_target, cmap="Reds", vmin=0, vmax=1)
            axes[1].set_title("Target Contact Map")
            plt.savefig(save_path)
            plt.close(fig)

        return contact_map_prob
