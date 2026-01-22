import random
from pathlib import Path
from typing import Literal, cast

import numpy as np
import torch
import torch.nn.functional as F
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
    cal_long_range_f1,
    cal_long_range_precision,
    cal_long_range_recall,
    extract_contact_map,
)

from .diffusion import (
    D3PMDiffuser,
    D3PMScheduler,
    D3PMSolver,
    SEDDDiffuser,
    SEDDScheduler,
    SEDDSolver,
)
from .model import ContactMapGenerationModel


class ContactMapGenerationClient(BaseClient):
    """Client for training and inference of contact map generation model."""

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
        name: str = "ContactMapGenerator"
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

    class DiscreteDiffusionConfig(BaseModel):
        """Configuration for discrete diffusion contact map generation."""

        method: Literal["none", "d3pm", "sedd"] = "none"
        transition_mode: Literal["absorbing", "other"] = "absorbing"
        scheduler: D3PMScheduler.D3PMSchedulerConfig | SEDDScheduler.SEDDSchedulerConfig
        diffuser: D3PMDiffuser.D3PMConfig | SEDDDiffuser.SEDDConfig
        solver: D3PMSolver.D3PMSolverConfig | SEDDSolver.SEDDSolverConfig

    class Config(BaseModel):
        """Configuration for the ContactMapGenerator client."""

        data: "ContactMapGenerationClient.DataConfig"
        model: "ContactMapGenerationModel.Config"
        experiment: "ContactMapGenerationClient.ExperimentsConfig"
        discrete_diffusion: "ContactMapGenerationClient.DiscreteDiffusionConfig"

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.config = config
        self.set_seed(config.experiment.seed)
        self.register_model(ContactMapGenerationModel(config.model, config.discrete_diffusion.transition_mode))
        if (
            config.discrete_diffusion is not None
            and config.discrete_diffusion.method != "none"
        ):
            self._init_discrete_diffusion(config.discrete_diffusion)

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

    def _init_discrete_diffusion(
        self,
        cfg: "ContactMapGenerationClient.DiscreteDiffusionConfig",
    ) -> None:
        """Instantiate scheduler/diffuser/solver triplet for generation."""
        if cfg.method == "sedd":
            required = (cfg.scheduler, cfg.diffuser, cfg.solver)
            if any(item is None for item in required):
                msg = "SEDD method requires scheduler, diffuser, and solver configs."
                raise ValueError(msg)
            cfg.scheduler = cast("SEDDScheduler.SEDDSchedulerConfig", cfg.scheduler)
            cfg.diffuser = cast("SEDDDiffuser.SEDDConfig", cfg.diffuser)
            cfg.solver = cast("SEDDSolver.SEDDSolverConfig", cfg.solver)
            scheduler = SEDDScheduler(cfg.scheduler)
            diffuser = SEDDDiffuser(cfg.diffuser, scheduler)
            if scheduler.config.transition_mode == "absorbing" and scheduler.num_classes - 1 != self.config.model.contact_num_classes:
                    msg = (
                        "For absorbing SEDD, diffuser num_classes must be"
                        " contact_num_classes + 1 (ContactMapHead outputs 2)."
                    )
                    raise ValueError(msg)
            if scheduler.num_classes != self.config.model.contact_num_classes:
                msg = "SEDD diffuser num_classes must match model.contact_num_classes (ContactMapHead outputs 2)."
                raise ValueError(msg)
            solver = SEDDSolver(
                cfg.solver,
                scheduler,
            )
            self.diffusion_scheduler = scheduler
            self.diffusion_diffuser = diffuser
            self.diffusion_solver = solver
            return
        if cfg.method == "d3pm":
            required = (cfg.scheduler, cfg.diffuser, cfg.solver)
            if any(item is None for item in required):
                msg = "D3PM method requires scheduler, diffuser, and solver configs."
                raise ValueError(msg)
            cfg.scheduler = cast("D3PMScheduler.D3PMSchedulerConfig", cfg.scheduler)
            cfg.diffuser = cast("D3PMDiffuser.D3PMConfig", cfg.diffuser)
            cfg.solver = cast("D3PMSolver.D3PMSolverConfig", cfg.solver)
            scheduler = D3PMScheduler(cfg.scheduler)
            diffuser = D3PMDiffuser(cfg.diffuser, scheduler)
            solver = D3PMSolver(
                cfg.solver,
                scheduler,
            )
            self.diffusion_scheduler = scheduler
            self.diffusion_diffuser = diffuser
            self.diffusion_solver = solver
            return
        msg = f"Unknown discrete diffusion method: {cfg.method}"
        raise ValueError(msg)

    def loss_fn(self, batch: Batch) -> tuple[torch.Tensor, dict]:
        """Compute discrete diffusion loss with noisy contact maps."""
        contact_target, residue_pair_mask = extract_contact_map(
            batch.structure.atom_pos,
            batch.structure.atom_pos_mask,
            batch.scheme.atom_to_residue_idx_map,
        )
        contact_labels = contact_target.long()

        sample_output = self.diffusion_diffuser.sample(
            contact_labels,
            residue_pair_mask,
        )
        xt_one_hot = sample_output.xt_one_hot
        tau = sample_output.t_emb
        model_output = self.model.forward(
            batch,
            xt_one_hot,
            tau,
        )  # d3pm : logit | sedd : ratio
        loss = self.diffusion_diffuser.cal_loss(
            sample_output, # pyright: ignore[reportArgumentType]
            model_output,
        )

        return loss, {
            "main_loss": loss.item(),
            "total_loss": loss.item(),
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

        return self.generate_contact_map(batch)

    @torch.no_grad()
    def generate_contact_map(
        self,
        batch: Batch,
        num_steps: int | None = None,
        save_dir: Path | None = None,
    ) -> dict[str, float]:
        """Generate contact maps via discrete diffusion solver."""
        self.model.eval()
        batch = batch.to(self.device)
        solver = self.diffusion_solver
        steps = num_steps or self.config.experiment.eval_timesteps

        pair_mask = (
            batch.structure.residue_mask[:, :, None]
            * batch.structure.residue_mask[
                :,
                None,
                :,
            ]
        )

        def model_fn(xt_one_hot: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
            try:
                output = self.model.forward(batch, xt_one_hot, tau)
            except:
                # save batch for debugging
                torch.save(batch, "debug_batch.pt")
                torch.save(xt_one_hot, "debug_xt_one_hot.pt")
                torch.save(tau, "debug_tau.pt")
                raise ValueError("Error in model forward during sampling. Debug data saved.")
            return self.model.forward(batch, xt_one_hot, tau)

        sample = solver.sample(
            model_fn,
            shape=torch.Size(
                (
                    batch.shape[0],
                    batch.residue_length,
                    batch.residue_length,
                ),
            ),
            num_steps=steps,
            device=self.device,
        )
        sample_labels = sample[0] if isinstance(sample, tuple) else sample

        sample_labels[sample_labels >= self.config.model.contact_num_classes] = 0.0

        contact_map_prob = F.one_hot(
            sample_labels,
            num_classes=self.config.model.contact_num_classes,
        ).float()
        contact_map_prob = contact_map_prob * pair_mask.unsqueeze(-1)

        # Long-range weighting hyperparameters (fallback to function defaults)
        lr_min_seq_sep = (
            self.config.experiment.long_range_min_seq_sep
            if self.config.experiment.long_range_min_seq_sep is not None
            else 16
        )

        contact_map_logit = contact_map_prob  # (B, L, L, C)

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

        if save_dir is not None:
            contact_map_np = contact_map_prob[0, ..., 1].cpu().numpy()
            mask_np = pair_mask[0].cpu().numpy()
            contact_map_np = contact_map_np * mask_np
            save_path = save_dir / f"{batch.name[0]}_contact_map.png"
            if not save_dir.exists():
                save_dir.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.imshow(contact_map_np, cmap="Reds", vmin=0, vmax=1)
            ax.set_title("Generated Contact Map")
            plt.savefig(save_path)
            plt.close(fig)

        return {
            "long_range_precision": long_range_precision.mean().item(),
            "long_range_recall": long_range_recall.mean().item(),
            "long_range_f1": long_range_f1.mean().item(),
        }

        return {"test" : 0.0}
