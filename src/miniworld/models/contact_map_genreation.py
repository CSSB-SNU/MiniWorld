import random
from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from pydantic import BaseModel
from team_gm import BaseClient
from team_gm.core.callbacks import ModelEMA
from torch import nn

from miniworld.data.dataloader.dataloader_multistate import (
    CropConfig,
    KmerFastAlignConfig,
    MSAConfig,
    MultistateConfig,
    MultiStatedbConfig,
)
from miniworld.data.features.features_biomol import Batch
from miniworld.loss.auxiliary import (
    cal_contact_map_focal_loss,
    cal_contact_map_weighted_bce_loss,
    cal_long_range_auroc,
    cal_long_range_f1,
    cal_long_range_precision,
    cal_long_range_recall,
    extract_contact_map,
)
from miniworld.modules.configs import (
    CommonConfig,
    DiffusionConfig,
)
from miniworld.modules.feature_embedder import InputFeatureEmbedder, fourier_embedding
from miniworld.modules.head import ContactMapHead
from miniworld.modules.msa_module import MSAModule
from miniworld.modules.pairformer import Pairformer
from miniworld.modules.primitives import (
    LayerNorm,
    Linear,
)
from miniworld.utils.diffusion.diffuser import D3PMDiffuser, SEDDDiffuser
from miniworld.utils.diffusion.scheduler import D3PMScheduler, SEDDScheduler
from miniworld.utils.diffusion.solver import DiffusionSolver, D3PMSolver, SEDDSolver
from miniworld.utils.precision_manager import PrecisionConfig, precision_manager


class ContactMapGenerationModel(nn.Module):
    """Structure Contact Map Generation model."""

    class ConditionConfig(BaseModel):
        """Configuration for condition modules."""

        pairformer: Pairformer.Config
        msa_module: MSAModule.Config
        n_recycle_max: int = 4

    class Config(BaseModel):
        """Configuration for the AF3 model."""

        common: CommonConfig
        trunk: "ContactMapGenerationModel.ConditionConfig"
        diffusion: DiffusionConfig
        precision: PrecisionConfig
        contact_num_classes: int = 2

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

        # Embed noisy contact map and time
        self.contact_map_embedder = nn.Linear(
            config.contact_num_classes,
            config.common.d_token_pair,
            bias=False,
        )
        self.tau_proj = nn.Sequential(
            LayerNorm(
                config.common.d_time,
                implementation=config.common.implementation,
            ),
            Linear(
                config.common.d_time,
                config.common.d_token_pair,
                bias=False,
            ),
        )

        # ContactMap prediction
        self.final_head = ContactMapHead(config.common)

    def forward(
        self,
        batch: Batch,
        noisy_contact_map: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass conditioned on noisy contact map and time embedding."""
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
        pair_mask = batch.structure.residue_mask[:, :, None] * batch.structure.residue_mask[
            :, None, :
        ]
        contact_pair = self.contact_map_embedder(
            noisy_contact_map.to(token_pair_init.dtype),
        )
        token_pair_init = token_pair_init + contact_pair * pair_mask.unsqueeze(-1)
        tau_embed = fourier_embedding(tau).to(token_pair_init.dtype)
        tau_embed = self.tau_proj(tau_embed)
        token_pair_init = token_pair_init + tau_embed[:, None, None, :]

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

        return self.final_head(token_pair)


class ContactMapGenerationClient(BaseClient):
    """Client for training and inference of contact map generation model."""

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
        long_range_min_seq_sep: int | None = None
        long_range_sigmoid_k: float | None = None
        long_range_sigmoid_amp: float | None = None

    class DiscreteDiffusionConfig(BaseModel):
        """Configuration for discrete diffusion contact map generation."""

        method: Literal["none", "d3pm", "sedd"] = "none"
        d3pm_scheduler: D3PMScheduler.D3PMSchedulerConfig | None = None
        d3pm_diffuser: D3PMDiffuser.D3PMConfig | None = None
        d3pm_solver: DiffusionSolver.SolverConfig | None = None
        sedd_scheduler: SEDDScheduler.SEDDSchedulerConfig | None = None
        sedd_diffuser: SEDDDiffuser.SEDDConfig | None = None
        sedd_solver: DiffusionSolver.SolverConfig | None = None

    class Config(BaseModel):
        """Configuration for the ContactMapGenerator client."""

        data: "ContactMapGenerationClient.DataConfig"
        model: "ContactMapGenerationModel.Config"
        experiment: "ContactMapGenerationClient.ExperimentsConfig"
        discrete_diffusion: (
            "ContactMapGenerationClient.DiscreteDiffusionConfig" | None
        ) = None

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.config = config
        self.set_seed(config.experiment.seed)
        self.register_model(ContactMapGenerationModel(config.model))
        self.discrete_diffusion = None
        if (
            config.discrete_diffusion is not None
            and config.discrete_diffusion.method != "none"
        ):
            self.discrete_diffusion = self._init_discrete_diffusion(
                config.discrete_diffusion,
            )

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
    ) -> dict[str, object]:
        """Instantiate scheduler/diffuser/solver triplet for generation."""
        if cfg.method == "sedd":
            required = (cfg.sedd_scheduler, cfg.sedd_diffuser, cfg.sedd_solver)
            if any(item is None for item in required):
                msg = "SEDD method requires scheduler, diffuser, and solver configs."
                raise ValueError(msg)
            scheduler = SEDDScheduler(cfg.sedd_scheduler)
            diffuser = SEDDDiffuser(cfg.sedd_diffuser, scheduler)
            if (
                cfg.sedd_diffuser.num_classes
                != self.config.model.contact_num_classes
            ):
                msg = "SEDD diffuser num_classes must match model.contact_num_classes (ContactMapHead outputs 2)."
                raise ValueError(msg)
            solver = SEDDSolver(
                cfg.sedd_solver,
                scheduler,
                num_classes=cfg.sedd_diffuser.num_classes,
                enforce_symmetric=cfg.sedd_diffuser.enforce_symmetric,
                min_ratio=cfg.sedd_diffuser.min_ratio,
            )
            return {
                "method": cfg.method,
                "scheduler": scheduler,
                "diffuser": diffuser,
                "solver": solver,
            }
        if cfg.method == "d3pm":
            required = (cfg.d3pm_scheduler, cfg.d3pm_diffuser, cfg.d3pm_solver)
            if any(item is None for item in required):
                msg = "D3PM method requires scheduler, diffuser, and solver configs."
                raise ValueError(msg)
            scheduler = D3PMScheduler(cfg.d3pm_scheduler)
            diffuser = D3PMDiffuser(cfg.d3pm_diffuser, scheduler)
            if (
                cfg.d3pm_diffuser.num_classes
                != self.config.model.contact_num_classes
            ):
                msg = "D3PM diffuser num_classes must match model.contact_num_classes (ContactMapHead outputs 2)."
                raise ValueError(msg)
            solver = D3PMSolver(
                cfg.d3pm_solver,
                scheduler,
                num_classes=cfg.d3pm_diffuser.num_classes,
                enforce_symmetric=cfg.d3pm_diffuser.enforce_symmetric,
            )
            return {
                "method": cfg.method,
                "scheduler": scheduler,
                "diffuser": diffuser,
                "solver": solver,
            }
        msg = f"Unknown discrete diffusion method: {cfg.method}"
        raise ValueError(msg)

    def loss_fn(self, batch: Batch) -> tuple[torch.Tensor, Mapping]:
        """Compute discrete diffusion loss with noisy contact maps."""
        if self.discrete_diffusion is None:
            msg = "Discrete diffusion must be configured for generation."
            raise RuntimeError(msg)
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

        contact_target, residue_pair_mask = extract_contact_map(
            batch.structure.atom_pos,
            batch.structure.atom_pos_mask,
            batch.scheme.atom_to_residue_idx_map,
        )
        contact_labels = contact_target.long()
        dd = self.discrete_diffusion

        xt_one_hot, _, tau = dd["diffuser"].sample(
            contact_labels,
            residue_pair_mask,
        )
        contact_pred = self.model.forward(batch, xt_one_hot, tau)
        loss = dd["diffuser"].cal_loss(contact_pred)

        if dd["method"] == "d3pm":
            contact_map_logit = contact_pred
        else:
            contact_map_logit = torch.log(contact_pred.clamp_min(1e-8))

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

        aux_bce = cal_contact_map_weighted_bce_loss(
            contact_map_logit,
            batch.structure.atom_pos,
            batch.structure.atom_pos_mask,
            batch.scheme.atom_to_residue_idx_map,
            pos_weight=self.config.experiment.bce_pos_weight,
            long_range_min_seq_sep=lr_min_seq_sep,
            long_range_sigmoid_k=lr_sigmoid_k,
            long_range_sigmoid_amp=lr_sigmoid_amp,
        )

        return loss, {
            "focal_loss": focal_loss.item(),
            "bce_metric": aux_bce.item(),
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
    def generate_contact_map(
        self,
        batch: Batch,
        num_steps: int | None = None,
        save_dir: Path | None = None,
    ) -> torch.Tensor:
        """Generate contact maps via discrete diffusion solver."""
        if self.discrete_diffusion is None:
            msg = "Discrete diffusion must be configured for generation."
            raise RuntimeError(msg)
        self.model.eval()
        batch = batch.to(self.device)
        dd = self.discrete_diffusion
        solver = dd["solver"]
        steps = num_steps or self.config.experiment.eval_timesteps

        pair_mask = batch.structure.residue_mask[:, :, None] * batch.structure.residue_mask[
            :, None, :
        ]

        def model_fn(xt_one_hot: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
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
        if isinstance(sample, tuple):
            sample_labels = sample[0]
        else:
            sample_labels = sample
        contact_map_prob = F.one_hot(
            sample_labels,
            num_classes=self.config.model.contact_num_classes,
        ).float()
        contact_map_prob = contact_map_prob * pair_mask.unsqueeze(-1)

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

        return contact_map_prob
