import random
from collections.abc import Generator, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
from lightning.fabric.wrappers import _FabricDataLoader
from pydantic import BaseModel
from team_gm import BaseClient
from team_gm.core.callbacks import ModelEMA
from team_gm.modules import Pairformer
from team_gm.utils import data_utils as du
from torch import nn
from torch.utils.data import DataLoader

from miniworld.data.dataloader.dataloader_edge_backprop import (
    AdaptiveEdgeSampler,
    BioMolDBConfig,
    CropConfig,
    EdgeWeightConfig,
    MSAConfig,
)
from miniworld.data.features.features_biomol import Batch, NoisyBatch
from miniworld.loss import metrics  # , losses
from miniworld.loss.auxiliary import (
    cal_contact_map_focal_loss,
    cal_contact_map_weighted_bce_loss,
    cal_long_range_auroc,
    cal_long_range_f1,
    cal_long_range_precision,
    cal_long_range_recall,
)
from miniworld.modules.configs import (
    CommonConfig,
    DiffusionConfig,
)
from miniworld.modules.diffusion_module import (
    DiffusionModule,
)
from miniworld.modules.feature_embedder import InputFeatureEmbedder
from miniworld.modules.heads import ContactMapHead
from miniworld.modules.msa_module import MSAModule
from miniworld.modules.primitives import (
    LayerNorm,
    Linear,
)
from miniworld.utils.diffusion.diffuser import EuclideanDiffuser
from miniworld.utils.diffusion.scheduler import EDMScheduler
from miniworld.utils.diffusion.solver import AF3Solver
from miniworld.utils.precision_manager import PrecisionConfig, precision_manager


class ContactMapEmbedder(nn.Module):
    """ContactMap Embedder."""

    def __init__(
        self,
        common_config: CommonConfig,
        pairformer_config: Pairformer.Config,
    ) -> None:
        super().__init__()

        self.to_pair = nn.Sequential(
            LayerNorm(
                2,
                implementation=common_config.implementation,
            ),
            Linear(
                2,
                common_config.d_token_pair,
                init="zero",
            ),
        )
        self.pairformer = Pairformer(pairformer_config)

    def forward(
        self,
        contact_map: torch.Tensor,  # (B, L, L, 2)
        token_single_init: torch.Tensor,  # (B, L, d_single)
        token_pair_init: torch.Tensor,  # (B, L, L, d_pair)
        token_mask: torch.Tensor,  # (B, L), bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of ContactMapEmbedder."""
        token_pair = token_pair_init + self.to_pair(contact_map)  # (B, L, L, d_pair)
        token_pair, token_single = self.pairformer(
            token_pair,
            single=token_single_init,
            mask=token_mask,
        )  # (B, L, L, d_pair), (B, L, d_single)
        return token_pair, token_single  # (B, L, L, d_pair)


class MiniWorldModel(nn.Module):
    """Structure MiniWorld model."""

    class ConditionConfig(BaseModel):
        """Configuration for condition modules."""

        pairformer: Pairformer.Config
        msa_module: MSAModule.Config
        n_recycle_max: int = 4

    class ContactMapConfig(BaseModel):
        """Configuration for contact map embedding."""

        pairformer: Pairformer.Config

    class Config(BaseModel):
        """Configuration for the MiniWorld model."""

        common: CommonConfig
        trunk: "MiniWorldModel.ConditionConfig"
        diffusion: DiffusionConfig
        precision: PrecisionConfig
        contact_map: "MiniWorldModel.ContactMapConfig"

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
        self.add_single_recycle = nn.Sequential(
            LayerNorm(
                config.common.d_token_single,
                implementation=config.common.implementation,
            ),
            Linear(
                config.common.d_token_single,
                config.common.d_token_single,
                init="zero",
            ),
        )

        # Trunk forward
        self.msa_module = MSAModule(config.common, config.trunk.msa_module)
        self.pairformer_blocks = Pairformer(config.trunk.pairformer)
        self.contact_map_head = ContactMapHead(config.common)

        self.contact_map_embedder = ContactMapEmbedder(
            config.common,
            config.contact_map.pairformer,
        )

        # Diffusion module
        self.diffusion_module = DiffusionModule(config.common, config.diffusion)

    def condition_forward(self, noisy_batch: NoisyBatch) -> tuple[torch.Tensor, ...]:
        """Forward pass of the condition modules with recycling."""
        if self.training:
            n_recycle = random.randint(1, self.n_recycle_max)
        else:
            n_recycle = self.n_recycle_max
        if noisy_batch.msa.aligned_sequences.shape[1] != self.n_recycle_max:
            msg = (
                "The number of MSA sequences should match the number of recycle steps."
            )
            raise ValueError(msg)

        # input feature embedding
        (
            token_single_input,
            token_single_init,
            token_pair_init,
        ) = self.input_feature_embedder(noisy_batch)

        token_pair = torch.zeros_like(token_pair_init)
        token_single = torch.zeros_like(token_single_init)
        # Trunk forward with recycling
        for i_cycle in range(n_recycle):
            with ExitStack() as stack:
                if i_cycle < n_recycle - 1:
                    stack.enter_context(torch.no_grad())
                    stack.enter_context(torch.inference_mode())
                token_pair = token_pair_init + self.add_pair_recycle(token_pair)

                token_pair = token_pair + self.msa_module(
                    noisy_batch,
                    i_cycle,
                    token_pair,
                    token_single_input,
                    noisy_batch.structure.residue_mask,
                )
                token_single = token_single_init + self.add_single_recycle(token_single)

                token_pair, token_single = self.pairformer_blocks.forward(
                    token_pair,
                    token_single,
                    noisy_batch.structure.residue_mask,
                )
        # reduce token_pair information to contact map
        contact_map_logit = self.contact_map_head(token_pair)

        # grad hack
        token_single_init = token_single_init + token_single * 0.0

        # expand contact map information to pair representation
        token_pair_exp, token_single_exp = self.contact_map_embedder(
            contact_map_logit.detach(),
            token_single_init,
            token_pair_init,
            noisy_batch.structure.residue_mask,
        )

        return (
            token_single_input,
            token_single_exp,
            token_pair_exp,
            contact_map_logit,
        )

    def diffusion_forward(
        self,
        noisy_batch: NoisyBatch,
        token_single_input: torch.Tensor,
        token_single_trunk: torch.Tensor,
        token_pair_trunk: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the diffusion module.

        Parameters
        ----------
        noisy_batch: NoisyBatch
            Batch of noisy data.
        token_single_input: FloatTensor, (B, L, d_single)
            Input single representation.
        token_single_trunk: FloatTensor, (B, L, d_single)
            Single representation after trunk forward.
        token_pair_trunk: FloatTensor, (B, L, L, d_pair)
            Pair representation after trunk forward.
        atom_single_cond: FloatTensor, (B, L, d_atom_single)
            Atom single condition representation.
        atom_pair: FloatTensor, (B, L, L, d_atom_pair)
            Atom pair representation.

        """
        return self.diffusion_module(
            noisy_batch,
            token_single_input,
            token_single_trunk,
            token_pair_trunk,
        )

    def forward(self, noisy_batch: NoisyBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the MiniWorld model."""
        (
            token_single_input,
            token_single_trunk,
            token_pair_trunk,
            contact_map_logit,
        ) = self.condition_forward(noisy_batch)
        # Diffusion forward
        atom_pos_update = self.diffusion_forward(
            noisy_batch,
            token_single_input,
            token_single_trunk,
            token_pair_trunk,
        )

        return atom_pos_update, contact_map_logit


class MiniWorldModelWrapper(nn.Module):
    """Wrapper for MiniWorldModel to handle the input and output using solver."""

    def __init__(self, model: MiniWorldModel, use_self_condition: bool = True) -> None:
        super().__init__()
        self.batch_loaded = False
        self.conditioned_forwarded = False
        self.model = model
        self.use_self_condition = use_self_condition
        self.z_sc = None  # Placeholder for self-conditioned input

    def load_batch(self, batch: Batch) -> None:
        """Load a new batch to the model."""
        self.batch = batch
        self.z_sc = None
        self.batch_loaded = True

    def prepare_condition(self, batch: Batch) -> None:
        """Prepare the model for conditioned forward pass."""
        if self.batch_loaded:
            msg = "Batch is already loaded. Please create a new MiniWorldModelWrapper instance for a new batch."
            raise ValueError(msg)
        if self.conditioned_forwarded:
            msg = "Conditioned forward is already done. Please create a new MiniWorldModelWrapper instance for a new batch."
            raise ValueError(msg)

        # Load the batch and prepare the model for conditioned forward pass
        self.load_batch(batch)
        (
            token_single_input,
            token_single_trunk,
            token_pair_trunk,
            contact_map_logit,
        ) = self.model.condition_forward(self.batch)
        self.conditioned_forwarded = True

        self.condition = {
            "token_single_input": token_single_input,
            "token_single_trunk": token_single_trunk,
            "token_pair_trunk": token_pair_trunk,
            "contact_map_logit": contact_map_logit,
        }

    def forward(self, z_i: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model wrapper."""
        if not self.batch_loaded or not self.conditioned_forwarded:
            msg = "Batch must be loaded and conditioned forward must be called before forward pass."
            raise ValueError(msg)

        n_str = z_i.shape[0]
        x_mask = self.batch.structure.atom_mask.repeat(n_str, 1).unsqueeze(0)
        noisy_batch = NoisyBatch(
            **self.batch.__dict__,
            x_t=z_i.unsqueeze(0),  # (B, L, 3) -> (1, B, L, 3)
            t=t_emb[None, None, None, None],  # (,) -> (1, 1, 1, 1)
            x_sc=self.z_sc,
            x_mask=x_mask,
        )

        z_update = self.model.diffusion_forward(
            noisy_batch,
            self.condition["token_single_input"],
            self.condition["token_single_trunk"],
            self.condition["token_pair_trunk"],
        )
        z_update = z_update.squeeze(0)  # (1, B, L, 3) -> (B, L, 3)
        if self.use_self_condition:
            self.z_sc = z_update

        return z_update


@dataclass
class MiniWorldInferenceOutput:
    """Output of the MiniWorld model inference."""

    # Tensor of final predicted atom coordinate.
    atom_pos_pred: torch.Tensor  # (B, L, 3)

    # Contact map logits
    contact_map_logit: torch.Tensor  # (B, L, L, 2)

    # Array of predicted atom coordinate trajectory for timesteps T.
    model_traj: np.ndarray  # (B, T, L, 3)

    # Array of interpolant predicted atom coordinate trajectory for timesteps T.
    inter_traj: np.ndarray  # (B, T, L, 3)

    batch: Batch


class MiniWorldClient(BaseClient):
    """Client for training and inference of MiniWorld model."""

    class DataConfig(BaseModel):
        """Configuration for data loading."""

        train_db: BioMolDBConfig
        valid_db: BioMolDBConfig
        edge_weight: EdgeWeightConfig
        crop: CropConfig
        msa: MSAConfig

    class LossConfig(BaseModel):
        """Configuration for loss weights."""

        diffusion_loss: float = 4.0
        contact_map_loss: float = 0.03

    class ExperimentsConfig(BaseModel):
        """Configuration for experiments."""

        comment: str = "default"
        name: str = "MiniWorld-PSK-2"
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
        bce_pos_weight: float = 2.0
        long_range_min_seq_sep: int | None = None
        long_range_sigmoid_k: float | None = None
        long_range_sigmoid_amp: float = 3.0

    class DiffuserConfig(BaseModel):
        """Configuration for the diffuser."""

        seed: int = 0
        scheduler: EDMScheduler.EDMSchedulerConfig
        method: Literal["AF3", "EDM"] = "AF3"

    class Config(BaseModel):
        """Configuration for the MiniWorld client."""

        data: "MiniWorldClient.DataConfig"
        model: MiniWorldModel.Config
        experiment: "MiniWorldClient.ExperimentsConfig"
        diffuser: "MiniWorldClient.DiffuserConfig"
        loss: "MiniWorldClient.LossConfig"

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.config = config
        self.set_seed(config.experiment.seed)
        self.register_model(MiniWorldModel(config.model))

        if config.experiment.use_ema:
            self.add_callback(ModelEMA(config.experiment.ema_decay))
        diffuser_method = config.diffuser.method
        if diffuser_method == "AF3":
            self.diffusion_scheduler = EDMScheduler(config.diffuser.scheduler)
            self.diffuser = EuclideanDiffuser(
                config=EuclideanDiffuser.EuclideanConfig(
                    seed=config.diffuser.seed,
                ),
                scheduler=self.diffusion_scheduler,
            )
            self.solver = AF3Solver(
                config=AF3Solver.SolverConfig(seed=config.diffuser.seed),
                scheduler=self.diffusion_scheduler,
            )
        else:
            msg = f"Diffuser method {diffuser_method} is not implemented yet."
            raise NotImplementedError(msg)

    def set_seed(self, seed: int) -> None:
        """Set the random seed for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)

    def contact_map_quality(
        self,
        noisy_batch: NoisyBatch,
        contact_map_logit: torch.Tensor,
    ) -> Mapping:
        """Compute contact map quality metrics."""
        # Placeholder for contact map quality metrics
        # Long-range weighting hyperparameters (fallback to function defaults)
        lr_min_seq_sep = (
            self.config.experiment.long_range_min_seq_sep
            if self.config.experiment.long_range_min_seq_sep is not None
            else 16
        )

        focal_loss = cal_contact_map_focal_loss(
            contact_map_logit,
            noisy_batch.structure.atom_pos,
            noisy_batch.structure.atom_pos_mask,
            noisy_batch.scheme.atom_to_residue_idx_map,
        )

        # Long-range metrics (|i-j| >= min_seq_sep)
        min_seq_sep = lr_min_seq_sep
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
        return {
            "focal_loss": focal_loss.item(),
            "lr_precision": long_range_precision.mean().item(),
            "lr_recall": long_range_recall.mean().item(),
            "lr_f1": long_range_f1.mean().item(),
            "lr_auroc": long_range_auroc.item(),
        }

    def loss_fn(self, noisy_batch: NoisyBatch) -> tuple[torch.Tensor, Mapping]:
        """Compute the loss given a noisy batch."""
        atom_pos_update, contact_map_logit = self.model.forward(noisy_batch)

        structure_loss = self.diffuser.cal_loss(atom_pos_update)

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

        contact_map_loss = cal_contact_map_weighted_bce_loss(
            contact_map_logit,
            noisy_batch.structure.atom_pos,
            noisy_batch.structure.atom_pos_mask,
            noisy_batch.scheme.atom_to_residue_idx_map,
            pos_weight=self.config.experiment.bce_pos_weight,
            long_range_min_seq_sep=lr_min_seq_sep,
            long_range_sigmoid_k=lr_sigmoid_k,
            long_range_sigmoid_amp=lr_sigmoid_amp,
        )

        loss = (
            self.config.loss.diffusion_loss * structure_loss
            + self.config.loss.contact_map_loss * contact_map_loss
        )

        contact_map_quality_dict = self.contact_map_quality(
            noisy_batch,
            contact_map_logit,
        )
        focal_loss = contact_map_quality_dict["focal_loss"]
        lr_precision = contact_map_quality_dict["lr_precision"]
        lr_recall = contact_map_quality_dict["lr_recall"]
        lr_f1 = contact_map_quality_dict["lr_f1"]
        lr_auroc = contact_map_quality_dict["lr_auroc"]

        return loss, {
            "diffusion_loss": structure_loss.item(),
            "contact_map_loss": contact_map_loss.item(),
            "total_loss": loss.item(),
            "main_loss": loss.item(),
            "focal_loss": focal_loss,
            "lr_precision": lr_precision,
            "lr_recall": lr_recall,
            "lr_f1": lr_f1,
            "lr_auroc": lr_auroc,
        }

    def training_step(self, batch: Batch) -> dict[str, float]:
        """Train the model on a batch."""
        with precision_manager(self.model, self.config.model.precision):
            num_augment = self.config.experiment.num_augment
            noisy_atom_pos, x_mask, t_emb = self.diffuser.sample(
                batch.structure.atom_pos,
                num_augment=num_augment,
                mask=batch.structure.atom_mask,
            )
            noisy_batch = NoisyBatch(
                **batch.__dict__,
                t=t_emb,
                x_t=noisy_atom_pos,
                x_mask=x_mask,
            )

            if self.config.experiment.self_condition and random.random() > 0.5:
                with torch.no_grad():
                    atom_pos_update = self.model.forward(noisy_batch)
                    noisy_batch.x_sc = atom_pos_update
            loss, loss_dict = self.loss_fn(noisy_batch)

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
        batch = batch.duplicate(self.config.experiment.eval_sample_num)
        return self.test_inference_quality(
            batch,
            self.config.experiment.eval_timesteps,
        )

    def training_epoch(self, dataloader: DataLoader) -> Generator[Any, None, None]:
        """Yield results from training step over the dataloader for one epoch."""
        if not isinstance(dataloader, _FabricDataLoader):
            fabric_dataloader = self.fabric.setup_dataloaders(dataloader)
        else:
            fabric_dataloader = dataloader
        sampler: AdaptiveEdgeSampler = dataloader.sampler

        self.model.train()
        self.call_callbacks("on_train_epoch_start")

        fabric_iter = iter(fabric_dataloader)
        for batch_idx, batch in enumerate(fabric_iter):
            self.call_callbacks("on_train_step_start", batch, batch_idx)
            loss_dict = self.training_step(batch)
            loss = (
                self.config.loss.diffusion_loss * loss_dict["diffusion_loss"]
                + self.config.loss.contact_map_loss * loss_dict["contact_map_loss"]
            )

            sampler.stats.update(
                batch.scheme.edge_index,
                torch.tensor(loss, device=batch.device),
            )
            is_accumulating = (batch_idx + 1) % self.gradient_accumulation_steps != 0
            if not is_accumulating:
                self._optimizer_step()
            self.call_callbacks("on_train_step_end", batch, batch_idx, loss_dict)
            yield loss_dict
        self.optimizer.zero_grad()

        self._epoch += 1
        self.call_callbacks("on_train_epoch_end")

    @torch.no_grad()
    def test_inference_quality(
        self,
        batch: Batch,
        timesteps: int = 100,
    ) -> dict[str, float]:
        """Test the inference quality of the model on a batch."""
        batch = batch.to(device=self.device)

        output = self.inference(batch, timesteps=timesteps)
        contact_map_logit = output.contact_map_logit
        contact_map_loss = cal_contact_map_weighted_bce_loss(
            contact_map_logit,
            batch.structure.atom_pos,
            batch.structure.atom_pos_mask,
            batch.scheme.atom_to_residue_idx_map,
        )

        max_lddt, min_rmsd = 0, float("inf")

        lddt = metrics.cal_atom_lddt(
            output.atom_pos_pred[0],
            batch.structure.atom_pos[0],
            batch.structure.atom_mask[0],
        )
        max_lddt = max(max_lddt, lddt)

        rmsd = metrics.cal_aligned_rmsd(
            output.atom_pos_pred[0],
            batch.structure.atom_pos[0],
            batch.structure.atom_mask[0],
        )
        category_lddt = metrics.category_lddt(
            batch,
            output.atom_pos_pred[0],
        )
        min_rmsd = min(min_rmsd, rmsd)
        print(f"<<<category_lddt[{batch.name}]: {category_lddt}>>>")

        contact_map_quality_dict = self.contact_map_quality(
            batch,
            contact_map_logit,
        )

        return {
            "best_rmsd": min_rmsd,
            "best_lddt": max_lddt,
            "vald_contact_map_loss": contact_map_loss.item(),
            **contact_map_quality_dict,
        }

    @torch.no_grad()
    def inference(
        self,
        batch: Batch,
        timesteps: int = 100,
    ) -> MiniWorldInferenceOutput:
        """Inference using the diffusion solver."""
        raw_model = getattr(self.model, "module", self.model)
        model_wrapper = MiniWorldModelWrapper(
            raw_model,
            use_self_condition=self.config.experiment.self_condition,
        )
        batch = batch.to(device=self.device)
        model_wrapper.prepare_condition(batch)
        shape = batch.structure.atom_pos.shape

        atom_pos_pred, inter_traj, model_traj = self.solver.sample(
            model_fn=model_wrapper,
            shape=shape,
            num_steps=timesteps,
            device=self.device,
            return_intermediate=True,
        )
        inter_traj = [du.to_numpy(x) for x in inter_traj]
        model_traj = [du.to_numpy(x) for x in model_traj]
        contact_map_logit = model_wrapper.condition["contact_map_logit"]
        return MiniWorldInferenceOutput(
            atom_pos_pred=atom_pos_pred,
            model_traj=np.stack(model_traj, axis=1),
            inter_traj=np.stack(inter_traj, axis=1),
            batch=batch,
            contact_map_logit=contact_map_logit,
        )

    def sample(self) -> None:
        """Sample from the diffusion model using the ODE Euler solver."""
