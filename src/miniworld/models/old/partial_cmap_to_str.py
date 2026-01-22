import random
from collections.abc import Generator, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
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
from miniworld.data.features.batch_edge_backprop import Batch, NoisyBatch
from miniworld.loss import metrics  # , losses
from miniworld.modules.configs import (
    CommonConfig,
    DiffusionConfig,
)
from miniworld.modules.diffusion_module import (
    DiffusionModule,
)
from miniworld.modules.feature_embedder import InputFeatureEmbedder
from miniworld.modules.primitives import (
    LayerNorm,
    Linear,
)
from miniworld.utils.diffusion.diffuser import EuclideanDiffuser
from miniworld.utils.diffusion.scheduler import EDMScheduler
from miniworld.utils.diffusion.solver import AF3Solver
from miniworld.utils.precision_manager import PrecisionConfig, precision_manager
from miniworld.utils.structure.distance import (
    get_contact_map,
)


class ContactMapEmbedder(nn.Module):
    """ContactMap Embedder."""

    def __init__(self, config: Pairformer.Config) -> None:
        super().__init__()

        self.to_pair = Linear(
            3,
            128,
            bias=False,
        )  # 0 : non-contact, 1: contact, 2: unknown(masked)
        # use_single is False
        config.use_single = False
        self.pairformer = Pairformer(config)

    def forward(
        self,
        contact_map: torch.Tensor,  # (B, L, L, 3)
        token_mask: torch.Tensor,  # (B, L), bool
    ) -> torch.Tensor:
        """Forward pass of ContactMapEmbedder."""
        token_pair = self.to_pair(contact_map)  # (B, L, L, d_pair)
        token_pair, _ = self.pairformer.forward(
            token_pair,
            single=None,
            mask=token_mask,
        )  # (B, L, L, d_pair), None
        return token_pair  # (B, L, L, d_pair)


class ContactMap2StrModel(nn.Module):
    """Contact Map to Structure Model."""

    class ConditionConfig(BaseModel):
        """Configuration for the trunk condition module."""

        pairformer: Pairformer.Config
        n_recycle_max: int = 4

    class Config(BaseModel):
        """Configuration for ContactMap2StrModel."""

        contact_map_dropout: float = 0.5
        common: CommonConfig
        contact_map_pairformer: "Pairformer.Config"
        trunk: "ContactMap2StrModel.ConditionConfig"
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

        # embed contact map
        self.contact_map_embedder = ContactMapEmbedder(config.contact_map_pairformer)
        self.contact_map_to_pair = nn.Sequential(
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
        self.pairformer_blocks = Pairformer(config.trunk.pairformer)

        # Diffusion module
        self.diffusion_module = DiffusionModule(config.common, config.diffusion)

    def condition_forward(
        self,
        noisy_batch: NoisyBatch,
    ) -> tuple[torch.Tensor, ...]:
        """Forward pass of the trunk condition module."""
        if self.training:
            n_recycle = random.randint(1, self.n_recycle_max)
        else:
            n_recycle = self.n_recycle_max
        if noisy_batch.msa.aligned_sequences.shape[1] != self.n_recycle_max:
            msg = (
                "The number of MSA sequences should match the number of recycle steps."
            )
            raise ValueError(msg)

        # gen randomly dropped out contact map
        with torch.no_grad():
            contact_map, contact_map_mask = get_contact_map(
                noisy_batch.structure.atom_pos,
                noisy_batch.structure.atom_pos_mask,
                noisy_batch.scheme.atom_to_residue_idx_map,
            )
            # randomly drop out contact map
            residue_length = contact_map.shape[1]
            dropout_length = int(residue_length * self.config.contact_map_dropout)
            start = random.randint(0, residue_length - dropout_length)
            dropout_mask = torch.ones_like(contact_map_mask)
            dropout_mask[:, start : start + dropout_length] = 0.0  # dropped
            dropout_mask[:, :, start : start + dropout_length] = 0.0  # dropped
            dropout_mask[
                :,
                start + dropout_length :,
                :start,
            ] = 0.0  # dropped
            dropout_mask[:, :start, start + dropout_length :] = 0.0  # dropped
            contact_map_mask = contact_map_mask * dropout_mask
            contact_map[~contact_map_mask.bool()] = 2.0  # unknown
            contact_map = contact_map.long()
            contact_map = nn.functional.one_hot(contact_map, num_classes=3).float()
        # input feature embedding
        (
            token_single_input,
            token_single_init,
            token_pair_init,
        ) = self.input_feature_embedder(noisy_batch)

        # embed contact map to pair token
        token_pair_contact_map = self.contact_map_embedder(
            contact_map,
            noisy_batch.structure.residue_mask,
        )
        token_pair_contact_map = self.contact_map_to_pair(token_pair_contact_map)
        token_pair_init = token_pair_init + token_pair_contact_map

        token_pair = torch.zeros_like(token_pair_init)
        token_single = torch.zeros_like(token_single_init)
        # Trunk forward with recycling
        for i_cycle in range(n_recycle):
            with ExitStack() as stack:
                if i_cycle < n_recycle - 1:
                    stack.enter_context(torch.no_grad())
                    stack.enter_context(torch.inference_mode())
                token_pair = token_pair_init + self.add_pair_recycle(token_pair)
                token_single = token_single_init + self.add_single_recycle(token_single)

                token_pair, token_single = self.pairformer_blocks.forward(
                    token_pair,
                    token_single,
                    noisy_batch.structure.residue_mask,
                )

        return (
            token_single_input,
            token_single,
            token_pair,
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

    def forward(self, noisy_batch: NoisyBatch) -> torch.Tensor:
        """Forward pass of ContactMap2StrModel."""
        (
            token_single_input,
            token_single_trunk,
            token_pair_trunk,
        ) = self.condition_forward(noisy_batch)
        # Diffusion forward
        return self.diffusion_forward(
            noisy_batch,
            token_single_input,
            token_single_trunk,
            token_pair_trunk,
        )


class ContactMap2StrModelWrapper(nn.Module):
    """Wrapper for ContactMap2StrModel to handle the input and output using solver."""

    def __init__(
        self,
        model: ContactMap2StrModel,
        use_self_condition: bool = True,
    ) -> None:
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
        if self.use_self_condition:
            msg = "Batch is already loaded. Cannot prepare condition again."
            raise ValueError(msg)
        if self.conditioned_forwarded:
            msg = "Conditioned forward is already done. Cannot prepare condition again."
            raise ValueError(msg)

        # Load the batch and prepare the model for conditioned forward pass
        self.load_batch(batch)
        (
            token_single_input,
            token_single_trunk,
            token_pair_trunk,
        ) = self.model.condition_forward(self.batch)
        self.conditioned_forwarded = True

        self.condition = {
            "token_single_input": token_single_input,
            "token_single_trunk": token_single_trunk,
            "token_pair_trunk": token_pair_trunk,
        }

    def forward(self, z_i: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model wrapper."""
        if not self.batch_loaded:
            msg = "Batch must be loaded before forward pass."
            raise ValueError(msg)
        if not self.conditioned_forwarded:
            msg = "Conditioned forward must be called before forward pass."
            raise ValueError(msg)

        n_str = z_i.shape[0]
        t_emb = t_emb[None, None, None, None].repeat(n_str, 1, 1, 1)
        x_mask = self.batch.structure.atom_pos_mask.repeat(n_str, 1)

        noisy_batch = NoisyBatch(
            **self.batch.__dict__,
            x_t=z_i,
            t=t_emb,
            x_sc=self.z_sc,
            x_mask=x_mask,
        )

        z_update = self.model.diffusion_forward(
            noisy_batch,
            self.condition["token_single_input"],
            self.condition["token_single_trunk"],
            self.condition["token_pair_trunk"],
        )
        if self.use_self_condition:
            self.z_sc = z_update

        return z_update


@dataclass
class ContactMap2StrInferenceOutput:
    """Output of ContactMap2Str inference."""

    # Tensor of final predicted atom coordinate.
    atom_pos_pred: torch.Tensor  # (B, L, 3)

    # Array of predicted atom coordinate trajectory for timesteps T.
    model_traj: np.ndarray | None  # (B, T, L, 3)

    # Array of interpolant predicted atom coordinate trajectory for timesteps T.
    inter_traj: np.ndarray | None  # (B, T, L, 3)

    batch: Batch


class ContactMap2StrClient(BaseClient):
    """Client for training and inference of ContactMap2StrModel."""

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
        distogram_loss: float = 0.03

    class ExperimentsConfig(BaseModel):
        """Configuration for experiments."""

        comment: str = "default"
        name: str = "AF3-PSK-2"
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
        ema_decay: float = 0.9999

    class DiffuserConfig(BaseModel):
        """Configuration for the diffuser."""

        seed: int = 0
        scheduler: EDMScheduler.EDMSchedulerConfig
        method: Literal["AF3", "EDM"] = "AF3"

    class Config(BaseModel):
        """Configuration for the AF3 client."""

        data: "ContactMap2StrClient.DataConfig"
        model: ContactMap2StrModel.Config
        experiment: "ContactMap2StrClient.ExperimentsConfig"
        diffuser: "ContactMap2StrClient.DiffuserConfig"
        loss: "ContactMap2StrClient.LossConfig"

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.config = config
        self.set_seed(config.experiment.seed)
        self.register_model(ContactMap2StrModel(config.model))

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

    def loss_fn(
        self,
        noisy_batch: NoisyBatch,
    ) -> tuple[torch.Tensor, Mapping]:
        """Compute the loss for a batch."""
        atom_pos_update = self.model.forward(noisy_batch)

        structure_loss = self.diffuser.cal_loss(atom_pos_update)

        loss = 4.0 * structure_loss  # + aux_losses

        return loss, {"EDMLoss": loss.item()}

    def training_step(self, batch: Batch) -> dict[str, float]:
        """Train the model on a batch."""
        with precision_manager(self.model, self.config.model.precision):
            num_augment = self.config.experiment.num_augment
            noisy_atom_pos, x_mask, t_emb = self.diffuser.sample(
                batch.structure.atom_pos,
                num_augment=num_augment,
                mask=batch.structure.atom_pos_mask,
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
        """Validate the model on a batch."""
        # Note that when doing validation, we measure inference quality, not a loss.
        # Please keep in mind that batch is duplicated to eval_sample_num, sample quality
        # is measured by the best sample in the batch. Therefore the batch size should be
        # give as 1.
        if batch.shape[0] != 1:
            msg = "Batch size for validation must be 1."
            raise ValueError(msg)
        return self.test_inference_quality(
            batch,
            self.config.experiment.eval_timesteps,
        )

    def training_epoch(self, dataloader: DataLoader) -> Generator[Any, None, None]:
        """Yield results from training step over the dataloader for one epoch."""
        sampler: AdaptiveEdgeSampler = dataloader.sampler

        self.model.train()
        self.call_callbacks("on_train_epoch_start")

        for batch_idx, batch_cpu in enumerate(iter(dataloader)):
            batch = batch_cpu.to(device=self.device)
            self.call_callbacks("on_train_step_start", batch, batch_idx)
            loss_dict = self.training_step(batch)
            loss = loss_dict["EDMLoss"]

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

        max_lddt, min_rmsd = 0, float("inf")

        lddt = metrics.cal_atom_lddt(
            output.atom_pos_pred[0],
            batch.structure.atom_pos[0],
            batch.structure.atom_mask[0],
        )
        max_lddt = max(max_lddt, lddt)

        rmsd = metrics.cal_aligned_rmsd(
            output.atom_pos_pred[0, 0],
            batch.structure.atom_pos[0],
            batch.structure.atom_mask[0],
        )
        min_rmsd = min(min_rmsd, rmsd)

        category_lddt = metrics.category_lddt(
            batch,
            output.atom_pos_pred[0, 0],
        )

        # test
        output = {"best_rmsd": min_rmsd, "best_lddt": max_lddt}
        # for key in category_lddt:
        #     output[key] = category_lddt[key]
        return output

    @torch.no_grad()
    def inference(
        self,
        batch: Batch,
        timesteps: int = 100,
    ) -> ContactMap2StrInferenceOutput:
        """Inference using the diffusion solver."""
        raw_model = getattr(self.model, "module", self.model)
        model_wrapper = ContactMap2StrModelWrapper(
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
        return ContactMap2StrInferenceOutput(
            atom_pos_pred=atom_pos_pred,
            model_traj=np.stack(model_traj, axis=1),
            inter_traj=np.stack(inter_traj, axis=1),
            batch=batch,
        )

    def sample(self) -> None:
        """Sample from the diffusion model using the ODE Euler solver."""
