from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import torch
from jaxtyping import Bool, Float
from lightning.fabric.wrappers import _FabricDataLoader
from pydantic import BaseModel
from team_gm import BaseClient
from team_gm.core.callbacks import ModelEMA
from team_gm.core.client import _SetEpochProtocol
from torch.utils.data import DataLoader

from miniworld.configs import EDMDiffuserConfig
from miniworld.data.features.batch import Batch
from miniworld.diffusion import AF3Solver, EDMScheduler, EuclideanDiffuser
from miniworld.loss import metrics
from miniworld.loss.auxiliary import (
    cal_atom_distogram_loss,
)
from miniworld.models.af3_like.model import (
    InferenceOutput,
    Model,
    ModelWrapper,
)

if TYPE_CHECKING:
    from collections.abc import Generator


class Client(BaseClient):
    """Client for training and inference of AF3Like model."""

    class TrainConfig(BaseModel):
        """Configuration for trains."""

        comment: str = "default"
        name: str = "AF3Like-PSK-2"
        run_dir: str = "runs/af3like"
        overfitting: bool = False
        overfitting_dir: str | None = None  # Directory for overfitting mode
        train_item: int = 25600
        valid_item: int = 2560
        num_batch: int = 1
        num_epoch: int = 1000
        optimizer: Literal["AdamW", "Adam"] = "AdamW"
        max_lr: float = 1e-4
        min_lr: float = 1e-5
        weight_decay: float = 0.01
        warmup_steps: int = int(5e3)
        decay_steps: int = int(5e6)
        decay_factor: float = 0.95
        compile: bool = False
        num_augment: int = 8
        save_freq: int = 5
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

        bucket_msa_multiple: int | None = 128
        bucket_token_multiple: int | None = 128
        bucket_atom_multiple: int | None = 1024

        verbose: bool = False
        use_wandb: bool = False
        wandb_project: str = "MiniWorld"

    class LossConfig(BaseModel):
        """Configuration for loss weights."""

        diffusion_loss: float = 4.0
        distogram_loss: float = 0.03

    class Config(BaseModel):
        """Configuration for the AF3Like client."""

        model: Model.Config
        diffuser: EDMDiffuserConfig
        train: Client.TrainConfig
        loss: Client.LossConfig

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.config = config
        self.set_seed(config.train.seed)
        self.register_model(Model(config.model))

        if config.train.use_ema:
            self.add_callback(ModelEMA(config.train.ema_decay))
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
        batch: Batch,
        x0: Float[torch.Tensor, "... L 3"],
        x_input: Float[torch.Tensor, "... L 3"],
        t_emb: Float[torch.Tensor, ...],
        sigma: Float[torch.Tensor, ...],
        x_mask: Bool[torch.Tensor, "... L"] | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Compute the loss given a noisy batch."""
        atom_pos_update, distogram_logit = self.model.forward(
            msa=batch.msa,
            reference=batch.reference,
            scheme=batch.scheme,
            sequence=batch.sequence,
            structure=batch.structure,
            x_t=x_input,
            x_mask=x_mask,
            t_emb=t_emb,
        )

        structure_loss = self.diffuser.cal_loss(
            x0=x0,
            x_input=x_input,
            x_update=atom_pos_update,
            sigma=sigma,
            mask=x_mask,
        )

        distogram_loss = cal_atom_distogram_loss(
            distogram_logit,
            batch.structure.atom_pos,
            batch.structure.atom_pos_mask,
            batch.scheme.atom_to_token_idx_map,
        )

        loss = (
            self.config.loss.diffusion_loss * structure_loss
            + self.config.loss.distogram_loss * distogram_loss
        )

        return loss, {
            "diffusion_loss": structure_loss.item(),
            "distogram_loss": distogram_loss.item(),
            "total_loss": loss.item(),
            "main_loss": loss.item(),
        }

    def training_step(self, batch: Batch) -> dict[str, float]:
        """Train the model on a batch."""
        num_augment = self.config.train.num_augment
        x0, x_input, x_mask, t_emb, sigma = self.diffuser.sample(
            batch.structure.atom_pos,
            num_augment=num_augment,
            mask=batch.structure.atom_mask,
        )

        loss, loss_dict = self.loss_fn(
            batch=batch,
            x0=x0,
            x_input=x_input,
            t_emb=t_emb,
            sigma=sigma,
            x_mask=x_mask,
        )

        self.backward(loss)
        del loss
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
        batch = batch.duplicate(self.config.train.eval_sample_num)
        output = self.inference(batch, timesteps=self.config.train.eval_timesteps)

        return self.test_inference_quality(
            batch,
            output,
        )

    def training_epoch(self, dataloader: DataLoader) -> Generator[Any, None, None]:
        """Yield results from training step over the dataloader for one epoch."""
        if not isinstance(dataloader, _FabricDataLoader) and isinstance(
            dataloader.sampler,
            _SetEpochProtocol,
        ):
            dataloader.sampler.set_epoch(self.epoch)

        self.model.train()
        self.call_callbacks("on_train_epoch_start")

        try:
            for batch_idx, _batch in enumerate(dataloader):
                batch = cast("Batch", _batch)
                batch = batch.to(device=self.device)
                if batch_idx % self.gradient_accumulation_steps == 0:
                    self.call_callbacks("on_train_step_start", batch, batch_idx)
                self.call_callbacks("on_train_batch_start", batch, batch_idx)
                is_accumulating = (batch_idx + 1) % self.gradient_accumulation_steps != 0
                with self.fabric.no_backward_sync(
                    self.model,  # pyright: ignore[reportArgumentType]
                    enabled=is_accumulating,
                ):
                    loss_dict = self.training_step(batch)
                self.call_callbacks(
                    "on_train_batch_end",
                    batch,
                    batch_idx,
                    loss_dict,
                )
                if not is_accumulating:
                    self._optimizer_step()
                self.call_callbacks("on_train_step_end", batch, batch_idx, loss_dict)
                yield loss_dict
        finally:
            self.optimizer.zero_grad()

            self._epoch += 1
            self.call_callbacks("on_train_epoch_end")

    @torch.no_grad()
    def test_inference_quality(
        self,
        batch: Batch,
        output: InferenceOutput,
    ) -> dict[str, float]:
        """Test the inference quality of the model on a batch."""
        batch = batch.to(device=self.device)

        distogram_logit = output.distogram_logit
        distogram_loss = cal_atom_distogram_loss(
            distogram_logit,
            batch.structure.atom_pos,
            batch.structure.atom_pos_mask,
            batch.scheme.atom_to_token_idx_map,
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
        min_rmsd = min(min_rmsd, rmsd)

        return {
            "best_rmsd": min_rmsd,
            "best_lddt": max_lddt,
            "vald_distogram_loss": distogram_loss.item(),
        }

    @torch.no_grad()
    def inference(
        self,
        batch: Batch,
        timesteps: int = 100,
    ) -> InferenceOutput:
        """Inference using the diffusion solver."""
        raw_model = getattr(self.model, "module", self.model)
        raw_model = cast("Model", raw_model)
        model_wrapper = ModelWrapper(
            raw_model,
        )
        batch = batch.to(device=self.device)
        model_wrapper.prepare_condition(
            msa=batch.msa,
            reference=batch.reference,
            scheme=batch.scheme,
            sequence=batch.sequence,
            structure=batch.structure,
        )
        shape = batch.structure.atom_pos.shape
        atom_pos_pred, inter_traj, model_traj = self.solver.sample(
            model_fn=model_wrapper,
            shape=shape,
            num_steps=timesteps,
            device=self.device,
            return_intermediate=True,
        )
        inter_traj = [x.detach().cpu().numpy() for x in inter_traj]
        model_traj = [x.detach().cpu().numpy() for x in model_traj]
        distogram_logit = model_wrapper.condition["distogram_logit"]
        return InferenceOutput(
            atom_pos_pred=atom_pos_pred,
            model_traj=np.stack(model_traj, axis=1),
            inter_traj=np.stack(inter_traj, axis=1),
            distogram_logit=distogram_logit,
        )

    def sample(self) -> None:
        """Sample from the diffusion model using the ODE Euler solver."""
