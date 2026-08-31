"""Phase3 client: EDM diffusion-loss-only training over a FROZEN mini-SWA trunk.

Mirrors :mod:`miniworld.models.default_client` (EDM ``EuclideanDiffuser`` +
``EDMScheduler`` + ``AF3Solver``) but:

  * computes the DIFFUSION loss ONLY (no distogram / smooth-lDDT aux),
  * freezes + loads the epoch-900 trunk via the ``param_policy`` mechanism
    (reused from :mod:`miniworld.models.miniworld.client`): with
    ``default: freeze_loaded`` every checkpoint param (the trunk, whose keys
    match this model exactly) is loaded and frozen, while the brand-new
    ``to_token_single_trunk`` + ``diffusion_module`` params are re-initialized
    and left trainable.

Build the optimizer over :func:`miniworld.training.trainable_parameters` after
:meth:`maybe_apply_param_policy` so the frozen trunk is excluded.
"""

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
from team_gm.diffusion import AF3Solver, EDMScheduler, EuclideanDiffuser
from torch.utils.data import DataLoader

from miniworld.configs import EDMDiffuserConfig
from miniworld.data.features.batch import Batch
from miniworld.models.phase3.model import (
    InferenceOutput,
    ModelWrapper,
    Phase3Model,
)
from miniworld.training import ParamPolicyConfig

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path


class Client(BaseClient):
    """Client for phase3: train ONLY the diffusion head with EDM diffusion loss."""

    class TrainConfig(BaseModel):
        """Configuration for training."""

        comment: str = "phase3-diffusion"
        name: str = "MiniWorld-phase3"
        run_dir: str = "runs/phase3"
        overfitting: bool = False
        overfitting_dir: str | None = None
        train_item: int = 25600
        valid_item: int = 2560
        num_batch: int = 1
        num_epoch: int = 1000
        optimizer: Literal["AdamW", "Adam"] = "Adam"
        max_lr: float = 1e-4
        min_lr: float = 1e-5
        weight_decay: float = 0.01
        warmup_steps: int = int(5e3)
        decay_steps: int = int(5e6)
        decay_factor: float = 0.95
        compile: bool = False
        # EXPERIMENTAL (default "" = OFF): when non-empty (e.g. "reduce-overhead")
        # AND ``compile`` is True, the FROZEN trunk conditioning path is compiled
        # with this inductor mode (cudagraph-trees) and its outputs are cloned
        # before the grad diffusion path, while the diffusion module uses the
        # normal ``compile(dynamic=False)``. Empty string keeps the current
        # whole-model ``compile(dynamic=False)`` behaviour unchanged. Needs GPU
        # validation before use.
        trunk_compile_mode: str = ""
        num_augment: int = 48
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

        bucket_msa_multiple: int | None = 128
        bucket_token_multiple: int | None = 128
        bucket_atom_multiple: int | None = 1024

        verbose: bool = False
        use_wandb: bool = False
        wandb_project: str = "MiniWorld"

        # Selective freeze / re-init / load-existing policy. For phase3 set
        # enabled=True with default=freeze_loaded to load+freeze the trunk and
        # train only the new diffusion params.
        param_policy: ParamPolicyConfig = ParamPolicyConfig()

    class LossConfig(BaseModel):
        """Configuration for loss weights — diffusion loss ONLY."""

        diffusion_loss: float = 4.0

    class Config(BaseModel):
        """Configuration for the phase3 client."""

        model: Phase3Model.Config
        diffuser: EDMDiffuserConfig
        train: Client.TrainConfig
        loss: Client.LossConfig

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.config = config
        self.set_seed(config.train.seed)
        self.register_model(Phase3Model(config.model))

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

    # -- param policy (freeze + load epoch-900 trunk) -----------------------
    @staticmethod
    def _summarize_keys(title: str, keys: list[str], limit: int = 20) -> list[str]:
        if not keys:
            return [f"{title}: 0"]
        lines = [f"{title}: {len(keys)}"]
        lines.extend(f"  - {key}" for key in keys[:limit])
        if len(keys) > limit:
            lines.append(f"  ... and {len(keys) - limit} more")
        return lines

    def _warn_non_strict_model_load(
        self,
        missing_keys: list[str],
        unexpected_keys: list[str],
        incompatible_keys: list[tuple[str, Any, Any]],
    ) -> None:
        lines = [
            "=" * 88,
            "NON-STRICT CHECKPOINT LOAD: model architecture/state mismatch detected.",
            (
                "New, removed, or resized layers were not restored exactly. "
                "Missing or incompatible parameters will keep their current "
                "initialization."
            ),
            *self._summarize_keys("Missing model keys", missing_keys),
            *self._summarize_keys("Unexpected checkpoint keys", unexpected_keys),
            f"Shape-mismatched checkpoint keys: {len(incompatible_keys)}",
        ]
        if incompatible_keys:
            limit = 20
            lines.extend(
                (
                    f"  - {key} "
                    f"(checkpoint={checkpoint_shape}, current={current_shape})"
                )
                for key, checkpoint_shape, current_shape in incompatible_keys[:limit]
            )
            if len(incompatible_keys) > limit:
                lines.append(f"  ... and {len(incompatible_keys) - limit} more")
        lines.append("=" * 88)
        self.logger.warning("%s", "\n".join(lines))

    def maybe_apply_param_policy(
        self,
        state_dict: dict[str, Any] | None,
    ) -> dict[str, list[str]] | None:
        """Apply ``self.config.train.param_policy`` to the model in-place.

        Returns ``None`` (and the caller falls back to the standard
        ``load_state_dict`` path) when the policy is disabled. Otherwise loads +
        freezes checkpoint params, re-inits the rest, and restores epoch/step.
        Build the optimizer over :func:`trainable_parameters` afterwards.
        """
        policy = self.config.train.param_policy
        if not policy.enabled:
            return None

        from miniworld.training import apply_param_policy, format_summary

        ckpt_model_sd = (
            state_dict.get("model_state_dict") if state_dict is not None else None
        )
        summary = apply_param_policy(
            self.model, ckpt_model_sd, policy, log=self.logger,
        )
        self.logger.info("Param policy applied:\n%s", format_summary(summary))

        if state_dict is not None:
            self._epoch = state_dict.get("epoch", 0)
            self._global_step = state_dict.get("global_step", 0)

        return summary

    def load_state_dict(
        self,
        state_dict: dict[str, Any],
        *,
        model_only: bool = False,
        strict: bool = True,
    ) -> None:
        """Load model (and optionally optimizer/scheduler) state.

        With ``strict=False`` only key- and shape-matching params load; the rest
        keep their current init and a strong warning is emitted.
        """
        model_state_dict = state_dict["model_state_dict"]
        if strict:
            self.model.load_state_dict(model_state_dict)
        else:
            current_model_state = self.model.state_dict()
            filtered_model_state: dict[str, Any] = {}
            unexpected_keys: list[str] = []
            incompatible_keys: list[tuple[str, Any, Any]] = []

            for key, value in model_state_dict.items():
                if key not in current_model_state:
                    unexpected_keys.append(key)
                    continue
                current_value = current_model_state[key]
                if (
                    isinstance(value, torch.Tensor)
                    and isinstance(current_value, torch.Tensor)
                    and value.shape != current_value.shape
                ):
                    incompatible_keys.append(
                        (key, tuple(value.shape), tuple(current_value.shape)),
                    )
                    continue
                filtered_model_state[key] = value

            missing_keys = [
                key for key in current_model_state if key not in model_state_dict
            ]

            self.model.load_state_dict(filtered_model_state, strict=False)
            if missing_keys or unexpected_keys or incompatible_keys:
                self._warn_non_strict_model_load(
                    missing_keys=missing_keys,
                    unexpected_keys=unexpected_keys,
                    incompatible_keys=incompatible_keys,
                )

        self._epoch = state_dict["epoch"]
        self._global_step = state_dict["global_step"]

        if not model_only:
            optimizer_state = state_dict.get("optimizer_state_dict")
            scheduler_state = state_dict.get("scheduler_state_dict")
            if optimizer_state is not None and self._optimizer is not None:
                self.optimizer.load_state_dict(optimizer_state)
            if scheduler_state is not None and self.scheduler is not None:
                self.scheduler.load_state_dict(scheduler_state)

        self.call_callbacks("on_load_state_dict", state_dict)
        self.logger.info(
            "Loaded checkpoint (epoch=%d, step=%d)",
            self.epoch,
            self.global_step,
        )

    @classmethod
    def from_checkpoint(
        cls,
        filepath: str | Path,
        *,
        strict: bool = True,
        **extra_kwargs: Any,
    ) -> Client:
        """Restore model weights from a checkpoint file."""
        state_dict = torch.load(filepath, map_location="cpu")
        config = cls.Config.model_validate(state_dict["config"])
        client = cls(config, **extra_kwargs)
        client.load_state_dict(state_dict, model_only=True, strict=strict)
        return client

    # -- EDM diffusion training (loss = diffusion ONLY) ---------------------
    def loss_fn(
        self,
        batch: Batch,
        x0: Float[torch.Tensor, "... L 3"],
        x_input: Float[torch.Tensor, "... L 3"],
        t_emb: Float[torch.Tensor, ...],
        sigma: Float[torch.Tensor, ...],
        x_mask: Bool[torch.Tensor, "... L"] | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Compute the EDM diffusion loss (no distogram / lddt aux)."""
        atom_pos_update = self.model.forward(
            msa=batch.msa,
            template=batch.template,
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

        loss = self.config.loss.diffusion_loss * structure_loss

        return loss, {
            "diffusion_loss": structure_loss.item(),
            "total_loss": loss.item(),
            "main_loss": loss.item(),
        }

    def training_step(self, batch: Batch) -> dict[str, float]:
        """Train the diffusion head on a batch."""
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
        """Measure inference quality (best-of-N) on a single-item batch."""
        if batch.shape[0] != 1:
            msg = "Batch size for validation must be 1."
            raise ValueError(msg)
        batch = batch.duplicate(self.config.train.eval_sample_num)
        output = self.inference(batch, timesteps=self.config.train.eval_timesteps)
        return self.test_inference_quality(batch, output)

    def training_epoch(self, dataloader: DataLoader) -> Generator[Any, None, None]:
        """Yield results from training step over the dataloader for one epoch."""
        if not isinstance(dataloader, _FabricDataLoader) and isinstance(
            dataloader.sampler,
            _SetEpochProtocol,
        ):
            dataloader.sampler.set_epoch(self.epoch)
            self.model.set_seed(self.config.train.seed + self.epoch)  # pyright: ignore[reportCallIssue]

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
        """Best RMSD / lDDT over the sampled structures."""
        from miniworld.loss import metrics

        batch = batch.to(device=self.device)
        max_lddt, min_rmsd = 0.0, float("inf")
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
        return {"best_rmsd": min_rmsd, "best_lddt": max_lddt}

    @torch.no_grad()
    def inference(
        self,
        batch: Batch,
        timesteps: int = 100,
    ) -> InferenceOutput:
        """Inference using the EDM diffusion solver."""
        raw_model = getattr(self.model, "module", self.model)
        raw_model = cast("Phase3Model", raw_model)
        model_wrapper = ModelWrapper(raw_model)
        batch = batch.to(device=self.device)
        model_wrapper.prepare_condition(
            msa=batch.msa,
            template=batch.template,
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
        return InferenceOutput(
            atom_pos_pred=atom_pos_pred,
            model_traj=np.stack(model_traj, axis=1),
            inter_traj=np.stack(inter_traj, axis=1),
        )
