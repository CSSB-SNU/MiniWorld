from __future__ import annotations

import random
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

import numpy as np
import torch
from lightning.fabric.wrappers import _FabricDataLoader
from pydantic import BaseModel, Discriminator, Tag
from team_gm import BaseClient
from team_gm.core.callbacks import ModelEMA
from team_gm.core.client import _SetEpochProtocol
from torch.utils.data import DataLoader

from miniworld.data.features.batch import Batch
from miniworld.loss.auxiliary import (
    cal_atom_distogram_loss,
)
from miniworld.models.distogram_only.model import Model
from miniworld.models.distogram_only.model_mini_swa import MiniSWAModel


def _model_variant_discriminator(value: object) -> str:
    """Route Client.Config.model to Model vs MiniSWAModel by presence of atom_swa.

    Duck-typed membership check works for plain ``dict`` and OmegaConf's
    ``DictConfig`` (which pydantic hands us when the config comes from hydra).
    """
    if isinstance(value, MiniSWAModel.Config):
        return "mini_swa"
    if isinstance(value, Model.Config):
        return "model"
    try:
        has_atom_swa = "atom_swa" in value  # type: ignore[operator]
    except (TypeError, AttributeError):
        has_atom_swa = False
    return "mini_swa" if has_atom_swa else "model"

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path


class Client(BaseClient):
    """Client for training the distogram-only trunk model."""

    class TrainConfig(BaseModel):
        """Configuration for trains."""

        comment: str = "default"
        name: str = "MiniWorld-Distogram"
        run_dir: str = "runs/distogram_only"
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
        # Trainer dispatch (was MW_FORCE_CUDAGRAPH / MW_FORCE_FABRIC): "auto" picks
        # CUDA-graph for fixed recycle (n_recycle_max==1) else Fabric; force either.
        force_trainer: Literal["auto", "cudagraph", "fabric"] = "auto"
        # torch.compile mode for the Fabric path (was MW_COMPILE_MODE); None = default.
        compile_mode: str | None = None
        # DDP find_unused_parameters (was MW_DDP_FIND_UNUSED); recycle skips some params
        # per step so default True keeps the collective count identical across ranks.
        ddp_find_unused: bool = True
        save_freq: int = 5
        eval_freq: int = 10
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

    class LossConfig(BaseModel):
        """Configuration for loss weights."""

        distogram_loss: float = 1.0
        # CB/pseudo-beta distogram target (rep-atom mask); False keeps the legacy
        # shortest-inter-atom-distance target. Was MW_DISTOGRAM_CB.
        distogram_cb_target: bool = False

    class Config(BaseModel):
        """Configuration for the distogram-only client.

        ``model`` is a discriminated union: MiniSWAModel.Config is picked when
        the payload has an ``atom_swa`` key (ESMFold2-style SWA/3D-RoPE atom
        embedder variant); otherwise falls back to the plain Model.Config.
        """

        model: Annotated[
            Annotated[Model.Config, Tag("model")] | Annotated[MiniSWAModel.Config, Tag("mini_swa")],
            Discriminator(_model_variant_discriminator),
        ]
        train: Client.TrainConfig
        loss: Client.LossConfig

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.config = config
        self.set_seed(config.train.seed)
        if isinstance(config.model, MiniSWAModel.Config):
            self.register_model(MiniSWAModel(config.model))
        else:
            self.register_model(Model(config.model))

        if config.train.use_ema:
            self.add_callback(ModelEMA(config.train.ema_decay))

    def set_seed(self, seed: int) -> None:
        """Set the random seed for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)

    @staticmethod
    def _summarize_keys(
        title: str,
        keys: list[str],
        limit: int = 20,
    ) -> list[str]:
        """Format checkpoint key summaries for warning logs."""
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
        """Emit a strong warning when model checkpoint loading is non-strict."""
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

    def load_state_dict(  # noqa: C901, PLR0912, PLR0915
        self,
        state_dict: dict[str, Any],
        *,
        model_only: bool = False,
        strict: bool = True,
    ) -> None:
        """Load model and optimizer state from a checkpoint file.

        With ``strict=False``, model parameters are loaded only when both key and
        shape match. Missing, unexpected, or resized parameters are skipped with a
        strong warning, so newly added layers stay at their current initialization.
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

            if strict:
                if optimizer_state is None:
                    msg = "Optimizer state not found in checkpoint."
                    raise ValueError(msg)
                if self._optimizer is None:
                    msg = "Optimizer is not set in the client."
                    raise ValueError(msg)
                if scheduler_state is not None and self.scheduler is None:
                    msg = (
                        "Scheduler state found in checkpoint, but no scheduler is set "
                        "in the client."
                    )
                    raise ValueError(msg)
                if scheduler_state is None and self.scheduler is not None:
                    msg = "Scheduler state not found in checkpoint."
                    raise ValueError(msg)

            if optimizer_state is not None and self._optimizer is not None:
                if strict:
                    self.optimizer.load_state_dict(optimizer_state)
                else:
                    try:
                        self.optimizer.load_state_dict(optimizer_state)
                    except ValueError as exc:
                        self.logger.warning(
                            "%s",
                            "\n".join(
                                [
                                    "=" * 88,
                                    "NON-STRICT CHECKPOINT LOAD: optimizer state mismatch.",
                                    "Optimizer state was skipped and current optimizer state is kept.",
                                    f"  - {type(exc).__name__}: {exc}",
                                    "=" * 88,
                                ],
                            ),
                        )

            if scheduler_state is not None and self.scheduler is not None:
                if strict:
                    self.scheduler.load_state_dict(scheduler_state)
                else:
                    try:
                        self.scheduler.load_state_dict(scheduler_state)
                    except (KeyError, ValueError) as exc:
                        self.logger.warning(
                            "%s",
                            "\n".join(
                                [
                                    "=" * 88,
                                    "NON-STRICT CHECKPOINT LOAD: scheduler state mismatch.",
                                    "Scheduler state was skipped and current scheduler state is kept.",
                                    f"  - {type(exc).__name__}: {exc}",
                                    "=" * 88,
                                ],
                            ),
                        )

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

    def loss_fn(self, batch: Batch) -> tuple[torch.Tensor, dict]:
        """Compute the distogram loss for a batch."""
        distogram_logit = self.model.forward(
            msa=batch.msa,
            reference=batch.reference,
            scheme=batch.scheme,
            sequence=batch.sequence,
            structure=batch.structure,
            template=batch.template,
        )

        # CB/pseudo-beta distogram target (config.loss.distogram_cb_target); default off
        # keeps the legacy shortest-inter-atom-distance target.
        _use_cb = self.config.loss.distogram_cb_target
        distogram_loss = cal_atom_distogram_loss(
            distogram_logit,
            batch.structure.atom_pos,
            batch.structure.atom_pos_mask,
            batch.scheme.atom_to_token_idx_map,
            rep_atom_mask=batch.structure.atom_is_rep if _use_cb else None,
        )

        loss = self.config.loss.distogram_loss * distogram_loss

        return loss, {
            "distogram_loss": distogram_loss.item(),
            "total_loss": loss.item(),
            "main_loss": loss.item(),
        }

    def training_step(self, batch: Batch) -> dict[str, float]:
        """Train the model on a batch."""
        loss, loss_dict = self.loss_fn(batch)
        self.backward(loss)
        del loss
        return loss_dict

    def validation_step(self, batch: Batch) -> dict[str, float]:
        """Inference path is not implemented for the distogram-only branch."""
        msg = (
            "validation_step is not implemented for the distogram-only client. "
            "Inference will be added in a follow-up."
        )
        raise NotImplementedError(msg)

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
