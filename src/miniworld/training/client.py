from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

import torch
import torch.nn as nn
from lightning.fabric.wrappers import _FabricDataLoader
from typing_extensions import Self

from .callbacks import Callback

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    from lightning import Fabric
    from lightning.fabric.wrappers import _FabricOptimizer
    from pydantic import BaseModel
    from torch.utils.data import DataLoader


@runtime_checkable
class _SetEpochProtocol(Protocol):
    def set_epoch(self, epoch: int) -> None: ...


class BaseClient(ABC):
    """Fabric-based training client."""

    Config: ClassVar[type[BaseModel]]

    @abstractmethod
    def training_step(self, batch: Any) -> Any:
        """Implement a single training step."""

    @abstractmethod
    def validation_step(self, batch: Any) -> Any:
        """Implement a single validation step."""

    @property
    def model(self) -> nn.Module:
        """Registered model instance."""
        if self._model is None:
            msg = (
                "Model has not been registered yet. "
                "Please call client.register_model() to register a model."
            )
            raise RuntimeError(msg)
        return self._model

    @property
    def fabric(self) -> Fabric:
        """Lightning fabric instance."""
        if self._fabric is None:
            msg = "Fabric has not been set up yet. Please call client.setup() first."
            raise ValueError(msg)
        return self._fabric

    @property
    def optimizer(self) -> _FabricOptimizer:
        """Optimizer used for training."""
        if self._optimizer is None:
            msg = "Optimizer not set. Please call setup() with an optimizer."
            raise ValueError(msg)
        if isinstance(self._optimizer, tuple):
            msg = "Multiple optimizers are not supported yet."
            raise NotImplementedError(msg)
        return self._optimizer

    @property
    def scheduler(self) -> torch.optim.lr_scheduler.LRScheduler | None:
        """Scheduler used for training."""
        return self._scheduler

    @property
    def epoch(self) -> int:
        """Current training epoch."""
        return self._epoch

    @property
    def global_step(self) -> int:
        """Current global training step."""
        return self._global_step

    @property
    def device(self) -> torch.device:
        """Current device of the model."""
        if self._fabric is not None:
            return self.fabric.device
        return next(self.model.parameters()).device

    @property
    def lr(self) -> float:
        """Current learning rate."""
        return self.optimizer.param_groups[0]["lr"]

    @property
    def is_global_zero(self) -> bool:
        """Returns True if the current process is global rank zero."""
        if self._fabric is None:
            return True
        return self.fabric.is_global_zero

    @property
    def logger(self) -> logging.Logger:
        """Client logger."""
        return logging.getLogger(self.__class__.__name__)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "Config"):
            msg = "Subclasses must define a Config class variable."
            raise NotImplementedError(msg)

    def __init__(self, config: BaseModel) -> None:
        self.Config.model_validate(config)

        self._model: nn.Module | None = None
        self._config = config
        self._fabric: Fabric | None = None
        self._optimizer = None
        self._scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self.gradient_accumulation_steps: int = 1
        self.gradient_clip_norm: float | None = None

        self._epoch: int = 0
        self._global_step: int = 0
        self._callbacks: list[Callback] = []

        self._setup_logger()

    def _setup_logger(self) -> None:
        if self.logger.hasHandlers():
            return

        rank = int(os.getenv("LOCAL_RANK", "0"))
        if rank == 0:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.addHandler(logging.NullHandler())

        self.logger.propagate = False

    def register_model(self, model: nn.Module) -> None:
        """Register the model to the client.

        Model can be accessed via the `model` property.
        """
        if not hasattr(self, "_model"):
            msg = (
                f"{self.__class__.__name__}.__init__() must call "
                "super().__init__(config) first"
            )
            raise RuntimeError(msg)
        self._model = model

    def add_callback(self, callback: Callback) -> None:
        """Add a callback."""
        if not isinstance(callback, Callback):
            msg = f"Expected Callback, got {type(callback).__name__}"
            raise TypeError(msg)
        self._callbacks.append(callback)

    def call_callbacks(self, hook_name: str, *args: Any, **kwargs: Any) -> None:
        """Call hook on all callbacks."""
        for callback in self._callbacks:
            if hasattr(callback, hook_name):
                try:
                    getattr(callback, hook_name)(self, *args, **kwargs)
                except Exception as e:
                    msg = f"Error in {callback.__class__.__name__}.{hook_name}()"
                    raise RuntimeError(msg) from e

    def setup(
        self,
        fabric: Fabric,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        gradient_accumulation_steps: int = 1,
        gradient_clip_norm: float | None = None,
    ) -> None:
        """Set up the client with Fabric, optimizer, scheduler, and training settings.

        Parameters
        ----------
        fabric : Fabric
            Lightning Fabric instance for distributed training.
        optimizer : torch.optim.Optimizer, optional
            PyTorch optimizer to use.
        scheduler : torch.optim.lr_scheduler.LRScheduler, optional
            PyTorch learning rate scheduler to use.
        gradient_accumulation_steps : int
            Steps for gradient accumulation.
        gradient_clip_norm : float, optional
            Gradient clipping norm value.

        """
        self._fabric = fabric
        self._model = fabric.setup_module(self.model)
        if optimizer:
            self._optimizer = fabric.setup_optimizers(optimizer)
        if scheduler:
            self._scheduler = scheduler
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_clip_norm = gradient_clip_norm

    def backward(self, loss: torch.Tensor) -> None:
        """Backward pass for the given loss.

        Loss will be scaled according to gradient accumulation steps. Any gradients
        accumulated from fewer batches than the accumulation step at the end of an epoch
        may not be applied.
        """
        loss = loss / self.gradient_accumulation_steps
        self.fabric.backward(loss)

    def training_epoch(self, dataloader: DataLoader) -> Generator[Any, None, None]:
        """Yield results from training step over the dataloader for one epoch.

        Accepts both raw ``DataLoader`` and ``_FabricDataLoader``.

        * To pass a ``_FabricDataLoader``, call ``fabric.setup_dataloaders()``
          before invoking this method.  Fabric's ``__iter__`` handles
          ``set_epoch`` internally.
        * A raw ``DataLoader`` is iterated as-is.  If its sampler implements
          ``set_epoch``, it will be called automatically.  Note that device
          transfer and distributed sampler wrapping are **not** applied for
          raw DataLoaders.
        """
        if not isinstance(dataloader, _FabricDataLoader) and isinstance(
            dataloader.sampler, _SetEpochProtocol
        ):
            dataloader.sampler.set_epoch(self.epoch)

        self.model.train()
        self.call_callbacks("on_train_epoch_start")

        try:
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx % self.gradient_accumulation_steps == 0:
                    self.call_callbacks("on_train_step_start", batch, batch_idx)

                self.call_callbacks("on_train_batch_start", batch, batch_idx)
                accumed_steps = (batch_idx + 1) % self.gradient_accumulation_steps
                is_accumulating = accumed_steps != 0
                with self.fabric.no_backward_sync(self.model, enabled=is_accumulating):  # ty: ignore[invalid-argument-type]
                    result = self.training_step(batch)
                self.call_callbacks("on_train_batch_end", batch, batch_idx, result)

                if not is_accumulating:
                    self._optimizer_step()
                    self.call_callbacks("on_train_step_end", batch, batch_idx, result)

                yield result

        finally:
            self.optimizer.zero_grad()
            self._epoch += 1
            self.call_callbacks("on_train_epoch_end")

    @torch.inference_mode()
    def validation_epoch(self, dataloader: DataLoader) -> Generator[Any, None, None]:
        """Yield results from validation step over the dataloader for one epoch.

        Accepts both raw ``DataLoader`` and ``_FabricDataLoader``.

        * To pass a ``_FabricDataLoader``, call ``fabric.setup_dataloaders()``
          before invoking this method.
        * A raw ``DataLoader`` is iterated as-is.  Note that device transfer
          and distributed sampler wrapping are **not** applied for raw
          DataLoaders.
        """
        self.model.eval()
        self.call_callbacks("on_validation_epoch_start")

        try:
            for batch_idx, batch in enumerate(dataloader):
                self.call_callbacks("on_validation_step_start", batch, batch_idx)
                result = self.validation_step(batch)
                self.call_callbacks("on_validation_step_end", batch, batch_idx, result)
                yield result

        finally:
            self.call_callbacks("on_validation_epoch_end")

    def save_checkpoint(
        self,
        filepath: str | Path,
        *,
        model_only: bool = False,
    ) -> None:
        """Save model and optimizer state to a checkpoint file.

        Parameters
        ----------
        filepath : str | Path
            Path to save the checkpoint file.
        model_only : bool
            Only save the model weights, ignoring optimizer and scheduler states.

        """
        if self.is_global_zero:
            state_dict = {
                "config": self._config.model_dump(mode="json"),
                "model_state_dict": self.model.state_dict(),
                "epoch": self.epoch,
                "global_step": self.global_step,
            }
            if not model_only:
                if self._optimizer is None:
                    msg = (
                        "Optimizer is not set in the client. If you want to save only "
                        "the model weights, set model_only=True."
                    )
                    raise ValueError(msg)
                state_dict["optimizer_state_dict"] = self.optimizer.state_dict()
                if self.scheduler is not None:
                    state_dict["scheduler_state_dict"] = self.scheduler.state_dict()

            self.call_callbacks("on_save_checkpoint", state_dict)
            torch.save(state_dict, filepath)
            self.logger.info(
                "Saved checkpoint to %s (epoch=%d, step=%d)",
                filepath,
                self.epoch,
                self.global_step,
            )

    def load_state_dict(
        self,
        state_dict: dict[str, Any],
        *,
        model_only: bool = False,
        strict: bool = True,
    ) -> None:
        """Load model and optimizer state from a checkpoint file.

        Parameters
        ----------
        state_dict : dict[str, Any]
            Dictionary containing the checkpoint state.
        model_only : bool
            Only load the model weights, ignoring optimizer and scheduler states.
        strict : bool
            If True, raise error when optimizer/scheduler states are missing or
            mismatched.

        Raises
        ------
        ValueError
            If model_only=False and optimizer/scheduler states don't match.

        """
        self.model.load_state_dict(state_dict["model_state_dict"])
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
    def from_checkpoint(cls, filepath: str | Path, **extra_kwargs: Any) -> Self:
        """Restore model weights from a pretrained checkpoint file.

        Parameters
        ----------
        filepath : str | Path
            Path to the checkpoint file.
        **extra_kwargs : Any
            Extra keyword arguments to initialize the client instance.

        Notes
        -----
        This method only loads the model weights, optimizer and scheduler states are
        ignored. To load the full checkpoint, use `client.load_state_dict()` after
        setting optimizer and scheduler.

        """
        state_dict = torch.load(filepath, map_location="cpu")
        config = cls.Config.model_validate(state_dict["config"])
        client = cls(config, **extra_kwargs)
        client.load_state_dict(state_dict, model_only=True)
        return client

    def _optimizer_step(self) -> None:
        if self.gradient_clip_norm is not None:
            self.fabric.clip_gradients(
                self.model,
                self.optimizer,
                max_norm=self.gradient_clip_norm,
            )

        self.optimizer.step()
        self.optimizer.zero_grad()
        if self.scheduler is not None:
            self.scheduler.step()

        self._global_step += 1
