"""
Custom PyTorch training client. This is a base class for training clients.
It provides a simple interface for training and validation loops. It also
provides a method to save and load checkpoints.

This client is derived from PyTorch Lightning.
https://github.com/Lightning-AI/pytorch-lightning
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np

from dataclasses import dataclass
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing_extensions import Self
from typing import Any, Literal
from torch import Tensor
from pydantic import BaseModel
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from functools import cached_property

from .loggers import Logger, StreamLogger
from .callbacks import Callback, DefaultCallback
from team_gm.utils.data_utils import move_data_to_device
from team_gm.utils.dist_utils import rank_zero_only


class GradientHelper:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        clip_max_norm: float | None = None,
        accum_steps: int = 1,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.clip_max_norm = clip_max_norm
        self.accum_steps = accum_steps

        self._local_step = 0
        self._global_step = 0

    @property
    def global_step(self) -> int:
        return self._global_step

    @property
    def local_step(self) -> int:
        return self._local_step

    def to(self, device):
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

        if self.scheduler is not None:
            new_scheduler_state = {}
            for k, v in self.scheduler.state_dict().items():
                if torch.is_tensor(v):
                    new_scheduler_state[k] = v.to(device)
                else:
                    new_scheduler_state[k] = v
            self.scheduler.load_state_dict(new_scheduler_state)

    def backward_local_step(self, loss: torch.Tensor):
        self._local_step += 1
        if self.accum_steps == 1:
            loss.backward()
            self.backward_global_step()
            return

        loss = loss / self.accum_steps
        is_last_step = self._local_step % self.accum_steps == 0
        if isinstance(self.model, DDP) and not is_last_step:
            with self.model.no_sync():
                loss.backward()
        else:
            loss.backward()

        if is_last_step:
            self.backward_global_step()

    def backward_global_step(self):
        if self.clip_max_norm is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_max_norm)

        self.optimizer.step()
        self.optimizer.zero_grad()
        if self.scheduler is not None:
            self.scheduler.step()

        self._local_step = 0
        self._global_step += 1

    def finalize(self):
        if self._local_step == 0:
            return

        # In DDP with gradient accumulation, if epoch ends mid-accumulation,
        # the last backward was called with no_sync(), meaning gradients
        # are not synchronized across GPUs. We need to trigger sync before
        # updating weights. A dummy backward pass forces gradient synchronization.
        if isinstance(self.model, DDP):
            dummy_loss = torch.tensor(
                0.0, device=next(self.model.parameters()).device, requires_grad=True
            )
            dummy_loss.backward()  # Triggers gradient sync across GPUs

        self.backward_global_step()


class LogMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._loggers: list[Logger] = []
        self._buffer_step: dict[str, list[float]] = {}
        self._buffer_epoch: dict[str, list[float]] = {}
        self._in_training_epoch: bool = False
        self._in_validation_epoch: bool = False

    @rank_zero_only
    def add_logger(self: "BaseClient", logger: "Logger"):
        """Add a logger to the list of loggers."""
        if not isinstance(logger, Logger):
            raise TypeError(
                f"logger must be an instance of {Logger.__name__}, "
                f"but got {type(logger).__name__}."
            )
        if not hasattr(self, "config"):
            raise RuntimeError(
                "add_logger method must be called after the class is initialized."
            )
        logger.log_config(self.config)
        self._loggers.append(logger)

    def clear_loggers(self: "BaseClient"):
        """Clear all loggers."""
        self._loggers.clear()

    @rank_zero_only
    def log_message(
        self,
        message: Any,
        level: Literal["info", "debug", "warning", "error", "critical"] = "info",
    ):
        """
        Log a message immediately.

        Parameters
        ----------
        message : Any
            The message to log. It can be any type, but should be convertible to a string
            for logging purposes.
        level : Literal["info", "debug", "warning", "error", "critical"]
            The logging level for the message. Defaults to "info".
        """
        for logger in self._loggers:
            logger.log_message(message, level=level)

    def log_metrics(
        self: "BaseClient",
        metrics: Mapping[str, float | int | Tensor | np.ndarray],
        on_step: bool | None = None,
        on_epoch: bool | None = None,
        sync_dist: bool = True,
    ):
        """
        Log metrics with step-wise and epoch-wise support.

        Parameters
        ----------
        metrics : Mapping[str, float | int | Tensor | np.ndarray]
            Dictionary of metrics to log
        on_step : bool | None
            Log the metrics on global step.
        on_epoch : bool | None
            Log the average metrics on epoch end.
        sync_dist : bool
            If True, sync metrics across all processes. If you are using distributed
            training and want to log only on the master process, set this to False.

        Notes
        -----
        When both parameters are None, defaults to epoch-wise logging
        (on_step=False, on_epoch=True)
        """

        if on_step is None and on_epoch is None:
            on_step, on_epoch = False, True  # default to logging on epoch end
        elif on_step is None:
            on_step = not on_epoch
        elif on_epoch is None:
            on_epoch = not on_step

        assert on_step or on_epoch, (
            "At least one of `on_step` or `on_epoch` must be True."
        )
        metrics = self._convert_to_float(metrics)
        if self.is_distributed and sync_dist:
            metric_values = list(metrics.values())
            metric_values = torch.tensor(metric_values, device=self.device)
            dist.all_reduce(metric_values, op=dist.ReduceOp.SUM)
            metric_values /= self.world_size
            metrics = {k: v.item() for k, v in zip(metrics.keys(), metric_values)}

        if not self.is_master:
            return

        if not self._in_training_epoch and not self._in_validation_epoch:
            if on_step:
                for logger in self._loggers:
                    logger.log_step(metrics, step=self.step, epoch=self.epoch)
            if on_epoch:
                for logger in self._loggers:
                    logger.log_epoch(metrics, step=self.step, epoch=self.epoch)
            return

        for key, value in metrics.items():
            if on_step:
                if key not in self._buffer_step:
                    self._buffer_step[key] = []
                self._buffer_step[key].append(value)

            if on_epoch:
                if key not in self._buffer_epoch:
                    self._buffer_epoch[key] = []
                self._buffer_epoch[key].append(value)

    def _convert_to_float(
        self, metrics: Mapping[str, float | int | Tensor | np.ndarray]
    ) -> dict[str, float]:
        """Convert all values in the metrics to float."""

        def _to_scalar(k, v):
            if isinstance(v, Tensor):
                if v.numel() != 1:
                    raise ValueError(
                        f"Expected a scalar tensor, but got a tensor with {v.numel()} "
                        f"elements for key '{k}'."
                    )
                return v.item()
            elif isinstance(v, np.ndarray):
                if v.size != 1:
                    raise ValueError(
                        f"Expected a scalar numpy array, but got an array with {v.size} "
                        f"elements for key '{k}'."
                    )
                return v.item()
            elif isinstance(v, float | int):
                return float(v)
            else:
                raise TypeError(f"Unsupported type {type(v).__name__} for key '{k}'.")

        return {k: float(_to_scalar(k, v)) for k, v in metrics.items()}

    @rank_zero_only
    def _log_on_global_step_complete(self: "BaseClient"):
        """Callback when global step is completed."""
        if not self._buffer_step:
            return

        metrics = {
            key: float(np.mean(values)) for key, values in self._buffer_step.items()
        }
        for logger in self._loggers:
            logger.log_step(metrics, step=self.step, epoch=self.epoch)
        self._buffer_step.clear()

    @rank_zero_only
    def _log_on_epoch_complete(self: "BaseClient"):
        """Callback when epoch is completed."""

        metrics = {
            key: float(np.mean(values)) for key, values in self._buffer_epoch.items()
        }
        for logger in self._loggers:
            logger.log_epoch(metrics, step=self.step, epoch=self.epoch)
        self._buffer_epoch.clear()


class CallbackMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._callbacks: list[Callback] = []

    def add_callback(self, callback: Callback) -> None:
        if not isinstance(callback, Callback):
            raise TypeError(f"Expected Callback, got {type(callback).__name__}")
        self._callbacks.append(callback)

    def call_callbacks(self, hook_name: str, *args, **kwargs) -> None:
        for callback in self._callbacks:
            if hasattr(callback, hook_name):
                try:
                    getattr(callback, hook_name)(self, *args, **kwargs)
                except Exception as e:
                    raise RuntimeError(
                        f"Error in callback '{callback.__class__.__name__}' "
                        f"while calling '{hook_name}': {e}"
                    ) from e


@dataclass(kw_only=True, frozen=True)
class Checkpoint:
    epoch: int
    step: int
    name: str
    config: dict
    model: dict
    optimizer: dict | None = None
    scheduler: dict | None = None


class BaseClient(LogMixin, CallbackMixin, ABC, nn.Module):
    @abstractmethod
    class Config(BaseModel):
        pass

    @abstractmethod
    def training_step(self, **kwargs) -> Any:
        pass

    @abstractmethod
    def validation_step(self, **kwargs) -> Any:
        pass

    def __init__(self):
        super().__init__()
        self._setup_model_done = False
        self._setup_done = False
        if hasattr(self.__class__, "_checkpoint_name"):
            self._name = self.__class__._checkpoint_name
        else:
            self._name: str = "default"
        self._epoch: int = 0
        self._losses = []
        self._should_stop = False

    def setup_model(
        self,
        model: nn.Module,
        **ddp_kwargs,
    ) -> nn.Module:
        if self.is_distributed:
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(self.local_rank)
            model = model.to(self.local_rank)
            self.to(self.local_rank)
            model = DDP(model, device_ids=[self.local_rank], **ddp_kwargs)
            self._unwrapped_model = model.module
            self._wrapped_model = model
        else:
            self._unwrapped_model = model
            self._wrapped_model = model
        self._setup_model_done = True
        return model

    def setup(
        self,
        config: Config,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        clip_max_norm: float | None = None,
        accum_steps: int = 1,
        name: str | None = None,
    ):
        if not self._setup_model_done:
            raise RuntimeError(
                "You must call `setup_model` method before calling `setup`."
            )

        self.config = config
        self.gradient_helper = GradientHelper(
            self._wrapped_model,
            optimizer=optimizer,
            scheduler=scheduler,
            clip_max_norm=clip_max_norm,
            accum_steps=accum_steps,
        )
        if name is not None:
            self._name = name
        self.add_logger(StreamLogger())
        self.add_callback(DefaultCallback())
        self._setup_done = True

    def __init_subclass__(cls):
        ori_init = cls.__init__
        if (
            "config" not in ori_init.__annotations__
            or ori_init.__annotations__["config"] is not cls.Config
        ):
            config_type = cls.Config.__name__
            err_msg = (
                f"__init__ method of {cls.__name__} must have 'config' parameter of type"
                f" {config_type} (e.g. `def __init__(self, config: {config_type})`.)"
            )
            raise TypeError(err_msg)

        def new_init(self, *args, **kwargs):
            ori_init(self, *args, **kwargs)

            if not self._setup_done or not self._setup_model_done:
                raise RuntimeError(
                    "You must call `setup_model` and `setup` method on __init__ of the "
                    "client class."
                )

        cls.__init__ = new_init

    @property
    def name(self) -> str:
        return self._name

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def step(self) -> int:
        return self.gradient_helper.global_step

    @property
    def lr(self):
        return self.gradient_helper.optimizer.param_groups[0]["lr"]

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @cached_property
    def rank(self) -> int:
        return int(os.environ.get("RANK", 0))

    @cached_property
    def local_rank(self) -> int:
        return int(os.environ.get("LOCAL_RANK", 0))

    @cached_property
    def world_size(self) -> int:
        return int(os.environ.get("WORLD_SIZE", 1))

    @cached_property
    def is_master(self) -> bool:
        return self.rank == 0

    @cached_property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    def _setup_dataloader(self, dataloader: Iterable) -> Iterable:
        if not self.is_distributed:
            return dataloader
        if not isinstance(dataloader, torch.utils.data.DataLoader):
            raise TypeError(
                "Dataloader must be an instance of torch.utils.data.DataLoader "
                "when using distributed training."
            )
        if not isinstance(dataloader.sampler, torch.utils.data.DistributedSampler):
            raise TypeError(
                "Dataloader must use DistributedSampler for distributed training."
            )
        dataloader.sampler.set_epoch(self.epoch)
        return dataloader

    def stop(self) -> None:
        """Signal to stop training."""
        self._should_stop = True

    def cleanup(self) -> None:
        """Clean up resources after training."""
        if self.is_distributed:
            dist.destroy_process_group()

    def _sync_and_check_stop(self) -> None:
        """Check if epoch should stop and synchronize across processes."""
        if self.is_distributed:
            should_stop_tensor = torch.tensor(int(self._should_stop), device=self.device)
            dist.all_reduce(should_stop_tensor, op=dist.ReduceOp.MAX)
            if should_stop_tensor.item() > 0:
                self._should_stop = True

        if self._should_stop:
            self.cleanup()
            raise StopIteration("Epoch stopped by user request.")

    def backward(self, loss: torch.Tensor) -> None:
        assert self._in_training_epoch, (
            "Cannot call backward outside of training epoch. "
            "Use `training_epoch` method to run training loop."
        )
        self.gradient_helper.backward_local_step(loss)
        self._losses.append(loss.item())

    def training_epoch(self, dataloader: Iterable) -> list:
        dataloader = self._setup_dataloader(dataloader)
        self._epoch += 1
        self._in_training_epoch = True
        self.train()
        outputs = []

        self.call_callbacks("on_train_epoch_start")
        for batch_idx, batch in enumerate(dataloader):
            batch = move_data_to_device(batch, self.device)

            self.call_callbacks("on_train_batch_start", batch, batch_idx)
            output = self.training_step(batch)
            outputs.append(output)
            self.call_callbacks("on_train_batch_end", output, batch, batch_idx)
            if self.gradient_helper.local_step == 0:
                self._log_on_global_step_complete()
            self._sync_and_check_stop()

        self.gradient_helper.finalize()
        self._log_on_global_step_complete()
        self._log_on_epoch_complete()
        self._in_training_epoch = False

        if self._losses and self.is_distributed:
            local_mean_loss = torch.tensor(self._losses, device=self.device).mean()
            dist.all_reduce(local_mean_loss, op=dist.ReduceOp.SUM)
            mean_loss = local_mean_loss.cpu()
            mean_loss /= self.world_size
        else:
            mean_loss = torch.tensor(self._losses).mean()
        self._losses.clear()
        self.call_callbacks("on_train_epoch_end", mean_loss, outputs)
        self._sync_and_check_stop()
        return outputs

    @torch.no_grad()
    def validation_epoch(self, dataloader: Iterable) -> list:
        dataloader = self._setup_dataloader(dataloader)
        self._in_validation_epoch = True
        self.eval()
        outputs = []

        self.call_callbacks("on_validation_epoch_start")
        for batch_idx, batch in enumerate(dataloader):
            batch = move_data_to_device(batch, self.device)

            self.call_callbacks("on_validation_batch_start", batch, batch_idx)
            output = self.validation_step(batch)
            outputs.append(output)
            self.call_callbacks("on_validation_batch_end", output, batch, batch_idx)
            self._log_on_global_step_complete()
            self._sync_and_check_stop()

        self._log_on_epoch_complete()
        self._in_validation_epoch = False

        self.call_callbacks("on_validation_epoch_end", outputs)
        self._sync_and_check_stop()
        return outputs

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: str, *, model_only: bool = False, **kwargs
    ) -> Self:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        checkpoint = Checkpoint(**state_dict)
        cls._checkpoint_name = checkpoint.name
        kwargs.setdefault("config", cls.Config(**checkpoint.config))
        client = cls(**kwargs)
        delattr(cls, "_checkpoint_name")

        client._unwrapped_model.load_state_dict(checkpoint.model)
        if not model_only:
            if checkpoint.optimizer is not None:
                client.gradient_helper.optimizer.load_state_dict(checkpoint.optimizer)
            if checkpoint.scheduler is not None:
                client.gradient_helper.scheduler.load_state_dict(checkpoint.scheduler)
        client._epoch = checkpoint.epoch
        client.gradient_helper._global_step = checkpoint.step
        if client.is_distributed:
            dist.barrier()
        return client

    @rank_zero_only
    def save_checkpoint(self, checkpoint_path: str, model_only: bool = False):
        if model_only:
            optimizer_state = None
            scheduler_state = None
        else:
            optimizer_state = self.gradient_helper.optimizer.state_dict()
            if self.gradient_helper.scheduler is not None:
                scheduler_state = self.gradient_helper.scheduler.state_dict()
            else:
                scheduler_state = None

        checkpoint = Checkpoint(
            epoch=self.epoch,
            step=self.step,
            name=self.name,
            config=self.config.model_dump(),
            model=self._unwrapped_model.state_dict(),
            optimizer=optimizer_state,
            scheduler=scheduler_state,
        )
        torch.save(checkpoint.__dict__, checkpoint_path)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        if self._setup_done:
            self.gradient_helper.to(device=self.device)
        return self
