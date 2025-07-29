import wandb
import logging
import json
import os

from abc import ABC, abstractmethod
from typing import Any, Literal
from pydantic import BaseModel
from logging.handlers import RotatingFileHandler
from pathlib import Path

from team_gm.utils.dist_utils import is_rank_zero


DEFAULT_FORMATTER = logging.Formatter(
    fmt="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class Logger(ABC):
    @abstractmethod
    def log_step(self, metric: dict[str, float], step: int, epoch: int):
        """
        Log a step-wise metric.

        Parameters
        ----------
        metric : dict
            Dictionary of metrics to log.
        step : int
            Global step number.
        epoch : int
            Epoch number.
        """

    @abstractmethod
    def log_epoch(self, metric: dict[str, float], step: int, epoch: int):
        """
        Log an epoch-wise metric.

        Parameters
        ----------
        metric : dict
            Dictionary of metrics to log.
        step : int
            Global step number.
        epoch : int
            Epoch number.
        """

    @abstractmethod
    def log_config(self, config: BaseModel):
        """
        Log the configuration settings.

        Parameters
        ----------
        config : BaseModel
            Configuration object containing settings.
        """

    @abstractmethod
    def log_message(
        self,
        message: Any,
        level: Literal["info", "debug", "warning", "error", "critical"] = "info",
    ):
        """
        Log a general message.

        Parameters
        ----------
        message : Any
            The message to log.
        level : Literal["info", "debug", "warning", "error", "critical"]
            The logging level for the message. Defaults to "info".
        """


class DummyLogger(Logger):
    def __init__(self, *args, **kwargs):
        pass

    def log_step(self, metric: dict[str, float], step: int, epoch: int):
        pass

    def log_epoch(self, metric: dict[str, float], step: int, epoch: int):
        pass

    def log_config(self, config: BaseModel):
        pass

    def log_message(
        self,
        message: Any,
        level: Literal["info", "debug", "warning", "error", "critical"] = "info",
    ):
        pass


def rank_zero_logger(cls: type[Logger]) -> type[Logger]:
    if not is_rank_zero():
        return DummyLogger
    return cls


@rank_zero_logger
class StreamLogger(Logger):
    def __init__(self, level=logging.INFO):
        self.logger = logging.getLogger(__name__ + ".StreamLogger")
        self.logger.setLevel(level)
        self.logger.propagate = False

        handler = logging.StreamHandler()
        handler.setFormatter(DEFAULT_FORMATTER)
        self.logger.addHandler(handler)

    def log_step(self, metric: dict[str, float], step: int, epoch: int):
        msg = "  ".join(f"{k}={v:.4g}" for k, v in metric.items())
        self.logger.debug(f"Step {step:8d} (Epoch {epoch:5d}) │ {msg}")

    def log_epoch(self, metric: dict[str, float], step: int, epoch: int):
        msg = "  ".join(f"{k}={v:.4g}" for k, v in metric.items())
        self.logger.info(f"Epoch {epoch:5d} │ {msg}")

    def log_config(self, config: BaseModel):
        config_dict = config.model_dump()
        config_str = json.dumps(config_dict, indent=4)
        self.logger.debug(f"Configuration:\n{config_str}")

    def log_message(
        self,
        message: Any,
        level: Literal["info", "debug", "warning", "error", "critical"] = "info",
    ):
        getattr(self.logger, level)(message)


@rank_zero_logger
class FileLogger(Logger):
    def __init__(
        self,
        file_path: str,
        level=logging.DEBUG,
        max_bytes: int = 50 * 1024 * 1024,
        backup_count: int = 5,
    ):
        self.logger = logging.getLogger(__name__ + ".FileLogger")
        self.logger.setLevel(level)
        self.logger.propagate = False

        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            filename=file_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        handler.setFormatter(DEFAULT_FORMATTER)
        self.logger.addHandler(handler)

    def log_step(self, metric: dict[str, float], step: int, epoch: int):
        msg = "  ".join(f"{k}={v:.4g}" for k, v in metric.items())
        self.logger.debug(f"Step {step:8d} (Epoch {epoch:5d}) │ {msg}")

    def log_epoch(self, metric: dict[str, float], step: int, epoch: int):
        msg = "  ".join(f"{k}={v:.4g}" for k, v in metric.items())
        self.logger.info(f"Epoch {epoch:5d} │ {msg}")

    def log_config(self, config: BaseModel):
        config_dict = config.model_dump()
        config_str = json.dumps(config_dict, indent=4)
        self.logger.debug(f"Configuration:\n{config_str}")

    def log_message(
        self,
        message: Any,
        level: Literal["info", "debug", "warning", "error", "critical"] = "info",
    ):
        getattr(self.logger, level)(message)


@rank_zero_logger
class WandbLogger(Logger):
    def __init__(self, name: str = "default"):
        wandb.init(name=name)
        self.defined_epoch_metrics = set()

    def log_step(self, metric: dict[str, float], step: int, epoch: int):
        wandb.log({k + "_step": v for k, v in metric.items()}, step=step)

    def log_epoch(self, metric: dict[str, float], step: int, epoch: int):
        for key in metric:
            if key not in self.defined_epoch_metrics:
                wandb.define_metric(key, "epoch")
                self.defined_epoch_metrics.add(key)
        metric.update({"epoch": epoch})
        wandb.log(metric, step=step)

    def log_config(self, config: BaseModel):
        wandb.config.update(config.model_dump())
        wandb.config.update({"world_size": int(os.environ.get("WORLD_SIZE", 1))})

    def log_message(
        self,
        message: Any,
        level: Literal["info", "debug", "warning", "error", "critical"] = "info",
    ):
        pass
