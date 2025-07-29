import torch
import time

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .client import BaseClient
from team_gm.utils.dist_utils import rank_zero_only


class Callback:
    def on_train_epoch_start(self, client: "BaseClient") -> None:
        """Called when the train epoch begins."""
        pass

    def on_train_epoch_end(
        self, client: "BaseClient", mean_loss: torch.Tensor, outputs: list
    ) -> None:
        """Called when the train epoch ends."""
        pass

    def on_validation_epoch_start(self, client: "BaseClient") -> None:
        """Called when the validation epoch begins."""
        pass

    def on_validation_epoch_end(self, client: "BaseClient", outputs: list) -> None:
        """Called when the validation epoch ends."""
        pass

    def on_train_batch_start(
        self, client: "BaseClient", batch: Any, batch_idx: int
    ) -> None:
        """Called when the train batch begins."""
        pass

    def on_train_batch_end(
        self, client: "BaseClient", output: Any, batch: Any, batch_idx: int
    ) -> None:
        """Called when the train batch ends."""
        pass

    def on_validation_batch_start(
        self, client: "BaseClient", batch: Any, batch_idx: int
    ) -> None:
        """Called when the validation batch begins."""
        pass

    def on_validation_batch_end(
        self, client: "BaseClient", output: Any, batch: Any, batch_idx: int
    ) -> None:
        """Called when the validation batch ends."""
        pass


class DefaultCallback(Callback):
    def __init__(self):
        self.train_start = None
        self.valid_start = None

    def on_train_epoch_start(self, client: "BaseClient") -> None:
        self.train_start = time.time()
        client.log_message(f"Start training {client.epoch} epoch")

    def on_train_epoch_end(
        self, client: "BaseClient", mean_loss: torch.Tensor, outputs: list
    ) -> None:
        if self.train_start is None:
            raise RuntimeError(
                "on_train_epoch_start must be called before on_train_epoch_end."
            )

        client.log_metrics(
            {"train/lr": client.lr},
            on_step=True,
        )
        client.log_metrics(
            {"train/time": time.time() - self.train_start},
            on_epoch=True,
        )

    def on_validation_epoch_start(self, client: "BaseClient") -> None:
        self.valid_start = time.time()
        client.log_message(f"Start validation {client.epoch} epoch")

    def on_validation_epoch_end(self, client: "BaseClient", outputs: list) -> None:
        if self.valid_start is None:
            raise RuntimeError(
                "on_validation_epoch_start must be called before "
                "on_validation_epoch_end."
            )
        client.log_metrics(
            {"valid/time": time.time() - self.valid_start},
            on_epoch=True,
        )


class SaveCheckpointPeriodic(Callback):
    """Save the checkpoint periodically during training."""

    def __init__(self, dirpath: str | Path, every_n_epochs: int = 1):
        self.dirpath = Path(dirpath)
        self.every_n_epochs = every_n_epochs
        self.dirpath.mkdir(parents=True, exist_ok=True)

    def on_train_epoch_end(
        self, client: "BaseClient", mean_loss: torch.Tensor, outputs: list
    ) -> None:
        if client.epoch % self.every_n_epochs != 0:
            return

        filename = f"{client.name}_epoch={client.epoch:04d}.pt"
        client.save_checkpoint(self.dirpath / filename)
        client.log_message(
            f"Model checkpoint saved: {self.dirpath / filename}", level="info"
        )


class SaveCheckpointBest(Callback):
    """Save the best checkpoint based on train loss."""

    def __init__(self, dirpath: str | Path):
        self.dirpath = Path(dirpath)
        self.dirpath.mkdir(parents=True, exist_ok=True)
        self.best_loss = float("inf")

    def on_train_epoch_end(
        self, client: "BaseClient", mean_loss: torch.Tensor, outputs: list
    ) -> None:
        loss = mean_loss.item()
        if loss < self.best_loss:
            self.best_loss = loss
            filename = f"{client.name}_best.pt"
            client.save_checkpoint(self.dirpath / filename)
            client.log_message(
                f"Best model checkpoint saved: {self.dirpath / filename}", level="info"
            )


class EarlyStopping(Callback):
    """Stop training when a monitored metric has stopped improving."""

    def __init__(self, patience: int = 3, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = abs(min_delta)
        self.wait_count = 0
        self.best_loss = None

    @rank_zero_only
    def on_train_epoch_end(
        self, client: "BaseClient", mean_loss: torch.Tensor, outputs: list
    ) -> None:
        loss = mean_loss.item()
        if self.best_loss is None:
            self.best_loss = loss

        if loss < (self.best_loss - self.min_delta):
            self.best_loss = loss
            self.wait_count = 0
        else:
            self.wait_count += 1

        if self.wait_count >= self.patience:
            client.log_message(
                f"Early stopping triggered after {self.wait_count} epochs "
                f"without improvement. Best loss: {self.best_loss}"
            )
            client.stop()
