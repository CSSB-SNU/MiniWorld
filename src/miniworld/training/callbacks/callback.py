from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from miniworld.training.client import BaseClient


class Callback:
    """Base class for all callbacks."""

    def on_train_epoch_start(self, client: BaseClient) -> None:
        """Event called when the train epoch begins."""

    def on_train_epoch_end(self, client: BaseClient) -> None:
        """Event called when the train epoch ends."""

    def on_validation_epoch_start(self, client: BaseClient) -> None:
        """Event called when the validation epoch begins."""

    def on_validation_epoch_end(self, client: BaseClient) -> None:
        """Event called when the validation epoch ends."""

    def on_train_step_start(
        self,
        client: BaseClient,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Event called when the training step starts."""

    def on_train_batch_start(
        self,
        client: BaseClient,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Event called before every training batch, including accumulation steps."""

    def on_train_batch_end(
        self,
        client: BaseClient,
        batch: Any,
        batch_idx: int,
        results: Any,
    ) -> None:
        """Event called after every training batch, including accumulation steps."""

    def on_train_step_end(
        self,
        client: BaseClient,
        batch: Any,
        batch_idx: int,
        results: Any,
    ) -> None:
        """Event called when the training step ends (after optimizer step)."""

    def on_validation_step_start(
        self,
        client: BaseClient,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Event called when the validation step starts."""

    def on_validation_step_end(
        self,
        client: BaseClient,
        batch: Any,
        batch_idx: int,
        results: Any,
    ) -> None:
        """Event called when the validation step ends."""

    def on_save_checkpoint(self, client: BaseClient, state_dict: dict) -> None:
        """Event called when a checkpoint is saved."""

    def on_load_state_dict(self, client: BaseClient, state_dict: dict) -> None:
        """Event called when a state dict is loaded."""
