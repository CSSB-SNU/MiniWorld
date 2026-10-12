import time
from collections import defaultdict
from typing import cast

import wandb

from miniworld.training.client import BaseClient


class MetricsAggregator:
    """Metrics aggregator for logging training metrics."""

    def __init__(
        self,
        client: BaseClient,
        group: str = "train",
        *,
        use_wandb: bool = False,
    ) -> None:
        self.client = client
        self.group = group
        self.use_wandb = use_wandb

        self.defined_epoch_keys = set()
        self.total_results = defaultdict(list)
        self.step_results = defaultdict(list)
        self.epoch_start_time: float | None = None
        self._last_global_step: int = 0

    def log_step(self, result: dict[str, float], *, ignore_step: bool = False) -> None:
        """Log metrics for a training step.

        Parameters
        ----------
        result : dict[str, float]
            A dictionary containing metric names and their corresponding float values.
        ignore_step : bool, optional
            Whether to ignore logging for this step, by default False.

        """
        if self.epoch_start_time is None:
            self.epoch_start_time = time.perf_counter()

        for k, v in result.items():
            self.total_results[k].append(v)
            self.step_results[k].append(v)

        if self.client.global_step > self._last_global_step:
            step_means = {k: sum(v) / len(v) for k, v in self.step_results.items()}
            step_means = cast(
                "dict[str, float]", self.client.fabric.all_reduce(step_means)
            )

            if not ignore_step and self.client.is_global_zero:
                msg = f"Step {self.client.global_step:8d} "
                msg += f"(Epoch {self.client.epoch:5d}) │ "
                msg += "  ".join(
                    f"{self.group}/{k}={v:.4g}" for k, v in step_means.items()
                )
                self.client.logger.debug(msg)

                if self.use_wandb:
                    wandb.log(
                        {f"{self.group}/{k}_step": v for k, v in step_means.items()},
                        step=self.client.global_step,
                    )

            self.step_results = defaultdict(list)
            self._last_global_step = self.client.global_step

    def log_epoch(self) -> dict[str, float]:
        """Log metrics for an epoch.

        Flushes the accumulated metrics and computes their mean values for the epoch.

        Returns
        -------
        dict[str, float]
            A dictionary containing the mean values of metrics for the epoch.

        """
        means = {k: sum(v) / len(v) for k, v in self.total_results.items()}
        means = cast("dict[str, float]", self.client.fabric.all_reduce(means))

        if self.epoch_start_time is not None:
            epoch_time = time.perf_counter() - self.epoch_start_time
            means["epoch_time"] = epoch_time
            self.epoch_start_time = None

        self.total_results = defaultdict(list)
        if not self.client.is_global_zero:
            return means

        msg = f"Epoch {self.client.epoch:5d} │ "
        msg += "  ".join(f"{self.group}/{k}={v:.4g}" for k, v in means.items())
        self.client.logger.info(msg)

        if self.use_wandb:
            reports = {f"{self.group}/{k}": v for k, v in means.items()}
            for key in reports:
                if key not in self.defined_epoch_keys:
                    wandb.define_metric(key, step_metric="epoch")
                    self.defined_epoch_keys.add(key)
            reports["epoch"] = self.client.epoch
            wandb.log(reports, step=self.client.global_step)

        return means
