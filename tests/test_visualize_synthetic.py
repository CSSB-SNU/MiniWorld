"""Quick test: generate synthetic diffusion trajectory and visualize."""
from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from tests.experiment_schedulers import (
    MetricRow,
    TrajectoryFrame,
    generate_visualizations,
    visualize_rmsd_vs_sigma,
    visualize_sigma_schedule,
    write_metrics_csv,
)


def make_synthetic_trajectory(
    num_atoms: int = 200,
    num_steps: int = 32,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, list[TrajectoryFrame], list[MetricRow]]:
    """Create a synthetic protein-like structure and noisy diffusion trajectory."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Ground truth: helix-like structure
    t = torch.linspace(0, 4 * np.pi, num_atoms)
    x0 = torch.stack(
        [5.0 * torch.cos(t), 5.0 * torch.sin(t), 1.5 * t / np.pi],
        dim=-1,
    ).unsqueeze(0)
    mask = torch.ones(1, num_atoms, dtype=torch.bool)

    # EDM-like sigma schedule: log-linear from high to low
    sigmas = torch.exp(torch.linspace(np.log(160.0), np.log(0.002), num_steps + 1))

    frames = []
    metrics = []

    for step in range(num_steps + 1):
        sigma = sigmas[step].item()
        noise = torch.randn_like(x0) * sigma
        noisy_pos = x0 + noise

        diff = (noisy_pos - x0).pow(2).sum(dim=-1)
        global_rmsd = torch.sqrt(diff[mask].mean()).item()

        frames.append(
            TrajectoryFrame(
                step=step,
                tag="init" if step == 0 else "sample",
                atom_pos=noisy_pos.detach().clone(),
            ),
        )
        metrics.append(
            MetricRow(
                step=step,
                tag="init" if step == 0 else "sample",
                sigma_y=sigma,
                sigma_rotation=None,
                sigma_translation=None,
                global_rmsd=global_rmsd,
                local_rmsd=None,
            ),
        )

    return x0, mask, frames, metrics


def main() -> None:
    output_dir = Path("tests/output/synthetic_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic diffusion trajectory...")
    x0, mask, frames, metrics = make_synthetic_trajectory()

    print(f"  num_atoms={x0.shape[1]}, num_steps={len(frames)-1}")
    print(f"  sigma range: {metrics[0].sigma_y:.2f} -> {metrics[-1].sigma_y:.4f}")
    print(f"  RMSD range:  {metrics[0].global_rmsd:.2f} -> {metrics[-1].global_rmsd:.4f}")

    # Save metrics CSV
    write_metrics_csv(metrics, output_dir / "metrics.csv")

    # Generate all visualizations
    generate_visualizations(output_dir, frames, x0, mask)

    print(f"\nDone! Check {output_dir / 'visualizations'} for outputs.")


if __name__ == "__main__":
    main()
