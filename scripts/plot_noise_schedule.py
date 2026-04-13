"""Plot decoupled x-prediction noise schedule: sigma_R, sigma_T vs sigma_y with training density overlay."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra import compose, initialize_config_dir

from miniworld.configs import XPredDecoupledDiffuserConfig
from miniworld.diffusion.decoupled_xpred.scheduler import DecoupledXPredScheduler


def plot_noise_schedule(scheduler: DecoupledXPredScheduler, save_dir: Path) -> None:  # noqa: PLR0915
    save_dir.mkdir(parents=True, exist_ok=True)

    cfg = scheduler.config
    sigma_y_min = cfg.sigma_y_min * cfg.sigma_data
    sigma_y_max = cfg.sigma_y_max * cfg.sigma_data

    # Deterministic mapping: sigma_R, sigma_T vs sigma_y
    sigma_y_vals = torch.logspace(np.log10(sigma_y_min), np.log10(sigma_y_max), 500)
    sigma_R, sigma_T = scheduler.convert_to_sigma_rt(sigma_y_vals)
    sigma_y_np = sigma_y_vals.numpy()

    # Training sigma_y samples for density
    n = 200_000
    sigma_y_samples, _, _ = scheduler.sample_noise(n)
    sigma_y_s = sigma_y_samples.numpy()

    # Histogram density in log10 space
    log_bins = np.linspace(np.log10(sigma_y_np.min()), np.log10(sigma_y_np.max()), 150)
    hist, bin_edges = np.histogram(np.log10(sigma_y_s), bins=log_bins, density=True)
    bin_centers = 10 ** (0.5 * (bin_edges[:-1] + bin_edges[1:]))

    phase_1_boundary = cfg.sigma_y_phase_1_boundary
    phase_2_boundary = cfg.sigma_y_phase_2_boundary

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # sigma_R vs sigma_y + density
    ax1 = axes[0]
    ax1_twin = ax1.twinx()
    ax1.plot(sigma_y_np, sigma_R.numpy(), linewidth=2, color="tab:blue", label="sigma_R")
    ax1_twin.fill_between(
        bin_centers,
        0,
        hist,
        alpha=0.2,
        color="gray",
        label="sigma_y density",
    )
    ax1_twin.plot(bin_centers, hist, color="gray", alpha=0.5, linewidth=0.8)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("sigma_y")
    ax1.set_ylabel("sigma_R", color="tab:blue")
    ax1_twin.set_ylabel("sigma_y density (log10 scale)", color="gray")
    ax1.set_title("sigma_R vs sigma_y")
    ax1.axvline(
        phase_1_boundary,
        color="red",
        linestyle="--",
        alpha=0.5,
        label="phase 1/2",
    )
    ax1.axvline(
        phase_2_boundary,
        color="red",
        linestyle=":",
        alpha=0.5,
        label="phase 2/3",
    )
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    ax1.grid(visible=True, alpha=0.2)

    # sigma_T vs sigma_y + density
    ax2 = axes[1]
    ax2_twin = ax2.twinx()
    ax2.plot(
        sigma_y_np,
        sigma_T.numpy(),
        linewidth=2,
        color="tab:orange",
        label="sigma_T",
    )
    ax2_twin.fill_between(
        bin_centers,
        0,
        hist,
        alpha=0.2,
        color="gray",
        label="sigma_y density",
    )
    ax2_twin.plot(bin_centers, hist, color="gray", alpha=0.5, linewidth=0.8)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("sigma_y")
    ax2.set_ylabel("sigma_T", color="tab:orange")
    ax2_twin.set_ylabel("sigma_y density (log10 scale)", color="gray")
    ax2.set_title("sigma_T vs sigma_y")
    ax2.axvline(
        phase_1_boundary,
        color="red",
        linestyle="--",
        alpha=0.5,
        label="phase 1/2",
    )
    ax2.axvline(
        phase_2_boundary,
        color="red",
        linestyle=":",
        alpha=0.5,
        label="phase 2/3",
    )
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    ax2.grid(visible=True, alpha=0.2)

    fig.tight_layout()
    out_path = save_dir / "sigma_RT_with_density.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    # --- Stats ---
    (sigma_y_s >= phase_1_boundary).mean() * 100
    ((sigma_y_s >= phase_2_boundary) & (sigma_y_s < phase_1_boundary)).mean() * 100
    (sigma_y_s < phase_2_boundary).mean() * 100


if __name__ == "__main__":
    import click

    @click.command()
    @click.option(
        "--config",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        default=Path("configs/miniworld/config_test.yaml"),
        help="config file (reads diffuser section)",
    )
    @click.option(
        "--output-dir",
        type=click.Path(path_type=Path),
        default=Path("outputs"),
        help="directory to save plots",
    )
    def main(config: Path, output_dir: Path) -> None:
        with initialize_config_dir(str(config.parent.absolute()), version_base=None):
            cfg = compose(config_name=config.name)

        diffuser_cfg = XPredDecoupledDiffuserConfig(**cfg["diffuser"])
        scheduler = DecoupledXPredScheduler(diffuser_cfg.scheduler)
        plot_noise_schedule(scheduler, output_dir)

    main()
