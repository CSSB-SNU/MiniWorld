#!/usr/bin/env python3
"""Plot method-comparison diffusion loss curves from local MiniWorld logs."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "lossplots"

LOSS_RE = re.compile(
    r"Step\s+(?P<step>\d+)\s+\(Epoch\s+(?P<epoch>\d+)\).*?"
    r"train/diffusion_loss=(?P<diff>[0-9.eE+-]+).*?"
    r"train/distogram_loss=(?P<dist>[0-9.eE+-]+).*?"
    r"train/total_loss=(?P<total>[0-9.eE+-]+)"
)


@dataclass(frozen=True)
class Run:
    key: str
    label: str
    log: str
    color: str
    linestyle: str = "-"


RUNS = {
    "trunk_single": Run(
        "trunk_single",
        "With trunk single",
        "logs/exp/msa3_24_3-edm/2026-06-12/"
        "010351_exp-msa3_24_3-edm-revisit420/train.log",
        "#7f7f7f",
    ),
    "qknorm_off": Run(
        "qknorm_off",
        "No trunk single, no QKNorm",
        "logs/exp/msa3_24_3-no-single-edm/2026-06-13/"
        "133709_exp-msa3_24_3-no-single-edm-revisit420/train.log",
        "#d55e00",
    ),
    "qknorm_on": Run(
        "qknorm_on",
        "QKNorm + bf16",
        "logs/exp/msa3_24_3-no-single-qknorm-bf16-edm/2026-06-15/"
        "161239_exp-msa3_24_3-no-single-qknorm-bf16-edm-revisit420/train.log",
        "#0072b2",
    ),
    "qknorm_fp32attn": Run(
        "qknorm_fp32attn",
        "QKNorm fp32 + bf16 attn core",
        "logs/exp/msa3_24_3-no-single-qknorm-fp32attnbf16-edm/2026-06-17/"
        "203509_exp-msa3_24_3-no-single-qknorm-fp32attn-edm-revisit420/train.log",
        "#009e73",
    ),
    "biasnorm": Run(
        "biasnorm",
        "QKNorm + bf16 + biasnorm",
        "logs/exp/msa3_24_3-no-single-qknorm-bf16-biasnorm-edm/2026-06-19/"
        "134744_exp-msa3_24_3-no-single-qknorm-bf16-biasnorm-edm-revisit420/train.log",
        "#cc79a7",
    ),
    "swa": Run(
        "swa",
        "SWA atom, ESM-style",
        "logs/exp/msa3_24_3-no-single-swa-fixinit-edm/2026-06-23/"
        "173915_exp-msa3_24_3-no-single-swa-fixinit-edm-revisit420/train.log",
        "#56b4e9",
    ),
    "ropeswa": Run(
        "ropeswa",
        "RoPE-SWA, AF3-style",
        "logs/exp/msa3_24_3-no-single-ropeswa-af3-edm/2026-06-25/"
        "131716_exp-msa3_24_3-no-single-ropeswa-af3-edm-revisit420/train.log",
        "#e69f00",
    ),
    "ropeglobal": Run(
        "ropeglobal",
        "RoPE-global, AF3-style",
        "logs/exp/msa3_24_3-no-single-ropeglobal-af3-edm/2026-06-27/"
        "212523_exp-msa3_24_3-no-single-ropeglobal-af3-edm-revisit420/train.log",
        "#000000",
    ),
    "edm2": Run(
        "edm2",
        "EDM2 forced-WN",
        "logs/exp/msa3_24_3-no-single-qknorm-bf16-biasnorm-mp-edm/2026-06-22/"
        "002127_exp-msa3_24_3-no-single-qknorm-bf16-biasnorm-mp-edm-revisit420/train.log",
        "#999999",
    ),
}


PLOTS = [
    (
        "00_trunk_single_vs_no_single",
        "Trunk single is not required",
        ["trunk_single", "qknorm_off"],
        "No-single trunk still learns the diffusion module.",
        (0.0, 0.45),
    ),
    (
        "01_qknorm_vs_no_qknorm",
        "QKNorm prevents loss drift",
        ["qknorm_off", "qknorm_on"],
        "QKNorm-off run stays much higher / less stable than QKNorm.",
        (0.0, 0.38),
    ),
    (
        "02_precision_biasnorm",
        "Precision matters; biasnorm does not",
        ["qknorm_on", "qknorm_fp32attn", "biasnorm"],
        "bf16 vs fp32-attn differs modestly; biasnorm overlaps QKNorm.",
        (0.02, 0.08),
    ),
    (
        "03_pairbias_removed_swa",
        "SWA works without atom pair bias",
        ["biasnorm", "swa"],
        "SWA atom attention reaches the same low-loss regime.",
        (0.02, 0.10),
    ),
    (
        "04_af3_vs_esm_style",
        "AF3-style drops faster early",
        ["swa", "ropeswa"],
        "AF3-style RoPE-SWA drops faster early; final loss is similar.",
        (0.02, 0.12),
    ),
    (
        "05_swa_vs_global",
        "Global attention is not better",
        ["ropeswa", "ropeglobal"],
        "RoPE-global remains stable but sits a little above RoPE-SWA.",
        (0.022, 0.045),
    ),
]


def parse_log(run: Run) -> list[dict[str, float]]:
    path = ROOT / run.log
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = LOSS_RE.search(line)
            if not match:
                continue
            rows.append(
                {
                    "step": int(match.group("step")),
                    "epoch": int(match.group("epoch")),
                    "epoch_from_420": int(match.group("epoch")) - 420,
                    "diffusion_loss": float(match.group("diff")),
                    "distogram_loss": float(match.group("dist")),
                    "total_loss": float(match.group("total")),
                }
            )
    if not rows:
        raise RuntimeError(f"No loss rows parsed from {path}")
    return rows


def epoch_median(rows: list[dict[str, float]]) -> list[tuple[int, float]]:
    by_epoch: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        by_epoch[int(row["epoch_from_420"])].append(row["diffusion_loss"])
    medians = []
    for epoch in sorted(by_epoch):
        values = sorted(by_epoch[epoch])
        mid = len(values) // 2
        if len(values) % 2:
            median = values[mid]
        else:
            median = 0.5 * (values[mid - 1] + values[mid])
        medians.append((epoch, median))
    return medians


def rolling(points: list[tuple[int, float]], window: int = 5) -> list[tuple[int, float]]:
    smoothed = []
    for index, (epoch, _) in enumerate(points):
        start = max(0, index - window + 1)
        values = [value for _, value in points[start : index + 1]]
        smoothed.append((epoch, sum(values) / len(values)))
    return smoothed


def write_csv(parsed: dict[str, list[dict[str, float]]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "diffusion_loss_points.csv"
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_key",
                "run_label",
                "step",
                "epoch",
                "epoch_from_420",
                "diffusion_loss",
                "distogram_loss",
                "total_loss",
            ],
        )
        writer.writeheader()
        for key, rows in parsed.items():
            for row in rows:
                writer.writerow({"run_key": key, "run_label": RUNS[key].label, **row})


def plot_one(
    filename: str,
    title: str,
    keys: list[str],
    subtitle: str,
    ylim: tuple[float, float],
    parsed: dict[str, list[dict[str, float]]],
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=180, constrained_layout=True)
    for key in keys:
        run = RUNS[key]
        points = epoch_median(parsed[key])
        smooth = rolling(points)
        x_raw = [x for x, _ in points]
        y_raw = [y for _, y in points]
        x_smooth = [x for x, _ in smooth]
        y_smooth = [y for _, y in smooth]
        ax.plot(x_raw, y_raw, color=run.color, alpha=0.22, linewidth=1.0)
        ax.plot(
            x_smooth,
            y_smooth,
            color=run.color,
            linestyle=run.linestyle,
            linewidth=2.4,
            label=run.label,
        )

    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=8)
    ax.set_xlabel("Epochs after pretraining checkpoint (epoch - 420)")
    ax.set_ylabel("diffusion loss (epoch median, 5-epoch rolling mean)")
    ax.set_ylim(*ylim)
    ax.grid(True, axis="y", color="#dddddd", linewidth=0.8)
    ax.grid(True, axis="x", color="#eeeeee", linewidth=0.5)
    ax.legend(frameon=True, framealpha=0.88, facecolor="white", edgecolor="#dddddd", fontsize=8, loc="upper right")
    fig.savefig(OUT_DIR / f"{filename}.png", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def main() -> None:
    parsed = {key: parse_log(run) for key, run in RUNS.items()}
    write_csv(parsed)
    for args in PLOTS:
        plot_one(*args, parsed=parsed)


if __name__ == "__main__":
    main()
