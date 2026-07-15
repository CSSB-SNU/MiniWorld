#!/usr/bin/env python3
"""Parallel 8-way eval summarizer.

worker  --dirname X : score one variant (eval_dir), dump JSON cache
combine             : load all cached variants, write csv/md + 3 plots (same as summarize_eval_e510_40targets_8way.py)
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = [
    ("nosingle", "nosingle_e510", "#0d9488"),
    ("qknorm", "qknorm_bf16_e510", "#c0392b"),
    ("biasnorm", "biasnorm_e510", "#2b6cb0"),
    ("fp32attn", "fp32attn_e510", "#2a8a3a"),
    ("SWA", "swa_fixinit_e510", "#8e44ad"),
    ("ropeswa-af3", "ropeswa_af3_e510", "#e67e22"),
    ("ropeglobal-af3", "ropeglobal_af3_e510_centerstep", "#111827"),
    ("ropeglobal-af3-pt", "ropeglobal_af3_e510_pytorch_transition", "#db2777"),
]
ORACLE = ("oracle-best", "oracle-best", "#64748b")
METRICS = [
    ("lddt", "lDDT up", (0.50, 1.0), "up"),
    ("tm", "TM-score up", (0.25, 1.0), "up"),
    ("rmsd", "protein RMSD (A) down", (0.0, 9.0), "down"),
    ("rmsd_lig", "ligand RMSD (A) down", (0.0, 25.0), "down"),
]
SUMMARY_YLIMS = {
    "lddt": (0.82, 0.88),
    "tm": (0.85, 0.93),
    "rmsd": (0.0, 2.2),
    "rmsd_lig": (0.0, 7.0),
}
CACHE = "_cache8"


def load_eval_module():
    spec = importlib.util.spec_from_file_location("usalign_lddt_eval", "scripts/usalign_lddt_eval.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load scripts/usalign_lddt_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def target_tokens(log_path: Path) -> dict[str, int]:
    tokens: dict[str, int] = {}
    if not log_path.exists():
        return tokens
    for line in log_path.read_text(errors="replace").splitlines():
        match = re.search(r"target \d+/\d+ (\S+) \| n_tokens=(\d+)", line)
        if match:
            tokens[match.group(1)] = int(match.group(2))
    return tokens


def short_target(name: str) -> str:
    return name.replace("['", "").replace("']", "").split("_")[0]


def oracle_value(values, direction):
    values = [v for v in values if not np.isnan(v)]
    if not values:
        return float("nan")
    return max(values) if direction == "up" else min(values)


def build_oracle(results, common):
    oracle = {}
    for target in common:
        oracle[target] = {}
        for metric, _, _, direction in METRICS:
            oracle[target][metric] = oracle_value(
                [results[label][target].get(metric, np.nan) for label, _, _ in METHODS],
                direction,
            )
    return oracle


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, summary_rows):
    headers = ["method", "n", "mean_lddt", "mean_tm", "mean_rmsd", "mean_rmsd_lig",
               "median_lddt", "median_tm", "median_rmsd", "median_rmsd_lig"]
    lines = ["|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in summary_rows:
        lines.append("|" + "|".join([
            str(row["method"]), str(row["n"]),
            f"{row['mean_lddt']:.4f}", f"{row['mean_tm']:.4f}",
            f"{row['mean_rmsd']:.3f}", f"{row['mean_rmsd_lig']:.3f}",
            f"{row['median_lddt']:.4f}", f"{row['median_tm']:.4f}",
            f"{row['median_rmsd']:.3f}", f"{row['median_rmsd_lig']:.3f}",
        ]) + "|")
    path.write_text("\n".join(lines) + "\n")


def style_axes(ax):
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_violin(out_dir, all_results, common, labs):
    rng = np.random.RandomState(0)
    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    for ax, (metric, title, ylim, _) in zip(axes.flat, METRICS):
        data = [[all_results[label][t].get(metric, np.nan) for t in common
                 if not np.isnan(all_results[label][t].get(metric, np.nan))]
                for label, _, _ in labs]
        parts = ax.violinplot(data, showmeans=True, widths=0.8)
        for i, body in enumerate(parts["bodies"]):
            body.set_facecolor(labs[i][2]); body.set_alpha(0.45)
        for key in ("cbars", "cmins", "cmaxes", "cmeans"):
            if key in parts:
                parts[key].set_color("black")
        for i, values in enumerate(data):
            ax.scatter(rng.normal(i + 1, 0.05, len(values)), values, s=8, color=labs[i][2], alpha=0.55, zorder=3, edgecolors="none")
            if values:
                y = max(values) if "down" in title else 1.0
                ax.text(i + 1, y, f"{np.nanmean(values):.3f}", ha="center", fontsize=7, va="bottom")
        ax.set_xticks(range(1, len(labs) + 1))
        ax.set_xticklabels([label for label, _, _ in labs], rotation=15, fontsize=8)
        ax.set_title(title)
        if ylim:
            ax.set_ylim(*ylim)
        style_axes(ax)
    fig.suptitle("Per-metric over 40 targets (ep510, CoM fix, crystallization aids excluded; oracle is metric-wise best)")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_dir / "violin_metrics_8way.png", dpi=130)
    plt.close(fig)


def plot_per_target(out_dir, all_results, common, labs, tokens):
    order = sorted(common, key=lambda t: tokens.get(t, 0))
    x = np.arange(len(order))
    width = 0.82 / len(labs)
    target_labels = [short_target(t) for t in order]
    fig, axes = plt.subplots(4, 1, figsize=(26, 20))
    for ax, (metric, title, ylim, _) in zip(axes, METRICS):
        center = (len(labs) - 1) / 2
        for j, (label, _, color) in enumerate(labs):
            ax.bar(x + (j - center) * width, [all_results[label][t].get(metric, np.nan) for t in order], width, label=label, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(target_labels, rotation=90, fontsize=7)
        ax.set_title(title, loc="left")
        ax.legend(fontsize=8, ncol=len(labs), loc="upper right")
        if ylim:
            ax.set_ylim(*ylim)
        style_axes(ax)
    fig.suptitle("Per-target grouped bars (ep510, CoM fix, crystallization aids excluded; oracle is metric-wise best)")
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(out_dir / "per_target_scores_8way.png", dpi=100)
    plt.close(fig)


def plot_summary_bars(out_dir, summary_rows, labs):
    labels = [row["method"] for row in summary_rows]
    colors = [color for _, _, color in labs]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for ax, (metric, title, _, direction) in zip(axes.flat, METRICS):
        values = [float(row[f"mean_{metric}"]) for row in summary_rows]
        ax.bar(range(len(labels)), values, color=colors)
        ax.set_title("Mean " + title, loc="left")
        ax.set_xticks(range(len(labels)), labels, rotation=15, ha="right", fontsize=8)
        ax.set_ylim(*SUMMARY_YLIMS[metric])
        style_axes(ax)
    fig.suptitle("Mean metrics over 40 targets (oracle is metric-wise best)")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_dir / "summary_metrics_8way.png", dpi=130)
    plt.close(fig)


def summarize(all_results, common, labs):
    summary_rows, per_target_rows = [], []
    for label, dirname, _ in labs:
        row = {"method": label, "dirname": dirname, "n": len(common)}
        for metric, _, _, _ in METRICS:
            values = np.array([all_results[label][t].get(metric, np.nan) for t in common], dtype=float)
            row[f"mean_{metric}"] = float(np.nanmean(values))
            row[f"median_{metric}"] = float(np.nanmedian(values))
        summary_rows.append(row)
        for index, target in enumerate(common, start=1):
            tr = {"method": label, "dirname": dirname, "rank": index, "target": target}
            for metric, _, _, _ in METRICS:
                tr[metric] = all_results[label][target].get(metric, np.nan)
            per_target_rows.append(tr)
    return summary_rows, per_target_rows


def do_worker(base, dirname, usalign):
    eval_module = load_eval_module()
    res = eval_module.eval_dir(base / dirname, usalign, True)
    out = base / CACHE / f"{dirname}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        json.dump(res, fh, default=float)
    print(f"WORKER_DONE {dirname} n={len(res)}")


def do_combine(base):
    results = {}
    for label, dirname, _ in METHODS:
        with (base / CACHE / f"{dirname}.json").open() as fh:
            results[label] = json.load(fh)
    common = sorted(set.intersection(*[set(results[label]) for label, _, _ in METHODS]))
    if not common:
        raise RuntimeError("No common targets across methods")
    oracle = build_oracle(results, common)
    all_results = dict(results)
    all_results[ORACLE[0]] = oracle
    labs = METHODS + [ORACLE]
    tokens = target_tokens(base / "nosingle_e510" / "driver.log")

    summary_rows, per_target_rows = summarize(all_results, common, labs)
    write_csv(base / "summary_8way.csv", summary_rows,
              ["method", "dirname", "n", "mean_lddt", "mean_tm", "mean_rmsd", "mean_rmsd_lig",
               "median_lddt", "median_tm", "median_rmsd", "median_rmsd_lig"])
    write_csv(base / "per_target_scores_8way.csv", per_target_rows,
              ["method", "dirname", "rank", "target", "lddt", "tm", "rmsd", "rmsd_lig"])
    write_markdown(base / "summary_8way.md", summary_rows)
    plot_violin(base, all_results, common, labs)
    plot_per_target(base, all_results, common, labs, tokens)
    plot_summary_bars(base, summary_rows, labs)

    print(f"MEANS ({len(common)} targets)")
    print(f"{'metric':<12}" + "".join(f"{label:>18}" for label, _, _ in labs))
    for metric, _, _, _ in METRICS:
        print(f"{metric:<12}" + "".join(f"{row[f'mean_{metric}']:>18.3f}" for row in summary_rows))
    print("SAVED")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["worker", "combine"])
    parser.add_argument("--dirname")
    parser.add_argument("--base-dir", type=Path, default=Path("outputs/eval_e510_40targets_25samples"))
    parser.add_argument("--usalign", type=Path, default=Path("tools/USalign/USalign"))
    args = parser.parse_args()
    if args.mode == "worker":
        do_worker(args.base_dir, args.dirname, args.usalign)
    else:
        do_combine(args.base_dir)


if __name__ == "__main__":
    main()
