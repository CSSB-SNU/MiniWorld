"""Plot non-EMA validation curves for three atom_token model variants.

Reads JSONL files written by the three run_af3like*_rev_no_ema_valid.py
scripts and draws one panel per validation metric (best_rmsd, best_lddt,
vald_distogram_loss) with one curve per variant:

    atom_token              -> red
    atom_token_fingerprint  -> blue
    atom_token_explicit     -> yellow
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import click
import matplotlib.pyplot as plt


VALID_FIELDS = ["best_rmsd", "best_lddt", "vald_distogram_loss"]


@dataclass
class Series:
    epochs: list[int] = field(default_factory=list)
    values: list[float] = field(default_factory=list)


def load_jsonl(path: Path) -> dict[str, Series]:
    series: dict[str, Series] = {f: Series() for f in VALID_FIELDS}
    if not path.exists():
        return series
    rows: list[dict] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    rows.sort(key=lambda r: int(r.get("epoch", -1)))
    seen: set[int] = set()
    for row in rows:
        ep = int(row.get("epoch", -1))
        if ep in seen:
            continue
        seen.add(ep)
        for f in VALID_FIELDS:
            if f in row:
                series[f].epochs.append(ep)
                series[f].values.append(float(row[f]))
    return series


DEFAULT_VARIANTS = {
    "atom_token": {
        "color": "red",
        "jsonl": "loss_figure/no_ema_valid/atom_token.jsonl",
    },
    "atom_token_fingerprint": {
        "color": "blue",
        "jsonl": "loss_figure/no_ema_valid/atom_token_fingerprint.jsonl",
    },
    "atom_token_explicit": {
        "color": "yellow",
        "jsonl": "loss_figure/no_ema_valid/atom_token_explicit.jsonl",
    },
}


@click.command()
@click.option(
    "--atom-token",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path(DEFAULT_VARIANTS["atom_token"]["jsonl"]),
    show_default=True,
)
@click.option(
    "--atom-token-fingerprint",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path(DEFAULT_VARIANTS["atom_token_fingerprint"]["jsonl"]),
    show_default=True,
)
@click.option(
    "--atom-token-explicit",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path(DEFAULT_VARIANTS["atom_token_explicit"]["jsonl"]),
    show_default=True,
)
@click.option(
    "--out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("loss_figure/no_ema_valid/valid_curves.png"),
    show_default=True,
)
def main(
    atom_token: Path,
    atom_token_fingerprint: Path,
    atom_token_explicit: Path,
    out: Path,
) -> None:
    variants = {
        "atom_token": {"color": "red", "jsonl": atom_token},
        "atom_token_fingerprint": {"color": "blue", "jsonl": atom_token_fingerprint},
        "atom_token_explicit": {"color": "yellow", "jsonl": atom_token_explicit},
    }

    loaded: dict[str, dict[str, Series]] = {}
    for name, spec in variants.items():
        series = load_jsonl(Path(spec["jsonl"]))
        n = len(series["best_rmsd"].epochs) if "best_rmsd" in series else 0
        print(f"{name}: {n} points from {spec['jsonl']}")
        loaded[name] = series

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = {
        "best_rmsd": "valid/best_rmsd (no-EMA)",
        "best_lddt": "valid/best_lddt (no-EMA)",
        "vald_distogram_loss": "valid/vald_distogram_loss (no-EMA)",
    }
    for ax, field_name in zip(axes, VALID_FIELDS):
        for name, spec in variants.items():
            s = loaded[name].get(field_name)
            if s is None or not s.epochs:
                continue
            ax.plot(
                s.epochs,
                s.values,
                color=spec["color"],
                marker="o",
                linestyle="-",
                linewidth=1.5,
                markersize=5,
                markeredgecolor="black",
                markeredgewidth=0.3,
                label=name,
            )
        ax.set_xlabel("epoch")
        ax.set_ylabel(titles[field_name])
        ax.set_title(titles[field_name])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="best")

    fig.suptitle(
        "Validation curves (no EMA) — atom_token variants",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    main()
