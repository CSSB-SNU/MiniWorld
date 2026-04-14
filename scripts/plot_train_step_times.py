from __future__ import annotations

import argparse
import re
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TypeVar

import matplotlib.pyplot as plt


STEP_RE = re.compile(
    r"^\[(?P<ts>[^\]]+)\]"
    r"(?:\[rank=(?P<rank>\d+)\])?"
    r"\[Client\]\[DEBUG\] Step\s+(?P<step>\d+) \(Epoch\s+(?P<epoch>\d+)\)"
)
EPOCH_RE = re.compile(
    r"^\[(?P<ts>[^\]]+)\]"
    r"(?:\[rank=(?P<rank>\d+)\])?"
    r"\[Client\]\[INFO\] Training Epoch (?P<epoch>\d+)"
)


@dataclass(frozen=True)
class StepRecord:
    ts: datetime
    rank: int | None
    epoch: int
    step: int


@dataclass(frozen=True)
class EpochMarker:
    ts: datetime
    rank: int | None
    epoch: int


RecordT = TypeVar("RecordT")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot step-to-step training times from a MiniWorld train.log and "
            "draw vertical lines at epoch boundaries."
        ),
    )
    parser.add_argument("log_path", type=Path, help="Path to train.log")
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help=(
            "Rank to plot. If omitted, rank 0 is used when the log contains "
            "rank tags; otherwise the unranked log is used."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="PNG output path. Default: next to the log file.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="CSV output path. Default: next to the log file.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional plot title override.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="PNG DPI. Default: 180",
    )
    return parser.parse_args()


def parse_timestamp(raw: str) -> datetime:
    return datetime.strptime(raw, "%Y-%m-%d %H:%M:%S")


def load_records(log_path: Path) -> tuple[list[StepRecord], list[EpochMarker], set[int]]:
    step_records: list[StepRecord] = []
    epoch_markers: list[EpochMarker] = []
    seen_ranks: set[int] = set()

    for line in log_path.read_text(encoding="utf-8").splitlines():
        step_match = STEP_RE.match(line)
        if step_match:
            rank = (
                int(step_match.group("rank"))
                if step_match.group("rank") is not None
                else None
            )
            if rank is not None:
                seen_ranks.add(rank)
            step_records.append(
                StepRecord(
                    ts=parse_timestamp(step_match.group("ts")),
                    rank=rank,
                    epoch=int(step_match.group("epoch")),
                    step=int(step_match.group("step")),
                ),
            )
            continue

        epoch_match = EPOCH_RE.match(line)
        if epoch_match:
            rank = (
                int(epoch_match.group("rank"))
                if epoch_match.group("rank") is not None
                else None
            )
            if rank is not None:
                seen_ranks.add(rank)
            epoch_markers.append(
                EpochMarker(
                    ts=parse_timestamp(epoch_match.group("ts")),
                    rank=rank,
                    epoch=int(epoch_match.group("epoch")),
                ),
            )

    return step_records, epoch_markers, seen_ranks


def choose_rank(requested_rank: int | None, seen_ranks: set[int]) -> int | None:
    if requested_rank is not None:
        return requested_rank
    if not seen_ranks:
        return None
    if 0 in seen_ranks:
        return 0
    return min(seen_ranks)


def filter_by_rank(records: list[RecordT], rank: int | None) -> list[RecordT]:
    return [record for record in records if getattr(record, "rank") == rank]


def default_output_path(log_path: Path, rank: int | None, suffix: str) -> Path:
    rank_tag = f"_rank{rank}" if rank is not None else ""
    return log_path.with_name(f"{log_path.stem}{rank_tag}_step_times{suffix}")


def format_rank_label(rank: int | None) -> str:
    if rank is None:
        return "unranked"
    return f"rank={rank}"


def build_plot(
    steps: list[StepRecord],
    epoch_markers: list[EpochMarker],
    output_path: Path,
    title: str,
    dpi: int,
) -> list[tuple[int, float]]:
    xs: list[int] = []
    ys: list[float] = []

    prev_step = steps[0]
    for step in steps[1:]:
        xs.append(step.step)
        ys.append((step.ts - prev_step.ts).total_seconds())
        prev_step = step

    plt.figure(figsize=(14, 6))
    plt.plot(xs, ys, marker="o", markersize=2.5, linewidth=1, label="step delta")

    mean_sec = statistics.mean(ys)
    median_sec = statistics.median(ys)
    plt.axhline(
        mean_sec,
        color="tab:red",
        linestyle="--",
        linewidth=1,
        label=f"mean={mean_sec:.1f}s",
    )
    plt.axhline(
        median_sec,
        color="tab:green",
        linestyle=":",
        linewidth=1,
        label=f"median={median_sec:.1f}s",
    )

    boundary_steps = {
        marker.epoch: next(
            (step.step for step in steps if step.epoch == marker.epoch),
            None,
        )
        for marker in epoch_markers
    }
    for epoch, boundary_step in sorted(boundary_steps.items()):
        if boundary_step is None:
            continue
        plt.axvline(
            boundary_step,
            color="gray",
            linestyle="--",
            linewidth=0.8,
            alpha=0.5,
        )
        plt.text(
            boundary_step,
            plt.ylim()[1] * 0.98,
            f"E{epoch}",
            rotation=90,
            va="top",
            ha="right",
            fontsize=8,
            color="gray",
        )

    plt.title(title)
    plt.xlabel("Global step")
    plt.ylabel("Seconds since previous logged step")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()

    return list(zip(xs, ys))


def write_csv(points: list[tuple[int, float]], csv_path: Path) -> None:
    with csv_path.open("w") as handle:
        handle.write("step,delta_seconds\n")
        for step, delta_seconds in points:
            handle.write(f"{step},{delta_seconds}\n")


def main() -> None:
    args = parse_args()
    step_records, epoch_markers, seen_ranks = load_records(args.log_path)
    rank = choose_rank(args.rank, seen_ranks)

    filtered_steps = filter_by_rank(step_records, rank)
    filtered_epoch_markers = filter_by_rank(epoch_markers, rank)

    if len(filtered_steps) < 2:
        msg = (
            f"Not enough step records to plot for rank={rank}. "
            "Check --rank or the log contents."
        )
        raise SystemExit(msg)

    output_path = args.output or default_output_path(args.log_path, rank, ".png")
    csv_path = args.csv_output or default_output_path(args.log_path, rank, ".csv")
    title = args.title or f"Step-to-Step Time ({format_rank_label(rank)})"

    points = build_plot(
        steps=filtered_steps,
        epoch_markers=filtered_epoch_markers,
        output_path=output_path,
        title=title,
        dpi=args.dpi,
    )
    write_csv(points, csv_path)

    deltas = [delta for _, delta in points]
    print(f"log={args.log_path}")
    print(f"rank={rank}")
    if seen_ranks:
        print(f"available_ranks={sorted(seen_ranks)}")
    print(f"steps={len(filtered_steps)}")
    print(f"epochs={sorted({step.epoch for step in filtered_steps})}")
    print(f"mean_sec={statistics.mean(deltas):.3f}")
    print(f"median_sec={statistics.median(deltas):.3f}")
    print(f"min_sec={min(deltas):.3f}")
    print(f"max_sec={max(deltas):.3f}")
    print(f"plot={output_path}")
    print(f"csv={csv_path}")


if __name__ == "__main__":
    main()
