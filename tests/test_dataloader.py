"""CLI helpers for inspecting and benchmarking the MiniWorld dataloader."""

from __future__ import annotations

import csv
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, ParamSpec, TypeVar, cast

import click
import matplotlib.pyplot as plt
import numpy as np

from miniworld.configs.data import (
    BioMolDBConfig,
    CropConfig,
    DynamicTokenizationConfig,
    MSAConfig,
    TokenizerConfig,
)
from miniworld.data.dataloader.dataloader import BioMolData, DataBias
from miniworld.data.features import Batch

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from torch.utils.data import DataLoader

plt.switch_backend("Agg")

DEFAULT_CIF_DB_PATH = Path(
    "/NHNHOME/WORKSPACE/0226010152_A/data/cif_attached_train_20260224_res9_chain300.lmdb",
)
DEFAULT_A3M_DB_PATH = Path("/NHNHOME/WORKSPACE/0226010152_A/data/a3m_16k.lmdb")
DEFAULT_EDGE_PATH = Path(
    "/NHNHOME/WORKSPACE/0226010152_A/data/metadata/train_20260224_edge_node.tsv",
)
DEFAULT_TEMPLATE_DB_PATH = Path("/NHNHOME/WORKSPACE/0226010152_A/data/template.lmdb")
DEFAULT_CCD_DB_PATH = Path(
    "/NHNHOME/WORKSPACE/0226010152_A/data/CCD/preprocessed_CCD.lmdb",
)
DEFAULT_OUTPUT_PREFIX = Path("tests/artifacts/dataloader_benchmark")
P = ParamSpec("P")
R = TypeVar("R")
MissingPolicy = Literal["gap", "query"]
TokenizerLevel = Literal["atom", "dynamic", "lte", "residue"]
FetchedItem = tuple[Batch, float, int | None, str, int | None]


@dataclass(frozen=True)
class DirectTiming:
    """Timing result for a direct ``dataset[index]`` fetch."""

    iteration: int
    dataset_index: int
    elapsed_sec: float
    sample_name: str


@dataclass(frozen=True)
class LoaderTiming:
    """Timing result for one exposed dataloader wait."""

    step: int
    wait_sec: float
    batch_size: int
    per_sample_wait_sec: float
    wait_over_train_ratio: float
    loader_share: float
    sample_names: tuple[str, ...]


@dataclass(frozen=True)
class DatasetOptions:
    """Shared dataset construction options."""

    max_tokens: int = 384
    max_atoms: int = 4096
    max_msa_depth: int = 384
    missing_policy: MissingPolicy = "query"
    tokenizer_level: TokenizerLevel = "dynamic"
    tokenizer_seed: int = 42
    sigma_flat_prob: float = 0.0
    sigma_min: float = 4.0
    sigma_max: float = 8.0
    cif_db_path: Path = DEFAULT_CIF_DB_PATH
    a3m_db_path: Path = DEFAULT_A3M_DB_PATH
    edge_path: Path = DEFAULT_EDGE_PATH
    template_db_path: Path = DEFAULT_TEMPLATE_DB_PATH
    ccd_db_path: Path = DEFAULT_CCD_DB_PATH


@dataclass(frozen=True)
class BenchmarkOptions(DatasetOptions):
    """Options for the benchmark command."""

    steps: int = 32
    direct_samples: int = 32
    warmup_steps: int = 4
    warmup_direct: int = 4
    train_seconds: float = 0.5
    batch_size: int = 1
    num_workers: int = 4
    prefetch_factor: int = 8
    seed: int = 42
    shuffle: bool = True
    drop_last: bool = False
    bucket_token_multiple: int = 128
    bucket_atom_multiple: int = 1024
    output_prefix: Path = DEFAULT_OUTPUT_PREFIX
    dpi: int = 180


@dataclass(frozen=True)
class ItemOptions(DatasetOptions):
    """Options for the item inspection command."""

    index: int | None = None
    name: str | None = None
    pdb_id: str | None = None
    assembly_id: str | None = None
    model_id: str | None = None
    alt_id: str | None = None
    chain_ids: tuple[str, ...] = ()
    match: int = 0
    seed: int = 42
    epoch: int = 0
    crop_indices: str | None = None
    allow_fallback: bool = False


def dataset_options(func: Callable[P, R]) -> Callable[P, R]:
    """Attach common dataset options to a click command."""
    options = [
        click.option("--max-tokens", type=int, default=384, help="Crop max tokens."),
        click.option("--max-atoms", type=int, default=4096, help="Crop max atoms."),
        click.option(
            "--max-msa-depth",
            type=int,
            default=384,
            help="Maximum sampled MSA depth.",
        ),
        click.option(
            "--missing-policy",
            type=click.Choice(["gap", "query"]),
            default="query",
            help="Policy used when MSA data are missing.",
        ),
        click.option(
            "--tokenizer-level",
            type=click.Choice(["atom", "dynamic", "lte", "residue"]),
            default="dynamic",
            help="Tokenizer level.",
        ),
        click.option("--tokenizer-seed", type=int, default=42, help="Tokenizer seed."),
        click.option(
            "--sigma-flat-prob",
            type=float,
            default=0.0,
            help="Dynamic tokenizer flat-sigma probability.",
        ),
        click.option(
            "--sigma-min",
            type=float,
            default=4.0,
            help="Dynamic tokenizer minimum sigma.",
        ),
        click.option(
            "--sigma-max",
            type=float,
            default=8.0,
            help="Dynamic tokenizer maximum sigma.",
        ),
        click.option(
            "--cif-db-path",
            type=click.Path(path_type=Path),
            default=DEFAULT_CIF_DB_PATH,
            help="CIF LMDB path.",
        ),
        click.option(
            "--a3m-db-path",
            type=click.Path(path_type=Path),
            default=DEFAULT_A3M_DB_PATH,
            help="A3M LMDB path.",
        ),
        click.option(
            "--edge-path",
            type=click.Path(path_type=Path),
            default=DEFAULT_EDGE_PATH,
            help="Edge metadata TSV path.",
        ),
        click.option(
            "--template-db-path",
            type=click.Path(path_type=Path),
            default=DEFAULT_TEMPLATE_DB_PATH,
            help="Template LMDB path.",
        ),
        click.option(
            "--ccd-db-path",
            type=click.Path(path_type=Path),
            default=DEFAULT_CCD_DB_PATH,
            help="Preprocessed CCD LMDB path.",
        ),
    ]
    decorated = func
    for option in reversed(options):
        decorated = cast("Callable[P, R]", option(decorated))
    return decorated


def ensure_paths_exist(paths: list[Path]) -> None:
    """Raise a click error when required data paths are missing."""
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        msg = "Missing required input paths:\n" + "\n".join(missing)
        raise click.ClickException(msg)


def build_dataset(args: DatasetOptions) -> BioMolData:
    """Build a BioMol dataset from CLI options."""
    dynamic_config = (
        DynamicTokenizationConfig(
            minimum_resolution_ratio=[0.2, 0.6, 0.2],
            sigma_flat_prob=args.sigma_flat_prob,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
        )
        if args.tokenizer_level == "dynamic"
        else None
    )
    config = BioMolData.BioMolConfig(
        crop_config=CropConfig(
            max_tokens=args.max_tokens,
            max_atoms=args.max_atoms,
            remain_invalid_tokens=False,
        ),
        msa_config=MSAConfig(
            max_msa_depth=args.max_msa_depth,
            missing_policy=args.missing_policy,
        ),
        DB_config=BioMolDBConfig(
            cif_db_path=args.cif_db_path,
            a3m_db_path=args.a3m_db_path,
            edge_id_to_bias_path=args.edge_path,
            template_db_path=args.template_db_path,
            ccd_preprocessed_path=args.ccd_db_path,
        ),
        tokenizer_config=TokenizerConfig(
            level=args.tokenizer_level,
            seed=args.tokenizer_seed,
            dynamic_config=dynamic_config,
        ),
    )
    return BioMolData(config)


def draw_indices(count: int, dataset_len: int, seed: int) -> list[int]:
    """Draw reproducible random dataset indices."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, dataset_len, size=count).tolist()


def normalize_names(names: list[object]) -> tuple[str, ...]:
    """Convert batch names to printable strings."""
    return tuple(str(name) for name in names)


def parse_sample_name(name: str) -> tuple[str, str, str, str]:
    """Parse a batch sample name into pdb, assembly, model, and alt ids."""
    parts = name.strip().rsplit("_", 3)
    if len(parts) != 4:
        msg = (
            "Expected --name in '<pdb>_<assembly>_<model>_<alt>' form, "
            "for example 3JB1_1_1_. or ['3JB1']_1_1_."
        )
        raise click.ClickException(msg)

    pdb_id = parts[0].strip().strip("[]'\"").lower()
    if not pdb_id:
        msg = "Parsed an empty pdb_id from --name."
        raise click.ClickException(msg)
    return pdb_id, parts[1], parts[2], parts[3]


def parse_crop_indices(value: str | None) -> np.ndarray | None:
    """Parse a comma-separated residue-index/range specification."""
    if value is None:
        return None

    indices: list[int] = []
    for raw_chunk in value.split(","):
        chunk = raw_chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            indices.append(int(chunk))
            continue

        range_parts = chunk.split(":")
        if len(range_parts) not in {2, 3} or range_parts[1] == "":
            msg = f"Invalid crop range '{chunk}'. Use start:stop or start:stop:step."
            raise click.ClickException(msg)
        start = int(range_parts[0]) if range_parts[0] else 0
        stop = int(range_parts[1])
        step = int(range_parts[2]) if len(range_parts) == 3 and range_parts[2] else 1
        indices.extend(range(start, stop, step))

    if not indices:
        msg = "--crop-indices did not contain any indices."
        raise click.ClickException(msg)
    return np.asarray(indices, dtype=np.int64)


def format_bias(bias: DataBias) -> str:
    """Format one metadata bias row for display."""
    chain_ids = [bias.chain_id1] + ([bias.chain_id2] if bias.chain_id2 else [])
    return (
        f"{bias.pdb_id}_{bias.assembly_id}_{bias.model_id}_{bias.alt_id} "
        f"chains={','.join(chain_ids)}"
    )


def bias_matches(
    bias: DataBias,
    *,
    pdb_id: str | None,
    assembly_id: str | None,
    model_id: str | None,
    alt_id: str | None,
) -> bool:
    """Return whether a metadata row matches optional id filters."""
    return (
        (pdb_id is None or bias.pdb_id == pdb_id.lower())
        and (assembly_id is None or bias.assembly_id == assembly_id)
        and (model_id is None or bias.model_id == model_id)
        and (alt_id is None or bias.alt_id == alt_id)
    )


def resolve_item_fields(
    args: ItemOptions,
) -> tuple[str | None, str | None, str | None, str | None]:
    """Resolve id fields from explicit options and optional sample name."""
    pdb_id = args.pdb_id.lower() if args.pdb_id else None
    assembly_id = args.assembly_id
    model_id = args.model_id
    alt_id = args.alt_id

    if args.name is None:
        return pdb_id, assembly_id, model_id, alt_id

    parsed_pdb, parsed_assembly, parsed_model, parsed_alt = parse_sample_name(args.name)
    explicit = {
        "pdb_id": (pdb_id, parsed_pdb),
        "assembly_id": (assembly_id, parsed_assembly),
        "model_id": (model_id, parsed_model),
        "alt_id": (alt_id, parsed_alt),
    }
    conflicts = [
        field
        for field, (given, parsed) in explicit.items()
        if given is not None and given != parsed
    ]
    if conflicts:
        msg = f"--name conflicts with explicit fields: {', '.join(conflicts)}"
        raise click.ClickException(msg)

    return (
        pdb_id or parsed_pdb,
        assembly_id or parsed_assembly,
        model_id or parsed_model,
        alt_id or parsed_alt,
    )


def fetch_direct_item(dataset: BioMolData, args: ItemOptions) -> FetchedItem:
    """Fetch one batch directly by index or id fields."""
    rng = np.random.default_rng(args.seed)
    crop_indices = parse_crop_indices(args.crop_indices)
    dataset.set_epoch(args.epoch)

    if args.index is not None:
        if args.index < 0 or args.index >= len(dataset):
            msg = (
                f"--index {args.index} is outside dataset range [0, {len(dataset) - 1}]."
            )
            raise click.ClickException(msg)
        bias = dataset.items[args.index]
        if args.allow_fallback and crop_indices is None:
            start = time.perf_counter()
            batch = dataset[args.index]
            return (
                batch,
                time.perf_counter() - start,
                args.index,
                format_bias(bias),
                None,
            )

        chain_ids = [bias.chain_id1] + ([bias.chain_id2] if bias.chain_id2 else [])
        start = time.perf_counter()
        batch = dataset.get_item_by_id(
            pdb_id=bias.pdb_id,
            assembly_id=bias.assembly_id,
            model_id=bias.model_id,
            alt_id=bias.alt_id,
            chain_ids=chain_ids,
            crop_indices=crop_indices,
            rng=rng,
        )
        return batch, time.perf_counter() - start, args.index, format_bias(bias), None

    pdb_id, assembly_id, model_id, alt_id = resolve_item_fields(args)
    if pdb_id is None:
        msg = "Provide either --index, --name, or --pdb-id."
        raise click.ClickException(msg)

    chain_ids = list(args.chain_ids)
    match_count: int | None = None
    selected_index: int | None = None
    source = (
        f"{pdb_id}_{assembly_id or '*'}_{model_id or '*'}_{alt_id or '*'} "
        f"chains={','.join(chain_ids) if chain_ids else '*'}"
    )

    if not chain_ids:
        matches = [
            (idx, bias)
            for idx, bias in enumerate(dataset.items)
            if bias_matches(
                bias,
                pdb_id=pdb_id,
                assembly_id=assembly_id,
                model_id=model_id,
                alt_id=alt_id,
            )
        ]
        match_count = len(matches)
        if not matches:
            msg = (
                "No dataset item matched the requested id fields. "
                "Pass --chain-id explicitly if the id is not listed in edge metadata."
            )
            raise click.ClickException(msg)
        if args.match < 0 or args.match >= len(matches):
            msg = (
                f"--match {args.match} is outside matched item range "
                f"[0, {len(matches) - 1}]."
            )
            raise click.ClickException(msg)
        selected_index, bias = matches[args.match]
        pdb_id = bias.pdb_id
        assembly_id = bias.assembly_id
        model_id = bias.model_id
        alt_id = bias.alt_id
        chain_ids = [bias.chain_id1] + ([bias.chain_id2] if bias.chain_id2 else [])
        source = format_bias(bias)

    start = time.perf_counter()
    batch = dataset.get_item_by_id(
        pdb_id=pdb_id,
        assembly_id=assembly_id,
        model_id=model_id,
        alt_id=alt_id,
        chain_ids=chain_ids,
        crop_indices=crop_indices,
        rng=rng,
    )
    return batch, time.perf_counter() - start, selected_index, source, match_count


def print_item_summary(
    batch: Batch,
    *,
    dataset: BioMolData,
    dataset_elapsed_sec: float,
    elapsed_sec: float,
    selected_index: int | None,
    source: str,
    match_count: int | None,
) -> None:
    """Print a compact summary for one fetched batch."""
    click.echo("=== Direct dataloader item ===")
    click.echo(f"Dataset size           : {len(dataset)}")
    if selected_index is not None:
        click.echo(f"Dataset index          : {selected_index}")
    if match_count is not None:
        click.echo(f"Matched items          : {match_count}")
    click.echo(f"Source                 : {source}")
    click.echo(f"Dataset build time     : {dataset_elapsed_sec:.4f}s")
    click.echo(f"Fetch time             : {elapsed_sec:.4f}s")
    click.echo(f"Total measured time    : {dataset_elapsed_sec + elapsed_sec:.4f}s")
    click.echo(f"Sample name            : {'|'.join(normalize_names(batch.name))}")
    click.echo(f"Batch shape            : {tuple(batch.shape)}")
    click.echo(f"Tokens / atoms         : {batch.token_length} / {batch.atom_length}")
    click.echo(f"MSA valid / depth      : {batch.msa_count} / {batch.msa_depth}")
    click.echo(
        f"Template valid / count : {batch.template_count} / {batch.template_number}",
    )
    click.echo(f"Device / dtype         : {batch.device} / {batch.dtype}")


def benchmark_direct_loading(
    dataset: BioMolData,
    *,
    measured_samples: int,
    warmup_samples: int,
    seed: int,
) -> list[DirectTiming]:
    """Measure direct ``dataset[index]`` construction latency."""
    timings: list[DirectTiming] = []
    indices = draw_indices(measured_samples + warmup_samples, len(dataset), seed)

    for draw_idx, dataset_index in enumerate(indices):
        start = time.perf_counter()
        batch = dataset[dataset_index]
        elapsed_sec = time.perf_counter() - start
        if draw_idx < warmup_samples:
            continue
        sample_name = str(batch.name[0]) if batch.name else f"dataset[{dataset_index}]"
        timings.append(
            DirectTiming(
                iteration=draw_idx - warmup_samples + 1,
                dataset_index=dataset_index,
                elapsed_sec=elapsed_sec,
                sample_name=sample_name,
            ),
        )

    return timings


def benchmark_dataloader_wait(
    dataset: BioMolData,
    args: BenchmarkOptions,
) -> list[LoaderTiming]:
    """Measure exposed wait time when iterating through a DataLoader."""
    timings: list[LoaderTiming] = []
    if args.num_workers > 0:
        dataloader = dataset.create_ddp_dataloader(
            rank=0,
            world_size=1,
            shuffle=args.shuffle,
            seed=args.seed,
            drop_last=args.drop_last,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            bucket_token_multiple=args.bucket_token_multiple,
            bucket_atom_multiple=args.bucket_atom_multiple,
        )
    else:
        dataloader = dataset.create_ddp_dataloader(
            rank=0,
            world_size=1,
            shuffle=args.shuffle,
            seed=args.seed,
            drop_last=args.drop_last,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            bucket_token_multiple=args.bucket_token_multiple,
            bucket_atom_multiple=args.bucket_atom_multiple,
        )
    typed_dataloader = cast("DataLoader[Batch]", dataloader)
    iterator: Iterator[Batch] = iter(typed_dataloader)

    def fetch_next_batch() -> Batch:
        nonlocal iterator
        try:
            return next(iterator)
        except StopIteration:
            iterator = iter(typed_dataloader)
            return next(iterator)

    for step_idx in range(args.steps + args.warmup_steps):
        start = time.perf_counter()
        batch = fetch_next_batch()
        wait_sec = time.perf_counter() - start

        if step_idx >= args.warmup_steps:
            batch_size = max(len(batch.name), 1)
            timings.append(
                LoaderTiming(
                    step=step_idx - args.warmup_steps + 1,
                    wait_sec=wait_sec,
                    batch_size=batch_size,
                    per_sample_wait_sec=wait_sec / batch_size,
                    wait_over_train_ratio=(
                        wait_sec / args.train_seconds if args.train_seconds > 0 else 0.0
                    ),
                    loader_share=(
                        wait_sec / (wait_sec + args.train_seconds)
                        if (wait_sec + args.train_seconds) > 0
                        else 0.0
                    ),
                    sample_names=normalize_names(batch.name),
                ),
            )

        if args.train_seconds > 0:
            time.sleep(args.train_seconds)

    return timings


def describe_distribution(values: list[float]) -> dict[str, float]:
    """Compute common summary statistics for measured durations."""
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(array.max()),
        "stdev": float(statistics.stdev(array)) if len(array) > 1 else 0.0,
    }


def write_csv(
    output_path: Path,
    direct_timings: list[DirectTiming],
    loader_timings: list[LoaderTiming],
    train_seconds: float,
) -> None:
    """Write benchmark measurements to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "mode",
                "iteration",
                "dataset_index",
                "step",
                "batch_size",
                "elapsed_sec",
                "per_sample_sec",
                "train_seconds",
                "wait_over_train_ratio",
                "loader_share",
                "sample_names",
            ],
        )
        writer.writeheader()

        for timing in direct_timings:
            writer.writerow(
                {
                    "mode": "direct_dataset",
                    "iteration": timing.iteration,
                    "dataset_index": timing.dataset_index,
                    "step": "",
                    "batch_size": 1,
                    "elapsed_sec": f"{timing.elapsed_sec:.6f}",
                    "per_sample_sec": f"{timing.elapsed_sec:.6f}",
                    "train_seconds": f"{train_seconds:.6f}",
                    "wait_over_train_ratio": (
                        f"{timing.elapsed_sec / train_seconds:.6f}"
                        if train_seconds > 0
                        else ""
                    ),
                    "loader_share": (
                        f"{timing.elapsed_sec / (timing.elapsed_sec + train_seconds):.6f}"
                        if (timing.elapsed_sec + train_seconds) > 0
                        else ""
                    ),
                    "sample_names": timing.sample_name,
                },
            )

        for timing in loader_timings:
            writer.writerow(
                {
                    "mode": "dataloader_wait",
                    "iteration": "",
                    "dataset_index": "",
                    "step": timing.step,
                    "batch_size": timing.batch_size,
                    "elapsed_sec": f"{timing.wait_sec:.6f}",
                    "per_sample_sec": f"{timing.per_sample_wait_sec:.6f}",
                    "train_seconds": f"{train_seconds:.6f}",
                    "wait_over_train_ratio": f"{timing.wait_over_train_ratio:.6f}",
                    "loader_share": f"{timing.loader_share:.6f}",
                    "sample_names": "|".join(timing.sample_names),
                },
            )


def build_plot(
    output_path: Path,
    direct_timings: list[DirectTiming],
    loader_timings: list[LoaderTiming],
    args: BenchmarkOptions,
) -> None:
    """Render benchmark plots for direct and DataLoader latency."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    direct_x = [timing.iteration for timing in direct_timings]
    direct_y = [timing.elapsed_sec for timing in direct_timings]

    loader_x = [timing.step for timing in loader_timings]
    loader_wait = [timing.wait_sec for timing in loader_timings]
    loader_per_sample = [timing.per_sample_wait_sec for timing in loader_timings]
    bottleneck_pct = [timing.wait_over_train_ratio * 100.0 for timing in loader_timings]
    loader_share_pct = [timing.loader_share * 100.0 for timing in loader_timings]

    per_sample_train = args.train_seconds / max(args.batch_size, 1)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 14),
        sharex=False,
        constrained_layout=True,
    )

    axes[0].plot(
        direct_x,
        direct_y,
        marker="o",
        linewidth=1.25,
        markersize=3,
        label="direct dataset[idx] latency",
    )
    axes[0].plot(
        loader_x,
        loader_per_sample,
        marker="o",
        linewidth=1.25,
        markersize=3,
        label="exposed dataloader wait / sample",
    )
    axes[0].axhline(
        per_sample_train,
        color="tab:red",
        linestyle="--",
        linewidth=1,
        label=f"assumed train time / sample = {per_sample_train:.3f}s",
    )
    axes[0].set_title("Per-sample latency")
    axes[0].set_xlabel("Measured sample")
    axes[0].set_ylabel("seconds")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].bar(
        loader_x,
        [args.train_seconds] * len(loader_x),
        label="assumed train compute",
        color="tab:blue",
        alpha=0.8,
    )
    axes[1].bar(
        loader_x,
        loader_wait,
        bottom=[args.train_seconds] * len(loader_x),
        label="dataloader stall",
        color="tab:orange",
        alpha=0.85,
    )
    axes[1].set_title("Step time decomposition")
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("seconds")
    axes[1].grid(alpha=0.3, axis="y")
    axes[1].legend()

    colors = [
        "tab:green" if value < 20 else "tab:orange" if value < 100 else "tab:red"
        for value in bottleneck_pct
    ]
    axes[2].bar(loader_x, bottleneck_pct, color=colors, alpha=0.9)
    axes[2].plot(
        loader_x,
        loader_share_pct,
        color="black",
        linewidth=1.2,
        marker="o",
        markersize=3,
        label="loader share of total step (%)",
    )
    axes[2].axhline(
        100.0,
        color="tab:red",
        linestyle="--",
        linewidth=1,
        label="wait = train compute",
    )
    axes[2].set_title("Bottleneck ratio")
    axes[2].set_xlabel("Training step")
    axes[2].set_ylabel("percent")
    axes[2].grid(alpha=0.3, axis="y")
    axes[2].legend()

    loader_summary = describe_distribution(loader_wait)
    direct_summary = describe_distribution(direct_y)
    fig.suptitle(
        (
            "MiniWorld dataloader benchmark\n"
            f"workers={args.num_workers}, batch_size={args.batch_size}, "
            f"train={args.train_seconds:.3f}s/step, "
            f"direct median={direct_summary['median']:.3f}s, "
            f"wait median={loader_summary['median']:.3f}s"
        ),
        fontsize=13,
    )
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


def print_summary(
    direct_timings: list[DirectTiming],
    loader_timings: list[LoaderTiming],
    args: BenchmarkOptions,
    figure_path: Path,
    csv_path: Path,
) -> None:
    """Print a text summary of benchmark measurements."""
    direct_stats = describe_distribution(
        [timing.elapsed_sec for timing in direct_timings],
    )
    wait_stats = describe_distribution([timing.wait_sec for timing in loader_timings])
    wait_per_sample_stats = describe_distribution(
        [timing.per_sample_wait_sec for timing in loader_timings],
    )
    bottleneck_stats = describe_distribution(
        [timing.wait_over_train_ratio for timing in loader_timings],
    )

    click.echo("=== Dataloader benchmark summary ===")
    click.echo(f"Dataset size           : {len(loader_timings)} measured steps")
    click.echo(f"Assumed train time     : {args.train_seconds:.4f} s/step")
    click.echo(
        "Direct dataset[idx]    : "
        f"mean={direct_stats['mean']:.4f}s, "
        f"median={direct_stats['median']:.4f}s, "
        f"p90={direct_stats['p90']:.4f}s, "
        f"max={direct_stats['max']:.4f}s",
    )
    click.echo(
        "Dataloader wait/step   : "
        f"mean={wait_stats['mean']:.4f}s, "
        f"median={wait_stats['median']:.4f}s, "
        f"p90={wait_stats['p90']:.4f}s, "
        f"max={wait_stats['max']:.4f}s",
    )
    click.echo(
        "Dataloader wait/sample : "
        f"mean={wait_per_sample_stats['mean']:.4f}s, "
        f"median={wait_per_sample_stats['median']:.4f}s, "
        f"p90={wait_per_sample_stats['p90']:.4f}s, "
        f"max={wait_per_sample_stats['max']:.4f}s",
    )
    click.echo(
        "Wait / train ratio     : "
        f"mean={bottleneck_stats['mean'] * 100:.1f}%, "
        f"median={bottleneck_stats['median'] * 100:.1f}%, "
        f"p90={bottleneck_stats['p90'] * 100:.1f}%, "
        f"max={bottleneck_stats['max'] * 100:.1f}%",
    )
    click.echo(f"Figure saved to        : {figure_path}")
    click.echo(f"CSV saved to           : {csv_path}")


def run_benchmark(args: BenchmarkOptions) -> None:
    """Run the benchmark command."""
    ensure_paths_exist(
        [
            args.cif_db_path,
            args.a3m_db_path,
            args.edge_path,
            args.template_db_path,
            args.ccd_db_path,
        ],
    )

    dataset = build_dataset(args)
    direct_timings = benchmark_direct_loading(
        dataset,
        measured_samples=args.direct_samples,
        warmup_samples=args.warmup_direct,
        seed=args.seed,
    )
    loader_timings = benchmark_dataloader_wait(dataset, args)

    figure_path = args.output_prefix.with_suffix(".png")
    csv_path = args.output_prefix.with_suffix(".csv")

    write_csv(csv_path, direct_timings, loader_timings, args.train_seconds)
    build_plot(figure_path, direct_timings, loader_timings, args)
    print_summary(direct_timings, loader_timings, args, figure_path, csv_path)


def run_item(args: ItemOptions) -> None:
    """Run the item inspection command."""
    ensure_paths_exist(
        [
            args.cif_db_path,
            args.a3m_db_path,
            args.edge_path,
            args.template_db_path,
            args.ccd_db_path,
        ],
    )

    start = time.perf_counter()
    dataset = build_dataset(args)
    dataset_elapsed_sec = time.perf_counter() - start
    batch, elapsed_sec, selected_index, source, match_count = fetch_direct_item(
        dataset,
        args,
    )
    print_item_summary(
        batch,
        dataset=dataset,
        dataset_elapsed_sec=dataset_elapsed_sec,
        elapsed_sec=elapsed_sec,
        selected_index=selected_index,
        source=source,
        match_count=match_count,
    )


@click.group(context_settings={"show_default": True})
def cli() -> None:
    """Inspect and benchmark the MiniWorld dataloader."""


@cli.command()
@click.option("--steps", type=int, default=32, help="Measured dataloader steps.")
@click.option(
    "--direct-samples",
    type=int,
    default=32,
    help="Measured dataset[idx] calls for raw sample-build latency.",
)
@click.option(
    "--warmup-steps",
    type=int,
    default=4,
    help="Warmup dataloader steps excluded from reporting.",
)
@click.option(
    "--warmup-direct",
    type=int,
    default=4,
    help="Warmup dataset[idx] calls excluded from reporting.",
)
@click.option(
    "--train-seconds",
    type=float,
    default=0.5,
    help="Assumed model compute time per training step in seconds.",
)
@click.option("--batch-size", type=int, default=1, help="Batch size.")
@click.option("--num-workers", type=int, default=4, help="DataLoader workers.")
@click.option(
    "--prefetch-factor",
    type=int,
    default=8,
    help="Prefetch factor used when num_workers > 0.",
)
@click.option("--seed", type=int, default=42, help="Random seed.")
@click.option("--shuffle/--no-shuffle", default=True, help="Shuffle the sampler.")
@click.option(
    "--drop-last/--no-drop-last",
    default=False,
    help="Drop the last incomplete batch.",
)
@click.option(
    "--bucket-token-multiple",
    type=int,
    default=128,
    help="Token padding multiple for bucketed collate.",
)
@click.option(
    "--bucket-atom-multiple",
    type=int,
    default=1024,
    help="Atom padding multiple for bucketed collate.",
)
@click.option(
    "--output-prefix",
    type=click.Path(path_type=Path),
    default=DEFAULT_OUTPUT_PREFIX,
    help="Output prefix for PNG and CSV files.",
)
@click.option("--dpi", type=int, default=180, help="PNG DPI.")
@dataset_options
def benchmark(**kwargs: object) -> None:
    """Benchmark direct dataset and DataLoader latency."""
    run_benchmark(BenchmarkOptions(**cast("dict[str, Any]", kwargs)))


@cli.command()
@click.option("--index", type=int, help="Dataset index to fetch directly.")
@click.option(
    "--name",
    type=str,
    help="Sample name such as 3JB1_1_1_. or ['3JB1']_1_1_.",
)
@click.option("--pdb-id", type=str, help="PDB id to fetch.")
@click.option("--assembly-id", type=str, help="Assembly id.")
@click.option("--model-id", type=str, help="Model id.")
@click.option("--alt-id", type=str, help="Alt id.")
@click.option(
    "--chain-id",
    "chain_ids",
    multiple=True,
    help="Chain id to crop around. Can be provided twice for an interface item.",
)
@click.option(
    "--match",
    type=int,
    default=0,
    help="When id fields match multiple metadata rows, choose this match index.",
)
@click.option("--seed", type=int, default=42, help="Crop RNG seed.")
@click.option("--epoch", type=int, default=0, help="Dataset/tokenizer epoch.")
@click.option(
    "--crop-indices",
    type=str,
    help="Comma-separated 0-based residue indices/ranges, e.g. 10,12,20:30.",
)
@click.option(
    "--allow-fallback/--no-allow-fallback",
    default=False,
    help="Use dataset[index] fallback behavior when --index crop fails.",
)
@dataset_options
def item(**kwargs: object) -> None:
    """Fetch one dataset item and print its feature shapes."""
    run_item(ItemOptions(**cast("dict[str, Any]", kwargs)))


if __name__ == "__main__":
    cli()
