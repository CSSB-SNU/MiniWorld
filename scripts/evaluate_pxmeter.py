"""Evaluate predicted structures against references using PXMeter.

Expects the output directory layout produced by the ``infer`` commands::

    <input-dir>/
        predicted/   # {name}_sample{i}.cif
        reference/   # {name}.cif

Usage (run inside the pxmeter conda environment)::

    conda run -n pxmeter python evaluate_pxmeter.py \
        --input-dir /path/to/inference/output \
        --output-dir /path/to/metrics/output \
        --num-workers 8
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import click

from pxmeter.configs.run_config import RUN_CONFIG
from pxmeter.eval import evaluate

# Trust the model CIF's entity_id assignment (it already mirrors the reference);
# disabling the auto-split avoids spurious "Unmapped model entities" warnings
# from chains whose residue subrange differs but biological entity is the same.
RUN_CONFIG.mapping.auto_fix_model_entities = False

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger("evaluate_pxmeter")


def _collect_pairs(input_dir: Path) -> list[dict]:
    """Match predicted CIF files with their reference CIF files."""
    pred_dir = input_dir / "predicted"
    ref_dir = input_dir / "reference"

    if not pred_dir.is_dir():
        msg = f"Predicted directory not found: {pred_dir}"
        raise FileNotFoundError(msg)
    if not ref_dir.is_dir():
        msg = f"Reference directory not found: {ref_dir}"
        raise FileNotFoundError(msg)

    pairs = []
    for pred_path in sorted(pred_dir.glob("*.cif")):
        stem = pred_path.stem
        parts = stem.rsplit("_sample", maxsplit=1)
        if len(parts) == 2:
            name, sample_idx_str = parts
            sample_idx = int(sample_idx_str)
        else:
            name = stem
            sample_idx = 0

        ref_path = ref_dir / f"{name}.cif"
        if not ref_path.exists():
            logger.warning("Reference not found for %s, skipping", pred_path.name)
            continue

        pairs.append({
            "name": name,
            "sample_idx": sample_idx,
            "pred_path": str(pred_path),
            "ref_path": str(ref_path),
        })

    logger.info("Found %d predicted/reference pairs", len(pairs))
    return pairs


def _run_pxmeter_single(
    pred_path: str,
    ref_path: str,
    detail_path: str,
) -> dict:
    """Run PXMeter on a single predicted/reference pair via Python API."""
    result = evaluate(
        ref_cif=ref_path,
        model_cif=pred_path,
    )
    detail = result.to_json_dict()
    Path(detail_path).write_text(json.dumps(detail, indent=2))

    metrics = {}
    if result.complex:
        for key, val in result.complex.items():
            if isinstance(val, (int, float)):
                metrics[f"complex_{key}"] = val

    for chain_id, chain_metrics in result.chain.items():
        for key, val in chain_metrics.items():
            if isinstance(val, (int, float)):
                metrics[f"chain_{chain_id}_{key}"] = val

    for iface_key, iface_metrics in result.interface.items():
        if isinstance(iface_key, tuple):
            iface_label = "_".join(str(k) for k in iface_key)
        else:
            iface_label = str(iface_key)
        for key, val in iface_metrics.items():
            if isinstance(val, (int, float)):
                metrics[f"interface_{iface_label}_{key}"] = val

    return metrics


def _aggregate_results(all_results: list[dict]) -> dict:
    """Aggregate per-sample results into summary statistics."""
    by_name: dict[str, list[dict]] = defaultdict(list)
    for r in all_results:
        by_name[r["name"]].append(r)

    metric_keys: set[str] = set()
    for r in all_results:
        metric_keys.update(
            k for k in r
            if k not in {"name", "sample_idx", "pred_path", "ref_path", "status", "error"}
        )

    per_structure = {}
    for name, samples in by_name.items():
        struct_summary: dict[str, float | None] = {}
        for key in sorted(metric_keys):
            values = [s[key] for s in samples if key in s and isinstance(s[key], (int, float))]
            if values:
                struct_summary[f"{key}_mean"] = sum(values) / len(values)
                # higher is better for lddt, lower is better for rmsd
                if "lddt" in key.lower() or "dockq" in key.lower():
                    struct_summary[f"{key}_best"] = max(values)
                else:
                    struct_summary[f"{key}_best"] = min(values)
            else:
                struct_summary[f"{key}_mean"] = None
                struct_summary[f"{key}_best"] = None
        struct_summary["num_samples"] = len(samples)
        per_structure[name] = struct_summary

    # Overall averages across structures (using best per structure)
    overall: dict[str, float | None] = {}
    for key in sorted(metric_keys):
        best_key = f"{key}_best"
        values = [
            s[best_key] for s in per_structure.values()
            if s.get(best_key) is not None
        ]
        if values:
            overall[f"{key}_avg_best"] = sum(values) / len(values)
        else:
            overall[f"{key}_avg_best"] = None

    return {
        "num_structures": len(per_structure),
        "num_total_samples": len(all_results),
        "overall": overall,
        "per_structure": per_structure,
    }


@click.command()
@click.option(
    "--input-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="inference output directory (contains predicted/ and reference/ subdirs)",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    required=True,
    help="directory to write metric results",
)
@click.option("--num-workers", type=int, default=1, help="number of parallel workers")
def main(
    input_dir: Path,
    output_dir: Path,
    num_workers: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_dir = output_dir / "detail"
    detail_dir.mkdir(parents=True, exist_ok=True)

    pairs = _collect_pairs(input_dir)
    if not pairs:
        logger.error("No predicted/reference pairs found in %s", input_dir)
        sys.exit(1)

    all_results = []

    def _process(pair: dict) -> dict:
        detail_path = str(detail_dir / f"{pair['name']}_sample{pair['sample_idx']}.json")
        try:
            metrics = _run_pxmeter_single(
                pred_path=pair["pred_path"],
                ref_path=pair["ref_path"],
                detail_path=detail_path,
            )
            return {**pair, **metrics, "status": "success"}
        except Exception as e:
            logger.warning(
                "Failed on %s sample %d: %s", pair["name"], pair["sample_idx"], e,
            )
            return {**pair, "status": "error", "error": str(e)}

    if num_workers <= 1:
        for pair in pairs:
            all_results.append(_process(pair))
    else:
        from joblib import Parallel, delayed
        all_results = Parallel(n_jobs=num_workers)(
            delayed(_process)(pair) for pair in pairs
        )
        all_results = [r for r in all_results if r is not None]

    # Save raw per-sample results
    raw_path = output_dir / "results_raw.json"
    raw_path.write_text(json.dumps(all_results, indent=2))
    logger.info("Saved raw results to %s", raw_path)

    # Aggregate and save summary
    successful = [r for r in all_results if r.get("status") == "success"]
    if successful:
        summary = _aggregate_results(successful)
        summary_path = output_dir / "results_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        logger.info("Saved summary to %s", summary_path)
        logger.info("Overall metrics:\n%s", json.dumps(summary["overall"], indent=2))
    else:
        logger.warning("No successful evaluations to aggregate")

    n_failed = len(all_results) - len(successful)
    logger.info(
        "Done: %d/%d successful, %d failed",
        len(successful), len(all_results), n_failed,
    )


if __name__ == "__main__":
    main()
