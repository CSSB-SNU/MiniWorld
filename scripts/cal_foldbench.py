from pathlib import Path

import click


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--pred-dir",
    type=Path,
    required=True,
    help="Directory containing predicted structures in PDB format.",
)
@click.option(
    "--output-dir",
    type=Path,
    required=True,
    help="Output directory where the generated CSV file will be saved.",
)
@click.option(
    "--ranking-score-path",
    type=Path,
    default=None,
    help="Optional path to a file containing ranking scores for the predictions.",
)
@click.option(
    "--seed-path",
    type=Path,
    default=None,
    help="Optional path to a file containing seed information for the predictions.",
)
def build_foldbench_csv(
    pred_dir: Path,
    output_dir: Path,
    ranking_score_path: Path | None = None,
    seed_path: Path | None = None,
):
    header = ["pdb_id", "seed", "sample", "ranking_score", "prediction_path"]
    if ranking_score_path is not None:
        # yet to be implemented
        msg = f"Ranking score parsing is not implemented yet. Please provide a ranking score file at {ranking_score_path}."
        raise NotImplementedError(msg)
    if seed_path is not None:  # yet to be implemented
        msg = f"Seed parsing is not implemented yet. Please provide a seed file at {seed_path}."
        raise NotImplementedError(msg)
    ranking_score = "NA"  # default ranking score for all predictions
    seed = "NA"  # default seed value for all predictions
    sample = "1"  # default sample value for all predictions
    pred_cif_paths = list(pred_dir.glob("*.cif"))

    # convert to absolute path to avoid issues when foldbench read the csv file.
    pred_cif_paths = [p.resolve() for p in pred_cif_paths]
    out_csv = output_dir / "prediction_reference.csv"  # hard coded in foldbench.
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w") as f:
        f.write(",".join(header) + "\n")
        for pred_path in pred_cif_paths:
            pdb_id = pred_path.stem.split("_")[0]
            assembly_id = pred_path.stem.split("_")[1]
            pdb_id = f"{pdb_id}-assembly{assembly_id}"
            row = [pdb_id, seed, sample, ranking_score, str(pred_path)]
            f.write(",".join(row) + "\n")


if __name__ == "__main__":
    cli()
