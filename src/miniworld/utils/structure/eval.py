import json
import os
import subprocess
from pathlib import Path
from typing import Literal

from joblib import Parallel, delayed
from tqdm import tqdm

OST_COMPARE_LIGAND_STRUCTURE = r"""
singularity run ost.img compare-ligand-structures \
-m {model_file} \
-r {reference_file} \
--fault-tolerant \
--lddt-pli --rmsd \
-o {output_path}
"""

OST_COMPARE_STRUCTURE = r"""
    singularity run ost.img compare-structures \
        -m {model_file} \
        -r {reference_file} \
        -o {output_path} \
        --fault-tolerant \
        --min-pep-length 4 \
        --min-nuc-length 4 \
        --lddt --rigid-scores --tm-score --dockq \
"""


def get_structure_value(
    output_path: Path,
    native_chain_id_1: str | None,
    native_chain_id_2: str | None,
) -> tuple[
    float | None,
    float | None,
    float | None,
    int | None,
    float | None,
    float | None,
    float | None,
    float | None,
]:
    """Get the structure evaluation values from the OST output JSON file."""
    dockq = None
    irmsd = None
    lrmsd = None
    len_dockq = None
    lddt = None
    tm_score = None
    gdt_ts = None
    rmsd = None
    with output_path.open("r") as f:
        data = json.load(f)

    for i, interface in enumerate(data["dockq_interfaces"]):
        if native_chain_id_1 in interface and native_chain_id_2 in interface:
            dockq = data["dockq"][i]
            irmsd = data["irmsd"][i]
            lrmsd = data["lrmsd"][i]
    len_dockq = len(data["dockq"])
    lddt = data["lddt"]
    tm_score = data["tm_score"]
    gdt_ts = data["oligo_gdtts"]
    rmsd = data["rmsd"]
    return dockq, irmsd, lrmsd, len_dockq, lddt, tm_score, gdt_ts, rmsd


def get_ligand_value(
    output_path: Path,
    native_chain_id_2: str | None,
) -> tuple[float | None, float | None, float | None]:
    """Get the ligand evaluation values from the OST output JSON file."""
    rmsd = None
    lddt_lp = None
    lddt_pli = None
    with output_path.open("r") as f:
        data = json.load(f)

    for item in data["rmsd"]["assigned_scores"]:
        reference_ligand_name = item["reference_ligand"]
        #  native_chain_id_2 is ligand by default
        if native_chain_id_2 == reference_ligand_name.split(".")[0]:
            rmsd = item["score"]
            lddt_lp = item["lddt_lp"]

    for item in data["lddt_pli"]["assigned_scores"]:
        reference_ligand_name = item["reference_ligand"]
        #  native_chain_id_2 is ligand by default
        if native_chain_id_2 == reference_ligand_name.split(".")[0]:
            lddt_pli = item["score"]

    return rmsd, lddt_lp, lddt_pli


def ost_evaluation(
    ground_truth_path: Path,
    prediction_path: Path,
    interface_chain_id_1: str,
    interface_chain_id_2: str,
    output_path: Path,
    mode: Literal["ligand", "structure"],
    executable: str = "/bin/bash",
) -> dict[str, str | float | None]:
    """Evaluate the predicted structure using OST and return the evaluation results."""
    if output_path.exists():
        return {"status": "exist"}
    if not prediction_path.exists():
        return {"status": "prediction_path is None"}

    result: dict[str, str | float | None] = {}
    if mode == "ligand":
        subprocess.run(
            OST_COMPARE_LIGAND_STRUCTURE.format(
                model_file=str(prediction_path),
                reference_file=str(ground_truth_path),
                output_path=str(output_path),
            ),
            shell=True,
            check=False,
            executable=executable,
            capture_output=True,
        )

        rmsd, lddt_lp, lddt_pli = get_ligand_value(
            output_path,
            interface_chain_id_2,
        )
        result = {
            "rmsd": rmsd,
            "lddt-lp": lddt_lp,
            "lddt-pli": lddt_pli,
        }
    elif mode == "structure":
        subprocess.run(
            OST_COMPARE_STRUCTURE.format(
                model_file=str(prediction_path),
                reference_file=str(ground_truth_path),
                output_path=str(output_path),
            ),
            shell=True,
            check=False,
            executable=executable,
            capture_output=True,
        )

        dockq_ost, irmsd, lrmsd, len_dockq, lddt, tm_score, gdt_ts, rmsd = (
            get_structure_value(
                output_path,
                interface_chain_id_1,
                interface_chain_id_2,
            )
        )
        result = {
            "dockq_score": dockq_ost,
            "irmsd": irmsd,
            "lrmsd": lrmsd,
            "len_dockq": len_dockq,
            "lddt": lddt,
            "tm_score": tm_score,
            "gdt_ts": gdt_ts,
            "rmsd": rmsd,
        }
    else:
        msg = f"Invalid mode: {mode}. Mode should be either 'ligand' or 'structure'."
        raise ValueError(msg)

    result["status"] = "success"
    return result


def eval_by_ost(
    target_df,
    target_type,
    evaluation_dir,
    ground_truth_dir,
    max_workers=64,
):
    detail_path = os.path.join(evaluation_dir, "detail")
    if not os.path.exists(detail_path):
        os.makedirs(detail_path)

    if target_type in [
        "interface_protein_protein",
        "interface_antibody_antigen",
        "interface_protein_peptide",
        "interface_protein_dna",
        "interface_protein_rna",
        "monomer_dna",
        "monomer_rna",
        "monomer_protein",
    ]:
        mode = "structure"
    elif target_type == "interface_protein_ligand":
        mode = "ligand"
    else:
        msg = f"Invalid target type: {target_type}. Target type should be one of 'interface_protein_protein', 'interface_antibody_antigen', 'interface_protein_peptide', 'interface_protein_dna', 'interface_protein_rna', 'monomer_dna', 'monomer_rna', 'monomer_protein' or 'interface_protein_ligand'."
        raise ValueError(msg)

    tasks = []

    def _run(task):
        row, ground_truth_dir, detail_path, mode = task

        ground_truth_path = Path(ground_truth_dir) / row["ground_truth_file"]
        prediction_path = Path(row["prediction_file"])
        interface_chain_id_1 = row["interface_chain_id_1"]
        interface_chain_id_2 = row["interface_chain_id_2"]
        output_path = Path(detail_path) / f"{row['id']}.json"

        return ost_evaluation(
            ground_truth_path=ground_truth_path,
            prediction_path=prediction_path,
            interface_chain_id_1=interface_chain_id_1,
            interface_chain_id_2=interface_chain_id_2,
            output_path=output_path,
            mode=mode,
        )

    results = Parallel(n_jobs=max_workers)(
        delayed(_run)(task) for task in tqdm(tasks, total=len(tasks))
    )
    results = [r for r in results if r is not None]
