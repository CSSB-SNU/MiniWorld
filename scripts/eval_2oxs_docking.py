"""Evaluate 2OXS + BEN docking predictions against the GT structure.

For each run dir under ``--root`` (or a single dir with ``--run-dir``):
walks ``<run_dir>/structures/`` for ``*_pred.cif`` files and reports

  protein_ca_rmsd_aligned     Kabsch-aligned CA RMSD on chain A (= spec chain
                              ``"0"`` in the prediction) between prediction
                              and the GT 2OXS structure.
  ben_rmsd_post_protein_align RMSD of BEN heavy atoms after applying the
                              protein Kabsch transform to predicted BEN
                              (the canonical "docking" metric).
  ben_rmsd_self_aligned       Kabsch-aligned BEN RMSD against the GT BEN
                              (intra-ligand conformation lower bound).

Writes ``<run_dir>/metrics.json`` per run. Walks idempotently — already-
written metrics.json files are rewritten unless ``--skip-existing``.

Usage:
    python scripts/eval_2oxs_docking.py --run-dir outputs/<...>/<job>/
    python scripts/eval_2oxs_docking.py --root outputs/miniworld_test/2026-05-11/2oxs/
"""
from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
from Bio.PDB.MMCIFParser import MMCIFParser

from miniworld.data.io import load_cifmol


DEFAULT_CIF_DB = "/public_data02/BioMolDB_20260224/cif_attached_train.lmdb"


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def _gt_chain_residues(cifmol_chain) -> list[tuple[str, dict[str, np.ndarray]]]:
    """Return [(resname, {atom_name: xyz}), ...] in residue order."""
    out: list[tuple[str, dict[str, np.ndarray]]] = []
    residues = cifmol_chain.residues
    for ri in range(len(residues)):
        r = residues[ri]
        atom_ids = [str(a) for a in np.asarray(r.atoms.id.value)]
        xyz = np.asarray(r.atoms.xyz.value, dtype=np.float64)
        atoms = {}
        for i, name in enumerate(atom_ids):
            if np.isfinite(xyz[i]).all():
                atoms[name] = xyz[i]
        resname = str(np.asarray(r.chem_comp_id.value).item()) if hasattr(r, "chem_comp_id") else ""
        out.append((resname, atoms))
    return out


def _parse_pred_cif(cif_path: Path) -> dict[str, list[tuple[int, str, dict[str, np.ndarray]]]]:
    """Parse a single-model CIF, grouping by label_asym_id.

    Returns ``{chain_id: [(seq_id, resname, {atom_name: xyz}), ...]}`` with
    residues sorted by ``seq_id``. ``auth_chains=False`` makes ``chain.id``
    match ``label_asym_id`` (which is the numeric string ``"0"``, ``"1"``,
    ... in our predicted CIFs).
    """
    parser = MMCIFParser(QUIET=True, auth_chains=False)
    structure = parser.get_structure("pred", str(cif_path))
    model = next(iter(structure))
    out: dict[str, list[tuple[int, str, dict[str, np.ndarray]]]] = {}
    for chain in model:
        residues = []
        for residue in chain:
            atoms = {
                atom.get_name(): np.array(atom.get_coord(), dtype=np.float64)
                for atom in residue
            }
            residues.append((residue.id[1], residue.resname, atoms))
        residues.sort(key=lambda x: x[0])
        out[chain.id] = residues
    return out


# ---------------------------------------------------------------------------
# Kabsch + RMSD
# ---------------------------------------------------------------------------


def _kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (R, t) minimizing ||R @ P + t - Q||."""
    cP = P.mean(0)
    cQ = Q.mean(0)
    H = (P - cP).T @ (Q - cQ)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = cQ - R @ cP
    return R, t


def _apply(R: np.ndarray, t: np.ndarray, P: np.ndarray) -> np.ndarray:
    return P @ R.T + t


def _rmsd(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((A - B) ** 2, axis=-1))))


# ---------------------------------------------------------------------------
# Per-CIF metric computation
# ---------------------------------------------------------------------------


def _collect_protein_ca(
    gt_residues: list[tuple[str, dict[str, np.ndarray]]],
    pred_chain: list[tuple[int, str, dict[str, np.ndarray]]],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (gt_ca, pred_ca, n_matched) for residues where both have CA."""
    if len(gt_residues) != len(pred_chain):
        msg = (
            f"protein residue count mismatch: GT={len(gt_residues)} "
            f"vs pred={len(pred_chain)}"
        )
        raise ValueError(msg)
    gt_pts, pred_pts = [], []
    for (_, gt_atoms), (_, _, pred_atoms) in zip(gt_residues, pred_chain):
        if "CA" in gt_atoms and "CA" in pred_atoms:
            gt_pts.append(gt_atoms["CA"])
            pred_pts.append(pred_atoms["CA"])
    if not gt_pts:
        msg = "no matched CA atoms between GT and pred"
        raise ValueError(msg)
    return np.asarray(gt_pts), np.asarray(pred_pts), len(gt_pts)


def _collect_ben(
    gt_ben_atoms: dict[str, np.ndarray],
    pred_ben_atoms: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Match BEN atoms by name (heavy-atom only)."""
    common = sorted(set(gt_ben_atoms.keys()) & set(pred_ben_atoms.keys()))
    # Skip hydrogens (BioMolDB has none for BEN, but be safe).
    common = [n for n in common if not n.startswith("H")]
    if not common:
        msg = (
            f"no shared BEN atom names. gt={sorted(gt_ben_atoms)} "
            f"pred={sorted(pred_ben_atoms)}"
        )
        raise ValueError(msg)
    gt_pts = np.asarray([gt_ben_atoms[n] for n in common])
    pred_pts = np.asarray([pred_ben_atoms[n] for n in common])
    return gt_pts, pred_pts, common


def _evaluate_one(
    cif_path: Path,
    gt_protein: list[tuple[str, dict[str, np.ndarray]]],
    gt_ben_atoms: dict[str, np.ndarray],
    *,
    pred_protein_chain: str,
    pred_ligand_chain: str,
) -> dict:
    pred = _parse_pred_cif(cif_path)
    if pred_protein_chain not in pred:
        msg = (
            f"{cif_path.name}: predicted CIF has no chain {pred_protein_chain!r}. "
            f"Available: {sorted(pred)}"
        )
        raise ValueError(msg)
    if pred_ligand_chain not in pred:
        msg = (
            f"{cif_path.name}: predicted CIF has no chain {pred_ligand_chain!r}. "
            f"Available: {sorted(pred)}"
        )
        raise ValueError(msg)

    gt_ca, pred_ca, n_ca = _collect_protein_ca(gt_protein, pred[pred_protein_chain])
    R_prot, t_prot = _kabsch(pred_ca, gt_ca)
    aligned_pred_ca = _apply(R_prot, t_prot, pred_ca)
    protein_ca_rmsd = _rmsd(aligned_pred_ca, gt_ca)

    pred_ligand = pred[pred_ligand_chain]
    if len(pred_ligand) != 1:
        msg = (
            f"{cif_path.name}: expected exactly 1 BEN residue in chain "
            f"{pred_ligand_chain}, found {len(pred_ligand)}"
        )
        raise ValueError(msg)
    pred_ben_atoms = pred_ligand[0][2]
    gt_ben_pts, pred_ben_pts, ben_names = _collect_ben(gt_ben_atoms, pred_ben_atoms)

    # Apply the protein Kabsch transform to the predicted BEN.
    pred_ben_in_gt_frame = _apply(R_prot, t_prot, pred_ben_pts)
    ben_post_protein_align = _rmsd(pred_ben_in_gt_frame, gt_ben_pts)

    # Lower bound: align the BEN to itself.
    R_ben, t_ben = _kabsch(pred_ben_pts, gt_ben_pts)
    ben_self_aligned = _rmsd(_apply(R_ben, t_ben, pred_ben_pts), gt_ben_pts)

    return {
        "cif": cif_path.name,
        "protein_ca_rmsd_aligned": round(protein_ca_rmsd, 4),
        "ben_rmsd_post_protein_align": round(ben_post_protein_align, 4),
        "ben_rmsd_self_aligned": round(ben_self_aligned, 4),
        "n_ca_matched": n_ca,
        "n_ben_atoms_matched": len(ben_names),
        "ben_atom_names_matched": ben_names,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--run-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="A single run directory (containing structures/*_pred.cif).",
)
@click.option(
    "--root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Root containing many run subdirectories — recursively process every "
         "subdir that has structures/*_pred.cif inside.",
)
@click.option(
    "--cif-db",
    type=click.Path(path_type=Path),
    default=Path(DEFAULT_CIF_DB),
    show_default=True,
)
@click.option("--pdb-id", default="2oxs", show_default=True)
@click.option("--asm", default="1", show_default=True)
@click.option("--model-id", default="1", show_default=True)
@click.option("--alt-id", default=".", show_default=True)
@click.option(
    "--gt-protein-chain", default="A_1", show_default=True,
    help="BioMolDB chain id for the GT protein chain.",
)
@click.option(
    "--gt-ligand-chain", default="D_1", show_default=True,
    help="BioMolDB chain id for the GT ligand chain.",
)
@click.option(
    "--pred-protein-chain", default="0", show_default=True,
    help="label_asym_id of the protein chain in the predicted CIF.",
)
@click.option(
    "--pred-ligand-chain", default="1", show_default=True,
    help="label_asym_id of the ligand chain in the predicted CIF.",
)
@click.option(
    "--skip-existing/--rewrite",
    default=False, show_default=True,
    help="If set, skip run dirs that already have metrics.json.",
)
def main(  # noqa: PLR0913
    run_dir: Path | None,
    root: Path | None,
    cif_db: Path,
    pdb_id: str,
    asm: str,
    model_id: str,
    alt_id: str,
    gt_protein_chain: str,
    gt_ligand_chain: str,
    pred_protein_chain: str,
    pred_ligand_chain: str,
    skip_existing: bool,
) -> None:
    if (run_dir is None) == (root is None):
        raise click.UsageError("provide exactly one of --run-dir / --root")

    # Load GT once.
    cm = load_cifmol(cif_db, pdb_id, asm, model_id, alt_id)
    gt_prot_cm = cm.chains[cm.chains.chain_id == gt_protein_chain].extract()
    gt_lig_cm = cm.chains[cm.chains.chain_id == gt_ligand_chain].extract()
    gt_protein = _gt_chain_residues(gt_prot_cm)
    gt_ben_residues = _gt_chain_residues(gt_lig_cm)
    if len(gt_ben_residues) != 1:
        msg = f"GT ligand chain {gt_ligand_chain!r} has {len(gt_ben_residues)} residues, expected 1"
        raise SystemExit(msg)
    gt_ben_atoms = gt_ben_residues[0][1]

    run_dirs: list[Path]
    if run_dir is not None:
        run_dirs = [run_dir]
    else:
        run_dirs = sorted(
            p.parent for p in root.glob("**/structures") if p.is_dir()
        )
    if not run_dirs:
        click.echo("no run directories found")
        return

    for rd in run_dirs:
        metrics_path = rd / "metrics.json"
        if skip_existing and metrics_path.exists():
            click.echo(f"skip {rd} (metrics.json exists)")
            continue
        pred_cifs = sorted((rd / "structures").glob("*_pred.cif"))
        # Exclude GT-rendered cifs the validation path writes (just in case).
        pred_cifs = [c for c in pred_cifs if not c.name.endswith("_gt.cif")]
        if not pred_cifs:
            click.echo(f"skip {rd} (no *_pred.cif)")
            continue
        per_cif: list[dict] = []
        for cif in pred_cifs:
            try:
                m = _evaluate_one(
                    cif,
                    gt_protein,
                    gt_ben_atoms,
                    pred_protein_chain=pred_protein_chain,
                    pred_ligand_chain=pred_ligand_chain,
                )
            except Exception as e:  # noqa: BLE001
                m = {"cif": cif.name, "error": f"{type(e).__name__}: {e}"}
            per_cif.append(m)

        # Aggregate (best by ben_rmsd_post_protein_align).
        ok = [m for m in per_cif if "error" not in m]
        summary = {}
        if ok:
            keys = (
                "protein_ca_rmsd_aligned",
                "ben_rmsd_post_protein_align",
                "ben_rmsd_self_aligned",
            )
            summary = {f"min_{k}": min(m[k] for m in ok) for k in keys}
            summary["best_pose_cif"] = min(
                ok, key=lambda m: m["ben_rmsd_post_protein_align"],
            )["cif"]
        out = {
            "gt": {
                "cif_id": f"{pdb_id}_{asm}_{model_id}_{alt_id}",
                "protein_chain": gt_protein_chain,
                "ligand_chain": gt_ligand_chain,
            },
            "summary": summary,
            "per_cif": per_cif,
        }
        metrics_path.write_text(json.dumps(out, indent=2) + "\n")
        click.echo(f"{rd}/metrics.json  ({len(per_cif)} cif)")
        for m in per_cif:
            if "error" in m:
                click.echo(f"  {m['cif']}: {m['error']}")
            else:
                click.echo(
                    f"  {m['cif']}: protein_ca={m['protein_ca_rmsd_aligned']:.3f}  "
                    f"ben_post={m['ben_rmsd_post_protein_align']:.3f}  "
                    f"ben_self={m['ben_rmsd_self_aligned']:.3f}",
                )


if __name__ == "__main__":
    main()
