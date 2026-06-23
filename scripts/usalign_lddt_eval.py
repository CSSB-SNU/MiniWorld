"""Score sampled structures with US-align (chain-mapping RMSD + TM-score) and lDDT.

For each target in a run dir's structures/ folder:
  - run US-align (-mm 1 -ter 0) on pred vs gt  -> chain-mapped RMSD, TM-score, aligned length
  - compute lDDT (cal_atom_lddt) from the index-matched coords (superposition-invariant,
    so US-align superposition does not change it; reported alongside).

Compares several run dirs side by side over the targets common to all of them.

Usage:
    pixi run python scripts/usalign_lddt_eval.py DIR1 DIR2 ... \
        [--usalign tools/USalign/USalign] [--label l1 l2 ...]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from miniworld.loss import metrics
from miniworld.utils.structure.align import weighted_align

RMSD_RE = re.compile(r"Aligned length=\s*(\d+),\s*RMSD=\s*([\d.]+)")
TM_REF_RE = re.compile(r"TM-score=\s*([\d.]+).*normalized by length of Structure_2")

# _atom_site field indices (whitespace-split): atom_id=3, comp=5, asym=6, seq=8, x=10
_ATOM, _COMP, _ASYM, _SEQ, _X = 3, 5, 6, 8, 10
STD_AA = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
          "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"}

# Common crystallization aids: precipitants/anions, cryoprotectants/polyols, PEGs,
# buffers, solvents, reducing agents, and monatomic ions. Dropped from scoring as
# they are crystallization artifacts, not biologically meaningful ligands.
CRYST_ADDITIVES = {
    # anions / precipitants
    "SO4", "PO4", "2HP", "1PO", "NO3", "NO2", "SCN", "NCO", "CO3", "BCT", "OXL",
    "ACT", "ACY", "FMT", "EDT", "CIT", "FLC", "TLA", "MLA", "MLI", "MLT", "TAR",
    # cryoprotectants / polyols / PEGs
    "GOL", "EDO", "PGO", "PDO", "MPD", "MRD", "BU3", "BU1", "DEG", "TRD",
    "PEG", "PGE", "PG4", "PG0", "PG5", "PG6", "P6G", "1PE", "2PE", "PE3", "PE4",
    "PE5", "PE8", "7PE", "12P", "15P", "211", "DIO", "DOX",
    # solvents / reducing agents / buffers
    "DMS", "DMF", "MOH", "EOH", "IPA", "IOH", "ACN", "ACE", "BME", "DTT", "DTU",
    "DTV", "TCE", "EPE", "MES", "TRS", "BTB", "BIS", "IMD", "CAC", "PIN", "POP",
    "B3P", "NHE", "TAU", "GAI", "TBU", "144", "MG8",
    # monatomic ions (and a few common ionic species)
    "NA", "K", "LI", "RB", "CS", "MG", "CA", "SR", "BA", "ZN", "MN", "FE", "FE2",
    "NI", "CO", "3CO", "CU", "CU1", "CD", "HG", "AU", "AG", "PT", "PB", "TL",
    "CL", "BR", "IOD", "FLO", "NH4", "OH", "OXY", "PER",
}


def excluded_comps(gt_path: Path) -> set[str]:
    """Crystallization aids present in the structure: any comp in CRYST_ADDITIVES,
    plus any unlisted monatomic species (single-atom residue) as a catch-all ion."""
    inst_atoms: dict[tuple, int] = defaultdict(int)  # (comp,asym,seq) -> n_atoms
    for ln in gt_path.read_text().splitlines():
        if ln.startswith(("ATOM ", "HETATM")):
            f = ln.split()
            inst_atoms[(f[_COMP], f[_ASYM], f[_SEQ])] += 1
    comp_max: dict[str, int] = defaultdict(int)
    for (comp, _a, _s), n in inst_atoms.items():
        comp_max[comp] = max(comp_max[comp], n)
    drop = set()
    for comp, mx in comp_max.items():
        if comp in CRYST_ADDITIVES or (mx == 1 and comp not in STD_AA):
            drop.add(comp)
    return drop


def parse_cif_coords(path: Path, drop: set[str] | None = None) -> np.ndarray:
    """Parse Cartn_x/y/z from an _atom_site loop, in file order, skipping drop comps."""
    drop = drop or set()
    xyz = []
    for ln in path.read_text().splitlines():
        if ln.startswith(("ATOM ", "HETATM")):
            f = ln.split()
            if f[_COMP] in drop:
                continue
            xyz.append((float(f[_X]), float(f[_X + 1]), float(f[_X + 2])))
    return np.asarray(xyz, dtype=np.float32)


def write_filtered_cif(src: Path, dst: Path, drop: set[str]) -> None:
    """Copy a CIF, dropping ATOM/HETATM lines whose comp is in drop (header kept)."""
    out = []
    for ln in src.read_text().splitlines():
        if ln.startswith(("ATOM ", "HETATM")) and ln.split()[_COMP] in drop:
            continue
        out.append(ln)
    dst.write_text("\n".join(out) + "\n")


def run_usalign(usalign: Path, pred: Path, gt: Path) -> dict | None:
    """US-align pred onto gt (multimer chain mapping); protein RMSD/TM/aligned-length."""
    out = subprocess.run(
        [str(usalign), str(pred), str(gt), "-mm", "1", "-ter", "0"],
        capture_output=True, text=True, check=False,
    ).stdout
    m_rmsd, m_tm = RMSD_RE.search(out), TM_REF_RE.search(out)
    if not m_rmsd or not m_tm:
        return None
    return {"aligned": int(m_rmsd.group(1)), "rmsd": float(m_rmsd.group(2)),
            "tm": float(m_tm.group(1))}


_ELEM = 2  # type_symbol


def parse_atoms(path: Path):
    """Return (xyz [N,3], comp [N], atom_id [N], elem [N]) in file order."""
    xyz, comp, aid, elem = [], [], [], []
    for ln in path.read_text().splitlines():
        if ln.startswith(("ATOM ", "HETATM")):
            f = ln.split()
            xyz.append((float(f[_X]), float(f[_X + 1]), float(f[_X + 2])))
            comp.append(f[_COMP]); aid.append(f[_ATOM]); elem.append(f[_ELEM])
    return (np.asarray(xyz, np.float32), np.asarray(comp),
            np.asarray(aid), np.asarray(elem))


def ligand_matched_sqdists(aligned: np.ndarray, gt: np.ndarray, comp: np.ndarray,
                           elem: np.ndarray, lig_mask: np.ndarray) -> np.ndarray:
    """Per-(comp,element) optimal one-to-one assignment between pred & gt ligand atoms.

    pred/gt share atom order, so each (comp,element) group spans the same indices;
    we re-match within the group by structural proximity (Hungarian, min total sq
    distance) to fix permuted ligand copies / atom ordering after protein alignment.
    """
    from collections import defaultdict

    from scipy.optimize import linear_sum_assignment
    groups: dict = defaultdict(list)
    for i in np.where(lig_mask)[0]:
        groups[(comp[i], elem[i])].append(i)
    out = []
    for idx in groups.values():
        g = np.asarray(idx)
        cost = ((aligned[g][:, None, :] - gt[g][None, :, :]) ** 2).sum(-1)
        r, c = linear_sum_assignment(cost)
        out.extend(cost[r, c].tolist())
    return np.asarray(out)


def score_target(usalign: Path, pred: Path, gt: Path, drop: set[str]) -> dict | None:
    # Superposition + chain mapping from the protein (SO4/ions dropped), as before.
    if drop:  # filter both structures (identical composition -> stays index-matched)
        with tempfile.TemporaryDirectory() as td:
            fp, fg = Path(td) / "p.cif", Path(td) / "g.cif"
            write_filtered_cif(pred, fp, drop)
            write_filtered_cif(gt, fg, drop)
            ua = run_usalign(usalign, fp, fg)
    else:
        ua = run_usalign(usalign, pred, gt)
    if ua is None:
        return None
    # lDDT on the protein (SO4/ions dropped), superposition-invariant.
    p, g = parse_cif_coords(pred, drop), parse_cif_coords(gt, drop)
    lddt = float("nan")
    if p.shape == g.shape and len(p) > 0:
        lddt = metrics.cal_atom_lddt(p, g, torch.ones(len(g), dtype=torch.bool))
    # RMSD including ligands. Superpose by Kabsch on the protein backbone (CA of
    # standard residues, reproducing US-align's core RMSD). Protein atoms use the
    # identity correspondence (each is unique); ligand/hetero atoms are re-matched to
    # the structurally-closest partner (optimal assignment) since their index order /
    # copy order need not correspond after alignment. SO4 + abundant ions are dropped.
    pf, cf, af, ef = parse_atoms(pred); gf, _, _, _ = parse_atoms(gt)
    rmsd_all = rmsd_lig = float("nan")
    if pf.shape == gf.shape and len(pf) > 0:
        keep = ~np.isin(cf, list(drop))
        pf, gf, cf, af, ef = pf[keep], gf[keep], cf[keep], af[keep], ef[keep]
        prot = np.isin(cf, list(STD_AA))
        fit = prot & (af == "CA")
        if fit.sum() < 3:  # nucleic / odd cases: fall back to all atoms for the fit
            fit = np.ones(len(pf), bool)
        wt = torch.from_numpy(fit.astype(np.float32))[None]
        aligned = weighted_align(torch.from_numpy(pf)[None], torch.from_numpy(gf)[None],
                                 weight=wt)[0].numpy()
        d2_prot = ((aligned[prot] - gf[prot]) ** 2).sum(-1)
        d2_lig = ligand_matched_sqdists(aligned, gf, cf, ef, ~prot)
        all_d2 = np.concatenate([d2_prot, d2_lig])
        rmsd_all = float(np.sqrt(all_d2.mean())) if len(all_d2) else float("nan")
        rmsd_lig = float(np.sqrt(d2_lig.mean())) if len(d2_lig) else float("nan")
    return {**ua, "lddt": lddt, "rmsd_all": rmsd_all, "rmsd_lig": rmsd_lig,
            "natom_full": len(gf)}


def eval_dir(run_dir: Path, usalign: Path, exclude: bool) -> dict[str, dict]:
    sd = run_dir / "structures"
    cifs = sorted(sd.iterdir())
    res = {}
    for gt in [p for p in cifs if p.name.endswith("_gt.cif")]:
        name = gt.name[: -len("_gt.cif")]
        drop = excluded_comps(gt) if exclude else set()
        preds = [p for p in cifs if p.name.startswith(f"{name}_pred_")
                 and p.name.endswith(".cif")]
        best = None
        for pred in preds:  # best-of-N by TM-score
            s = score_target(usalign, pred, gt, drop)
            if s and (best is None or s["tm"] > best["tm"]):
                best = s
        if best:
            res[name] = {**best, "dropped": sorted(drop)}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--usalign", type=Path, default=Path("tools/USalign/USalign"))
    ap.add_argument("--label", nargs="+", default=None)
    ap.add_argument("--keep-all", action="store_true",
                    help="Do NOT drop SO4 / abundant monatomic ions (default drops them).")
    args = ap.parse_args()
    labels = args.label or [d.name for d in args.run_dirs]
    exclude = not args.keep_all

    per = {lab: eval_dir(d, args.usalign, exclude) for lab, d in zip(labels, args.run_dirs)}
    common = set.intersection(*[set(v) for v in per.values()]) if per else set()
    common = sorted(common)
    print(f"\ncommon targets to all {len(per)} runs: {len(common)}")
    if exclude:
        ref = next(iter(per.values()))
        print("dropped comps (SO4 + monatomic ions >=4 copies) per target:")
        for n in common:
            d = ref[n].get("dropped", [])
            if d:
                print(f"  {n}: {', '.join(d)}")

    titles = {"rmsd": "RMSD (protein, US-align aligned residues)",
              "rmsd_all": "RMSD ALL-ATOM incl. ligands (Kabsch fit on protein-CA)",
              "rmsd_lig": "RMSD ligand/hetero-only (protein-CA superposed)",
              "tm": "TM-score (protein)", "lddt": "lDDT (SO4/ions dropped)"}
    for metric in ("rmsd", "rmsd_all", "rmsd_lig", "tm", "lddt"):
        print(f"\n=== {titles[metric]}  (best-of-N by TM) ===")
        hdr = f"{'target':<20} " + " ".join(f"{l[:14]:>14}" for l in labels)
        print(hdr)
        for n in common:
            print(f"{n[:20]:<20} " + " ".join(f"{per[l][n][metric]:>14.3f}" for l in labels))
        print("-" * len(hdr))
        print(f"{'MEAN(common)':<20} " + " ".join(
            f"{np.nanmean([per[l][n][metric] for n in common]):>14.3f}" for l in labels))


if __name__ == "__main__":
    main()
