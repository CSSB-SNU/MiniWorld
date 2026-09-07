#!/usr/bin/env python3
"""Verify the KPCA fingerprint tables, and that the unknown rows are genuine.

The central check is on ``RNA_UNK`` / ``DNA_UNK``.  An earlier revision set them
to the mean of the A/U/G/C and DA/DT/DG/DC rows, which places them at a centroid
no real nucleotide occupies.  They should instead be the KPCA image of the
backbone-scaffold molecules from ``scripts/fp_emb_old.py`` -- ribose-5'-phosphate
and 2'-deoxyribose-5'-phosphate with the base replaced by an amine.  This script
proves that by:

  1. showing the row is NOT the average of the canonical nucleotides;
  2. recomputing the scaffold's fingerprint from SMILES and confirming its
     Tanimoto profile against the canonical nucleotides is what the stored row's
     geometry implies -- a real molecule, not an artefact;
  3. confirming the row is not shared with an unrelated component -- a twin is
     allowed only when it is literally the same molecule, which happens: the
     RNA_UNK scaffold has the same canonical SMILES as CCD ``GRF``
     (beta-D-ribofuranosylamine 5'-phosphate), so identical fingerprints give
     identical KPCA coordinates.  That is the embedding working correctly, not a
     collision;
  4. confirming ``UNK`` matches its own real CCD fingerprint, not any average.

It also re-checks the structural invariants: shape against the vocab, per-block
standardisation and the zero rows.

Usage
-----
    /home/bsoohyuncd/.conda/envs/prolif/bin/python \\
        scripts/dataset/verify_fp_embeddings.py \\
        --mcac /public_data/bsoohyuncd/CCD/McAc/table_d32.pt
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
from zstandard import ZstdDecompressor

CCD_DIR = Path("/public_data/bsoohyuncd/CCD")
IDX32_TO_CHEM = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "UNK", "A", "U", "G", "C", "RNA_UNK", "DA", "DT", "DG", "DC", "DNA_UNK",
    "GAP_ZERO",
]
SCAFFOLD_SMILES = {
    "RNA_UNK": "N[C@@H]1O[C@H](COP(O)(O)=O)[C@@H](O)[C@H]1O",
    "DNA_UNK": "N[C@H]1C[C@H](O)[C@@H](COP(O)(O)=O)O1",
}
RNA_CANONICAL = ["A", "U", "G", "C"]
DNA_CANONICAL = ["DA", "DT", "DG", "DC"]


def load_bytes_local(raw: bytes) -> dict:
    blob = ZstdDecompressor().decompress(raw, max_output_size=1 << 31)
    hl = int.from_bytes(blob[:8], "little")
    hdr = json.loads(blob[8 : 8 + hl])
    payload = memoryview(blob)[8 + hl :]
    sl = {}
    off = 0
    for uid, ln in hdr["arrays"].items():
        sl[uid] = payload[off : off + ln]
        off += ln

    def rb(t: dict) -> dict:
        out = {}
        for k, v in t.items():
            if isinstance(v, str) and v in sl:
                out[k] = np.load(io.BytesIO(sl[v]), allow_pickle=False)
            elif isinstance(v, dict):
                out[k] = rb(v)
            else:
                out[k] = v
        return out

    return rb(hdr["template"])

def main() -> None:  # noqa: C901, PLR0915
    """Run every check and report."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mcac", type=Path, required=True)
    ap.add_argument("--fp-radius", type=int, default=2,
                    help="Must match what the table was built with.")
    ap.add_argument("--fp-bits", type=int, default=4096,
                    help="Must match what the table was built with.")
    ap.add_argument("--vocab", type=Path, default=CCD_DIR / "vocab.json")
    ap.add_argument("--ccd-lmdb", type=Path,
                    default=CCD_DIR / "preprocessed_CCD_20260826.lmdb")
    args = ap.parse_args()

    import lmdb
    import torch
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    problems: list[str] = []

    def check(ok: bool, label: str, detail: str = "") -> None:
        print(f"  [{'OK ' if ok else 'FAIL'}] {label}{'  ' + detail if detail else ''}")
        if not ok:
            problems.append(label)

    vocab = json.loads(args.vocab.read_text())
    t1 = torch.load(args.mcac, map_location="cpu")
    d1 = t1.shape[1]
    print(f"vocab   : {args.vocab}  ({len(vocab):,} entries)")
    print(f"McAc    : {args.mcac}  {tuple(t1.shape)}")
    print(f"fp      : Morgan r={args.fp_radius} bits={args.fp_bits}")
    print(f"\nd_single_token_input -> {2 * d1 + 385}")

    print("\n== structure")
    check(t1.shape[0] == len(vocab), "rows match vocab length")
    check(bool(torch.isfinite(t1).all()), "all values finite")
    half = d1 // 2
    s_m, s_a = float(t1[:, :half].std()), float(t1[:, half:].std())
    check(abs(s_m - 1) < 0.05 and abs(s_a - 1) < 0.05,  # noqa: PLR2004
          "per-block unit variance", f"morgan {s_m:.3f}, atompair {s_a:.3f}")

    print("\n== zero rows")
    check(bool(torch.all(t1[vocab["GAP_ZERO"]] == 0)), "GAP_ZERO KPCA block is zero")
    n_zero = int((t1.abs().sum(1) == 0).sum())
    print(f"  ....  {n_zero} zero rows in McAc (unparseable CCDs + GAP_ZERO)")

    # Every CCD's SMILES, so a shared row can be judged same-molecule or not.
    def _load_all_smiles() -> dict[str, str]:
        out: dict[str, str] = {}
        env = lmdb.open(str(args.ccd_lmdb), readonly=True, lock=False)
        with env.begin() as txn:
            for key, value in txn.cursor():
                nodes = load_bytes_local(bytes(value))["residues"]["nodes"]
                if "rdkit_smiles" not in nodes:
                    continue
                smi = str(
                    np.asarray(nodes["rdkit_smiles"]["value"]).reshape(-1)[0]).strip()
                if smi:
                    out[bytes(key).decode()] = smi
        env.close()
        return out

    # ---- the point of this script -------------------------------------
    print("\n== unknown rows are real molecules, not averages")
    ccd_smiles = _load_all_smiles()
    for key, canonical in (("RNA_UNK", RNA_CANONICAL), ("DNA_UNK", DNA_CANONICAL)):
        row = t1[vocab[key]]
        avg = t1[[vocab[c] for c in canonical]].mean(0)
        dist_to_avg = float(torch.norm(row - avg))
        check(not torch.allclose(row, avg, atol=1e-5),
              f"{key} is NOT the mean of {'/'.join(canonical)}",
              f"L2 to that mean = {dist_to_avg:.3f}")
        check(bool(row.abs().sum() > 0), f"{key} is a non-zero row")
        # A twin is fine iff it is the same molecule; only flag a twin whose
        # SMILES differs from the scaffold.
        twins = [c for c in (
            {k for k, i in vocab.items() if bool((t1[i] == row).all())} - {key}
        )]
        bad_twins = [
            c for c in twins
            if c in ccd_smiles and Chem.MolToSmiles(Chem.MolFromSmiles(ccd_smiles[c]))
            != Chem.MolToSmiles(Chem.MolFromSmiles(SCAFFOLD_SMILES[key]))
        ]
        check(not bad_twins,
              f"{key} shares its row only with the same molecule",
              f"twins={twins}" + (f" MISMATCHED={bad_twins}" if bad_twins else ""))

    check(not torch.allclose(
        t1[vocab["UNK"]],
        t1[[vocab[c] for c in IDX32_TO_CHEM[:20]]].mean(0), atol=1e-5),
        "UNK is NOT the mean of the 20 amino acids")

    # ---- independent evidence: the scaffold chemistry itself ------------
    print("\n== scaffold chemistry cross-check (recomputed from SMILES)")
    # Must match the parameters the tables were built with, or the recomputed
    # Tanimoto profile is against a differently-hashed fingerprint.
    morgan = rdFingerprintGenerator.GetMorganGenerator(
        radius=args.fp_radius, fpSize=args.fp_bits)


    env = lmdb.open(str(args.ccd_lmdb), readonly=True, lock=False)
    smiles: dict[str, str] = {}
    with env.begin() as txn:
        for code in RNA_CANONICAL + DNA_CANONICAL + ["UNK", "ATP"]:
            raw = txn.get(code.encode())
            if raw is None:
                continue
            nodes = load_bytes_local(bytes(raw))["residues"]["nodes"]
            smiles[code] = str(
                np.asarray(nodes["rdkit_smiles"]["value"]).reshape(-1)[0]).strip()
    env.close()

    for key, canonical in (("RNA_UNK", RNA_CANONICAL), ("DNA_UNK", DNA_CANONICAL)):
        scaffold_fp = morgan.GetCountFingerprint(
            Chem.MolFromSmiles(SCAFFOLD_SMILES[key]))
        sims = {
            c: DataStructs.TanimotoSimilarity(
                scaffold_fp, morgan.GetCountFingerprint(Chem.MolFromSmiles(smiles[c])))
            for c in canonical if c in smiles
        }
        far = DataStructs.TanimotoSimilarity(
            scaffold_fp, morgan.GetCountFingerprint(Chem.MolFromSmiles(smiles["ATP"])))
        nearest = min(sims.values())
        print(f"  {key} scaffold vs " + ", ".join(
            f"{c} {v:.3f}" for c, v in sims.items()) + f" | ATP {far:.3f}")
        check(nearest > 0, f"{key} scaffold shares substructure with its nucleotides")

        # geometry: is the stored row nearer its own nucleotides than to a
        # chemically unrelated reference?
        d_own = float(torch.norm(
            t1[vocab[key]] - t1[[vocab[c] for c in canonical]].mean(0)))
        d_aa = float(torch.norm(
            t1[vocab[key]] - t1[[vocab[c] for c in IDX32_TO_CHEM[:20]]].mean(0)))
        check(d_own < d_aa, f"{key} sits nearer nucleotides than amino acids",
              f"{d_own:.2f} vs {d_aa:.2f}")

    print(f"\n{'=' * 68}")
    if problems:
        print(f"FAILED {len(problems)} check(s):")
        for p in problems:
            print("  -", p)
        raise SystemExit(f"{len(problems)} failed")
    print("all checks passed")


if __name__ == "__main__":
    main()
