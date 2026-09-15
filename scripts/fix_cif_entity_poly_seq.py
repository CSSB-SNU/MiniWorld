"""Add `_entity_poly` sequence strings to predicted CIFs so OpenStructure accepts them.

Our writer emits `_entity_poly_seq` (the residue list) but no sequence string. OST then
derives seqres from the `mon_id`s, which puts `X` at a modified residue, reads the
residue's own one-letter code as `?`, and rejects the whole file:

    RuntimeError: Sequence mismatch of residue "B.OGC2" with olc "?" ...
                  Character at that location in seqres: "X"

23 of 1,493 targets die that way, and a rejected file scores nothing at all. The
FoldBench ground truth carries both `pdbx_seq_one_letter_code` (bracket notation,
`(OGU)(OGC)...`) and `pdbx_seq_one_letter_code_can` (canonical, `XXX...`), which
removes the ambiguity.

This rewrites the header of an already-written CIF; the coordinates are untouched, so
no re-prediction is needed.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

# canonical one-letter for the standard monomers; anything else is X
ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "A": "A", "C": "C", "G": "G", "U": "U",
    "DA": "A", "DC": "C", "DG": "G", "DT": "T", "DU": "U",
}


def build_blocks(text: str) -> str | None:
    """Return the CIF with _entity_poly carrying both sequence strings, or None."""
    if "_entity_poly.pdbx_seq_one_letter_code" in text:
        return None  # already fixed

    m = re.search(
        r"#\nloop_\n_entity_poly\.entity_id\s*\n_entity_poly\.type\s*\n"
        r"_entity_poly\.pdbx_strand_id\s*\n(.*?)\n#\n",
        text, re.S,
    )
    if m is None:
        return None
    poly_rows = [ln.split(None, 2) for ln in m.group(1).splitlines() if ln.strip()]

    seqs: dict[str, list[str]] = {}
    ms = re.search(
        r"#\nloop_\n_entity_poly_seq\.entity_id\s*\n_entity_poly_seq\.num\s*\n"
        r"_entity_poly_seq\.mon_id\s*\n_entity_poly_seq\.hetero\s*\n(.*?)\n#\n",
        text, re.S,
    )
    if ms is None:
        return None
    for ln in ms.group(1).splitlines():
        f = ln.split()
        if len(f) >= 3:
            seqs.setdefault(f[0], []).append(f[2])

    # _chem_comp: without it OST cannot tell that a modified residue is peptide- or
    # nucleotide-linking, reads its one-letter code as "?", and rejects the file even
    # when the sequence strings above are present. The ground truth always carries it.
    link = {"polypeptide(L)": "L-peptide linking", "polypeptide(D)": "D-peptide linking",
            "polyribonucleotide": "RNA linking",
            "polydeoxyribonucleotide": "DNA linking",
            "polydeoxyribonucleotide/polyribonucleotide hybrid": "DNA linking"}
    comp: dict[str, str] = {}
    for eid, etype, _strand in poly_rows:
        t = link.get(etype.strip("'\""), "L-peptide linking")
        for mon in seqs.get(eid, []):
            comp.setdefault(mon, t)
    for ln in text.splitlines():
        f = ln.split()
        if f and f[0] in ("ATOM", "HETATM") and len(f) > 5 and f[0] == "HETATM":
            comp.setdefault(f[5], "non-polymer")
    chem = ["#", "loop_", "_chem_comp.id", "_chem_comp.type", "_chem_comp.mon_nstd_flag"]
    chem += [f"{k} '{v}' {'y' if k in ONE else 'n'}" for k, v in sorted(comp.items())]

    out = ["#", "loop_", "_entity_poly.entity_id", "_entity_poly.type",
           "_entity_poly.nstd_monomer", "_entity_poly.pdbx_seq_one_letter_code",
           "_entity_poly.pdbx_seq_one_letter_code_can", "_entity_poly.pdbx_strand_id"]
    for eid, etype, strand in poly_rows:
        mons = seqs.get(eid, [])
        nstd = "yes" if any(x not in ONE for x in mons) else "no"
        # bracket notation for non-standard monomers, as the ground truth writes it
        raw = "".join(ONE.get(x, f"({x})") for x in mons)
        can = "".join(ONE.get(x, "X") for x in mons)
        out += [f"{eid} {etype} {nstd}", f";{raw}", ";", f";{can}", ";", strand]
    return (text[:m.start()] + "\n".join(chem) + "\n"
            + "\n".join(out) + "\n#\n" + text[m.end():])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--glob", default="*/*_pred.cif")
    args = ap.parse_args()
    n = fixed = 0
    for p in args.paths:
        files = sorted(p.glob(args.glob)) if p.is_dir() else [p]
        for cif in files:
            n += 1
            new = build_blocks(cif.read_text())
            if new is not None:
                cif.write_text(new)
                fixed += 1
    print(f"{fixed}/{n} rewritten")


if __name__ == "__main__":
    main()
