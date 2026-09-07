#!/usr/bin/env python3
"""Build the protein-ligand subset in one pass.

Three rules, applied in this order:

1. *Scope.*  The set is the PDB entries named by a protein-ligand row
   (``cL`` against ``cP``/``cQ``/``cA``) in the source edge table.  Nothing else
   is read.

2. *Nucleic acid excludes the whole entry.*  If any cif in an entry contains a
   DNA/RNA/hybrid chain, or a nucleotide or nucleotide-analogue modelled as a
   ligand, the ENTIRE entry is dropped -- from the LMDB and from the edge table
   alike.  This is exclusion, not stripping: a structure that has nucleic acid
   in it is not a protein-ligand example, and deleting the nucleic acid would
   leave a protein posed around an absent partner.  Those entries are held out
   for the nucleotide validation set instead.

3. *Crystallisation aids are stripped, the entry stays.*  The sixteen codes of
   the AF3 supplement's aid list (``AID_CODES``) are deleted residue-by-residue
   from the coordinates, and any protein-aid row is dropped from the edge table.
   The entry and everything else in it survive -- an aid is a bystander, not a
   reason to discard a structure.

On top of those, a protein-ligand row whose ligand is *covalently bonded to that
row's own protein chain* is dropped from the edge table, but its CHAIN stays in
the coordinates: deleting it would leave a dangling bond on the residue it is
attached to.

Everything else is left exactly as it is -- ions, sugars, lipids, fragments and
ligand chains that no surviving row names all remain in the coordinates.

Why nucleic acid is detected two ways
-------------------------------------
By chain TYPE (``cluster_id`` beginning ``cD``/``cR``/``cN``), which is exact for
polymers; and by CCD code within non-polymer chains, which is what catches a
nucleotide modelled as a ligand.  The code test is confined to ``cL`` chains
because the nucleotide codes include the bare monomers ``A``/``U``/``G``/``C``/
``DA``/``DT``/``DG``/``DC`` -- the residues of every real nucleic-acid polymer,
already covered exactly by the type test.

Nucleotide tiers
----------------
``code``      the code says so: ``D?[ATGCU](TP|DP|MP)`` -- ATP, GTP, dCDP, UMP --
              plus the bare monomers.  No chemistry, no false positives.
``scaffold``  and anything that IS essentially a nucleotide: it matches a
              nucleoside substructure (furanose carrying a purine or pyrimidine)
              and that scaffold, grown over the fused base rings, attached
              phosphates and terminal N/O, covers at least ``--coverage`` of its
              heavy atoms.  Catches modified nucleotides whose codes look like
              nothing in particular.
``similar``   and anything within ``--similarity`` Tanimoto of the above.  This
              is the tier that reaches the nucleotide-DERIVED cofactors -- NAD,
              FAD, SAM, coenzyme A, ADP-ribose, the UDP-sugars -- which carry a
              large second moiety and so fall below the coverage bar by design.
              The fingerprint is binary Morgan, radius 2, 4096 bits
              (``--fp-radius`` / ``--fp-bits``).  4096 rather than 2048 halves
              the hash-collision rate, which matters here because a collision
              can only ever inflate a similarity -- two molecules sharing a
              folded bit they do not really share -- and every inflated
              similarity above the threshold discards an entry.

Covalency, per row
------------------
Tested against the row's own protein chain, not against "any polymer", so it
answers the question the spec asks.  Two signals, unioned, because neither is
complete alone: a ``struct_conn`` edge of type ``covale`` joining the two
chains, or any atom pair closer than ``--covalent-angstrom``.  The coordinates
hold heavy atoms only, so 1.8 A separates a bond from a contact cleanly.  Both
are derived once per sub-entry from a single KD-tree, not once per row.

Cluster ids
-----------
The output TSV keeps the ligand cluster ids exactly as ``train_edge_node.tsv``
writes them.  Relabelling ligands to Tanimoto cluster ids is a separate step
(``apply_ligand_clusters_to_edges.py``) that must run LAST, on the finished
table -- never mid-pipeline, which is what produced the two-namespace bug in the
previous build.

Environment
-----------
Needs biomol + structcooker (for the structures) and RDKit (for the nucleotide
chemistry).  No single installed environment has all three: the StructCooker
pixi env lacks RDKit, and the prolif conda env lacks biomol.  biomol is pure
Python, so ``scripts/dataset/_shim`` symlinks it next to structcooker's source
and prolif imports both against its own numpy.  Encoding under prolib and
decoding under pixi round-trips byte-for-byte.

Reproducing both splits
-----------------------
One script, one flag apart.  ``--split train`` takes no holdout arguments, so
its code path and its output are unchanged by anything the validation split
needs.  valid2 depends on train's finished table, which fixes the order::

    PY="PYTHONPATH=<this dir>/_shim:/home/bsoohyuncd/software/StructCooker/src \\
        /home/bsoohyuncd/.conda/envs/prolif/bin/python"

    # 1. train
    $PY scripts/dataset/build_prt_lig_set.py --split train
    # 2. cluster the ligands (binary Morgan r=2/4096, greedy sphere excl. > 0.5)
    $PY scripts/dataset/cluster_ligands_tanimoto.py --radius 2 --bits 4096
    # 3. relabel train's ligand ids to cluster ids -- LAST for that split
    $PY scripts/dataset/apply_ligand_clusters_to_edges.py \\
        --in-tsv  metadata/train_edge_node_onlyPrtLig.tsv \\
        --out-tsv metadata/train_edge_node_onlyPrtLig.tsv
    # 4. valid2 -- holds out train's ligand clusters (defaults come from SPLITS)
    $PY scripts/dataset/build_prt_lig_set.py --split valid2
    # 5. relabel valid2 the same way
    $PY scripts/dataset/apply_ligand_clusters_to_edges.py \\
        --in-tsv  metadata/valid2_edge_node_onlyPrtLig.tsv \\
        --out-tsv metadata/valid2_edge_node_onlyPrtLig.tsv

The relabel rewrites only fields beginning ``cL``; the protein cluster columns
are left exactly as the source wrote them.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import lmdb
import numpy as np
from biomol.core.utils import load_bytes, to_bytes
from scipy.spatial import cKDTree
from structcooker.mols import CIFMolAttached

BASEDIR = Path("/public_data/bsoohyuncd/BioMolDB_20260224")
CCD_LMDB = Path("/public_data/bsoohyuncd/CCD/preprocessed_CCD_20260826.lmdb")

# The AF3 supplement's crystallisation-aid list.
AID_CODES = np.array([
    "SO4", "GOL", "EDO", "PO4", "ACT", "PEG", "DMS", "TRS",
    "PGE", "PG4", "FMT", "EPE", "MPD", "MES", "CD", "IOD",
])

NUCLEOTIDE_CODE_PATTERN = r"^D?[ATGCU](TP|DP|MP)$"
NUCLEOTIDE_MONOMERS = frozenset({
    "A", "U", "G", "C", "DA", "DT", "DG", "DC", "I", "DI", "N", "DN",
})
NUCLEOSIDE_SMARTS = (
    "[$([nX3]1cnc2c1ncnc2),$([nX3]1ccc(=O)[nX3]c1=O),$([nX3]1ccc(N)nc1=O)]"
    "[CX4H]1O[CX4H]([CH2X4,CH3X4,CX4])[CX4H]([OX2H,OX2])[CX4H,CX4H2]1"
)
PHOSPHATE_SMARTS = "[PX4](=O)([OX2,OX1-])([OX2,OX1-])"
TIERS = ("code", "scaffold", "similar")

SPLITS: dict[str, dict[str, Path]] = {
    "train": {
        "in_lmdb": BASEDIR / "cif_attached_train.lmdb",
        "in_tsv": BASEDIR / "metadata/train_edge_node.tsv",
        "out_lmdb": BASEDIR / "cif_attached_train_onlyPrtLig.lmdb",
        "out_tsv": BASEDIR / "metadata/train_edge_node_onlyPrtLig.tsv",
    },
    # valid2 additionally holds out chemistry: any ligand cluster that appears
    # in the built training table is excluded.  These two keys are set ONLY
    # here, so `--split train` cannot pick them up and the training build stays
    # byte-for-byte reproducible from this same script.  The dependency does fix
    # an order, though: train must be built and the ligands clustered before
    # valid2 can be.
    "valid2": {
        "in_lmdb": BASEDIR / "cif_attached_valid_2.lmdb",
        "in_tsv": BASEDIR / "metadata/valid2_edge_node.tsv",
        "out_lmdb": BASEDIR / "cif_attached_valid_2_onlyPrtLig.lmdb",
        "out_tsv": BASEDIR / "metadata/valid2_edge_node_onlyPrtLig.tsv",
        "exclude_ligands_from": BASEDIR / "metadata/train_edge_node_onlyPrtLig.tsv",
        "ligand_map": BASEDIR / "metadata/ligand_cluster_map_tanimoto50.tsv",
    },
}

TSV_HEADER = (
    "cluster1\tcluster2\tpdb_id\tassembly_id\tmodel_id\talt_id\tchain_id1\tchain_id2\n"
)

PROTEIN_TYPES = frozenset("PQA")
NUCLEIC_TYPES = frozenset("DRN")  # cD DNA, cR RNA, cN DNA/RNA hybrid
LIGAND_TYPE = "L"
COVALENT_ANGSTROM = 1.8
ZSTD_LEVEL = 6
COMMIT_EVERY = 2000
REPORT_EVERY = 5000


# ---------------------------------------------------------------------------
# Nucleotide-like CCD codes
# ---------------------------------------------------------------------------


def _decode_ccd(byte_data: bytes) -> dict:
    """Decode one CCD blob without importing biomol's reader."""
    from zstandard import ZstdDecompressor

    raw = ZstdDecompressor().decompress(byte_data, max_output_size=1 << 31)
    header_len = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8 : 8 + header_len])
    payload = memoryview(raw)[8 + header_len :]
    flat: dict[str, memoryview] = {}
    offset = 0
    for key, length in header["arrays"].items():
        flat[key] = payload[offset : offset + length]
        offset += length

    def rebuild(template: dict) -> dict:
        out = {}
        for key, value in template.items():
            if isinstance(value, str) and value in flat:
                out[key] = np.load(io.BytesIO(flat[value]), allow_pickle=False)
            elif isinstance(value, dict):
                out[key] = rebuild(value)
            else:
                out[key] = value
        return out

    return rebuild(header["template"])


def nucleotide_coverage(mol, nucleoside, phosphate) -> float:
    """Heavy-atom fraction belonging to a nucleoside scaffold plus phosphates."""
    hits = mol.GetSubstructMatches(nucleoside)
    if not hits:
        return 0.0
    covered: set[int] = set()
    for hit in hits:
        covered.update(hit)
    for ring in mol.GetRingInfo().AtomRings():
        if covered.intersection(ring):
            covered.update(ring)
    for hit in mol.GetSubstructMatches(phosphate):
        covered.update(hit)
    for idx in list(covered):
        for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
            if nbr.GetDegree() == 1 and nbr.GetAtomicNum() in (7, 8):
                covered.add(nbr.GetIdx())
    return len(covered) / max(mol.GetNumHeavyAtoms(), 1)


def nucleotide_codes(
    ccd_lmdb: Path, tier: str, coverage: float, similarity: float,
    radius: int = 2, bits: int = 4096,
) -> tuple[np.ndarray, dict[str, str]]:
    """Every CCD code that counts as DNA/RNA-like at the requested tier."""
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    nucleoside = Chem.MolFromSmarts(NUCLEOSIDE_SMARTS)
    phosphate = Chem.MolFromSmarts(PHOSPHATE_SMARTS)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
    code_re = re.compile(NUCLEOTIDE_CODE_PATTERN)

    started = time.time()
    env = lmdb.open(str(ccd_lmdb), readonly=True, lock=False)
    mols: dict[str, object] = {}
    assigned: dict[str, str] = {}
    with env.begin() as txn:
        for key, value in txn.cursor():
            ccd = bytes(key).decode()
            if code_re.match(ccd) or ccd in NUCLEOTIDE_MONOMERS:
                assigned[ccd] = "code"
            nodes = _decode_ccd(bytes(value))["residues"]["nodes"]
            if "rdkit_smiles" not in nodes:
                continue
            smi = str(np.asarray(nodes["rdkit_smiles"]["value"]).reshape(-1)[0]).strip()
            mol = Chem.MolFromSmiles(smi) if smi else None
            if mol is not None and mol.GetNumHeavyAtoms() >= 2:  # noqa: PLR2004
                mols[ccd] = mol
    env.close()
    print(f"  ccd store scanned : {len(mols):,} parseable "
          f"({time.time() - started:.0f}s)")

    if tier in ("scaffold", "similar"):
        for ccd, mol in mols.items():
            if ccd not in assigned and nucleotide_coverage(
                    mol, nucleoside, phosphate) >= coverage:
                assigned[ccd] = "scaffold"

    if tier == "similar":
        reference = [c for c in assigned if c in mols]
        ref_fps = [gen.GetFingerprint(mols[c]) for c in reference]
        if ref_fps:
            for ccd, mol in mols.items():
                if ccd in assigned:
                    continue
                sims = DataStructs.BulkTanimotoSimilarity(
                    gen.GetFingerprint(mol), ref_fps)
                if max(sims) > similarity:
                    assigned[ccd] = "similar"

    counts = Counter(assigned.values())
    for name in TIERS:
        if name in counts:
            print(f"  tier {name:9s}: {counts[name]:>6,} codes")
    print(f"  nucleotide codes  : {len(assigned):,} total "
          f"({time.time() - started:.0f}s)")
    return np.array(sorted(assigned)), assigned


# ---------------------------------------------------------------------------
# Structures
# ---------------------------------------------------------------------------


def load_ligand_map(path: Path) -> dict[str, str]:
    """``old_cluster -> new_cluster`` from the Tanimoto clustering."""
    mapping: dict[str, str] = {}
    with path.open() as fh:
        next(fh)
        for line in fh:
            old, _, new = line.rstrip("\n").partition("\t")
            if new:
                mapping[old] = new
    return mapping


def seen_ligand_clusters(path: Path, mapping: dict[str, str]) -> set[str]:
    """Ligand CLUSTER ids appearing in an already-built edge table.

    Every id is pushed through ``mapping``, so this works whether that table is
    still in per-ligand original ids or has already been relabelled: a cluster
    representative maps to itself, making the operation idempotent.
    """
    seen: set[str] = set()
    with path.open() as fh:
        next(fh)
        for line in fh:
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 8:  # noqa: PLR2004
                continue
            lig = fields[0] if fields[0][1] == LIGAND_TYPE else fields[1]
            seen.add(mapping.get(lig, lig))
    return seen


def read_rows(
    path: Path,
    exclude: set[str] | None = None,
    mapping: dict[str, str] | None = None,
) -> tuple[dict[str, list[tuple[str, ...]]], Counter]:
    """Protein-ligand rows grouped by lowercase pdb_id.

    ``exclude`` is optional and unset for the training split, which therefore
    takes exactly the code path it always did.  When it IS given, a row is
    skipped if its ligand's CLUSTER id is in that set -- which is what makes a
    validation split hold out chemistry as well as proteins.  Filtering here
    rather than afterwards means an entry whose every row is excluded is never
    opened, and the surviving entry set follows from the surviving rows.
    """
    rows: dict[str, list[tuple[str, ...]]] = defaultdict(list)
    tally: Counter[str] = Counter()
    mapping = mapping or {}
    with path.open() as fh:
        if next(fh).rstrip("\n") != TSV_HEADER.rstrip("\n"):
            msg = f"unexpected header in {path}"
            raise ValueError(msg)
        for line in fh:
            fields = tuple(line.rstrip("\n").split("\t"))
            tally["rows_total"] += 1
            if len(fields) != 8 or fields[1] == "None":  # noqa: PLR2004
                continue
            pair = {fields[0][1], fields[1][1]}
            if LIGAND_TYPE not in pair or not pair & PROTEIN_TYPES:
                continue
            tally["rows_protein_ligand"] += 1
            if exclude:
                lig = fields[0] if fields[0][1] == LIGAND_TYPE else fields[1]
                if mapping.get(lig, lig) in exclude:
                    tally["rows_dropped_ligand_seen_in_train"] += 1
                    continue
            rows[fields[2].lower()].append(fields)
    tally["pdb_candidate"] = len(rows)
    return rows, tally


def covalent_partners(
    cif_dict: dict,
    xyz: np.ndarray,
    atom_chain: np.ndarray,
    ligand_chains: set[int],
    angstrom: float,
) -> dict[int, set[int]]:
    """For each ligand chain, the chain indices it is bonded to."""
    partners: dict[int, set[int]] = {c: set() for c in ligand_chains}

    conn = cif_dict.get("atoms", {}).get("edges", {}).get("struct_conn")
    if conn is not None:
        value = np.asarray(conn["value"])
        src = np.asarray(conn["src_indices"])
        dst = np.asarray(conn["dst_indices"])
        n_atoms = len(atom_chain)
        for row in range(len(src)):
            if "covale" not in str(value[row]):
                continue
            a, b = int(src[row]), int(dst[row])
            if not (0 <= a < n_atoms and 0 <= b < n_atoms):
                continue
            ca, cb = int(atom_chain[a]), int(atom_chain[b])
            if ca == cb:
                continue
            if ca in partners:
                partners[ca].add(cb)
            if cb in partners:
                partners[cb].add(ca)

    finite = np.isfinite(xyz).all(1)
    if not finite.any():
        return partners
    idx_finite = np.where(finite)[0]
    tree = cKDTree(xyz[idx_finite])
    for chain in ligand_chains:
        sel = np.where((atom_chain == chain) & finite)[0]
        if len(sel) == 0:
            continue
        for neighbours in tree.query_ball_point(xyz[sel], angstrom):
            if neighbours:
                partners[chain].update(atom_chain[idx_finite[neighbours]].tolist())
        partners[chain].discard(chain)
    return partners


def build_entry(
    raw: bytes,
    rows: list[tuple[str, ...]],
    counts: Counter,
    angstrom: float,
    nuc_codes: np.ndarray,
) -> tuple[bytes | None, list[tuple[str, ...]]]:
    """Filter one PDB entry; return (blob or None, surviving rows)."""
    entry = load_bytes(raw)
    local: Counter[str] = Counter()

    by_key: dict[str, list[tuple[str, ...]]] = defaultdict(list)
    for row in rows:
        by_key[f"{row[3]}_{row[4]}_{row[5]}"].append(row)

    # ---- rule 2: nucleic acid anywhere in the entry excludes all of it ----
    # Checked across every sub-entry in the blob, not only the ones a row names,
    # and across every residue, not only ligand chains -- "appears in the cif"
    # is meant literally.  A nucleic-acid polymer is caught exactly by chain
    # type; a nucleotide or analogue modelled as a ligand, by CCD code.
    mols: dict[str, CIFMolAttached] = {}
    for asm_key, item in entry.items():
        mol = CIFMolAttached.from_dict(item["cifmol_attached_dict"])
        mols[asm_key] = mol
        cluster_id = np.asarray(mol.chains.cluster_id.value).astype(str)
        res_ccd = np.asarray(mol.residues.chem_comp_id.value).astype(str)
        if any(c[1:2] in NUCLEIC_TYPES for c in cluster_id):
            local["entry_dropped_nucleic_chain"] += 1
        elif np.isin(res_ccd, nuc_codes).any():
            local["entry_dropped_nucleotide_ligand"] += 1
        else:
            continue
        local["rows_dropped_with_nucleic_entry"] += len(rows)
        counts.update(local)
        return None, []

    # ---- rule 3 and covalency, per sub-entry ------------------------------
    kept: dict[str, dict] = {}
    surviving: list[tuple[str, ...]] = []
    modified = False

    for asm_key, item in entry.items():
        want = by_key.get(asm_key)
        if not want:
            modified = True
            local["subentry_no_row"] += 1
            continue

        mol = mols[asm_key]
        chain_id = np.asarray(mol.chains.chain_id.value).astype(str)
        cluster_id = np.asarray(mol.chains.cluster_id.value).astype(str)
        chain_index = {name: i for i, name in enumerate(chain_id)}
        n_chains = len(chain_id)

        # Aid residues, confined to non-polymer chains.
        chain_is_ligand = np.array(
            [c[1:2] == LIGAND_TYPE for c in cluster_id], dtype=bool)
        res_ccd = np.asarray(mol.residues.chem_comp_id.value).astype(str)
        res_to_chain = np.asarray(mol.index_table.res_to_chain)
        is_aid_res = chain_is_ligand[res_to_chain] & np.isin(res_ccd, AID_CODES)

        # A chain made entirely of aids ceases to exist, so its rows go too.
        res_per_chain = np.bincount(res_to_chain, minlength=n_chains)
        aid_per_chain = np.bincount(
            res_to_chain, weights=is_aid_res.astype(float), minlength=n_chains)
        dead_chain = (res_per_chain > 0) & (aid_per_chain == res_per_chain)

        present = set(chain_id)
        rows_here = []
        for row in want:
            if row[6] not in present or row[7] not in present:
                local["row_dropped_chain_absent"] += 1
                continue
            if dead_chain[chain_index[row[6]]] or dead_chain[chain_index[row[7]]]:
                local["row_dropped_ligand_is_aid"] += 1
                continue
            rows_here.append(row)

        # Covalency, tested against each row's own protein chain.
        if rows_here:
            atom_chain = res_to_chain[np.asarray(mol.index_table.atom_to_res)]
            xyz = np.asarray(mol.atoms.xyz.value)
            lig_chains = {
                chain_index[row[f]] for row in rows_here for f in (6, 7)
                if chain_is_ligand[chain_index[row[f]]]
            }
            partners = covalent_partners(
                item["cifmol_attached_dict"], xyz, atom_chain, lig_chains, angstrom)
            free_rows = []
            for row in rows_here:
                i6, i7 = chain_index[row[6]], chain_index[row[7]]
                lig, prot = (i6, i7) if chain_is_ligand[i6] else (i7, i6)
                if prot in partners.get(lig, ()):
                    local["row_dropped_covalent"] += 1
                    continue
                free_rows.append(row)
            rows_here = free_rows

        if not rows_here:
            modified = True
            local["subentry_no_row"] += 1
            continue

        if is_aid_res.any():
            modified = True
            local["aid_residues_removed"] += int(is_aid_res.sum())
            local["subentry_stripped"] += 1
            mol = mol.residues[~is_aid_res].extract()
        else:
            local["subentry_clean"] += 1

        new_item = dict(item)
        new_item["cifmol_attached_dict"] = mol.to_dict()
        kept[asm_key] = new_item
        surviving.extend(rows_here)

    counts.update(local)
    if not surviving:
        return None, []
    if modified:
        counts["entry_reencoded"] += 1
        return to_bytes(kept, level=ZSTD_LEVEL), surviving
    counts["entry_verbatim"] += 1
    return raw, surviving


def write_tsv(path: Path, rows: list[tuple[str, ...]]) -> None:
    """Write the filtered edge_node TSV in the source's sort order."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as fh:
        fh.write(TSV_HEADER)
        for row in sorted(rows, key=lambda r: (r[2], r[3], r[4], r[5], r[6], r[7])):
            fh.write("\t".join(row) + "\n")
    tmp.replace(path)
    print(f"\nwrote {len(rows):,} rows -> {path}")


def write_lmdb(path: Path, values: dict[bytes, bytes], payload: int) -> None:
    """Write the filtered LMDB, sized to what it actually holds."""
    if path.exists():
        if not path.name.endswith(".lmdb"):
            msg = f"refusing to delete non-.lmdb path: {path}"
            raise ValueError(msg)
        print(f"removing existing {path}")
        shutil.rmtree(path)

    map_size = int(payload * 1.5) + 512 * 1024 * 1024
    env = lmdb.open(str(path), map_size=map_size)
    pending: list[tuple[bytes, bytes]] = []
    for key in sorted(values):
        pending.append((key, values[key]))
        if len(pending) >= COMMIT_EVERY:
            with env.begin(write=True) as txn:
                for k, v in pending:
                    txn.put(k, v)
            pending.clear()
    if pending:
        with env.begin(write=True) as txn:
            for k, v in pending:
                txn.put(k, v)
    env.sync()
    written = env.stat()["entries"]
    env.close()
    if written != len(values):
        msg = f"wrote {len(values)} keys but the new env holds {written}"
        raise SystemExit(msg)
    print(f"wrote {written:,} keys -> {path} "
          f"({payload / 1e9:.3f} GB payload, map_size={map_size / 1e9:.2f} GB)")


def process(name: str, paths: dict[str, Path], args: argparse.Namespace) -> None:  # noqa: PLR0915
    """Filter one split and, unless dry-running, write its LMDB + TSV."""
    print(f"{'=' * 72}\n{name} -> protein-ligand\n{'=' * 72}")
    for label in ("in_lmdb", "in_tsv", "out_lmdb", "out_tsv"):
        print(f"{label:10s}: {paths[label]}")
    print(f"aid codes : {' '.join(sorted(AID_CODES))}")
    print(f"covalent  : struct_conn covale OR contact < {args.covalent_angstrom} A "
          f"to the row's own protein chain")

    print(f"\nnucleotide tier '{args.nucleotide_tier}' "
          f"(coverage {args.coverage}, similarity {args.similarity}):")
    nuc_codes, _ = nucleotide_codes(
        args.ccd_lmdb, args.nucleotide_tier, args.coverage, args.similarity,
        args.fp_radius, args.fp_bits)

    exclude: set[str] | None = None
    mapping: dict[str, str] = {}
    seen_from = paths.get("exclude_ligands_from")
    if seen_from is not None:
        mapping = load_ligand_map(paths["ligand_map"])
        exclude = seen_ligand_clusters(seen_from, mapping)
        print(f"\nligand holdout: excluding clusters seen in {seen_from.name}")
        print(f"  ligand map            : {paths['ligand_map'].name} "
              f"({len(mapping):,} ligands)")
        print(f"  clusters seen in train: {len(exclude):,}")

    rows_by_pdb, tally = read_rows(paths["in_tsv"], exclude, mapping)
    print(f"\nrows in edge table                     : {tally['rows_total']:,}")
    print(f"protein-ligand rows                    : {tally['rows_protein_ligand']:,}")
    if exclude:
        print(f"  dropped, ligand seen in train        : "
              f"{tally['rows_dropped_ligand_seen_in_train']:,}")
        print(f"  novel-ligand rows                    : "
              f"{tally['rows_protein_ligand'] - tally['rows_dropped_ligand_seen_in_train']:,}")
    print(f"candidate pdb entries                  : {tally['pdb_candidate']:,}")

    env = lmdb.open(str(paths["in_lmdb"]), readonly=True, lock=False, max_readers=2048)
    print(f"source entries in lmdb                 : {env.stat()['entries']:,}")

    kept_rows: list[tuple[str, ...]] = []
    out_values: dict[bytes, bytes] = {}
    counts: Counter[str] = Counter()
    started = time.time()

    with env.begin(buffers=True) as txn:
        for n_done, pdb in enumerate(sorted(rows_by_pdb), start=1):
            raw = txn.get(pdb.encode())
            if raw is None:
                counts["pdb_missing_from_lmdb"] += 1
                continue
            blob, surviving = build_entry(
                bytes(raw), rows_by_pdb[pdb], counts,
                args.covalent_angstrom, nuc_codes)
            if blob is None:
                counts["pdb_dropped"] += 1
            else:
                out_values[pdb.encode()] = blob
                kept_rows.extend(surviving)
                counts["pdb_kept"] += 1
            if n_done % REPORT_EVERY == 0:
                elapsed = time.time() - started
                rate = n_done / max(elapsed, 1e-9)
                eta = (len(rows_by_pdb) - n_done) / max(rate, 1e-9)
                print(f"  scanned {n_done:,}/{len(rows_by_pdb):,} "
                      f"({elapsed:.0f}s, {rate:.0f}/s, eta {eta / 60:.0f}m, "
                      f"kept {counts['pdb_kept']:,}, rows {len(kept_rows):,})",
                      flush=True)
    env.close()

    payload = sum(len(v) for v in out_values.values())
    pl = max(tally["rows_protein_ligand"], 1)
    print(f"\n-- rows ({time.time() - started:.0f}s)")
    print(f"protein-ligand rows in                 : {tally['rows_protein_ligand']:,}")
    counts["rows_dropped_ligand_seen_in_train"] = \
        tally["rows_dropped_ligand_seen_in_train"]
    for label, key in (
        ("dropped, ligand seen in train       ", "rows_dropped_ligand_seen_in_train"),
        ("lost with a nucleic-acid entry      ", "rows_dropped_with_nucleic_entry"),
        ("dropped, ligand is an aid           ", "row_dropped_ligand_is_aid"),
        ("dropped, ligand covalent to protein ", "row_dropped_covalent"),
        ("dropped, chain absent from structure", "row_dropped_chain_absent"),
    ):
        print(f"  {label} : {counts[key]:>9,}  ({100 * counts[key] / pl:5.1f}%)")
    print(f"  kept                                 : {len(kept_rows):>9,}  "
          f"({100 * len(kept_rows) / pl:5.1f}%)")

    print("\n-- entries excluded for nucleic acid")
    print(f"contained a DNA/RNA/hybrid chain       : "
          f"{counts['entry_dropped_nucleic_chain']:,}")
    print(f"contained a nucleotide/analogue ligand : "
          f"{counts['entry_dropped_nucleotide_ligand']:,}")

    print("\n-- structures")
    print(f"sub-entries kept, no aid present       : {counts['subentry_clean']:,}")
    print(f"sub-entries kept, aids stripped        : {counts['subentry_stripped']:,}")
    print(f"sub-entries dropped, no surviving row  : {counts['subentry_no_row']:,}")
    print(f"aid residues removed                   : {counts['aid_residues_removed']:,}")
    print(f"pdb entries kept                       : {counts['pdb_kept']:,}")
    print(f"  copied verbatim                      : {counts['entry_verbatim']:,}")
    print(f"  re-encoded                           : {counts['entry_reencoded']:,}")
    print(f"pdb entries dropped                    : {counts['pdb_dropped']:,}")
    if counts["pdb_missing_from_lmdb"]:
        print(f"pdb entries missing from lmdb          : {counts['pdb_missing_from_lmdb']:,}")
    print(f"output payload                         : {payload / 1e9:.3f} GB")

    if args.dry_run:
        print("\n[dry-run] nothing written.")
        return

    write_tsv(paths["out_tsv"], kept_rows)
    write_lmdb(paths["out_lmdb"], out_values, payload)


def main() -> None:
    """Parse the command line and run."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=[*SPLITS], default="train")
    ap.add_argument("--in-lmdb", type=Path)
    ap.add_argument("--in-tsv", type=Path)
    ap.add_argument("--out-lmdb", type=Path)
    ap.add_argument("--out-tsv", type=Path)
    ap.add_argument("--exclude-ligands-from", type=Path,
                    help="Built edge table whose ligand clusters to hold out. "
                         "Default: set for valid2, unset for train.")
    ap.add_argument("--ligand-map", type=Path,
                    help="Tanimoto old->new cluster map, for the holdout comparison.")
    ap.add_argument("--ccd-lmdb", type=Path, default=CCD_LMDB)
    ap.add_argument("--nucleotide-tier", choices=TIERS, default="similar")
    ap.add_argument("--coverage", type=float, default=0.7)
    ap.add_argument("--similarity", type=float, default=0.5)
    ap.add_argument("--fp-radius", type=int, default=2, help="Morgan radius (ECFP4).")
    ap.add_argument("--fp-bits", type=int, default=4096, help="Fingerprint bits.")
    ap.add_argument("--covalent-angstrom", type=float, default=COVALENT_ANGSTROM)
    ap.add_argument("--dry-run", action="store_true", help="Report only; write nothing.")
    args = ap.parse_args()

    paths = dict(SPLITS[args.split])
    for key, value in (("in_lmdb", args.in_lmdb), ("in_tsv", args.in_tsv),
                       ("out_lmdb", args.out_lmdb), ("out_tsv", args.out_tsv),
                       ("exclude_ligands_from", args.exclude_ligands_from),
                       ("ligand_map", args.ligand_map)):
        if value is not None:
            paths[key] = value
    process(args.split, paths, args)


if __name__ == "__main__":
    main()
