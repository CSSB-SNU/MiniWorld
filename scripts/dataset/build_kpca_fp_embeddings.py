#!/usr/bin/env python3
"""Build KPCA fingerprint embedding tables for the token embedding.

Replaces the deleted ``fp_emb_table.pt``:

  ``CCD/McAc/table_d32.pt``  KPCA(Morgan count) || KPCA(atom-pair count)
  ``CCD/vocab.json``         the ``chem_comp_id -> row`` map

The ``_d<N>`` suffix is taken from the tensor's actual width, so it cannot drift
from the file it names -- and ``d_single_token_input`` is read straight off it.

Both match the interface ``TokenEmbeddingConfig`` expects: ``torch.load`` yields
an ``(V, D)`` tensor, and the vocab maps a code to its row with ``UNK`` as the
fallback.

Dimensions
----------
``init_token_single_msa_explicit`` (modules/msa_util_explicit.py) concatenates
the embedding twice -- once as the token's own row, once as the MSA profile
projected through the table -- plus a scalar deletion mean::

    d_single_token_input = 2 * D + 385

which is why the current config reads 1921 for the old D=768 table.  At the
default 16 components per block, McAc is D=32 (-> 449).  The script prints the
value to set.

The kernel
----------
RDKit's ``TanimotoSimilarity`` on a count fingerprint (``UIntSparseIntVect``) is
MinMax / Ruzicka, ``sum(min)/sum(max)`` -- verified, and *not* the dot-product
form.  That is the intended count-Tanimoto and is PSD, so it serves directly as
a kernel for KPCA.

KPCA is exact rather than Nyström: the full ~49k x 49k kernel is 9.7 GB in
float32, so ``eigsh`` over a ``LinearOperator`` that applies the double-centering
on the fly gets the leading components in seconds without ever materialising the
centred matrix.

Building a kernel is by far the slow part -- serially 50 min for Morgan and 2 h
for atom-pair over ~49k components, because real CCD count fingerprints are far
denser than drug-sized molecules and MinMax cost scales with the number of
nonzero elements.  The eigendecomposition is 6 s.  Two things address that:

*Caching.*  Kernels are written to ``--cache-dir`` as ``.npy`` (9.7 GB each) and
memory-mapped on reuse, so sweeping ``--n-components`` costs seconds rather than
hours.  ``--no-cache`` skips them.

A cache is only trusted when a sidecar ``.done`` marker sits beside it.  The
parallel builder allocates the full-size ``.npy`` up front and fills it in place,
so a run killed midway leaves a file of exactly the right shape holding partly
zeros -- indistinguishable from a finished kernel by shape alone, and silently
reusable as garbage.  The marker is written only after the last chunk lands.

*Parallelism.*  With ``--workers N`` the rows are split across a forked pool that
writes straight into the memory-mapped ``.npy``, so the kernel is never held in
RAM and the cache file is the shared buffer rather than a separate save.  Each
worker computes COMPLETE rows rather than upper-triangle slices: the triangle
would require mirroring each row into a column that other workers own, which is
a write race on shared pages.  Owning whole rows means no cell is written twice
and no locking is needed, at the cost of 2x the arithmetic -- which the workers
more than absorb.  Fingerprints reach the workers by ``fork`` copy-on-write
through a module global, never by pickling.

Each block is standardised to unit variance before concatenation.  Raw KPCA
components carry a ``sqrt(eigenvalue)`` scale, so one block's leading axis can
dwarf the whole of the other; unscaled, the concatenation would be dominated by
whichever fingerprint has the larger spectrum and the 0/1 one-hot would vanish
beside both.

Special rows
------------
``UNK`` is a real CCD in the store and keeps its own fingerprint.

``RNA_UNK`` and ``DNA_UNK`` are built from the backbone-scaffold SMILES in
``scripts/fp_emb_old.py`` (there keyed ``R_X`` and ``d_X``): ribose-5'-phosphate
and 2'-deoxyribose-5'-phosphate, each with the variable base replaced by an
amine.  Because those are real molecules they are fingerprinted and fed into the
kernel alongside every CCD, so they receive genuine KPCA coordinates -- not a
synthetic average of A/U/G/C, which would place them at a centroid no real
nucleotide occupies.

``GAP_ZERO`` is a zero row, being a gap.

A CCD whose SMILES will not parse also gets a zero KPCA row.

Usage
-----
    /home/bsoohyuncd/.conda/envs/prolif/bin/python \\
        scripts/dataset/build_kpca_fp_embeddings.py [--n-components 16]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

CCD_DIR = Path("/public_data/bsoohyuncd/CCD")

# Set before forking so workers inherit the fingerprints copy-on-write.
_WORKER_FPS: list | None = None
CCD_LMDB = CCD_DIR / "preprocessed_CCD_20260826.lmdb"
VOCAB_PATH = CCD_DIR / "vocab.json"
V1_DIR = CCD_DIR / "McAc"

# Backbone scaffolds for the unknown-nucleotide rows, from scripts/fp_emb_old.py
# (keyed R_X and d_X there).  Real molecules, so they join the kernel directly.
SCAFFOLD_SMILES = {
    "RNA_UNK": "N[C@@H]1O[C@H](COP(O)(O)=O)[C@@H](O)[C@H]1O",
    "DNA_UNK": "N[C@H]1C[C@H](O)[C@@H](COP(O)(O)=O)O1",
}
ZERO_KEYS = ["GAP_ZERO"]


def load_bytes(byte_data: bytes) -> dict:
    """Decode one BioMolDB blob (standalone copy of biomol.core.utils.load_bytes)."""
    from io import BytesIO

    from zstandard import ZstdDecompressor

    with ZstdDecompressor().stream_reader(BytesIO(byte_data)) as reader:
        raw = reader.read()
    header_len = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8 : 8 + header_len].decode("utf-8"))
    payload = memoryview(raw)[8 + header_len :]
    flat = {}
    offset = 0
    for key, length in header["arrays"].items():
        flat[key] = payload[offset : offset + length]
        offset += length

    def rebuild(template: dict) -> dict:
        out = {}
        for key, value in template.items():
            if isinstance(value, str) and value in flat:
                out[key] = np.load(BytesIO(flat[value]), allow_pickle=False)
            elif isinstance(value, dict):
                out[key] = rebuild(value)
            else:
                out[key] = value
        return out

    return rebuild(header["template"])


def read_smiles(path: Path) -> dict[str, str]:
    """CCD code -> rdkit_smiles for every component in the store."""
    import lmdb

    out: dict[str, str] = {}
    env = lmdb.open(str(path), readonly=True, lock=False)
    with env.begin() as txn:
        for key, value in txn.cursor():
            nodes = load_bytes(bytes(value))["residues"]["nodes"]
            if "rdkit_smiles" not in nodes:
                continue
            smi = str(np.asarray(nodes["rdkit_smiles"]["value"]).reshape(-1)[0]).strip()
            if smi:
                out[bytes(key).decode()] = smi
    env.close()
    return out


def done_marker(cache: Path) -> Path:
    """Sidecar path proving a cached kernel was written to completion."""
    return cache.with_suffix(cache.suffix + ".done")


def cache_is_complete(cache: Path) -> bool:
    """True only when the kernel exists AND its .done marker does."""
    return cache.exists() and done_marker(cache).exists()


def _init_worker(fps: list) -> None:
    """Runs in each forked child; binds the inherited fingerprint list."""
    global _WORKER_FPS  # noqa: PLW0603
    _WORKER_FPS = fps


def _fill_rows(job: tuple[Path, int, int]) -> int:
    """Compute complete kernel rows [start, end) into the memmapped .npy."""
    from rdkit import DataStructs

    path, start, end = job
    fps = _WORKER_FPS
    kernel = np.load(path, mmap_mode="r+")
    for i in range(start, end):
        kernel[i, :] = np.asarray(
            DataStructs.BulkTanimotoSimilarity(fps[i], fps), dtype=np.float32,
        )
    kernel.flush()
    return end - start


def build_kernel_parallel(
    fps: list,
    label: str,
    cache: Path,
    workers: int,
    chunk: int = 500,
) -> np.ndarray:
    """Kernel built by a forked pool writing into a memmapped .npy.

    Returns the memmap, which is also the cache file.
    """
    import multiprocessing as mp
    from numpy.lib.format import open_memmap

    n = len(fps)
    cache.parent.mkdir(parents=True, exist_ok=True)
    open_memmap(cache, mode="w+", dtype=np.float32, shape=(n, n)).flush()
    jobs = [(cache, s, min(s + chunk, n)) for s in range(0, n, chunk)]
    print(f"    {label}: {len(jobs)} chunks of <={chunk} rows across {workers} workers",
          flush=True)
    started = time.time()
    done = 0
    ctx = mp.get_context("fork")
    with ctx.Pool(workers, initializer=_init_worker, initargs=(fps,)) as pool:
        for got in pool.imap_unordered(_fill_rows, jobs):
            done += got
            if done % 5000 < chunk:
                elapsed = max(time.time() - started, 1e-9)
                print(f"    {label}: {done:,}/{n:,} rows ({elapsed:.0f}s, "
                      f"{done * n / 1e6 / elapsed:.1f}M pairs/s)", flush=True)
    print(f"    {label}: kernel {n:,}x{n:,} in {time.time() - started:.0f}s "
          f"-> {cache}")
    done_marker(cache).write_text(f"rows={n}\nworkers={workers}\n")
    return np.load(cache, mmap_mode="r")


def build_kernel(
    fps: list,
    label: str,
    cache: Path | None = None,
) -> np.ndarray:
    """Full symmetric MinMax-Tanimoto kernel as float32, cached to disk.

    A cached kernel is memory-mapped rather than read, so reuse costs no RAM
    beyond the pages eigsh actually touches.
    """
    from rdkit import DataStructs

    n = len(fps)
    if cache is not None and cache_is_complete(cache):
        kernel = np.load(cache, mmap_mode="r")
        if kernel.shape == (n, n):
            print(f"    {label}: reusing cached kernel {cache} "
                  f"({kernel.nbytes / 1e9:.1f} GB, memory-mapped)")
            return kernel
        print(f"    {label}: cached kernel is {kernel.shape}, need {(n, n)} -- rebuilding")
    elif cache is not None and cache.exists():
        print(f"    {label}: {cache} has no .done marker -- treating as incomplete "
              f"and rebuilding")

    kernel = np.empty((n, n), dtype=np.float32)
    started = time.time()
    for i in range(n):
        row = np.asarray(
            DataStructs.BulkTanimotoSimilarity(fps[i], fps[i:]), dtype=np.float32,
        )
        kernel[i, i:] = row
        kernel[i:, i] = row
        if (i + 1) % 5000 == 0:
            elapsed = max(time.time() - started, 1e-9)
            done = (i + 1) * n - (i + 1) * i // 2
            print(f"    {label}: {i + 1:,}/{n:,} rows ({elapsed:.0f}s, "
                  f"{done / 1e6 / elapsed:.1f}M pairs/s)", flush=True)
    print(f"    {label}: kernel {n:,}x{n:,} built in {time.time() - started:.0f}s "
          f"({kernel.nbytes / 1e9:.1f} GB)")
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        started = time.time()
        np.save(cache, kernel)
        done_marker(cache).write_text(f"rows={n}\nworkers=1\n")
        print(f"    {label}: cached -> {cache} ({time.time() - started:.0f}s)")
    return kernel


def kpca(kernel: np.ndarray, n_components: int, label: str) -> np.ndarray:
    """Exact kernel PCA via eigsh on a double-centering LinearOperator."""
    from scipy.sparse.linalg import LinearOperator, eigsh

    n = kernel.shape[0]
    row_sums = kernel.sum(axis=1, dtype=np.float64)
    total = float(row_sums.sum())

    def matvec(v: np.ndarray) -> np.ndarray:
        vec = np.asarray(v, dtype=np.float32).reshape(-1)
        out = (kernel @ vec).astype(np.float64)
        sv = float(vec.sum())
        out -= out.sum() / n              # - (1/n) J K v
        out -= row_sums * (sv / n)        # - (1/n) K J v
        out += total * sv / (n * n)       # + (1/n^2) J K J v
        return out

    started = time.time()
    values, vectors = eigsh(
        LinearOperator((n, n), matvec=matvec, dtype=np.float64),
        k=n_components, which="LA",
    )
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    n_neg = int((values <= 0).sum())
    values = np.clip(values, 0.0, None)
    print(f"    {label}: eigsh {n_components} components in {time.time() - started:.0f}s"
          f"{f', {n_neg} non-positive eigenvalues clipped' if n_neg else ''}")
    print(f"    {label}: top eigenvalues {np.round(values[:6], 2).tolist()}")
    trace = float(np.trace(kernel)) - total / n
    if trace > 0:
        print(f"    {label}: captures {100 * values.sum() / trace:.1f}% of the "
              f"centred kernel trace")
    return (vectors * np.sqrt(values)[None, :]).astype(np.float32)


def main() -> None:  # noqa: PLR0915
    """Build and write both embedding tables plus the shared vocab."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ccd-lmdb", type=Path, default=CCD_LMDB)
    ap.add_argument("--vocab", type=Path, default=VOCAB_PATH)
    ap.add_argument("--v1-dir", type=Path, default=V1_DIR)
    ap.add_argument("--n-components", type=int, default=16,
                    help="KPCA components per fingerprint block (default 16).")
    ap.add_argument("--morgan-radius", type=int, default=2)
    ap.add_argument("--fp-size", type=int, default=4096)
    ap.add_argument("--cache-dir", type=Path, default=CCD_DIR / "kernel_cache",
                    help="Where to cache the Tanimoto kernels (9.7 GB each).")
    ap.add_argument("--no-cache", action="store_true",
                    help="Do not read or write cached kernels.")
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel workers for kernel building (default 1 = serial). "
                         "Requires caching, since the .npy is the shared buffer.")
    ap.add_argument("--verify-parallel", action="store_true",
                    help="Before using the parallel path, recompute a sample of rows "
                         "from an already-cached kernel and require an exact match.")
    args = ap.parse_args()

    import torch
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")

    print(f"ccd store : {args.ccd_lmdb}")
    smiles = read_smiles(args.ccd_lmdb)
    print(f"components with rdkit_smiles : {len(smiles):,}")

    morgan = rdFingerprintGenerator.GetMorganGenerator(
        radius=args.morgan_radius, fpSize=args.fp_size)
    atompair = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=args.fp_size)

    for key, smi in SCAFFOLD_SMILES.items():
        if key in smiles:
            msg = f"{key} unexpectedly present in the CCD store; refusing to overwrite"
            raise SystemExit(msg)
        smiles[key] = smi
    print(f"  + scaffold rows              : {', '.join(SCAFFOLD_SMILES)}")

    codes: list[str] = []
    mfps: list = []
    afps: list = []
    unparseable: list[str] = []
    for code in sorted(smiles):
        mol = Chem.MolFromSmiles(smiles[code])
        if mol is None:
            unparseable.append(code)
            continue
        codes.append(code)
        mfps.append(morgan.GetCountFingerprint(mol))
        afps.append(atompair.GetCountFingerprint(mol))
    print(f"fingerprinted                : {len(codes):,}")
    print(f"unparseable SMILES           : {len(unparseable):,}")

    blocks = []
    for label, fps in (("morgan", mfps), ("atompair", afps)):
        print(f"\n-- {label} count fingerprint, MinMax-Tanimoto kernel")
        cache = None if args.no_cache else (
            args.cache_dir / f"{label}_r{args.morgan_radius}_b{args.fp_size}.npy"
            if label == "morgan" else args.cache_dir / f"{label}_b{args.fp_size}.npy"
        )
        if cache is not None and cache_is_complete(cache):
            kernel = build_kernel(fps, label, cache)
        elif args.workers > 1:
            if cache is None:
                msg = "--workers needs caching; drop --no-cache"
                raise SystemExit(msg)
            kernel = build_kernel_parallel(fps, label, cache, args.workers)
        else:
            kernel = build_kernel(fps, label, cache)
        block = kpca(kernel, args.n_components, label)
        del kernel
        std = float(block.std())
        print(f"    {label}: raw std {std:.4f}, standardised to unit variance")
        blocks.append(block / std if std > 0 else block)

    kpca_block = np.concatenate(blocks, axis=1)
    d_kpca = kpca_block.shape[1]

    # ---- vocab: parseable CCDs, then unparseable, then synthetic keys -----
    vocab = {code: i for i, code in enumerate(codes)}
    for code in unparseable:
        vocab.setdefault(code, len(vocab))
    for key in ZERO_KEYS:
        vocab.setdefault(key, len(vocab))
    if "UNK" not in vocab:
        msg = "UNK absent from the CCD store; the model needs it as the fallback row"
        raise SystemExit(msg)
    n_vocab = len(vocab)

    table = np.zeros((n_vocab, d_kpca), dtype=np.float32)
    for i, code in enumerate(codes):
        table[vocab[code]] = kpca_block[i]
    # RNA_UNK / DNA_UNK were fingerprinted above, so they already have real rows.
    # GAP_ZERO and unparseable CCDs keep zero rows.

    v1 = torch.from_numpy(table)

    args.v1_dir.mkdir(parents=True, exist_ok=True)
    args.vocab.parent.mkdir(parents=True, exist_ok=True)
    with args.vocab.open("w") as fh:
        json.dump(vocab, fh)
    v1_path = args.v1_dir / f"table_d{v1.shape[1]}.pt"
    torch.save(v1, v1_path)

    print(f"\n{'=' * 70}")
    print(f"vocab entries : {n_vocab:,}   UNK row = {vocab['UNK']}")
    print("  scaffold    : " + ", ".join(
        f"{k}={vocab[k]}" for k in SCAFFOLD_SMILES if k in vocab))
    print("  zero        : " + ", ".join(f"{k}={vocab[k]}" for k in ZERO_KEYS))
    print(f"  zero rows   : {len(unparseable) + len(ZERO_KEYS):,} "
          f"({len(unparseable)} unparseable + {len(ZERO_KEYS)} gap)")
    print(f"\nMcAc   {tuple(v1.shape)} -> {v1_path}")
    print(f"       d_single_token_input: {2 * v1.shape[1] + 385}")
    print(f"vocab  -> {args.vocab}")


if __name__ == "__main__":
    main()
