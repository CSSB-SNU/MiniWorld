# CCD reference-coordinate fallback

Training and inference share `miniworld.data.reference`. The policy is:

1. Keep complete, finite CCD model coordinates.
2. For an incomplete component, reconstruct its full chemical graph and generate
   a complete reference conformer using RDKit ETKDGv3. Replace the whole component
   coordinate frame, then map by atom name to the cropped/stripped atom list.
3. If chemistry cannot be reconstructed or embedding fails, retain only valid
   model coordinates. Missing atoms have `reference.mask=False`; they do not enter
   the centroid and their output positions remain zero even after augmentation.

`?`, `.`, `None`, NaN and infinity are invalid coordinates. A partly specified
atom is masked as a whole. Reference validity does not change the structural atom
existence mask or the ground-truth coordinates used by the loss.

## Relationship to AF3

[AF3 get_reference](https://github.com/google-deepmind/alphafold3/blob/main/src/alphafold3/model/features.py)
tries RDKit first and falls back to CCD ideal/model coordinates. This implementation
uses RDKit as a **missing-coordinate fallback**, retaining MiniWorld's complete
existing references. It is not an exact AF3 featurization implementation. Existing
MiniWorld LMDBs retain model coordinates but do not contain the ideal coordinate
fields, so ideal-coordinate and CCD-date fallback policies are not implemented.

The RDKit molecule uses CCD elements, charges, bond orders and R/S and E/Z
annotations. Generated stereochemistry is checked from the 3D coordinates. An
unsupported/inconsistent chemical description falls back to explicit masks.

## Determinism and cost

Conformer seeds derive from a SHA-256 of the full topology, not Python's randomized
hash or the sample RNG. Generation uses one CPU thread, at most 200 ETKDG
iterations, a five-second timeout when supported by RDKit, and a 512-atom budget.
A per-process LRU holds 1,024 generation results, including failed attempts.
Independent per-residue SE(3) augmentation still uses the existing sample RNG.
This deliberately caches one conformer per topology rather than reproducing AF3's
random conformer distribution.

The training dataloader supplies its existing full CCD fragmentation cache. This
also handles crops that remove every originally missing atom: training and
inference still choose the same full-component conformer. Callers constructing
features directly should pass `ccd_mols` to `make_batch`/`to_reference_features`;
without a CCD lookup, missing coordinates remain masked rather than being repaired.

No LMDB rebuild is required. `pyproject.toml` and `pixi.lock` include RDKit; install
the updated environment before launching a new run. Existing frozen training
snapshots and the running training environment were not changed.

## Validation

CPU regression tests cover missing-value masks, all-missing components, failed
embedding, unsupported chemistry, deterministic cached generation, atom-name
mapping, crop/terminal removal, training/inference agreement, aromatic chemistry,
tetrahedral chirality and E/Z geometry. Run:

```bash
python -m pytest -q tests/test_reference_fallback.py
```

Real local CCD checks for DRP, Y7G, A1CAA and 4TJ generated complete RDKit references.
ALA retained its original coordinates. Evidence is recorded locally in
`runs/ccd-reference-audit-20260925/fallback-validation.json` (not a training-quality
evaluation). The ongoing training job was not restarted or migrated.
