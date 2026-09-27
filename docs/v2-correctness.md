# MiniWorld v2: data, targets, and inference consistency

Status: **2.0.0.dev0**, development consolidation, 2026-09-25.
This is the model/pipeline version, independent of miniworld-engine 2.0.0.
No existing job, checkpoint, W&B run, frozen snapshot, or runtime environment was restarted or migrated.

## Report audit

| # | Finding | Current action/status |
|---|---|---|
| 1 | Unresolved atoms supervise pLDDT | Fixed: resolved GT mask, separate existence mask for head inputs. |
| 2 | Diffusion loss diluted by bucket padding | Fixed in local `libs/team-gm`: valid coordinate mean per example/augmentation, then batch mean. Atom-type and sigma weights remain numerator multipliers. Empty observations produce zero loss/gradient. |
| 3 | Inference noncanonical tokenization differs | Report stale for default policy: current inference already atomizes noncanonical components. Explicit user fragmentation overrides still intentionally differ. |
| 4 | Missing CCD coordinates marked valid | Shared missing-only RDKit fallback, deterministic whole-component regeneration, valid-only centering, masked failures. See `ccd-reference-fallback.md`. |
| 5 | Inference lacks representative atoms | Fixed: inference builds chemical representatives and frame indices with the same selectors as training. GT mask is false without experimental coordinates; existence mask remains true. |
| 6 | Template padding/order mismatch | Empty train/inference slots use `ProteinTemplate.empty/padded`; configured count defaults to 4, fixed-slot mean. Inference no longer randomizes database template order. Entirely absent templates produce zero update. Explicit extra complex-template slots are retained. |
| 7 | Nucleic representative atom | Already C4 for purines and C2 for pyrimidines. No new change to that selection rule. |
| 8 | Sampling-step rigid augmentation missing | Already in AF3Solver via shared EulerSampler's augmentation callback. No duplicate augmentation added. |
| 9 | Interchain query/residue features zeroed | Fixed: multichain mask gates geometry and geometric validity, not query or residue identity projections. |
| 10 | Validation batching/best-of-N | Sample N structures from one trunk pass; compute best RMSD and best lDDT over all samples. Multi-complex validation processes B=1 kernels per complex and averages metrics. |
| 11 | Confidence GT index fixed to zero | Fixed: batch-specific lDDT, molecule masks, frame indices/validity. |
| 12 | Confidence chain/atom symmetry alignment | Partial, conservative implementation: equivalent chains with identical crop signatures plus standard ASP/GLU/ARG/LEU/VAL/PHE/TYR atom-name symmetries. Coordinates and observed masks move together. See limits below. |

## Symmetry limits — not a complete general CCD solution

`loss/symmetry.py` runs before confidence target construction under no-grad, outside CUDA graphs.
Chains must have matching entity IDs and exact cropped residue/atom signatures. Up to 720 chain assignments are enumerated; larger groups use improving pair swaps, not a guaranteed global optimum. Cropped chains with different atom sets are not interchanged. Known covalent-link endpoints are protected. Aromatic name swaps are coupled.

Arbitrary ligand/modified-residue automorphisms are **not implemented**. Batch `atom_bond` is currently a placeholder, and its separate `bond_atom_pairs` only contains a resolved, distance-filtered subset of structural links. Complete CCD and inter-residue bond/stereo metadata is required before expanding symmetry support safely. This is a remaining v2 release requirement, as are end-to-end/GPU validation and real-complex symmetry checks. The development version is not a declaration that all 12 findings are fully closed.

## Version/configuration policy

- New `phase1{a,b}_distogram_{,medium_}v200.yaml` inherits v1.2 recipes, including interface upweight, and selects new v2 run directories.
- New `phase3{a,b}_confidence_v200.yaml` enables conservative target alignment.
- Existing `phase2{a,b}_diffusion_v200.yaml` already means BiasOnlyTokenDiT + BF16 diffusion; preserved, not silently replaced with an architectural baseline.
- Source-level correctness fixes are not conditional on config filename. To reproduce old loss/template behavior, use the old commit/frozen environment, not a v1-named config with new source.
- The corrected loss can be substantially larger for small samples (e.g. roughly 8.2x for 1000 valid vs 8192 padded atoms). Old/new loss curves cannot be compared as identical objectives; review diffusion loss weights when launching v2.
- `libs/team-gm` is a separate git repository/submodule. Its loss patch must be committed/published and the parent gitlink updated together before distributing the release.
- RDKit is declared but not installed into the running training environment. CPU tests use the isolated CCD audit venv.

## Validation

Run (CPU, no training jobs):

```sh
OMP_NUM_THREADS=1 PYTHONPATH=src:libs/team-gm/src MPLCONFIGDIR=/tmp/mw-ccd-mpl \
  runs/ccd-reference-audit-20260925/venv/bin/python -m pytest \
  tests/test_reference_fallback.py tests/test_v2_correctness.py -q
```

Tests cover fallback chemistry/crop mapping, padding-invariant loss and gradients, empty losses, batch-specific confidence targets, head inputs independent of observed-mask availability, second-sample best-of-N, absent/interchain templates, actual inference batch construction, chain identity/mask permutation, and standard atom symmetry. Real training convergence, H100/CUDA-graph execution, general CCD automorphisms, and large-homomer optimal assignment are not validated by these CPU tests.
