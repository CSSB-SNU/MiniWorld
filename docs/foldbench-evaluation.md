# Scoring MiniWorld on FoldBench

How to turn a set of predicted CIFs into FoldBench's own numbers, and the three
output-format bugs that had to be fixed before any of it worked.

FoldBench (BEAM-Labs, *Nat. Commun.* 2025) lives at `/home/psk6950/data/foldbench`:
`upstream/` is the benchmark repo, `targets/` the nine task CSVs, `ground_truth/`
the 1,522 answer CIFs, `meta/paper_scores/` the published per-target scores for
AlphaFold 3, Protenix, Boltz-1, Chai-1 and HelixFold 3.

## Sampling: 5 seeds x 5 diffusion samples

`scripts/run_miniworld_diffusion_inference.py foldbench --n-seeds 5 --n-samples 5`
is the AF3/FoldBench protocol and what the numbers are reported on. A seed is
**not** just a diffusion noise seed: `build_inference_batch`'s rng drives the
per-residue SE(3) randomisation of the reference conformers, the MSA subsample and
the template sampling, so each seed feeds the trunk a different input and gets its
own recycled representation. Drawing 25 samples from one trunk pass varies only the
diffusion noise and understates the ensemble.

Predictions land at `<output_dir>/<target>/<target>_seed{s}_sample{k}_pred.cif`.

## Running the evaluation

```bash
PY=/home/psk6950/data/foldbench/upstream/.venv/bin/python
$PY scripts/foldbench_evaluate.py \
    --pred-dir runs/foldbench/<run> --eval-dir eval_results/foldbench --name <run>
$PY scripts/foldbench_compare.py \
    --eval-dir eval_results/foldbench --name <run>
```

`foldbench_evaluate.py` wraps upstream `evaluate.py` + `task_score_summary.py`, so
the metrics are the benchmark's own code. It adds three things: the
`prediction_reference.csv` upstream expects, task CSVs filtered to the targets we
actually predicted (upstream left-joins onto all 1,522 and every unpredicted row
becomes a NaN that crashes its workers), and both aggregations (see below).
`foldbench_compare.py` then restricts the paper's per-target scores to the same
interfaces and recomputes each published method's category metric over that subset,
so the comparison is like-for-like.

Two pieces of environment this needs, neither of which the cluster has:

* **OpenStructure.** Every metric goes through `ost compare-structures` /
  `compare-ligand-structures`. `upstream/images/ost.sif` is the upstream container
  (`singularity pull docker://registry.scicore.unibas.ch/schwede/openstructure:latest`)
  and `upstream/bin/ost` is a shim that runs it; `foldbench_evaluate.py` puts that
  shim on PATH.
* **`upstream/.venv`** — a `--system-site-packages` venv on the pixi python adding
  biopython and parallelbar, which DockQv2 imports and the pixi env does not carry.

## Reading the two columns

This checkpoint has no confidence head, so it cannot rank its own samples.

* **`ours@1`** — one arbitrary sample per interface. This is the honest comparison:
  the published methods submit the sample their own ranker chose, and an unranked
  model can only offer a random one. Upstream's default `--metric_type rank` is an
  `idxmax` over `ranking_score`, which for a constant degenerates to exactly this.
* **`ours@best`** — best of N per interface. An **oracle upper bound**: what a
  working ranking head could reach, not what we have. Never quote it against a
  published top-1 number.

## The category metrics

From `task_score_summary.py`:

| categories | metrics | success threshold |
|---|---|---|
| antibody-antigen, protein-protein, -peptide, -dna, -rna | DockQ success rate, iRMSD, LRMSD, lDDT | DockQ >= 0.23 |
| protein-ligand | rmsd_lddt-pli success rate, lDDT-LP, lDDT-PLI | RMSD < 2 A **and** lDDT-PLI > 0.8 |
| monomer protein / dna / rna | GDT-TS, TM-score, RMSD, lDDT | — |

Per-chain RMSD is not among them. DockQ for protein-DNA and protein-RNA comes from
DockQv2, not OST.

## Three bugs this surfaced

All three were in our output path, and each silently zeroed out metrics rather than
failing loudly.

1. **`_atom_site.pdbx_formal_charge` written as `0.0`.** mmCIF declares it an
   integer column, so OpenStructure refuses the whole file
   (`Expecting integer value ... found '0.0'`) — i.e. *every* FoldBench metric was
   unobtainable. Fixed in `data/io/to_cif.py` with an `int()` cast.
2. **No `_entity` / `_entity_poly` / `_entity_poly_seq`.** We emitted only
   `_atom_site`. OST limps along under `--fault-tolerant` by guessing polymers from
   the CCD and chem-grouping by sequence identity, but `compare-ligand-structures`
   hard-requires `_entity.type` and DockQv2's parser hard-requires
   `_entity_poly_seq.entity_id`. So **protein-ligand, protein-DNA and protein-RNA
   scored nothing at all**. `to_cif.py` now derives all three tables from the same
   masked per-atom arrays it writes into `_atom_site`, matching the ground-truth
   convention (1-based entity ids; `label_seq_id` 1-based within a chain for
   polymers and `.` for non-polymers). Note `token_residue_idx` runs globally across
   the complex, so a homomer's copies must be renumbered per chain.
3. **Template dropout was active at inference.** `apply_template_dropout` is a plain
   function, not an `nn.Dropout`, so `model.eval()` did not switch it off: every
   inference forward erased each of the four template pair classes with p=0.25. Those
   channels are a one-hot encoding (contact / negative / ambiguous / multistate,
   unknown = all zeros) with no `1/(1-p)` rescale, so a dropped channel deletes a
   whole class of template evidence rather than scaling it down. Now gated on
   `self.training` in `models/miniworld/model.py`.

## Checked against the AF3 reference implementation

Compared against `/home/psk6950/alphafold3` (`model/network/diffusion_head.py`,
`evoformer.py`, `run_alphafold.py`). Matching, i.e. **not** a source of error: the
noise schedule (`sigma_data * (smax^(1/p) + t*(smin^(1/p)-smax^(1/p)))^p`, p=7,
smin=4e-4, smax=160), the preconditioning `c_in`/`c_out`/`c_skip`, the noise
embedding (`1/4 * log(sigma/sigma_data)` into the same fixed Fourier weights),
churn (`gamma_0=0.8`, `gamma_min=1.0`, tested on the *next* noise level),
`noise_scale=1.003`, `step_scale=1.5`, the ODE derivative taken at the *noised*
iterate, per-step centre+rotate+translate, recycling as
`init + Linear(LayerNorm(prev))` with a zero-init Linear, MSA truncation to the
first N rows, and 5 diffusion samples x 200 steps per seed.

Two divergences found:

* **The per-step centring was unmasked.** AF3's `random_augmentation` takes a
  `mask_mean` and zeroes the padding slots; ours averaged over the whole atom axis.
  Padding slots hold pure noise at `sigma_0 = 16 * 160 = 2560 A`, so the centre gets
  dragged in proportion to how much of the axis is padding: a half-padded bucket
  leaves the real structure ~25 A off origin (measured), a FoldBench inference batch
  only ~7 stride-padding atoms and so effectively nothing. It matters for **phase 3
  confidence training**, which samples structures from bucket-padded training
  batches. Fixed by threading `batch.structure.atom_mask` through
  `AF3Solver.sample -> step -> _centre_random_augmentation`.
* **Only half the seed was set.** The per-step rotation comes from
  `scipy.spatial.transform.Rotation.random`, which reads numpy's global RNG, while
  the runner seeded only torch. Sampling was therefore not reproducible across
  shard splits. The runner now seeds numpy per target as well.

A third difference — AF3 subsamples the MSA and we did not — is covered under **MSA
subsampling** below.

## Sampling settings: what FoldBench runs the published methods at

`upstream/algorithms/Protenix/make_predictions.sh` is the benchmark's own reference
invocation:

```bash
N_sample=5   N_step=200   N_cycle=10   seed=42,66,101,2024,8888
```

AF3 agrees: `DiffusionHead.Config.eval` defaults to `num_samples=5, steps=200`, and
`run_alphafold.py` defaults to `num_recycles=10`. We match all four
(`--n-seeds 5 --n-samples 5 --timesteps 200 --num-recycles 10`). 200 denoising steps
is not excessive — it is what both reference implementations use.

Recycling is still an extrapolation for us: training samples `n_recycle` uniformly
from **1..4** (`n_recycle_max: 4`). 10 is the same extrapolation everyone else makes;
an earlier run used 20, which is a bigger one and was never validated.

## MSA subsampling

AF3 subsamples the MSA, and it does it **inside the model**, not in featurisation:

```python
# evoformer.py, _embed_process_msa, called once per recycle iteration
msa_batch, key = featurization.shuffle_msa(key, msa_batch)   # uniform over all rows
msa_batch = featurization.truncate_msa_batch(msa_batch, 1024)
```

(`features.MSA.compute_features` takes no `random_state`, which is why looking only
at the data pipeline gives the wrong answer.) We do it once per seed rather than once
per recycle, via `--msa-subsample` (default on): depth stays at `max_msa_depth`, but
*which* homologs are used varies with the seed.

**It must be done per chain.** Sampling rows of the stacked alignment uniformly — the
obvious reading of "shuffle then truncate" — is wrong for our layout and destroys
predictions. Under `no_pairing` the stack is positional (row r = every chain's r-th
homolog) and as tall as the *deepest* chain, with shallower chains holding query/gap
fill past their own depth. A uniform draw then:

* keeps only `depth/total` of a shallow chain's real homologs — 7xld chain 0 fell
  from 2047 to ~460 of its 3665, and an 8tuz chain with 89 homologs in the entire
  pool kept 12-28 of them;
* throws away the close homologs, since a3m rows are ordered best-first — 7xld
  chain 0 went from 4 rows above 70% identity to 0-1.

Measured effect on 7xld (antibody-antigen, DockQ per seed over 5 diffusion samples):

| | seed 0 | 1 | 2 | 3 | 4 | success >= 0.23 | median |
|---|---|---|---|---|---|---|---|
| uniform over the stack | 0.866 | 0.006 | 0.005 | 0.015 | 0.043 | 5/25 | 0.015 |
| per chain (`_subsample_no_pairing`) | 0.591 | 0.559 | 0.570 | 0.555 | 0.580 | **25/25** | **0.577** |

Four of five seeds collapsed completely under uniform sampling (RMSD ~20 A); per-chain
sampling removes the collapse and every seed lands at DockQ ~0.57, RMSD ~2 A. Note the
one lucky seed under the broken scheme scored higher (0.866) than anything the fixed
scheme produces — with no confidence head there is no way to pick that seed, so the
random-sample expectation, 0.015 vs 0.577, is the number that matters.

AF3 escapes the problem because the 16,384 rows it shuffles are all real homologs:
each chain is cropped best-first to its own budget (`msa_crop_size`, `features.py`)
before the shuffle, so uniform sampling preserves per-chain depth.

The remaining tradeoff is real and shared with AF3: a *deep* chain still loses its
best-first ordering (7xld chain 1, mean identity to query 0.507 -> 0.356, rows above
50% identity 1256 -> ~360). Turn the whole thing off with `--no-msa-subsample` if a
measurement says that costs more than the ensemble diversity buys.
