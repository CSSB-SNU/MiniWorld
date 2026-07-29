# Distogram-diffusion EDM `sigma_data` measurement (PDB)

EDM (Karras et al. 2022) preconditions on two data statistics of the target being
diffused: its **mean** (subtract to make the data zero-mean) and its **std**
(`sigma_data`, the scale the preconditioning normalises against). Our diffusion target
is the distogram **bin-index image** (0..D-1, D=64) of the CB/pseudo-beta
representative-atom distance — the same target `cal_atom_distogram_loss(rep_atom_mask=…)`
builds (`edges = linspace(2.25, 25.75, 63)`).

Because the distance distribution is **far-heavy** (most residue pairs are distant), the
bin distribution is skewed: its mean is NOT the geometric centre (31.5) and its std is
**crop-size dependent** — a larger crop L has proportionally more far pairs, pushing more
mass into the top bin, raising the mean and shrinking the std.

## Measured on the TRAINING MIX (the actual training distribution)

`source_weights = pdb 0.5 / protein_monomer 0.495 / short_protein_monomer 0.005 /
disordered_pdb 0.02` — the distribution the diffusion model actually trains on (NOT
pdb-only). Requires the dataloader fix `8a4bd96` for source_weights to be honoured on a
catalog-cache hit.

Script: [`scripts/measure_distogram_sigma_data.py`](../scripts/measure_distogram_sigma_data.py)
· job wrapper: [`submits/measure_sigma_data.sh`](../submits/measure_sigma_data.sh)
· 800 structures per crop, token-level CB `cdist`, valid i<j pairs.

| crop (max_tokens) | mean_bin (= `bin_center`) | `sigma_data` (raw bin) | sigma ([-1,1] norm) | frac in top bin (far) |
|------------------:|--------------------------:|-----------------------:|--------------------:|----------------------:|
| 384  | 53.48 | 14.676 | 0.4659 | 0.579 |
| 512  | 54.96 | 13.841 | 0.4394 | 0.638 |
| **768**  | **56.70** | **12.619** | **0.4006** | **0.711** |
| 1024 | 57.77 | 11.690 | 0.3711 | 0.757 |
| 2048 | 59.92 |  9.256 | 0.2938 | 0.854 |

(An earlier run that set `source_weights` to pdb-only but predated fix `8a4bd96` gave
essentially identical numbers — the override was silently ignored, so it too measured the
mix. This confirms the distillation-monomer CB-distance distribution barely differs from
PDB's.)

### Reading it
- `sigma_data` decreases monotonically with crop (14.68 → 9.60) as the image gets more
  far-heavy (`frac_top` 0.58 → 0.84, mean_bin 53.5 → 59.7).
- The module encodes `x0 = bin - bin_center` and `sigma_data` is the std of that centred
  image; **both are crop-dependent and MUST match the training `crop.max_tokens`.**

## Chosen basis: **crop = 768, D = 96 bins** (matches main)

The model uses **n_distogram_bins = 96** (matching main's small/large_H100 runs), so the
table above (D=64) was re-measured at D=96, crop=768, TRAINING MIX, CB target, 1000
structures:

| D | crop | bin_center (mean) | sigma_data (std) | frac_top |
|--:|-----:|------------------:|-----------------:|---------:|
| 64 | 768 | 56.57 | 12.69 | 0.71 |
| **96** | **768** | **85.70** | **18.90** | **0.72** |

(Same distance range [2.25, 25.75] Å; 96 bins are ~95/63 finer, so mean/std scale up
~1.5x, as observed.)

Set in the configs:
- [`config_distogram_swa_bioai_small_pf4_d512_diffusion.yaml`](../configs/miniworld/config_distogram_swa_bioai_small_pf4_d512_diffusion.yaml):
  `data.crop.max_tokens = 768`, `max_atoms = 8192`.
- [`model/medium_distogram_swa_pf4_notmpl_diffusion.yaml`](../configs/miniworld/model/medium_distogram_swa_pf4_notmpl_diffusion.yaml):
  `n_distogram_bins = 96`, `bin_center = 85.70`, `scheduler.sigma_data = 18.90`.

If the training crop changes, re-pick `bin_center` (= mean_bin) and `sigma_data`
(= sigma_raw) from the table row for that crop (or re-run the measurement script).
