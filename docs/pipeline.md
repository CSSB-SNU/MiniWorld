# MiniWorld training pipeline — v1.0.0

Canonical definition of the training pipeline. This document is the source of truth for
phase naming; configs, run directories and checkpoints follow it.

## Version policy

| version | meaning |
|---|---|
| **1.0.0** | the setup described in this document |
| **1.0.1** | 1.0.0 + `max_lr` 1.8e-3 -> 5e-3 (see below) |
| **1.0.2** | 1.0.1 + the two AF3 auxiliary diffusion losses (see below) |
| 1.0.x | anything that does **not** change the model architecture — config, data mix, hyper-parameters, curriculum lengths, bug fixes |
| 1.1.x | **architecture changes** (module added/removed/rewired, dimensions changed) |
| 2.x.x | reserved for a pipeline restructure (phases added/removed/reordered) |

The version lives in `pyproject.toml` (`version`) and `miniworld.__version__`. Stamp it into
run directory names and checkpoint metadata so a checkpoint can be identified later.

## Phases

Every phase pair `Na` / `Nb` is the same training stage run as a **crop curriculum**: a long
cheap run at `L=384` followed by a short run at `L=768`. Phase 1 already worked this way
(384 for epochs 0–800, then 768 for 800–900); phases 2 and 3 adopt the same shape.

| phase | stage | crop | epochs | trains | frozen | initialised from |
|---|---|---|---|---|---|---|
| **1a** | distogram trunk | L=384 / 4096 atoms | 800 | everything | — | scratch |
| **1b** | distogram trunk | L=768 / 8192 atoms | +100 (→900) | everything | — | 1a full state (weights + optimizer + scheduler + epoch counter) |
| **2a** | EDM diffusion | L=384 / 4096 atoms | 50 | diffusion module, `to_token_single_trunk` | trunk | **1b SWA_last12** (see below) |
| **2b** | EDM diffusion | L=768 / 8192 atoms | 20 | diffusion module | trunk | 2a |
| **3a** | confidence (pLDDT / PAE / PDE) | L=384 / 4096 atoms | 50 | confidence head | trunk + diffusion | 2b |
| **3b** | confidence | L=768 / 8192 atoms | 20 | confidence head | trunk + diffusion | 3a |

**Epoch counters restart at each numbered phase.** The trainer seeds `client.epoch` from the
`epoch` field of the `--ckpt` it is given and loops `while client.epoch < num_epoch`, so the
phase-2a seed checkpoint carries `epoch: 0` (provenance in `swa.source_epoch`). Within a
phase the counter continues across the a→b hand-off, which is why 2b/3b use
`num_epoch: 70` (= 50 + 20) rather than 20.

Freezing is enforced by `train.param_policy` (`default: freeze_loaded`, with `load_existing`
listing the modules that must stay trainable across a requeue).

`bucket_token_multiple` / `bucket_atom_multiple` must equal the crop of the phase (single
bucket = one CUDA-graph shape). At `L=384` that is 384 / 4096.

## What v1.0.0 fixes relative to the pre-1.0.0 code

Two changes are part of the 1.0.0 baseline and must be in before phase 2a is trained:

* **AF3 Eq.4 per-atom loss weighting** in the EDM diffusion loss:
  `w_l = 1 + is_dna*5 + is_rna*5 + is_ligand*10`, gathered from `chain.entity_type` to
  atoms. Before this the EDM path weighted every atom equally, so ligand and nucleic-acid
  coordinates were swamped by the far more numerous protein atoms. (The decoupled diffuser
  already had it; only the EDM stack was missing it.) Exposed as
  `loss.alpha_{dna,rna,ligand}`.
* **QK-norm on the token DiT** (`model.diffusion_module.token_dit.use_qk_norm: true`).
  The atom DiT already defaulted to it; the trunk pairformer deliberately stays off,
  because its weights are frozen phase-1b checkpoints and adding norms there would break
  key compatibility. The token DiT is trained from scratch in phase 2a, so enabling it
  costs nothing.

Already fixed before 1.0.0: the diffusion loss masks with `structure.atom_pos_mask`
(resolved atoms), not `atom_mask` — the latter trained the model to place unresolved
atoms at the origin (the observed origin-collapse of disordered termini / His-tags).

## v1.0.1 — learning rate 1.8e-3 -> 5e-3

`max_lr: 1.8e-3` in v1.0.0 was inherited by mistake. The pre-1.0.0 phase-3 work had already
moved past it: two runs exist on disk and the one carried forward into phase 4 was the
raised-LR one.

| pre-1.0.0 run | max_lr | checkpoints |
|---|---|---|
| `logs/phase3/large_H100_diffusion` | 1.8e-3 | 3 |
| **`logs/phase3/large_H100_diffusion_hlr`** | **5e-3** | **9** (the one phase 4 seeded from) |

The rationale recorded with the raise — *"frozen trunk + from-scratch diffusion head"* —
applies identically to phase 2a, which also trains a scratch diffusion head on a frozen
trunk. v1.0.1 is v1.0.0 with only this one value changed: `phase2{a,b}_diffusion_v101.yaml`,
writing to `runs/v1.0.1/...`. (`phase2b_diffusion_hlr.yaml` is the legacy config that first
carried 5e-3; it is kept for provenance.)

Note that with `warmup_steps: 1000`, `decay_steps: 5e4` and `decay_factor: 0.95`, the step
scheduler never actually decays inside this curriculum: 2a is 50 x 100 = 5000 steps, so the
first decay (step 51000) is never reached and `min_lr: 1e-4` is unused by `LambdaLR`. The
effective schedule is "linear warmup to `max_lr` over 1000 steps, then constant". That is
also why post-hoc checkpoint averaging pays off so well here (see SWA below) — the weights
keep oscillating instead of annealing.

## v1.0.2 — AF3 auxiliary diffusion losses

AF3's diffusion objective (supplement Eq. 6) has three terms; v1.0.0/v1.0.1 have only the
first:

```
L_diffusion = (t^2 + sd^2)/(t + sd)^2 * (L_MSE + alpha_bond * L_bond) + L_smooth_lDDT
```

v1.0.2 adds the other two to the EDM path (they existed only in the decoupled diffuser),
**with AF3's per-stage weights**, and carries the v1.0.1 learning rate (versions are
cumulative). AF3 trains at crops 384 -> 640 -> 768; our 2a is its initial-training stage and
our 2b its fine-tuning stage, so the two phases differ:

| | our phase | AF3 stage | smooth lDDT | alpha_bond |
|---|---|---|---|---|
| initial | **2a** (L=384) | initial training | **1.0** | **0** |
| fine-tune | **2b** (L=768) | fine tuning 1/2 | **0.0** | **1.0** |

Sources: Eq. 6 — *"alpha_bond is 0 for regular training, and 1 for both fine tuning
stages"*; Table 6 row *Polymer-ligand bond loss weight* = 0 / 1 / 1 / 1; and section 5.2 —
*"In the first stage of fine tuning, smooth lddt loss was turned off"* (fine tuning 2
changes things only "in addition", so it stays off). AF3 Eq. 15 also confirms
`alpha_diffusion = 4`, which is our `diffusion_loss: 4.0`.

We deliberately keep `num_augment: 48` in both phases; AF3 drops its diffusion batch size
to 32 for fine tuning, but that is a throughput choice, not part of the objective.

Implementation notes:

* **smooth lDDT** — AF3 Algorithm 27, `miniworld.loss.smooth_lddt.cal_smooth_lddt`.
  Nucleotide atoms use the 30 A inclusion radius, others 15 A. Checkpointed per augment,
  because A x [L_atom, L_atom] pair tensors would otherwise dominate memory at 48 augments.
* **bond loss** — AF3 Eq. 5, MSE of bonded-atom-pair distances vs GT over
  `structure.bond_atom_pairs` (bonded-ligand/glycan-to-parent-chain bonds).
* Both read the same preconditioned prediction the MSE term uses, via the new
  `Diffuser.get_x_pred`; both are distance-based and therefore rigid-invariant, so no
  alignment is needed.
* **Both default to 0.0 in `Client.LossConfig`**, so a v1.0.0 or v1.0.1 job requeued onto
  this code is bit-identical to what it was running; the terms only switch on through the
  `loss: diffusion_v102{,_ft}` groups.

## Version ladder

Each step changes exactly one thing, so the comparisons are clean:

| version | max_lr | smooth lDDT | alpha_bond | isolates |
|---|---|---|---|---|
| 1.0.0 | 1.8e-3 | off | 0 | baseline (currently training) |
| 1.0.1 | **5e-3** | off | 0 | the LR raise |
| 1.0.2 | 5e-3 | **2a: 1.0** | **2b: 1.0** | the AF3 auxiliary losses |

v1.1.0 and v1.2.0 are phase-1 (distogram) data/target changes on top of the v1.0.x
trunk recipe: [v1.1](v1.1-distogram.md) fixes the pseudo-beta targets, weights interchain
pairs 2.0 and forces Ab-Ag crops onto the interface; [v1.2.0](v1.2-msa-sampling.md) draws
PDB MSA depth over the full stored alignment into an 8192-row pool (AF3 SI 2.2) and
re-draws 1024 rows from the pool on every recycle (AF3 SI 3.3).

## Trunk initialisation for phase 2a: SWA_last12

Phase 2a does **not** start from the raw epoch-900 trunk. It starts from the uniform average
of the last 12 phase-1b checkpoints (epochs 845–900, 5-epoch spacing):

```
theta = (1/12) * sum(theta_ep) for ep in 845, 850, ..., 900
```

Measured on 256 held-out items drawn from the PDB-only training stream, against the raw
epoch-900 trunk:

| weights | distogram loss | vs raw ep900 |
|---|---|---|
| raw epoch-900 | 0.87551 | — |
| **SWA_last12** | **0.84529** | **−3.45%** |
| SWA_last8 | 0.84542 | −3.44% |
| decay 0.85 | 0.84543 | −3.44% |
| LCSC (evolutionary search, ~450 evals) | 0.84615 | −3.35% |

SWA_last12, SWA_last8 and the decay 0.75–0.85 profiles are statistically indistinguishable
from each other (paired 2·SE ≈ 0.0009–0.0012 on the same items), and all beat raw ep900 by a
wide, significant margin. A full LCSC search — 180-checkpoint basis, affine coefficients with
negatives allowed, difference parameterisation, EMA-seeded truncation GA — reached a *lower
search objective* than every fixed profile but did **not** improve held-out loss, so the
simplest member of the plateau is the one to ship. Averaging is done per parameter in fp32
and cast back to each parameter's original dtype (the trunk is mixed bf16/fp32).

Do **not** use the online EMA weights from phase 1b: they are degenerate (loss ≈ 3.75, near
the ln(96) = 4.56 random baseline) because the EMA buffer was kept in bf16.

## Naming of historical artifacts

Runs and checkpoints produced before this document use the old numbering. The mapping:

| old name | new phase |
|---|---|
| phase1 | 1a |
| phase2 | 1b |
| phase3 | 2 (2a / 2b) |
| phase4 | 3 (3a / 3b) |

Affected on-disk paths keep their old names — `logs/autoscale/large_H100_full_d16k/...`
(phases 1a+1b), `runs/phase3/...`, `runs/phase4/...`. New runs use
`runs/v1.0.0/<phase>/<name>`.

## Configs

| phase | config |
|---|---|
| 1a | `configs/miniworld/phase1a_distogram.yaml` |
| 1b | `configs/miniworld/phase1b_distogram.yaml` |
| 2a | `configs/miniworld/phase2a_diffusion.yaml` |
| 2b | `configs/miniworld/phase2b_diffusion.yaml` |
| 3a | `configs/miniworld/phase3a_confidence.yaml` |
| 3b | `configs/miniworld/phase3b_confidence.yaml` |

Model / loss / data groups are shared and unchanged: `model: large_swa_pf48{,_diffusion,_confidence}`,
`loss: {distogram,diffusion,confidence}_only`, `data: H100_default`.
