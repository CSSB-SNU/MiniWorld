# v2.4.0 launch scripts (phase 1a scratch run, phase 2a hand-off)

- `train_phase1a_v240_local.sbatch`: the phase-1a run (`phase1a_distogram_medium_v240_local`, 4 x H100 on node02, data on the node's local NVMe, chained epochs, a crash restarts from the run's own `last.pt`). It ran epochs 0-600 (job 41417, 2026-11-01 .. 11-05) and was stopped after `epoch=0600.pt`.
- `make_seed_v240.py <ckpt> <epoch> <out>`: the phase-2a trunk seed = the online EMA weights of a phase-1a checkpoint, weights only, epoch 0 (`runs/v2.4.0/phase2a/seed_ema600/epoch=0600_ema.pt`).
- `check_2a_v240.py <seed>`: CPU check that `phase2a_diffusion_medium_v240` composes and every seed tensor matches the phase-2a model (the only model-only parameters are `diffusion_module` and `to_token_single_trunk`).
- `train_phase2a_v240_h100.sbatch`: the phase-2a run (frozen trunk, new EDM diffusion head, 200 epochs configured, 4 x H100 on node02; started 2026-11-05 12:50, job 42537).

Notes: both training jobs keep the long `--time` of a training run; every verification job elsewhere in this repo's analysis scripts uses `--time=00:10:00` and one GPU.
`/scripts/slurm/` is git-ignored in this repository, so these live here.
