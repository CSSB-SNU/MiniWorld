# Eight GPU continuation, 2026-09-23

Both stages reuse this source/engine snapshot and the existing training directory,
Slurm append log, and W&B run `team_gm/MiniWorld/tapgki9e`.

| Stage | Node / GPUs | PF blocks | Token / atom bucket | Accumulation | Effective batch | Epoch endpoint |
|---|---|---:|---|---:|---:|---:|
| Phase 1a | node02 / 8 H100 | 16 | 384 / 4096 | 32 | 256 | 800 |
| Phase 1b | node02 / 8 H100 | 16 | 768 / 8192 | 32 | 256 | 1000 |

MSA sampled per recycle: 1024; recycle range unchanged. Fabric and static compile,
engine auto dispatch with native OPM/PWA training remain enabled. No CUDA graph
training change is included. Phase 1b checkpoints each of 4 MSA and 16 PF blocks.

`switch8_at_checkpoint.py` waits for a completed checkpoint after epoch 468,
copies and validates it, submits `train8.sbatch` after the old job, submits
`phase1b8.sbatch` with `afterok` of Phase 1a, and then cancels old job 16681.
The job IDs and checkpoint SHA are recorded in `switch8.json`.

Phase 1b loads the final live `last.pt` only after Phase 1a succeeds; its preflight
requires epoch 800 / step 80000. Full model, optimizer, scheduler and EMA state
continue. The target is cumulative epoch 1000, not an additional 1000 epochs.

Original `config.yaml` is historical and intentionally preserved; Phase 1b's
resolved configuration is recorded in `phase1b-resolved.json`. Launch overrides
in the two sbatch files document current accumulation and backend. Existing run
name is preserved despite containing the original `triton-4xh100` label.
