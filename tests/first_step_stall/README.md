# First-step stall isolation benchmarks

The four scripts here were used to reproduce and pull apart the 15–20 min
first-step wait on `distogram_swa_af3_mix` training. See
`docs/first_step_stall.md` for the analysis; this directory is just the
harness.

All four expect the MiniWorld pixi env and a CUDA GPU. They run against
synthetic inputs — none of them touches the dataloader or the LMDBs.

## `synth_dataloader_launcher.py`

Full-training entry-point wrapper that monkey-patches
`BioMolData.create_ddp_dataloader` to return an infinite iterator of the same
synthetic `Batch` (`_build_precompile_batch`) — no LMDB reads, no workers, no
pickle IPC. Use it in place of the real training launcher to prove the stall
is model-side, not dataloader-side.

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  pixi run -e default torchrun --nproc_per_node=2 --master_port=29512 \
  tests/first_step_stall/synth_dataloader_launcher.py train \
  --config configs/miniworld/config_distogram_swa_af3_mix.yaml \
  --job-name distogram-swa-SYNTH
```

Same "Training Epoch 0 → EMA initialized → 15 min GPU-busy silence" pattern
appears with a synthetic loader — confirmed dataloader is not the culprit.

## `bench_pairformer.py`

`MiniPairformer` only — measures first-step + steady-state cost as a function
of `n_block`, `compile`, `impl`. Shape defaults to `L=384, d_pair=128`
matching the yaml. Baseline for isolating trunk compile cost with no MSA / no
input embedder.

```bash
python tests/first_step_stall/bench_pairformer.py \
  --compile 1 --n_block 48 --token 384 --d_pair 128 --steps 15 --gpu 4
```

## `bench_msa_pairformer.py`

`MiniMSAModule` + `MiniPairformer`. Adds `--impl {pytorch,triton}` and
`--msa_ln_pytorch 1` to replace only the fused-LN inside `OuterProductMean` /
`MSAPairWeightedAveraging` with PyTorch (isolates the fused-LN autotune cost).

```bash
# baseline all-triton
python tests/first_step_stall/bench_msa_pairformer.py \
  --compile 0 --n_msa_block 4 --n_pf_block 48 --N 2048 --L 384 \
  --impl triton --steps 8 --gpu 4

# swap OPM/MPWA LN to pytorch only (keep everything else triton)
python tests/first_step_stall/bench_msa_pairformer.py \
  --compile 0 --n_msa_block 4 --n_pf_block 48 --N 2048 --L 384 \
  --impl triton --msa_ln_pytorch 1 --steps 8 --gpu 5
```

## `bench_embed_pairformer.py`

`InputFeatureEmbedderESMFold2Style` + `MiniPairformer` — same interface as
above plus `--n_embed_block`, `--swa_backend`, and `--n_recycle` to loop the
PF branch N times per step (models the `n_recycle_max` recycling schedule in
`MiniSWAModel`). The input embedder is *not* cast to `bfloat16` here because
the real `MiniSWAModel` does the same (only MSA / PF get `.to(bfloat16)`);
`RelativePositionEmbedding.forward` internally casts to fp32 and would crash a
bf16 Linear if we cast the embedder wholesale.

```bash
# eager, single-pass
python tests/first_step_stall/bench_embed_pairformer.py \
  --compile 0 --n_embed_block 3 --n_pf_block 48 --L 384 --N_atom 4096 \
  --N_msa 2048 --impl triton --n_recycle 1 --steps 6 --gpu 4

# eager + recycle=4
python tests/first_step_stall/bench_embed_pairformer.py \
  --compile 0 --n_embed_block 3 --n_pf_block 48 --L 384 --N_atom 4096 \
  --N_msa 2048 --impl triton --n_recycle 4 --steps 6 --gpu 5
```
