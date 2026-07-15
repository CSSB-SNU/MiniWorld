# First-step stall on `MiniSWAModel` training — what is actually happening

**Context.** On the 2-GPU (and 8-GPU) `distogram_swa_af3_mix` runs (2026-07-15), the
training loop reaches `EMA initialized (874 params)` and then produces no
`log_step` output for 15–20 minutes while GPU util hovers around 70–100%. There
is no `torch._dynamo hit config.recompile_limit` warning, no dataloader queue
starvation, no NaN. It is not a hang — it eventually starts stepping — but the
first step is that long.

This note records what the wait actually consists of, based on isolated
GPU-4..7 experiments run against synthetic inputs on the same B200 rig.

## What we ruled out

- **`swa_rope_attention.py` recompile hell (fixed earlier).** The old MiniWorld
  copy of `SlidingWindowAttention` used `flash_attn_varlen_func` with
  `torch.nonzero` / `torch.unique_consecutive` packing per batch. Dynamo saw a
  fresh packed shape each time and hit `config.recompile_limit=128`. Swapping
  MiniWorld → team-gm's static-shape `flash_window_seqused`
  (`@torch.compiler.disable` + `cu_seqlens = [0, S, 2S, ...]` +
  `seqused_q/seqused_k`) removed the recompile warnings entirely. Fixed in
  MiniWorld `b62831c` / team-gm `4846cfe`.

- **Dataloader.** Repeated the same stall on a `InfiniteSyntheticLoader` that
  yields a pre-generated `Batch` forever — no LMDB, no workers, no pickle. Same
  first-step wait, same GPU util pattern.

- **BioMolData init + worker startup.** A separate ~15 min was seen on the
  first real run, but that came from a different mechanism — see appendix; it
  is *not* the ongoing stall in later runs where `persistent_workers=True`
  keeps the workers alive.

- **MSA species-index Python loop.** A pathological AF3-distillation MSA
  (`|S121` × 12k rows) took 18.4 s per `MSA.__init__` call under CPython's
  `dict.setdefault + list.append`. Replaced with an `np.argsort` + boundary
  scan (msa.py, commit `8344922`). This was a real dataloader hotspot that
  moved throughput 0.58 → 1.31 items/s; it is unrelated to the first-step
  stall.

## What actually causes the wait

Every isolated first step consists of a `miniworld_kernels` **triton autotune
sweep** on the shape it sees for the first time. Autotune has to benchmark the
full config grid (30–50 configs × 10 reps × a few kernels), and at
`bucket_max` shapes (`N_msa=2048, L=384, N_atom=4096`) each benchmark run costs
tens of milliseconds. Numbers below are all `impl=triton`, single B200,
`bfloat16`, batch 1, on freshly-built inductor/triton caches.

### Per-component autotune cost (single first step, eager, no compile)

| stack | first step | steady | delta vs prev |
|-------|-----------:|-------:|--------------:|
| `MiniPairformer` alone (48 blk) | 771 ms | 257 ms | baseline |
| `MiniMSAModule` (4) + `MiniPairformer` (48) | 80.5 s | 125.7 ms | +80 s |
| `InputFeatureEmbedderESMFold2Style` (3) + `MiniPairformer` (48) | 55.3 s | 95 ms | +55 s |

The +80 s from MSA and the +55 s from the input embedder come from *different*
triton kernels; they compound, not overlap. Full trunk (input embedder + 4 MSA
+ 48 Pairformer, all triton, eager) first step ≈ **130–140 s** by
construction.

Inside MSA the ~80 s splits roughly:

| variant | first step |
|---------|-----------:|
| all triton (baseline)                                     | 80.5 s |
| triton, **`OuterProductMean` + `MSAPairWeightedAveraging` fused-LN → PyTorch** | **59.3 s** |

so ~20 s of the MSA cost is specifically the fused-LN autotune inside `OPM` and
`MPWA`. `layernorm_linear_stats` warnings in the training log come from here.
The remaining ~60 s is the OPM main op, the MPWA main op, `Transition` and
`BidirectionalTriangleMultiplication` — all going through
`miniworld_kernels.modules`.

`impl=pytorch` (bypassing miniworld-kernels entirely) collapses the first step
to **0.66 s** with a steady-state penalty of 2.7× (336 vs 125 ms) — confirming
autotune is the entire first-step budget, not model math.

### `torch.compile` on top

| stack | eager step 0 | compile step 0 | Δ |
|-------|-------------:|---------------:|--:|
| `MiniPairformer` alone (48 blk)               | 771 ms | 59.8 s | +59 s |
| `MiniMSAModule` (4) + `MiniPairformer` (48)   | 80.5 s | 86.4 s |  +6 s |
| `InputFeatureEmbedderESMFold2Style` (3) + `MiniPairformer` (48) | 55.3 s | 62.8 s | +7 s |

`torch.compile` adds ~7 s on top of the underlying triton autotune when triton
kernels are the dominant work (MSA + embedder cases). When the model is
kernel-cheap (PF-only) the compile cost dominates instead. Steady state after
compile is 1.9–2× faster.

### Recycling

Wrapping the PF branch in a `for _ in range(n_recycle=4)` loop:

| stack | step 0 | steady |
|-------|-------:|-------:|
| eager, `n_recycle=1` | 55.3 s | 95 ms |
| eager, `n_recycle=4` | **56.2 s** (+1 s) | 365 ms (≈4×) |

Reusing the *same* shape across recycles adds nothing to autotune. **But** the
real training path uses
`n_recycle = self.rng.integers(1, self.n_recycle_max + 1)` — a fresh random
count each step. Under `torch.compile(dynamic=False)` each recycle count
compiles as a distinct graph, so within the first few real steps you will pay
the ~55–140 s autotune × (up to 4) different unique `n_recycle` values seen.
That is the largest single multiplier we identified.

## Why the isolated 130–140 s becomes 15–20 minutes in real training

None of the isolated tests above talk to DDP, don't hit `grad_accum_steps=16`,
don't randomize `n_recycle`, and don't set up wandb/callbacks. Multiplying the
factors that stack in the real script:

- **triton autotune** (input embedder + MSA + PF): ~130–140 s (one shape)
- **× unique `n_recycle` values seen in the first few real steps**: up to 4
  (`n_recycle_max=4`, uniform in `[1, 4]`). torch.compile stores them as
  distinct graphs; each one autotunes the first time it fires.
- **× sync / no-sync branches** from
  `fabric.no_backward_sync(enabled=is_accumulating)` — dynamo compiles two
  branches; each sees the autotune sweep once. With `grad_accum_steps=16` the
  no-sync branch is hit 15× per step (accumulating) and the sync branch once
  (the step boundary), but only two graph compiles.
- **DDP first-collective warmup** + `_sync_module_states` + gradient bucket
  init on the first step: ~1–3 min at this parameter count on 2× B200.
- **wandb setup + callback `on_train_step_start` / `on_train_batch_start` +
  first `MetricsAggregator.log_step`**: seconds, but they don't fire until
  after the first `training_step` returns, which is what we're waiting on.

An 8-way compile fan-out at ~60 s each is 8 min; add ~2 min DDP warmup and the
~2 min baseline autotune floor and the 10–20 min window matches what we see on
the wall clock. There is no single 20-minute call — it is a cascade.

## Mitigations, in order of effort

1. **Pin `n_recycle` for the first few steps** (or the whole warmup schedule)
   so the compiler sees one graph, not four. Simplest change; largest expected
   win.
2. **Pre-build the miniworld-kernels autotune cache** for the target shape at
   run start — see `docs/operations/dispatch-cache.md` (the warning
   `no tuned autotune cache entry for this shape for op 'layernorm_linear_stats'`
   points to the cache-builder). Reuses across runs on the same GPU.
3. **`train.compile=False` for iteration runs.** Steady-state throughput
   halves, but the first step is under one second instead of ten minutes.
   Correct default for short experiments, not for real training.
4. **`impl=pytorch`** — zero autotune (0.66 s first step) at the cost of ~2.7×
   slower steady state. Only useful as a debugging fallback; don't ship real
   training with it.

None of these change the SWA path; the earlier switch to team-gm's
`flash_window_seqused` already removed the *compile-limit* class of stall.

## Appendix — the ~15 min *worker* startup, one time only

Separately from the model-side stall above, the very first run of a fresh
session pays a ~15-min cost that has nothing to do with autotune: the
DataLoader pickles the 5.3 GB `BioMolData` catalog (26 M `DataRecord` items) and
each of the 4 workers per rank spends ~2.5 min `pickle.loads`-ing it. Workers
spawn staggered because the DataLoader triggers each one lazily on its first
`__getitem__`, and IPC over the pipe serializes the 5.3 GB payload. Once the
workers are up, `persistent_workers=True` keeps them alive across epochs, so
subsequent epochs and reruns (that reuse the workers) don't pay this again.

This is out of scope for the model-side stall, but it looks similar from the
outside (log stuck at `Training Epoch 0`), so it is worth naming here.
