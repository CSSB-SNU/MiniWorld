# Known issues / deferred fixes

Running list of confirmed correctness issues that are **not yet fixed** (or need
a retrain to take effect). Fixed issues are removed from here once the fix lands.

---

## EMA is broken under compiled / CUDA-graph training (does not track live weights)

**Status:** confirmed, NOT fixed. The EMA weights are unusable; keep loading raw.

**Evidence:** on the phase 1b distogram checkpoint (`large_H100_full_d16k`
epoch-900), evaluating over 256 validation items:
- **raw** distogram loss mean **0.864**
- **ema** distogram loss mean **3.749** (≈ ln(n_bins) — a uniform, no-information
  distogram), **worse on 256/256 items**.
- The EMA sits ~0.072 from *every* recent checkpoint (epochs 855–900), which
  themselves span only 0.0265 — i.e. the EMA is not on the recent training
  trajectory at all; it is stuck near the untrained init, with the deep
  pairformer layers (44–47) most off (EMA norms ~30% smaller than raw).

**Likely cause:** phase 1b / phase 2 train with `compile: True` (whole-graph /
CUDA-graph capture). `ModelEMA._update_ema` reads `client.model.state_dict()`,
but under capture the optimizer updates the captured static parameter buffers
in place while the callback appears to read a stale (near-initial) view, so the
EMA accumulates the same early weights every step and never moves.

**Consequences / actions taken:**
- phase 2 must load the **raw** trunk (`model_state_dict`), NOT `ema_state_dict`.
  A change to load EMA was made and then reverted after this measurement — see
  `models/diffusion/client.py` `maybe_apply_param_policy`.
- A separate, defensive fix landed in
  [ema.py](../libs/team-gm/src/team_gm/core/callbacks/ema.py): keep the EMA
  master in FP32 (a BF16 accumulator would round away the decay update). This is
  correct but does **not** fix the compile/stale-read problem above.

**Real fix (when addressed):** make `_update_ema` read the *live* parameters
under CUDA-graph capture (e.g. update EMA from `named_parameters()` tensors that
the optimizer actually writes, or exclude the EMA update from capture / force a
sync), then re-verify EMA distogram loss < raw before enabling EMA loading.

**Files:** `libs/team-gm/src/team_gm/core/callbacks/ema.py`,
`src/miniworld/models/diffusion/client.py`.

---

## Batched diffusion sampling (`--sample-batch`) corrupts the CUDA context

**Status:** confirmed, NOT fixed. Worked around by sampling one trajectory at a time.

**Symptom:** `CUDA error: an illegal memory access was encountered` on a diffusion
step. The context is dead afterwards, so every remaining target in that worker fails
with `AcceleratorError` -- one bad launch costs a whole 190-target shard, not one
target. It killed two full FoldBench sweeps (jobs 12894, 12903).

**Trigger:** drawing all `n_samples` trajectories in one solver call (`shape =
(n_samples, L, 3)` instead of looping at `shape = (1, L, 3)`), which rides the same
leading axis training fills with `num_augment`. It fails the first time a worker
crosses into a **new padding bucket** -- not on the bucket itself, which is why a
single-target probe passes.

**Evidence** (job 12905: replay the exact target sequence that killed rank 4 of 12903
-- two targets in bucket (256, 2048), then 8j71 in (384, 4096)):

| batching | steps | TRIMUL_DISPATCH | result |
|---|---|---|---|
| on | 200 | 1 | illegal access (x2 ranks) |
| on | 20 | 1 | illegal access |
| on | 200 | 0 | illegal access (x3 ranks) |
| **off** | 200 | 1 | **OK** |
| **off** | 200 | 0 | **OK** |

Batching is the necessary condition in every failure observed. It is not the step
count, and it is not `dispatch.pick()`'s per-shape benchmark (`TRIMUL_DISPATCH=0`
skips that and changes nothing). `torch.cuda.empty_cache()` is a second, independent
way to trigger it -- see below -- but only ever in combination with batching: with
sequential sampling, empty_cache at 5 seeds x 200 steps is fine.

**Cost of the workaround:** 2.80 min/target instead of 0.80 (measured over ~100
targets at 147-647 tokens), GPU utilisation 25% instead of 33-56%. That is ~14 h
instead of ~4 h for a full 1,522-target sweep. Worth chasing in the engine: AF3 vmaps
its 5 samples, and at these sizes one trajectory cannot fill an H100.

---

## `torch.cuda.empty_cache()` corrupts the engine's cute kernels

**Status:** confirmed, NOT fixed in the engine. Worked around by never calling
`empty_cache()` in the inference path.

**Symptom:** `CUDA error: an illegal memory access was encountered`, surfacing on the
next diffusion step (or, because CUDA reports asynchronously, on an unrelated later
call). Once it fires the context is dead and every remaining target in that worker
fails with `AcceleratorError`. It killed a full FoldBench sweep (job 12894) on its
first computed target.

**Evidence** (job 12899 / 12902, one configuration per GPU, everything else held
fixed, same target):

| batch | seeds | steps | empty_cache | result |
|---|---|---|---|---|
| yes | 5 | 200 | **on** | illegal access |
| yes | 5 | 200 | **off** | OK |
| no | 5 | 200 | on | OK |
| yes | 1 | 200 | on | OK |

It is not shape-dependent: of six heavy buckets run identically with the cache on,
three died and three did not; with it off, all eight of the heaviest buckets -- up to
21,303 atoms and 2,412 tokens -- passed at 5 seeds x 5 samples x 200 steps.

**Cause:** `miniworld_engine`'s cute kernels keep module-level `_CACHE` dicts and pass
`data_ptr()` of cached tensors (barrier storage, tmem buffers) into the kernel launch.
`empty_cache()` returns those blocks to the driver while the cached pointer lives on.
Sequential sampling survived it because the freed block was usually handed straight
back; batched sampling changes the allocation pattern enough that something else takes
it. That asymmetry is what makes this look like a batching bug at first.

**Consequences / actions taken:**
- `scripts/run_miniworld_diffusion_inference.py` calls `empty_cache()` nowhere, not
  even on the OOM-recovery path where it is the standard remedy: poisoning the context
  would cost the whole shard rather than the one target.
- Fragmentation, the reason it was added, is handled by
  `PYTORCH_ALLOC_CONF=expandable_segments:True` in the launchers. That alone already
  turned the 8pnp OOM (22 GiB transient refused while 45 GiB sat reserved-but-
  unallocated) into a pass.
- A real fix belongs in the engine: cache the *tensors*, or re-read `data_ptr()` per
  launch, instead of baking the address into a cached launch config.

---

## Distogram representative atoms — fixed in v1.1 work

**Status:** fixed on 2026-09-16. Canonical proteins use CB (CA only for glycine),
purines use C4 and pyrimidines use C2. Missing designated atoms are masked,
without switching to a different atom. Atomized ligands/noncanonical residues
use their own atom. This also corrects the representative feature consumed by
confidence. Parameter/checkpoint shapes are unchanged.

The earlier description of the distogram fallback as an all-atom minimum was
outdated: the active single-structure helper selected the lowest-index marked
atom. The fix removes that atom-order-dependent target for canonical residues.

See [implementation, weighting, cropping and validation](v1.1-distogram.md).

---

## Training MSA never saw rows past 2048 (PDB) — fixed in v1.2.0

**Status:** fixed on 2026-09-17 for the `pdb` source; distillation sources are 2048-capped
in storage and unchanged.

`sample_depth="uniform"` drew k ~ U[1, min(n, 2048)] and took the first k rows. The PDB a3m
store holds up to 16k rows (88% of entries exceed 2048), so rows past 2048 were never used and
the median fed depth was ~1024 — while AF3 draws U[1, n] over the full alignment, shuffles, then
crops. v1.2.0 adds `sample_depth="af3"` (full-depth draw, random per-chain crop to the 2048
budget) selected per source via `MSAConfig.sample_depth_by_source`. Shapes and buckets are
unchanged. v1.2.0 also adds the AF3 SI 3.3 model-stage step: every recycle draws a fresh
1024-row subset from the pool (`trunk.msa_subsample_per_recycle`), so the MSA module no longer
reuses one embedded pool across cycles. See [v1.2-msa-sampling.md](v1.2-msa-sampling.md).
