# `miniworld.inference` — fast inference path

Independent inference stack that runs on top of the same trained
`miniworld.models.miniworld.Model` weights. It hoists every diffusion-
step-invariant op (atom_pair MLP, token_pair_cond, scatter mapping,
per-step token_single_cond) out of the solver loop, so the per-step
kernel only does the work that actually depends on `x_t` / the
augmentation axis.

Default-on in `casp17/scripts/run_miniworld.py` since 2026-05-22 (after
the T1313 / H2339 v_quick benches). Fall back with
`--legacy-inference` or run yaml `infer.fast_inference=false`.


## API

```python
from miniworld.inference import Predictor

# `client` is an already-loaded miniworld.models.miniworld.Client
# (EMA shadow, Fabric, optional torch.compile all applied)
predictor = Predictor.from_client(client)
cache = predictor.prepare(batch)              # trunk + static cache
output = predictor.sample(
    cache,
    n_samples=8,
    timesteps=100,
    update_rule="x0_centered",
    combine_all=False,
)
# output.atom_pos_pred: (n_samples, L_atom, 3)
# output.distogram_logit: (B, L, L, n_distogram_bins)
# output.model_traj / inter_traj / input_traj: (n_samples, T, L_atom, 3)
```

`PredictorOutput`'s field layout matches the legacy `InferenceOutput`,
so downstream tooling (`batch_to_cif`, the relax / iface phases) needs
no changes.


## Layout

```
miniworld/inference/
  __init__.py       # exports Predictor, InferenceCache, StepSchedule
  cache.py          # InferenceCache + StepSchedule frozen dataclasses
                    # + build_inference_cache + build_step_schedule
  diffusion.py      # diffusion_step(model, cache, schedule, x_t, t_index)
  solver.py         # sample_trajectory(model, cache, schedule, ...)
  predictor.py      # Predictor — orchestrates prepare + sample
```

The training stack (`miniworld.models.miniworld.Client`,
`ModelWrapper`, `Model`, `DiffusionModule`, etc.) is untouched. The
inference path reaches into the model's leaf `nn.Module`s
(`dm.diffusion_conditioning`, `dm.atom_attention_encoder`, ...) and
calls them with cached inputs.


## What gets cached

### `InferenceCache` — built once per `prepare(batch)`

Reused by every solver step and every `sample(cache, ...)` call against
the same batch. Independent of the diffusion timestep `t`, the
augmentation axis `A`, and the noisy coordinates `x_t`.

| field | shape | what it replaces |
|---|---|---|
| `token_pair_cond` | `(B, L_token, L_token, d_pair)` | `DiffusionConditioning` pair branch (full `linear_token_pair` + `pair_transitions`); ran 100x per sample before |
| `token_single_pre_time` | `(B, L_token, d_single)` | `linear_token_single(cat([input, trunk]))` |
| `atom_single_cond_base` | `(B, L_atom, d_single_atom)` | `to_atom_single_cond(atom_single_init)` + token→atom gather |
| `atom_pair` | `(B, L_atom, L_atom, d_pair_atom)` | **The big hoist.** Post `mlp_atom_pair`, fully baked — for L_atom in the 10k range this was step-dominant. |
| `scatter_mapping` | `(B, L_atom, L_token)` one-hot | `F.one_hot(atom_to_token_idx_map)` per step |
| `scatter_count_inv` | `(B, L_token, 1)` | `1 / count.clamp(min=1.0)` |
| `atom_mask` / `token_mask` | `(B, L_atom)` / `(B, L_token)` bool | pre-cast to bool |

Plus the trunk outputs (`token_single_input/trunk`, `token_pair_trunk`,
`distogram_logit`).

### `StepSchedule` — built once per `sample(cache, ...)`

T = `timesteps`. Deterministic given the scheduler config + `timesteps`
+ optional `start_sigma_y`.

| field | shape | meaning |
|---|---|---|
| `sigma_i` / `sigma_hat` / `sigma_next` | `(T,)` | per-step EDM sigmas |
| `sigma_t_hat` | `(T,)` | translation-noise scale at `sigma_hat` |
| `c_in` | `(T,)` | input scaling `1/sqrt(σ² + σ_t² + σ_data²)` |
| `gamma` | `(T,)` | EDM stochastic-injection scale |
| `noise_scale` | `(T,)` | x0_centered residual-noise scale |
| `token_single_cond` | `(T, B, L_token, d_single)` | `DiffusionConditioning` single branch — fourier + add_time + transitions + final_LN, T rows pre-stacked |
| `added_token_cond` | `(T, B, L_token, d_single_token)` | `add_single_token_cond(token_single_cond[t])` pre-applied |
| `time_steps` | `(T+1,)` | original EDM schedule (kept for downstream tooling) |


## Per-step kernel (`diffusion_step`)

Everything below already excludes the cached ops. The full sequence is
in `diffusion.py`:

1. `atom_single_rep = cache.atom_single_cond_base + noisy_to_atom_single_rep(x_t) * x_mask`
2. encoder `atom_transformer(atom_single_rep, atom_single_cond_aug,
   cache.atom_pair, atom_mask)`
3. scatter atoms → tokens via cached `scatter_mapping` /
   `scatter_count_inv` (no `F.one_hot` call)
4. `+= schedule.added_token_cond[t]` (constant lookup)
5. `dm.diffusion_transformer(token_single_rep,
   schedule.token_single_cond[t], cache.token_pair_cond, mask)`
6. decoder: `add_token_info` + `atom_transformer` (reuses `cache.atom_pair`
   and `atom_single_cond_aug`) + `final_denoising`

`@typecheck` is dropped throughout — jaxtyping's runtime checks show up
in flamegraphs of the tight loop.


## Solver (`sample_trajectory`)

Same math as `team_gm.diffusion.decoupled_xpred.solver.XPredDecoupledSolver`,
slimmed:
- All sigmas / c_in / gamma are looked up from `StepSchedule` instead
  of being recomputed.
- Model call goes through `diffusion_step(...)` and takes `t_index`
  directly — no scalar `t_emb` is threaded.
- No `@typecheck`, no `@torch.no_grad` (the function is always called
  under `torch.inference_mode()`).
- R/T noise is still sampled per step (state-free but needs fresh
  randomness — same `sample_rigid` from `miniworld.utils.structure.se3`).
- Supports `update_rule="ode" | "ode_aligned" | "x0_centered"`,
  `init_x0` warm start (flexdock).


## What's NOT cached (and why)

- **Atom transformers' attention** — per-step output depends on
  `atom_single_rep` (which depends on `x_t`), so the attention has to
  rerun. The pair bias `atom_pair` IS cached, just not the output.
- **Token DiT** — same, depends on `token_single_rep` which depends on
  the atom branch.
- **R/T noise sampling** — needs fresh randomness each step, by design.
- **`atom_pair` augmentation** — kept as `(B, L_atom, L_atom, d_pair_atom)`
  with no leading A axis. PyTorch broadcasts it across A in the
  transformer pair-bias path, so there's no need to materialise an
  `(A, B, L_atom, L_atom, d_pair_atom)` tensor.


## Validation

- **`libs/MiniWorld/tests/test_inference_equivalence.py`** —
  single-step `diffusion_step` vs canonical `DiffusionModule.forward`
  on a randomly-initialised module. Passes at `atol=rtol=1e-4`.
- **`casp17/scripts/bench_inference.py`** — A/B benchmark that loads
  the model once, runs both paths back-to-back with the same seed,
  dumps both cifs under `targets/<T>/structures/pred/_bench/...`, and
  reports wallclock + atom_pos_pred RMSD + distogram diff.

Measured 2026-05-22 (ckpt `epoch=1020.pt`, timesteps=100, A6000):

| target | n_tokens / n_atoms | sample (legacy → fast) | speedup | VRAM peak (legacy → fast) |
|---|---|---|---|---|
| T1313_wo_Nterm_A1 | 587 / 4568 | 45.1s → 37.6s | **1.20x** | 8.08 → 6.90 GiB (−15%) |
| H2339 (v_quick) | 1044 / 8112 | 139.2s → 115.7s | **1.20x** | 20.95 → 17.25 GiB (−18%) |

Trunk `distogram_logit` is bit-identical across both paths
(`max_abs=0`). Final `atom_pos_pred` diverges by ~20–40 Å RMSD between
legacy and fast — consistent with chaotic divergence of a stochastic
sampler from per-step float-precision deltas, not a bug (sample-to-sample
RMSD between two random seeds is in the same ballpark). The user
inspected the cifs and signed off on the fast path output.


## Caveats

- **`torch.compile` interaction is asymmetric.** `client.model.compile()`
  wraps the whole `Model` module; the fast path bypasses `Client.sample`
  and calls the diffusion module's leaves directly, so compile coverage
  may differ. Bench wallclocks above are with `compile=False`; if you
  flip compile on for production, validate timing.
- **`from_client` requires a constructed `Client`.** That's how EMA
  shadow / Fabric / compile keep working transparently — `Predictor`
  reads the loaded model from the client and never re-loads the
  state dict.
- **Fixed batch.** Each `InferenceCache` is tied to one `batch`.
  Reusing across batches requires a fresh `predictor.prepare(batch)`.
- **CUDA graph capture not used** (yet). Per-step kernel is shape-static
  except for `x_t`, so it's a good capture target — left as a follow-up.


## Future work (not implemented yet)

- CUDA graph capture of the per-step kernel.
- Persistent `StepSchedule` keyed on `(timesteps, start_sigma_y)` so
  repeated `sample(...)` calls with the same schedule skip the
  schedule build entirely.
- A higher-level "quality" comparison (lDDT / iface energies) against
  legacy on real CASP targets — RMSD alone undersells whether the two
  paths are statistically equivalent.
