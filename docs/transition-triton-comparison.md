# Transition: forced Triton versus H100 auto — 2026-09-15

## Result

For the two small token-row inputs, the existing Triton split implementation is
faster than the current H100 auto route. It is 2.60x faster at width384 and 1.27x
at width768, and nearly catches compiled PyTorch at width384. The pair cases
favor H100 auto over the tested Triton alternatives, with modest margins against
the best Triton route. There is no universal winner based only on kernel language.

Milliseconds per forward + backward + gradient reset; lower is better:

| Input / width | Compiled PyTorch | H100 auto | Forced Triton fused | Triton split |
| --- | ---: | ---: | ---: | ---: |
| Pair 128 | 4.70885 | 4.10025 | 4.23536 | 4.42265 |
| Pair 512 | 25.58869 | 30.26088 | 112.87184 | 30.75667 |
| Single 384 | 0.08016 | 0.21434 | 0.19928 | 0.08240 |
| Token 768 | 0.14596 | 0.20069 | 1.18802 | 0.15761 |

Pair inputs are [1,768,768,D]; Single/Token inputs are [1,768,D]. These are
unconditioned Transition modules, not complete ConditionedTransition blocks or
whole-model training. The active model uses pair width128; width512 is the wider
qualification case from the previous audit.

## What each label actually runs

The public implementation="triton" option resolves to the same auto family as
"miniworld". It does not guarantee Triton kernels on H100. This audit bypasses
that dispatch only in benchmark subclasses:

- **H100 auto:** the installed Transition module, with its default native
  settings. Width128 forward launches hand-CUDA WGMMA b2b; width512/768 launch
  CuTe LN/expand/SwiGLU (and eligible fused squeeze/residual). Width384 already
  uses Triton k-tiled b2b forward, plus native CUDA LN backward.
- **Forced Triton fused:** directly calls triton_transition_fused(save_xn=False).
  Disables transition_cuda_b2b, transition_gatebwd_wgmma, transition_lnbwd_cuda,
  and transition_dab_lnbwd. Width128 uses _transition_b2b_kernel; larger widths
  use _transition_b2b_ktiled_kernel.
- **Triton split:** explicitly calls triton_layernorm, then triton_transition,
  then adds the input residual. Dense Triton LN backward is used. It saves the
  normalized input and recomputes SwiGLU activations in backward.
- **Compiled PyTorch:** ordinary reference module with saved activations.

Both Triton alternatives still use cuBLAS GEMMs and some ordinary PyTorch CUDA
operations; these are not claims that every kernel is written in Triton. Native
engine kernels were excluded from the forced routes, verified in actual traces.
Switches use settings.configure, not the stale environment variables in comments.
Each variant's settings are restored before warmup, profiling and capture; later
CUDA graph replay executes those captured kernels regardless of global settings.

## Interpretation

- Pair128: H100 auto is about 3% faster than fused Triton in this run; both beat
  compiled PyTorch. This is a small difference with some per-round variability.
- Pair512: split Triton is about 2% behind H100 auto. Both are slower than ordinary
  compiled PyTorch. Forcing the k-tiled fused path is much worse: its forward
  kernel alone is about88.34ms in the profile.
- Single384: the auto route's k-tiled forward costs about0.114ms. Split avoids
  that kernel and brings the entire training step to about0.0824ms, versus
  compiled PyTorch0.0802ms. The regression is primarily route/shape selection.
- Token768: split is faster than H100 auto, while k-tiled fused is much slower.
  Compiled PyTorch still leads split by about8% in this probe.

The split/fused alternatives have different saved-state and temporary-memory
policies. Same mathematical module and precision do not mean equal activation
storage; see transition-performance-recheck.md for the checkpoint comparison.
This audit changed no production kernel or model dispatch.

## Method and validation

H100 80GB, BF16 inputs/linear weights, expansion4, batch1, TF32 disabled. Norm
parameters stay FP32; wrappers follow each kernel API's normal casts. All variants
use torch.compile(fullgraph=True), with compiler-owned CUDA graphs disabled, and
one manually captured graph for the complete step. Each shape compares variants
in the same process/GPU, in12 alternating-order rounds of20 graph replays.
Separate shapes ran in parallel on allocated GPUs. Compilation and initial tuning
are outside the timed section. No optimizer update is included.

Identical copied weights/input/upstream gradient were used. Outputs, input
gradients and all parameter gradients are finite and pass the existing relative
L2 threshold0.025. The largest observed error across these cases is0.004559
(about0.456%); this is a tensor-norm comparison, not a maximum elementwise bound.
Both wrappers include LN and exactly one residual. Transition has no dropout.

Actual GPU kernel events in Chrome traces verify the selected routes, including
Triton forward and recompute backward and absence of native WGMMA/CuTe/LN
kernels in forced variants. GPU annotation spans are not counted as kernels.
The harness now asserts those route checks when --profile is enabled.

The installed caches were used with their runtime fallback tuning. Several
missing shape entries triggered a heuristic subset of the full configuration
grid. This is an installed-policy comparison, not an exhaustive best-tile search
or a claim about the ultimate performance limit of Triton or H100 kernels.

## Reproduce and artifacts

On an allocated H100 with the project's cu128 environment and engine:

```bash
python scripts/audit_transition_triton.py --width 128 --layout pair --profile --output runs/transition_triton/repeat128.json
python scripts/audit_transition_triton.py --width 512 --layout pair --profile --output runs/transition_triton/repeat512.json
python scripts/audit_transition_triton.py --width 384 --layout token --profile --output runs/transition_triton/repeat384.json
python scripts/audit_transition_triton.py --width 768 --layout token --profile --output runs/transition_triton/repeat768.json
```

Main artifacts under runs/transition_triton:

- pair128_13036.json, pair512_13038.json, single384_13035.json,
  token768_13037.json: complete four-variant measurements and kernel summaries.
- comparison.json: combined main results, with job IDs and original artifact paths.
- Corresponding *_trace.json files: actual kernel traces for each variant.
- pair512_split_13039.json: independent supplementary comparison excluding the
  slow-to-tune fused alternative; PT25.6049ms, auto30.2786ms, split30.8349ms,
  consistent with the main four-way result.

Initial jobs13031/13032/13034 failed during profiling because the new wrapper's
module import collided with the profiler's local summary variable. Job13033 had
the same script issue and was cancelled. Their partial traces are excluded from
all timing conclusions. The collision was fixed before the successful jobs.
Logs are runs/v1.0.1/phase2b/full_wiring_<job>.out.
