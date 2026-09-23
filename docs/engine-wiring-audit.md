# Complete engine wiring audit — 2026-09-15

## Scope and changes

The shared `miniworld_engine` implementation selection now reaches the active
MiniWorld SWA model's embedding, template/MSA, Pairformer, diffusion and confidence
paths. Explicit PyTorch overrides remain available. Checkpoint parameter names
and tensor shapes are preserved.

| Area | Connection or correction |
| --- | --- |
| Embedding / template / recycling | Engine RMSNorm and LayerNorm; SWA receives the shared implementation selection. |
| MSA | OPM and pair-weighted averaging pass the selection into their normalizations. |
| Diffusion conditioning | All four Transition modules use the selected engine implementation. |
| Diffusion atom/token paths | Standalone LayerNorm, attention pair LayerNorm, and query/key RMSNorm use engine dispatch. |
| ESMFold2-style SWA atom blocks | Existing RMSNorm + modulation and SwiGLU kernels are connected, retaining the original six modulation chunks and parameter layout. |
| AF3-style RoPE SWA alternative | Updated constructors to the installed engine API; selection reaches AdaLN, RMSNorm, attention and ConditionedTransition. |
| Confidence | Inherits shared selection unless explicitly overridden; single attention uses the engine QK-plus-pair-bias kernel; single/pair Transitions are connected. |
| Native mixed precision | Projection weights are differentiably cast for BF16 kernels; FP32 master parameters retain gradients. |
| Native triangular batching | B=1 native kernels are invoked per sample to preserve the public batched training and inference contract. |

RMSNorm preserves the installed PyTorch 2.10 behavior for `eps=None`: BF16/FP16
use the FP32 accumulation epsilon. Using BF16 epsilon here caused a substantial
error on small inputs and is covered by the regression tests.

This connects existing compatible kernels. It does not replace ordinary geometry,
tensor layout operations, or unsupported complete OPM/MSA operations with a
different algorithm. The engine's standalone SWADiTBlock has a different
conditioning formula and is not a compatible replacement for this model's SWA
block. Magnitude-preserving FFNs retain their original weight normalization;
legacy AF3 rotation/full-MP constructor combinations, unsupported by the pinned
engine, raise explicit errors.

## Real model verification

H100 80GB, PyTorch 2.10.0+cu128, BF16 autocast, epoch-93 model checkpoint, L768,
8192 padded atoms, MSA depth 2048, two recycles. Evidence:
`runs/v1.0.1/phase2b/wiring_runtime_12981.json` and its trace/log.

- No called audited module was configured to use the PyTorch default.
- The profiler recorded native LayerNorm, RMSNorm, RMSNorm/modulation, SwiGLU,
  Flash SWA, augmented attention, AdaLN and conditioned-transition operations.
- All **685** trainable diffusion/projection parameter tensors received finite
  gradients. The probe used a squared-output loss; it was not a production loss,
  optimizer update, full training run, or structure-quality evaluation.
- Same-input relative L2 difference before/after these additional connections:
  single **0.0746%**, pair **0.0115%**, valid-atom coordinate output **0.6234%**.
  Maximum coordinate difference was **0.107421875** in model output units.
  These BF16 fusion differences are distinct from CUDA Graph ON/OFF differences;
  the earlier approximately 0.01% Graph result is not a bound on this change.

The before-wiring comparison temporarily restores the prior normalization,
modulation and FFN backends on identical weights, tensors and diffusion noise.
It retains the already connected Flash attention and conditioning transitions.

### Regression and CUDA Graph results

- RMSNorm, SWA modulation/FFN and confidence attention parity/gradient checks:
  11 tests passed in job 12983. Its initially failing B=2 confidence inference
  exposed the native batch restriction described above.
- Final confidence tests, job **12987**: **2 passed**, covering batches 1 and 2,
  four blocks, FP32 master parameters with BF16 autocast, all parameter gradients,
  and training/inference output agreement. Residual output weights are randomized
  so zero initialization cannot hide a disconnected branch.
- Existing compilation/AMP/wiring regressions and the ESMFold2 SWA test,
  job **12984**: **21 passed**. The AF3 alternative is checked separately in job
  **12988**: **1 passed**. Together the three test files cover 34 distinct cases.
- Fatal Python lint, byte compilation and whitespace checks passed. Two existing
  single-name jaxtyping dimension annotations in the template module require a
  per-file F821 exclusion with this installed Ruff version.

Full-model job **12985**, `fullgraph=True`, recorded **one replay per conditioning
call**, six replays for six measurements, three CUDA graph launches for three
profiled calls, and no graph-break/unimplemented entries. After warmup there were
no measured recaptures. Same-input OFF/ON, changed-input OFF/ON, and repeated calls
within each mode were bitwise equal in this final run. Changed coordinates
affected the output. Paired medians were approximately **0.4831 s OFF / 0.4736 s
ON** (1.0202x); these measure only the frozen conditioning trunk.

This is not a universal bitwise-equality guarantee. Earlier job 12979 on the new
wiring showed a 0.222% pair relative-L2 difference for changed-input OFF/ON; it
did not reproduce in 12985, including the added repeat and valid-token checks.
The source of that earlier difference has not been isolated. Both runs produced
finite outputs. Evidence is in `runs/v1.0.1/phase2b/fullgraph_12985/trees.json`
and `fullgraph_12979/trees.json`.

## Reproducibility

### Integration with the shared team-gm branch

The MiniWorld consumer now references `team-gm` commit `aaf9356` on
`exp/miniworld`, including the earlier standalone/FoldForge history at `dbc204f`.
The integrated source was independently retested in H100 job **12989**:
**34 tests passed**, and the real L768 probe produced finite output with all
**685 parameter gradients present and finite**, with no audited PyTorch-default
module calls. Evidence is in `runs/publish_validation/runtime_12989.json` and
`runs/v1.0.1/phase2b/full_wiring_12989.out`.

Python 3.10 support is retained for MiniWorld core; the shared repository's
Biohub ESMFold2 processor dependencies remain gated to Python 3.12. The matching
Pixi lock adds safetensors and updates team-gm metadata without changing existing
package versions. Both the Pixi and team-gm uv locks passed consistency checks.
See [the shared-branch integration record](../libs/team-gm/docs/miniworld-integration-20260915.md).

The dependency remains pinned to engine commit
`1bc0803e3b2fef3b963fdc383e090c0adcccdb43`. Local engine fixes are stored in the
five patch files under `patches/`, rather than depending solely on site-packages edits.
After reinstalling that dependency, run:

```bash
pixi run -e cu128 engine-setup
```

The patch helper verifies compatibility before applying changes and tolerates
repeat execution. Applying all patches to a pristine copy of the pinned source
and comparing the resulting files against the installed engine checks that the
fixes survive reinstallation (**28 affected engine files matched** after the H100 upgrade, with repeat
application verified). Team-gm caller changes also live in this workspace's
`libs/team-gm` submodule and must be included when publishing these changes.

Regression coverage is in `tests/test_engine_full_wiring.py`,
`tests/test_engine_wiring.py` and `tests/test_engine_cudagraph_regressions.py`.
The real model check is reproducible with `scripts/audit_engine_wiring.py
--assert-engine --compare-before-wiring`, supplying the checkpoint, catalog
snapshot and output path.

The earlier CUDA Graph investigation and intermediate wiring results are in
[phase2-cudagraph-bias-audit.md](phase2-cudagraph-bias-audit.md).

H100 kernel classification, fusion changes and new measurements are documented in
[h100-kernel-upgrade.md](h100-kernel-upgrade.md).
