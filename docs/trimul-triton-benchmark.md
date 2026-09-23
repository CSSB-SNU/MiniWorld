# Existing Triton TriMul benchmark — 2026-09-15

**Compile-policy follow-up:** these multi-length runs used automatic dynamic
shape promotion. They are not the inspected production training entry's
`dynamic=False` result. The [paired component audit](trimul-training-components.md)
reproduces the dynamic regression and shows CuTe slightly ahead under static
compilation; it also measures the corresponding kernels directly.

These measurements explicitly select implementation="triton". The preceding
TriMul tuning audit selected implementation="miniworld", which routes to the
H100 CuTe/CUDA/Triton composition. Its timings must not be labeled as standalone
Triton-backend module timings.

## Conditions

H100 80GB, BF16 inputs and linear weights, FP32 norm parameters, batch1,
d_pair=d_hidden=128, pair inputs [1,L,L,128]. Holed residue masks; training dropout
0.25 with the module's shared-row semantics and fused residual. TF32 disabled.
Training times include forward, backward and gradient reset; inference uses eval
plus no_grad. There is no optimizer update or whole-model benchmark.

All variants use manual CUDA graphs. Compiled variants additionally use
fullgraph torch.compile with compiler-owned CUDA graphs disabled. Timing is the
median of5 rounds, each with20 graph replays, after compilation/tuning warmup.
Each family ran on a separate GPU; compiled and eager cases within a family ran
sequentially on that same GPU. These are not alternating paired rounds.

## Compiled training — milliseconds

| L | Outgoing | Incoming | Bidirectional |
| --- | ---: | ---: | ---: |
| 128 | 0.22265 | 0.22248 | 0.31685 |
| 384 | 1.20559 | 1.20874 | 2.12151 |
| 768 | 4.44368 | 4.44742 | 8.15858 |

## Compiled inference — milliseconds

| L | Outgoing | Incoming | Bidirectional |
| --- | ---: | ---: | ---: |
| 128 | 0.03814 | 0.03827 | 0.06428 |
| 384 | 0.24827 | 0.24938 | 0.52346 |
| 768 | 0.91302 | 0.92915 | 1.99149 |

## Without outer torch.compile — milliseconds

Manual CUDA graphs are still enabled.

| L | Mode | Outgoing | Incoming | Bidirectional |
| --- | --- | ---: | ---: | ---: |
| 128 | Inference | 0.05565 | 0.05608 | 0.08573 |
| 128 | Training | 0.23977 | 0.24010 | 0.33509 |
| 384 | Inference | 0.26963 | 0.27209 | 0.49484 |
| 384 | Training | 1.20994 | 1.21429 | 1.96412 |
| 768 | Inference | 0.94436 | 0.95243 | 1.79127 |
| 768 | Training | 4.35054 | 4.35545 | 7.46254 |

## Relation to the H100 auto route

The immediately preceding audit measured the same benchmark shapes/settings for
the auto route in separate allocations (jobs13040–13042). At L768:

| Mode / family | Triton backend | H100 auto |
| --- | ---: | ---: |
| Compiled training, outgoing | 4.4437 | 4.4266 |
| Compiled training, incoming | 4.4474 | 4.4190 |
| Compiled training, bidirectional | 8.1586 | 9.1252 |
| Eager training, outgoing | 4.3505 | 4.3366 |
| Eager training, incoming | 4.3554 | 4.3368 |
| Eager training, bidirectional | 7.4625 | 7.3527 |
| Compiled inference, outgoing | 0.9130 | 1.0298 |
| Compiled inference, incoming | 0.9292 | 1.0345 |
| Compiled inference, bidirectional | 1.9915 | 2.6422 |

Single-direction training is essentially tied in these samples. Compiled
bidirectional training favors Triton by about1.12x, while eager bidirectional
training is close (auto about1.5% ahead, too small for a strong conclusion from
separate runs). The Triton path also retains a compiled-versus-eager penalty in
bidirectional training. Tile tuning and whole-module dispatch/layout optimization
are separate questions; these results do not identify one specific cause of every
timing difference.

## Route and execution verification

All36 cases completed: jobs13043/13044/13045 exited0. Every case's actual GPU trace
contains _bidir_front_kernel and no CuTe/CUTLASS native engine kernel. Both module
families report KernelBackend.TRITON. The backend includes cuBLAS GEMMs and ordinary
PyTorch CUDA operations; it does not mean every launched kernel is written in
Triton. The installed Triton cache lookups observed in these jobs were all hits.

All outputs, input gradients and parameter gradients checked finite. Training graph
replay advances dropout RNG. This performance run does not add a numerical error
comparison against PyTorch; it is not an elementwise-accuracy or full-model claim.

The reusable audit now accepts --implementation triton; its existing default
remains miniworld. No production kernel, dispatch or persistent tuning data changed.

## Reproduce and artifacts

On an allocated H100 in the cu128 environment:

```bash
python scripts/audit_trimul_tuning.py --family outgoing --implementation triton --output runs/trimul_triton_benchmark/outgoing.json
python scripts/audit_trimul_tuning.py --family incoming --implementation triton --output runs/trimul_triton_benchmark/incoming.json
python scripts/audit_trimul_tuning.py --family bidir --implementation triton --output runs/trimul_triton_benchmark/bidir.json
```

Original results and per-case traces are under runs/trimul_triton_benchmark:

- outgoing_13044.json
- incoming_13045.json
- bidir_13043.json
- comparison.json: combined results with job IDs
- *_trace.json: all36 actual kernel traces

See [the tuning audit](trimul-tuning-audit.md) for H100 auto-route cache/driver gaps.
