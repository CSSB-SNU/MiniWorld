# PyTorch vs upgraded Miniworld-engine — 2026-09-15

## Scope and method

These are module benchmarks, not whole-model or optimizer-step timings. All times
are milliseconds. H100 80GB HBM3, PyTorch 2.10.0+cu128, batch1, BF16 parameters and
inputs, TF32 disabled. TriMul width128; Transition width512 / expansion4. Pair
shape is [1,L,L,D]. Every third token is masked in TriMul, with shared-row dropout
0.25 during training. Transition has no dropout in this API.

All four variants use manual CUDA graph replay, including the uncompiled PyTorch
reference. Thus “uncompiled” excludes Python dispatch overhead from the timed GPU
work; it does not mean a normal Python training-loop wall time. Compiled variants
use torch.compile(fullgraph=True), default Inductor settings with internal CUDA
graphs disabled, so each timed replay launches just the explicitly captured graph.
Four warmups precede capture; five samples of 30 replays are summarized by their
median. Compilation, autotuning, construction and validation are outside timing.

Families ran concurrently on separate GPUs on node02. Each original family job
measured PyTorch, compiled PyTorch and the uncompiled engine in one allocation.
Compiled TriMul engine retries used new H100 allocations after the compile fixes
below. The final bidirectional compiled-engine and compiled-PyTorch comparison
ran together in job13020. No GPU clocks were locked; these are measured samples,
not universal speedup guarantees.

All 72 selected cases completed. The report checks identical parameter SHA-256,
seeded input samples, shapes, precision and dropout probability across each
comparison. Warmup checks finite outputs and all input/parameter gradients.
Training TriMul graph replay must advance dropout RNG; no new Dynamo graph may
appear during timing. Random dropout masks need not be bitwise identical across
compiled and uncompiled implementations. Independent numerical regression against
FP32 checks outputs and all gradients (job13021: six cold-compile cases passed).

## Training (forward + backward + gradient reset)

| Module | L | PyTorch | PyTorch compile | Engine | Engine compile | Compiled PT / compiled engine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| outgoing | 128 | 0.7902 | 0.3629 | 0.2428 | 0.2242 | 1.62x |
| outgoing | 384 | 5.4500 | 3.0372 | 1.2060 | 1.1902 | 2.55x |
| outgoing | 768 | 41.3065 | 21.6140 | 4.3388 | 4.4212 | 4.89x |
| bidir | 128 | 1.0091 | 0.5631 | 0.3473 | 0.3123 | 1.80x |
| bidir | 384 | 7.3293 | 4.5197 | 1.9523 | 2.3636 | 1.91x |
| bidir | 768 | 48.2045 | 37.2653 | 7.3475 | 9.1423 | 4.08x |
| transition | 128 | 1.2946 | 0.7926 | 0.9273 | 0.9229 | 0.86x |
| transition | 384 | 10.1473 | 6.3222 | 7.6011 | 7.6934 | 0.82x |
| transition | 768 | 40.2752 | 25.5673 | 30.1788 | 30.2224 | 0.85x |

## Inference

| Module | L | PyTorch | PyTorch compile | Engine | Engine compile | Compiled PT / compiled engine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| outgoing | 128 | 0.2056 | 0.0879 | 0.0512 | 0.0499 | 1.76x |
| outgoing | 384 | 1.7037 | 0.7519 | 0.2811 | 0.2745 | 2.74x |
| outgoing | 768 | 10.0238 | 6.1799 | 1.0334 | 1.0154 | 6.09x |
| bidir | 128 | 0.2937 | 0.1501 | 0.1216 | 0.0928 | 1.62x |
| bidir | 384 | 2.5758 | 1.3736 | 0.5859 | 0.6911 | 1.99x |
| bidir | 768 | 16.7083 | 11.9640 | 2.1059 | 2.6448 | 4.52x |
| transition | 128 | 0.3976 | 0.2566 | 0.2103 | 0.2090 | 1.23x |
| transition | 384 | 3.3156 | 2.2243 | 1.8913 | 1.9058 | 1.17x |
| transition | 768 | 13.2165 | 8.8640 | 7.6106 | 7.5591 | 1.17x |

## Interpretation

- At L768, compiled engine vs compiled PyTorch training: outgoing TriMul 4.89x,
  bidirectional TriMul 4.08x. This is a different comparison from the earlier
  1.05–1.10x incremental improvement over the old engine.
- Transition width512 training is about 18% slower than compiled PyTorch at L768
  (30.222 vs25.567 ms). Beating uncompiled PyTorch does not establish an advantage
  over Inductor. This benchmark does not isolate the source of the remaining cost.
- Bidirectional engine training is faster without the outer torch.compile at L768
  (7.347 vs9.142 ms). Compilation is not a universal performance improvement for
  a module already composed of native kernels. The compiled cache-dispatch policy
  below differs from eager calibration, so these are distinct execution paths.

## Compile errors found and fixed

The extra cold-compile comparison exposed three gaps beyond prior training-only
and warmed-up checks:

1. The masked-front fake implementation omitted the public save_preact=False
   default. Inference can omit that argument, so its fake now has the same default.
2. Bidirectional inference initialized CuTe imports while Dynamo was tracing.
   Its constructor now initializes those imports when the resolved backend is CuTe.
3. Bidirectional backward queried tensor strides to construct a calibration-cache
   key during speculative autograd tracing, when those strides can be unknown.
   Compiled dispatch now directly selects the policy-allowed cuBLAS variant before
   reading the cache key. Eager execution still calibrates candidates and uses its
   cached winner. This is an explicit compiled-dispatch policy, not a claim that
   cuBLAS is always the fastest candidate.

The fixes are in both copies of the H100 patch and the installed package. Six
cold-compile inference/training regressions passed for outgoing, incoming and
bidirectional TriMul. Pristine engine + four patches reproduces all28 affected
source files; repeat application changes zero files.

## Reproduce and artifacts

Run on an allocated H100 GPU, after engine-setup:

```bash
python scripts/benchmark_engine_h100.py --family outgoing --implementation pytorch --compile --label pytorch_compiled --output pytorch.json
python scripts/benchmark_engine_h100.py --family outgoing --implementation miniworld --compile --label miniworld_compiled --output engine.json
```

Use --family bidir or transition, omit --compile for the other variants, and use
--mode training / --lengths 768 to narrow the run. Each JSON records samples,
parameter hash, compile count and RNG verification.

Local results: runs/h100_vs_pytorch/comparison.json (all72 selected cases),
manifest.json (patch/harness SHA-256), and each named source JSON. Job logs:
runs/v1.0.1/phase2b/full_wiring_<job>.out. PyTorch/uncompiled-engine jobs13014–13016;
final compiled outgoing13017, compiled bidirectional plus PyTorch rerun13020.
Jobs13014/13015 failed only at the later compiled-engine stage; their six-case
PyTorch and uncompiled-engine artifacts completed and remain valid. Job13018's
partial result is excluded. Regression jobs13019 (three inference cases) and13021
(six inference/training cases) passed.

## Transition follow-up

[The paired Transition recheck](transition-performance-recheck.md) reproduces the
width512 slowdown, identifies activation recomputation as the main extra backward
cost, compares checkpoint memory policy, and adds actual-model width probes.
