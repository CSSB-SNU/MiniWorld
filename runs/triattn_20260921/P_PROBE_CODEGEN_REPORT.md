# P ordering and current CUDA compiler controls

No new serving speedup qualified. Installed package SHA256 remains
`9365a4f35e018afed4adba21590ae2c1734bbccf98d1e53904ebfe2537b3e225`.
Its latest qualified NCU results remain L768 **795.904 us / SM61.349%** and
L1024 **1751.712 us / SM65.249%**. SOL90 is unachieved.

## Packed probability dependency

Job14947 was cancelled before GPU execution because its compiler was stuck
in file-read waits on an overloaded node. The isolated builds finished on
the build host. A source audit also found that the initial experiment's
`kNoProbe` condition compiled out its intended P-dependent barrier test.
That legacy binary is recorded as an invalid probe experiment, not a result.
The builder now enables the fast bias probe by default; reproducing the old
control requires `--legacy-probe-disabled`.

The corrected variants keep native BF16 arithmetic, TMA transfers and
producer/consumer ownership. All eight packed P words feed the address of
the actual bias-ready barrier test. Both emitted code and resource use were
checked before any GPU launch:

| Variant | Emitted PTX probes | Compiler result | Decision |
|---|---:|---|---|
| `basepprobebothv2` | 16 | C7512; 78 WGMMA fences and 78 waits | Reject before GPU |
| `basepprobeevenv2` | 8 | C7520; 64 B stack, 512 B spill stores, 576 B spill loads | Reject before GPU |

The installed fast code has 39 fences and 22 waits. All three experimental
binaries, including the invalid legacy control, preserve the installed
generic and SAFE machine words exactly. These static counts are codegen
evidence, not timing measurements.

## Compiler scheduling comparison

Twelve offline configurations of the current L768 PTX produce three unique
machine-code groups. Levels4/5 and disabled expensive optimizations exactly
reproduce the installed code. Levels0–3 form a second group, and levels6–10
a third. All three groups compile without hot spills or serialization.

Job14964 compared the representatives using actual captured core/block
graphs. Only the hot kernel node was replaced through the CUDA Driver API;
all other nodes and argument values were retained. The harness verifies
one replacement, the 1216-byte argument layout, 512 threads and 191488 bytes
of dynamic shared memory. Full outputs match bitwise before and after
24 balanced timing rounds in both directions, using distinct output clones.

| Representative | Starting core ratio | Ending core ratio | Decision |
|---|---:|---:|---|
| Reassembled identical control (`ru5`) | 1.000194 | 1.000210 | Control |
| Levels0–3 (`ru0`) | 1.005191 | 1.004870 | Slower |
| Levels6–10 (`ru6`) | 1.025604 | 1.027162 | Slower |

Ratios are candidate/serving medians of paired rounds. The baseline complete
core medians were857.014 us starting and853.284 us ending; these include the
core graph's preparation and SAFE dispatch and are distinct from the NCU
main-kernel time above. Neither candidate improved block time either.

A separate native full-Q control changes only the remaining shared QK
operand to its cached BF16 register fragment. All12 offline compiler settings
retain C7512 and78 fences/78 waits. None was launched on the GPU. This is
distinct from the earlier scaled-FP16 Q/K experiment14934, which was also
rejected because conversion and SAFE costs exceeded its small hot gain.

## Evidence and reuse

- [Probability probe resources and codegen](core_sol90/probability-probe-results.json)
- [Current PTX compiler screen](core_sol90/ptxas_current_screen/results.json)
- [Starting graph comparison](core_sol90/ptxas-graph-e0.json)
- [Ending graph comparison](core_sol90/ptxas-graph-e1.json)
- [Full-Q compiler screen](core_sol90/fullqcurrent_screen/results.json)
- [CUDA graph comparison harness](core_sol90/bench_ptxas_graph.py)

The graph harness can screen compatible standalone CUDA cubins without
rebuilding unrelated host translation units. A future winner still needs
normal final-package compilation and the established full qualification
before serving installation. Original expected vectors were not changed.
