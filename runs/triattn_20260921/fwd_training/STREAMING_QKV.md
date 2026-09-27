# Streaming Q/K/V/gate projection plus attention

Completed measurements: [QKV_STREAM_RESULTS.md](QKV_STREAM_RESULTS.md). No candidate
was selected. The resource table below records the initial design targets, not
qualified compiled allocations. The original C2 96/64 redistribution was invalid:
its allocated pool was30720 registers, short of32768 requested. The valid
`qkv_stream_c2_late88` uses88/64, exactly30720. `qkv_stream_c4_static` uses93
compiler registers (96 allocated) without redistribution and has zero spills.
All valid streaming launches first pass `audit_stream_registers.py` against SASS.

Baseline: installed checkpoint18006 (`qg_scoped`). Native API and all backward
save layouts stay identical. These candidates are isolated experiments until
correctness, synchronization and complete-workload measurements pass.

The previous resident designs kept a full row of K/V in shared memory. This
design gives a CTA one head and C adjacent64-query tiles. C consumer warpgroups
project their Q/gate tiles in parallel. One producer warpgroup then projects
64-key K/V tiles into a two-slot shared-memory pipeline. Consumers use those
tiles directly; attention never loads Q/K/V from global memory.

| Resource / work | C=2 | C=4 |
|---|---:|---:|
| Warpgroups per CTA | 3 | 5 |
| Consumer register target | 96 | 96 |
| Producer register target | 64 | 64 |
| Register budget after redistribution | 32,768 | 57,344 |
| Shared allocation, including barrier padding | about89KiB | about129KiB |
| Desired CTA capacity per SM | 2 | 1 |
| K/V computation count, L768 | 6 | 3 |
| K/V computation count, L1024 | 8 | 4 |

L64 uses one consumer; L128/256/384 use two. The C=4 control changes only
L768/1024. These lengths divide the chosen query-group size exactly.
Compiler output and SASS register redistribution must be checked before
claiming these capacities. Increased arithmetic is intentional; HBM/L2 traffic
and complete time must be measured, not inferred from the number of kernels.

Ownership and lifetime:

- Each consumer owns one Q/gate output tile and its bias pipeline. It loads Z
  once for Q and gate, preserves BF16 projection rounding, saves both outputs,
  and keeps Q in shared memory.
- A CTA barrier retires all projection readers before their scratch becomes
  bias slots and producer weights. Producer Z has separate storage.
- The producer owns shared K/V slots. `empty[slot]` counts C consumers;
  `ready[slot]` has one producer arrival. Both use alternating phases.
- QK completion in iteration k also retires PV from iteration k-1. Only then
  does each consumer release the preceding K/V slot. The final PV is explicitly
  retired before releasing its slot. This preserves overlap across iterations.
- Only query group0 saves K/V for backward. Other query groups recompute and
  consume private shared tiles, avoiding duplicate global writes and races.
- All score/softmax/PV operations retain ascending key-tile order. No split
  softmax merge, atomics or extra partial-output buffers are introduced.

Primary comparison boundary is Q/K/V/gate projections plus attention, against
the frozen18006 native artifact. Any selection also requires full module FWD,
BWD and F+B with all parameter gradients, changed-input graph replay, independent
FP64 fixtures and memcheck/racecheck/synccheck. Production dispatch remains18006
unless a candidate passes the complete promotion checks.
