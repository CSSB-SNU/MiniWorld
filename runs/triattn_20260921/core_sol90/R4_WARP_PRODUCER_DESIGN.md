# Next core architecture: four consumers, one producer warp

Current installed6d52 remains804.416us/SM60.630086%; SOL90 is unachieved.
The544-thread producer-warp design FAILED resource screening in job14424.
The original120-register estimate was WRONG: CUDA's SM90 launch resource
check rounds17 warps to20 (four SM subpartitions), not17. At120 registers
this assumes76800 registers, exceeding65536. See local primary source
/usr/local/cuda-12.9/include/cuda_occupancy.h, lines1460-1505. The compiler
used96 registers with224B stack and1360/1312B spills, plus C7520. No launch.
Do not use setmaxnreg on a partial producer warp to bypass this constraint.

The replacement m128r4coop2 uses512 threads: four complete consumer WGs.
Thread0 services TMA between MMA phases; every warp still participates in
its full WG. Sixteen warps permit128 registers. Job14425 compiled hot to116
registers, zero stack/spills, but C7520 serialized WGMMA. It was NOT launched.
Job14426 rescale/repack fences did not remove C7520. Job14427 adds an
unconditional PV operand fence/WG.AR, removing all warnings/spills (118regs).
Initial results are bitwise but hot931.411us versus installed795.276us;
whole block1477.031us versus1371.593us. This design is REJECTED for latency.

Use M128/N32/LN128/R4, two KV stages, eight bias halves. A single producer
thread issues all Q TMA first, then for each128-key tile: wait KV empty, issue
K/V TMA for all4 rows, issue8 bias halves with their empty waits. Interleave
these operations by tile; independent blocking producer threads within the
same warp would deadlock. Consumers use shared Q (SS), not Q registers.

Reduce consumer registers with two S and two P buffers. Proposed per-step k:

1. wait1 retires QK(k), leaves prior PV(k-2) in flight.
2. Load next bias into free S(k+1), fence P(k-1) and that score, issue QK(k+1).
3. Compute E(k) while QK(k+1) and PV(k-2) run.
4. wait1 now retires PV(k-2), leaving QK(k+1). Reuse its P buffer to pack P(k).
5. Issue PV(k-1) using the P operand fenced before QK; commit it last.

The newly packed P(k) is the other buffer, so it does not require an extra
fence before PV(k-1). Check actual C7519/SASS; do not assume compiler ordering.
Score2=32GPR, P2=16GPR, N40 outputs2=40GPR:88 registers plus temporaries.
Cooperative target128 static registers maximum, zero spills. Drain and exact power-of-two rescale
at established16-half-chunk boundaries. Initialization/final drain, SAFE,
KV-stage release after final PV, uniform means must all remain correct.

Shared storage before padding/barriers:
Q32768 +KV131072 +bias65536 =229376 bytes.
Installed N40 ones8192 bytes would not fit. A per-K32-chunk anchored ones tile
needs only32*32 BF16=2048 bytes, total231424 before barriers. Static assert
<=232448 is mandatory. `build_small_ones.py`/job14423
verified this2KB descriptor view initially in the installed R3 family. The
initial core/block were bitwise, but SAFE spilled and graph measurements
drifted substantially during warmup; no speedup or installation is claimed. Subtract both
KV-stage and K32-column offset from the N32-group leading stride, so both K16
MMAs read immutable ones in the same2KB tile. Never change V global layout.

Use N40 to keep the PV/denominator fusion. For a simpler first prototype, the
old separate8x32 denominator tile fits512B but adds LS instructions; do not
attribute its timing to the fused target. Measure core and whole block against
current installed6d52; retain only real speedups with established accuracy.
