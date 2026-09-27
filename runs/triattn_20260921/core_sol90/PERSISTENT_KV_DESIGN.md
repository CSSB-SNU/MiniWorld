# Next bounded experiment: retain all K/V across the six query tiles

Implemented in persistent_kv_body.cuh. Initial candidates are bitwise equal but slower; see STATUS.md entries14372–14375. The text below records the design and does not establish a speedup.
Current installed core remains N40 descriptor fusion, 811.648 us / SM59.943772%.

The R3/M128 pipeline reloads K/V for each of six query tiles. R2/M256 SS and
QRS both compiled cleanly but took about952 us; QRS alone did not help that
short pipeline. The next distinct opportunity is to remove repeated K/V TMA
traffic, K/V ring releases, repeated CTA initialization and teardown.

Target B1,N=S768,H4,D32,BF16,broadcast mask. One CTA owns two pair rows and one
head, and loops over all six M128 query tiles. Grid384*4=1536 CTAs. Use two
224-register consumer warpgroups and one32-register producer warpgroup;
384 threads, require initial allocation>=160 registers per thread. Hold both
Q halves in registers (16 GPR per consumer), four score buffers, two P buffers,
two output accumulators and two separate P*ones denominator accumulators.
The larger224 budget should avoid the R3/QRS C7512 problem.

Shared memory plan:

- K for all768 keys and two rows:98304 bytes.
- V for all768 keys and two rows:98304 bytes.
- One M128 Q tile for two rows:16384 bytes.
- Two64x32 FP32 bias half slots:16384 bytes.
- Separate8x32 BF16 ones tile:512 bytes.
- Barriers/alignment: must statically fit the232448-byte H100 per-block limit.

Use the separate denominator MMA initially. The installed N40 fusion's8KB
ones tile would exceed this tight shared budget. Do not silently assume N40
is inherited. K/V can be loaded with six128-key TMA boxes per row; a768-key
box would exceed TMA box limits. Each immutable K/V tile has a transaction
barrier with two arrivals (K and V); no empty barrier or K/V release is needed.

Producer roles:

1. Warp1 issues Q tile0 and all K boxes, then streams Q tiles1..5 through one
   shared Q buffer. It waits on Q-empty before each reuse.
2. Warp2 independently issues all V boxes, then finishes.
3. Warp0 streams bias halves for all6*48 chunks through two independent slots.
   Each slot has a transaction-full barrier and an8-warp-arrival empty barrier.

Q buffer reuse requires a real data dependency. Consumer QRS loads alone plus
a compiler fence do not prove LDSM has completed. After QK chunks0 and1 have
issued, every Q register has been consumed by an instruction, so the warp
leaders may release Q-empty. Q registers stay unchanged until the query tile's
final WGMMA drain; producers may prefetch the next Q tile meanwhile. Validate
this with full-shape racecheck, not just numerical comparisons.

For each query tile, preserve the installed four-score/two-P order: QK(k+2),
E(k), PV(k-1), with wait2 retiring QK(k) and PV(k-3). Prepack P(k-1) before
QK's hardware fence and retain prefenced PV after E. Bias S(k+3) is loaded into
the freed score buffer at the end. Start with16-chunk periods and drains;
full unrolling/no intermediate drains was slower in earlier tests.

Two bias slots can work because their registers are released after QK issue,
two chunks ahead of E. Prologue: load/issue/release S0,S1, then S2,S3, exp0/
pack/PV0, exp1, init S4, drain. Do not let phantom chunks48/49 consume the next
query tile's bias. For local seq>=48, initialize a harmless score without TMA
wait or bias release, and wrap the unused K descriptor into valid immutable
storage. The real next query will consume those prefetched bias halves.

For a first prototype, reuse the generic prepare/uniform/forward scaffolding
and count its preparation cost. If competitive, adapt TMA bias addressing to
the existing M1 staged format so no extra repack is introduced. Bias has the
same64x32 accumulator order; existing staged halves0..47 of each query tile
are sufficient, with padded halves48..55 unused.

SAFE can recompute all six query tiles of a row-group CTA if any hot tile fails
validation. Store a per-CTA fix flag, zeroed by preparation; only failing warps
write it. SAFE uses the same immutable K/V storage, per-chunk running max and
full drains. This avoids incompatible mapping to the installed R3 fix list.
An all-masked batch still needs the uniform-mean V path. Do not qualify a
prototype that omits these semantics.

Potential later extensions only if the prototype is faster: TMA cluster
multicast for bias, revised bias capacity, and a four-score M256 variant.
None has been implemented here. Do not call a higher utilization metric a
success without a lower measured runtime and qualified numerical behavior.
