# Resident K/V, four M64 consumers

Experiment `resident4m64s3qr`, first job14909. Installed package remains
`9365a4f35e018afed4adba21590ae2c1734bbccf98d1e53904ebfe2537b3e225`.
This is a standalone L768/B1 prototype, not a serving path.

Each CTA has one producer WG and four consumer WGs. Consumers `wg/2` share
one pair row's K/V; `wg%2` chooses one M64 half of each M128 query tile.
The CTA computes six query tiles while retaining all six K/V stages.
This changes the old resident2 design's two M128 consumers into four M64
consumers; the previous resident2 design already had QK lookahead.

Initial register budget: 640 threads * 96 registers = 61440. After full-WG
reallocation: 128 * 32 + 512 * 112 = 61440. The build rejects spills and
compiler MMA serialization. The host also checks actual function registers
and one active CTA at the requested dynamic shared-memory allocation.

Shared storage: K96KiB, V96KiB, bias32KiB, ones2KiB. The Q16KiB tile
aliases two bias half-slots; alignment/barriers must fit 232448 bytes.
Each consumer owns one Q register fragment, three score fragments, two P
fragments, and an N40 accumulator (32 output columns plus repeated row sum).

Barrier ownership and phases:

- Each resident K/V stage's transaction barrier has two arrivals: K and V
  producer warps each issue both rows. Consumers wait phase0 before use.
  K/V is never overwritten inside a CTA.
- Q transaction barrier has one producer arrival per query tile. Its phase
  is query-tile parity. Q producer waits query-done for the previous tile.
- Q-consumed barrier has 16 consumer-warp arrivals. Every warp's release
  address depends on all eight Q register words and opaque host zero.
  Bias producer waits this barrier at the start of every query tile.
- Bias sequence is `2*key_chunk+query_half`, slot `sequence%4`, phase
  `(sequence/4)%2`. Each half-slot is shared by two consumers: eight warp
  arrivals release it. There are 48 half-slots per query, so each ring slot
  is used 12 times and phase0 starts the next query naturally.
- Query-done has 16 consumer-warp arrivals after the last WGMMA drain and
  output stores. Only then can the next Q tile overwrite the bias aliases.
- The two-group body packs E(k-1), loads bias and issues QK(k+2), computes
  E(k), then commits PV(k-1). `wait_group 2` retires all earlier groups.
  Initial QK0/1 are fully drained; a zero-P PV makes the first body's group
  count identical. Three scores and two P buffers use compile-time indices.
- Drains at key chunks7/15/23 precede exact power-of-two denominator/output
  rescaling; the one exponentiated but unpacked score is rescaled too.

Accuracy, sanitizers and performance remain unqualified until measured.

## Follow-up variants

Jobs14911/14915: six-step three-score and eight-step two-score register-Q
controls have zero spills but C7512. Adding packed-P dependencies to K/V
descriptor addresses does not fix it. All four were rejected before GPU.

Job14920: shared Q with two bias slots and three scores still has C7512.
The two-score version is clean96 initial registers, 232448B shared, one
resident CTA. Initial full core/block results are bitwise equal and FP64
RMS is unchanged, but hot1608.1512us versus795.6288us. Rejected.

Job14924 hybrid: keep K96KiB resident, stream V through two32KiB-total
stages, retain Q16KiB, and expand bias to64KiB/eight half-slots. Clean96
initial registers,216064B shared,one CTA. Initial full core/block bitwise
equal,unchanged FP64RMS. Hot1003.3872us versus791.5738us: still rejected.
This combines changed bias depth with changed V transport, so it is not an
isolated measurement of bias-ring depth.

Hybrid V ownership: each transaction barrier has one V producer arrival
and16384 expected bytes for both rows. Empty barriers have16 arrivals,one
per owning consumer warp. V iteration g=6*query_tile+key_tile,slot=g%2,
phase=(g/2)%2. In the hot loop, wait_group1/2 retires PV(seq-3), so key
chunks3/7/11/15/19 release at loop steps6/10/14/18/22. The finalPV23 is
fully drained before its release. SAFE releases each tile after its last
PV drains. The producer can refill only after all16 owning warps release.

Next shared-P variant: three scores, two P shared tiles per consumerWG,
six bias half-slots. K96+V32+Q16+P32+bias48+ones2=226KiB before alignment
and barriers. Each P publication uses STSM, fence.proxy.async.shared, and
a128-thread named barrier8+wg. wait_group2 retires the previous use of
the same P slot before it is overwritten. The hot path no longer retains
two register-P tiles. This is not installed or qualified.

Job14928 completed: shared-P variant is clean96/232448B/oneCTA and initial
full core/blockBITWISE with unchanged FP64RMS. Hot1244.9552us versus
788.0504us. Rejected for latency; no extended qualification or install.
resident-four-results.json includes all9 candidates; no job is pending.
