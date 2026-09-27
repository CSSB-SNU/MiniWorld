# Four-head KV materialization and tiled attention

Inference-only experiment following qualified `hot6t`. The four heads share the same
normalized C128 input, but their QK products and softmax denominators remain independent.
Use the dense N128 projection to produce all four D32 K/V heads from a single Z tile.
This is ordinary multi-head attention, not shared K/V or a change to model semantics.

`h4kv_stream1/2/4` materialize BF16 K and V once, then retain the existing fused
Q/gate projection, tensor-core P*V/P*1 attention and stable retry. Each warpgroup
owns a 64-query tile and a private two-slot TMA K/V+bias ring. Q, gate and softmax
state remain local. Static query ownership removes the 6+6+4 tail at L1024.
Adjacent head CTAs preserve the existing Z cache locality. No query CTA recomputes
K/V projections. The projection uses one Z load for both K and V across all heads.

The explicit cost is two L*L*C BF16 intermediate allocations and their writes/reads.
The benefit sought is shorter shared-memory lifetimes, more independently schedulable
query CTAs, and no resident6 register cap. It must beat full `hot6t` and original
Anthropic including the projection kernel and its allocations under CUDA Graph replay.
This is an intentional partial-fusion control, not a claim of lower HBM traffic.

`h4kv_resident6` uses the same all-head projection but retains the old resident KV,
query batching and attention arithmetic. It isolates the choice to materialize
K/V from the attention scheduling change, although projection organization also changes.

Ring ownership: a transaction barrier covers all K/V/bias loads for a slot;
consumers wait before QK. Completed PV WGMMA plus a 128-thread named barrier
retires every reader before that slot is reused. Retry traverses the same ring
and advances its phases. Projection output TMA stores retire before scratch reuse.
Candidates have separate source/binary manifests; production and the original `hot6t`
artifact are untouched. Numerical pilots precede selection; any selected candidate must also
pass retry/graph, full-module FP64 and all three compute-sanitizer tools.

## Locality controls

The first `stream1/2/4` grid has `(head, outer row, query group)` dimensions. A query
group traverses all outer rows before the next group reuses that row's K/V. The
`local1/2/4` controls flatten `(query group, head)` into grid X, keeping outer row in
grid Y. Thus four neighboring heads and every query group for one row are adjacent.
Arithmetic, per-CTA storage, TMA ring and the projection kernel are unchanged.
This is a scheduling/locality experiment, not a claim that CUDA guarantees CTA
execution order. Nsight traffic and complete FWD determine its practical effect.

The narrow `h1kv_stream2` projector computes one D32 head per CTA, with head as the
slow grid dimension. Its paired head4 comparison includes input-cache locality.
`h1adj_local1` also makes the four narrow projection CTAs neighbors, allowing a
stronger comparison that separates a wide projection from that simple grid fix.
Never present the first comparison as proof against the best single-head layout.

`hot4t` and `hot4t_s1` retain the complete on-chip K/V row. At L768/1024 they use four
warpgroups, with two/one bias slots respectively; L384 remains identical to hot6t.
The two-slot variant has 126 registers and zero stack/spills at the larger lengths,
versus hot6t's 80 registers and stack traffic. It trades fewer active warpgroups for
less register spilling and an exact four-batch query partition at L1024.

## Front fusion control

`h4kv_front1` fuses LayerNorm, masked bias and all-head KV projection. TMA loads X
in either orientation, eight lanes normalize each row directly in the swizzled
projection input, and the same normalized Z is used by K/V WGMMA and saved for Q/gate.
This removes a Z reread. Z/bias are bitwise equal to front8, and KV passes sampled
FP64; changed-input graphs and all three front sanitizers pass at128/384/768/1024.
`h4kv_front1v` vectorizes the shared Z accesses to128 bits and limits bias weight
liveness. It improves the front control, but neither wins the whole FWD comparison.

The final experimental default is shape dependent: `h4kv_local1` at384, original
`hot6t` at768, and `hot4t` at1024. See the qualified results for the per-shape gates
and the actual default-entry dispatch/graph checks. Four heads give projection and
input-locality opportunities; the four query warpgroups in hot4t are a separate choice.
