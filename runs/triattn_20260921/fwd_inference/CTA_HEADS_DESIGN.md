# Four heads in one CTA with head-last bias

The current selected inference path directly writes head-major bias; it already has no
separate bias transpose. This experiment changes the producer/consumer contract to
contiguous BF16 [1,L,L,4]. Front8 computes and stores the four values as one eight-byte
vector. No head-major buffer or transpose is used in the complete inference path.

One CTA owns (outer row, query tile64), with four cooperative warpgroups owning heads0–3.
The warpgroups share one normalized Z tile for Q/gate projection. K/V are projected once
for all heads by the existing native all-head projector, then streamed through one/two
ring slots. A slot contains all four heads' K/V plus one64x256 packed bias tile. One
transaction barrier covers all64KiB. All heads retire the slot before refill.

Each head has independent QK, normalization and PV accumulators. Shared bias is read
with128-bit loads containing two keys' four head values. Unsafe softmax in any head
retries the whole CTA with the existing stable path, keeping shared barrier phases
uniform. Retry counters count head-tiles recomputed, including the other heads in a
retrying CTA. Output stores remain disjoint and retire before scratch reuse.

The planar-bias control has identical CTA/head/query ownership and shared Z projection,
but retains four head-major bias TMA loads and the existing ldmatrix reader. This
separates grouping the heads from the new bias layout/reader implementation. Full FWD
includes KV projection, the selected bias producer, output projection and residual.
Compare against both the current per-length selected entry and original Anthropic.

Candidates stay isolated. Numerical/graph pilots precede performance selection; any
selected implementation also requires FP64, all three sanitizers and full-module tests.

Follow-up controls retain the same producer/consumer layout contract. `repack` loads
packed bias into registers cooperatively, retires all readers, writes head planes back
into the same shared allocation and uses ldmatrix. `scalar` reads only the desired
BF16 head element directly. `indep` is a planar control with per-head TMA completion
barriers and independently retired KV/bias rings, separating all-head synchronization
from four-head CTA ownership.

For a query/key tile across four heads, both old and new paths request32KiB KV plus
32KiB bias. The guaranteed reduction is in logical Q/gate input loads:64KiB becomes
16KiB per query tile. Cache reuse determines whether those saved requests would have
reached HBM. A single CuTe tiled-copy call is not a claim of one hardware TMA instruction.
