# Single-head attention pipeline and output/residual fusion

These are independent experiments following the18909 bottleneck analysis. Four-head
attention CTAs remain closed. Production dispatch and the qualified selected paths
remain unchanged until a candidate passes whole-module qualification and paired timing.

Output `outproj_c1` subsequently passed qualification18946 and was connected to the
experimental default for verification18962. Production and training are unchanged.
See [results](PIPELINE_OUTPUT_RESULTS.md) for the final selection and core controls.

The output family computes128 output channels from the128-channel gated attention
tensor, rounds the projection to BF16, adds the original residual in FP32 and rounds
the final result to BF16. TMA loads the residual and stores the result in starting or
ending orientation directly. Input preservation and the inner BF16 rounding are part
of the contract. CTA variants use1/2/4 warpgroups, each owning64 independent rows and
sharing the projection weight tile. No attention heads are combined in this operation.
The eliminated buffer is the post-projection BF16 tensor, not the gated attention input.

The attention family keeps one head and one outer row per CTA with resident KV.
Each64-query tile traverses32-key score tiles. The serial32 control changes only tile
width. The pipelined fast path owns two16-register FP32 score fragments and one reusable
BF16 P fragment. On entry to an ordinary step, the outstanding group queue contains
current QK followed by previous PV/P*1. wait_group1 completes current QK. Next QK is
then issued into the other score fragment before current bias/exp work. Before P or
output/sum accumulators are reused, wait_group1 retires previous PV while leaving next
QK in flight. At the last step there is no newer QK, so a full wait is required.

The first step also needs a full wait because there is no previous PV group. Final
completion retires all GMMA register readers/writers. Named barriers protect each bias
ring slot before TMA refills it. Stable retry remains serial with the32-key arithmetic;
each pass advances transaction phases by the actual number of32-key iterations.
Native/FP64/mask/changed-graph and sanitizer evidence is required; compiling async
instructions alone is not proof that the generated kernel overlaps execution.

Follow-up `pipecopy` controls use one QK accumulator and copy its completed values to
separate scalar work registers before issuing the next QK. Both32-key/four-group and
64-key/two-group versions test the register/loop-width tradeoff. `pipeclean` additionally
drains all earlier WGMMA groups before this copy. These attempts still emit compiler
serialization diagnostics; the intended schedules must not be described as achieved
hardware overlap. Stable retry remains serial for all versions.

Output fusion and the core pipeline are timed independently against the currently
selected per-length entry, then together only if both earn selection. Whole inference
FWD is also compared to the original Anthropic release, including both directions.
