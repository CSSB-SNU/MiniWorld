# Bounded backward follow-up

The current training package remains checkpoint18246: optimized training
forward plus backward `rs8_async_q4` / `rs_softmax_overlap`. The new inference
forward has no training saves and is not the forward used in these measurements.
All48 package source/binary hashes are verified and recorded in baseline.json.
No production/autograd edits are part of the pilot.

Keep the existing fusion boundaries: gate/output/delta, row-group8 dK/dV+bias,
projection/LN/residual and shared-input weight gradients. The dK/dV partial
buffer and final reduction remain accounted for. The historical profile puts
dK/dV near half of backward and dQ around one quarter at L1024; attribution19245
refreshes both against the actual current training path.

Three native CUDA candidates are the initial scope:

* `reuse_q`: dQ loads the resident64x32 Q tile into8 packed registers per thread
  once, then uses RS QK across all key iterations. Existing TMA K/V/bias staging
  and barriers remain. No additional HBM buffers or math changes.
* `reuse_qdo`: also retains dO in8 packed registers for RS dP. The additional
  register lifetime must fit the128-thread/four-block launch budget without
  harmful spills. This applies the inference Q-reuse idea to both repeated
  left operands.
* `rs8_prob_overlap`: dK/dV commits score and dP as separate WGMMA groups,
  waits for score, computes exp while dP is pending, then waits before reading
  dP. Delta loads occur after this wait. The existing producer and two consumer
  warpgroups, asynchronous bias reducer, four Q/dO slots, all reader barriers,
  BF16 dS and deterministic FP32 bias reduction order remain.

These changes target on-chip data reuse and waits. They do not claim reduced
global intermediate traffic. Persisting all four rows of K/V in dK/dV registers
would add64 registers/thread to a224-register consumer and exceed its budget,
so that direct extension is excluded from this bounded pass.

First require native bitwise equality, resources and paired target-length
timings. All three controls receive complete backward and F+B checks; only
`reuse_qdo` receives full qualification. Selection requires all input/parameter gradients, changed graphs, independent
FP64, relevant full-shape sanitizers and actual candidate dispatch attribution.
Keep the optimized forward unchanged. SOL is distinct from speedup.
