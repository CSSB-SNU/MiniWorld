# Four-consumer resident K/V experiments

No serving improvement was found. **SOL90 remains unachieved**. Installed
package `9365a4f35e018afed4adba21590ae2c1734bbccf98d1e53904ebfe2537b3e225`
is unchanged. Its last installed NCU measurements remain L768 SM61.349%
and L1024 SM65.249%; this experiment did not produce a new installed profile.

Nine independent L768/B1 CUDA candidates were built. Each uses a640-thread
CTA with one producer WG and four M64 consumer WGs. Initial96 registers
per thread fund producer32 and consumer112. This fits the actual CTA pool;
the three runnable variants also checked one active CTA at runtime.

| Runnable candidate | Candidate hot kernel | Same-run installed hot kernel | Initial output comparison |
|---|---:|---:|---|
| Full resident K/V, shared Q, two scores |1608.151us|795.629us|Full core/block bitwise equal|
| Resident K, two-stage V TMA, eight bias slots |1003.387us|791.574us|Full core/block bitwise equal|
| Resident K, two-stage V TMA, three scores/two shared P slots |1244.955us|788.050us|Full core/block bitwise equal|

All three preserve the initial FP64 RMS,0.00028883485479432215. Their
three-round CUDA-graph core timings also lose to the installed path. They
are rejected for latency, so no extended numerical qualification,
sanitizers, paired installation qualification, or serving publication was
performed. The initial checks cover one broadcast-mask case; they are not
a claim of broad numerical or race qualification.

The other six candidates are rejected at the compiler gate for C7512
WGMMA serialization. The first fully unrolled three-score/register-Q
candidate additionally spills184 bytes. Six/eight-step loops remove spills,
but packed-P dependencies for descriptor setup and shared Q alone do not
make the three-score/register-P variants fit without serialization.

The hybrid's extra bias slots and changed V transport improve on the fully
resident prototype; both factors changed, so their individual effects are
not isolated. Moving P to shared memory makes three-score overlap compile
cleanly, but its stores and synchronization do not yield a faster kernel in
this design. These results do not justify another slot-count sweep of the
same family.

Jobs14909/14911/14915/14920/14924/14928 are terminal. Job14909 stops with
exit1 at the intended compiler rejection gate; the remaining jobs complete
their respective screens/measurements. No experiment is pending.

Evidence and reproduction:

- [Machine-readable results](core_sol90/resident-four-results.json) include
  module hashes, compiler resources, SASS counts and all three comparisons.
- [Barrier and storage notes](core_sol90/RESIDENT_FOUR_NOTES.md) describe
  consumer ownership, alias phases, V release and shared-P publication.
- [CUDA generator](core_sol90/build_resident_four_m64.py) and
  [shared-P generator](core_sol90/build_resident_shared_p.py) retain all
  candidates in separate modules and namespaces.
- [Result collector](core_sol90/summarise_resident_four.py) verifies the
  installed hash and rebuilds the result document from the saved logs,
  binaries and benchmark JSONs.
