# Engine 2 validation and publication — 2026-09-23

The engine source used by the current node02 training continuation matches commit
`55304541` of SanggeunParrk/miniworld-engine (source-file comparison).
The remote publication branch is `publish/h100-training-20260923`.
The existing v2.0.0 tag is not moved.

The JSON/XML records retain the completed GPU validation and comparison details;
large tensor dumps, checkpoints, compiler caches and profiler binaries are excluded.
The Python scripts are the original site-specific validation harnesses; paths need
adaptation outside this cluster. See [training integration](../engine2-training-connection.md)
and [continuation records](../operations/continuations/20260923/README.md).

Publication-time CPU checks:
- Engine host launch preparation / descriptor caching: 8 passed.
- MiniWorld Adam resume, distogram, cropping and MSA policies: 107 passed, 1 skipped.
- Patch-stack checks across both repositories: 4 passed, 4 skipped.

These checks do not replace GPU validation. The saved GPU records are from the
prior validation runs, not a new GPU run during publication.
