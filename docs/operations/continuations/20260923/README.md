# Production continuation record

These are the exact site-specific launch and preflight records for jobs 16779 and
16780 on node02. They reuse the immutable September 17 source snapshot with
`snapshot-resume.patch` and the Engine 2 source described in the training profile.
Paths are deployment records, not portable defaults; no checkpoints or credentials
are included. `CONTINUATION_8GPU.md` describes state, logging and dependencies.

Do not rerun the switch script: it refers to the already-cancelled job 16681.
For another deployment, adapt paths, predecessor IDs and checkpoint boundaries.
