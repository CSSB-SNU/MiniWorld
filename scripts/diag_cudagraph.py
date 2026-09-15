"""Compatibility entry point for the measured phase-2 CUDA graph audit.

Output addresses do not determine CUDA graph activity. This diagnostic records
actual launch/capture counts, profiler traces, timings and same-input parity.
"""

from __future__ import annotations

import sys
from datetime import datetime

from audit_phase2_cudagraph import main


if __name__ == "__main__":
    if "--graph" not in sys.argv:
        sys.argv += ["--graph", "trees"]
    if "--output" not in sys.argv:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sys.argv += ["--output", f"runs/v1.0.1/phase2b/cgdiag_{stamp}.json"]
    if "--paired" not in sys.argv:
        sys.argv.append("--paired")
    main()
