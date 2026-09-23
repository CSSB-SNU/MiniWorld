"""Build the SM90 module plan and Triton driver complement, deferring native tuning.

Uses the engine's normal incremental builder, source checks, merge and coverage
checks. Native kernels may execute through existing module dispatch, but their
dedicated configuration search jobs are excluded.
"""
from __future__ import annotations

import sys
import argparse
from unittest.mock import patch


def main(argv=None):
    from miniworld_engine import cli
    from miniworld_engine.autotune import derive, native

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--driver-ops', help='Comma-separated driver kernels to revisit; omitted means all Triton drivers')
    options, remaining = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    requested = set(options.driver_ops.split(',')) if options.driver_ops else None

    original = derive.uncovered_kernels

    def triton_complement(*args, **kwargs):
        names = original(*args, **kwargs) - set(native.BUILD_OPS)
        if requested is not None:
            if requested - names:
                raise ValueError(f'Not in the Triton driver complement: {sorted(requested - names)}')
            names &= requested
        return names

    print("Scope: Triton module coverage and Triton driver tuning; native tuning deferred.",
          flush=True)
    with patch.object(derive, "uncovered_kernels", triton_complement):
        return cli.main(["build", "all", *remaining])


if __name__ == "__main__":
    raise SystemExit(main())
