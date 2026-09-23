"""Report staleness for the requested Triton scope, keeping native status separate."""
from __future__ import annotations
import argparse,json
from dataclasses import asdict
from pathlib import Path


def main():
    from miniworld_engine.autotune import cache_status,native
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    rows=cache_status.scan(gpu_substr='H100')
    triton=[r for r in rows if r.op not in native.BUILD_OPS]
    deferred=[r for r in rows if r.op in native.BUILD_OPS]
    report={'scope':'Triton on H100; native tuning deferred','triton':[asdict(r) for r in triton],
            'deferred_native':[asdict(r) for r in deferred]}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(cache_status.format_report(triton,gpu_substr='H100'))
    print(f'{len(deferred)} native cache files outside this build scope.')
    return int(any(r.stale or r.env_matches is False for r in triton))


if __name__=='__main__':
    raise SystemExit(main())
