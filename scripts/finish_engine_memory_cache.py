"""Merge isolated AdaLN/attention recovery shards and publish compatible cache results.

Run only after the recorded Slurm array jobs exit. Publication keeps the ordinary
source, installation, and patch-stack guards; partial measurements stay usable
without claiming that unfinished units completed.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    args=parser.parse_args()
    run=args.run.resolve()
    metadata=json.loads((run/'metadata.json').read_text())
    import miniworld_engine
    from miniworld_engine.autotune import capture,native
    assert Path(miniworld_engine.__file__).resolve().parent == run/'package/miniworld_engine'
    assert native.source_identity()==metadata['native_source_identity']
    completed=[];incomplete=[];readable=[]
    for filename in metadata['recovery_shards']:
        path=Path(filename)
        try:
            data=json.loads(path.read_text())
        except (OSError,ValueError) as exc:
            incomplete.append({'path':filename,'error':str(exc)})
            continue
        readable.append(path)
        if data.get('_unit_complete') is True:
            completed.append(filename)
        else:
            incomplete.append({'path':filename,'error':'unit did not complete'})
    merged=capture.merge_shards(readable,gpu=metadata['gpu'])
    skipped=list(capture._MERGE_SKIPPED)
    code=int(bool(incomplete or skipped))
    report={'exit_code':code,'completed':completed,'incomplete':incomplete,
            'merged_ops':merged,'rejected_shards':skipped,
            'scope':'Failed L8192 workloads only; other affected shape/dtype profiles need new-source tuning.'}
    (run/'job_exit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    publisher=Path(__file__).with_name('publish_engine_build_cache.py')
    result=subprocess.run([sys.executable,str(publisher),str(run),'--build-exit',str(code)])
    return result.returncode or code


if __name__=='__main__':
    raise SystemExit(main())
