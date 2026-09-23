"""Package validated audit changes as a reversible, hash-checked engine patch."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--validation',required=True)
    p.add_argument('--patch',default='miniworld-engine-trimul-training-audit.patch')
    p.add_argument('--manifest',default='trimul-training-audit-manifest.json')
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    run=args.run.resolve()
    base=run/'baseline_package/miniworld_engine'
    after=run/'package/miniworld_engine'
    spec=importlib.util.spec_from_file_location('audit_patch_applier',root/'scripts/apply_engine_audit_patches.py')
    ap=importlib.util.module_from_spec(spec);spec.loader.exec_module(ap)
    sha=lambda content:hashlib.sha256(content).hexdigest()
    changed={}
    for path in after.rglob('*'):
        if not path.is_file() or '__pycache__' in path.parts or path.suffix=='.lock':
            continue
        relative=str(path.relative_to(after));old=base/relative
        data=path.read_bytes();prior=old.read_bytes() if old.exists() else None
        if prior!=data:
            changed[relative]=(prior,data)
    assert changed
    patch=args.patch
    text=''.join(ap.unified_file_diff(a.decode() if a is not None else None,b.decode(),
                                      'src/miniworld_engine/'+name)
                 for name,(a,b) in sorted(changed.items()))
    for directory in [root/'patches',root/'libs/team-gm/patches']:
        (directory/patch).write_text(text)
    names=tuple(ap.PATCH_NAMES)
    assert patch not in names
    current=root/'.pixi/envs/cu128/lib/python3.10/site-packages/miniworld_engine'
    changes=ap.apply_patches(current,root/'patches',(*names,patch),check_only=True)
    with tempfile.TemporaryDirectory() as directory:
        stage=Path(directory)
        for name,(a,b) in changed.items():
            if a is not None:
                dest=stage/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(a)
        ap.apply_patches(stage,root/'patches',(patch,))
        assert ap.apply_patches(stage,root/'patches',(patch,))==[]
        for name,(_,b) in changed.items():
            assert (stage/name).read_bytes()==b

    def identity(package):
        return subprocess.check_output([sys.executable,'-c',
            'from miniworld_engine.autotune.native import source_identity; print(source_identity())'],
            env={**os.environ,'PYTHONPATH':str(package.parent)},text=True).strip().splitlines()[-1]

    manifest=dict(patch=patch,patch_sha256=sha(text.encode()),
                  files={name:dict(before=sha(a) if a is not None else None,after=sha(b))
                         for name,(a,b) in changed.items()},base_patches=list(names),
                  base_patch_hashes={name:sha((root/'patches'/name).read_bytes()) for name in names},
                  before_identity=identity(base),after_identity=identity(after),
                  run=str(run.relative_to(root)),validation=args.validation)
    for directory in [root/'patches',root/'libs/team-gm/patches']:
        (directory/args.manifest).write_text(json.dumps(manifest,indent=2)+'\n')
    (run/'patch-validation.json').write_text(json.dumps(dict(files=list(changed),
        full_stack_check_only_changes=[str(p) for p in changes],idempotent=True,manifest=manifest),indent=2))
    print('Validated patch:',len(changed),'files; source identity',manifest['after_identity'])


if __name__=='__main__':
    main()
