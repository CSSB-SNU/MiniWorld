"""Install the qualified tuning patch after an immutable cache build has published.

Preserve compatible runtime winners only after verifying byte-identical kernel and
launcher sources. Old unprofiled timings are never attributed to the new benchmark
profile; the next native build measures that profile and preserves its full history.
"""
from __future__ import annotations
import argparse
import ast
import difflib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import subprocess

PATCH = 'miniworld-engine-native-tuning.patch'
COMPAT = 'miniworld-engine-native-tuning-cache-compat.patch'


def digest(data):
    return hashlib.sha256(data).hexdigest() if data is not None else None


def read_optional(path):
    return path.read_bytes() if path.exists() else None


def kernel_identity(package):
    result = {}
    for suffix in ('*.py','*.cu'):
        for path in sorted((package/'kernels').rglob(suffix)):
            if 'notes' not in path.parts:
                result[str(path.relative_to(package))] = digest(path.read_bytes())
    return result


def installer_names(source):
    for node in ast.parse(source).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PATCH_NAMES' for t in node.targets):
            return tuple(ast.literal_eval(node.value)),node
    raise ValueError('PATCH_NAMES missing')


def activate(project, package, *, after_build=None, check_only=False):
    project,package=Path(project).resolve(),Path(package).resolve()
    manifest=json.loads((project/'patches/native-tuning-manifest.json').read_text())
    patch_paths=[project/'patches',project/'libs/team-gm/patches']
    installers=[project/'scripts/apply_engine_audit_patches.py',
                project/'libs/team-gm/scripts/apply_engine_audit_patches.py']
    if after_build is not None:
        if not (Path(after_build)/'job_exit.json').exists():
            raise RuntimeError('cache build has not exited; preserve its publication guards')
    texts=[p.read_text() for p in installers]
    if texts[0] != texts[1]:
        raise RuntimeError('root and team-gm installer stacks differ')
    names,node=installer_names(texts[0])
    if PATCH in names:
        for relative,row in manifest['files'].items():
            if digest(read_optional(package/relative)) != row['after']:
                raise RuntimeError(f'activated source changed: {relative}')
        return {'installed':True,'already_active':True}
    if kernel_identity(package) != manifest['kernels']:
        raise RuntimeError('kernel/launcher sources changed; cannot preserve old runtime winners')
    if manifest.get('verify_native_identity'):
        current=subprocess.check_output([sys.executable,'-c',
            'from miniworld_engine.autotune.native import source_identity; print(source_identity())'],
            env={**os.environ,'PYTHONPATH':str(package.parent)},cwd=project,text=True).strip().splitlines()[-1]
        if current != manifest['before_identity']:
            raise RuntimeError('native source or compiler dependencies changed before activation')
    for relative,row in manifest['files'].items():
        if digest(read_optional(package/relative)) != row['before']:
            raise RuntimeError(f'source changed before activation: {relative}')
    for directory in patch_paths:
        if digest((directory/PATCH).read_bytes()) != manifest['patch_sha256']:
            raise RuntimeError(f'patch changed: {directory/PATCH}')
    spec=importlib.util.spec_from_file_location('native_patch_applier',installers[0])
    applier=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(applier)
    # Preserve only cache files carrying the exact old implementation identity.
    # The reviewed source patch changes no kernel/launcher or math expression.
    changes=[]
    for path in sorted((package/'autotune/data').glob('*/*.json')):
        if path.parent.name not in manifest['native_ops']:
            continue
        old=path.read_text()
        data=json.loads(old)
        if data.get('op_identity') != manifest['before_identity']:
            continue
        if data.get('measurements'):
            raise RuntimeError('unexpected profiled legacy native cache; review migration')
        data['op_identity']=manifest['after_identity']
        data.setdefault('provenance',{})['native_identity_migration']={
            'from':manifest['before_identity'], 'to':manifest['after_identity'],
            'evidence':'byte-identical kernel/launcher tree; tuning policy and bookkeeping only',
            'timing_profile':'legacy-unattributed'}
        new=json.dumps(data,indent=2,sort_keys=True)+'\n'
        relative=str(path.relative_to(package))
        changes.append((path,old,new,relative))
    compat=''.join(applier.unified_file_diff(old,new,'src/miniworld_engine/'+rel)
                   for _,old,new,rel in changes)
    new_names=(*names,PATCH,*((COMPAT,) if compat else ()))
    # Validation needs the exact generated patch, but does not edit the package.
    for directory in patch_paths:
        if compat:
            target=directory/COMPAT
            if target.exists() and target.read_text()!=compat:
                raise RuntimeError(f'compatibility patch already differs: {target}')
            if not check_only:
                target.write_text(compat)
    applier.apply_patches(package,patch_paths[0],(*names,PATCH),check_only=True)
    if check_only:
        return {'installed':False,'check_only':True,'compatible_cache_files':len(changes)}
    for path,old,_,_ in changes:
        if path.read_text()!=old:
            raise RuntimeError(f'cache edited concurrently: {path}')
    for path,text in zip(installers,texts):
        if path.read_text()!=text:
            raise RuntimeError(f'installer edited concurrently: {path}')
    applier.apply_patches(package,patch_paths[0],new_names)
    for path, _, new, relative in changes:
        if path.read_text() != new:
            raise RuntimeError(f'published cache bytes differ: {relative}')
    lines=texts[0].splitlines(keepends=True)
    assignment='PATCH_NAMES = (\n'+''.join(f'    "{name}",\n' for name in new_names)+')\n'
    updated=''.join(lines[:node.lineno-1])+assignment+''.join(lines[node.end_lineno:])
    for path in installers:
        temp=path.with_suffix('.tmp')
        temp.write_text(updated)
        os.replace(temp,path)
    return {'installed':True,'patches':list(new_names),'compatible_cache_files':len(changes),
            'native_identity':manifest['after_identity']}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--package',type=Path)
    parser.add_argument('--after-build',type=Path)
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    if args.package is None:
        import miniworld_engine
        args.package=Path(miniworld_engine.__file__).parent
    try:
        result=activate(args.project,args.package,after_build=args.after_build,check_only=args.check_only)
    except Exception as exc:
        result={'installed':False,'error':f'{type(exc).__name__}: {exc}'}
        if args.report:
            args.report.write_text(json.dumps(result,indent=2)+'\n')
        raise
    if args.report:
        args.report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
