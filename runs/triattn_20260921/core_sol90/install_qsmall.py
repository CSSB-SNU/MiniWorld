"""Publish the exact Q K16 / small-ones package after recorded qualification."""
from pathlib import Path
import hashlib, json, os, re, shutil, subprocess, sys

HERE = Path(__file__).resolve().parent
R = HERE.parent
SRC = HERE / 'qsmall_package'
DST = R / 'oc/opt_core/kernels/triattn_core_broadcast'
report = (HERE / 'stage-qsmall-14520.log').read_text()
assert 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' in report
assert report.count('ERROR SUMMARY: 0 errors') == 2
assert report.count('PASS total 5') == 3
assert 'Traceback' not in report
for name, count in [('check_qsmallpackaged-full.json',20), ('check_qsmallpackaged.json',40), ('check_qsmallpackaged-scale-full.json',2)]:
    assert len(json.loads((HERE / name).read_text())) == count, name
build = (HERE / 'build-qsmall-package.log').read_text()
assert not re.search(r'C751[245]|C7520', build)
m = re.search(r'Function properties for (\S*1073741824\S*)\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads\nptxas info\s*: Used (\d+) registers', build)
assert m and [int(v) for v in m.groups()[1:]] == [0,0,0,128]
for length in (768,1024):
    for ending in (0,1):
        result = json.loads((HERE / ('baseqrk0ip4small-paired-currentL%d-%d.json' % (length,ending))).read_text())
        assert result['length']==length and result['ending']==bool(ending)
        assert result['summary']['core']['paired_ratio'] < (.995 if length==768 else 1.005), result
        assert result['summary']['block']['paired_ratio'] < 1.005, result
        assert all(len(v)==16 for scope in result['rounds'].values() for v in scope.values())
manifest = json.loads((SRC / 'manifest.json').read_text())
for name, expected in manifest['sha256'].items():
    assert hashlib.sha256((SRC / name).read_bytes()).hexdigest()==expected, name
previous = '6d52ebfc4b43ca8fcbeac7de165d4572cf30d9cd2efe6ee6f63508425dc284fe'
assert hashlib.sha256((DST / 'triattn_broadcast.so').read_bytes()).hexdigest()==previous
backup = HERE / 'before_qsmall_install'
assert not backup.exists(), backup
shutil.copytree(DST, backup, ignore=shutil.ignore_patterns('build','__pycache__'))
def replace_file(source, destination):
    temporary = destination.with_name(destination.name+'.install-qsmall-'+str(os.getpid()))
    try:
        shutil.copy2(source,temporary)
        os.replace(temporary,destination)
    finally:
        if temporary.exists():
            temporary.unlink()
for name in manifest['sha256']:
    destination = DST / name
    destination.parent.mkdir(parents=True,exist_ok=True)
    replace_file(SRC / name,destination)
replace_file(SRC / 'manifest.json',DST / 'manifest.json')
subprocess.run([sys.executable,str(R / 'core_pipeline/refresh_core_record.py')],check=True)
print('INSTALLED',manifest['sha256']['triattn_broadcast.so'])
