"""Install the qualified exact L768 dispatch and SAFE reset correction."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys
HERE=Path(__file__).resolve().parent;R=HERE.parent
SRC=HERE/'fast_package';DST=R/'oc/opt_core/kernels/triattn_core_broadcast'
report=(HERE/'sanitize-14308.log').read_text()
assert 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' in report
assert report.count('ERROR SUMMARY: 0 errors')==2
assert len(json.loads((HERE/'check_packaged-full.json').read_text()))==20
manifest=json.loads((SRC/'manifest.json').read_text())
for name,expected in manifest['sha256'].items():
 assert hashlib.sha256((SRC/name).read_bytes()).hexdigest()==expected,name
backup=HERE/'before_fast_install'
if not backup.exists():shutil.copytree(DST,backup,ignore=shutil.ignore_patterns('build','__pycache__'))
for name in manifest['sha256']:
 destination=DST/name;destination.parent.mkdir(parents=True,exist_ok=True)
 shutil.copy2(SRC/name,destination)
shutil.copy2(SRC/'manifest.json',DST/'manifest.json')
subprocess.run([sys.executable,str(R/'core_pipeline/refresh_core_record.py')],check=True)
print('INSTALLED',manifest['sha256']['triattn_broadcast.so'])
