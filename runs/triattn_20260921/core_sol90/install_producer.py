"""Install the qualified independent K/V producer and correct barrier IDs."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys
HERE=Path(__file__).resolve().parent;R=HERE.parent
SRC=HERE/'producer_package';DST=R/'oc/opt_core/kernels/triattn_core_broadcast'
report=(HERE/'sanitize-14326.log').read_text()
assert 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' in report
assert report.count('ERROR SUMMARY: 0 errors')==2
assert report.count('PASS total 5')==3
for name,count in [('check_producerpackaged-full.json',20),('check_producerpackaged.json',40),('check_producerpackaged-scale-full.json',2)]:
 assert len(json.loads((HERE/name).read_text()))==count,name
for ending in [0,1]:
 result=json.loads((HERE/('base3producerids-paired-%d.json'%ending)).read_text())
 assert result['summary']['core']['paired_ratio']<.99,result
manifest=json.loads((SRC/'manifest.json').read_text())
for name,expected in manifest['sha256'].items():
 assert hashlib.sha256((SRC/name).read_bytes()).hexdigest()==expected,name
previous='f44286f39cac8a061c72157b6a6df154cbb82249bc788adea164481d27d489ab'
assert hashlib.sha256((DST/'triattn_broadcast.so').read_bytes()).hexdigest()==previous
backup=HERE/'before_producer_install'
assert not backup.exists(),backup
shutil.copytree(DST,backup,ignore=shutil.ignore_patterns('build','__pycache__'))
for name in manifest['sha256']:
 destination=DST/name;destination.parent.mkdir(parents=True,exist_ok=True)
 shutil.copy2(SRC/name,destination)
shutil.copy2(SRC/'manifest.json',DST/'manifest.json')
subprocess.run([sys.executable,str(R/'core_pipeline/refresh_core_record.py')],check=True)
print('INSTALLED',manifest['sha256']['triattn_broadcast.so'])
