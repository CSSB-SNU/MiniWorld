"""Install only after N40 descriptor fusion has all recorded qualifications."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys
HERE=Path(__file__).resolve().parent;R=HERE.parent
SRC=HERE/'f40_package';DST=R/'oc/opt_core/kernels/triattn_core_broadcast'
report=(HERE/'sanitize-14356.log').read_text()
assert 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' in report
assert report.count('ERROR SUMMARY: 0 errors')==2
assert report.count('PASS total 5')==3
for name,count in [('check_f40packaged-full.json',20),('check_f40packaged.json',40),('check_f40packaged-scale-full.json',2)]:
 assert len(json.loads((HERE/name).read_text()))==count,name
for length in (768,1024):
 for ending in (0,1):
  suffix='' if length==768 else '-L1024'
  result=json.loads((HERE/('basef40desc-paired%s-%d.json'%(suffix,ending))).read_text())
  assert result['length']==length and result['ending']==bool(ending)
  assert result['summary']['core']['paired_ratio']<.99,result
manifest=json.loads((SRC/'manifest.json').read_text())
for name,expected in manifest['sha256'].items():
 assert hashlib.sha256((SRC/name).read_bytes()).hexdigest()==expected,name
previous='893a0a0a32ef03572936fcb7378abca1358a5af13cb075f85aa36a2806c641fa'
assert hashlib.sha256((DST/'triattn_broadcast.so').read_bytes()).hexdigest()==previous
backup=HERE/'before_f40_install'
assert not backup.exists(),backup
shutil.copytree(DST,backup,ignore=shutil.ignore_patterns('build','__pycache__'))
def replace_file(source,destination):
 # Existing processes may still map the previous .so. Replace its directory
 # entry, preserving the old inode for those processes, instead of truncating.
 temporary=destination.with_name(destination.name+'.install-f40-'+str(os.getpid()))
 try:
  shutil.copy2(source,temporary)
  os.replace(temporary,destination)
 finally:
  if temporary.exists():temporary.unlink()
for name in manifest['sha256']:
 destination=DST/name;destination.parent.mkdir(parents=True,exist_ok=True)
 replace_file(SRC/name,destination)
replace_file(SRC/'manifest.json',DST/'manifest.json')
subprocess.run([sys.executable,str(R/'core_pipeline/refresh_core_record.py')],check=True)
print('INSTALLED',manifest['sha256']['triattn_broadcast.so'])
