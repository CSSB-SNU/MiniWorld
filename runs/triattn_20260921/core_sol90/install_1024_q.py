"""Publish L1024 specialization only after final-codegen and sanitizer gates."""
from pathlib import Path
import hashlib,json,os,re,shutil,subprocess,sys
HERE=Path(__file__).resolve().parent;R=HERE.parent
SRC=HERE/'q1024_package';DST=R/'oc/opt_core/kernels/triattn_core_broadcast'
previous='4be5b7cdd291150436b9b048318b6cac7977b0db82b9a4f748520d398cfb4c32'
assert hashlib.sha256((DST/'triattn_broadcast.so').read_bytes()).hexdigest()==previous
report=(HERE/'stage-q1024-14856.log').read_text()
assert 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' in report
assert report.count('ERROR SUMMARY: 0 errors')==2
assert report.count('PASS total 5')==3
assert 'Traceback' not in report
for name,count,length in [('check_q1024packaged-full1024.json',20,1024),('check_q1024packaged-full.json',20,768),('check_q1024packaged.json',40,None),('check_q1024packaged-scale-full1024.json',2,1024)]:
 d=json.loads((HERE/name).read_text());assert len(d)==count,name
 if length:assert all(row[0]==length for row in d),name
subprocess.run([sys.executable,str(HERE/'screen_installed_experiment.py'),str(HERE/'build-q1024-package.log')],check=True)
s=(HERE/'build-q1024-package.log').read_text()
m=re.search(r'Function properties for (\S*536870912\S*)\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads\nptxas info\s*: Used (\d+) registers',s)
assert m and list(map(int,m.groups()[1:]))==[0,0,0,128],m.groups() if m else None
for direction in (0,1):
 d=json.loads((HERE/('baseqthree1024-paired-L1024-'+str(direction)+'.json')).read_text())
 assert d['length']==1024 and d['ending']==bool(direction)
 assert d['summary']['core']['paired_ratio']<.9975,d['summary']
 assert d['summary']['block']['paired_ratio']<1.002,d['summary']
 assert all(len(times)==16 for scope in d['rounds'].values() for times in scope.values())
subprocess.run([sys.executable,str(HERE/'compare_1024_q_package.py')],check=True)
manifest=json.loads((SRC/'manifest.json').read_text())
changed={'build_source.py','csrc/m1/m1_binding.cu','csrc/m1/triattn_m1_sm90.cuh','triattn_broadcast.so'}
for name,expected in manifest['sha256'].items():
 assert hashlib.sha256((SRC/name).read_bytes()).hexdigest()==expected,name
 if name not in changed:assert (SRC/name).read_bytes()==(DST/name).read_bytes(),name
backup=HERE/'before_q1024_install';assert not backup.exists(),backup
shutil.copytree(DST,backup,ignore=shutil.ignore_patterns('build','__pycache__'))
def publish(source,destination):
 temporary=destination.with_name(destination.name+'.install-q1024-'+str(os.getpid()))
 try:
  shutil.copy2(source,temporary);os.replace(temporary,destination)
 finally:
  if temporary.exists():temporary.unlink()
for name in manifest['sha256']:
 destination=DST/name;destination.parent.mkdir(parents=True,exist_ok=True)
 publish(SRC/name,destination)
publish(SRC/'manifest.json',DST/'manifest.json')
subprocess.run([sys.executable,str(R/'core_pipeline/refresh_core_record.py')],check=True)
print('INSTALLED',manifest['sha256']['triattn_broadcast.so'],flush=True)
