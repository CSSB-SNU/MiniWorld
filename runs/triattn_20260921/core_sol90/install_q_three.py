"""Atomically install qualified three-quarter Q cache; no vector regeneration."""
from pathlib import Path
import hashlib,json,os,re,shutil,subprocess,sys
HERE=Path(__file__).resolve().parent;R=HERE.parent
SRC=HERE/'qthree_package';DST=R/'oc/opt_core/kernels/triattn_core_broadcast'
previous='97634810cdba1e91fe4aecb8bdfa610dd050dd39e7504ffb8babee8c46d4077b'
assert hashlib.sha256((DST/'triattn_broadcast.so').read_bytes()).hexdigest()==previous
report=(HERE/'stage-qthree-14832.log').read_text()
assert 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' in report
assert report.count('ERROR SUMMARY: 0 errors')==2
assert report.count('PASS total 5')==3
assert 'Traceback' not in report
for name,count in [('check_qthreepackaged-full.json',20),('check_qthreepackaged.json',40),('check_qthreepackaged-scale-full.json',2)]:
 assert len(json.loads((HERE/name).read_text()))==count,name
subprocess.run([sys.executable,str(HERE/'screen_installed_experiment.py'),str(HERE/'build-qthree-package.log')],check=True)
for length,suffix in [(768,'-cloned'),(1024,'-L1024-cloned')]:
 for ending in (0,1):
  d=json.loads((HERE/('q-three-interleaved%s-%d.json'%(suffix,ending))).read_text())
  assert d['length']==length and d['ending']==bool(ending) and d['full_bitwise']
  assert d['rounds']==72 and d['subrounds']==8 and d['replays_per_sample']==2
  for scope in ('core','block'):
   v=d['summary'][scope]['baseqthreehalf0']
   limit=(.9975 if scope=='core' else 1.002) if length==768 else 1.005
   assert v['paired_ratio']<limit,(length,ending,scope,v)
   assert v['paired_median_bootstrap95'][1]<limit,(length,ending,scope,v)
for ending in (0,1):
 d=json.loads((HERE/('baseqthreehalf0-paired-qthree-%d.json'%ending)).read_text())
 assert d['summary']['core']['paired_ratio']<.9975 and d['summary']['block']['paired_ratio']<1.002
subprocess.run([sys.executable,str(HERE/'compare_q_three_package.py')],check=True)
manifest=json.loads((SRC/'manifest.json').read_text())
for name,expected in manifest['sha256'].items():
 assert hashlib.sha256((SRC/name).read_bytes()).hexdigest()==expected,name
 # This candidate changes exactly the CUDA Q-cache source and its binary.
 if name not in ('csrc/m1/triattn_m1_sm90.cuh','triattn_broadcast.so'):
  assert (SRC/name).read_bytes()==(DST/name).read_bytes(),name
backup=HERE/'before_qthree_install';assert not backup.exists(),backup
shutil.copytree(DST,backup,ignore=shutil.ignore_patterns('build','__pycache__'))
def publish(source,destination):
 temporary=destination.with_name(destination.name+'.install-qthree-'+str(os.getpid()))
 try:
  shutil.copy2(source,temporary);os.replace(temporary,destination)
 finally:
  if temporary.exists():temporary.unlink()
for name in manifest['sha256']:
 destination=DST/name;destination.parent.mkdir(parents=True,exist_ok=True)
 publish(SRC/name,destination)
publish(SRC/'manifest.json',DST/'manifest.json')
subprocess.run([sys.executable,str(R/'core_pipeline/refresh_core_record.py')],check=True)
print('INSTALLED',manifest['sha256']['triattn_broadcast.so'])
