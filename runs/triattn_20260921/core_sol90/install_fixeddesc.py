"""Install the descriptor-constant package only after recorded qualifications."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys
HERE=Path(__file__).resolve().parent;R=HERE.parent
SRC=HERE/'fixeddesc_package';DST=R/'oc/opt_core/kernels/triattn_core_broadcast'
report=(HERE/'sanitize-14390.log').read_text()
assert 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' in report
assert report.count('ERROR SUMMARY: 0 errors')==2
assert report.count('PASS total 5')==3
for name,count in [('check_fixeddescpackaged-full.json',20),('check_fixeddescpackaged.json',40),('check_fixeddescpackaged-scale-full.json',2)]:
 assert len(json.loads((HERE/name).read_text()))==count,name
for ending in (0,1):
 result=json.loads((HERE/('basef40fixeddesc-paired-currentL768-%d.json'%ending)).read_text())
 assert result['length']==768 and result['ending']==bool(ending)
 assert result['summary']['core']['paired_ratio']<.995,result
 assert all(len(x)==16 for x in result['rounds']['core'].values())
result=json.loads((HERE/'basef40fixeddesc-paired-currentL1024-0.json').read_text())
assert result['length']==1024 and not result['ending']
assert result['summary']['core']['paired_ratio']<=1.005,result
manifest=json.loads((SRC/'manifest.json').read_text())
for name,expected in manifest['sha256'].items():
 assert hashlib.sha256((SRC/name).read_bytes()).hexdigest()==expected,name
previous='6fe43326e87c87bb5512b39b66437493123e6e68cdc6799bdcc7659f6173f694'
assert hashlib.sha256((DST/'triattn_broadcast.so').read_bytes()).hexdigest()==previous
backup=HERE/'before_fixeddesc_install'
assert not backup.exists(),backup
shutil.copytree(DST,backup,ignore=shutil.ignore_patterns('build','__pycache__'))
def replace_file(source,destination):
 temporary=destination.with_name(destination.name+'.install-fixeddesc-'+str(os.getpid()))
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
