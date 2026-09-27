from pathlib import Path
import hashlib,json,re,subprocess
HERE=Path(__file__).resolve().parent
installed=HERE/'before_qthree_install/triattn_broadcast.so'
if not installed.exists():
 installed=HERE.parent/'oc/opt_core/kernels/triattn_core_broadcast/triattn_broadcast.so'
assert hashlib.sha256(installed.read_bytes()).hexdigest()=='97634810cdba1e91fe4aecb8bdfa610dd050dd39e7504ffb8babee8c46d4077b'
package=HERE/'qthree_package/triattn_broadcast.so'
prototype=HERE/'build/triattn_sol_baseqthreehalf0/triattn_sol_baseqthreehalf0.so'
def get_words(path):
 text=subprocess.check_output(['/usr/local/cuda-12.9/bin/cuobjdump','--dump-sass',str(path)]).decode()
 blocks=re.findall(r'Function : ([^\n]+)\n(.*?)(?=\n\s*Function :|\Z)',text,re.S)
 result={}
 for flag in (0,1024,1073741824):
  matching=[body for name,body in blocks if 'TraitsILi%dE'%flag in name]
  assert len(matching)==1,(path,flag)
  words=re.findall(r'/\* (0x[0-9a-f]{16}) \*/',matching[0]);assert words
  result[str(flag)]=words
 return result
a,b,c=get_words(installed),get_words(package),get_words(prototype)
result=dict(installed_sha256=hashlib.sha256(installed.read_bytes()).hexdigest(),
 package_sha256=hashlib.sha256(package.read_bytes()).hexdigest(),
 prototype_sha256=hashlib.sha256(prototype.read_bytes()).hexdigest(),flags={})
for flag in a:
 result['flags'][flag]=dict(final_matches_prototype=b[flag]==c[flag],
  final_matches_installed=b[flag]==a[flag],final_machine_words=len(b[flag]))
assert result['flags']['1073741824']['final_matches_prototype'],result
assert all(result['flags'][f]['final_matches_installed'] for f in ('0','1024')),result
(HERE/'qthree-package-sass.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
