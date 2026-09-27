from pathlib import Path
import hashlib,json,re,subprocess
HERE=Path(__file__).resolve().parent
baseline=HERE/'before_q1024_install/triattn_broadcast.so'
if not baseline.exists():baseline=HERE.parent/'oc/opt_core/kernels/triattn_core_broadcast/triattn_broadcast.so'
assert hashlib.sha256(baseline.read_bytes()).hexdigest()=='4be5b7cdd291150436b9b048318b6cac7977b0db82b9a4f748520d398cfb4c32'
package=HERE/'q1024_package/triattn_broadcast.so'
prototype=HERE/'build/triattn_sol_baseqthree1024/triattn_sol_baseqthree1024.so'
def get_words(path,flags):
 text=subprocess.check_output(['/usr/local/cuda-12.9/bin/cuobjdump','--dump-sass',str(path)]).decode()
 blocks=re.findall(r'Function : ([^\n]+)\n(.*?)(?=\n\s*Function :|\Z)',text,re.S)
 result={}
 for flag in flags:
  matching=[body for name,body in blocks if 'TraitsILi%dE'%flag in name]
  assert len(matching)==1,(path,flag)
  words=re.findall(r'/\* (0x[0-9a-f]{16}) \*/',matching[0]);assert words
  result[str(flag)]=words
 return result
old_flags=(0,1024,1073741824);new_flags=old_flags+(536870912,)
a,b,c=get_words(baseline,old_flags),get_words(package,new_flags),get_words(prototype,new_flags)
result=dict(baseline_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),
 package_sha256=hashlib.sha256(package.read_bytes()).hexdigest(),
 prototype_sha256=hashlib.sha256(prototype.read_bytes()).hexdigest(),flags={})
for flag in b:
 result['flags'][flag]=dict(final_matches_prototype=b[flag]==c[flag],
  final_matches_baseline=b[flag]==a[flag] if flag in a else None,final_machine_words=len(b[flag]))
 assert b[flag]==c[flag],(flag,'prototype codegen changed')
 if flag in a:assert b[flag]==a[flag],(flag,'previous codegen changed')
(HERE/'q1024-package-sass.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
