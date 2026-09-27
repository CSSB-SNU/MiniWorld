"""Inspect emitted code; static instruction counts are not latency evidence."""
from pathlib import Path
import hashlib,json,re,subprocess,sys

HERE=Path(__file__).resolve().parent
variant=sys.argv[1]
build=HERE/'build'/('triattn_sol_'+variant)
installed=HERE.parent/'oc/opt_core/kernels/triattn_core_broadcast/triattn_broadcast.so'
expected='9365a4f35e018afed4adba21590ae2c1734bbccf98d1e53904ebfe2537b3e225'
assert hashlib.sha256(installed.read_bytes()).hexdigest()==expected

def kernels(path):
 text=subprocess.check_output(['/usr/local/cuda-12.9/bin/cuobjdump','-sass',str(path)]).decode()
 out={}
 for part in text.split('Function : ')[1:]:
  m=re.search(r'triattn_m1_kernel.*TraitsILi(\d+)',part.splitlines()[0])
  if m:out[int(m.group(1))]=part
 return out

def words(s):return re.findall(r'/\* (0x[0-9a-f]{16}) \*/',s)

binary=build/('triattn_sol_'+variant+'.so')
base=kernels(installed);cand=kernels(binary)
assert set(base)==set(cand)=={0,1024,536870912,1073741824}
out=dict(variant=variant,installed_sha256=expected,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),kernels={})
for flag in sorted(cand):
 s=cand[flag]
 same=words(s)==words(base[flag])
 item=dict(machine_words=len(words(s)),matches_installed=same)
 if flag in (0,1024):assert same,('generic/SAFE changed',flag)
 else:
  ptx=(build/('m1_'+str(flag)+'.ptx')).read_text()
  item['ptx_bias_probes']=ptx.count('mbarrier.test_wait.parity.acquire.cta.shared::cta.b64')
  item['sass']={op:len(re.findall(r'\b'+re.escape(op)+r'\b',s)) for op in ('HGMMA','WARPGROUP.ARRIVE','WARPGROUP.DEPBAR','MUFU.EX2','LOP3.LUT','STL','LDL')}
  item['sync_opcodes']=sorted(set(re.findall(r'\bSYNCS\.[A-Z0-9.]+',s)))
  if '--require-probe' in sys.argv:assert item['ptx_bias_probes']>0,('probe compiled out',flag)
  (HERE/(variant+'-'+str(flag)+'-hot.sass')).write_text(s)
 out['kernels'][str(flag)]=item
(HERE/(variant+'-codegen.json')).write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
