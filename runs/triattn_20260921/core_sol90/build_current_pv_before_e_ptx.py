"""Recheck early PV on current N40/three-Q source, plus native full-Q.

The older SS-Q version had parity. Current P packing already precedes QK;
moving PV directly after QK can shorten the packed-P-to-use interval and
let its fence serve both operations. Two commits and wait2 remain unchanged.
No zero-dependency tricks, producer changes, extra waits or changed arithmetic.
"""
from pathlib import Path
import argparse,hashlib,json,shlex,subprocess
HERE=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--full-q',action='store_true')
ap.add_argument('--after-release',action='store_true');args=ap.parse_args()
variant='currentpvbeforeefullq' if args.full_q else 'currentpvbeforee'
if args.after_release:variant+='rel'
src=HERE.parent/'oc/opt_core/kernels/triattn_core_broadcast'
assert hashlib.sha256((src/'triattn_broadcast.so').read_bytes()).hexdigest()=='9365a4f35e018afed4adba21590ae2c1734bbccf98d1e53904ebfe2537b3e225'
dst=HERE/variant;dst.mkdir(exist_ok=True)
for p in (src/'csrc').rglob('*'):
 if not p.is_file():continue
 s=p.read_text()
 if p.name=='triattn_m1_sm90.cuh':
  a=s.index('    auto body =');b=s.index('    // drained state:',a)
  body=s[a:b]
  pv='        if constexpr (kFast) { issue_pv_prefenced(PCb[pb], tV, Int<hp>{}, Int<cp>{}, Int<(tp & 1) * R>{}); }'
  assert body.count(pv)==1
  body=body.replace(pv+'\n','')
  # Put PV immediately after QK's commit, before bias release and E.
  needle='        ring_wait();' if args.after_release else '        if constexpr (!kEarlyRel) { release2(dep2); }'
  assert body.count(needle)==1;body=body.replace(needle,pv+'\n'+needle)
  s=s[:a]+body+s[b:]
  if args.full_q:
   old='if constexpr(hh==0){';assert s.count(old)==1
   s=s.replace(old,'if constexpr(true){')
   old='''if constexpr(kFast) {
            auto a1=tQr1(_,_,Int<0>{});
            warpgroup_fence_operand(tQr0);warpgroup_fence_operand(a1);
        } else {warpgroup_fence_operand(tQr0);warpgroup_fence_operand(tQr1);}'''
   assert s.count(old)==2
   s=s.replace(old,'warpgroup_fence_operand(tQr0);warpgroup_fence_operand(tQr1);')
 s=s.replace('ta_core_broadcast','ta_sol_'+variant)
 q=dst/p.relative_to(src);q.parent.mkdir(parents=True,exist_ok=True);q.write_text(s)
inst=dst/'inst_m1';inst.mkdir(exist_ok=True)
cu=inst/'m1_1073741824.cu'
cu.write_text('#include "launch_m1.cuh"\nnamespace ta_sol_'+variant+' {\nvoid run_hot(Args const& a) { launch_m1<Traits<1073741824>>(a); }\n}\n')
ninja=(HERE/'build/triattn_sol_baseqthree1024/build.ninja').read_text()
line=next(l.split(' = ',1)[1] for l in ninja.splitlines() if l.startswith('cuda_cflags = '))
flags=[]
for f in shlex.split(line):
 if f=='--keep' or f.startswith('--keep-dir='):continue
 flags.append(f.replace(str(HERE/'baseqthree1024'),str(dst)).replace(str(HERE/'build/triattn_sol_baseqthree1024/inst_m1'),str(inst)))
ptx=dst/'m1_1073741824.ptx'
command=['/usr/local/cuda-12.9/bin/nvcc']+flags+['--ptx',str(cu),'-o',str(ptx)]
(dst/'command.json').write_text(json.dumps(command,indent=2)+'\n')
subprocess.run(command,check=True)
print('BUILT',variant,ptx,flush=True)
