"""Native current full-Q control, one hot translation unit, no shared install."""
from pathlib import Path
import hashlib,json,shlex,subprocess
HERE=Path(__file__).resolve().parent
src=HERE.parent/'oc/opt_core/kernels/triattn_core_broadcast'
assert hashlib.sha256((src/'triattn_broadcast.so').read_bytes()).hexdigest()=='9365a4f35e018afed4adba21590ae2c1734bbccf98d1e53904ebfe2537b3e225'
dst=HERE/'fullqcurrent';dst.mkdir(exist_ok=True)
for p in (src/'csrc').rglob('*'):
 if not p.is_file():continue
 s=p.read_text()
 if p.name=='triattn_m1_sm90.cuh':
  old='''            if constexpr(hh==0){
                cute::gemm(tiled_mma_qkr, tQr(_,_,1), tK(_,_,1,c,st), acc);
            } else {
                cute::gemm(tiled_mma_qk, tQ(_,_,1,hh,cwg), tK(_,_,1,c,st), acc);
            }'''
  assert s.count(old)==1
  s=s.replace(old,'            cute::gemm(tiled_mma_qkr, tQr(_,_,1), tK(_,_,1,c,st), acc);')
  old='''            auto a1=tQr1(_,_,Int<0>{});
            warpgroup_fence_operand(tQr0);warpgroup_fence_operand(a1);'''
  assert s.count(old)==2
  s=s.replace(old,'            warpgroup_fence_operand(tQr0);warpgroup_fence_operand(tQr1);')
 s=s.replace('ta_core_broadcast','ta_sol_fullqcurrent')
 q=dst/p.relative_to(src);q.parent.mkdir(parents=True,exist_ok=True);q.write_text(s)
inst=dst/'inst_m1';inst.mkdir(exist_ok=True)
cu=inst/'m1_1073741824.cu'
cu.write_text('#include "launch_m1.cuh"\nnamespace ta_sol_fullqcurrent {\nvoid run_hot(Args const& a) { launch_m1<Traits<1073741824>>(a); }\n}\n')
ninja=(HERE/'build/triattn_sol_baseqthree1024/build.ninja').read_text()
line=next(l.split(' = ',1)[1] for l in ninja.splitlines() if l.startswith('cuda_cflags = '))
flags=[]
for f in shlex.split(line):
 if f=='--keep' or f.startswith('--keep-dir='):continue
 f=f.replace(str(HERE/'baseqthree1024'),str(dst)).replace(str(HERE/'build/triattn_sol_baseqthree1024/inst_m1'),str(inst))
 flags.append(f)
ptx=dst/'m1_1073741824.ptx'
command=['/usr/local/cuda-12.9/bin/nvcc']+flags+['--ptx',str(cu),'-o',str(ptx)]
(dst/'command.json').write_text(json.dumps(command,indent=2)+'\n')
subprocess.run(command,check=True)
print('BUILT native full-Q PTX',ptx,flush=True)
