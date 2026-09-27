"""Pin P through the existing bias-empty arrival, without a new probe.

The arrival depends on every score word read from bias, not an assumption
that the compiler will retain LDS.128. Optional P words extend that same
dependency to force pack completion before the following QK fence.
"""
from pathlib import Path
import hashlib,json,shlex,subprocess,sys
HERE=Path(__file__).resolve().parent
mode=sys.argv[1];assert mode in ('bias','p')
variant='releaseorder'+mode
src=HERE.parent/'oc/opt_core/kernels/triattn_core_broadcast'
assert hashlib.sha256((src/'triattn_broadcast.so').read_bytes()).hexdigest()=='9365a4f35e018afed4adba21590ae2c1734bbccf98d1e53904ebfe2537b3e225'
dst=HERE/variant;dst.mkdir(exist_ok=True)
for p in (src/'csrc').rglob('*'):
 if not p.is_file():continue
 s=p.read_text()
 if p.name=='triattn_m1_sm90.cuh':
  old='        if constexpr (kFast) { pack_chunk(accC[bp], PCb[pb]); warpgroup_fence_operand(PCb[pb]); }'
  assert s.count(old)==1
  new='''        if constexpr (kFast) {
            pack_chunk(accC[bp], PCb[pb]);
            warpgroup_fence_operand(PCb[pb]);
            warpgroup_fence_operand(accC[bn]);
            uint32_t dep=0;
            #pragma unroll
            for(int j=0;j<16;++j)dep|=__float_as_uint(accC[bn](j));
            P_DEPENDENCY
            asm volatile("" ::: "memory");
            __syncwarp();
            pipe_b.release_dep(e2&7,kRelAlways?leadk2:(rel2&&warp_leader),dep,params.zero);
        }'''
  pdep='''auto pw=recast<uint32_t>(PCb[pb]);
            #pragma unroll
            for(int j=0;j<8;++j)dep|=pw(j);''' if mode=='p' else ''
  s=s.replace(old,new.replace('P_DEPENDENCY',pdep))
  old='        if constexpr (!kEarlyRel) { release2(dep2); }'
  assert s.count(old)==1
  s=s.replace(old,'        if constexpr (!kFast && !kEarlyRel) { release2(dep2); }')
  old='        if constexpr (kEarlyRel) { release2(dep2); }'
  assert s.count(old)==1
  s=s.replace(old,'        if constexpr (!kFast && kEarlyRel) { release2(dep2); }')
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
 f=f.replace(str(HERE/'baseqthree1024'),str(dst)).replace(str(HERE/'build/triattn_sol_baseqthree1024/inst_m1'),str(inst))
 flags.append(f)
ptx=dst/'m1_1073741824.ptx'
command=['/usr/local/cuda-12.9/bin/nvcc']+flags+['--ptx',str(cu),'-o',str(ptx)]
(dst/'command.json').write_text(json.dumps(command,indent=2)+'\n')
subprocess.run(command,check=True)
print('BUILT',variant,ptx,flush=True)
