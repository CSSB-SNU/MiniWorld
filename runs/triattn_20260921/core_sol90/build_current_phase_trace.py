"""Coarse CTA phase timing on installed9365, outside steady chunk bodies.

Diagnostic only. Three fixed CTAs, all three consumer-WG leaders. No kernel
argument changes: timestamps live in a private standalone-cubin global.
Clock differences are only taken within one CTA/SM, never between SMs.
"""
from pathlib import Path
import hashlib,json,shlex,subprocess
HERE=Path(__file__).resolve().parent
variant='currentphasetrace'
src=HERE.parent/'oc/opt_core/kernels/triattn_core_broadcast'
assert hashlib.sha256((src/'triattn_broadcast.so').read_bytes()).hexdigest()=='9365a4f35e018afed4adba21590ae2c1734bbccf98d1e53904ebfe2537b3e225'
dst=HERE/variant;dst.mkdir(exist_ok=True)
helper='''
extern "C" { __device__ unsigned long long diag_phase_clock[3*3*20]; }
__device__ __forceinline__ void diag_phase_mark(int point) {
    unsigned x,y,z,t;
    asm volatile("mov.u32 %0, %%ctaid.x;" : "=r"(x));
    asm volatile("mov.u32 %0, %%ctaid.y;" : "=r"(y));
    asm volatile("mov.u32 %0, %%ctaid.z;" : "=r"(z));
    asm volatile("mov.u32 %0, %%tid.x;" : "=r"(t));
    int const sample=x==1 && y==1 && z==0 ? 0 :
                     x==2 && y==128 && z==1 ? 1 :
                     x==5 && y==255 && z==3 ? 2 : -1;
    if(sample>=0 && t>=128 && (t&127)==0) {
        int const base=(sample*3+(t/128-1))*20;
        diag_phase_clock[base+point]=clock64();
        if(point==0) {
            unsigned sm;
            asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
            diag_phase_clock[base+19]=sm;
        }
    }
}
'''
for p in (src/'csrc').rglob('*'):
 if not p.is_file():continue
 s=p.read_text()
 if p.name=='triattn_m1_sm90.cuh':
  s=s.replace('using namespace cute;','using namespace cute;\n'+helper,1)
  old='    auto init_barriers = [&]() {'
  assert s.count(old)==1;s=s.replace(old,'    if constexpr(kFast)diag_phase_mark(0);\n'+old)
  old='    cutlass::arch::warpgroup_reg_alloc<160>();'
  assert s.count(old)==1;s=s.replace(old,'    if constexpr(kFast)diag_phase_mark(1);\n'+old+'\n    if constexpr(kFast)diag_phase_mark(2);')
  a=s.index('    // ---- prologue = chunks 0, 1 (tile 0, chunk column 0, both halves)')
  b=s.index('    // ---- epilogue:',a)
  h=s[a:b]
  old='    shared.barrier_q.wait(0);'
  assert h.count(old)==1;h=h.replace(old,'    if constexpr(kFast)diag_phase_mark(3);\n'+old)
  old='    // a tile of the CTA stream this warpgroup does not consume:'
  assert h.count(old)==1;h=h.replace(old,'    if constexpr(kFast)diag_phase_mark(4);\n'+old)
  old='    init_chunk(accC[0], _0{}, _0{}); init_chunk(accC[1], _0{}, _1{});'
  assert h.count(old)==1;h=h.replace(old,'    if constexpr(kFast)diag_phase_mark(5);\n'+old)
  old='    pipe_b.wait_full(1, 0);'
  assert h.count(old)==1;h=h.replace(old,'    if constexpr(kFast)diag_phase_mark(6);\n'+old)
  old='    if (!lead_follow) { cutlass::arch::NamedBarrier::sync'
  assert h.count(old)==1;h=h.replace(old,'    if constexpr(kFast)diag_phase_mark(7);\n'+old)
  old='    // ---- the loop: period p ='
  assert h.count(old)==1;h=h.replace(old,'    if constexpr(kFast)diag_phase_mark(8);\n'+old)
  old='drain(_3{}); return true;'
  assert h.count(old)==2;h=h.replace(old,'if constexpr(kFast)diag_phase_mark(13); drain(_3{}); if constexpr(kFast)diag_phase_mark(14); return true;')
  old='        drain(_1{});'
  assert h.count(old)==1;h=h.replace(old,'        if constexpr(kFast)diag_phase_mark(9+2*p);\n'+old+'\n        if constexpr(kFast)diag_phase_mark(10+2*p);')
  old='    #pragma unroll 1\n    for (int t = 0; t < n_trail;'
  assert h.count(old)==1;h=h.replace(old,'    if constexpr(kFast)diag_phase_mark(15);\n'+old)
  s=s[:a]+h+s[b:]
  old='    }   // consumer tile loop'
  # Put the last marker before the one-shot hot exit, not after its break.
  needle='    if constexpr (!kList) { break; }\n'+old
  assert s.count(needle)==1;s=s.replace(needle,'    if constexpr(kFast)diag_phase_mark(16);\n'+needle)
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
