"""Exact descriptor-coordinate producer control and mixed full-Q experiment.

Fast-only, isolated source. Registers are assigned per COMPLETE warpgroup:
32+160+160+160 or 24+168+160+160, each summing to 512 per lane quartet.
No consumer barrier or arithmetic changes; the 168-register consumer caches
the fourth Q fragment while the other two retain installed three-fragment Q.
"""
from pathlib import Path
import argparse,hashlib,json,shlex,subprocess

HERE=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--mixed',action='store_true')
ap.add_argument('--unroll-bias',action='store_true');args=ap.parse_args()
variant='leanprod24q168' if args.mixed else 'leanprod32current'
if args.unroll_bias:variant+='b8v2'
src=HERE.parent/'oc/opt_core/kernels/triattn_core_broadcast'
assert hashlib.sha256((src/'triattn_broadcast.so').read_bytes()).hexdigest()=='9365a4f35e018afed4adba21590ae2c1734bbccf98d1e53904ebfe2537b3e225'
dst=HERE/variant;dst.mkdir(exist_ok=True)
for p in (src/'csrc').rglob('*'):
 if not p.is_file():continue
 s=p.read_text()
 if p.name=='triattn_m1_sm90.cuh':
  producer_anchor='    if (wg_idx == 0) {\n        // =============================================== PRODUCER'
  assert s.count(producer_anchor)==1
  injection='    if (wg_idx == 0) {\n        constexpr int kLeanProducerRegs=%d;\n'% (24 if args.mixed else 32)
  producer=(HERE/'lean_producer_body.cuh').read_text()
  if args.unroll_bias:
   old='''        for(int seq=0;seq<8*nt;++seq) {
            int const slot=seq&7;'''
   new='''        for(int j=0;j<nt;++j) {
          #pragma unroll
          for(int slot=0;slot<8;++slot) {
            int const seq=8*j+slot;'''
   assert producer.count(old)==1;producer=producer.replace(old,new)
   old='''                0,0,2*first+seq,g_qtile,g_bh);
        }'''
   assert producer.count(old)==1;producer=producer.replace(old,old+'\n        }')
  injection+=producer
  injection+='\n        // =============================================== PRODUCER'
  s=s.replace(producer_anchor,injection)
  assert s.count('constexpr int kLeanProducerRegs=%d;'%(24 if args.mixed else 32))==1
  assert 'for(int seq=0;seq<8*nt;++seq)' in s or 'int const seq=8*j+slot;' in s
  if args.mixed:
   old='''    {
    // ================================================= CONSUMERS ======================================================
    cutlass::arch::warpgroup_reg_alloc<160>();'''
   new='''    auto consume=[&](auto fullq,auto regs) __attribute__((always_inline)) {
    constexpr bool kFullQR=decltype(fullq)::value;
    cutlass::arch::warpgroup_reg_alloc<decltype(regs)::value>();'''
   assert s.count(old)==1;s=s.replace(old,new)
   old='if constexpr(hh==0){'
   assert s.count(old)==1;s=s.replace(old,'if constexpr(hh==0 || kFullQR){')
   # Existing fast fences keep half0's two slices and half1's first slice.
   # The full-Q branch additionally fences the second half's last K16 slice.
   start=s.index('    auto consume=')
   a=s[:start];b=s[start:]
   # These are the two Q-ready sites (W and ordinary four-score schedules).
   needle='''if constexpr(kFast) {
            auto a1=tQr1(_,_,Int<0>{});'''
   assert b.count(needle)==2
   b=b.replace(needle,'''if constexpr(kFast && !kFullQR) {
            auto a1=tQr1(_,_,Int<0>{});''')
   old='''    }   // consumer tile loop
    }   // consumers
    if constexpr (CR > 1)'''
   assert b.count(old)==1
   b=b.replace(old,'''    }   // consumer tile loop
    };
    if constexpr(kFast) {
        if(wg_idx==1)consume(cute::true_type{},Int<168>{});
        else consume(cute::false_type{},Int<160>{});
    } else consume(cute::false_type{},Int<160>{});
    if constexpr (CR > 1)''')
   s=a+b
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
