"""Isolated experiments based on the installed, qualified source."""
from pathlib import Path
import importlib.util, os, sys
HERE=Path(__file__).resolve().parent
SRC=HERE.parent/'oc/opt_core/kernels/triattn_core_broadcast'
variant=sys.argv[1]
assert variant in ('base3producer','base3producerids','baseablateexp','baseablateexp64','baseablateexpxor','basehalfring','basefixeddesc','basestaticseed','baseablateqk','basef40tma','basestream48','basef40tmasw32','basef40desc','basephasepv','basephaselds','baseablatepv','basegroup','basef40group')
snapshot=HERE/('before_producer_install' if variant.startswith(('base3producer','baseablateexp')) else 'producer_package')
if snapshot.exists():SRC=snapshot
DST=HERE/variant
DST.mkdir(exist_ok=True)
wrapper=(HERE/'basefastdispatchsyn/triattn_m1.py').read_text().replace('basefastdispatchsyn',variant)
(DST/'triattn_m1.py').write_text(wrapper)
for p in (SRC/'csrc').rglob('*'):
 if not p.is_file() or p.suffix not in ('.cu','.cuh','.h'):continue
 q=DST/p.relative_to(SRC);q.parent.mkdir(parents=True,exist_ok=True)
 s=p.read_text().replace('ta_core_broadcast','ta_sol_'+variant).replace('triattn_broadcast','triattn_sol_'+variant)
 if p.name=='triattn_m1_sm90.cuh':
  if variant.startswith('basef40') or variant=='basegroup':pass
  elif variant.startswith('base3producer'):
   a=s.index('        if (warp_idx_in_wg == 1 && lane_predicate) {\n            // ---- the K/V stream:')
   b=s.index('        } else if (warp_idx_in_wg == 0 && lane_predicate) {',a)
   block=s[a:b]
   block=block.replace('if (warp_idx_in_wg == 1 && lane_predicate)',
                       'if ((warp_idx_in_wg == 1 || (kFast && warp_idx_in_wg == 2)) && lane_predicate)',1)
   loops=[];start=0
   while True:
    idx=block.find('                #pragma unroll\n                for (int r = 0; r < R; ++r) {',start)
    if idx<0:break
    loops.append(idx);start=idx+1
   assert len(loops)==2
   end=block.index('\n            }',loops[1])
   block=block[:end]+'\n                }'+block[end:]
   block=block[:loops[1]]+'                }\n                if (!kFast || warp_idx_in_wg == 2) {\n'+block[loops[1]:]
   block=block[:loops[0]]+'                if (!kFast || warp_idx_in_wg == 1) {\n'+block[loops[0]:]
   s=s[:a]+block+s[b:]
   if variant.endswith('ids'):
    s=s.replace('constexpr uint32_t kRingBar0 = uint32_t(cutlass::arch::ReservedNamedBarriers::FirstUserBarrier);',
                'constexpr uint32_t kRingBar0 = 0; // user-relative IDs 0..3 map to hardware 8..11')
  elif variant=='baseablatepv':
   # DIAGNOSTIC: omit only output PV. Keep P*ones so softmax and P packing
   # remain live and the original asynchronous commit count is preserved.
   import re
   s,n=re.subn(r'cute::gemm\(tiled_mma_pv,([^;]+);',r'if constexpr(!kFast) { cute::gemm(tiled_mma_pv,\1; }',s)
   assert n==4
  elif variant=='baseablateqk':
   # Deliberately incorrect diagnostic. Commit count is retained; each
   # shared-bias LDS is consumed before its producer receives the slot.
   a=s.index('    auto issue_qk =')
   b=s.index('    auto issue_pv =',a)
   block=s[a:b]
   start=block.index('        if constexpr (kQinRegs) {')
   end=block.index('        warpgroup_commit_batch();',start)
   original=block[start:end]
   replacement='        if constexpr(kFast) {\n            #pragma unroll\n            for(int z=0;z<16;++z) {\n                float v=acc(z);\n                asm volatile("add.f32 %0,%0,0f00000000;" : "+f"(v));\n                acc(z)=v;\n            }\n        } else {\n'+original+'        }\n'
   block=block[:start]+replacement+block[end:]
   s=s[:a]+block+s[b:]
  elif variant=='basestream48':
   # Keep the 4-score/2-P ordering across the two intermediate period
   # boundaries. Final validation still routes invalid sums to SAFE.
   old='else if constexpr (dd != 0) { warpgroup_wait<kQK1 ? 1 : 2>(); }'
   assert s.count(old)==1
   s=s.replace(old,old+'\n        else if constexpr(kFast) { if(p>0)warpgroup_wait<2>(); }')
   assert s.count('        drain(_1{});')==1
   s=s.replace('        drain(_1{});','        if constexpr(!kFast) { drain(_1{}); }')
   a=s.index('    constexpr bool kHasSteady =')
   b=s.index('    {   // last chunk K-1',a)
   original=s[a:b]
   s=s[:a]+'    if constexpr(kFast) {\n        if(!period(cute::false_type{},0,0u)) {\n            if(!period(cute::false_type{},1,1u)) {\n                period(cute::false_type{},2,0u);\n            }\n        }\n    } else {\n'+original+'    }\n'+s[b:]
   s=s.replace('int const zo = params.zero * p;','int const zo = kFast ? 0 : params.zero * p;')
  elif variant in ('basephasepv','basephaselds'):
   a=s.index('    auto body =')
   b=s.index('    // drained state:',a)
   block=s[a:b]
   x=block.index('        ring_wait();')
   y=block.index('        stamp(k, 4);',x)
   expblock=block[x:y]
   block=block[:x]+'        if(!kFast || cwg==0) {\n'+expblock+'        }\n'+block[y:]
   marker2='        stamp(k, 5);'
   late='cwg!=0' if variant=='basephasepv' else 'cwg==1'
   block=block.replace(marker2,'        if(kFast && '+late+') {\n'+expblock+'        }\n'+marker2)
   if variant=='basephaselds':
    marker2='        stamp(k, 6);'
    block=block.replace(marker2,'        if(kFast && cwg==2) {\n'+expblock+'        }\n'+marker2)
   s=s[:a]+block+s[b:]
  elif variant=='basehalfring':
   old='kRing = (T::kFlags & 32) != 0'
   assert s.count(old)==1
   s=s.replace(old,'kRing = (T::kFlags & 32) != 0 || kFast')
   old='''            for (int ni = 0; ni < kNC; ++ni) { s_rc(mi, ni) = ex2_approx(fmaf(s_rc(mi, ni), c_l2, nm[hh][mi])); }
        }
        warpgroup_fence_operand(acc);'''
   assert s.count(old)==1
   s=s.replace(old,'''            for (int ni = 0; ni < kNC; ++ni) { s_rc(mi, ni) = ex2_approx(fmaf(s_rc(mi, ni), c_l2, nm[hh][mi])); }
            // Scheduling token only: release after the first row's exponentials
            // have issued, overlapping their tail with the next warpgroup.
            if constexpr(kFast) {
                if(mi==0)cutlass::arch::NamedBarrier::arrive(256, uint32_t(cwg==R-1 ? 0 : cwg+1));
            }
        }
        warpgroup_fence_operand(acc);''')
   old='auto ring_pass = [&]() __attribute__((always_inline)) { if constexpr (kRing)'
   assert s.count(old)==1
   s=s.replace(old,'auto ring_pass = [&]() __attribute__((always_inline)) { if constexpr (kRing && !kFast)')
  elif variant=='basefixeddesc':
   old='int const zo = params.zero * p;'
   assert s.count(old)==2
   s=s.replace(old,'int const zo = kFast ? 0 : params.zero * p;')
  elif variant=='basestaticseed':
   # A different common softmax offset changes BF16 P rounding. This is
   # a numerical candidate, not a bitwise-equivalent transformation.
   assert s.count('uint32_t seeded = 0;')==1
   assert s.count('bool need_seed = true;')==1
   assert s.count('nm[hh][0] = 0.f; nm[hh][1] = 0.f;')==1
   s=s.replace('uint32_t seeded = 0;','uint32_t seeded = kFast ? 0xfu : 0u;')
   s=s.replace('bool need_seed = true;','bool need_seed = !kFast;')
   s=s.replace('nm[hh][0] = 0.f; nm[hh][1] = 0.f;',
               'nm[hh][0] = kFast ? -64.f : 0.f; nm[hh][1] = kFast ? -64.f : 0.f;')
  else:
   # DIAGNOSTIC ONLY: replace the transcendental with a cheap dependent
   # bit transform yielding positive finite values. This is not attention.
   old='s_rc(mi, ni) = ex2_approx(fmaf(s_rc(mi, ni), c_l2, nm[hh][mi]));'
   assert s.count(old)==1
   s=s.replace(old,'''if constexpr(kFast) {
                float z=fmaf(s_rc(mi, ni), c_l2, nm[hh][mi]);
                s_rc(mi,ni)=__uint_as_float((__float_as_uint(z)&0x007fffffu)|PROB_BITS);
            } else { '''.replace('PROB_BITS','0x1f000000u' if variant.endswith('64') else '0x3f000000u')+old+' }')
   if variant.endswith('xor'):
    s=s.replace('(__float_as_uint(z)&0x007fffffu)|0x3f000000u', '__float_as_uint(z)^0xdd000000u')
 if variant.startswith('basef40tma'):
  from f40_tma import transform,sw32
  s=(sw32 if variant.endswith('sw32') else transform)(p.name,s)
 if variant in ('basef40desc','basef40group'):
  from f40_descriptor import transform
  s=transform(p.name,s)
 if variant in ('basegroup','basef40group'):
  from merge_qk_pv_group import transform
  s=transform(p.name,s)
 q.write_text(s)
os.environ['TORCH_EXTENSIONS_DIR']=str(HERE/'build')
os.environ['TORCH_CUDA_ARCH_LIST']='9.0a';os.environ['MAX_JOBS']='2'
os.environ['CUTLASS_PATH']=str(HERE.parents[1]/'anthropic_adoption_20260919/cutlass-4.2')
os.environ['TRIATTN_PTXAS']='image'
spec=importlib.util.spec_from_file_location('installed_variant_builder',DST/'triattn_m1.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m._build(verbose=True)
