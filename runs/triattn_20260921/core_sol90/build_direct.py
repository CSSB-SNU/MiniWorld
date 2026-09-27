from pathlib import Path
import os,sys,shutil,importlib.util
HERE=Path(__file__).resolve().parent
SRC=HERE.parent/'core_tiles/broadcast'
variant=sys.argv[1] if len(sys.argv)>1 else 'direct'
DST=HERE/variant
for p in SRC.rglob('*'):
 if p.is_file() and p.suffix in ('.py','.cu','.cuh','.h'):
  q=DST/p.relative_to(SRC);q.parent.mkdir(parents=True,exist_ok=True)
  s=p.read_text().replace('ta_core_broadcast','ta_sol_'+variant).replace('triattn_broadcast','triattn_sol_'+variant)
  if q.name=='triattn_m1.py':s=s.replace('FLAGS = [0, 256]','FLAGS = [0]')
  q.write_text(s)
p=DST/'csrc/m1/triattn_m1_sm90.cuh';s=p.read_text()
pos=s.index('template <int kFlags_ = 0>')
s=s[:pos]+'''struct DirectBiasPipe {
 struct SharedStorage {};
 CUTLASS_DEVICE DirectBiasPipe(SharedStorage&) {}
 CUTLASS_DEVICE static void init(SharedStorage&,int,int=1) {}
 CUTLASS_DEVICE void wait_full(int,uint32_t) {}
 CUTLASS_DEVICE bool test_full(int,uint32_t) {return true;}
 CUTLASS_DEVICE void release(int,bool) {}
 CUTLASS_DEVICE void release_dep(int,bool,uint32_t,int) {}
 CUTLASS_DEVICE void release2(int,bool,uint32_t) {}
};
''' +s[pos:]
s=s.replace('using PipeB = Pipe<kHalves>;', 'using PipeB = std::conditional_t<kSafe,Pipe<kHalves>,DirectBiasPipe>;')
s=s.replace('cute::array_aligned<float, kSlotElems * kSlotsB, 1024>', 'cute::array_aligned<float, kSafe ? kSlotElems * kSlotsB : 1, 1024>')
s=s.replace('int const* rowkc1;                                             //', 'float const* direct_bias;\n        int const* rowkc1;                                             //')
s=s.replace('} else if (warp_idx_in_wg == 0 && lane_predicate) {','} else if constexpr (kSafe) { if (warp_idx_in_wg == 0 && lane_predicate) {')
s=s.replace('        if constexpr (!kList) { break; }\n        }   // producer tile loop','        }\n        if constexpr (!kList) { break; }\n        }   // producer tile loop')
start=s.index('    auto bias_release_by =');end=s.index('    auto bias_release =',start)
part=s[start:end];part=part.replace('        asm volatile(', '        if constexpr(kSafe) { asm volatile(',1).replace('    };','        }\n    };')
s=s[:start]+part+s[end:]
s=s.replace('    auto init_chunk =', '''    int direct_chunk=0;
    float const* direct_base=params.direct_bias+(int64_t(bh)*params.n_qtiles+qtile)*params.wpr*T::kSlotElems+t128*4;
    auto init_chunk =''')
s=s.replace('        if constexpr (kNoBias) {\n            #pragma unroll\n            for (int v = 0;', '''        if constexpr (!kSafe) {
            int dc=direct_chunk++;
            int col=min(kc0+4*jb_w+(dc>>1),params.wpr-1);
            #pragma unroll
            for(int u=0;u<4;u++) {
                float const* ptr=direct_base+col*T::kSlotElems+((dc&1)*4+u)*512;
                float4 x;
                asm volatile("ld.global.ca.v4.f32 {%0,%1,%2,%3},[%4];" : "=f"(x.x),"=f"(x.y),"=f"(x.z),"=f"(x.w) : "l"(ptr) : "memory");
                acc(4*u)=x.x;acc(4*u+1)=x.y;acc(4*u+2)=x.z;acc(4*u+3)=x.w;
            }
        } else if constexpr (kNoBias) {
            #pragma unroll
            for (int v = 0;''')
if variant=='directj':
 s=s.replace('    int direct_chunk=0;\n','')
 s=s.replace('auto cc, auto hc) __attribute__((always_inline)) {     // acc <- bias', 'auto cc, auto hc, int direct_tile) __attribute__((always_inline)) {     // acc <- bias')
 s=s.replace('int dc=direct_chunk++;\n            int col=min(kc0+4*jb_w+(dc>>1),params.wpr-1);', 'int col=kc0+4*jb_w+4*min(direct_tile,n_w-1)+c;')
 s=s.replace('((dc&1)*4+u)*512', '(hh*4+u)*512')
 s=s.replace('init_chunk(accC[bp], Int<c3>{}, Int<h3>{});','init_chunk(accC[bp], Int<c3>{}, Int<h3>{},j3);')
 s=s.replace('init_chunk(accC[bq], Int<c2>{}, Int<h2>{});','init_chunk(accC[bq], Int<c2>{}, Int<h2>{},j2);')
 import re
 s=re.sub(r'init_chunk\((accC\[\d\], _\d\{\}, _\d\{\})\);',r'init_chunk(\1,0);',s)
p.write_text(s)
p=DST/'csrc/m1/launch_m1.cuh';s=p.read_text().replace('n_kcol, a.rowkc0, a.rowkc1}', 'n_kcol, a.rowkc0, a.bias.data_ptr<float>(), a.rowkc1}')
p.write_text(s)
os.environ['TORCH_EXTENSIONS_DIR']=str(HERE/'build')
os.environ['TORCH_CUDA_ARCH_LIST']='9.0a';os.environ['MAX_JOBS']='2'
os.environ['CUTLASS_PATH']=str(HERE.parents[1]/'anthropic_adoption_20260919/cutlass-4.2')
spec=importlib.util.spec_from_file_location('direct_builder',DST/'triattn_m1.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module._build(verbose=True)
