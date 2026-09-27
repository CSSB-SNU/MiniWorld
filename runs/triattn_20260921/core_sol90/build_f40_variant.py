"""Experiments pinned to the qualified N40 descriptor package (6fe43326...)."""
from pathlib import Path
import importlib.util,os,sys,re
HERE=Path(__file__).resolve().parent
variant=sys.argv[1]
assert variant in ('basef40ip4','basef40qrip4','basef40qrip8','basef40fixeddesc','basef40qrh0','basef40qrh1','basef40qrh0ip4','basef40qrk0','basef40qrk0ip4','basef40qrh1ip4','basef40qrh0ip4fixed','basef40qrh0fixed')
group=int(re.search(r'ip(\d+)',variant).group(1)) if 'ip' in variant else 0;src=HERE/'f40_package';dst=HERE/variant
dst.mkdir(exist_ok=True)
for p in src.rglob('*'):
 if not p.is_file() or p.suffix not in ('.py','.cu','.cuh','.h'):continue
 relative=p.relative_to(src)
 if '__pycache__' in relative.parts:continue
 if str(relative) in ('__init__.py','build_native.py'):continue
 if str(relative)=='build_source.py':relative=Path('triattn_m1.py')
 target=dst/relative;target.parent.mkdir(parents=True,exist_ok=True)
 s=p.read_text().replace('ta_core_broadcast','ta_sol_'+variant).replace('triattn_broadcast','triattn_sol_'+variant)
 if p.name=='triattn_m1_sm90.cuh' and 'qrk' in variant:
  from partial_q_kblock import transform
  s=transform(s,kblock=int(variant.split('qrk')[1][0]))
 if p.name=='triattn_m1_sm90.cuh' and 'qrh' in variant:
  from partial_q_registers import transform
  s=transform(s,half=int(variant.split('qrh')[1][0]))
 if p.name=='triattn_m1_sm90.cuh' and (variant=='basef40fixeddesc' or variant.endswith('fixed')):
  old='int const zo = params.zero * p;'
  assert s.count(old)==2
  s=s.replace(old,'int const zo = kFast ? 0 : params.zero * p;')
 if p.name=='triattn_m1_sm90.cuh' and group:
  if 'qr' in variant and not any(x in variant for x in ('qrh','qrk')):
   old='static constexpr bool kQinRegs = (kFlags_ & 1048576) != 0;'
   assert s.count(old)==1
   s=s.replace(old,'static constexpr bool kQinRegs = (kFlags_ & 1048576) != 0 || (kFlags_ & 1073741824) != 0;')
  old='''        #pragma unroll
        for (int mi = 0; mi < kNRows; ++mi) {
            #pragma unroll
            for (int ni = 0; ni < kNC; ++ni) { s_rc(mi, ni) = ex2_approx(fmaf(s_rc(mi, ni), c_l2, nm[hh][mi])); }
        }
        warpgroup_fence_operand(acc);'''
  assert s.count(old)==1
  instructions=[]
  for i in range(group):instructions.append('fma.rn.ftz.f32 %%%d,%%%d,%%%d,%%%d;'%(i,i,group,group+1))
  for i in range(group):instructions.append('ex2.approx.ftz.f32 %%%d,%%%d;'%(i,i))
  asm='\n'.join('                "'+line+'\\n"' for line in instructions)
  outputs=', '.join('"+f"(s_rc(mi, ni+%d))'%i for i in range(group))
  new='''        if constexpr(kFast) {
            #pragma unroll
            for(int mi=0;mi<kNRows;++mi) {
                #pragma unroll
                for(int ni=0;ni<kNC;ni+=GROUP) {
                    asm volatile(
ASM
                        : OUTPUTS : "f"(c_l2), "f"(nm[hh][mi]));
                }
            }
        } else {
'''.replace('GROUP',str(group)).replace('ASM',asm).replace('OUTPUTS',outputs)+old.split('        warpgroup_fence_operand(acc);')[0]+'''        }
        warpgroup_fence_operand(acc);'''
  s=s.replace(old,new)
 target.write_text(s)
os.environ['TORCH_EXTENSIONS_DIR']=str(HERE/'build')
os.environ['TORCH_CUDA_ARCH_LIST']='9.0a';os.environ['MAX_JOBS']='2'
os.environ['CUTLASS_PATH']=str(HERE.parents[1]/'anthropic_adoption_20260919/cutlass-4.2')
os.environ['TRIATTN_PTXAS']='image'
spec=importlib.util.spec_from_file_location('f40_variant_builder',dst/'triattn_m1.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m._build(verbose=True)
