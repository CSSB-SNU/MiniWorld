"""Exact exponentiation scheduling with the installed partial-Q allocation."""
from pathlib import Path
import hashlib,importlib.util,os,sys
HERE=Path(__file__).resolve().parent
group=int(sys.argv[1]);assert group in (2,8)
variant='baseqsmallip%d'%group
src=HERE.parent/'oc/opt_core/kernels/triattn_core_broadcast'
assert hashlib.sha256((src/'triattn_broadcast.so').read_bytes()).hexdigest()=='97634810cdba1e91fe4aecb8bdfa610dd050dd39e7504ffb8babee8c46d4077b'
dst=HERE/variant;dst.mkdir(exist_ok=True)
for p in src.rglob('*'):
    if not p.is_file() or p.suffix not in ('.py','.cu','.cuh','.h'):continue
    rel=p.relative_to(src)
    if '__pycache__' in rel.parts or str(rel) in ('__init__.py','build_native.py'):continue
    if str(rel)=='build_source.py':rel=Path('triattn_m1.py')
    s=p.read_text().replace('ta_core_broadcast','ta_sol_'+variant).replace('triattn_broadcast','triattn_sol_'+variant)
    if p.name=='triattn_m1_sm90.cuh':
        a=s.index('        if constexpr(kFast){',s.index('    auto exp_chunk ='))
        b=s.index('        }else{',a)
        instr=['fma.rn.ftz.f32 %%%d,%%%d,%%%d,%%%d;'%(i,i,group,group+1) for i in range(group)]
        instr+=['ex2.approx.ftz.f32 %%%d,%%%d;'%(i,i) for i in range(group)]
        asm='\n'.join('                    "'+v+'\\n"' for v in instr)
        operands=', '.join('"+f"(s_rc(mi,ni+%d))'%i for i in range(group))
        s=s[:a]+'''        if constexpr(kFast){
            #pragma unroll
            for(int mi=0;mi<kNRows;++mi){
                #pragma unroll
                for(int ni=0;ni<kNC;ni+=GROUP){
                    asm volatile(
'''.replace('GROUP',str(group))+asm+'\n                        : '+operands+''' : "f"(c_l2), "f"(nm[hh][mi]));
                }
            }
''' + s[b:]
    target=dst/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(s)
os.environ['TORCH_EXTENSIONS_DIR']=str(HERE/'build')
os.environ['TORCH_CUDA_ARCH_LIST']='9.0a';os.environ['MAX_JOBS']='2'
os.environ['CUTLASS_PATH']=str(HERE.parents[1]/'anthropic_adoption_20260919/cutlass-4.2')
os.environ['TRIATTN_PTXAS']='image'
spec=importlib.util.spec_from_file_location('exp_group_builder',dst/'triattn_m1.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m._build(verbose=True)
print('BUILT',variant,flush=True)
