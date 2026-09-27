"""Installed-source control: half of every QK uses retained Q register A."""
from pathlib import Path
import hashlib, importlib.util, os, sys
from partial_q_kblock import transform

HERE = Path(__file__).resolve().parent
src = HERE.parent / 'oc/opt_core/kernels/triattn_core_broadcast'
assert hashlib.sha256((src / 'triattn_broadcast.so').read_bytes()).hexdigest() == '6d52ebfc4b43ca8fcbeac7de165d4572cf30d9cd2efe6ee6f63508425dc284fe'
variant = sys.argv[1] if len(sys.argv)>1 else 'baseqrk0'
assert variant in ('baseqrk0', 'baseqrk0ip4fixed', 'baseqrk0ip4small')
dst = HERE / variant
dst.mkdir(exist_ok=True)
for p in src.rglob('*'):
    if not p.is_file() or p.suffix not in ('.py', '.cu', '.cuh', '.h'):
        continue
    rel = p.relative_to(src)
    if '__pycache__' in rel.parts or str(rel) in ('__init__.py', 'build_native.py'):
        continue
    if str(rel) == 'build_source.py':
        rel = Path('triattn_m1.py')
    s = p.read_text().replace('ta_core_broadcast', 'ta_sol_' + variant).replace('triattn_broadcast', 'triattn_sol_' + variant)
    if p.name == 'triattn_m1_sm90.cuh':
        s = transform(s)
        if variant in ('baseqrk0ip4fixed', 'baseqrk0ip4small'):
            old = '''        #pragma unroll
        for (int mi = 0; mi < kNRows; ++mi) {
            #pragma unroll
            for (int ni = 0; ni < kNC; ++ni) { s_rc(mi, ni) = ex2_approx(fmaf(s_rc(mi, ni), c_l2, nm[hh][mi])); }
        }
        warpgroup_fence_operand(acc);'''
            assert s.count(old) == 1
            instructions = ['fma.rn.ftz.f32 %%%d,%%%d,%%4,%%5;' % (i,i) for i in range(4)]
            instructions += ['ex2.approx.ftz.f32 %%%d,%%%d;' % (i,i) for i in range(4)]
            asm = '\n'.join('                    "' + x + '\\n"' for x in instructions)
            operands = ', '.join('"+f"(s_rc(mi,ni+%d))' % i for i in range(4))
            new = '''        if constexpr(kFast){
            #pragma unroll
            for(int mi=0;mi<kNRows;++mi){
                #pragma unroll
                for(int ni=0;ni<kNC;ni+=4){
                    asm volatile(
''' + asm + '\n                        : ' + operands + ''' : "f"(c_l2), "f"(nm[hh][mi]));
                }
            }
        }else{
''' + old.split('        warpgroup_fence_operand(acc);')[0] + '''        }
        warpgroup_fence_operand(acc);'''
            s = s.replace(old, new)
        if variant == 'baseqrk0ip4small':
            # Same standalone small-ones transformation, with explicit checks.
            def rep(old, new, count=1):
                global s
                assert s.count(old)==count,(old,s.count(old),count)
                s=s.replace(old,new)
            rep('cute::array_aligned<Element, kBlockN*kHeadDim, 1024> smem_ones;', 'cute::array_aligned<Element, CW*kHeadDim, 1024> smem_ones;')
            rep('idx < T::kBlockN*T::kHeadDim', 'idx < T::CW*T::kHeadDim')
            rep('auto const& operand, auto stage) __attribute__', 'auto const& operand, auto stage, auto column) __attribute__')
            rep('constexpr int st=decltype(stage)::value;', 'constexpr int st=decltype(stage)::value, col=decltype(column)::value;')
            rep('uint32_t(st*T::kStageElemsV/8)<<16', 'uint32_t((st*T::kStageElemsV+col*CW*kHeadDim)/8)<<16')
            rep('fused_v_operand(tV(_, _, kb, c, st),Int<st>{})', 'fused_v_operand(tV(_, _, kb, c, st),Int<st>{},Int<c>{})', 2)
            rep('fused_v_operand(tV(_, _, kb, cp, stp),Int<stp>{})', 'fused_v_operand(tV(_, _, kb, cp, stp),Int<stp>{},Int<cp>{})')
    target = dst / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(s)
os.environ['TORCH_EXTENSIONS_DIR'] = str(HERE / 'build')
os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0a'
os.environ['MAX_JOBS'] = '2'
os.environ['CUTLASS_PATH'] = str(HERE.parents[1] / 'anthropic_adoption_20260919/cutlass-4.2')
os.environ['TRIATTN_PTXAS'] = 'image'
spec = importlib.util.spec_from_file_location('qr_k_builder', dst / 'triattn_m1.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m._build(verbose=True)
print('BUILT', variant, flush=True)
