"""Use a 32-entry warp-register exp2 table for one quarter of hot logits."""
from pathlib import Path
import hashlib, importlib.util, os

HERE = Path(__file__).resolve().parent
src = HERE.parent / 'oc/opt_core/kernels/triattn_core_broadcast'
assert hashlib.sha256((src / 'triattn_broadcast.so').read_bytes()).hexdigest() == '6d52ebfc4b43ca8fcbeac7de165d4572cf30d9cd2efe6ee6f63508425dc284fe'
variant = 'baseshufexp1'
dst = HERE / variant
dst.mkdir(exist_ok=True)
helper = '''
// Each lane owns exp2(lane/32); shuffle performs the lookup without an LDS.
// Rounded grid residual lies in [-1/64,1/64]. Linear interpolation has
// relative error below 6e-5 in the normal range (CPU screening only).
__device__ __forceinline__ float ex2_shuffle(float x,float lane_lut){
    float rounded=__fadd_rn(x,393216.f);
    float nearest=__fadd_rn(rounded,-393216.f);
    float residual=__fadd_rn(x,-nearest);
    unsigned bits=__float_as_uint(rounded);
    float tab=__shfl_sync(0xffffffffu,lane_lut,bits&31u);
    float scale=__uint_as_float(__float_as_uint(tab)+((bits>>5)<<23));
    float y=scale*fmaf(residual,0.6931471805599453f,1.f);
    return x < -126.f ? 0.f : (x >= 128.f ? INFINITY : y);
}
'''
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
        pos = s.index('template <class T>', s.index('__device__ __forceinline__ float ex2_approx'))
        s = s[:pos] + helper + s[pos:]
        marker = '    auto exp_chunk ='
        assert s.count(marker) == 1
        s = s.replace(marker, '    float const exp_lut_lane=kFast?ex2_approx(float(t128&31)*0.03125f):0.f;\n' + marker)
        old = 'for (int ni = 0; ni < kNC; ++ni) { s_rc(mi, ni) = ex2_approx(fmaf(s_rc(mi, ni), c_l2, nm[hh][mi])); }'
        assert s.count(old) == 1
        s = s.replace(old, '''for (int ni = 0; ni < kNC; ++ni) {
                if(kFast && ni%4==0)s_rc(mi,ni)=ex2_shuffle(fmaf(s_rc(mi,ni),c_l2,nm[hh][mi]),exp_lut_lane);
                else s_rc(mi,ni)=ex2_approx(fmaf(s_rc(mi,ni),c_l2,nm[hh][mi]));
            }''')
    target = dst / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(s)
os.environ['TORCH_EXTENSIONS_DIR'] = str(HERE / 'build')
os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0a'
os.environ['MAX_JOBS'] = '2'
os.environ['CUTLASS_PATH'] = str(HERE.parents[1] / 'anthropic_adoption_20260919/cutlass-4.2')
os.environ['TRIATTN_PTXAS'] = 'image'
spec = importlib.util.spec_from_file_location('shuffle_exp_builder', dst / 'triattn_m1.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m._build(verbose=True)
print('BUILT', variant, flush=True)
