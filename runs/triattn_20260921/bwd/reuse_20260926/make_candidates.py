"""Small backward follow-up: reuse resident operands and overlap dP with exp."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parent
bwd = root.parent
pkg = bwd.parents[1] / 'trimul_sm90_parity_20260917/engine/src/miniworld_engine/kernels/triangle_attention'
snapshot = json.loads((bwd.parent / 'fwd_training/checkpoint18246/snapshot.json').read_text())['sha256']
for name, digest in snapshot.items():
    assert hashlib.sha256((pkg / name).read_bytes()).hexdigest() == digest, name
(root / 'baseline.json').write_text(json.dumps(snapshot, indent=2) + '\n')


def emit(kind, name, source):
    folder = bwd / ('dq' if kind == 'dq' else 'bias_fusion') / name
    assert not (folder / 'build-ready.json').exists(), 'Never mutate a built artifact'
    folder.mkdir(exist_ok=True)
    (folder / ('fused.cu' if kind == 'dq' else 'grouped.cu')).write_text(source)


base = (pkg / 'cuda/dq.cu').read_text()
for name, keep_dout in [('reuse_q', False), ('reuse_qdo', True)]:
    s = base.replace(' using Score=', ' using ScoreRS=decltype(make_tiled_mma(SM90_64x64x16_F32BF16BF16_RS<GMMA::Major::K,GMMA::Major::K>{}));\n using Score=', 1)
    marker = ' uint32_t mp=cast_smem_ptr_to_uint(s.lse.data())'
    code = ' Config::ScoreRS rs_mma;\n'
    for tag, tensor, member in [('qr', 'sq', 'q')] + ([('dr', 'sd', 'dout')] if keep_dout else []):
        code += f''' auto {tag}=rs_mma.get_slice(lane).partition_fragment_A({tensor});
 auto {tag}_words=recast<uint32_t>({tag});static_assert(size({tag}_words)==8);
 #pragma unroll
 for(int slice=0;slice<2;++slice){{
  int row=(lane/32)*16+lane%16,col=(lane%32/16)*8+slice*16;
  uint32_t addr=cast_smem_ptr_to_uint(s.{member}.data())+
    as_position_independent_swizzle_layout(Config::QL{{}})(make_coord(row,col))*2;
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%0,%1,%2,%3}},[%4];"
   :"=r"({tag}_words(slice*4)),"=r"({tag}_words(slice*4+1)),"=r"({tag}_words(slice*4+2)),"=r"({tag}_words(slice*4+3)):"r"(addr):"memory");
 }}
'''
    assert marker in s
    s = s.replace(marker, code + marker, 1)
    s = s.replace('flash::gemm<true,-1>(smma,qa,kb,score);', 'flash::gemm<true,-1>(rs_mma,qr,kb,score);')
    if keep_dout:
        s = s.replace('flash::gemm<true,-1>(smma,da,vb,dp);', 'flash::gemm<true,-1>(rs_mma,dr,vb,dp);')
    s = s.replace('dq_tma', 'dq_tma_' + name)
    emit('dq', name, s)

base = (pkg / 'cuda/bias_fusion.cu').read_text()
s = base.replace('gemm_pair<true>(smma,ka,qb,score,va,dob,dp);', '''flash::gemm<true,-1>(smma,ka,qb,score);
      flash::gemm<true,-1>(smma,va,dob,dp);
      warpgroup_wait<1>();warpgroup_fence_operand(score);''')
assert s != base
s = s.replace('          float dd0=sm_load_f32(deltap+query0*4);\n          float dd1=sm_load_f32(deltap+(query0+1)*4);\n', '')
s = s.replace('dp(x)=pr*(dp(x)-(xi%2?dd1:dd0));score(x)=pr;', 'score(x)=pr;')
marker = '      #pragma unroll\n      for(int x=0;x<size(score);x+=2) {'
code = '''      warpgroup_wait<0>();warpgroup_fence_operand(dp);
      #pragma unroll
      for(int xb=0;xb<size(score);xb+=4) {
        int query0=get<1>(scoords(xb));
        float dd0=sm_load_f32(deltap+query0*4),dd1=sm_load_f32(deltap+(query0+1)*4);
        #pragma unroll
        for(int xi=0;xi<4;++xi)dp(xb+xi)=score(xb+xi)*(dp(xb+xi)-(xi%2?dd1:dd0));
      }
'''
assert s.count(marker) == 1
s = s.replace(marker, code + marker)
s = s.replace('grouped_dkdv', 'grouped_dkdv_prob_overlap')
emit('bias', 'rs8_prob_overlap', s)

# Reuse established harnesses with strict equality and fresh report locations.
s = (bwd / 'below80/check.py').read_text()
s = s.replace('assert max(errors) < .01, record', 'assert all(record["bitwise"]), record')
(root / 'check_native.py').write_text(s)
s = (bwd / 'core90/attribution.py').read_text()
s = s.replace('activities=[torch.profiler.ProfilerActivity.CUDA]',
              'activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]')
(root / 'attribution.py').write_text(s)
print('Created two dQ variants and one dK/dV overlap variant; baseline files', len(snapshot))
