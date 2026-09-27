"""Overlap deterministic R8 bias reduction with the next query tile.

Two producer warps issue TMA as before. Their two idle sibling warps reduce dS.
An explicit two-stage ready/empty protocol retains every shared-memory reader
barrier. No global intermediate size or rounding order changes.
"""
from pathlib import Path

root = Path(__file__).resolve().parent.parent / 'bias_fusion'
s = (root / 'rs8_producer32/grouped.cu').read_text()
s = s.replace('cutlass::arch::ClusterBarrier q_empty',
              'cutlass::arch::ClusterBarrier ds_ready[2],ds_empty[2];\n    cutlass::arch::ClusterBarrier q_empty')
s = s.replace('s.b_full[st].init(1);s.b_empty[st].init(C::NWG);',
              's.b_full[st].init(1);s.b_empty[st].init(C::NWG);\n      s.ds_ready[st].init(C::NWG);s.ds_empty[st].init(1);')
s = s.replace('warpgroup_reg_dealloc<32>()', 'warpgroup_reg_dealloc<56>()')
s = s.replace('warpgroup_reg_alloc<232>()', 'warpgroup_reg_alloc<224>()')
start = s.index('    cutlass::arch::NamedBarrier::sync(256,5);')
end = s.index('    // The other dS slot', start)
reduce = s[start:end]
# Both reducers form one 64-thread group. Process eight float4 chunks per lane
# while preserving the installed per-coordinate outer-row addition order.
ra = reduce.index('        int key=')
rb = reduce.index('        values[local]=', ra)
sum_code = reduce[ra:rb]
reducer = '''    if(tid>=64){
      int rl=tid-64;
      for(int jt=0;jt<nt;++jt){
        int bs=jt%2,phase=(jt/2)%2;
        s.ds_ready[bs].wait(phase);
        int64_t base=(((int64_t(h)*(L/R)+group)*nt+jt)*nkt+kt)*2048;
        #pragma unroll
        for(int chunk=0;chunk<8;++chunk){
          int vec=chunk*64+rl;
          int lane=vec%128,xbase=(vec/128)*4;
          float values[4];
          #pragma unroll
          for(int local=0;local<4;local+=2){
            int x=xbase+local;
''' + sum_code + '''            values[local]=sum0;values[local+1]=sum1;
          }
          reinterpret_cast<float4*>(p.db+base)[vec]=make_float4(values[0],values[1],values[2],values[3]);
        }
        cutlass::arch::NamedBarrier::sync(64,6);
        if(rl==0)s.ds_empty[bs].arrive();
      }
    }
'''
s = s[:start] + '''    // The last per-WG reader barrier also publishes every dS store.
    if(lane==0)s.ds_ready[bs].arrive();
''' + s[end:]
s = s.replace('    return;\n  }\n  cutlass::arch::warpgroup_reg_alloc',
              reducer + '    return;\n  }\n  cutlass::arch::warpgroup_reg_alloc', 1)
# Only consumer outer loop gets the empty wait; producer bias loop is distinct.
s = s.replace('    s.b_full[bs].wait(bphase);',
              '    s.ds_empty[bs].wait(bphase^1);\n    s.b_full[bs].wait(bphase);')
s = s.replace('// The other dS slot protects readers until the next pre-reduction barrier.',
              '// Bias input can retire; dS reuse waits for the independent reducer.')
target = root / 'rs8_async_bias64'
target.mkdir(exist_ok=True)
(target / 'grouped.cu').write_text(s)
