from pathlib import Path
root=Path(__file__).resolve().parent.parent/'bias_fusion'
s=(root/'ldmatrix_bias/grouped.cu').read_text()
s=s.replace('bias[2],prob[NWG],ds[NWG];','bias[2],prob[NWG],ds[NWG][2];')
s=s.replace('  auto ss=make_tensor(make_smem_ptr(s.ds[wg].data()),typename C::SL{});\n','')
s=s.replace('uint32_t pp=cast_smem_ptr_to_uint(s.prob[wg].data()),ps=cast_smem_ptr_to_uint(s.ds[wg].data());','uint32_t pp=cast_smem_ptr_to_uint(s.prob[wg].data());')
s=s.replace('    s.b_full[bs].wait(bphase);','    s.b_full[bs].wait(bphase);\n    auto ss=make_tensor(make_smem_ptr(s.ds[wg][bs].data()),typename C::SL{});\n    uint32_t ps=cast_smem_ptr_to_uint(s.ds[wg][bs].data());')
s=s.replace('s.ds[w].data()', 's.ds[w][bs].data()')
s=s.replace('    cutlass::arch::NamedBarrier::sync(512,5);\n    if(lane==0) s.b_empty[bs].arrive();','    // The other dS slot protects readers until the next pre-reduction barrier.\n    if(lane==0) s.b_empty[bs].arrive();')
# Keep the WG-local release fence in the first candidate to isolate lifetime change.
d=root/'ds_double';d.mkdir(exist_ok=True);(d/'grouped.cu').write_text(s)
v=s.replace('''      #pragma unroll
      for(int local=0;local<4;local+=2) {''','''      float values[4];
      #pragma unroll
      for(int local=0;local<4;local+=2) {''')
v=v.replace('        p.db[idx]=sum0;p.db[idx+128]=sum1;','        values[local]=sum0;values[local+1]=sum1;')
v=v.replace('''    // The other dS slot protects readers''','''    // The other dS slot protects readers''')
old='''        values[local]=sum0;values[local+1]=sum1;
      }
    }'''
new='''        values[local]=sum0;values[local+1]=sum1;
      }
      int64_t base=(((int64_t(h)*(L/R)+group)*nt+jt)*nkt+kt)*2048;
      reinterpret_cast<float4*>(p.db+base)[wg*128+lane]=make_float4(values[0],values[1],values[2],values[3]);
    }'''
assert old in v;v=v.replace(old,new)
v=v.replace('lane=idx%128,x=idx/128;', 'lane=(idx/4)%128,x=(idx/512)*4+((idx%4)/2)*2+(idx%2);')
d=root/'ds_double_vec4';d.mkdir(exist_ok=True);(d/'grouped.cu').write_text(v)
