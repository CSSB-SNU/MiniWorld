from pathlib import Path

root = Path(__file__).resolve().parent.parent

# Combine the independently successful register-source MMA and TMA epilogue
# with a cooperative four-warp CTA. Retire Q only after its last consumer.
s = (root/'dq/rs_coop/fused.cu').read_text()
t = (root/'dq/rs_tma_store/fused.cu').read_text()
line = next(x for x in t.splitlines(True) if 'using TO=' in x)
s = s.replace(' using Score=', line+' using Score=')
s = s.replace('Element* dq;int L;};', 'Element* dq;int L;TO out;};')
a = s.index(' #pragma unroll\n for(int x=0;x<size(dq);++x)')
b = s.index('\n}\ntorch::Tensor backward', a)
ta = t.index(' uint32_t op=cast_smem_ptr_to_uint(s.q.data());')
tb = t.index('\n}\ntorch::Tensor backward', ta)
s = s[:a]+t[ta:tb].replace('cutlass::arch::NamedBarrier::sync(128,1);','__syncthreads();')+s[b:]
line = next(x for x in t.splitlines(True) if 'p.out=make_tma_copy' in x)
s = s.replace(' C10_CUDA_CHECK(cudaFuncSetAttribute(dq_tma', line+' C10_CUDA_CHECK(cudaFuncSetAttribute(dq_tma')
s = s.replace('(Element*)result.data_ptr(),L};','(Element*)result.data_ptr(),L,{}};')
d = root/'dq/rs_coop_tma'; d.mkdir(exist_ok=True); (d/'fused.cu').write_text(s)

# R4 cooperative producer, retaining four consumer warpgroups. No dynamic
# register transfer and no persistent extra-row gradient registers.
s = (root/'bias_fusion/rs_r8_coop/grouped.cu').read_text()
s = s.replace('static_assert(R==8);','static_assert(R==4);')
s = s.replace('return launch<8>', 'return launch<4>').replace('sizeof(Config<8>::Shared)','sizeof(Config<4>::Shared)')
s = s.replace('TORCH_CHECK(R==4 || R==8,"serving ABI4 or explicit row-group8 required"); // ABI4 retained for A/B harness; actual group is8.', 'TORCH_CHECK(R==4,"row group4 required");')
t = (root/'bias_fusion/rs_tma_store/grouped.cu').read_text()
line = next(x for x in t.splitlines(True) if 'using TO=' in x)
s = s.replace('  using ScoreMMA=', line+'  using ScoreMMA=')
s = s.replace('    int L;\n  };','    int L; TO outk,outv;\n  };')
a = s.index('  #pragma unroll\n  for(int rr=0;rr<C::RP;++rr) {',s.index('void grouped_dkdv'))
b = s.index('\n}\n\n__global__ void reduce_bias', a)
ta = t.index('  #pragma unroll\n  for(int rr=0;rr<C::RP;++rr) {',t.index('void grouped_dkdv'))
tb = t.index('\n}\n\n__global__ void reduce_bias', ta)
s = s[:a]+t[ta:tb]+s[b:]
ta = t.index('  auto make_out='); tb = t.index('  C10_CUDA_CHECK(cudaFuncSetAttribute(grouped_dkdv',ta)
s = s.replace('  C10_CUDA_CHECK(cudaFuncSetAttribute(grouped_dkdv',t[ta:tb]+'  C10_CUDA_CHECK(cudaFuncSetAttribute(grouped_dkdv')
s = s.replace('part.data_ptr<float>(),L};','part.data_ptr<float>(),L,{},{}};')
d = root/'bias_fusion/rs_coop_tma'; d.mkdir(exist_ok=True); (d/'grouped.cu').write_text(s)

# R8 with two consumer warpgroups. Four rows per WG use the larger register
# allocation; bias partials are halved without any extra global tensor.
s = (root/'bias_fusion/rs_tma_store/grouped.cu').read_text()
s = s.replace('NWG=4, RP=R/NWG','NWG=2, RP=R/NWG').replace('static_assert(R==4);','static_assert(R==8);')
s = s.replace('ds[NWG][2];','ds[NWG][2][RP];')
s = s.replace('__launch_bounds__(640,1)','__launch_bounds__(384,1)').replace('warpgroup_reg_alloc<112>()','warpgroup_reg_alloc<232>()')
s = s.replace('    auto ss=make_tensor(make_smem_ptr(s.ds[wg][bs].data()),typename C::SL{});\n','')
s = s.replace('    uint32_t ps=cast_smem_ptr_to_uint(s.ds[wg][bs].data());\n','')
s = s.replace('      constexpr int rr=decltype(rr_)::value;', '      constexpr int rr=decltype(rr_)::value;\n      uint32_t ps=cast_smem_ptr_to_uint(s.ds[wg][bs][rr].data());')
s = s.replace('NamedBarrier::sync(512,5)','NamedBarrier::sync(256,5)')
s = s.replace('float values[4];','float values[8];').replace('local<4;local+=2','local<8;local+=2').replace('int x=wg*4+local;','int x=wg*8+local;')
old = '''          uint32_t pair,ptr=cast_smem_ptr_to_uint(s.ds[w][bs].data())+off;
          asm volatile("ld.shared.b32 %0,[%1];":"=r"(pair):"r"(ptr):"memory");
          sum0+=float(Element::bitcast(uint16_t(pair)));
          sum1+=float(Element::bitcast(uint16_t(pair>>16)));'''
new = '''          #pragma unroll
          for(int rr=0;rr<C::RP;++rr){
            uint32_t pair,ptr=cast_smem_ptr_to_uint(s.ds[w][bs][rr].data())+off;
            asm volatile("ld.shared.b32 %0,[%1];":"=r"(pair):"r"(ptr):"memory");
            sum0+=float(Element::bitcast(uint16_t(pair)));
            sum1+=float(Element::bitcast(uint16_t(pair>>16)));
          }'''
assert old in s; s = s.replace(old,new)
s = s.replace('reinterpret_cast<float4*>(p.db+base)[wg*128+lane]=make_float4(values[0],values[1],values[2],values[3]);', '''reinterpret_cast<float4*>(p.db+base)[wg*256+lane]=make_float4(values[0],values[1],values[2],values[3]);
      reinterpret_cast<float4*>(p.db+base)[wg*256+128+lane]=make_float4(values[4],values[5],values[6],values[7]);''')
s = s.replace('grouped_dkdv<R><<<dim3(L/64,L/R,4),640,', 'grouped_dkdv<R><<<dim3(L/64,L/R,4),384,')
s = s.replace('return launch<4>', 'return launch<8>').replace('sizeof(Config<4>::Shared)','sizeof(Config<8>::Shared)')
s = s.replace('TORCH_CHECK(R==4,"row group4 required");','TORCH_CHECK(R==4 || R==8,"ABI4 or actual group8 required");')
d = root/'bias_fusion/rs_r8_two'; d.mkdir(exist_ok=True); (d/'grouped.cu').write_text(s)
