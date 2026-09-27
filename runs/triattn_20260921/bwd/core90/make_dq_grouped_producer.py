from pathlib import Path

root=Path(__file__).resolve().parent.parent/'dq'
s=(root/'rs_group2/fused.cu').read_text()
t=(root/'rs_ldmatrix/fused.cu').read_text()
a=s.index('  #pragma unroll\n  for(int x=0;x<size(score);x+=2)');b=s.index('  auto acc_a=',a)
ta=t.index('  #pragma unroll\n  for(int xb=0;');tb=t.index('  auto acc_a=',ta)
s=s[:a]+t[ta:tb]+s[b:]
s=s.replace('__launch_bounds__(256,2)','__launch_bounds__(384,2)')
s=s.replace('wg=tid/128,row=', 'wg=tid/128-1,row=')
s=s.replace('s.empty[i].init(1);','s.empty[i].init(8);')
s=s.replace('   int slot=kt%2;', '   int slot=kt%2;\n   s.empty[slot].wait(((kt/2)%2)^1);')
a=s.index(' auto load=[&]');b=s.index(' int lane=tid%128;',a)
producer=s[a:b].replace('   load(0);','   for(int kt=0;kt<L/64;++kt)load(kt);')
s=s[:a]+''' if(tid<128){
  cutlass::arch::warpgroup_reg_dealloc<24>();
'''+producer+'''  return;
 }
 cutlass::arch::warpgroup_reg_alloc<104>();
'''+s[b:]
s=s.replace('  if(tid==0 && kt+1<L/64)load(kt+1);\n','')
s=s.replace('flash::gemm<false,0>(gmma,dsa,ktb,dq);__syncthreads();','flash::gemm<false,0>(gmma,dsa,ktb,dq);__syncwarp();if(lane%32==0)s.empty[slot].arrive();')
s=s.replace('cutlass::arch::fence_view_async_shared();__syncthreads();','cutlass::arch::fence_view_async_shared();cutlass::arch::NamedBarrier::sync(128,wg+1);')
s=s.replace('dim3(L/64,L/Config::R,4),256,','dim3(L/64,L/Config::R,4),384,')
d=root/'rs_group2_producer';d.mkdir(exist_ok=True);(d/'fused.cu').write_text(s)
