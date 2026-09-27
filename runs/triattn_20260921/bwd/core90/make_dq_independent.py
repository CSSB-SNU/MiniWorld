from pathlib import Path

root=Path(__file__).resolve().parent.parent/'dq'
for rows in (2,4):
    s=(root/f'rs_group{rows}/fused.cu').read_text()
    t=(root/'rs_ldmatrix/fused.cu').read_text()
    a=s.index('  #pragma unroll\n  for(int x=0;x<size(score);x+=2)');b=s.index('  auto acc_a=',a)
    ta=t.index('  #pragma unroll\n  for(int xb=0;');tb=t.index('  auto acc_a=',ta)
    s=s[:a]+t[ta:tb]+s[b:]
    s=s.replace('resident,full[2];','resident,full[R][2],bias_full[2];')
    s=s.replace('s.resident.init(1);for(int i=0;i<2;++i){s.full[i].init(1);s.empty[i].init(1);}',
                's.resident.init(Config::R);for(int i=0;i<2;++i){for(int w=0;w<Config::R;++w)s.full[w][i].init(1);s.bias_full[i].init(1);s.empty[i].init(Config::R);}')
    a=s.index(' auto load=[&]');b=s.index(' int lane=tid%128;',a)
    s=s[:a]+''' auto load_kv=[&](int kt){
   int slot=kt%2;
   s.full[wg][slot].arrive_and_expect_tx(2*2048*sizeof(Element));
   auto shape=make_shape(L,_32{},_4{},L);
   auto kg=p.k.get_tma_tensor(shape);auto vg=p.v.get_tma_tensor(shape);
   auto kk=local_tile(kg(_,_,h,row),Shape<_64,_32>{},make_coord(kt,0));auto vv=local_tile(vg(_,_,h,row),Shape<_64,_32>{},make_coord(kt,0));
   copy_tile(p.k,kk,make_tensor(make_smem_ptr(s.k[wg][slot].data()),Config::QL{}),s.full[wg][slot]);
   copy_tile(p.v,vv,make_tensor(make_smem_ptr(s.v[wg][slot].data()),Config::QL{}),s.full[wg][slot]);
 };
 auto load_bias=[&](int kt){
   int slot=kt%2;s.empty[slot].wait(((kt/2)%2)^1);
   s.bias_full[slot].arrive_and_expect_tx(4096*sizeof(Element));
   auto bg=p.bias.get_tma_tensor(make_shape(L,L,_4{}));auto bb=local_tile(bg(_,_,h),Shape<_64,_64>{},make_coord(qt,kt));
   copy_tile(p.bias,bb,make_tensor(make_smem_ptr(s.bias[slot].data()),Config::SL{}),s.bias_full[slot]);
 };
 if(tid%128==0){
   auto shape=make_shape(L,_32{},_4{},L);auto qg=p.q.get_tma_tensor(shape);auto dog=p.dout.get_tma_tensor(shape);
   s.resident.arrive_and_expect_tx(2*2048*sizeof(Element)+2*64*sizeof(float));
   auto qtile=local_tile(qg(_,_,h,row),Shape<_64,_32>{},make_coord(qt,0));auto dotile=local_tile(dog(_,_,h,row),Shape<_64,_32>{},make_coord(qt,0));
   copy_tile(p.q,qtile,make_tensor(make_smem_ptr(s.q[wg].data()),Config::QL{}),s.resident);copy_tile(p.dout,dotile,make_tensor(make_smem_ptr(s.dout[wg].data()),Config::QL{}),s.resident);
   int stat=(h*L+row)*L+qt*64;
   SM90_BULK_COPY_G2S::copy(p.lse+stat,reinterpret_cast<uint64_t*>(&s.resident),s.lse[wg].data(),64*sizeof(float));SM90_BULK_COPY_G2S::copy(p.delta+stat,reinterpret_cast<uint64_t*>(&s.resident),s.delta[wg].data(),64*sizeof(float));
   load_kv(0);
 }
 if(tid==0)load_bias(0);
'''+s[b:]
    s=s.replace('  if(tid==0 && kt+1<L/64)load(kt+1);', '  if(lane==0 && kt+1<L/64)load_kv(kt+1);\n  if(tid==0 && kt+1<L/64)load_bias(kt+1);')
    s=s.replace('s.full[slot].wait(phase);','s.full[wg][slot].wait(phase);s.bias_full[slot].wait(phase);')
    s=s.replace('flash::gemm<false,0>(gmma,dsa,ktb,dq);__syncthreads();','flash::gemm<false,0>(gmma,dsa,ktb,dq);cutlass::arch::NamedBarrier::sync(128,wg+1);if(lane==0)s.empty[slot].arrive();')
    s=s.replace('cutlass::arch::fence_view_async_shared();__syncthreads();','cutlass::arch::fence_view_async_shared();cutlass::arch::NamedBarrier::sync(128,wg+1);')
    d=root/f'rs_group{rows}_independent';d.mkdir(exist_ok=True);(d/'fused.cu').write_text(s)
