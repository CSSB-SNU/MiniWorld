from pathlib import Path
root=Path(__file__).resolve().parent.parent
s=(root/'dq/rs3/fused.cu').read_text()
s=s.replace('ClusterBarrier empty[2];','ClusterBarrier empty[2];')
s=s.replace('__launch_bounds__(256,3)', '__launch_bounds__(128,4)')
a=s.index(' __syncthreads();');b=s.index(' int lane=tid-128;',a)
s=s[:a]+''' __syncthreads();
 auto load=[&](int kt){
   int slot=kt%2;
   s.full[slot].arrive_and_expect_tx((2*2048+4096)*sizeof(Element));
   auto shape=make_shape(L,_32{},_4{},L);
   auto kg=p.k.get_tma_tensor(shape);auto vg=p.v.get_tma_tensor(shape);auto bg=p.bias.get_tma_tensor(make_shape(L,L,_4{}));
   auto kk=local_tile(kg(_,_,h,row),Shape<_64,_32>{},make_coord(kt,0));auto vv=local_tile(vg(_,_,h,row),Shape<_64,_32>{},make_coord(kt,0));auto bb=local_tile(bg(_,_,h),Shape<_64,_64>{},make_coord(qt,kt));
   copy_tile(p.k,kk,make_tensor(make_smem_ptr(s.k[slot].data()),Config::QL{}),s.full[slot]);copy_tile(p.v,vv,make_tensor(make_smem_ptr(s.v[slot].data()),Config::QL{}),s.full[slot]);copy_tile(p.bias,bb,make_tensor(make_smem_ptr(s.bias[slot].data()),Config::SL{}),s.full[slot]);
 };
 if(tid==0){
   auto shape=make_shape(L,_32{},_4{},L);auto qg=p.q.get_tma_tensor(shape);auto dog=p.dout.get_tma_tensor(shape);
   s.resident.arrive_and_expect_tx(2*2048*sizeof(Element)+2*64*sizeof(float));
   auto qtile=local_tile(qg(_,_,h,row),Shape<_64,_32>{},make_coord(qt,0));auto dotile=local_tile(dog(_,_,h,row),Shape<_64,_32>{},make_coord(qt,0));
   copy_tile(p.q,qtile,make_tensor(make_smem_ptr(s.q.data()),Config::QL{}),s.resident);copy_tile(p.dout,dotile,make_tensor(make_smem_ptr(s.dout.data()),Config::QL{}),s.resident);
   int stat=(h*L+row)*L+qt*64;
   SM90_BULK_COPY_G2S::copy(p.lse+stat,reinterpret_cast<uint64_t*>(&s.resident),s.lse.data(),64*sizeof(float));SM90_BULK_COPY_G2S::copy(p.delta+stat,reinterpret_cast<uint64_t*>(&s.resident),s.delta.data(),64*sizeof(float));
   load(0);
 }
'''+s[b:]
s=s.replace(' int lane=tid-128;', ' int lane=tid;')
s=s.replace(' for(int kt=0;kt<L/64;++kt){\n  int slot=', ' for(int kt=0;kt<L/64;++kt){\n  if(tid==0 && kt+1<L/64)load(kt+1);\n  int slot=')
s=s.replace('cutlass::arch::NamedBarrier::sync(128,1);if(lane==0)s.empty[slot].arrive();', '__syncthreads();')
s=s.replace('),256,sizeof(Config::Shared)', '),128,sizeof(Config::Shared)')
d=root/'dq/rs_coop';d.mkdir(exist_ok=True);(d/'fused.cu').write_text(s)
# Current register-source + vector partial store, compact static producer warp.
s=(root/'bias_fusion/rs_tma_store/grouped.cu').read_text().replace('__launch_bounds__(640,1)','__launch_bounds__(544,1)')
s=s.replace('wg=tid/128-1','wg=tid/128').replace('if(tid==0)','if(tid==512)').replace('if(tid<128)','if(tid>=512)').replace('grouped_dkdv<R><<<dim3(L/64,L/R,4),640,','grouped_dkdv<R><<<dim3(L/64,L/R,4),544,')
s=s.replace('    cutlass::arch::warpgroup_reg_dealloc<24>();\n','').replace('  cutlass::arch::warpgroup_reg_alloc<112>();\n','')
d=root/'bias_fusion/rs_tma_warp';d.mkdir(exist_ok=True);(d/'grouped.cu').write_text(s)
