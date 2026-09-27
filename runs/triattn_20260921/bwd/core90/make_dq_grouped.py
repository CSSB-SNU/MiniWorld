from pathlib import Path

root=Path(__file__).resolve().parent.parent/'dq'
for rows in (2,4):
    s=(root/'rs_coop_tma/fused.cu').read_text()
    s=s.replace('struct Config {',f'struct Config {{\n static constexpr int R={rows};')
    s=s.replace('q,dout,k[2],v[2];','q[R],dout[R],k[R][2],v[R][2];')
    s=s.replace('lse,delta;','lse[R],delta[R];')
    s=s.replace('__launch_bounds__(128,4)',f'__launch_bounds__({rows*128},{4//rows})')
    s=s.replace('int tid=threadIdx.x,row=blockIdx.y,h=', 'int tid=threadIdx.x,wg=tid/128,row=blockIdx.y*Config::R+wg,h=')
    a=s.index(' auto load=[&]'); b=s.index(' int lane=tid;',a)
    s=s[:a]+''' auto load=[&](int kt){
   int slot=kt%2;
   s.full[slot].arrive_and_expect_tx((Config::R*2*2048+4096)*sizeof(Element));
   auto shape=make_shape(L,_32{},_4{},L);
   auto kg=p.k.get_tma_tensor(shape);auto vg=p.v.get_tma_tensor(shape);auto bg=p.bias.get_tma_tensor(make_shape(L,L,_4{}));
   auto bb=local_tile(bg(_,_,h),Shape<_64,_64>{},make_coord(qt,kt));
   copy_tile(p.bias,bb,make_tensor(make_smem_ptr(s.bias[slot].data()),Config::SL{}),s.full[slot]);
   #pragma unroll
   for(int w=0;w<Config::R;++w){
     int r=blockIdx.y*Config::R+w;
     auto kk=local_tile(kg(_,_,h,r),Shape<_64,_32>{},make_coord(kt,0));auto vv=local_tile(vg(_,_,h,r),Shape<_64,_32>{},make_coord(kt,0));
     copy_tile(p.k,kk,make_tensor(make_smem_ptr(s.k[w][slot].data()),Config::QL{}),s.full[slot]);copy_tile(p.v,vv,make_tensor(make_smem_ptr(s.v[w][slot].data()),Config::QL{}),s.full[slot]);
   }
 };
 if(tid==0){
   auto shape=make_shape(L,_32{},_4{},L);auto qg=p.q.get_tma_tensor(shape);auto dog=p.dout.get_tma_tensor(shape);
   s.resident.arrive_and_expect_tx(Config::R*(2*2048*sizeof(Element)+2*64*sizeof(float)));
   #pragma unroll
   for(int w=0;w<Config::R;++w){
     int r=blockIdx.y*Config::R+w;
     auto qtile=local_tile(qg(_,_,h,r),Shape<_64,_32>{},make_coord(qt,0));auto dotile=local_tile(dog(_,_,h,r),Shape<_64,_32>{},make_coord(qt,0));
     copy_tile(p.q,qtile,make_tensor(make_smem_ptr(s.q[w].data()),Config::QL{}),s.resident);copy_tile(p.dout,dotile,make_tensor(make_smem_ptr(s.dout[w].data()),Config::QL{}),s.resident);
     int stat=(h*L+r)*L+qt*64;
     SM90_BULK_COPY_G2S::copy(p.lse+stat,reinterpret_cast<uint64_t*>(&s.resident),s.lse[w].data(),64*sizeof(float));SM90_BULK_COPY_G2S::copy(p.delta+stat,reinterpret_cast<uint64_t*>(&s.resident),s.delta[w].data(),64*sizeof(float));
   }
   load(0);
 }
'''+s[b:]
    s=s.replace('int lane=tid;', 'int lane=tid%128;')
    for var in ('q','dout','lse','delta'):
        s=s.replace(f's.{var}.data()',f's.{var}[wg].data()')
    for var in ('k','v'):
        s=s.replace(f's.{var}[slot].data()',f's.{var}[wg][slot].data()')
    s=s.replace('dim3(L/64,L,4),128,',f'dim3(L/64,L/Config::R,4),{rows*128},')
    d=root/f'rs_group{rows}';d.mkdir(exist_ok=True);(d/'fused.cu').write_text(s)
