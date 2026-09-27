from pathlib import Path
root=Path(__file__).resolve().parent.parent/'bias_fusion'
s=(root/'rs_double_vec4/grouped.cu').read_text()
s=s.replace('static_assert(R==4);','static_assert(R==8);').replace('ds[NWG][2];','ds[NWG][2][RP];')
s=s.replace('wg=tid/128-1','wg=tid/128').replace('__launch_bounds__(640,1)','__launch_bounds__(512,1)')
a=s.index('  if(tid<128) {');b=s.index('  typename C::ScoreMMA',a)
s=s[:a]+'''  auto load_query=[&](int it){
    int jt=it/C::RP,rr=it%C::RP,bs=jt%2,bphase=(jt/2)%2,st=it%2,phase=(it/2)%2;
    auto shape=make_shape(L,_32{},_4{},L);
    if(rr==0){
      s.b_empty[bs].wait(bphase^1);s.b_full[bs].arrive_and_expect_tx(2048*sizeof(Element));
      auto mb=p.bias.get_tma_tensor(make_shape(L,L,_4{}));auto gb=local_tile(mb(_,_,h),Shape<_32,_64>{},make_coord(jt,kt));
      tma_copy(p.bias,gb,make_tensor(make_smem_ptr(s.bias[bs].data()),typename C::BL{}),s.b_full[bs]);
    }
    auto mq=p.q.get_tma_tensor(shape);auto md=p.dout.get_tma_tensor(shape);
    #pragma unroll
    for(int w=0;w<C::NWG;++w){
      int row=group*R+w*C::RP+rr;
      s.q_empty[w][st].wait(phase^1);s.q_full[w][st].arrive_and_expect_tx(2*1024*sizeof(Element)+2*32*sizeof(float));
      auto gq=local_tile(mq(_,_,h,row),Shape<_32,_32>{},make_coord(jt,0));auto gd=local_tile(md(_,_,h,row),Shape<_32,_32>{},make_coord(jt,0));
      tma_copy(p.q,gq,make_tensor(make_smem_ptr(s.q[w][st].data()),typename C::QL{}),s.q_full[w][st]);
      tma_copy(p.dout,gd,make_tensor(make_smem_ptr(s.dout[w][st].data()),typename C::QL{}),s.q_full[w][st]);
      int stat=(h*L+row)*L+jt*32;
      SM90_BULK_COPY_G2S::copy(p.lse+stat,reinterpret_cast<uint64_t*>(&s.q_full[w][st]),s.lse[w][st].data(),32*sizeof(float));
      SM90_BULK_COPY_G2S::copy(p.delta+stat,reinterpret_cast<uint64_t*>(&s.q_full[w][st]),s.delta[w][st].data(),32*sizeof(float));
    }
  };
  if(tid==0){
    auto shape=make_shape(L,_32{},_4{},L);auto mk=p.k.get_tma_tensor(shape);auto mv=p.v.get_tma_tensor(shape);
    s.kv_full.arrive_and_expect_tx(R*2*2048*sizeof(Element));
    #pragma unroll
    for(int r=0;r<R;++r){auto gk=local_tile(mk(_,_,h,group*R+r),Shape<_64,_32>{},make_coord(kt,0));auto gv=local_tile(mv(_,_,h,group*R+r),Shape<_64,_32>{},make_coord(kt,0));tma_copy(p.k,gk,make_tensor(make_smem_ptr(s.k[r].data()),typename C::KL{}),s.kv_full);tma_copy(p.v,gv,make_tensor(make_smem_ptr(s.v[r].data()),typename C::KL{}),s.kv_full);}
    load_query(0);
  }
'''+s[b:]
s=s.replace('    auto ss=make_tensor(make_smem_ptr(s.ds[wg][bs].data()),typename C::SL{});\n','').replace('    uint32_t ps=cast_smem_ptr_to_uint(s.ds[wg][bs].data());\n','')
s=s.replace('      int it=jt*C::RP+rr,slot=it%2,phase=(it/2)%2;', '''      int it=jt*C::RP+rr,slot=it%2,phase=(it/2)%2;
      if(tid==0 && it+1<nt*C::RP)load_query(it+1);
      uint32_t ps=cast_smem_ptr_to_uint(s.ds[wg][bs][rr].data());''')
old='''          uint32_t pair,ptr=cast_smem_ptr_to_uint(s.ds[w][bs].data())+off;
          asm volatile("ld.shared.b32 %0,[%1];":"=r"(pair):"r"(ptr):"memory");
          sum0+=float(Element::bitcast(uint16_t(pair)));
          sum1+=float(Element::bitcast(uint16_t(pair>>16)));'''
new='''          #pragma unroll
          for(int rr=0;rr<C::RP;++rr){
            uint32_t pair,ptr=cast_smem_ptr_to_uint(s.ds[w][bs][rr].data())+off;
            asm volatile("ld.shared.b32 %0,[%1];":"=r"(pair):"r"(ptr):"memory");
            sum0+=float(Element::bitcast(uint16_t(pair)));
            sum1+=float(Element::bitcast(uint16_t(pair>>16)));
          }'''
assert old in s;s=s.replace(old,new)
s=s.replace('grouped_dkdv<R><<<dim3(L/64,L/R,4),640,', 'grouped_dkdv<R><<<dim3(L/64,L/R,4),512,')
s=s.replace('return launch<4>', 'return launch<8>').replace('sizeof(Config<4>::Shared)','sizeof(Config<8>::Shared)')
s=s.replace('TORCH_CHECK(R==4,"row group4 required");','TORCH_CHECK(R==4 || R==8,"serving ABI4 or explicit row-group8 required"); // ABI4 retained for A/B harness; actual group is8.')
d=root/'rs_r8_coop';d.mkdir(exist_ok=True);(d/'grouped.cu').write_text(s)
