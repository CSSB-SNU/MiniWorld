from pathlib import Path

root=Path(__file__).resolve().parent.parent
s=(root/'dq/rs_coop_tma/fused.cu').read_text()
s=s.replace('s.full[i].init(1);','s.full[i].init(3);')
a=s.index(' auto load=[&]');b=s.index(' if(tid==0){',a)
s=s[:a]+''' auto load=[&](int kt){
   int slot=kt%2,role=tid/32;
   s.full[slot].arrive_and_expect_tx((role==2?4096:2048)*sizeof(Element));
   auto shape=make_shape(L,_32{},_4{},L);
   if(role==0){
     auto kg=p.k.get_tma_tensor(shape);auto kk=local_tile(kg(_,_,h,row),Shape<_64,_32>{},make_coord(kt,0));
     copy_tile(p.k,kk,make_tensor(make_smem_ptr(s.k[slot].data()),Config::QL{}),s.full[slot]);
   }else if(role==1){
     auto vg=p.v.get_tma_tensor(shape);auto vv=local_tile(vg(_,_,h,row),Shape<_64,_32>{},make_coord(kt,0));
     copy_tile(p.v,vv,make_tensor(make_smem_ptr(s.v[slot].data()),Config::QL{}),s.full[slot]);
   }else{
     auto bg=p.bias.get_tma_tensor(make_shape(L,L,_4{}));auto bb=local_tile(bg(_,_,h),Shape<_64,_64>{},make_coord(qt,kt));
     copy_tile(p.bias,bb,make_tensor(make_smem_ptr(s.bias[slot].data()),Config::SL{}),s.full[slot]);
   }
 };
'''+s[b:]
s=s.replace('   load(0);\n }',' }\n if(tid<96 && tid%32==0)load(0);')
s=s.replace('if(tid==0 && kt+1<L/64)load(kt+1);','if(tid<96 && tid%32==0 && kt+1<L/64)load(kt+1);')
d=root/'dq/rs_parallel_tma';d.mkdir(exist_ok=True);(d/'fused.cu').write_text(s)

s=(root/'bias_fusion/rs_r8_pair/grouped.cu').read_text()
s=s.replace('s.kv_full.init(1);','s.kv_full.init(C::NWG);')
s=s.replace('''    if(tid==0) {
      auto shape''','''    if(tid%32==0 && tid/32<C::NWG) {
      int producer=tid/32;
      auto shape''')
s=s.replace('s.kv_full.arrive_and_expect_tx(R*2*2048*sizeof(Element));','s.kv_full.arrive_and_expect_tx(C::RP*2*2048*sizeof(Element));')
s=s.replace('for(int r=0;r<R;++r) {','for(int rr=0;rr<C::RP;++rr) {\n        int r=producer*C::RP+rr;',1)
a=s.index('        s.b_empty[bs].wait');b=s.index('        for(int rr=0;rr<C::RP;++rr)',a)
s=s[:a]+'        if(producer==0){\n'+s[a:b]+'        }\n'+s[b:]
s=s.replace('          for(int w=0;w<C::NWG;++w) {','          {\n            int w=producer;',1)
d=root/'bias_fusion/rs_r8_parallel_tma';d.mkdir(exist_ok=True);(d/'grouped.cu').write_text(s)
