from pathlib import Path

root=Path(__file__).resolve().parent.parent/'bias_fusion'
s=(root/'rs_r8_two/grouped.cu').read_text()
s=s.replace('NWG=2, RP=R/NWG','NWG=4, RP=R/NWG')
s=s.replace('__launch_bounds__(384,1)','__launch_bounds__(640,1)')
s=s.replace('warpgroup_reg_alloc<232>()','warpgroup_reg_alloc<112>()')
s=s.replace('NamedBarrier::sync(256,5)','NamedBarrier::sync(512,5)')
s=s.replace('float values[8];','float values[4];').replace('local<8;local+=2','local<4;local+=2').replace('int x=wg*8+local;','int x=wg*4+local;')
s=s.replace('''reinterpret_cast<float4*>(p.db+base)[wg*256+lane]=make_float4(values[0],values[1],values[2],values[3]);
      reinterpret_cast<float4*>(p.db+base)[wg*256+128+lane]=make_float4(values[4],values[5],values[6],values[7]);''','''reinterpret_cast<float4*>(p.db+base)[wg*128+lane]=make_float4(values[0],values[1],values[2],values[3]);''')
s=s.replace('grouped_dkdv<R><<<dim3(L/64,L/R,4),384,','grouped_dkdv<R><<<dim3(L/64,L/R,4),640,')
s=s.replace('    // dBias reduction reads', '    array_aligned<float,2048,128> probability[NWG];\n    // dBias reduction reads')
s=s.replace('      auto dp=partition_fragment_C(smma,Shape<_64,_32>{});\n','')
s=s.replace('''      flash::gemm<true,-1>(smma,ka,qb,score);
      flash::gemm<true,0>(smma,va,dob,dp);''','''      flash::gemm<true,0>(smma,ka,qb,score);''')
s=s.replace('          float dd0=sm_load_f32(deltap+query0*4);\n','').replace('          float dd1=sm_load_f32(deltap+(query0+1)*4);\n','')
s=s.replace('            dp(x)=pr*(dp(x)-(xi%2?dd1:dd0));score(x)=pr;','            score(x)=pr;')
needle='''      #pragma unroll
      for(int x=0;x<size(score);x+=2) {'''
replacement='''      // FP32 P remains on chip, freeing the first score fragment before dP.
      uint32_t pp=cast_smem_ptr_to_uint(s.probability[wg].data())+lane*sizeof(float);
      #pragma unroll
      for(int x=0;x<size(score);++x)sm_store_f32(pp+x*128*4,score(x));
      flash::gemm<true,0>(smma,va,dob,score);
      auto& dp=score;
      #pragma unroll
      for(int x=0;x<size(dp);++x){int query=get<1>(scoords(x));dp(x)=sm_load_f32(pp+x*128*4)*(dp(x)-sm_load_f32(deltap+query*4));}
      #pragma unroll
      for(int x=0;x<size(score);x+=2) {'''
assert s.count(needle)==1;s=s.replace(needle,replacement)
s=s.replace('      auto acc_p=make_tensor(score.data(),flash::convert_layout_acc_Aregs<typename C::GradMMA>(score.layout()));\n','')
s=s.replace('auto pa=make_tensor_like<Element>(acc_p);','auto pa=make_tensor_like<Element>(acc_ds);')
s=s.replace('      flash::convert_type_out(acc_p,pa);flash::convert_type_out(acc_ds,dsa);','''      #pragma unroll
      for(int x=0;x<size(pa);++x)pa.data()[x]=Element(sm_load_f32(pp+x*128*4));
      flash::convert_type_out(acc_ds,dsa);''')
s=s.replace('// One TMA producer, four consumer warpgroups; each WG owns one outer row.','// One TMA producer, four consumer warpgroups; each WG owns two outer rows.')
d=root/'rs_r8_staged_p';d.mkdir(exist_ok=True);(d/'grouped.cu').write_text(s)

# A full four-WG cooperative CTA has a static 128-register ceiling, avoiding
# producer register-transfer overhead while preserving the same tile lifetime.
t=(root/'rs_r8_coop/grouped.cu').read_text()
a=t.index('  auto load_query=');b=t.index('  typename C::ScoreMMA',a)
start=s.index('  if(tid<128) {');end=s.index('  typename C::ScoreMMA',start)
s=s[:start]+t[a:b]+s[end:]
s=s.replace('wg=tid/128-1','wg=tid/128').replace('__launch_bounds__(640,1)','__launch_bounds__(512,1)')
s=s.replace('      int it=jt*C::RP+rr,slot=it%2,phase=(it/2)%2;', '      int it=jt*C::RP+rr,slot=it%2,phase=(it/2)%2;\n      if(tid==0 && it+1<nt*C::RP)load_query(it+1);')
s=s.replace('grouped_dkdv<R><<<dim3(L/64,L/R,4),640,','grouped_dkdv<R><<<dim3(L/64,L/R,4),512,')
s=s.replace('// One TMA producer, four consumer warpgroups; each WG owns two outer rows.','// Four cooperative consumer warpgroups; each WG owns two outer rows.')
d=root/'rs_r8_staged_coop';d.mkdir(exist_ok=True);(d/'grouped.cu').write_text(s)
