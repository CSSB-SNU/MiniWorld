from pathlib import Path
root=Path(__file__).resolve().parent.parent
s=(root/'dq/vector_bias/fused.cu').read_text()
s=s.replace('using Grad=decltype(make_tiled_mma(GMMA::ss_op_selector', 'using Grad=decltype(make_tiled_mma(GMMA::rs_op_selector')
s=s.replace('bias[2],ds;', 'bias[2];')
s=s.replace(',dsp=cast_smem_ptr_to_uint(s.ds.data())','')
a=s.index('  #pragma unroll\n  for(int x=0;x<size(score);x+=2){int qr=');b=s.index('  flash::gemm<false,0>',a)
s=s[:a]+'''  auto acc_a=make_tensor(dp.data(),flash::convert_layout_acc_Aregs<Config::Grad>(dp.layout()));
  auto dsa=make_tensor_like<Element>(acc_a);flash::convert_type_out(acc_a,dsa);
  auto kk=make_tensor(make_smem_ptr(s.k[slot].data()),Config::KT{});auto ktb=gt.partition_fragment_B(kk);
'''+s[b:]
for variant in ['rs3','rs4']:
 v=s
 if variant=='rs4':
  v=v.replace('__launch_bounds__(256,3)','__launch_bounds__(160,4)').replace('if(tid==0)','if(tid==128)').replace('if(tid<128)','if(tid>=128)').replace('int lane=tid-128','int lane=tid').replace('warpgroup_reg_alloc<128>','warpgroup_reg_alloc<112>').replace('),256,sizeof(Config::Shared)', '),160,sizeof(Config::Shared)')
 d=root/'dq'/variant;d.mkdir(exist_ok=True);(d/'fused.cu').write_text(v)
for variant,base in [('rs','ldmatrix_bias'),('rs_double_vec4','ds_double_vec4')]:
 s=(root/'bias_fusion'/base/'grouped.cu').read_text()
 s=s.replace('using GradMMA=decltype(make_tiled_mma(GMMA::ss_op_selector','using GradMMA=decltype(make_tiled_mma(GMMA::rs_op_selector')
 s=s.replace('bias[2],prob[NWG],ds[NWG]', 'bias[2],ds[NWG]')
 s=s.replace('  auto sp=make_tensor(make_smem_ptr(s.prob[wg].data()),typename C::SL{});\n','')
 s=s.replace('uint32_t pp=cast_smem_ptr_to_uint(s.prob[wg].data()),ps=', 'uint32_t ps=')
 s=s.replace('  uint32_t pp=cast_smem_ptr_to_uint(s.prob[wg].data());\n','')
 s=s.replace('        sm_store_pair(pp+off,score(x),score(x+1));\n','')
 s=s.replace('      cutlass::arch::fence_view_async_shared();\n      cutlass::arch::NamedBarrier::sync(128,wg+1);\n','')
 s=s.replace('      auto pa=gt.partition_fragment_A(sp);auto dsa=gt.partition_fragment_A(ss);', '''      auto acc_p=make_tensor(score.data(),flash::convert_layout_acc_Aregs<typename C::GradMMA>(score.layout()));
      auto acc_ds=make_tensor(dp.data(),flash::convert_layout_acc_Aregs<typename C::GradMMA>(dp.layout()));
      auto pa=make_tensor_like<Element>(acc_p);auto dsa=make_tensor_like<Element>(acc_ds);
      flash::convert_type_out(acc_p,pa);flash::convert_type_out(acc_ds,dsa);''')
 d=root/'bias_fusion'/variant;d.mkdir(exist_ok=True);(d/'grouped.cu').write_text(s)
