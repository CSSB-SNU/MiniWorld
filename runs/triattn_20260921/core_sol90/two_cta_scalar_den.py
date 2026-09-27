"""N32 PV plus scalar sum of rounded P; different FP32 reduction order."""
def transform(s,den_mma=False):
 old='using PV=decltype(make_tiled_mma(SM90::GMMA::MMA_64x40x16_F32BF16BF16_RS<GMMA::Major::K,GMMA::Major::MN>{}));'
 assert old in s;s=s.replace(old,'using PV=PVBase;')
 s=s.replace('Shape<_64,_40>','Shape<_64,_32>')
 s=s.replace(' alignas(1024) Element ones[N*D];\n','')
 s=s.replace(' for(int x=tid;x<N*D;x+=Threads)s.ones[x]=Element(1.f);\n','')
 s=s.replace('bias[3][64*N]','bias[4][64*N]').replace('bf[3]','bf[4]').replace('be[3]','be[4]')
 s=s.replace('for(int st=0;st<3;++st)','for(int st=0;st<4;++st)')
 s=s.replace('seq%3','seq%4').replace('seq/3','seq/4').replace('seq>=3','seq>=4')
 old='''  auto d0=make_tensor(acc[0].data()+16,Den{}.layout());
  decltype(d0) den[2]={d0,make_tensor(acc[1].data()+16,Den{}.layout())};'''
 assert old in s;s=s.replace(old,'  Den den[2];')
 a=s.index('   auto raw=tp.partition_fragment_B(vs);')
 b=s.index('\n   ',s.index('auto vb=make_tensor(GMMA::DescriptorIterator{desc},raw.layout());',a))
 s=s[:a]+'''   auto vb=tp.partition_fragment_B(vs);
   auto bits=recast<uint32_t>(prob);
   auto lr=make_tensor(den[hh].data(),flash::convert_layout_acc_rowcol(den[hh].layout()));
   #pragma unroll
   for(int row=0;row<2;++row){
    float sum=0.f;
    #pragma unroll
    for(int pair=0;pair<4;++pair){
     uint32_t u=bits(2*pair+row);
     auto bf=*reinterpret_cast<__nv_bfloat162 const*>(&u);
     float2 f=__bfloat1622float2(bf);
     sum=__fadd_rn(sum,__fadd_rn(f.x,f.y));
    }
    sum=__fadd_rn(sum,__shfl_xor_sync(0xffffffffu,sum,1));
    sum=__fadd_rn(sum,__shfl_xor_sync(0xffffffffu,sum,2));
    lr(row,0)=__fadd_rn(lr(row,0),sum);
   }
 '''+s[b:]
 if den_mma:
  s=s.replace('struct Shared {','struct Shared {\n alignas(128) Element ones[8*N];')
  marker=' cutlass::arch::fence_view_async_shared();__syncthreads();'
  assert marker in s
  s=s.replace(marker,' for(int x=tid;x<8*N;x+=Threads)s.ones[x]=Element(1.f);\n'+marker)
  marker=' auto tq=qk.get_slice(0);auto tp=pvbase.get_slice(0);'
  assert marker in s
  s=s.replace(marker,marker+'\n auto tl=ls.get_slice(0);auto lb=tl.partition_fragment_B(make_tensor(make_smem_ptr(s.ones),S1{}));')
  a=s.index('   auto bits=recast<uint32_t>(prob);')
  b=s.index('   warpgroup_fence_operand(prob);warpgroup_arrive();',a)
  s=s[:a]+s[b:]
  old='   for(int kk=0;kk<2;++kk)gemm(pv,prob(_,_,kk),vb(_,_,kk),acc[hh]);'
  assert old in s
  s=s.replace(old,old+'\n   #pragma unroll\n   for(int kk=0;kk<2;++kk)gemm(ls,prob(_,_,kk),lb(_,_,kk),den[hh]);')
 return s
