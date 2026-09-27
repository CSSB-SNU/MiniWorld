TMN_DEVI void dnorm(const Params& p,int row,uint8_t* sm,uint64_t* bars,int& phase){
 int tid=threadIdx.x,wg=tid/128,lane=tid%32,warp=(tid/32)%4;
 if(tid==0){mbar_arrive_expect_tx(bars+1,32768);for(int k=0;k<4;++k)tma_load_2d(sm+DN+k*8192,&p.dp,bars+1,k*64,row);}
 mbar_wait(bars+1,(row/(gridDim.x*64))&1);__syncthreads();
 auto load=[&](int k,int slot){mbar_arrive_expect_tx(bars+2+slot,65536);for(int g=0;g<8;++g)tma_load_2d(sm+WB+slot*65536+g*8192,&p.wp,bars+2+slot,g*64,k);};
 if(tid==0)load(0,0);float v[128]={};
 #pragma unroll
 for(int k=0;k<4;++k){int slot=k%2;
  if(tid==0&&k<3)load((k+1)*64,1-slot);
  mbar_wait(bars+2+slot,(phase>>slot)&1);phase^=1<<slot;__syncthreads();
  fence_regs(v);wgmma_fence();
  static_for<4>([&](auto qq){constexpr int q=decltype(qq)::value;
   mma256<0,1>(v,smem_desc(smem_u32(sm+DN+k*8192+q*32),16,1024,1),smem_desc(smem_u32(sm+WB+slot*65536+wg*32768+q*2048),8192,1024,1),k>0||q>0);
  });wgmma_commit();wgmma_wait<0>();fence_regs(v);__syncthreads();
 }
 for(int j=0;j<128;++j){int r=warp*16+lane/4+8*((j/2)&1),c=(j/8)*16+2*(lane%4)+8*((j/2)%4/2)+(j%2);reinterpret_cast<bf*>(sm+DN+wg*32768+(c/64)*8192)[swz128(r,(c%64)*2)/2]=__float2bfloat16_rn(v[j]);}
 __syncthreads();
}
