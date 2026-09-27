// Bound WGMMA operand live ranges with descriptor offsets and two N128 tiles.
TMN_DEVI void dnorm(const Params& p,int row,uint8_t* sm,uint64_t* bars,int& phase){
 int tid=threadIdx.x,wg=tid/128,lane=tid%32,warp=(tid/32)%4;
 if(tid==0){mbar_arrive_expect_tx(bars+1,32768);for(int k=0;k<4;++k)tma_load_2d(sm+DN+k*8192,&p.dp,bars+1,k*64,row);}
 mbar_wait(bars+1,(row/(gridDim.x*64))&1);__syncthreads();
 auto load=[&](int k,int slot){mbar_arrive_expect_tx(bars+2+slot,65536);for(int g=0;g<8;++g)tma_load_2d(sm+WB+slot*65536+g*8192,&p.wp,bars+2+slot,g*64,k);};
 if(tid==0)load(0,0);float v0[64]={},v1[64]={};
 #pragma unroll 1
 for(int k=0;k<4;++k){int slot=k%2;
  if(tid==0&&k<3)load((k+1)*64,1-slot);
  mbar_wait(bars+2+slot,(phase>>slot)&1);phase^=1<<slot;__syncthreads();
  uint64_t ad=smem_desc(smem_u32(sm+DN+k*8192),16,1024,1);
  uint64_t b0=smem_desc(smem_u32(sm+WB+slot*65536+wg*32768),8192,1024,1);
  uint64_t b1=smem_desc(smem_u32(sm+WB+slot*65536+wg*32768+16384),8192,1024,1);
  fence_regs(v0);fence_regs(v1);wgmma_fence();
  static_for<4>([&](auto qq){constexpr int q=decltype(qq)::value;
   mma128_off<q*32,q*2048,0,1>(v0,ad,b0,k>0||q>0);
   mma128_off<q*32,q*2048,0,1>(v1,ad,b1,k>0||q>0);
  });wgmma_commit();wgmma_wait<0>();fence_regs(v0);fence_regs(v1);__syncthreads();
 }
 static_for<64>([&](auto jj){constexpr int j=decltype(jj)::value;
  int r=warp*16+lane/4+8*((j/2)&1),c=(j/8)*16+2*(lane%4)+8*((j/2)%4/2)+(j%2);
  reinterpret_cast<bf*>(sm+DN+wg*32768+(c/64)*8192)[swz128(r,(c%64)*2)/2]=__float2bfloat16_rn(v0[j]);
  reinterpret_cast<bf*>(sm+DN+wg*32768+16384+(c/64)*8192)[swz128(r,(c%64)*2)/2]=__float2bfloat16_rn(v1[j]);
 });
 __syncthreads();
}
