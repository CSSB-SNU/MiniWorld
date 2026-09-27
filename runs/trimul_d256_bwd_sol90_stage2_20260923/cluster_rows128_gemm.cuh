TMN_DEVI void load_gemm(const Params& p,uint8_t* sm,uint64_t* bar,int row,int col,int step){
 int slot=step%SLOTS;uint8_t* dst=sm+slot*STAGE;
 mbar_arrive_expect_tx(bar+slot,STAGE);
 for(int g=0;g<2;++g){
  tma_load_2d(dst+g*8192,&p.dp,bar+slot,step*64,row+g*64);
  tma_load_2d(dst+16384+g*8192,&p.wp,bar+slot,col+g*64,step*64);
 }
}
TMN_DEVI void gemm(const Params& p,uint8_t* sm,uint64_t* bar,int row,int col){
 int tid=threadIdx.x%128,lane=tid%32,warp=tid/32,wg=threadIdx.x/128;float v[64]={};
 if(threadIdx.x==0){load_gemm(p,sm,bar,row,col,0);load_gemm(p,sm,bar,row,col,1);
  mbar_arrive_expect_tx(bar+3,32768);
  for(int wm=0;wm<2;++wm)for(int c=0;c<2;++c)tma_load_2d(sm+TRI+(wm*2+c)*8192,&p.tri,bar+3,row+wm*64,col+c*64);
 }
 for(int step=0;step<D/64;++step){int slot=step%SLOTS;uint8_t* src=sm+slot*STAGE;
  mbar_wait(bar+slot,(step/SLOTS)&1);__syncthreads();
  if(threadIdx.x==0 && step+2<D/64)load_gemm(p,sm,bar,row,col,step+2);
  fence_regs(v);wgmma_fence();
  static_for<4>([&](auto qq){constexpr int q=decltype(qq)::value;
   mma128<0,1>(v,smem_desc(smem_u32(src+wg*8192+q*32),16,1024,1),smem_desc(smem_u32(src+16384+q*2048),8192,1024,1),step>0||q>0);
  });wgmma_commit();wgmma_wait<0>();fence_regs(v);__syncthreads();
 }
 static_for<32>([&](auto qq){constexpr int j=2*decltype(qq)::value;
  int r=warp*16+lane/4+8*((j/2)&1),c=(j/8)*16+2*(lane%4)+8*((j/2)%4/2);
  *reinterpret_cast<uint32_t*>(sm+wg*16384+2*pos(r,c))=pack_bf16(v[j],v[j+1]);
 });
 __syncthreads();
 if constexpr(EMIT_DN){fence_proxy_async();__syncthreads();
  if(threadIdx.x==0){for(int wm=0;wm<2;++wm)for(int c=0;c<2;++c)put(&p.dn,sm+(wm*2+c)*8192,col+c*64,row+wm*64);tma_store_commit();tma_store_wait_all();}
  __syncthreads();
 }
 mbar_wait(bar+3,0);__syncthreads();transpose<false>(sm+TRI+wg*16384);__syncthreads();
}
