// SPDX-License-Identifier: Apache-2.0
// One projection/GLU producer, two dW consumers; two shared input/derivative slots.
// SOURCE_HELPERS
TMN_DEVI void pipe_produce(const Params& p,uint8_t* sm,uint64_t* bar){
 int tid=threadIdx.x,lane=tid%32,warp=tid/32;
 int rank=blockIdx.x%RANKS,split=blockIdx.x/RANKS,tiles=p.M/64;
 int begin=tiles*split/WEIGHT_SPLITS,end=tiles*(split+1)/WEIGHT_SPLITS;
 if(tid==0){mbar_arrive_expect_tx(bar+2,CH);
  for(int c=0;c<D/64;++c)tma_load_2d(sm+WEIGHT+c*8192,&p.w,bar+2,c*64,rank*64);
 }
 mbar_wait(bar+2,0);named_bar_sync(1,128);
 for(int tile=begin,it=0;tile<end;++tile,++it){
  int slot=it%2,row=tile*64;
  if(it>=2)mbar_wait(bar+5+slot,((it/2)-1)&1);
  if(tid==0)load_input(p,sm,bar,row,rank,slot);
  mbar_wait(bar+slot,(it/2)&1);named_bar_sync(1,128);
  uint8_t* xn=sm+slot*INPUT;uint8_t* dg=sm+DERIV+slot*8192;
  float pre[32]={};fence_regs(pre);wgmma_fence();
  static_for<D/16>([&](auto kk){constexpr int k=decltype(kk)::value;
   mma64<0,0>(pre,smem_desc(smem_u32(xn+(k/4)*8192+(k%4)*32),16,1024,1),smem_desc(smem_u32(sm+WEIGHT+(k/4)*8192+(k%4)*32),16,1024,1),k>0);
  });wgmma_commit();wgmma_wait<0>();fence_regs(pre);
  int ra=warp*16+lane/4;
  uint32_t ma=uint32_t(__bfloat16_as_ushort(p.mask[row+ra]))*0x10001u;
  uint32_t mb=uint32_t(__bfloat16_as_ushort(p.mask[row+ra+8]))*0x10001u;
  packed_glu(pre,xn,dg,ma,mb);
  named_bar_sync(1,128);fence_proxy_async();named_bar_sync(1,128);
  if(tid==0){
   int side=rank/(H/32),col=(rank%(H/32))*32;
   store_gp(p.gmap+2*side,dg+4096,row,col);store_gp(p.gmap+2*side+1,dg,row,col);
   tma_store_commit();mbar_arrive(bar+3+slot);tma_store_wait_all();
  }
  named_bar_sync(1,128);
 }
}
template<int WG> TMN_DEVI void pipe_consume(const Params& p,uint8_t* sm,uint64_t* bar){
 int tid=threadIdx.x%128,lane=tid%32,warp=tid/32;
 int rank=blockIdx.x%RANKS,split=blockIdx.x/RANKS,tiles=p.M/64;
 int begin=tiles*split/WEIGHT_SPLITS,end=tiles*(split+1)/WEIGHT_SPLITS;
 float dw[NC][32]={};
 for(int tile=begin,it=0;tile<end;++tile,++it){
  int slot=it%2;uint8_t* xn=sm+slot*INPUT;uint8_t* dg=sm+DERIV+slot*8192;
  mbar_wait(bar+3+slot,(it/2)&1);named_bar_sync(2+WG,128);
  static_for<NC>([&](auto cc){constexpr int c=decltype(cc)::value;fence_regs(dw[c]);});wgmma_fence();
  static_for<4>([&](auto kk){constexpr int k=decltype(kk)::value;
   static_for<NC>([&](auto cc){constexpr int c=decltype(cc)::value;
    mma64<0,1>(dw[c],smem_desc(smem_u32(dg+k*32),16,1024,1),smem_desc(smem_u32(xn+(WG*NC+c)*8192+k*2048),8192,1024,1),it>0||k>0);
   });
  });wgmma_commit();wgmma_wait<0>();
  static_for<NC>([&](auto cc){constexpr int c=decltype(cc)::value;fence_regs(dw[c]);});
  named_bar_sync(2+WG,128);if(tid==0)mbar_arrive(bar+5+slot);
 }
 static_for<NC>([&](auto nn){constexpr int n=decltype(nn)::value;
  static_for<32>([&](auto jj){constexpr int j=decltype(jj)::value;
   int r=warp*16+lane/4+8*((j/2)&1),c=(j/8)*16+2*(lane%4)+8*((j/2)%4/2)+(j%2);
   int which=(rank/(H/32))*2+(r<32?1:0),outrow=(rank%(H/32))*32+r%32;
   size_t ix=size_t(split)*11*D*D+(3+2*which)*D*D+outrow*D+(WG*NC+n)*64+c;
   p.part[ix]=dw[n][j];
  });
 });
}
extern "C" __global__ __launch_bounds__(384,1)
void mw_wide_pipe_source(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];auto bar=reinterpret_cast<uint64_t*>(sm+BAR);
 if(threadIdx.x==0){for(int b=0;b<7;++b)mbar_init(bar+b,b>=5?2:1);fence_barrier_init();}
 __syncthreads();
 if(threadIdx.x<128){setmaxnreg_dec<96>();pipe_produce(p,sm,bar);}
 else{setmaxnreg_inc<200>();if(threadIdx.x<256)pipe_consume<0>(p,sm,bar);else pipe_consume<1>(p,sm,bar);}
}
