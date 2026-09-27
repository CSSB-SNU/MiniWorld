TMN_DEVI void load_gate_input(const Params& p,uint8_t* sm,uint64_t* bar,int row,int rank,int slot){
 uint8_t* in=sm+slot*40960;
 mbar_arrive_expect_tx(bar+slot,40960);
 for(int c=0;c<4;++c)tma_load_2d(in+c*8192,&p.xn,bar+slot,c*64,row);
 tma_load_2d(in+32768,&p.gatemap,bar+slot,row,(rank-32)*64);
}
TMN_DEVI void compute_gate_dw(const Params& p){
 extern __shared__ __align__(1024) uint8_t sm[];auto bar=reinterpret_cast<uint64_t*>(sm+114688);
 int tid=threadIdx.x%128,lane=tid%32,warp=tid/32,rank=blockIdx.x%36,split=blockIdx.x/36;
 int tiles=p.M/64,begin=tiles*split/WEIGHT_SPLITS,end=tiles*(split+1)/WEIGHT_SPLITS;
 float dw0[64]={},dw1[64]={};
 for(int tile=begin,it=0;tile<end;++tile,++it){
  int slot=it%2;uint8_t* xn=sm+slot*40960;
  mbar_wait(bar+slot,(it/2)&1);named_bar_sync(1,128);
  fence_regs(dw0);fence_regs(dw1);wgmma_fence();
  static_for<4>([&](auto kk){constexpr int k=decltype(kk)::value;
   uint64_t a=smem_desc(smem_u32(xn+32768),16,1024,1);
   mma128_off<k*32,k*2048,0,1>(dw0,a,smem_desc(smem_u32(xn),8192,1024,1),it>0||k>0);
   mma128_off<k*32,k*2048,0,1>(dw1,a,smem_desc(smem_u32(xn+16384),8192,1024,1),it>0||k>0);
  });wgmma_commit();wgmma_wait<0>();fence_regs(dw0);fence_regs(dw1);
  named_bar_sync(1,128);if(tid==0)mbar_arrive(bar+3+slot);
 }
 static_for<64>([&](auto jj){constexpr int j=decltype(jj)::value;
  int r=warp*16+lane/4+8*((j/2)&1),c=(j/8)*16+2*(lane%4)+8*((j/2)%4/2)+(j%2);
  size_t ix=size_t(split)*11*D*D+2*D*D+((rank-32)*64+r)*D+c;
  p.part[ix]=dw0[j];p.part[ix+128]=dw1[j];
 });
}
