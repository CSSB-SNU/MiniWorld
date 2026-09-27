"""Separate TMA issue from the resident recompute/dW warpgroup."""
def specialize(body):
 marker='extern "C" __global__ __launch_bounds__(128,2) void mw_d256_b7_tma(__grid_constant__ const Params p){'
 prefix,compute=body.split(marker)
 begin=compute.index(' if(tid==0){');end=compute.index(' float dw0[64]')
 compute=compute[:begin]+' mbar_wait(bar+2,0);named_bar_sync(1,128);\n'+compute[end:]
 compute=compute.replace('tid=threadIdx.x,lane','tid=threadIdx.x%128,lane')
 compute=compute.replace('  if(tid==0&&tile+1<end)load_input(p,sm,bar,(tile+1)*64,rank,1-slot);','')
 compute=compute.replace('__syncthreads();','named_bar_sync(1,128);')
 compute=compute.replace('if(tid==0)tma_store_wait_all();named_bar_sync(1,128);','if(tid==0)tma_store_wait_all();named_bar_sync(1,128);if(tid==0)mbar_arrive(bar+3+slot);')
 return prefix+'TMN_DEVI void compute_source(const Params& p){'+compute+'''
extern "C" __global__ __launch_bounds__(256,2) void mw_d256_b7_tma(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];auto bar=reinterpret_cast<uint64_t*>(sm+114688);
 if(threadIdx.x==0){for(int i=0;i<5;++i)mbar_init(bar+i,1);fence_barrier_init();}
 __syncthreads();
 if(threadIdx.x<128){
  setmaxnreg_dec<32>();
  if(threadIdx.x==0){
   int rank=blockIdx.x%32,split=blockIdx.x/32,tiles=p.M/64,begin=tiles*split/WEIGHT_SPLITS,end=tiles*(split+1)/WEIGHT_SPLITS;
   mbar_arrive_expect_tx(bar+2,CH);
   for(int c=0;c<4;++c)tma_load_2d(sm+WEIGHT+c*8192,&p.w,bar+2,c*64,rank*64);
   for(int tile=begin,it=0;tile<end;++tile,++it){int slot=it%2;if(it>=2)mbar_wait(bar+3+slot,((it/2)-1)&1);load_input(p,sm,bar,tile*64,rank,slot);}
  }
 }else{setmaxnreg_inc<224>();compute_source(p);}
}
'''
