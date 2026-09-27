// Row ownership replaces the serial inter-CTA LN sum chain.
constexpr int ROWS=64/CLSIZE,RDN=16384,RTRI=32768,AFF=RTRI,PARAM=BAR+256;
// ROW_TRANSPOSE_HELPERS
TMN_DEVI int rowpos(int r,int c){return (c/64)*(ROWS*64)+swz128(r,(c%64)*2)/2;}
TMN_DEVI void exchange_rows(uint8_t* sm,uint64_t* bar,int rank){
 fence_proxy_async();__syncthreads();
 if(threadIdx.x==0){
  for(int peer=0;peer<CLSIZE;++peer)for(int slab=0;slab<2;++slab){
   int src=slab*8192+peer*ROWS*128,dst=(rank*2+slab)*ROWS*128;
   asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0],[%1],%2,[%3];"::"r"(remote_addr(sm+RDN+dst,peer)),"r"(smem_u32(sm+src)),"n"(ROWS*128),"r"(remote_addr(bar+6,peer)):"memory");
   asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0],[%1],%2,[%3];"::"r"(remote_addr(sm+RTRI+dst,peer)),"r"(smem_u32(sm+TRI+src)),"n"(ROWS*128),"r"(remote_addr(bar+6,peer)):"memory");
  }
 }
}
TMN_DEVI void normalize_rows(const Params& p,uint8_t* sm,uint64_t* bar,int row,int rank){
 int tid=threadIdx.x,lane=tid%32,warp=tid/32;
 mbar_wait(bar+6,0);mbar_wait(bar+7,0);__syncthreads();
 auto dn=reinterpret_cast<bf*>(sm+RDN);auto tri=reinterpret_cast<bf*>(sm+RTRI);
 auto gamma=reinterpret_cast<float*>(sm+PARAM);
 auto mu=gamma+H;auto rs=mu+ROWS;
 float gg[H/32]={},bb[H/32]={};
 for(int r=warp;r<ROWS;r+=4){
  float s0=0,s1=0;
  #pragma unroll
  for(int q=0;q<H/32;++q){int c=lane+q*32;
   float z=(__bfloat162float(tri[rowpos(r,c)])-mu[r])*rs[r];
   float dy=__bfloat162float(dn[rowpos(r,c)]),v=dy*gamma[c];
   s0+=v;s1+=v*z;gg[q]+=dy*z;bb[q]+=dy;
  }
  s0=sumwarp(s0)/H;s1=sumwarp(s1)/H;
  #pragma unroll
  for(int q=0;q<H/32;++q){int c=lane+q*32;
   float z=(__bfloat162float(tri[rowpos(r,c)])-mu[r])*rs[r];
   float dy=__bfloat162float(dn[rowpos(r,c)]);
   dn[rowpos(r,c)]=__float2bfloat16_rn(fmaf(-z,s1,fmaf(dy,gamma[c],-s0))*rs[r]);
  }
 }
 __syncthreads();transpose32<true>(sm+RDN);fence_proxy_async();__syncthreads();
 if(tid==0){for(int c=0;c<H;c+=64)put_tile(&p.dt,sm+RDN+(c/64)*(ROWS*128),row+rank*ROWS,c);tma_store_commit();tma_store_wait_all();}
 __syncthreads();
 // Gather warp-local affine sums into a compact array in retired TRI storage.
 auto agg=reinterpret_cast<float*>(sm+RTRI);
 #pragma unroll
 for(int q=0;q<H/32;++q){int c=lane+q*32;agg[warp*H+c]=gg[q];agg[(4+warp)*H+c]=bb[q];}
 __syncthreads();
 // The source dNorm and all incoming messages are now retired locally.
 auto values=reinterpret_cast<float*>(sm+RDN);
 for(int c=tid;c<H;c+=128){float g=0,b=0;
  #pragma unroll
  for(int w=0;w<4;++w){g+=agg[w*H+c];b+=agg[(4+w)*H+c];}
  values[c]=g;values[H+c]=b;
 }
 __syncthreads();cooperative_groups::this_cluster().sync();
 if(rank<ACTIVE){int c=rank*128+tid;float g=0,b=0;
  #pragma unroll
  for(int peer=0;peer<CLSIZE;++peer){g+=remote_float(values+c,peer);b+=remote_float(values+H+c,peer);}
  p.partial[(size_t(row/64)*2)*H+c]=g;p.partial[(size_t(row/64)*2+1)*H+c]=b;
 }
}
extern "C" __global__ __launch_bounds__(128,2)
void mw_cluster_rows_dn_ln(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];auto bar=reinterpret_cast<uint64_t*>(sm+BAR);
 auto cluster=cooperative_groups::this_cluster();int rank=cluster.block_rank(),tid=threadIdx.x;
 int row=(blockIdx.x/CLSIZE)*64;
 for(int c=blockIdx.x*128+tid;c<H;c+=gridDim.x*128){p.dg[c]=0;p.db[c]=0;}
 if(tid==0){
  for(int i=0;i<8;++i)mbar_init(bar+i,1);fence_barrier_init();
  mbar_arrive_expect_tx(bar+6,ROWS*H*4);
  mbar_arrive_expect_tx(bar+7,H*4+ROWS*8);
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1],%2,[%3];"::"r"(smem_u32(sm+PARAM)),"l"(p.gamma),"n"(H*4),"r"(smem_u32(bar+7)):"memory");
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1],%2,[%3];"::"r"(smem_u32(sm+PARAM+H*4)),"l"(p.mu+row+rank*ROWS),"n"(ROWS*4),"r"(smem_u32(bar+7)):"memory");
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1],%2,[%3];"::"r"(smem_u32(sm+PARAM+H*4+ROWS*4)),"l"(p.rs+row+rank*ROWS),"n"(ROWS*4),"r"(smem_u32(bar+7)):"memory");
 }
 __syncthreads();cluster.sync();
 if(rank<ACTIVE)gemm(p,sm,bar,row,rank*128);
 // Every receiver's retired GEMM storage must be free before DSM writers start.
 cluster.sync();
 if(rank<ACTIVE)exchange_rows(sm,bar,rank);
 normalize_rows(p,sm,bar,row,rank);
 cluster.sync();
}
