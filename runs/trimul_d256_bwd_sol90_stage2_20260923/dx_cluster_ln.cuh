// Two CTA channel halves exchange only 2 x 64 FP32 LN row sums.
TMN_DEVI uint32_t remote_addr(const void* p,int rank){uint32_t a;asm volatile("mapa.shared::cluster.u32 %0,%1,%2;":"=r"(a):"r"(smem_u32(p)),"r"(rank));return a;}
TMN_DEVI float remote_float(const void* p,int rank){float x;asm volatile("ld.shared::cluster.f32 %0,[%1];":"=f"(x):"r"(remote_addr(p,rank)):"memory");return x;}
TMN_DEVI int spos(int r,int c){return (c/64)*4096+swz128(r,(c%64)*2)/2;}
TMN_DEVI void cluster_ln(const Params& p,uint8_t* sm,uint64_t* bars,int row,int col,int round,float (&gg)[4],float (&bb)[4]){
 int tid=threadIdx.x,lane=tid%32,warp=tid/32,peer=1-blockIdx.x%2;
 float* sums=reinterpret_cast<float*>(sm+NSLOT*STAGE+128);
 float* gamma=reinterpret_cast<float*>(sm+NSLOT*STAGE+640);
 if(tid==0){mbar_arrive_expect_tx(bars+2*NSLOT,32768);for(int c=0;c<128;c+=64){tma_load_2d(sm+16384+c*128,&p.xmap,bars+2*NSLOT,col+c,row);tma_load_2d(sm+32768+c*128,&p.resmap,bars+2*NSLOT,col+c,row);}}
 mbar_wait(bars+2*NSLOT,round&1);__syncthreads();
 bf* dxn=reinterpret_cast<bf*>(sm);bf* x=reinterpret_cast<bf*>(sm+16384);bf* res=reinterpret_cast<bf*>(sm+32768);
 for(int r=warp;r<64;r+=4){float mu=p.f[12][2*(row+r)],rs=p.f[12][2*(row+r)+1],s0=0,s1=0;
  #pragma unroll
  for(int q=0;q<4;++q){int c=lane+q*32;float z=(rd(x,spos(r,c))-mu)*rs,dy=rd(dxn,spos(r,c)),v=dy*gamma[c];s0+=v;s1+=v*z;gg[q]+=dy*z;bb[q]+=dy;}
  s0=wsum(s0);s1=wsum(s1);if(lane==0){sums[r]=s0;sums[64+r]=s1;}
 }
 cooperative_groups::this_cluster().sync();
 for(int r=warp;r<64;r+=4){float mu=p.f[12][2*(row+r)],rs=p.f[12][2*(row+r)+1],other0=0,other1=0;
  if(lane==0){other0=remote_float(sums+r,peer);other1=remote_float(sums+64+r,peer);}
  float s0=(sums[r]+__shfl_sync(0xffffffff,other0,0))/D,s1=(sums[64+r]+__shfl_sync(0xffffffff,other1,0))/D;
  #pragma unroll
  for(int q=0;q<4;++q){int c=lane+q*32;float z=(rd(x,spos(r,c))-mu)*rs,v=rd(dxn,spos(r,c))*gamma[c];float dx=(v-s0-z*s1)*rs;dxn[spos(r,c)]=cv(dx+rd(res,spos(r,c)));}
 }
 __syncthreads();fence_proxy_async();__syncthreads();
 if(tid==0){for(int c=0;c<128;c+=64)store_tile(&p.dxn,sm+c*128,col+c,row);tma_store_commit();tma_store_wait_all();}
 cooperative_groups::this_cluster().sync();
}
