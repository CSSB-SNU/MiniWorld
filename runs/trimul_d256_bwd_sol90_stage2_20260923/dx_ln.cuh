TMN_DEVI int pos(int r,int c){return (c/64)*4096+swz128(r,(c%64)*2)/2;}
TMN_DEVI void put_tile(const CUtensorMap* map,uint8_t* sm,int c,int row){asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"::"l"(map),"r"(smem_u32(sm)),"r"(c),"r"(row):"memory");}
extern "C" __global__ __launch_bounds__(256,2) void mw_d256_dx_ln(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];auto bars=reinterpret_cast<uint64_t*>(sm+98304);
 int tid=threadIdx.x,lane=tid%32,warp=tid/32,phase=0,lnphase=0;
 if(tid==0){for(int i=0;i<3;++i)mbar_init(bars+i,1);fence_barrier_init();}
 for(int c=blockIdx.x*256+tid;c<D;c+=gridDim.x*256){p.f[8][c]=0;p.f[9][c]=0;}
 __syncthreads();cooperative_groups::this_grid().sync();float gg[8]={},bb[8]={};
 for(int row=blockIdx.x*64;row<p.M;row+=gridDim.x*64){
  float v[64];int col=(tid/128)*128;
  gemm<0,1>(v,&p.map[5],&p.map[2],row,col,0,D,sm,bars,phase);
  for(int i=0;i<4;++i)gemm<1,1>(v,&p.map[6+i],&p.map[10+i],row,col,0,H,sm,bars,phase,true);
  for(int j=0;j<64;++j)reinterpret_cast<bf*>(sm)[pos(rr(j),col+cc(j))]=cv(v[j]);
  __syncthreads();
  if(tid==0){mbar_arrive_expect_tx(bars+2,65536);for(int c=0;c<D;c+=64){tma_load_2d(sm+32768+c*128,&p.xmap,bars+2,c,row);tma_load_2d(sm+65536+c*128,&p.resmap,bars+2,c,row);}}
  mbar_wait(bars+2,lnphase);lnphase^=1;__syncthreads();
  bf* dxn=reinterpret_cast<bf*>(sm);bf* x=reinterpret_cast<bf*>(sm+32768);bf* res=reinterpret_cast<bf*>(sm+65536);
  for(int r=warp;r<64;r+=8){
   float z=0;for(int c=lane;c<D;c+=32)z+=rd(x,pos(r,c));float mu=wsum(z)/D;z=0;
   for(int c=lane;c<D;c+=32){float val=rd(x,pos(r,c))-mu;z+=val*val;}float rs=rsqrtf(wsum(z)/D+1e-5f),s0=0,s1=0;
   for(int c=lane;c<D;c+=32){float zz=(rd(x,pos(r,c))-mu)*rs,dy=rd(dxn,pos(r,c)),vv=dy*p.f[0][c];s0+=vv;s1+=vv*zz;gg[c/32]+=dy*zz;bb[c/32]+=dy;}
   s0=wsum(s0)/D;s1=wsum(s1)/D;
   for(int c=lane;c<D;c+=32){float zz=(rd(x,pos(r,c))-mu)*rs,vv=rd(dxn,pos(r,c))*p.f[0][c];float dx=(vv-s0-zz*s1)*rs;dxn[pos(r,c)]=cv(dx+rd(res,pos(r,c)));}
  }
  __syncthreads();fence_proxy_async();__syncthreads();
  if(tid==0){for(int c=0;c<D;c+=64)put_tile(&p.dxmap,sm+c*128,c,row);tma_store_commit();tma_store_wait_all();}__syncthreads();
 }
 aggregate_ln<D,256>(gg,bb,reinterpret_cast<float*>(sm),p.f[8],p.f[9]);
 for(int which=0;which<4;++which){int offset=(H+D+which*H)*D;
  for(int i=blockIdx.x*256+tid;i<H*D;i+=gridDim.x*256){float v=0;for(int s=0;s<SPLITS;++s)v+=p.f[7][size_t(s)*11*D*D+offset+i];p.t[17+which][i]=cv(v);}
 }
}
