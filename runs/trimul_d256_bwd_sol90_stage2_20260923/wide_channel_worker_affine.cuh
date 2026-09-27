TMN_DEVI void affine(const Params& p,uint8_t* sm,uint64_t* bar){
 int tid=threadIdx.x%128;float gg[H/128]={},bb[H/128]={};
 for(int row=blockIdx.x*ROWS,it=0;row<p.M;row+=gridDim.x*ROWS,++it){
  mbar_wait(bar+5,it&1);named_bar_sync(2,128);
  #pragma unroll 1
  for(int r=0;r<ROWS;++r){float mu=p.mu[row+r],rs=p.rs[row+r];
   #pragma unroll
   for(int q=0;q<H/128;++q){int c=tid+q*128;
    float z=(__bfloat162float(reinterpret_cast<bf*>(sm)[pos(r,c)])-mu)*rs;
    float dy=__bfloat162float(reinterpret_cast<bf*>(sm+SB)[pos(r,c)]);gg[q]+=dy*z;bb[q]+=dy;
   }
  }
  named_bar_sync(2,128);if(tid==0)mbar_arrive(bar+2);
 }
 mbar_wait(bar+4,0);named_bar_sync(2,128);
 #pragma unroll
 for(int q=0;q<H/128;++q){int c=tid+q*128;atomicAdd(p.dg+c,gg[q]);atomicAdd(p.db+c,bb[q]);}
}
