"""Keep affine-gradient accumulators out of registers during dNorm WGMMA.

The triangle tile is idle until GEMM ends. Reuse it between rows to retain
the exact FP32 affine-gradient sums without adding a global intermediate.
"""
def transform(body):
    body=body.replace(
        ' float gg[16]={},bb[16]={};int phase=0,wp_phase=0;',
        ''' volatile float* sums=reinterpret_cast<volatile float*>(sm);
 for(int i=tid;i<2*(NT/32)*H;i+=NT)sums[i]=0;
 __syncthreads();int phase=0,wp_phase=0;''')
    old='''  if(tid==0){mbar_arrive_expect_tx(bar,SB);for(int c=0;c<H;c+=64)tma_load_2d(sm+c*128,&p.tri,bar,row,c);}
  mbar_wait(bar,phase);phase^=1;__syncthreads();
  dnorm(p,row,sm,bar,wp_phase);transpose(sm);'''
    new='''  dnorm(p,row,sm,bar,wp_phase);
  float gg[16],bb[16];
  #pragma unroll
  for(int q=0;q<16;++q){int c=lane+q*32;gg[q]=sums[warp*H+c];bb[q]=sums[(NT/32+warp)*H+c];}
  __syncthreads();
  if(tid==0){mbar_arrive_expect_tx(bar,SB);for(int c=0;c<H;c+=64)tma_load_2d(sm+c*128,&p.tri,bar,row,c);}
  mbar_wait(bar,phase);phase^=1;__syncthreads();transpose(sm);'''
    assert body.count(old)==1
    body=body.replace(old,new)
    old='''  __syncthreads();
 }
 for(int q=0;q<16;++q){atomicAdd(p.dg+lane+q*32,gg[q]);atomicAdd(p.db+lane+q*32,bb[q]);}
}'''
    new='''  __syncthreads();
  #pragma unroll
  for(int q=0;q<16;++q){int c=lane+q*32;sums[warp*H+c]=gg[q];sums[(NT/32+warp)*H+c]=bb[q];}
  __syncthreads();
 }
 for(int q=0;q<16;++q){int c=lane+q*32;atomicAdd(p.dg+c,sums[warp*H+c]);atomicAdd(p.db+c,sums[(NT/32+warp)*H+c]);}
}'''
    assert body.count(old)==1
    return body.replace(old,new)
