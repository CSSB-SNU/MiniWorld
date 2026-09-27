struct ReduceParams {const float *partial,*weights;float *tmp,*dg,*db;bf *dw[4];int rows,chunks;};
extern "C" __global__ __launch_bounds__(256,2)
void mw_prefix_input_affine_first(__grid_constant__ const ReduceParams p){
 int c=threadIdx.x;float g=0,b=0;
 for(int r=blockIdx.x*64;r<min(p.rows,int(blockIdx.x+1)*64);++r){g+=p.partial[r*512+c];b+=p.partial[r*512+256+c];}
 p.tmp[blockIdx.x*512+c]=g;p.tmp[blockIdx.x*512+256+c]=b;
}
extern "C" __global__ __launch_bounds__(128,4)
void mw_prefix_input_weight_finish(__grid_constant__ const ReduceParams p){
 int c=blockIdx.x*128+threadIdx.x;
 if(c<256){float g=0,b=0;for(int r=0;r<p.chunks;++r){g+=p.tmp[r*512+c];b+=p.tmp[r*512+256+c];}p.dg[c]=g;p.db[c]=b;}
 for(int i=c;i<8*256*256;i+=gridDim.x*128){int which=i/(2*256*256),j=i%(2*256*256);float v=0;
  #pragma unroll
  for(int s=0;s<8;++s)v+=p.weights[size_t(s)*11*256*256+(3+2*which)*256*256+j];
  p.dw[which][j]=__float2bfloat16_rn(v);
 }
}
