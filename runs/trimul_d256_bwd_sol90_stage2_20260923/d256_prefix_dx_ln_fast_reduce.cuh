struct ReduceParams {const float *partial,*weights;float *tmp,*dg,*db;bf *dw[4];int rows,chunks;};
extern "C" __global__ __launch_bounds__(256,4)
void mw_prefix_input_affine_first(__grid_constant__ const ReduceParams p){
 __shared__ float parts[8][64];
 int lane=threadIdx.x%32,warp=threadIdx.x/32,c=blockIdx.x*32+lane;float g=0,b=0;
 #pragma unroll
 for(int q=0;q<8;++q){int r=blockIdx.y*64+warp+q*8;if(r<p.rows){g+=p.partial[r*512+c];b+=p.partial[r*512+256+c];}}
 parts[warp][lane]=g;parts[warp][32+lane]=b;__syncthreads();
 if(warp==0){g=0;b=0;
  #pragma unroll
  for(int w=0;w<8;++w){g+=parts[w][lane];b+=parts[w][32+lane];}
  p.tmp[blockIdx.y*512+c]=g;p.tmp[blockIdx.y*512+256+c]=b;
 }
}
extern "C" __global__ __launch_bounds__(128,4)
void mw_prefix_input_weight_finish(__grid_constant__ const ReduceParams p){
 int c=blockIdx.x*128+threadIdx.x;
 if(c<256){float g=0,b=0;for(int r=0;r<p.chunks;++r){g+=p.tmp[r*512+c];b+=p.tmp[r*512+256+c];}p.dg[c]=g;p.db[c]=b;}
 for(int i=c*2;i<8*256*256;i+=gridDim.x*256){int which=i/(2*256*256),j=i%(2*256*256);float a=0,b=0;
  #pragma unroll
  for(int s=0;s<8;++s){float2 v=*reinterpret_cast<const float2*>(p.weights+size_t(s)*11*256*256+(3+2*which)*256*256+j);a+=v.x;b+=v.y;}
  *reinterpret_cast<uint32_t*>(p.dw[which]+j)=pack_bf16(a,b);
 }
}
