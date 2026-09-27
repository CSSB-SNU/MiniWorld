#include "base.cuh"
extern "C" __global__ __launch_bounds__(THREADS,2) void mw_d256_b7_reduce(__grid_constant__ const Params p){
 extern __shared__ __align__(128) uint8_t sm[];auto bar=reinterpret_cast<uint64_t*>(sm+2*STAGE);if(threadIdx.x==0){mbar_init(bar,1);mbar_init(bar+1,1);fence_barrier_init();}__syncthreads();int phase=0;auto grid=cooperative_groups::this_grid();
 for(int c=blockIdx.x*THREADS+threadIdx.x;c<D;c+=gridDim.x*THREADS){p.f[8][c]=0;p.f[9][c]=0;}stamp(p,8);stamp(p,9);__threadfence();asm volatile("fence.proxy.async.global;":::"memory");grid.sync();stamp(p,10);
 stamp(p,11);__threadfence();asm volatile("fence.proxy.async.global;":::"memory");grid.sync();ln_bwd<true>(p);stamp(p,12);for(int i=0;i<4;++i)reduce_w(p,(H+D+i*H)*D,H*D,p.t[17+i]);stamp(p,13);
}
