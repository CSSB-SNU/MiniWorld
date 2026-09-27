#include "base.cuh"
extern "C" __global__ __launch_bounds__(256,2) void mw_d256_b1_ln(__grid_constant__ const Params p){extern __shared__ __align__(128) uint8_t sm[];for(int c=blockIdx.x*THREADS+threadIdx.x;c<H;c+=gridDim.x*THREADS){p.f[10][c]=0;p.f[11][c]=0;}cooperative_groups::this_grid().sync();output_ln_bwd(p,sm);}
