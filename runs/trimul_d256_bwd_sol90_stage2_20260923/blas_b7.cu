// SPDX-License-Identifier: Apache-2.0
#include "tmn_kernels.cuh"
using namespace tmn;using bf=__nv_bfloat16;
struct Params{const bf *pre,*dl,*dr,*mask;bf *gp;int M;};
extern "C" __global__ void mw_d256_blas_gp(__grid_constant__ const Params p){
 for(int i=blockIdx.x*256+threadIdx.x;i<1024*p.M/2;i+=gridDim.x*256){
  int side=i/(512*p.M/2),u=i%(512*p.M/2),row=u%(p.M/2);
  const uint32_t* pr=reinterpret_cast<const uint32_t*>(p.pre);
  uint32_t v=pr[side*512*p.M+u],g=pr[side*512*p.M+256*p.M+u];
  uint32_t dy=reinterpret_cast<const uint32_t*>(side?p.dr:p.dl)[u],mask=reinterpret_cast<const uint32_t*>(p.mask)[row];
  float a=math::round_bf16(bf16lo(dy)*bf16lo(mask)),b=math::round_bf16(bf16hi(dy)*bf16hi(mask));
  float x=math::sigmoid(bf16lo(g)),y=math::sigmoid(bf16hi(g));
  uint32_t* out=reinterpret_cast<uint32_t*>(p.gp);
  out[side*512*p.M+u]=pack_bf16(a*x,b*y);
  out[side*512*p.M+256*p.M+u]=pack_bf16(((a*bf16lo(v))*x)*(1-x),((b*bf16hi(v))*y)*(1-y));
 }
}
