// SPDX-License-Identifier: Apache-2.0
#include "tmn_kernels.cuh"
using namespace tmn;using namespace tmn::sm90;using bf=__nv_bfloat16;
constexpr int D=WIDTH,H=2*D;
struct Params{const uint32_t *pre,*dl,*dr,*mask;uint32_t *gp[4];int M;};
extern "C" __global__ __launch_bounds__(256,4)
void mw_wide_saved_gp_linear(__grid_constant__ const Params p){
 int half=p.M/2;
 for(int i=blockIdx.x*256+threadIdx.x;i<2*H*half;i+=gridDim.x*256){
  int channel=i/half,pos=i%half,side=channel/H,col=channel%H;
  size_t pi=(size_t(side*(H/32)+col/32)*64+col%32)*half+pos;
  uint32_t gr=p.pre[pi],pr=p.pre[pi+size_t(32)*half],dy=(side?p.dr:p.dl)[size_t(col)*half+pos],masked;
  asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(masked):"r"(dy),"r"(p.mask[pos]));
  float ga=math::sigmoid(bf16lo(gr)),gb=math::sigmoid(bf16hi(gr)),pa=bf16lo(pr),pb=bf16hi(pr);
  p.gp[2*side][size_t(col)*half+pos]=pack_bf16(bf16lo(masked)*ga,bf16hi(masked)*gb);
  p.gp[2*side+1][size_t(col)*half+pos]=pack_bf16(((bf16lo(masked)*pa)*ga)*(1.f-ga),((bf16hi(masked)*pb)*gb)*(1.f-gb));
 }
}
