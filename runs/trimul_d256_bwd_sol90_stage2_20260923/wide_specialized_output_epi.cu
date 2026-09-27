#include "tmn_kernels.cuh"
using namespace tmn;using namespace tmn::sm90;using bf=__nv_bfloat16;
struct Epi {const bf *proj,*gate,*x,*ds;bf *y;int elements,period;};
extern "C" __global__ __launch_bounds__(256,4)
void mw_wide_specialized_output_epi(__grid_constant__ const Epi p){
 constexpr int PERIOD=WIDTH*LENGTH/2;
 for(int i=(blockIdx.x*256+threadIdx.x)*VEC;i<p.elements/2;i+=gridDim.x*256*VEC){
  uint4 prs,gas,xxs,dss,ys;
  if constexpr(VEC==4){
   prs=*reinterpret_cast<const uint4*>(p.proj+2*i);gas=*reinterpret_cast<const uint4*>(p.gate+2*i);
   xxs=*reinterpret_cast<const uint4*>(p.x+2*i);dss=*reinterpret_cast<const uint4*>(p.ds+2*(i%PERIOD));
  }
  #pragma unroll
  for(int q=0;q<VEC;++q){int j=i+q;
   uint32_t pr,ga,xx,ds;
   if constexpr(VEC==4){pr=reinterpret_cast<uint32_t*>(&prs)[q];ga=reinterpret_cast<uint32_t*>(&gas)[q];xx=reinterpret_cast<uint32_t*>(&xxs)[q];ds=reinterpret_cast<uint32_t*>(&dss)[q];}
   else {pr=reinterpret_cast<const uint32_t*>(p.proj)[j];ga=reinterpret_cast<const uint32_t*>(p.gate)[j];xx=reinterpret_cast<const uint32_t*>(p.x)[j];ds=reinterpret_cast<const uint32_t*>(p.ds)[j%PERIOD];}
   float a=bf16lo(pr)*math::sigmoid(bf16lo(ga)),b=bf16hi(pr)*math::sigmoid(bf16hi(ga));
   uint32_t y=pack_bf16(fmaf(a,bf16lo(ds),bf16lo(xx)),fmaf(b,bf16hi(ds),bf16hi(xx)));
   if constexpr(VEC==4)reinterpret_cast<uint32_t*>(&ys)[q]=y;
   else reinterpret_cast<uint32_t*>(p.y)[j]=y;
  }
  if constexpr(VEC==4)*reinterpret_cast<uint4*>(p.y+2*i)=ys;
 }
}
