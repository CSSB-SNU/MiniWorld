// Lossless BF16 representation: sign/mantissa bytes + exponent nibbles.
// Code15 reads the original BF16 location; zero/subnormal exponent uses code0.
#include <cuda_bf16.h>
#include <stdint.h>
struct Params {const uint32_t* pre;uint16_t* mant;uint8_t* exps;unsigned long long* escapes;int pairs;};
__device__ __forceinline__ unsigned code(unsigned raw){unsigned e=(raw>>7)&255u;return e==0?0u:(e>=116u&&e<=129u?e-115u:15u);}
extern "C" __global__ void mw_pack_pre_bf12(__grid_constant__ const Params p){
 unsigned count=0;
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<p.pairs;i+=gridDim.x*blockDim.x){
  unsigned raw=p.pre[i],a=raw&65535u,b=raw>>16,ca=code(a),cb=code(b);
  p.mant[i]=(a&127u)|((a>>8)&128u)|((b&127u)<<8)|(b&32768u);
  p.exps[i]=ca|(cb<<4);count+=(ca==15u)+(cb==15u);
 }
 if(count)atomicAdd(p.escapes,(unsigned long long)count);
}
