extern "C" __global__ void half_exp(unsigned const* p,unsigned* q){
 unsigned x=p[blockIdx.x*blockDim.x+threadIdx.x],y;
 asm("ex2.approx.f16x2 %0,%1;":"=r"(y):"r"(x));
 q[blockIdx.x*blockDim.x+threadIdx.x]=y;
}
