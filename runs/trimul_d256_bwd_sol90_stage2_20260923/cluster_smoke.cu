// SPDX-License-Identifier: Apache-2.0
// Exercise the two-slot DSM publication protocol before using it for B7.
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdint.h>
namespace cg=cooperative_groups;
constexpr int WORDS=2048,SLOTS=2,CLUSTER=8,READERS=4;
struct Params {int* out;const int* seed;int rounds;};
__device__ __forceinline__ uint32_t addr(const void* p){return uint32_t(__cvta_generic_to_shared(p));}
__device__ __forceinline__ uint32_t remote(const void* p,int rank){
 uint32_t r;asm volatile("mapa.shared::cluster.u32 %0,%1,%2;":"=r"(r):"r"(addr(p)),"r"(rank));return r;
}
__device__ __forceinline__ void arrive(const void* p,int rank){
 asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _,[%0];"::"r"(remote(p,rank)):"memory");
}
__device__ __forceinline__ void wait(const void* p,int phase){
 uint32_t done;
 do{asm volatile("{.reg .pred P; mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P,[%1],%2; selp.u32 %0,1,0,P;}"
 :"=r"(done):"r"(addr(p)),"r"(phase):"memory");}while(!done);
}
__device__ __forceinline__ int read_remote(const void* p,int rank){
 int v;asm volatile("ld.shared::cluster.u32 %0,[%1];":"=r"(v):"r"(remote(p,rank)):"memory");return v;
}
extern "C" __global__ __launch_bounds__(256) void mw_d256_cluster_smoke(Params p){
 extern __shared__ __align__(128) unsigned char sm[];
 int* data=reinterpret_cast<int*>(sm);
 uint64_t* ready=reinterpret_cast<uint64_t*>(sm+SLOTS*WORDS*4);
 uint64_t* free=ready+SLOTS;
 auto cluster=cg::this_cluster();int rank=cluster.block_rank();
 if(threadIdx.x==0){
  for(int slot=0;slot<SLOTS;++slot){
   asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;"::"r"(addr(ready+slot)),"r"(CLUSTER_DIRECT_ARRIVE?8*256:8):"memory");
   asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;"::"r"(addr(free+slot)),"r"(CLUSTER_DIRECT_ARRIVE?4*256:4):"memory");
  }
 }
 __syncthreads();cluster.sync();
 int seed=*p.seed;
 for(int it=0;it<p.rounds;++it){
  int slot=it%SLOTS;
  if(it>=SLOTS)wait(free+slot,((it/SLOTS)-1)&1);
  __syncthreads();
  for(int i=threadIdx.x;i<WORDS;i+=blockDim.x)data[slot*WORDS+i]=seed+int(blockIdx.x)*1000000+it*10000+i;
  __syncthreads();
  if(CLUSTER_DIRECT_ARRIVE||threadIdx.x==0)for(int peer=0;peer<READERS;++peer)arrive(ready+slot,peer);
  if constexpr(CLUSTER_FULL_SYNC)cluster.sync();
  if(rank<READERS){
   wait(ready+slot,(it/SLOTS)&1);
   for(int peer=0;peer<CLUSTER;++peer){
    int dst=((((it*(gridDim.x/CLUSTER)+blockIdx.x/CLUSTER)*READERS+rank)*CLUSTER+peer)*WORDS);
    for(int i=threadIdx.x;i<WORDS;i+=blockDim.x)p.out[dst+i]=read_remote(data+slot*WORDS+i,peer);
   }
   __syncthreads();
   if(CLUSTER_DIRECT_ARRIVE||threadIdx.x==0)for(int peer=0;peer<CLUSTER;++peer)arrive(free+slot,peer);
  }
  if constexpr(CLUSTER_FULL_SYNC)cluster.sync();
 }
 // Keep each DSM allocation alive until every remote reader has finished.
 cluster.sync();
}
