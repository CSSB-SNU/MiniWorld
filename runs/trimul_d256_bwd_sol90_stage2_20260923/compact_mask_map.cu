// SPDX-License-Identifier: Apache-2.0
#include <cuda_bf16.h>
#include <stdint.h>
extern "C" __global__ void mw_mask_local(const __nv_bfloat16* mask,int* rows,int* counts,int M){
 __shared__ int sums[8];
 int row=blockIdx.x*256+threadIdx.x,lane=threadIdx.x%32,warp=threadIdx.x/32;
 bool valid=row<M && (reinterpret_cast<const uint16_t*>(mask)[row]&0x7fff)!=0;
 unsigned ballot=__ballot_sync(0xffffffff,valid);
 if(lane==0)sums[warp]=__popc(ballot);
 __syncthreads();int base=0;for(int i=0;i<warp;++i)base+=sums[i];
 if(row<M)rows[row]=valid?base+__popc(ballot&((1u<<lane)-1)):-1;
 if(threadIdx.x==0){int total=0;for(int i=0;i<8;++i)total+=sums[i];counts[blockIdx.x]=total;}
}
extern "C" __global__ void mw_mask_counts(int* counts,int* metadata,int blocks,int capacity){
 __shared__ int sums[32];
 int tid=threadIdx.x,lane=tid%32,warp=tid/32,value=tid<blocks?counts[tid]:0,original=value;
 for(int delta=1;delta<32;delta*=2){int previous=__shfl_up_sync(0xffffffff,value,delta);if(lane>=delta)value+=previous;}
 if(lane==31)sums[warp]=value;
 __syncthreads();int base=0;for(int i=0;i<warp;++i)base+=sums[i];
 if(tid<blocks)counts[tid]=base+value-original;
 if(tid==blocks-1){metadata[0]=base+value;metadata[1]=base+value<=capacity;}
}
extern "C" __global__ void mw_mask_global(int* rows,const int* counts,const int* metadata,int M){
 int row=blockIdx.x*256+threadIdx.x;
 if(row<M){int local=rows[row];rows[row]=metadata[1]?(local<0?-1:local+counts[blockIdx.x]):row;}
}
extern "C" __global__ void mw_mask_pad(__nv_bfloat16* gp,const int* metadata,int M,int capacity,int channels){
 if(!metadata[1])return;
 int count=metadata[0],length=capacity-count;
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<channels*length;i+=gridDim.x*blockDim.x){
  int ch=i/length,col=i%length;
  reinterpret_cast<uint16_t*>(gp)[size_t(ch)*M+count+col]=0;
 }
}
