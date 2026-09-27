// SPDX-License-Identifier: Apache-2.0
// One warp owns eight rows and 64 channels, with uniform CTA barriers.
template<bool INVERSE> TMN_DEVI void transpose32(uint8_t* sm){
 int lane=threadIdx.x%32,warp=threadIdx.x/32,mat=lane/8,r8=lane%8;
 for(int first=0;first<H/64;first+=4){
  int kc=first+warp;uint32_t f[2][4],base=smem_u32(sm+kc*1024);
  if(kc<H/64){
   #pragma unroll
   for(int q=0;q<2;++q){
    uint32_t raw=(q*32+mat*8+r8)*16,row=swz128(r8,(q*32+mat*8)*2);
    ldsm_x4_t(f[q],base+(INVERSE?row:raw));
   }
  }
  named_bar_sync(1,128);
  if(kc<H/64){
   #pragma unroll
   for(int q=0;q<2;++q){
    uint32_t raw=(q*32+mat*8+r8)*16,row=swz128(r8,(q*32+mat*8)*2);
    stsm_x4(base+(INVERSE?raw:row),f[q][0],f[q][1],f[q][2],f[q][3]);
   }
  }
 }
 named_bar_sync(1,128);
}
TMN_DEVI void put_tile(const CUtensorMap* map,uint8_t* sm,int row,int c){
 asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"::"l"(map),"r"(smem_u32(sm)),"r"(row),"r"(c):"memory");
}
