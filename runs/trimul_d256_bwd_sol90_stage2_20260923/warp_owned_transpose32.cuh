// SPDX-License-Identifier: Apache-2.0
// One warp owns each complete 64-channel x32-row tile, including both row halves.
template<bool INVERSE> TMN_DEVI void transpose32(uint8_t* sm){
 static_assert(ROWS==32,"This layout owns all32 rows inside one warp");
 int tid=threadIdx.x%128,lane=tid%32,warp=tid/32,mat=lane/8,r8=lane%8;
 for(int kc=warp;kc<H/64;kc+=4){
  uint32_t f[2][4][4],base=smem_u32(sm+kc*(ROWS*128));
  #pragma unroll
  for(int w=0;w<2;++w){
   #pragma unroll
   for(int q=0;q<4;++q){
    uint32_t raw=rawpos(16*q+r8+((mat&2)?8:0),16*w+((mat&1)?8:0));
    uint32_t row=swz128(16*w+r8+((mat&1)?8:0),(16*q+((mat&2)?8:0))*2);
    ldsm_x4_t(f[w][q],base+(INVERSE?row:raw));
   }
  }
  __syncwarp();
  #pragma unroll
  for(int w=0;w<2;++w){
   #pragma unroll
   for(int q=0;q<4;++q){
    uint32_t raw=rawpos(16*q+r8+((mat&2)?8:0),16*w+((mat&1)?8:0));
    uint32_t row=swz128(16*w+r8+((mat&1)?8:0),(16*q+((mat&2)?8:0))*2);
    stsm_x4(base+(INVERSE?raw:row),f[w][q][0],f[w][q][1],f[w][q][2],f[w][q][3]);
   }
  }
 }
 named_bar_sync(1,128);
}
