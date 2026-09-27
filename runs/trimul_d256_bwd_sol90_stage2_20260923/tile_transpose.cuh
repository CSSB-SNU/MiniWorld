// SPDX-License-Identifier: Apache-2.0
// Uniform CTA barriers even when the number of channel tiles is not a
// multiple of the participating warp groups (e.g. D384 / 256 threads).
TMN_DEVI uint32_t rawpos(int c,int r){uint32_t v=c*(ROWS*2)+r*2;if constexpr(ROWS==32)return sw64(v);else return v^(((v>>7)&1u)<<4);}
template<bool INVERSE> TMN_DEVI void transpose32(uint8_t* sm){
 int tid=threadIdx.x,lane=tid%32,warp=tid/32,mat=lane/8,r8=lane%8,w=warp%(ROWS/16);
 constexpr int TILES=(NT/32)/(ROWS/16);
 for(int first=0;first<H/64;first+=TILES){
  int kc=first+warp/(ROWS/16);uint32_t f[4][4],base=smem_u32(sm+kc*(ROWS*128));
  if(kc<H/64){
   #pragma unroll
   for(int q=0;q<4;++q){
    uint32_t raw=rawpos(16*q+r8+((mat&2)?8:0),16*w+((mat&1)?8:0));
    uint32_t row=swz128(16*w+r8+((mat&1)?8:0),(16*q+((mat&2)?8:0))*2);
    ldsm_x4_t(f[q],base+(INVERSE?row:raw));
   }
  }
  named_bar_sync(1,NT);
  if(kc<H/64){
   #pragma unroll
   for(int q=0;q<4;++q){
    uint32_t raw=rawpos(16*q+r8+((mat&2)?8:0),16*w+((mat&1)?8:0));
    uint32_t row=swz128(16*w+r8+((mat&1)?8:0),(16*q+((mat&2)?8:0))*2);
    stsm_x4(base+(INVERSE?raw:row),f[q][0],f[q][1],f[q][2],f[q][3]);
   }
  }
 }
 named_bar_sync(1,NT);
}
