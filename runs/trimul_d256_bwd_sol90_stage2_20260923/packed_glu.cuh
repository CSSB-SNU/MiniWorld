// SPDX-License-Identifier: Apache-2.0
// D256 adaptation of MiniWorld D128 pair_glu: packed ldmatrix/stmatrix epilogue.
TMN_DEVI void packed_glu(float (&a)[32],uint8_t* s,uint8_t* sg,uint32_t ma,uint32_t mb){
 int lane=threadIdx.x%32,w=(threadIdx.x/32)%4,mat=lane/8;
 #pragma unroll
 for(int q=0;q<2;++q){uint32_t dy[4],dg[4],dp[4];ldsm_x4_t(dy,smem_u32(s+32768)+swz128(q*16+lane%8+8*(mat>>1),(w*16+8*(mat&1))*2));
  #pragma unroll
  for(int j=0;j<4;++j){uint32_t masked,m=(j&1)?mb:ma;asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(masked):"r"(dy[j]),"r"(m));
   uint32_t gr=pack_bf16(a[q*8+j*2],a[q*8+j*2+1]),pr=pack_bf16(a[(q+2)*8+j*2],a[(q+2)*8+j*2+1]);float ga=math::sigmoid(bf16lo(gr)),gb=math::sigmoid(bf16hi(gr)),pa=bf16lo(pr),pb=bf16hi(pr);
   dg[j]=pack_bf16(((bf16lo(masked)*pa)*ga)*(1.f-ga),((bf16hi(masked)*pb)*gb)*(1.f-gb));dp[j]=pack_bf16(bf16lo(masked)*ga,bf16hi(masked)*gb);
  }
  uint32_t addr=swz128(q*16+lane%8+8*(mat>>1),(w*16+8*(mat&1))*2);stsm_x4_t(smem_u32(sg)+addr,dg[0],dg[1],dg[2],dg[3]);stsm_x4_t(smem_u32(sg+4096)+addr,dp[0],dp[1],dp[2],dp[3]);
 }
}
