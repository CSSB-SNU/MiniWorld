from pathlib import Path

root=Path(__file__).resolve().parent.parent
# Issue shared bias loads while the two score MMAs are in flight. This does
# not change rounding or add global traffic; screen register pressure first.
s=(root/'dq/rs_coop_tma/fused.cu').read_text()
s=s.replace('flash::gemm<true,0>(smma,da,vb,dp);','flash::gemm<true,-1>(smma,da,vb,dp);')
old='''  uint32_t bp=cast_smem_ptr_to_uint(s.bias[slot].data());
  #pragma unroll
  for(int x=0;x<size(score);x+=2){
   int qr=get<0>(sc(x)),kr=get<1>(sc(x));int off=as_position_independent_swizzle_layout(Config::SL{})(make_coord(qr,kr))*2;
   uint32_t packed;asm volatile("ld.shared.b32 %0,[%1];":"=r"(packed):"r"(bp+off):"memory");'''
new='''  uint32_t bp=cast_smem_ptr_to_uint(s.bias[slot].data()),breg[16];
  #pragma unroll
  for(int x=0;x<size(score);x+=2){
   int qr=get<0>(sc(x)),kr=get<1>(sc(x));int off=as_position_independent_swizzle_layout(Config::SL{})(make_coord(qr,kr))*2;
   asm volatile("ld.shared.b32 %0,[%1];":"=r"(breg[x/2]):"r"(bp+off):"memory");
  }
  warpgroup_wait<0>();
  #pragma unroll
  for(int x=0;x<size(score);x+=2){
   uint32_t packed=breg[x/2];'''
assert old in s;s=s.replace(old,new)
d=root/'dq/rs_bias_prefetch';d.mkdir(exist_ok=True);(d/'fused.cu').write_text(s)

s=(root/'bias_fusion/rs_tma_store/grouped.cu').read_text()
s=s.replace('flash::gemm<true,0>(smma,va,dob,dp);','flash::gemm<true,-1>(smma,va,dob,dp);')
start=s.index('      // ldmatrix transposes')
loop=s.index('        #pragma unroll\n        for(int yi=0;',start)
pre=s[start:loop].replace('      #pragma unroll','      uint32_t bias_regs[8];\n      #pragma unroll',1)
pre=pre.replace('        uint32_t bv[4];','        uint32_t* bv=bias_regs+xb/2;')
pre+='''      }
      warpgroup_wait<0>();
      #pragma unroll
      for(int xb=0;xb<size(score);xb+=8) {
        uint32_t* bv=bias_regs+xb/2;
'''
s=s[:start]+pre+s[loop:]
d=root/'bias_fusion/rs_bias_prefetch';d.mkdir(exist_ok=True);(d/'grouped.cu').write_text(s)
