from pathlib import Path
root=Path(__file__).resolve().parent.parent/'dq'
s=(root/'rs_coop_tma/fused.cu').read_text()
a=s.index('  #pragma unroll\n  for(int x=0;x<size(score);x+=2)')
b=s.index('  auto acc_a=',a)
s=s[:a]+'''  #pragma unroll
  for(int xb=0;xb<size(score);xb+=8){
   // Four 8x8 blocks cover a 16x16 sub-tile in C-fragment order.
   int qr=(lane/32)*16+(lane%16),kr=xb*2+(lane%32/16)*8;
   uint32_t addr=bp+as_position_independent_swizzle_layout(Config::SL{})(make_coord(qr,kr))*2,bv[4];
   asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
     :"=r"(bv[0]),"=r"(bv[1]),"=r"(bv[2]),"=r"(bv[3]):"r"(addr):"memory");
   #pragma unroll
   for(int xi=0;xi<8;++xi){
    int x=xb+xi;float bias=float(Element::bitcast(uint16_t(bv[xi/2]>>((xi%2)*16))));
    float logit=score(x)+bias*(1.f/SCALE);float pr=ex2(logit*(SCALE*LOG2E)-((x%4)/2?m1:m0));dp(x)=pr*(dp(x)-((x%4)/2?d1:d0));
   }
  }
'''+s[b:]
d=root/'rs_ldmatrix';d.mkdir(exist_ok=True);(d/'fused.cu').write_text(s)
