"""Consume score/bias eight values at a time and immediately pack probabilities."""
import argparse,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--artifact',required=True)
a=p.parse_args();r=Path(__file__).resolve().parent;s=(r/a.base/'fused.cu').read_text()
decl='''        auto ar=make_tensor(score.data(),flash::convert_layout_acc_Aregs<typename C::PV>(score.layout()));
        auto pr=make_tensor_like<Element>(ar); flash::convert_type_out(ar,pr);'''
assert decl in s
s=s.replace('    auto score=partition_fragment_C(smma,Shape<_64,_64>{});',
'''    auto score=partition_fragment_C(smma,Shape<_64,_64>{});
    auto ar=make_tensor(score.data(),flash::convert_layout_acc_Aregs<typename C::PV>(score.layout()));
    auto pr=make_tensor_like<Element>(ar);''')
lo=s.index('        uint32_t bp=cast_smem_ptr_to_uint(')
hi=s.index(decl,lo)
safe=s[lo:hi]+'        flash::convert_type_out(ar,pr);\n'
fast='''        } else {
          uint32_t bp=cast_smem_ptr_to_uint(s.scratch.attn.bias[c][stage].data());
          auto packed=recast<uint32_t>(pr);
          #pragma unroll
          for(int xb=0;xb<size(score);xb+=8){
            int qr=(lane/32)*16+lane%16,kr=xb*2+(lane%32/16)*8;
            uint32_t addr=bp+as_position_independent_swizzle_layout(typename C::BL{})(make_coord(qr,kr))*2;
            uint32_t bv[4];
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
              :"=r"(bv[0]),"=r"(bv[1]),"=r"(bv[2]),"=r"(bv[3]):"r"(addr):"memory");
            #pragma unroll
            for(int pair=0;pair<4;++pair){
              int x=xb+pair*2;
              float b0=float(Element::bitcast(uint16_t(bv[pair])));
              float b1=float(Element::bitcast(uint16_t(bv[pair]>>16)));
              float v0=score(x)+b0*(1.f/SCALE),v1=score(x+1)+b1*(1.f/SCALE);
              v0=ex2(v0*(SCALE*LOG2E)-16.f);v1=ex2(v1*(SCALE*LOG2E)-16.f);
              asm("cvt.rn.bf16x2.f32 %0,%1,%2;":"=r"(packed(x/2)):"f"(v1),"f"(v0));
            }
          }
          if constexpr(Stages==1){
            cutlass::arch::NamedBarrier::sync(128,c+1);
            if(lane==0 && kt+1<nt)load_bias(kt+1);
          }
        }
'''
# alpha only belongs to the stable path, but the common following loop needs its name.
safe=safe.replace('float alpha[2], ls[2]={0.f,0.f};','float ls[2]={0.f,0.f};')
s=s[:lo]+'        float alpha[2];\n        if constexpr(Safe) {\n'+safe+fast+s[hi+len(decl):]
s=re.sub(r'qkv_attention_[a-z0-9_]+','qkv_attention_stream_pack_'+a.artifact,s)
folder=r/a.artifact;assert not (folder/'build-ready.json').exists()
folder.mkdir(exist_ok=True);(folder/'fused.cu').write_text(s)
