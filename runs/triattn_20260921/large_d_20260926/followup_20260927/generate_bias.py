from pathlib import Path
ROOT=Path(__file__).resolve().parent
s=(ROOT.parent/'native_head128/dkdv.cu').read_text()
def rep(a,b,n=None):
    global s
    assert s.count(a) and (n is None or s.count(a)==n),(a,s.count(a),n)
    s=s.replace(a,b)
rep('float *db;','Element *db;',1)
rep('reinterpret_cast<float4*>(p.db+base)[vec]=make_float4(values[0],values[1],values[2],values[3]);', '''uint32_t lo,hi;
          asm("cvt.rn.bf16x2.f32 %0,%1,%2;":"=r"(lo):"f"(values[1]),"f"(values[0]));
          asm("cvt.rn.bf16x2.f32 %0,%1,%2;":"=r"(hi):"f"(values[3]),"f"(values[2]));
          reinterpret_cast<uint2*>(p.db+base)[vec]=make_uint2(lo,hi);''',1)
rep('float const* part','Element const* part')
rep('float4 x=reinterpret_cast<float4 const*>(part)[base+j*256];\n      acc[j*4+0]+=x.x;acc[j*4+1]+=x.y;acc[j*4+2]+=x.z;acc[j*4+3]+=x.w;', '''uint2 x=reinterpret_cast<uint2 const*>(part)[base+j*256];
      acc[j*4+0]+=float(Element::bitcast(uint16_t(x.x)));
      acc[j*4+1]+=float(Element::bitcast(uint16_t(x.x>>16)));
      acc[j*4+2]+=float(Element::bitcast(uint16_t(x.y)));
      acc[j*4+3]+=float(Element::bitcast(uint16_t(x.y>>16)));''',1)
rep('float2 x=reinterpret_cast<float2 const*>(part)[base];a+=x.x;b+=x.y;', '''uint32_t x=reinterpret_cast<uint32_t const*>(part)[base];
    a+=float(Element::bitcast(uint16_t(x)));b+=float(Element::bitcast(uint16_t(x>>16)));''',1)
rep('auto part=torch::empty({4,L/R,L,L},q.options().dtype(torch::kFloat32));','auto part=torch::empty({4,L/R,L,L},q.options());',1)
rep('part.data_ptr<float>()','(Element*)part.data_ptr()')
rep('wide_grouped_dkdv','wide_grouped_dkdv_bias16')
rep('reduce_bias','reduce_bias16')
(ROOT/'bias.cu').write_text(s)
