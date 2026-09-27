"""Four heads per CTA; shared QG input and one packed head-last bias TMA ring."""
from pathlib import Path
import argparse

R=Path(__file__).resolve().parent
ap=argparse.ArgumentParser()
ap.add_argument('--artifact',required=True)
ap.add_argument('--stages',type=int,choices=(1,2),default=2)
ap.add_argument('--planar',action='store_true')
ap.add_argument('--repack',action='store_true',help='Transpose packed bias in shared memory, then use ldmatrix')
ap.add_argument('--independent',action='store_true',help='Planar control with independent per-head KV/bias rings')
ap.add_argument('--scalar',action='store_true',help='Read only the required BF16 head from packed shared bias')
a=ap.parse_args()
assert not(a.planar and a.repack)
assert not a.independent or a.planar
assert not a.scalar or not(a.planar or a.repack)
s=(R/'h4kv_local1/fused.cu').read_text()
s=s.replace('// Inference forward: no global Q/K/V/gate/LSE buffers.',
            '// Inference: one CTA owns four heads. Global KV; Q/gate and softmax stay local.')
if not a.planar:
    s=s.replace('using BL=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{},Shape<_64,_64>{}));',
                'using BL=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{},Shape<_64,_256>{}));')
    s=s.replace('using BS=Shape<int32_t,int32_t,_4>; using BStride=Stride<int64_t,_1,int64_t>;',
                'using BS=Shape<int32_t,int32_t>; using BStride=Stride<int64_t,_1>;')
    s=s.replace('BS{},BStride{}),BL{},Shape<_64,_64>', 'BS{},BStride{}),BL{},Shape<_64,_256>')
    s=s.replace('array_aligned<Element,4096,1024> bias[Consumers][Stages];',
                'array_aligned<Element,16384,1024> bias[Stages];')
else:
    s=s.replace('array_aligned<Element,4096,1024> bias[Consumers][Stages];',
                'array_aligned<Element,4096,1024> bias[Stages][Consumers];')
s=s.replace('array_aligned<Element,4096,1024> z[Consumers]; array_aligned<Element,8192,1024> w;',
            'array_aligned<Element,4096,1024> z[1]; array_aligned<Element,8192,1024> w[Consumers];')
s=s.replace('wfull,bfull[Consumers][Stages];', 'wfull,bfull[Stages];')
s=s.replace('h=blockIdx.x%4,row=blockIdx.y', 'h=c,row=blockIdx.y')
s=s.replace('for(int cc=0;cc<Consumers;++cc)for(int st=0;st<Stages;++st)s.bfull[cc][st].init(1);',
            'for(int st=0;st<Stages;++st)s.bfull[st].init(1);')
start=s.index('  auto load_weights=[&](bool kv) {')
end=s.index('  int bias_epoch=0;',start)
s=s[:start]+'''  auto load_weights=[&](bool) {
    if(tid==0) {
      s.wfull.arrive_and_expect_tx(Consumers*8192*sizeof(Element));
      for(int hh=0;hh<Consumers;++hh) {
        auto ws=make_tensor(make_smem_ptr(s.scratch.proj.w[hh].data()),typename C::WPL{});
        for(int which=0;which<2;++which) {
          auto const& wt=which==0?p.wq:p.wg;
          auto wg=wt.get_tma_tensor(make_shape(_128{},_128{}));
          for(int chunk=0;chunk<2;++chunk) {
            auto src=local_tile(wg,Shape<_32,_64>{},make_coord(hh,chunk));
            auto dst=local_tile(ws,Shape<_32,_64>{},make_coord(which,chunk));
            tma_load(wt,src,dst,s.wfull);
          }
        }
      }
    }
  };
'''+s[end:]
s=s.replace('int batch=(int(blockIdx.x)/4)*Consumers;\n    int qt=batch+c;',
            'int qt=int(blockIdx.x);')
s=s.replace('s.scratch.proj.z[c]', 's.scratch.proj.z[0]')
s=s.replace('s.scratch.proj.w.data()', 's.scratch.proj.w[c].data()')
# The same Z tile is consumed by all heads before the next input chunk overwrites it.
start=s.index('    // Q/gate are produced')
end=s.index('    if(qt<nt) {\n      auto bg=',start)
proj=s[start:end].replace('if(lane==0) {', 'if(tid==0) {')
proj=proj.replace('s.zfull[c]', 's.zfull[0]')
proj=proj.replace('cutlass::arch::NamedBarrier::sync(128,c+1);', '__syncthreads();')
s=s[:start]+proj+s[end:]
start=s.index('      auto bg=p.bias.get_tma_tensor(')
end=s.index('    auto out=partition_fragment_C',start)
load='''      typename C::Score smma;typename C::PV pmma;
      auto st=smma.get_slice(lane);auto pt=pmma.get_slice(lane);
      auto load_bias=[&](int kt) {
        int stage=kt%Stages;
        s.bfull[stage].arrive_and_expect_tx(65536);
        for(int hh=0;hh<4;++hh)for(int which=0;which<2;++which) {
          auto const& t=which==0?p.key:p.value;
          auto g=t.get_tma_tensor(make_shape(L,_32{},_4{},L));
          auto src=local_tile(g(_,_,hh,row),Shape<_64,_32>{},make_coord(kt,0));
          auto dst=make_tensor(make_smem_ptr(s.kv[hh][which][stage].data()),typename C::QL{});
          tma_load(t,src,dst,s.bfull[stage]);
        }
'''
if a.planar:
    load+='''        auto bg=p.bias.get_tma_tensor(make_shape(L,L,_4{}));
        for(int hh=0;hh<4;++hh){
          auto bb=local_tile(bg(_,_,hh),Shape<_64,_64>{},make_coord(qt,kt));
          auto dst=make_tensor(make_smem_ptr(s.scratch.attn.bias[stage][hh].data()),typename C::BL{});
          tma_load(p.bias,bb,dst,s.bfull[stage]);
        }
'''
else:
    load+='''        auto bg=p.bias.get_tma_tensor(make_shape(L,L*4));
        auto bb=local_tile(bg,Shape<_64,_256>{},make_coord(qt,kt));
        auto dst=make_tensor(make_smem_ptr(s.scratch.attn.bias[stage].data()),typename C::BL{});
        tma_load(p.bias,bb,dst,s.bfull[stage]);
'''
load+='      };\n'
s=s[:start]+load+s[end:]
s=s.replace('if(lane==0)for(int b=0;b<Stages && b<nt;++b)load_bias(b);',
            'if(tid==0)for(int b=0;b<Stages && b<nt;++b)load_bias(b);')
s=s.replace('s.bfull[c][stage]', 's.bfull[stage]')
if a.planar:
    s=s.replace('s.scratch.attn.bias[c][stage]', 's.scratch.attn.bias[stage][c]')
else:
    start=s.index('        uint32_t bp=cast_smem_ptr_to_uint(')
    end=s.index('\n        float alpha[2]',start)
    s=s[:start]+'''        float mx[2]={-INFINITY,-INFINITY};
        auto coords=st.partition_C(make_identity_tensor(Shape<_64,_64>{}));
        #pragma unroll
        for(int x=0;x<size(score);x+=2) {
          int qr=get<0>(coords(x)),kr=get<1>(coords(x));
          int offset=as_position_independent_swizzle_layout(typename C::BL{})(make_coord(qr,kr*4));
          uint4 bv=*reinterpret_cast<uint4 const*>(s.scratch.attn.bias[stage].data()+offset);
          unsigned b0=c<2?bv.x:bv.y,b1=c<2?bv.z:bv.w;
          float v0=float(Element::bitcast(uint16_t(b0>>((c%2)*16))));
          float v1=float(Element::bitcast(uint16_t(b1>>((c%2)*16))));
          score(x)+=v0*(1.f/SCALE);score(x+1)+=v1*(1.f/SCALE);
          if constexpr(Safe) {
            mx[(x%4)/2]=fmaxf(mx[(x%4)/2],score(x));
            mx[((x+1)%4)/2]=fmaxf(mx[((x+1)%4)/2],score(x+1));
          }
        }
''' + s[end:]
s=s.replace('''      cutlass::arch::NamedBarrier::sync(128,c+1);
      if(lane==0 && kt+Stages<nt)load_bias(kt+Stages);''',
'''      __syncthreads();
      if(tid==0 && kt+Stages<nt)load_bias(kt+Stages);''')
old='''    cutlass::arch::NamedBarrier::sync(128,c+1);
    bool retry=s.unsafe[c][0] || s.unsafe[c][1] || s.unsafe[c][2] || s.unsafe[c][3];'''
assert old in s
s=s.replace(old,'''    __syncthreads();
    bool retry=false;
    #pragma unroll
    for(int hh=0;hh<4;++hh)for(int ww=0;ww<4;++ww)retry=retry || s.unsafe[hh][ww];''')
s=s.replace('ep_h%=4;', 'ep_h=ep_tid/128;')
s=s.replace('<<<dim3(4*((L/64+Consumers-1)/Consumers),L),', '<<<dim3(L/64,L),')
if not a.planar:
    s=s.replace('make_shape(L,L,_4{}),typename C::BStride{L,_1{},int64_t(L)*L}',
                'make_shape(L,L*4),typename C::BStride{int64_t(L)*4,_1{}}')
    s=s.replace('bg,typename C::BL{},Shape<_64,_64>', 'bg,typename C::BL{},Shape<_64,_256>')
    s=s.replace('torch::IntArrayRef({1,4,L,L})', 'torch::IntArrayRef({1,L,L,4})')
for cap,st in ((128,1),(128,2),(384,2),(768,2),(1024,2)):
    s=s.replace(f'launch<{cap},1,{st},1>',f'launch<{cap},4,{1 if cap==128 and st==1 else a.stages},4>')
s=s.replace('Config<1024,1,2,1>::Shared',f'Config<1024,4,{a.stages},4>::Shared')
s=s.replace('qg_attention_head4kv','qg_attention_cta_heads4')
if a.independent:
    original=(R/'h4kv_local1/fused.cu').read_text()
    lo=original.index('      auto bg=p.bias.get_tma_tensor(')
    hi=original.index('    auto out=partition_fragment_C',lo)
    start=s.index('      typename C::Score smma;typename C::PV pmma;')
    end=s.index('    auto out=partition_fragment_C',start)
    s=s[:start]+original[lo:hi]+s[end:]
    s=s.replace('bias[Stages][Consumers]','bias[Consumers][Stages]')
    s=s.replace('wfull,bfull[Stages]','wfull,bfull[Consumers][Stages]')
    s=s.replace('for(int st=0;st<Stages;++st)s.bfull[st].init(1);','for(int cc=0;cc<Consumers;++cc)for(int st=0;st<Stages;++st)s.bfull[cc][st].init(1);')
    s=s.replace('if(tid==0)for(int b=0;b<Stages && b<nt;++b)load_bias(b);','if(lane==0)for(int b=0;b<Stages && b<nt;++b)load_bias(b);')
    s=s.replace('s.bfull[stage].wait(', 's.bfull[c][stage].wait(')
    s=s.replace('s.scratch.attn.bias[stage][c]', 's.scratch.attn.bias[c][stage]')
    s=s.replace('''      __syncthreads();
      if(tid==0 && kt+Stages<nt)load_bias(kt+Stages);''','''      cutlass::arch::NamedBarrier::sync(128,c+1);
      if(lane==0 && kt+Stages<nt)load_bias(kt+Stages);''')
if a.scalar:
    lo=s.index('        float mx[2]={-INFINITY,-INFINITY};')
    hi=s.index('\n        float alpha[2]',lo)
    s=s[:lo]+'''        float mx[2]={-INFINITY,-INFINITY};
        auto coords=st.partition_C(make_identity_tensor(Shape<_64,_64>{}));
        #pragma unroll
        for(int x=0;x<size(score);++x) {
          int qr=get<0>(coords(x)),kr=get<1>(coords(x));
          int offset=as_position_independent_swizzle_layout(typename C::BL{})(make_coord(qr,kr*4+c));
          float b=float(s.scratch.attn.bias[stage][offset]);
          score(x)+=b*(1.f/SCALE);
          if constexpr(Safe)mx[(x%4)/2]=fmaxf(mx[(x%4)/2],score(x));
        }
''' +s[hi:]
# Capacity does not affect streaming storage or arithmetic; one instantiation is sufficient.
if a.independent or a.scalar:
    import re
    s=re.sub(r'launch<(128|384|768),4,2,4>', 'launch<1024,4,2,4>',s)
if a.repack:
    s=s.replace('  using ZL=', '  using BPL=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{},Shape<_64,_64>{}));\n  using ZL=',1)
    mark='      s.bfull[stage].wait((bias_epoch+kt/Stages)%2);'
    assert mark in s
    s=s.replace(mark,mark+'''
      {
        // Every thread owns eight keys of one query and all four bias heads.
        int rq=tid/8,rk=(tid%8)*8;
        uint4 packed[4];
        #pragma unroll
        for(int t=0;t<4;++t) {
          int off=as_position_independent_swizzle_layout(typename C::BL{})(make_coord(rq,rk*4+t*8));
          packed[t]=*reinterpret_cast<uint4 const*>(s.scratch.attn.bias[stage].data()+off);
        }
        // In-place conversion: all packed loads retire before any planar store.
        __syncthreads();
        #pragma unroll
        for(int hh=0;hh<4;++hh) {
          uint4 plane;
          #pragma unroll
          for(int t=0;t<4;++t) {
            unsigned lo=hh<2?packed[t].x:packed[t].y;
            unsigned hi=hh<2?packed[t].z:packed[t].w;
            reinterpret_cast<unsigned*>(&plane)[t]=((lo>>((hh%2)*16))&65535u)|((hi>>((hh%2)*16))<<16);
          }
          int off=hh*4096+as_position_independent_swizzle_layout(typename C::BPL{})(make_coord(rq,rk));
          *reinterpret_cast<uint4*>(s.scratch.attn.bias[stage].data()+off)=plane;
        }
        __syncthreads();
      }
''')
    original=(R/'h4kv_local1/fused.cu').read_text()
    lo=original.index('        uint32_t bp=cast_smem_ptr_to_uint(')
    hi=original.index('\n        float alpha[2]',lo)
    reader=original[lo:hi].replace('s.scratch.attn.bias[c][stage].data()', 's.scratch.attn.bias[stage].data()+c*4096').replace('typename C::BL{}','typename C::BPL{}')
    lo=s.index('        float mx[2]={-INFINITY,-INFINITY};')
    hi=s.index('\n        float alpha[2]',lo)
    s=s[:lo]+reader+s[hi:]
s=s.replace('  m.def("forward_audit",&launch_forward);',
            '  m.def("forward_audit",&launch_forward);\n  m.def("bias_head_last",[](){return '+('false' if a.planar else 'true')+';});')
folder=R/a.artifact
assert not(folder/'build-ready.json').exists()
folder.mkdir(exist_ok=True);(folder/'fused.cu').write_text(s)

folder=R/'front8_hlast'
if not (folder/'fused.cu').exists():
    f=(R/'front8/fused.cu').read_text()
    f=f.replace('  #pragma unroll\n  for(int h=0;h<4;++h){',
                '  uint2 packed_bias;auto heads=reinterpret_cast<B*>(&packed_bias);\n  #pragma unroll\n  for(int h=0;h<4;++h){')
    f=f.replace('if(lane==0)bias[h*L*L+r]=', 'heads[h]=')
    f=f.replace('    heads[h]=__float2bfloat16_rn(mask && !mask[j]?-3.3895313892515355e38f:b);\n  }',
                '    heads[h]=__float2bfloat16_rn(mask && !mask[j]?-3.3895313892515355e38f:b);\n  }\n  if(lane==0)*reinterpret_cast<uint2*>(bias+r*4)=packed_bias;')
    f=f.replace('torch::empty({1,4,L,L},x.options())', 'torch::empty({1,L,L,4},x.options())')
    f=f.replace('m.def("front",&front);','m.def("bias_head_last",[](){return true;});m.def("front",&front);')
    assert 'bias+r*4' in f
    folder.mkdir(exist_ok=True);(folder/'fused.cu').write_text(f)
