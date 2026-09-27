"""Resident fused QKV: max-free normal path with on-chip stable retry for unsafe tiles."""
from pathlib import Path
r = Path(__file__).resolve().parent
s = (r/'resident4/fused.cu').read_text()
s = s.replace('    Scratch scratch;', '    int unsafe[Consumers][4];\n    Scratch scratch;')
s = s.replace('TB bias; TO out; int L;', 'TB bias; TO out; int L; int* retry_count;')
s = s.replace('  for(int batch=0;', '  int bias_epoch=0;\n  for(int batch=0;')
s = s.replace('((qt/Consumers)*(nt/Stages)+kt/Stages)%2', '(bias_epoch+kt/Stages)%2')
start = s.index('      if(lane==0)for(int b=0;')
end = s.index('    int ep_tid,', start)
old = s[start:end]
initial_load = old[:old.index('    auto score=')]
body = old[old.index('    auto score='):]
body = body.replace('    auto out=partition_fragment_C(pmma,Shape<_64,_32>{}); clear(out);', '    clear(out);')
body = body.replace('float m[2]={-INFINITY,-INFINITY},l[2]={1.f,1.f};',
                    'float m[2]={-INFINITY,-INFINITY}; l[0]=l[1]=Safe?1.f:0.f;')
body = body.replace('mx[mi]=fmaxf(mx[mi],score(x));', 'if constexpr(Safe) mx[mi]=fmaxf(mx[mi],score(x));')
begin = body.index('        #pragma unroll\n        for(int mi=0;mi<2;++mi) {\n          mx[mi]=')
finish = body.index('        #pragma unroll\n        for(int x=0;x<size(score);', begin)
body = body[:begin]+'        if constexpr(Safe) {\n'+body[begin:finish]+'''        } else { m[0]=m[1]=16.f; alpha[0]=alpha[1]=1.f; }
'''+body[finish:]
body = body.replace('          ls[mi]+=__shfl_xor_sync(0xffffffffu,ls[mi],1);\n          ls[mi]+=__shfl_xor_sync(0xffffffffu,ls[mi],2);',
'''          if constexpr(Safe) {
            ls[mi]+=__shfl_xor_sync(0xffffffffu,ls[mi],1);
            ls[mi]+=__shfl_xor_sync(0xffffffffu,ls[mi],2);
          }''')
body = body.replace('for(int x=0;x<size(out);++x) out(x)*=alpha[(x%4)/2];',
                    'for(int x=0;x<size(out);++x) if constexpr(Safe) out(x)*=alpha[(x%4)/2];')
replacement = '''    auto out=partition_fragment_C(pmma,Shape<_64,_32>{});
    float l[2];
    auto compute=[&](auto safe_tag) {
      constexpr bool Safe=decltype(safe_tag)::value;
'''+initial_load+body+'''
      if constexpr(!Safe) {
        #pragma unroll
        for(int mi=0;mi<2;++mi) {
          l[mi]+=__shfl_xor_sync(0xffffffffu,l[mi],1);
          l[mi]+=__shfl_xor_sync(0xffffffffu,l[mi],2);
        }
      }
      bias_epoch+=nt/Stages;
    };
    compute(cute::false_type{});
    bool bad=!(l[0]>=1e-16f && l[0]<=1e16f && l[1]>=1e-16f && l[1]<=1e16f);
    #pragma unroll
    for(int x=0;x<size(out);++x) bad=bad || !isfinite(out(x));
    bool warp_bad=__any_sync(0xffffffffu,bad);
    if(lane%32==0) s.unsafe[c][lane/32]=warp_bad;
    cutlass::arch::NamedBarrier::sync(128,c+1);
    bool retry=s.unsafe[c][0] || s.unsafe[c][1] || s.unsafe[c][2] || s.unsafe[c][3];
    if(retry) {
      if(lane==0 && p.retry_count) atomicAdd(p.retry_count,1);
      compute(cute::true_type{});
    }
'''
s = s[:start]+replacement+s[end:]
s = s.replace('            torch::Tensor out) {', '            torch::Tensor out,torch::Tensor counts) {')
s = s.replace('tb,save(out),L};', 'tb,save(out),L,counts.defined()?counts.data_ptr<int>():nullptr};')
s = s.replace('torch::Tensor wg,torch::Tensor b) {', 'torch::Tensor wg,torch::Tensor b,torch::Tensor counts={}) {')
s = s.replace('(z,wq,wk,wv,wg,b,out);', '(z,wq,wk,wv,wg,b,out,counts);')
s = s.replace('if(L<=128)launch<128,2,2,2>',
              'if(L==64)launch<128,2,1,2>(z,wq,wk,wv,wg,b,out,counts);\n  else if(L<=128)launch<128,2,2,2>')
s = s.replace('m.def("forward",&launch_forward);', '''m.def("forward",[](torch::Tensor z,torch::Tensor q,torch::Tensor k,torch::Tensor v,torch::Tensor g,torch::Tensor b){
    return launch_forward(z,q,k,v,g,b);
  });
  m.def("forward_audit",&launch_forward);''')
rounded = s.replace('ls[mi]+=score(x);',
                    'if constexpr(Safe) ls[mi]+=score(x); else ls[mi]+=float(Element(score(x)));')
for name, source in [('hot4r',rounded)]:
    path=r/name
    assert not (path/'build-ready.json').exists()
    path.mkdir(exist_ok=True)
    (path/'fused.cu').write_text(source)
