"""Inference producer/consumer variant: Q/gate local, streaming KV, gated output only."""
from pathlib import Path
R = Path(__file__).resolve().parent
s = (R.parent/'fwd_training/qkv_stream_wide4b/fused.cu').read_text()
s = s.replace('// Training forward: one score buffer, complete QK groups before scalar score reads.',
              '// Inference forward: streaming KV producer and attention consumers, no training saves.')
s = s.replace('q[Consumers],kv[2][Stages];','q[Consumers],gate[Consumers],kv[2][Stages];')
s = s.replace('TO saveq,savek,savev,saveg,out;float* lse;','TO out;')
s = s.replace('qkv_attention_stream','qkv_attention_inference_stream')
s = s.replace('uint32_t dst=cast_smem_ptr_to_uint(s.q[c].data());',
              'uint32_t dst=cast_smem_ptr_to_uint(which==0?s.gate[c].data():s.q[c].data());')
begin = s.index('      if(lane==0) {\n        auto const& save=which==0?p.saveg:p.saveq;')
end = s.index('      cutlass::arch::NamedBarrier::sync(128,c+1);',begin)
s = s[:begin]+s[end:]
begin = s.index('        // Only one query group saves training K/V;')
end = s.index('        s.ready[st].arrive();',begin)
s = s[:begin]+s[end:]
begin = s.index('      if(lane%4==0) {\n        int qr=get<0>(sc(2*mi));')
end = s.index('    uint32_t op=',begin)
s = s[:begin]+'    }\n'+s[end:]
s = s.replace('      store_pair(op+off,out(x)*inv[(x%4)/2],out(x+1)*inv[(x%4)/2]);', '''      float r0=float(Element(out(x)*inv[(x%4)/2]));
      float r1=float(Element(out(x+1)*inv[(x%4)/2]));
      float g0=float(s.gate[c][off/2]),g1=float(s.gate[c][off/2+1]);
      float sg0,sg1,e0=1.f+ex2(-g0*LOG2E),e1=1.f+ex2(-g1*LOG2E);
      asm("rcp.approx.ftz.f32 %0,%1;":"=f"(sg0):"f"(e0));
      asm("rcp.approx.ftz.f32 %0,%1;":"=f"(sg1):"f"(e1));
      store_pair(op+off,r0*sg0,r1*sg1);''')
s = s.replace('torch::Tensor q,torch::Tensor k,torch::Tensor v,torch::Tensor gate,torch::Tensor out,torch::Tensor lse',
              'torch::Tensor out')
s = s.replace('tb,save(q),save(k),save(v),save(gate),save(out),lse.data_ptr<float>(),L',
              'tb,save(out),L')
s = s.replace('std::vector<torch::Tensor> launch_forward','torch::Tensor launch_forward')
begin = s.index('  auto gate=torch::empty_like(z);')
end = s.index('  if(L==64)launch',begin)
s = s[:begin]+'  auto out=torch::empty_like(z);\n'+s[end:]
s = s.replace('(z,wq,wk,wv,wg,b,q,k,v,gate,o,lse)','(z,wq,wk,wv,wg,b,out)')
begin = s.index('  auto view='); end = s.index('\n}',begin)
s = s[:begin]+'  return out;'+s[end:]
for artifact,src in [('stream4',s),('stream2',s.replace('launch<1024,4,2>','launch<1024,2,2>')
                                     .replace('Config<1024,4,2>::Shared','Config<1024,2,2>::Shared'))]:
    path = R/artifact
    assert not (path/'build-ready.json').exists()
    path.mkdir(exist_ok=True)
    (path/'fused.cu').write_text(src)
