"""Diagnostic-only clock64 instrumentation; never select using this binary."""
from pathlib import Path
root=Path(__file__).resolve().parent
s=(root/'resident_qkv_reuse/fused.cu').read_text()
s=s.replace('float* lse; int L;','float* lse; int L; int64_t* trace;')
s=s.replace('  if(tid==0) {\n    s.zfull.init(1);',
'''  unsigned long long begin=clock64();
  if(tid==0)p.trace[(row*4+h)*(Consumers+2)]=begin;
  if(tid==0) {
    s.zfull.init(1);''')
s=s.replace('  auto bg=p.bias.get_tma_tensor',
'''  if(tid==0)p.trace[(row*4+h)*(Consumers+2)+1]=clock64();
  auto bg=p.bias.get_tma_tensor''')
s=s.replace('\n}\n\ntemplate<int Capacity,int Consumers,int Stages>\nvoid launch',
'''\n  if(lane==0)p.trace[(row*4+h)*(Consumers+2)+2+c]=clock64();
}

template<int Capacity,int Consumers,int Stages>
void launch''')
s=s.replace('torch::Tensor out,torch::Tensor lse) {','torch::Tensor out,torch::Tensor lse,torch::Tensor trace) {')
s=s.replace('lse.data_ptr<float>(),L};','lse.data_ptr<float>(),L,trace.data_ptr<int64_t>()};')
s=s.replace('  if(L<=128)launch', '  auto trace=torch::empty({L*4,L<=384?4:6},z.options().dtype(torch::kInt64));\n  if(L<=128)launch')
s=s.replace('b,q,k,v,o,lse);','b,q,k,v,o,lse,trace);')
s=s.replace('view(k),view(v)};','view(k),view(v),trace};')
d=root/'resident_qkv_phase';d.mkdir(exist_ok=True);(d/'fused.cu').write_text(s)
