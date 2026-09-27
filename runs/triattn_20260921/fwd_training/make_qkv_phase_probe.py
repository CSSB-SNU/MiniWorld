"""Instrument CTA projection/attention phase durations; never use for selection."""
from pathlib import Path
r=Path(__file__).resolve().parent
s=(r/'qkv_compact_async4/fused.cu').read_text()
s=s.replace('float* lse; int L;', 'float* lse; unsigned long long* cycles; int L;')
needle='  // Fewer projection WGs'
s=s.replace(needle,'  if(tid==0)p.cycles[(row*4+h)*3]=clock64();\n'+needle,1)
needle='  // QKV saves are complete. The projection scratch may become bias storage.\n  __syncthreads();'
assert needle in s
s=s.replace(needle,needle+'\n  if(tid==0)p.cycles[(row*4+h)*3+1]=clock64();',1)
needle='  }\n}\n\ntemplate<int Capacity,int Consumers,int Stages,int Projectors>\nvoid launch'
assert needle in s
s=s.replace(needle,'  }\n  __syncthreads();\n  if(tid==0)p.cycles[(row*4+h)*3+2]=clock64();\n}\n\ntemplate<int Capacity,int Consumers,int Stages,int Projectors>\nvoid launch',1)
s=s.replace('torch::Tensor out,torch::Tensor lse) {', 'torch::Tensor out,torch::Tensor lse,torch::Tensor cycles) {')
s=s.replace('save(out),lse.data_ptr<float>(),L};', 'save(out),lse.data_ptr<float>(),reinterpret_cast<unsigned long long*>(cycles.data_ptr<int64_t>()),L};')
needle='  if(L<=128)launch'
s=s.replace(needle,'  auto cycles=torch::empty({L,4,3},z.options().dtype(torch::kInt64));\n'+needle,1)
s=s.replace('b,q,k,v,gate,o,lse);', 'b,q,k,v,gate,o,lse,cycles);')
s=s.replace('view(v),gate};', 'view(v),gate,cycles};')
d=r/'qkv_compact_phase';assert not (d/'build-ready.json').exists()
d.mkdir(exist_ok=True);(d/'fused.cu').write_text(s)
