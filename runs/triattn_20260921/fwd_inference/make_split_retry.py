"""Separate rare stable repair so its register demand cannot spill the normal path."""
import argparse,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--artifact',required=True)
a=p.parse_args();r=Path(__file__).resolve().parent;s=(r/a.base/'fused.cu').read_text()
s=s.replace('int L; int* retry_count;','int L; int* retry_count; int* repair_flags;')
s=s.replace('template<int Capacity,int Consumers,int Stages,int Projectors>\n__global__',
            'template<int Capacity,int Consumers,int Stages,int Projectors,bool Repair>\n__global__')
needle='  if(tid==0) {\n    for(int cc=0;cc<Consumers;++cc)s.zfull[cc].init(1);'
repl='''  int flagbase=(h*L+row)*(nt+1);
  if constexpr(Repair) { if(p.repair_flags[flagbase]==0)return; }
  if constexpr(!Repair) { if(tid<=nt)p.repair_flags[flagbase+tid]=0; }
  if(tid==0) {
    for(int cc=0;cc<Consumers;++cc)s.zfull[cc].init(1);'''
assert needle in s;s=s.replace(needle,repl,1)
lo=s.index('    compute(cute::false_type{});');hi=s.index('    int ep_tid,',lo)
old=s[lo:hi]
old=old.replace('      compute(cute::true_type{});',
                '      if(lane==0){p.repair_flags[flagbase+qt+1]=1;atomicExch(p.repair_flags+flagbase,1);}')
s=s[:lo]+'    if constexpr(Repair)compute(cute::true_type{});\n    else {\n'+old+'    }\n'+s[hi:]
needle='  typename C::Params p{'
s=s.replace('    if(qt<nt) {\n      auto bg=',
            '    if(qt<nt && (!Repair || p.repair_flags[flagbase+qt+1])) {\n      auto bg=',1)
s=s.replace(needle,'  auto flags=torch::empty({4,L,L/64+1},z.options().dtype(torch::kInt32));\n'+needle,1)
s=s.replace('counts.defined()?counts.data_ptr<int>():nullptr};','counts.defined()?counts.data_ptr<int>():nullptr,flags.data_ptr<int>()};')
old='''  C10_CUDA_CHECK(cudaFuncSetAttribute(qkv_attention_inference<Capacity,Consumers,Stages,Projectors>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(typename C::Shared)));
  qkv_attention_inference<Capacity,Consumers,Stages,Projectors><<<dim3(4,L),Consumers*128,sizeof(typename C::Shared),at::cuda::getCurrentCUDAStream()>>>(p);'''
new='''  C10_CUDA_CHECK(cudaFuncSetAttribute(qkv_attention_inference<Capacity,Consumers,Stages,Projectors,false>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(typename C::Shared)));
  C10_CUDA_CHECK(cudaFuncSetAttribute(qkv_attention_inference<Capacity,Consumers,Stages,Projectors,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(typename C::Shared)));
  qkv_attention_inference<Capacity,Consumers,Stages,Projectors,false><<<dim3(4,L),Consumers*128,sizeof(typename C::Shared),at::cuda::getCurrentCUDAStream()>>>(p);
  qkv_attention_inference<Capacity,Consumers,Stages,Projectors,true><<<dim3(4,L),Consumers*128,sizeof(typename C::Shared),at::cuda::getCurrentCUDAStream()>>>(p);'''
assert old in s;s=s.replace(old,new)
s=s.replace('qkv_attention_inference','qkv_attention_split_retry')
folder=r/a.artifact;assert not (folder/'build-ready.json').exists()
folder.mkdir(exist_ok=True);(folder/'fused.cu').write_text(s)
