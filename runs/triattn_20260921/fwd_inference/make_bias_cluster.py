"""Two resident-QKV CTAs share their common pair bias by TMA multicast."""
from pathlib import Path
r=Path(__file__).resolve().parent
s=(r/'resident6/fused.cu').read_text()
s=s.replace('using TB=decltype(make_tma_copy(SM90_TMA_LOAD{},',
            'using TB=decltype(make_tma_copy(SM90_TMA_LOAD_MULTICAST{},')
s=s.replace('auto tb=make_tma_copy(SM90_TMA_LOAD{},',
            'auto tb=make_tma_copy(SM90_TMA_LOAD_MULTICAST{},')
s=s.replace('    Scratch scratch;',
            '    Scratch scratch;\n    cutlass::arch::ClusterBarrier bempty[Consumers][Stages];')
s=s.replace('void qkv_attention_inference(', 'void __cluster_dims__(1,2,1) qkv_attention_inference(')
s=s.replace('    prefetch_tma_descriptor(p.z.get_tma_descriptor());',
'''    for(int cc=0;cc<Consumers;++cc)for(int st=0;st<Stages;++st)s.bempty[cc][st].init(8);
    prefetch_tma_descriptor(p.z.get_tma_descriptor());''')
pos=s.index('  __syncthreads();')
s=s[:pos]+s[pos:].replace('  __syncthreads();', '  cute::cluster_arrive();cute::cluster_wait();',1)
s=s.replace('        s.bfull[c][stage].arrive_and_expect_tx(4096*sizeof(Element));',
'''        if(cute::block_rank_in_cluster()==0)
          s.bempty[c][stage].wait((((qt/Consumers)*(nt/Stages)+kt/Stages)%2)^1);
        s.bfull[c][stage].arrive_and_expect_tx(4096*sizeof(Element));''')
old='        tma_load(p.bias,bb,make_tensor(make_smem_ptr(s.scratch.attn.bias[c][stage].data()),typename C::BL{}),s.bfull[c][stage]);'
new='''        if(cute::block_rank_in_cluster()==0) {
          auto bs=make_tensor(make_smem_ptr(s.scratch.attn.bias[c][stage].data()),typename C::BL{});
          auto slice=p.bias.get_slice(_0{});
          copy(p.bias.with(reinterpret_cast<uint64_t&>(s.bfull[c][stage]),uint16_t(3)),
               slice.partition_S(bb),slice.partition_D(bs));
        }'''
assert old in s;s=s.replace(old,new)
s=s.replace('        if constexpr(Stages==1) {\n          cutlass::arch::NamedBarrier::sync(128,c+1);',
'''        cutlass::arch::NamedBarrier::sync(128,c+1);
        if(lane%32==0)s.bempty[c][stage].arrive(0,1u);
        if constexpr(Stages==1) {''')
end=s.index('\ntemplate<int Capacity,int Consumers,int Stages,int Projectors>\nvoid launch')
s=s[:end].rstrip()[:-1]+'  cute::cluster_arrive();cute::cluster_wait();\n}\n'+s[end:]
path=r/'cluster2'
assert not (path/'build-ready.json').exists()
path.mkdir(exist_ok=True)
(path/'fused.cu').write_text(s)
