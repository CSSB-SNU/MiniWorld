"""Parallel QKV producers; retain KV and reload training Q saves for each query."""
import argparse
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--consumers',type=int,choices=(4,6),required=True)
a=ap.parse_args();root=Path(__file__).resolve().parent
s=(root/'resident_qkv_reuse/fused.cu').read_text()
s=s.replace('  using TO=decltype', '  using TQ=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)nullptr),QS{},QStride{}),QL{},Shape<_64,_32>{},_1{}));\n  using TO=decltype')
s=s.replace('array_aligned<Element,4096,1024> z;','array_aligned<Element,4096,1024> z[Consumers];')
s=s.replace('array_aligned<Element,2048,1024> qkv[3][Capacity/64];','array_aligned<Element,2048,1024> q[Consumers],kv[2][Capacity/64];')
s=s.replace(' zfull,wfull,bfull[Consumers][Stages];',' zfull[Consumers],qfull[Consumers],wfull,bfull[Consumers][Stages];')
s=s.replace('TB bias; TO saveq','TB bias; TQ loadq; TO saveq')
s=s.replace('s.zfull.init(1); s.wfull.init(1);','for(int cc=0;cc<Consumers;++cc) {s.zfull[cc].init(1);s.qfull[cc].init(1);} s.wfull.init(1);')
s=s.replace('  // A single input Z chunk feeds all three projections. Full head weights stay resident.\n  if(c==0) {','  // Each warpgroup independently projects disjoint tiles; all reuse head weights.\n  {')
s=s.replace('s.scratch.proj.z.data()','s.scratch.proj.z[c].data()')
s=s.replace('    if(lane==0) {\n      s.wfull.arrive', '    if(tid==0) {\n      s.wfull.arrive')
s=s.replace('for(int qt=0;qt<nt;++qt)', 'for(int qt=c;qt<nt;qt+=Consumers)')
s=s.replace('s.zfull.', 's.zfull[c].')
s=s.replace('zs,s.zfull);', 'zs,s.zfull[c]);')
# Q is read back by TMA. Wait for global destination completion, not just
# the shared source consumption guaranteed by cute::tma_store_wait().
pos=s.index('  // QKV saves are complete.')
s=s[:pos].replace('tma_store_arrive();tma_store_wait<0>();',
                  'tma_store_arrive();asm volatile("cp.async.bulk.wait_group 0;":::"memory");')+s[pos:]
s=s.replace('NamedBarrier::sync(128,0)', 'NamedBarrier::sync(128,c+1)')
s=s.replace('s.qkv[which][qt].data()', '(which==0?s.q[c].data():s.kv[which-1][qt].data())')
s=s.replace('s.qkv[0][qt].data()', 's.q[c].data()')
s=s.replace('s.qkv[1][kt].data()', 's.kv[0][kt].data()')
s=s.replace('s.qkv[2][kt].data()', 's.kv[1][kt].data()')
needle='''    if(lane==0)for(int b=0;b<Stages && b<nt;++b)load_bias(b);'''
replacement='''    if(lane==0) {
      auto qg=p.loadq.get_tma_tensor(make_shape(L,_32{},_4{},L));
      auto qq=local_tile(qg(_,_,h,row),Shape<_64,_32>{},make_coord(qt,0));
      s.qfull[c].arrive_and_expect_tx(2048*sizeof(Element));
      tma_load(p.loadq,qq,make_tensor(make_smem_ptr(s.q[c].data()),typename C::QL{}),s.qfull[c]);
      for(int b=0;b<Stages && b<nt;++b)load_bias(b);
    }
    s.qfull[c].wait((qt/Consumers)%2);'''
assert needle in s;s=s.replace(needle,replacement)
needle='''  typename C::Params p{tz,weight(wq),weight(wk),weight(wv),tb,save(q),save(k),save(v),save(out),lse.data_ptr<float>(),L};'''
replacement='''  auto qg=make_tensor(make_gmem_ptr((Element const*)q.data_ptr()),make_shape(L,_32{},_4{},L),typename C::QStride{128,_1{},_32{},int64_t(L)*128});
  auto tq=make_tma_copy(SM90_TMA_LOAD{},qg,typename C::QL{},Shape<_64,_32>{},_1{});
  typename C::Params p{tz,weight(wq),weight(wk),weight(wv),tb,tq,save(q),save(k),save(v),save(out),lse.data_ptr<float>(),L};'''
assert needle in s;s=s.replace(needle,replacement)
if a.consumers==6:
    s=s.replace('launch<768,4,2>', 'launch<768,6,2>').replace('launch<1024,4,1>', 'launch<1024,6,1>')
    s=s.replace('sizeof(Config<1024,4,1>::Shared)', 'sizeof(Config<1024,6,1>::Shared)')
else:
    s=s.replace('launch<1024,4,1>', 'launch<1024,4,2>')
    s=s.replace('sizeof(Config<1024,4,1>::Shared)', 'sizeof(Config<1024,4,2>::Shared)')
d=root/('resident_kv_parallel%db'%a.consumers);d.mkdir(exist_ok=True);(d/'fused.cu').write_text(s)
