"""Two M256 consumers with 224 registers each and bias-prefetched QRS/SS."""
def transform(s,qr):
 from four_query_halves import transform as four
 s=four(s)
 old='  init_score(score,0);issue_qk(score,0,_0{});drain();'
 assert s.count(old)==1
 s=s.replace(old,'  init_score(score,0);init_score(next_score,1);issue_qk(score,0,_0{});drain();')
 assert s.count('   init_score(ns,seq+1);')==1
 s=s.replace('   init_score(ns,seq+1);','')
 old='   pack(sc,prob);issue_pv(seq,half,prob);'
 assert s.count(old)==1
 s=s.replace(old,old+'init_score(sc,seq+2);')
 s=s.replace('using namespace SOL_NAMESPACE;int L=q.size(-2);',
             'using namespace SOL_NAMESPACE;int L=q.size(-2);TORCH_CHECK(L==768,"M256 prototype requires L768");')
 if not qr:return s
 s=s.replace('using QK=decltype(make_tiled_mma(GMMA::ss_op_selector',
             'using QK=decltype(make_tiled_mma(GMMA::rs_op_selector')
 pos=s.index(' auto init_score=')
 s=s[:pos]+''' auto sqr=make_tensor(make_smem_ptr(s.q[wg]),SQ{});
 auto qproto=qk.get_thread_slice(t).partition_fragment_A(local_tile(sqr,Shape<_64,_32>{},make_coord(0,0)));
 decltype(qproto) qregs[4];
 auto qcopy=make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N,Element>{},qk);
 auto qthread=qcopy.get_thread_slice(t);
 #pragma unroll
 for(int hh=0;hh<4;++hh){
  auto qs=qthread.partition_S(local_tile(sqr,Shape<_64,_32>{},make_coord(hh,0)));
  auto qd=qthread.retile_D(qregs[hh]);copy(qcopy,qs,qd);
  warpgroup_fence_operand(qregs[hh]);
 }
'''+s[pos:]
 old='''  auto qall=make_tensor(make_smem_ptr(s.q[wg]),SQ{});
  auto qs=local_tile(qall,Shape<_64,_32>{},make_coord(hh,0));auto qa=tq.partition_fragment_A(qs);'''
 assert s.count(old)==1
 s=s.replace(old,'  auto& qa=qregs[hh];')
 return s
