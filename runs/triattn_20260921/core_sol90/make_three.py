from pathlib import Path
r=Path(__file__).resolve().parent
s=(r/'m128_pipe.cu').read_text()
s=s.replace('Score score;', 'Score score;Score next_score;Score third_score;')
s=s.replace(' auto pp=make_tensor_like<Element>(make_tensor(score.data(),flash::convert_layout_acc_Aregs<PV>(score.layout())));',' auto pp=make_tensor_like<Element>(make_tensor(score.data(),flash::convert_layout_acc_Aregs<PV>(score.layout())));\n auto pp1=make_fragment_like(pp);')
s=s.replace('auto issue_qk=[&](int seq,auto half)', 'auto issue_qk=[&](auto& score,int seq,auto half)')
s=s.replace('auto exponentiate=[&](auto half,auto seed)', 'auto exponentiate=[&](auto& score,auto half,auto seed)')
s=s.replace('auto pack=[&]()', 'auto pack=[&](auto& score,auto& pp)')
s=s.replace('auto issue_pv=[&](int seq,auto half)', 'auto issue_pv=[&](int seq,auto half,auto& pp)')
a=s.index('  if(seq<2*nk){s.bf');b=s.index('  auto qall=',a)
init=s[a:b]
s=s[:a]+s[b:]
a=s.index(' auto issue_qk=')
s=s[:a]+' auto init_score=[&](auto& score,int seq) __attribute__((always_inline)) {\n'+init+' };\n'+s[a:]
s=s.replace('warpgroup_fence_operand(score);warpgroup_fence_operand(pp);','warpgroup_fence_operand(score);warpgroup_fence_operand(next_score);warpgroup_fence_operand(third_score);warpgroup_fence_operand(pp);warpgroup_fence_operand(pp1);')
s=s.replace('issue_qk(seq,half);', 'init_score(score,seq);issue_qk(score,seq,half);').replace('exponentiate(half,cute::true_type{});pack();issue_pv(seq,half);', 'exponentiate(score,half,cute::true_type{});pack(score,pp);issue_pv(seq,half,pp);')
a=s.index(' }else{\n  issue_qk(0,');b=s.index(' auto id=',a)
s=s[:a]+(r/'m128_three_hot.cuh').read_text()+s[b:]
(r/'m128_three.cu').write_text(s)
s=s.replace('using PV=decltype', 'using QR=decltype(make_tiled_mma(GMMA::rs_op_selector<Element,Element,float,Shape<_64,Int<N>,_32>>()));\nusing PV=decltype',1)
mark=' s.qr.wait(0);asm volatile("":::"memory");'
qregs=r'''
 QR qr;qr.accumulate_=GMMA::ScaleOut::One;
 auto qshared=make_tensor(make_smem_ptr(s.q[wg]),SQ{});
 auto tqreg=qr.get_thread_slice(t);
 auto qa0=tqreg.partition_fragment_A(local_tile(qshared,Shape<_64,_32>{},make_coord(0,0)));
 auto qa1=make_fragment_like(qa0);
 auto qcopy=make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N,Element>{},qr);
 auto qtcopy=qcopy.get_thread_slice(t);
 #pragma unroll
 for(int hh=0;hh<2;hh++){
  auto qhalf=local_tile(qshared,Shape<_64,_32>{},make_coord(hh,0));
  auto qs=qtcopy.partition_S(qhalf);auto qd=qtcopy.retile_D(hh==0?qa0:qa1);
  copy(qcopy,qs,qd);
 }
 warpgroup_fence_operand(qa0);warpgroup_fence_operand(qa1);
'''
s=s.replace(mark,mark+qregs)
s=s.replace('  auto qall=make_tensor(make_smem_ptr(s.q[wg]),SQ{});\n  auto qs=local_tile(qall,Shape<_64,_32>{},make_coord(hh,0));auto qa=tq.partition_fragment_A(qs);', '  auto& qa=hh==0?qa0:qa1;')
s=s.replace('gemm(qk,qa(_,_,kk),kb(_,_,kk),score)', 'gemm(qr,qa(_,_,kk),kb(_,_,kk),score)')
(r/'m128_three_qr.cu').write_text(s)
