"""M128 N64, three score buffers, two full consumer warpgroups."""
def transform(s):
 s=s.replace('Score=decltype(partition_fragment_C(qk,Shape<_64,_32>{}))', 'Score=decltype(partition_fragment_C(qk,Shape<_64,Int<N>>{}))')
 s=s.replace('QK qk;PV pv;auto tq=qk.get_slice(0);auto tp=pv.get_slice(0);','QK qk;PV pv;LS ls;auto tq=qk.get_slice(0);auto tp=pv.get_slice(0);auto tl=ls.get_slice(0);')
 s=s.replace('float sum[2][2]={},mx[2][2]', '''using Den=decltype(partition_fragment_C(ls,Shape<_64,_8>{}));
 Den den[2];clear(den[0]);clear(den[1]);
 auto lb=tl.partition_fragment_B(make_tensor(make_smem_ptr(s.ones),S1{}));
 float mx[2][2]''')
 s=s.replace('for(int u=0;u<4;u++)', 'for(int u=0;u<N/8;u++)')
 s=s.replace('local_tile(kall,Shape<_32,_32>{}', 'local_tile(kall,Shape<Int<N>,_32>{}')
 s=s.replace('for(int col=1;col<8;col++)m=', 'for(int col=1;col<N/4;col++)m=')
 s=s.replace('for(int col=0;col<8;col++){\n    float x=sr', 'for(int col=0;col<N/4;col++){\n    float x=sr')
 s=s.replace('sum[hh][row]*=alpha;', 'auto lr=make_tensor(den[hh].data(),flash::convert_layout_acc_rowcol(den[hh].layout()));lr(row,0)*=alpha;lr(row,1)*=alpha;')
 s=s.replace('    sum[hh][row]+=sr(row,col);\n','')
 s=s.replace('for(int n=0;n<8;n++){auto x=__floats2bfloat162_rn', 'for(int n=0;n<N/4;n++){auto x=__floats2bfloat162_rn')
 s=s.replace('local_tile(vall,Shape<_32,_32>{}', 'local_tile(vall,Shape<_32,Int<N>>{}')
 s=s.replace('if constexpr(Safe)warpgroup_fence_operand(acc[hh]);', 'if constexpr(Safe){warpgroup_fence_operand(acc[hh]);warpgroup_fence_operand(den[hh]);}')
 s=s.replace('for(int kk=0;kk<2;kk++)gemm(pv,pp(_,_,kk),vb(_,_,kk),acc[hh]);', '''for(int kk=0;kk<N/16;kk++)gemm(pv,pp(_,_,kk),vb(_,_,kk),acc[hh]);
  #pragma unroll
  for(int kk=0;kk<N/16;kk++)gemm(ls,pp(_,_,kk),lb(_,_,kk),den[hh]);''')
 s=s.replace('warpgroup_fence_operand(acc[0]);warpgroup_fence_operand(acc[1]);','warpgroup_fence_operand(acc[0]);warpgroup_fence_operand(acc[1]);warpgroup_fence_operand(den[0]);warpgroup_fence_operand(den[1]);')
 s=s.replace('float val=sum[hh][row];val+=__shfl_xor_sync(0xffffffff,val,1);val+=__shfl_xor_sync(0xffffffff,val,2);', 'auto lr=make_tensor(den[hh].data(),flash::convert_layout_acc_rowcol(den[hh].layout()));float val=lr(row,0);')
 s=s.replace('u=(idx>>9)&3,hh=(idx/2048)%2,cc=idx/4096', 'u=(idx>>9)%(N/8),hh=(idx/(64*N))%2,cc=idx/(M*N)')
 s=s.replace('k=kt*LN+cc*32+8*u', 'k=kt*LN+cc*N+8*u')
 s=s.replace('p.L','768').replace('p.scale','0x1.6a09e6p-3f')
 s=s.replace('using namespace SOL_NAMESPACE;int L=q.size(-2);', 'using namespace SOL_NAMESPACE;int L=q.size(-2);TORCH_CHECK(L==768 && float(scale)==0x1.6a09e6p-3f,"L768 standard scale required");')
 return s
