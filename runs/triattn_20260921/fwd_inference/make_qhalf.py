"""Keep one or both K16 query slices in packed registers across the key loop."""
import argparse,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--artifact',required=True)
p.add_argument('--full',action='store_true')
a=p.parse_args();r=Path(__file__).resolve().parent;s=(r/a.base/'fused.cu').read_text()
s=s.replace('  using Score=', '  using ScoreRS=decltype(make_tiled_mma(SM90_64x64x16_F32BF16BF16_RS<GMMA::Major::K,GMMA::Major::K>{}));\n  using Score=',1)
old='    auto qa=st.partition_fragment_A(sq);'
new='''    auto qa=st.partition_fragment_A(sq);
    typename C::ScoreRS qr_mma;
    auto qr=qr_mma.get_slice(lane).partition_fragment_A(sq);
    auto qr_words=recast<uint32_t>(qr);
    static_assert(size(qr_words)==8);
    int qr_row=(lane/32)*16+lane%16,qr_col=(lane%32/16)*8;
    uint32_t qr_addr=cast_smem_ptr_to_uint(s.scratch.attn.q[0][c].data())+
      as_position_independent_swizzle_layout(typename C::QL{})(make_coord(qr_row,qr_col))*2;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
      :"=r"(qr_words(0)),"=r"(qr_words(1)),"=r"(qr_words(2)),"=r"(qr_words(3)):"r"(qr_addr):"memory");'''
assert old in s;s=s.replace(old,new)
if a.full:
    needle='    for(int kt=0;kt<nt;++kt) {'
    load='''    uint32_t qr_addr2=cast_smem_ptr_to_uint(s.scratch.attn.q[0][c].data())+
      as_position_independent_swizzle_layout(typename C::QL{})(make_coord(qr_row,qr_col+16))*2;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
      :"=r"(qr_words(4)),"=r"(qr_words(5)),"=r"(qr_words(6)),"=r"(qr_words(7)):"r"(qr_addr2):"memory");
'''
    assert needle in s;s=s.replace(needle,load+needle,1)
old='      flash::gemm<true,0>(smma,qa,kb,score);'
new='''      auto qr_first=qr(_,_,0);
      warpgroup_fence_operand(qr_first);warpgroup_fence_operand(score);warpgroup_arrive();
      qr_mma.accumulate_=GMMA::ScaleOut::Zero;smma.accumulate_=GMMA::ScaleOut::One;
      cute::gemm(qr_mma,qr_first,kb(_,_,0),score);
      cute::gemm(smma,qa(_,_,1),kb(_,_,1),score);
      warpgroup_commit_batch();warpgroup_wait<0>();
      warpgroup_fence_operand(qr_first);warpgroup_fence_operand(score);'''
assert old in s;s=s.replace(old,new)
if a.full:
    s=s.replace('warpgroup_fence_operand(qr_first);','warpgroup_fence_operand(qr);')
    needle='      cute::gemm(smma,qa(_,_,1),kb(_,_,1),score);'
    assert needle in s
    s=s.replace(needle,'      qr_mma.accumulate_=GMMA::ScaleOut::One;\n      cute::gemm(qr_mma,qr(_,_,1),kb(_,_,1),score);')
s=re.sub(r'qkv_attention_[a-z0-9_]+','qkv_attention_qhalf_'+a.artifact,s)
folder=r/a.artifact;assert not (folder/'build-ready.json').exists()
folder.mkdir(exist_ok=True);(folder/'fused.cu').write_text(s)
