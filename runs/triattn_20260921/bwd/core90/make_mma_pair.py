from pathlib import Path

root=Path(__file__).resolve().parent.parent
helper='''
// Two independent MMAs share one WGMMA fence/commit/wait group.
template<bool Zero, class MMA, class A0, class B0, class C0, class A1, class B1, class C1>
__device__ __forceinline__ void gemm_pair(MMA& mma,A0& a0,B0& b0,C0& c0,A1& a1,B1& b1,C1& c1){
  constexpr bool RS=!cute::is_base_of<cute::GMMA::DescriptorIterator,typename MMA::FrgTypeA>::value;
  if constexpr(RS){warpgroup_fence_operand(a0);warpgroup_fence_operand(a1);}
  warpgroup_fence_operand(c0);warpgroup_fence_operand(c1);warpgroup_arrive();
  mma.accumulate_=Zero?GMMA::ScaleOut::Zero:GMMA::ScaleOut::One;
  #pragma unroll
  for(int k=0;k<size<2>(a0);++k){cute::gemm(mma,a0(_,_,k),b0(_,_,k),c0);mma.accumulate_=GMMA::ScaleOut::One;}
  mma.accumulate_=Zero?GMMA::ScaleOut::Zero:GMMA::ScaleOut::One;
  #pragma unroll
  for(int k=0;k<size<2>(a1);++k){cute::gemm(mma,a1(_,_,k),b1(_,_,k),c1);mma.accumulate_=GMMA::ScaleOut::One;}
  warpgroup_commit_batch();warpgroup_wait<0>();
  warpgroup_fence_operand(c0);warpgroup_fence_operand(c1);
  if constexpr(RS){warpgroup_fence_operand(a0);warpgroup_fence_operand(a1);}
}
'''
s=(root/'dq/rs_coop_tma/fused.cu').read_text().replace('struct Config {',helper+'\nstruct Config {')
old='flash::gemm<true,-1>(smma,qa,kb,score);flash::gemm<true,0>(smma,da,vb,dp);'
assert old in s;s=s.replace(old,'gemm_pair<true>(smma,qa,kb,score,da,vb,dp);')
d=root/'dq/rs_mma_pair';d.mkdir(exist_ok=True);(d/'fused.cu').write_text(s)

for artifact,name in [('rs_tma_store','rs_mma_pair'),('rs_r8_two','rs_r8_pair')]:
    s=(root/'bias_fusion'/artifact/'grouped.cu').read_text().replace('template<int R> struct Config {',helper+'\ntemplate<int R> struct Config {')
    old='''      flash::gemm<true,-1>(smma,ka,qb,score);
      flash::gemm<true,0>(smma,va,dob,dp);'''
    assert old in s;s=s.replace(old,'      gemm_pair<true>(smma,ka,qb,score,va,dob,dp);')
    old='''      flash::gemm<false,-1>(gmma,pa,dbt,dv);
      flash::gemm<false,0>(gmma,dsa,qbt,dk);'''
    assert old in s;s=s.replace(old,'      gemm_pair<false>(gmma,pa,dbt,dv,dsa,qbt,dk);')
    d=root/'bias_fusion'/name;d.mkdir(exist_ok=True);(d/'grouped.cu').write_text(s)
