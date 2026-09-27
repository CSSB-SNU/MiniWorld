"""Mixed RS/SS QK: cache only one K16 slice of both query halves (8 GPR)."""
def transform(s,kblock=0):
 assert kblock in (0,1)
 old='static constexpr bool kQinRegs = (kFlags_ & 1048576) != 0;'
 assert s.count(old)==1
 s=s.replace(old,'static constexpr bool kQinRegs = (kFlags_ & 1048576) != 0 || (kFlags_ & 1073741824) != 0;')
 a=s.index('    auto issue_qk =');b=s.index('    auto issue_pv =',a)
 block=s[a:b]
 old='        if constexpr (kQinRegs) {'
 assert block.count(old)==1
 new='''        if constexpr(kFast) {
            tiled_mma_qkr.accumulate_ = GMMA::ScaleOut::One;
            tiled_mma_qk.accumulate_ = GMMA::ScaleOut::One;
            auto& tQr = hh==0?tQr0:tQr1;
'''
 for kb in range(2):
  if kb==kblock:new+='            cute::gemm(tiled_mma_qkr, tQr(_,_,KB), tK(_,_,KB,c,st), acc);\n'.replace('KB',str(kb))
  else:new+='            cute::gemm(tiled_mma_qk, tQ(_,_,KB,hh,cwg), tK(_,_,KB,c,st), acc);\n'.replace('KB',str(kb))
 new+='        } else if constexpr(kQinRegs) {'
 block=block.replace(old,new)
 s=s[:a]+block+s[b:]
 # The compiled fast flag never instantiates schedule W. Keep its generic QRS
 # behavior unchanged; whole-register fences below also remain for non-fast flags.
 old='if constexpr (kQinRegs) { warpgroup_fence_operand(tQr0); warpgroup_fence_operand(tQr1); }'
 assert s.count(old)==2
 new='''if constexpr(kQinRegs) {
        if constexpr(kFast) {
            auto a0=tQr0(_,_,Int<KB>{});auto a1=tQr1(_,_,Int<KB>{});
            warpgroup_fence_operand(a0);warpgroup_fence_operand(a1);
        } else {warpgroup_fence_operand(tQr0);warpgroup_fence_operand(tQr1);}
    }'''.replace('KB',str(kblock))
 return s.replace(old,new)
