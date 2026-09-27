"""Keep only one M64 Q half in registers in the qualified R3/M128 pipeline."""
def transform(s,half):
 assert half in (0,1)
 old='static constexpr bool kQinRegs = (kFlags_ & 1048576) != 0;'
 assert s.count(old)==1
 s=s.replace(old,'static constexpr bool kQinRegs = (kFlags_ & 1048576) != 0 || (kFlags_ & 1073741824) != 0;')
 old='for (int hq = 0; hq < 2; ++hq) {'
 assert s.count(old)==1
 s=s.replace(old,'for (int hq = kFast ? HALF : 0; hq < (kFast ? HALF+1 : 2); ++hq) {'.replace('HALF',str(half)))
 a=s.index('    auto issue_qk =');b=s.index('    auto issue_pv =',a)
 block=s[a:b];assert block.count('if constexpr (kQinRegs)')==1
 block=block.replace('if constexpr (kQinRegs)','if constexpr (kQinRegs && (!kFast || hh == HALF))'.replace('HALF',str(half)))
 s=s[:a]+block+s[b:]
 a=s.index('    auto issue_group =');b=s.index('    auto bodyW =',a)
 block=s[a:b];assert block.count('if constexpr (kQinRegs)')==1
 block=block.replace('if constexpr (kQinRegs)','if constexpr (kQinRegs && (!kFast || h1 == HALF))'.replace('HALF',str(half)))
 s=s[:a]+block+s[b:]
 old='if constexpr (kQinRegs) { warpgroup_fence_operand(tQr0); warpgroup_fence_operand(tQr1); }'
 assert s.count(old)==2
 new='if constexpr (kQinRegs) { if constexpr(!kFast || HALF==0)warpgroup_fence_operand(tQr0); if constexpr(!kFast || HALF==1)warpgroup_fence_operand(tQr1); }'.replace('HALF',str(half))
 return s.replace(old,new)
