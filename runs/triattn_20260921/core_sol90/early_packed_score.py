"""Compress each completed E into8 BF16-pair registers before next bias loads.

Isolated numerical experiment: hot period rescaling now acts on rounded P.
SAFE arithmetic is unchanged. This needs full accuracy qualification before
any installation, even if ordinary cases remain bitwise.
"""
def transform(s,qregs):
 if qregs:
  old='static constexpr bool kQinRegs = (kFlags_ & 1048576) != 0;'
  assert old in s
  s=s.replace(old,'static constexpr bool kQinRegs = (kFlags_ & 1048576) != 0 || (!kSafe && (kFlags_ & 1073741824));')
 old='''            if constexpr (kPrmtPack) {'''
 assert s.count(old)==1
 s=s.replace(old,'''            if constexpr(kFast){dst32(pr)=__float_as_uint(acc(pr));}
            else if constexpr (kPrmtPack) {''')
 old='''        #pragma unroll
        for (int mi = 0; mi < kNRows; ++mi) {
            #pragma unroll
            for (int ni = 0; ni < kNC; ++ni) { s_rc(mi, ni) = ex2_approx(fmaf(s_rc(mi, ni), c_l2, nm[hh][mi])); }
        }
        warpgroup_fence_operand(acc);'''
 assert old in s
 s=s.replace(old,'''        if constexpr(kFast){
            // Fragment pairs alternate the two accumulator rows. Compress
            // from low to high indices, so no unread score is overwritten.
            #pragma unroll
            for(int pr=0;pr<8;++pr){
                float x=ex2_approx(fmaf(acc(2*pr),c_l2,nm[hh][pr&1]));
                float y=ex2_approx(fmaf(acc(2*pr+1),c_l2,nm[hh][pr&1]));
                auto h2=__floats2bfloat162_rn(x,y);
                acc(pr)=__uint_as_float(reinterpret_cast<uint32_t const&>(h2));
            }
            #pragma unroll
            for(int pr=8;pr<16;++pr)acc(pr)=0.f;
        }else{
            #pragma unroll
            for (int mi = 0; mi < kNRows; ++mi) {
                #pragma unroll
                for (int ni = 0; ni < kNC; ++ni) { s_rc(mi, ni) = ex2_approx(fmaf(s_rc(mi, ni), c_l2, nm[hh][mi])); }
            }
        }
        warpgroup_fence_operand(acc);''')
 old='''                        #pragma unroll
                        for (int ni = 0; ni < kNC; ++ni) { p_rc(mi, ni) *= f; }'''
 assert s.count(old)==1
 s=s.replace(old,'''                        if constexpr(kFast){
                            #pragma unroll
                            for(int pair=0;pair<4;++pair){
                                int ix=2*pair+mi;
                                uint32_t bits=__float_as_uint(pend(ix));
                                auto bf=*reinterpret_cast<__nv_bfloat162 const*>(&bits);
                                float2 v=__bfloat1622float2(bf);
                                auto scaled=__floats2bfloat162_rn(v.x*f,v.y*f);
                                pend(ix)=__uint_as_float(reinterpret_cast<uint32_t const&>(scaled));
                            }
                        }else{
                            #pragma unroll
                            for (int ni = 0; ni < kNC; ++ni) { p_rc(mi, ni) *= f; }
                        }''')
 return s
