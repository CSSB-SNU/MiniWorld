"""Only odd query-half P uses shared memory; even P stays in registers.

Two shared slots hold odd chunks j. Reuse from j-4 is protected by the
previous odd publication j-2: its body j-1 wait retires PV(j-4) in every
warp before that publication barrier. Thus one publication barrier per odd
chunk suffices. This halves stores/barriers vs all-shared P and removes the
runtime modulo-three descriptor calculation. Full Q replaces exactly the
eight P registers removed from the installed partial-Q kernel.
"""
import re
from shared_probability import transform as shared_transform


def transform(s):
    s=shared_transform(s,full_q=True,matrix_store=True)
    old='using TiledMmaPV = std::conditional_t<kSharedP,TiledMmaPVS,TiledMmaPVR>;'
    assert s.count(old)==1
    s=s.replace(old,'using TiledMmaPV = TiledMmaPVR;')
    s=s.replace('R*3*64*CW','R*2*64*CW').replace('(cwg*3+pseq%3)*64*CW','(cwg*2+((pseq/2)&1))*64*CW')
    s,n=re.subn(r'if constexpr\(!kFast\)\{warpgroup_fence_operand\(PCb\[([^\]]+)\]\);\}',r'if constexpr(!kFast || (\1)==0){warpgroup_fence_operand(PCb[\1]);}',s)
    assert n>=8,n
    for helper in ('issue_pv','issue_pv_prefenced'):
        a=s.index('    auto '+helper+' =')
        b=s.index('\n    };',a)+len('\n    };')
        block=s[a:b]
        block=block.replace('if constexpr(!kFast){warpgroup_fence_operand(tP);}', 'if constexpr(!kFast || hh==0){warpgroup_fence_operand(tP);}')
        old='        if constexpr(kFast){'
        assert block.count(old)==1
        # Keep the T-dependent guard outside the generic-lambda hh guard:
        # nvcc checks non-dependent shared members before hh substitution.
        block=block.replace(old,'        if constexpr(kFast){\n        if constexpr(hh==1){')
        old='            auto pa=wg_mma_pv.partition_fragment_A(sp);'
        assert block.count(old)==1
        block=block.replace(old,'''            typename T::TiledMmaPVS pvs;pvs.accumulate_=GMMA::ScaleOut::One;
            auto pa=pvs.get_slice(0).partition_fragment_A(sp);''')
        block=block.replace('cute::gemm(tiled_mma_pv,pa(', 'cute::gemm(pvs,pa(')
        a_else=block.index('        }else{')
        b_else=block.index('        }\n        warpgroup_commit_batch();',a_else)
        rs_body=block[a_else+len('        }else{'):b_else]
        block=block[:b_else]+'''        }
        }else{'''+rs_body+block[b_else:]
        s=s[:a]+block+s[b:]
    a=s.index('    auto pack_chunk =');b=s.index('    auto chunk_rowmax =',a)
    block=s[a:b]
    left=block.index('        if constexpr(kFast){')
    split=block.index('        }else{',left)
    right=block.rindex('        }\n    };')
    hot=block[left+len('        if constexpr(kFast){'):split]
    regular=block[split+len('        }else{'):right]
    block=block[:left]+'''        if constexpr(kFast){
            if(pseq&1){'''+hot+'''
            }else{'''+regular+'''
            }
        }else{'''+regular+'''
        }
    };
'''
    return s[:a]+block+s[b:]
