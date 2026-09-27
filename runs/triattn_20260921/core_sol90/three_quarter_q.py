"""Cache three of four Q K16 fragments (12 GPR/thread) in installed pipeline."""
def transform(s,full_half):
    assert full_half in (0,1)
    old='''            cute::gemm(tiled_mma_qk, tQ(_,_,1,hh,cwg), tK(_,_,1,c,st), acc);'''
    assert s.count(old)==1
    s=s.replace(old,'''            if constexpr(hh==FULL_HALF){
                cute::gemm(tiled_mma_qkr, tQr(_,_,1), tK(_,_,1,c,st), acc);
            } else {
                cute::gemm(tiled_mma_qk, tQ(_,_,1,hh,cwg), tK(_,_,1,c,st), acc);
            }'''.replace('FULL_HALF',str(full_half)))
    old='''            auto a0=tQr0(_,_,Int<0>{});auto a1=tQr1(_,_,Int<0>{});
            warpgroup_fence_operand(a0);warpgroup_fence_operand(a1);'''
    assert s.count(old)==2
    partial=1-full_half
    new='''            auto aPARTIAL=tQrPARTIAL(_,_,Int<0>{});
            warpgroup_fence_operand(tQrFULL);warpgroup_fence_operand(aPARTIAL);'''
    s=s.replace(old,new.replace('PARTIAL',str(partial)).replace('FULL',str(full_half)))
    return s
