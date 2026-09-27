"""Explicit normalized-input reuse and optional native MMA stores."""
import triton as tr
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout, SliceLayout, DotOperandLayout, NVMMADistributedLayout, NVMMASharedLayout
from triton.experimental.gluon.language.nvidia.ampere import mma_v2
from triton.experimental.gluon.language.nvidia.hopper import warpgroup_mma, fence_async_shared

@g.jit
def prologue(Z,XLN,LNW,LNB,WQKVG,WB,Q,K,V,G,BIAS,XOUT,NI,NJ,s_zi,s_zj,eps,
             C:gl.constexpr,H:gl.constexpr,D:gl.constexpr,HB:gl.constexpr,BI:gl.constexpr,BJ:gl.constexpr,BN:gl.constexpr,
             LN_MODE:gl.constexpr,WRITE_X:gl.constexpr,FMA_M2:gl.constexpr,FMA_MERGE:gl.constexpr,FMA_AFF:gl.constexpr,
             NOASM_AFF:gl.constexpr,ADDR_SPLIT:gl.constexpr,QKV_PAIR:gl.constexpr,DOT_F32:gl.constexpr,J_FAST:gl.constexpr=False,
             MMA:gl.constexpr=3,NW:gl.constexpr=8,STORE:gl.constexpr=0,WM:gl.constexpr=4):
    BM:gl.constexpr=BI*BJ
    HD:gl.constexpr=H*D
    L:gl.constexpr=BlockedLayout([1,8],[2,16],[NW,1],[1,0])
    LS:gl.constexpr=BlockedLayout([1,8],[4,8],[NW,1],[1,0])
    LW:gl.constexpr=BlockedLayout([8,1],[16,2],[1,NW],[0,1])
    M:gl.constexpr=NVMMADistributedLayout([3,0],[NW,1],[16,BN,16]) if MMA==3 else NVMMADistributedLayout([2,0],[WM,NW//WM],[16,8])
    ST:gl.constexpr=M if STORE==1 else LS
    tile=gl.program_id(0)+gl.program_id(1)*gl.num_programs(0)
    pi=tile//gl.num_programs(1) if J_FAST else gl.program_id(0)
    pj=tile%gl.num_programs(1) if J_FAST else gl.program_id(1)
    r=gl.arange(0,BM,layout=SliceLayout(1,L))
    c=gl.arange(0,C,layout=SliceLayout(0,L))
    ii=pi*BI+r//BJ;jj=pj*BJ+r%BJ
    valid=(ii<NI)&(jj<NJ)
    if LN_MODE==0:
        x=gl.load(XLN+(ii.to(gl.int64)*NJ+jj)[:,None]*C+c[None,:],valid[:,None],other=0)
    else:
        z=gl.load(Z+ii.to(gl.int64)[:,None]*s_zi+jj.to(gl.int64)[:,None]*s_zj+c[None,:],valid[:,None],other=0).to(gl.float32)
        wln=gl.load(LNW+c).to(gl.float32);bln=gl.load(LNB+c).to(gl.float32)
        mean=gl.sum(z,1)/C
        centered=z-mean[:,None]
        var=gl.sum(centered*centered,1)/C
        inv=1./gl.sqrt_rn(var+eps)
        y=centered*inv[:,None]*wln[None,:]+bln[None,:]
        x=gl.where(valid[:,None],y,0.).to(gl.bfloat16)
    aa=gl.convert_layout(x,DotOperandLayout(0,M,2))
    wc=gl.arange(0,C,layout=SliceLayout(1,LW))
    wn=gl.arange(0,BN,layout=SliceLayout(0,LW))
    sr=gl.arange(0,BM,layout=SliceLayout(1,ST))
    sn=gl.arange(0,BN,layout=SliceLayout(0,ST))
    si=(pi*BI+sr//BJ).to(gl.int64);sj=(pj*BJ+sr%BJ).to(gl.int64)
    sm=(si<NI)&(sj<NJ)
    for n0 in gl.static_range(0,4*HD,BN):
        w=gl.load(WQKVG+(n0+wn)[None,:].to(gl.int64)*C+wc[:,None])
        zero=gl.full((BM,BN),0,gl.float32,M)
        if MMA==3:
            wb=gl.allocate_shared_memory(gl.bfloat16,(C,BN),NVMMASharedLayout(128,16,transposed=True),w)
            fence_async_shared()
            acc=warpgroup_mma(aa,wb,zero)
        else:
            bb=gl.convert_layout(w,DotOperandLayout(1,M,2))
            acc=mma_v2(aa,bb,zero)
        out=gl.convert_layout(acc.to(gl.bfloat16),ST)
        if n0//HD==3:
            off=(si*NJ+sj)[:,None]*HD+(n0%HD)+sn[None,:]
            gl.store(G+off,out,sm[:,None])
        else:
            off=((si[:,None]*H+((n0%HD+sn)//D)[None,:])*NJ+sj[:,None])*D+(sn%D)[None,:]
            if n0//HD==0:gl.store(Q+off,out,sm[:,None])
            elif n0//HD==1:gl.store(K+off,out,sm[:,None])
            else:gl.store(V+off,out,sm[:,None])
    MB:gl.constexpr=NVMMADistributedLayout([3,0],[NW,1],[16,HB,16]) if MMA==3 else M
    ba=gl.convert_layout(x,DotOperandLayout(0,MB,2))
    bn=gl.arange(0,HB,layout=SliceLayout(0,LW))
    w=gl.load(WB+bn[None,:].to(gl.int64)*C+wc[:,None])
    zero=gl.full((BM,HB),0,gl.float32,MB)
    if MMA==3:
        wb=gl.allocate_shared_memory(gl.bfloat16,(C,HB),NVMMASharedLayout(128,16,transposed=True),w)
        fence_async_shared()
        acc=warpgroup_mma(ba,wb,zero)
    else:
        bb=gl.convert_layout(w,DotOperandLayout(1,MB,2))
        acc=mma_v2(ba,bb,zero)
    LB:gl.constexpr=BlockedLayout([4,1],[32,1],[1,NW],[0,1])
    vals=gl.convert_layout(acc.to(gl.bfloat16).to(gl.float32),LB)
    br=gl.arange(0,BM,layout=SliceLayout(1,LB));bh=gl.arange(0,HB,layout=SliceLayout(0,LB))
    bii=pi*BI+br//BJ;bjj=pj*BJ+br%BJ
    gl.store(BIAS+(bh[None,:].to(gl.int64)*NI+bii[:,None])*NJ+bjj[:,None],vals,((bii<NI)&(bjj<NJ))[:,None]&(bh<H)[None,:])
