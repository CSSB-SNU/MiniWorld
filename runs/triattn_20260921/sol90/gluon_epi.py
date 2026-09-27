"""Explicit layouts: gate in the load layout, then convert once for tensor cores."""
import triton as tr
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout, SliceLayout, DotOperandLayout, NVMMADistributedLayout, NVMMASharedLayout
from triton.experimental.gluon.language.extra import libdevice
from triton.experimental.gluon.language.nvidia.ampere import mma_v2
from triton.experimental.gluon.language.nvidia.hopper import warpgroup_mma, fence_async_shared

@g.jit
def epilogue(O,G,WT,Z,OUT,I,J,so_b,so_i,so_h,so_j,sg_b,sg_i,sg_j,swt_k,sz_b,sz_r,sz_c,su_b,su_i,su_j,
             H:gl.constexpr,D:gl.constexpr,C:gl.constexpr,BI:gl.constexpr,BJ:gl.constexpr,
             ENDING:gl.constexpr,RESIDUAL:gl.constexpr,USE_LD:gl.constexpr,RESIDUAL_OUT:gl.constexpr=False,
             MMA:gl.constexpr=3,NW:gl.constexpr=4,VEC:gl.constexpr=4,WM:gl.constexpr=4):
    BM:gl.constexpr=BI*BJ
    K:gl.constexpr=H*D
    L:gl.constexpr=BlockedLayout([1,VEC],[4,8],[NW,1],[1,0])
    M:gl.constexpr=NVMMADistributedLayout([3,0],[NW,1],[16,C,16]) if MMA==3 else NVMMADistributedLayout([2,0],[WM,NW//WM],[16,8])
    r=gl.arange(0,BM,layout=SliceLayout(1,L))
    k=gl.arange(0,K,layout=SliceLayout(0,L))
    n=gl.arange(0,C,layout=SliceLayout(0,L))
    ii=gl.program_id(0)*BI+r//BJ
    jj=gl.program_id(1)*BJ+r%BJ
    b=gl.program_id(2).to(gl.int64)
    valid=(ii<I)&(jj<J)
    ii64=ii.to(gl.int64);jj64=jj.to(gl.int64)
    o=gl.load(O+b*so_b+ii64[:,None]*so_i+jj64[:,None]*so_j+((k//D)*so_h+k%D)[None,:],valid[:,None],other=0)
    gate=gl.load(G+b*sg_b+ii64[:,None]*sg_i+jj64[:,None]*sg_j+k[None,:],valid[:,None],other=0).to(gl.float32)
    if USE_LD:e=libdevice.exp(-gate)
    else:e=gl.exp(-gate)
    sigmoid=(1./(1.+e)).to(gl.bfloat16).to(gl.float32)
    gated=(o.to(gl.float32)*sigmoid).to(gl.bfloat16)
    kk=gl.arange(0,K,layout=SliceLayout(1,L))
    w=gl.load(WT+kk[:,None].to(gl.int64)*swt_k+n[None,:])
    aa=gl.convert_layout(gated,DotOperandLayout(0,M,2))
    zero=gl.full((BM,C),0,gl.float32,M)
    if MMA==3:
        wb=gl.allocate_shared_memory(gl.bfloat16,(K,C),NVMMASharedLayout(128,16),w)
        fence_async_shared()
        acc=warpgroup_mma(aa,wb,zero)
    else:
        bb=gl.convert_layout(w,DotOperandLayout(1,M,2))
        acc=mma_v2(aa,bb,zero)
    u=gl.convert_layout(acc.to(gl.bfloat16),L)
    zi=jj64 if ENDING else ii64
    zj=ii64 if ENDING else jj64
    if RESIDUAL:
        zp=Z+b*sz_b+zi[:,None]*sz_r+zj[:,None]*sz_c+n[None,:]
        z=gl.load(zp,valid[:,None],other=0)
        val=(z.to(gl.float32)+u.to(gl.float32)).to(gl.bfloat16)
        if RESIDUAL_OUT:dst=OUT+b*su_b+zi[:,None]*su_i+zj[:,None]*su_j+n[None,:]
        else:dst=zp
        gl.store(dst,val,valid[:,None])
    else:
        gl.store(OUT+b*su_b+ii64[:,None]*su_i+jj64[:,None]*su_j+n[None,:],u,valid[:,None])
