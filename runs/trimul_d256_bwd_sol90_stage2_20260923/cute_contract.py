"""Explicit batched contraction schedule; uses existing CuTe SM90 GEMM."""
from quack.gemm import gemm


class Contractions:
    def __init__(self,plan,tm=128,tn=128,ping=True,cm=1,cn=1):
        p=plan.p;ab=p.front.ab;d=p.D;h=2*d
        # CuTe interface takes A[B,M,K], B[B,N,K], D[B,M,N].
        self.args=[
            (p.dt[:d],ab[h:h+d].transpose(-1,-2),p.dl[:d]),
            (p.dt[:d].transpose(-1,-2),ab[:d].transpose(-1,-2),p.dr[:d]),
            (ab[h+d:],p.dt[d:],p.dl[d:]),
            (ab[d:h],p.dt[d:].transpose(-1,-2),p.dr[d:]),
        ]
        self.cfg=dict(tile_M=tm,tile_N=tn,cluster_M=cm,cluster_N=cn,pingpong=ping)

    def __call__(self):
        for a,b,c in self.args:gemm(a,b,c,None,None,**self.cfg)
