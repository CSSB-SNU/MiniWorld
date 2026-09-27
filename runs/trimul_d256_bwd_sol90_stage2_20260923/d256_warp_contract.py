from pathlib import Path
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

class WarpContract:
    def __init__(self,plan):
        p=plan.p;assert p.D==256;self.p=p;n=p.n;d=256;h=512;ab=p.front.ab
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc')]
        self.cubin=T.compile(Path(__file__).with_suffix('.cu'),flags)
        self.k=T.load_unit(str(self.cubin),'mw_d256_warp_contract').kernel('mw_d256_warp_contract')
        self.smem=163840+128;self.k.set_max_dynamic_smem(self.smem)
        L=T._launch_module()
        aa=(p.dt[:d],p.dt[:d],ab[h+d:],ab[d:h]);bb=(ab[h:h+d],ab[:d],p.dt[d:],p.dt[d:])
        def tm(x,box):return L.tensor_map(x,box,dims=[n,d*n],strides_bytes=[n*2],swizzle='none',l2='128B')
        self.params=L.Struct([*[tm(x,[16,16]) for x in aa],*[tm(x,[16,64] if i==2 else [64,16]) for i,x in enumerate(bb)],p.dl,p.dr,n])
    def __call__(self):self.k.launch((32*(self.p.n//16)*(self.p.n//64),1,1),(256,1,1),[self.params],self.smem)
