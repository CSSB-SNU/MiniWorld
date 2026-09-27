from pathlib import Path
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_width as W
from miniworld_engine.kernels.trimul_inproj.cuda import h100_wide_forward as F
R=Path(__file__).resolve().parent

class FusedB1:
    def __init__(self,p):
        self.p=p
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),'-I'+str(R),'-DWEIGHT_SPLITS=32','-DWIDTH=256']
        ks={}
        for name in ('source','finish'):
            body=(R/('b1_'+name+'.cu')).read_text()
            for h in ('mma.cuh','b1_ln.inc'):body=body.replace(f'#include "{h}"',(R/h).read_text())
            out=T.compile_text(body,flags)
            k=T.load_unit(str(out),'mw_d256_b1_'+name).kernel('mw_d256_b1_'+name)
            k.set_max_dynamic_smem(131200 if name=='source' else W.tuning(256)[2]);ks[name]=k
        self.source,self.finish=ks['source'],ks['finish']
        drv=self.finish.unit.drv
        occ=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(drv.d.CUfunction(int(self.finish.handle)),256,W.tuning(256)[2])))
        self.grid=torch.cuda.get_device_properties(p.x.device).multi_processor_count*occ
        self.params=T._launch_module().Struct([F.tm(p.tri,[64,64],[p.M,512],[p.M*2]),p.maps[0],p.maps[1],p.maps[2],p.dy,p.ds,p.floats[2],p.floats[3],p.tensors[7],p.dg,p.floats[5],p.floats[6],p.floats[7],p.M,p.n])
    def __call__(self):
        self.source.launch((128,1,1),(256,1,1),[self.params],131200)
        W.launch(self.finish,self.p.params,self.grid,D=256)
