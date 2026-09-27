from pathlib import Path
import torch,os
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_width as W

R=Path(__file__).resolve().parent

class B7:
    def __init__(self,p,splits=8):
        assert p.D==256
        self.p,self.splits=p,splits
        self.mask=p.mask.to(torch.bfloat16)
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v',
               '-I'+str(T._upstream()/'csrc'),'-I'+str(R),f'-DWEIGHT_SPLITS={splits}','-DWIDTH=256']
        # Hash the local included files into the translation unit cache key.
        kernels={}
        for name in ('source','finish'):
            body=(R/(('source_loop' if name=='source' and os.environ.get('B7_LOOP')=='1' else name)+'.cu')).read_text()
            for header in ('mma.cuh','base.cuh'):
                body=body.replace(f'#include "{header}"',(R/header).read_text())
            out=T.compile_text(body,flags)
            k=T.load_unit(str(out),'mw_d256_b7_'+name).kernel('mw_d256_b7_'+name)
            k.set_max_dynamic_smem(114816 if name=='source' else W.tuning(256)[2])
            kernels[name]=k
        self.source,self.finish=kernels['source'],kernels['finish']
        drv=self.finish.unit.drv
        occ=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(drv.d.CUfunction(int(self.finish.handle)),256,W.tuning(256)[2])))
        self.grid=torch.cuda.get_device_properties(p.x.device).multi_processor_count*occ
        L=T._launch_module()
        dy=lambda x:L.tensor_map(x,[64,32],dims=[p.M,512],strides_bytes=[p.M*2],swizzle='128B',l2='128B')
        self.params=L.Struct([p.maps[0],W.tm(p.w1),dy(p.dl),dy(p.dr),self.mask,*p.gp,p.floats[7],p.M])
    def __call__(self):
        self.source.launch((32*self.splits,1,1),(128,1,1),[self.params],114816)
        if hasattr(self,'wide_finish'):self.wide_finish()
        else:W.launch(self.finish,self.p.params7,self.grid,D=256)
        return self.p.outputs
