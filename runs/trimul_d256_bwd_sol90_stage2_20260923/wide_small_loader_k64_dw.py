"""A trailing loader warp feeds aligned WGMMA groups with K32 input tiles."""
from pathlib import Path
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_wide_forward as F
class SmallLoaderK64DW:
    def __init__(self,plan,columns=256,minblocks=2):
        p=plan.p;d=p.D;root=Path(__file__).resolve().parent;self.threads=columns//128*128+32
        body=(root/'wide_small_loader_k64_dw.cu').read_text().replace('// MMA_HELPERS',(root/'mma_offset.cuh').read_text())
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(F.headers()),'-DMW_MINB=1','-DMW_K1_STREAM=0',f'-DWIDTH={d}',f'-DWEIGHT_SPLITS={plan.input_splits}',f'-DDW_COLUMNS={columns}',f'-DDW_MINBLOCKS={minblocks}',f'-DDW_SLOTS={2 if columns==256 else 3}']
        self.cubin=T.compile_text(body,flags)
        self.k=T.load_unit(str(self.cubin),'mw_wide_small_loader_k64_dw').kernel('mw_wide_small_loader_k64_dw')
        self.smem=(8192+128*columns)*(2 if columns==256 else 3)+128;self.k.set_max_dynamic_smem(self.smem)
        drv=self.k.unit.drv
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(drv.d.CUfunction(int(self.k.handle)),self.threads,self.smem)))
        L=T._launch_module()
        gp=L.tensor_map(p.gp_all,[64,64],dims=[p.M,8*d],strides_bytes=[p.M*2],swizzle='128B',l2='128B')
        xn=L.tensor_map(p.xn,[64,64],dims=[d,p.M],strides_bytes=[d*2],swizzle='128B',l2='128B')
        self.params=L.Struct([gp,xn,p.floats[7],p.M]);self.grid=(8*d//64*plan.input_splits,(d+columns-1)//columns,1)
    def __call__(self):self.k.launch(self.grid,(self.threads,1,1),[self.params],self.smem)
