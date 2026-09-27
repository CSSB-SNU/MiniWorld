"""All four current D256 contractions, with a fixed-length native schedule."""
from pathlib import Path
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

class FixedContract:
    def __init__(self,plan,groups=2,order=1):
        p=plan.p;assert p.D==256 and p.n==384
        root=Path(__file__).resolve().parent
        body=(root/'d256_fixed_contract.cu').read_text().replace('// MMA_HELPER',(root/'mma_offset.cuh').read_text())
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DROW_GROUPS={groups}',f'-DGRID_ORDER={order}',f'-DMIN_BLOCKS={3 if groups==1 else 2}']
        self.cubin=T.compile_text(body,flags);self.k=T.load_unit(str(self.cubin),'mw_d256_fixed_contract').kernel('mw_d256_fixed_contract')
        self.smem=(groups+2)*8192*3+128;self.threads=128*groups;self.grid=4*256*(384//(64*groups))*3
        self.k.set_max_dynamic_smem(self.smem);L=T._launch_module();ab=p.front.ab;d=256;h=512;n=384
        tm=lambda t,c:L.tensor_map(t,[64,64],dims=[n,c*n],strides_bytes=[n*2],swizzle='128B',l2='256B')
        aa=(p.dt[:d],p.dt[:d],ab[h+d:],ab[d:h]);bb=(ab[h:h+d],ab[:d],p.dt[d:],p.dt[d:])
        self.params=L.Struct([*[tm(t,d) for t in aa],*[tm(t,d) for t in bb],tm(p.dl,h),tm(p.dr,h)])
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda n:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,n),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,self.threads,self.smem)))
    def __call__(self):self.k.launch((self.grid,1,1),(self.threads,1,1),[self.params],self.smem)
