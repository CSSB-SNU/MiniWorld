"""Plain row-tile GEMM/LN launch, followed by small affine/weight reduction."""
from pathlib import Path
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

class PrefixChunkDxLN:
    def __init__(self,plan,slots=3,minblocks=None):
        p=plan.p;self.p=p;d=p.D;cols=256 if d==512 else 128;root=Path(__file__).resolve().parent
        self.grid=p.M//64
        self.partial=torch.empty((self.grid,2,d),device=p.x.device,dtype=torch.float32)
        splits=plan.b7.splits if d==256 else plan.input_splits
        joint=int(d>256 and plan.joint_choice is not None)
        helpers=(root.parent/'trimul_d256_bwd_sol90_20260923/mma.cuh').read_text()
        helpers+='\n'+(root/'mma192_offset.cuh').read_text()+'\n'+(root/'mma256.cuh').read_text()
        body=(root/'prefix_chunk_dx_ln.cu').read_text().replace('// MMA_HELPERS',helpers)
        call=f'mma{cols//2}<1,1>(v,a,b,step>0||q>0);'
        body=body.replace('// WIDE_MMA',call)
        minblocks=minblocks or (3 if d<512 else 2)
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWIDTH={d}',f'-DDX_SLOTS={slots}',f'-DDX_COLS={cols}',f'-DDX_MINBLOCKS={minblocks}',f'-DWEIGHT_SPLITS={splits}',f'-DJOINT_INPUT={joint}']
        self.cubin=T.compile_text(body,flags)
        unit=T.load_unit(str(self.cubin),'mw_prefix_chunk_dx_ln');self.kernel=unit.kernel('mw_prefix_chunk_dx_ln');self.reducer=unit.kernel('mw_prefix_chunk_dx_finish')
        self.smem=max(slots*(1+cols//64)*4096,192*d)+128;self.kernel.set_max_dynamic_smem(self.smem)
        drv=self.kernel.unit.drv;fun=drv.d.CUfunction(int(self.kernel.handle))
        attr=lambda name:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,name),fun)))
        self.registers=attr('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=attr('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fun,256,self.smem)))
        L=T._launch_module()
        gp=L.tensor_map(plan.dx.input,[64,32],dims=[p.M,9*d],strides_bytes=[p.M*2],swizzle='128B',l2='128B')
        weights=L.tensor_map(plan.dx.weights,[64,32],dims=[d,9*d],strides_bytes=[d*2],swizzle='128B',l2='128B')
        tm=lambda t:L.tensor_map(t,[64,16],dims=[d,p.M],strides_bytes=[d*2],swizzle='128B',l2='128B')
        self.params=L.Struct([gp,weights,tm(p.x),tm(p.dy),tm(p.dx),p.floats[0],self.partial,p.floats[8],p.floats[9],p.floats[7],*p.tensors[17:21],p.dwg,p.M])
    def gemm(self):self.kernel.launch((self.grid,1,1),(256,1,1),[self.params],self.smem)
    def finish(self):self.reducer.launch((self.p.D//32*8,1,1),(256,1,1),[self.params],0)
    def __call__(self):self.gemm();self.finish()
