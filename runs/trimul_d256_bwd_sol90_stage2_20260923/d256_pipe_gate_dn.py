"""D256 gate/ordered-dNorm pipeline, independent of the selected dispatch."""
from pathlib import Path
from three_role_register_pool import initial_pool_three
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_wide_forward as F


class PipeGateDN:
    def __init__(self,plan,producer=64,consumer=192):
        p=plan.p;assert p.D==256 and p.n==384;root=Path(__file__).resolve().parent
        body=(root/'d256_pipe_gate_dn.cu').read_text().replace('// MMA256',(root/'mma256.cuh').read_text());self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(F.headers()),'-DMW_MINB=1','-DMW_K1_STREAM=0','-DTMN_SIGMOID_TANH=1',f'-DPRODUCER_REGS={producer}',f'-DCONSUMER_REGS={consumer}']
        cubin=T.compile_text(body,flags);initial=((producer+2*consumer+23)//24)*8
        self.cubin,self.pool_metadata=initial_pool_three(cubin,'mw_d256_pipe_gate_dn',initial,producer,consumer)
        self.k=T.load_unit(str(self.cubin),'mw_d256_pipe_gate_dn').kernel('mw_d256_pipe_gate_dn');self.smem=212992+128;self.k.set_max_dynamic_smem(self.smem)
        L=T._launch_module();fields=plan.prefix_gate.params.fields.copy();assert len(fields)==12,len(fields)
        wp=L.tensor_map(plan.leaves[6],[64,64,8],dims=[64,256,8],strides_bytes=[1024,128],swizzle='128B',l2='256B')
        dn=L.tensor_map(p.tensors[9],[64,64,4],dims=[64,p.M,8],strides_bytes=[1024,128],swizzle='128B',l2='128B')
        self.params=L.Struct([*fields[:6],wp,dn,*fields[6:]]);self.grid=p.M//64
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,384,self.smem)))

    def __call__(self):self.k.launch((self.grid,1,1),(384,1,1),[self.params],self.smem)
    def launch(self,*args,**kwargs):self()
