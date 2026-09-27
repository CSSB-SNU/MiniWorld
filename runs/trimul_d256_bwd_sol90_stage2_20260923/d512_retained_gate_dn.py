"""D512 retains full shared DP while two dNorm groups cover H in two passes."""
from pathlib import Path
from three_role_register_pool import initial_pool_three
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_wide_forward as F


class RetainedGateDN:
    def __init__(self,plan,producer=80,consumer=176):
        p=plan.p;assert (p.D,p.n)==(512,384);root=Path(__file__).resolve().parent
        body=(root/'d512_retained_gate_dn.cu').read_text().replace('// MMA256',(root/'mma256.cuh').read_text());self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(F.headers()),'-DMW_MINB=1','-DMW_K1_STREAM=0',f'-DPRODUCER_REGS={producer}',f'-DCONSUMER_REGS={consumer}']
        cubin=T.compile_text(body,flags);initial=((producer+2*consumer+23)//24)*8
        self.cubin,self.pool_metadata=initial_pool_three(cubin,'mw_d512_retained_gate_dn',initial,producer,consumer)
        self.k=T.load_unit(str(self.cubin),'mw_d512_retained_gate_dn').kernel('mw_d512_retained_gate_dn');self.smem=229376+128;self.k.set_max_dynamic_smem(self.smem)
        L=T._launch_module();fields=plan.prefix_gate.params.fields.copy();assert len(fields)==12
        wp=L.tensor_map(plan.b1.wp,[64,64,8],dims=[64,512,16],strides_bytes=[2048,128],swizzle='128B',l2='256B')
        dn=L.tensor_map(p.tensors[9],[64,64,4],dims=[64,p.M,16],strides_bytes=[2048,128],swizzle='128B',l2='128B')
        self.params=L.Struct([*fields[:6],wp,dn,*fields[6:]]);self.grid=p.M//64
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,384,self.smem)))

    def __call__(self):self.k.launch((self.grid,1,1),(384,1,1),[self.params],self.smem)
    def launch(self,*args,**kwargs):self()
