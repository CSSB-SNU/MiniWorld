"""Share dW accumulator lifetimes across regular and output-gate source ranks."""
from d256_joint_gate_source import JointGateSource
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

class JointGateSourceV2(JointGateSource):
    def __init__(self,plan):
        super().__init__(plan);body=self.source_text
        begin=body.index('TMN_DEVI void compute_gate_dw(');end=body.index('extern "C" __global__ __launch_bounds__(256,2)',begin)
        body=body[:begin]+body[end:]
        body=body.replace('if(blockIdx.x%36<32)compute_source(p);else compute_gate_dw(p);','compute_source(p);')
        marker=' mbar_wait(bar+2,0);named_bar_sync(1,128);';assert body.count(marker)==1
        body=body.replace(marker,' if(rank<32)mbar_wait(bar+2,0);named_bar_sync(1,128);')
        marker='uint8_t* xn=sm+slot*INPUT;';assert body.count(marker)==1
        body=body.replace(marker,'uint8_t* xn=sm+slot*(rank<32?INPUT:40960);')
        marker='  float pre[32]={};';assert body.count(marker)==1
        body=body.replace(marker,'  if(rank<32){\n'+marker)
        marker='  fence_regs(dw0);fence_regs(dw1);wgmma_fence();';assert body.count(marker)==1
        body=body.replace(marker,'  }\n'+marker)
        marker='smem_desc(smem_u32(sm+DERIV),16,1024,1)';assert body.count(marker)==1
        body=body.replace(marker,'smem_desc(smem_u32(rank<32?sm+DERIV:xn+32768),16,1024,1)')
        marker='size_t ix=size_t(split)*11*D*D+(3+2*which)*D*D+outrow*D+c;';assert body.count(marker)==1
        body=body.replace(marker,'size_t ix=size_t(split)*11*D*D+(rank<32?(3+2*which)*D*D+outrow*D+c:2*D*D+((rank-32)*64+r)*D+c);')
        body=body.replace('mw_d256_joint_gate_source','mw_d256_joint_gate_source_v2')
        self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWEIGHT_SPLITS={self.splits}']
        self.cubin=T.compile_text(body,flags);self.k=T.load_unit(str(self.cubin),'mw_d256_joint_gate_source_v2').kernel('mw_d256_joint_gate_source_v2');self.k.set_max_dynamic_smem(self.smem)
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
