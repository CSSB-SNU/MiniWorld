"""Exact prefix dX and input LN keep the intermediate tile in shared memory."""
from pathlib import Path
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from d256_full_width_prefix_dx import FullWidthPrefixDX

class FullPrefixDxLN(FullWidthPrefixDX):
    def __init__(self,plan,emit_dxn=False,slots=4,k_tile=64,depth=1):
        super().__init__(plan,slots,k_tile,depth,'256B')
        p=plan.p;assert p.n==384 and 768*k_tile*slots>=99328
        root=Path(__file__).resolve().parent;body=self.source_text
        body=body.replace('CUtensorMap input,weight,output;','CUtensorMap input,weight,output,x,res;const float* gamma;float* partial;bf* dxn;')
        marker='template<int WG> TMN_DEVI void consume('
        body=body.replace(marker,'TMN_DEVI float sumwarp(float v){for(int q=16;q;q>>=1)v+=__shfl_xor_sync(0xffffffff,v,q);return v;}\n'+marker)
        marker=' fence_proxy_async();named_bar_sync(WG+1,128);'
        assert body.count(marker)==1
        body=body.replace(marker,(root/'d256_prefix_dx_ln_epilogue.cuh').read_text()+'\n'+marker)
        body=body.replace('i<2*NSLOT;++i)mbar_init(bars+i,i<NSLOT?1:2)','i<2*NSLOT+2;++i)mbar_init(bars+i,(i<NSLOT||i>=2*NSLOT)?1:2)')
        body=body.replace('mw_d256_full_width_prefix_dx','mw_d256_full_prefix_dx_ln')
        body+='\n'+(root/'d256_prefix_dx_ln_reduce.cuh').read_text();self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DDX_FULL_SLOTS={slots}',f'-DDX_FULL_K={k_tile}',f'-DDX_FULL_DEPTH={depth}',f'-DEMIT_DXN={int(emit_dxn)}']
        self.cubin=T.compile_text(body,flags);unit=T.load_unit(str(self.cubin),'mw_d256_full_prefix_dx_ln')
        self.k=unit.kernel('mw_d256_full_prefix_dx_ln');self.k.set_max_dynamic_smem(self.smem)
        self.first=unit.kernel('mw_prefix_input_affine_first');self.finish=unit.kernel('mw_prefix_input_weight_finish')
        self.partial=torch.empty((self.grid*2,2,256),device=p.x.device,dtype=torch.float32)
        self.chunks=(self.grid*2+63)//64;self.tmp=torch.empty((self.chunks,2,256),device=p.x.device,dtype=torch.float32)
        L=T._launch_module();fields=self.params.fields
        rowmap=lambda t,rows:L.tensor_map(t,[64,rows,4],dims=[64,p.M,4],strides_bytes=[512,128],swizzle='128B',l2='128B')
        self.params=L.Struct(fields[:2]+[rowmap(p.dx,64),rowmap(p.x,16),rowmap(p.dy,16),p.floats[0],self.partial,p.tensors[10]])
        self.rp=L.Struct([self.partial,p.floats[7],self.tmp,p.floats[8],p.floats[9],*p.tensors[17:21],self.grid*2,self.chunks])
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        assert self.registers*384>=128*(32+224*2)
    def __call__(self):
        super().__call__()
        self.first.launch((self.chunks,1,1),(256,1,1),[self.rp],0)
        self.finish.launch((64,1,1),(128,1,1),[self.rp],0)
