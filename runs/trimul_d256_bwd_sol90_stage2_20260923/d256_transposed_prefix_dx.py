"""Transpose GEMM output axes: 128 channels by 256 spatial rows per CTA."""
from pathlib import Path
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

class TransposedPrefixDX:
    def __init__(self,plan,slots=4,depth=0,k_tile=64):
        p=plan.p;root=Path(__file__).resolve().parent
        assert p.D==256 and p.M%256==0
        body=(root/'d256_full_width_prefix_dx.cu').read_text().replace('// MMA_HELPER',(root/'mma256.cuh').read_text())
        body=body.replace('&p.input,bars+slot,0,step*KD,blockIdx.x*2);','&p.input,bars+slot,0,step*KD,(blockIdx.x%2)*2);')
        body=body.replace('&p.weight,bars+slot,0,step*KD,0);','&p.weight,bars+slot,0,step*KD,(blockIdx.x/2)*4);')
        marker='*reinterpret_cast<uint32_t*>(out+(c/64)*8192+swz128(r,(c%64)*2))=pack_bf16(acc[j],acc[j+1]);'
        assert body.count(marker)==1
        body=body.replace(marker,'*reinterpret_cast<bf*>(out+swz128(c,r*2))=__float2bfloat16_rn(acc[j]);\n  *reinterpret_cast<bf*>(out+swz128(c+1,r*2))=__float2bfloat16_rn(acc[j+1]);')
        begin=body.index('  int row=blockIdx.x*128+WG*64;');end=body.index('  tma_store_commit();',begin)
        body=body[:begin]+'''  int row=(blockIdx.x/2)*256,col=(blockIdx.x%2)*128+WG*64;
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"::"l"(&p.output),"r"(smem_u32(out)),"r"(col),"r"(row):"memory");
'''+body[end:]
        body=body.replace('mw_d256_full_width_prefix_dx','mw_d256_transposed_prefix_dx');self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DDX_FULL_SLOTS={slots}',f'-DDX_FULL_K={k_tile}',f'-DDX_FULL_DEPTH={depth}']
        self.cubin=T.compile_text(body,flags);self.k=T.load_unit(str(self.cubin),'mw_d256_transposed_prefix_dx').kernel('mw_d256_transposed_prefix_dx')
        self.smem=768*k_tile*slots+((16*slots+16+127)//128)*128;self.grid=p.M//256*2
        self.k.set_max_dynamic_smem(self.smem);L=T._launch_module()
        a=L.tensor_map(plan.dx.weights,[64,k_tile,2],dims=[64,2304,4],strides_bytes=[512,128],swizzle='128B',l2='256B')
        b=L.tensor_map(plan.dx.input,[64,k_tile,4],dims=[64,2304,p.M//64],strides_bytes=[p.M*2,128],swizzle='128B',l2='256B')
        y=L.tensor_map(p.tensors[10],[64,256],dims=[256,p.M],strides_bytes=[512],swizzle='128B',l2='128B')
        self.params=L.Struct([a,b,y]);drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,384,self.smem)))
        assert self.registers*384>=128*(32+224*2)
    def __call__(self):self.k.launch((self.grid,1,1),(384,1,1),[self.params],self.smem)
