"""Two row consumers and one loader warp fit two N128 CTAs per SM."""
from pathlib import Path
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

class TwoCtaPrefixDX:
    def __init__(self,plan,slots=3,depth=1):
        p=plan.p;root=Path(__file__).resolve().parent;k_tile=64
        assert p.D==256 and slots in (2,3)
        body=(root/'d256_full_width_prefix_dx.cu').read_text().replace('// MMA_HELPER',(root/'mma_offset.cuh').read_text())
        body=body.replace('STAGE=768*KD','STAGE=512*KD')
        body=body.replace('if(threadIdx.x)return;','if(threadIdx.x!=256)return;')
        body=body.replace('blockIdx.x*2);','(blockIdx.x/2)*2);')
        marker='&p.weight,bars+slot,0,step*KD,0);'
        assert body.count(marker)==1;body=body.replace(marker,'&p.weight,bars+slot,0,step*KD,(blockIdx.x%2)*2);')
        body=body.replace('float acc[128]={};','float acc[64]={};')
        marker='mma256<1,1>(acc,smem_desc(smem_u32(buf+WG*128*KD+q*2048),16,1024,1),smem_desc(smem_u32(buf+256*KD+q*2048),128*KD,1024,1),step>0||q>0);'
        assert body.count(marker)==1
        body=body.replace(marker,'mma128_off<q*2048,q*2048,1,1>(acc,smem_desc(smem_u32(buf+WG*128*KD),16,1024,1),smem_desc(smem_u32(buf+256*KD),128*KD,1024,1),step>0||q>0);')
        body=body.replace('sm+WG*32768','sm+WG*16384').replace('static_for<64>([&](auto jj)','static_for<32>([&](auto jj)')
        body=body.replace('int row=blockIdx.x*128+WG*64;','int row=(blockIdx.x/2)*128+WG*64,col=(blockIdx.x%2)*2;')
        body=body.replace('[%0,{0,%2,0}]','[%0,{0,%2,%3}]').replace('"r"(row):"memory"','"r"(row),"r"(col):"memory"')
        body=body.replace('__launch_bounds__(384,1)','__launch_bounds__(288,2)')
        marker=' if(tid<128){setmaxnreg_dec<32>();produce(p,sm,bars);}\n else{setmaxnreg_inc<224>();if(tid<256)consume<0>(p,sm,bars);else consume<1>(p,sm,bars);}'
        assert body.count(marker)==1
        body=body.replace(marker,' if(tid>=256)produce(p,sm,bars);else if(tid<128)consume<0>(p,sm,bars);else consume<1>(p,sm,bars);')
        body=body.replace('mw_d256_full_width_prefix_dx','mw_d256_two_cta_prefix_dx');self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DDX_FULL_SLOTS={slots}','-DDX_FULL_K=64',f'-DDX_FULL_DEPTH={depth}']
        self.cubin=T.compile_text(body,flags);self.k=T.load_unit(str(self.cubin),'mw_d256_two_cta_prefix_dx').kernel('mw_d256_two_cta_prefix_dx')
        self.smem=512*k_tile*slots+128;self.grid=p.M//128*2;self.k.set_max_dynamic_smem(self.smem)
        L=T._launch_module()
        a=L.tensor_map(plan.dx.input,[64,k_tile,2],dims=[64,2304,p.M//64],strides_bytes=[p.M*2,128],swizzle='128B',l2='256B')
        w=L.tensor_map(plan.dx.weights,[64,k_tile,2],dims=[64,2304,4],strides_bytes=[512,128],swizzle='128B',l2='256B')
        y=L.tensor_map(p.tensors[10],[64,64,2],dims=[64,p.M,4],strides_bytes=[512,128],swizzle='128B',l2='128B')
        self.params=L.Struct([a,w,y])
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,288,self.smem)))
    def __call__(self):self.k.launch((self.grid,1,1),(288,1,1),[self.params],self.smem)
