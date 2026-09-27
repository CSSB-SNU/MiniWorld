"""Wide dX: a 64-channel by 256-row tile with a single loader warp."""
from pathlib import Path
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


class WideOneGroupTransposedDX:
    def __init__(self,plan,slots=2,depth=1,k_tile=64):
        p=plan.p;d=p.D;q=9*d;channels=d//64;root=Path(__file__).resolve().parent
        assert d in (384,512) and p.M%256==0 and q%k_tile==0
        body=(root/'d256_full_width_prefix_dx.cu').read_text().replace('// MMA_HELPER',(root/'mma256.cuh').read_text())
        body=body.replace('STAGE=768*KD','STAGE=640*KD').replace('STEPS=2304/KD',f'STEPS={q}/KD')
        body=body.replace('if(threadIdx.x)return;','if(threadIdx.x!=128)return;')
        body=body.replace('&p.input,bars+slot,0,step*KD,blockIdx.x*2);',f'&p.input,bars+slot,0,step*KD,blockIdx.x%{channels});')
        body=body.replace('&p.weight,bars+slot,0,step*KD,0);',f'&p.weight,bars+slot,0,step*KD,(blockIdx.x/{channels})*4);')
        body=body.replace('256*KD','128*KD').replace('named_bar_sync(3,256);','')
        marker='*reinterpret_cast<uint32_t*>(out+(c/64)*8192+swz128(r,(c%64)*2))=pack_bf16(acc[j],acc[j+1]);'
        assert body.count(marker)==1
        body=body.replace(marker,'*reinterpret_cast<bf*>(out+swz128(c,r*2))=__float2bfloat16_rn(acc[j]);\n  *reinterpret_cast<bf*>(out+swz128(c+1,r*2))=__float2bfloat16_rn(acc[j+1]);')
        begin=body.index('  int row=blockIdx.x*128+WG*64;');end=body.index('  tma_store_commit();',begin)
        body=body[:begin]+f'''  int row=(blockIdx.x/{channels})*256,col=(blockIdx.x%{channels})*64;
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{{%2,%3}}],[%1];"::"l"(&p.output),"r"(smem_u32(out)),"r"(col),"r"(row):"memory");
'''+body[end:]
        body=body.replace('mbar_init(bars+i,i<NSLOT?1:2)','mbar_init(bars+i,1)')
        marker=' if(tid<128){setmaxnreg_dec<32>();produce(p,sm,bars);}\n else{setmaxnreg_inc<224>();if(tid<256)consume<0>(p,sm,bars);else consume<1>(p,sm,bars);}'
        assert body.count(marker)==1
        body=body.replace(marker,' if(tid>=128)produce(p,sm,bars);else consume<0>(p,sm,bars);')
        resident=2 if 640*k_tile*slots<115000 else 1
        body=body.replace('__launch_bounds__(384,1)',f'__launch_bounds__(160,{resident})')
        body=body.replace('mw_d256_full_width_prefix_dx','mw_wide_one_group_transposed_dx');self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DDX_FULL_SLOTS={slots}',f'-DDX_FULL_K={k_tile}',f'-DDX_FULL_DEPTH={depth}']
        self.cubin=T.compile_text(body,flags);self.k=T.load_unit(str(self.cubin),'mw_wide_one_group_transposed_dx').kernel('mw_wide_one_group_transposed_dx')
        self.smem=640*k_tile*slots+128;self.grid=p.M//256*channels;self.k.set_max_dynamic_smem(self.smem)
        L=T._launch_module()
        a=L.tensor_map(plan.dx.weights,[64,k_tile,1],dims=[64,q,channels],strides_bytes=[2*d,128],swizzle='128B',l2='256B')
        b=L.tensor_map(plan.dx.input,[64,k_tile,4],dims=[64,q,p.M//64],strides_bytes=[p.M*2,128],swizzle='128B',l2='256B')
        y=L.tensor_map(p.tensors[10],[64,256],dims=[d,p.M],strides_bytes=[2*d],swizzle='128B',l2='128B')
        self.params=L.Struct([a,b,y]);drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,160,self.smem)))

    def __call__(self):self.k.launch((self.grid,1,1),(160,1,1),[self.params],self.smem)
