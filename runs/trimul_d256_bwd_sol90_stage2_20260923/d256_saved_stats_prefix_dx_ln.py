"""Reuse forward input statistics in the full-width dX/LN epilogue."""
from d256_full_prefix_dx_ln import FullPrefixDxLN
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

class SavedStatsPrefixDxLN(FullPrefixDxLN):
    def __init__(self,plan,emit_dxn=False,slots=4,k_tile=64,depth=0):
        super().__init__(plan,emit_dxn,slots,k_tile,depth)
        body=self.source_text.replace('bf* dxn;','bf* dxn;const float* stats;')
        marker=' if(threadIdx.x)return;'
        assert body.count(marker)==1
        body=body.replace(marker,marker+'''
 mbar_arrive_expect_tx(bars+2*NSLOT+2,1024);
 asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1],1024,[%2];"::"r"(smem_u32(sm+BAR+256)),"l"(p.stats+size_t(blockIdx.x)*256),"r"(smem_u32(bars+2*NSLOT+2)):"memory");''')
        marker='  int ct=WG*128+tid;'
        assert body.count(marker)==1
        body=body.replace(marker,'  mbar_wait(bars+2*NSLOT+2,0);\n'+marker)
        begin=body.index('    float xv[8],s=0;');end=body.index('    #pragma unroll\n    for(int q=0;q<8;++q){int c=lane+q*32;float z=',begin)
        body=body[:begin]+'''    float xv[8];
    #pragma unroll
    for(int q=0;q<8;++q)xv[q]=read16(scratch,r,lane+q*32);
    const float* stats=reinterpret_cast<const float*>(sm+BAR+256)+(WG*64+half*16+r)*2;
    float mu=stats[0],rs=stats[1],s0=0,s1=0;
'''+body[end:]
        body=body.replace('i<2*NSLOT+2;++i','i<2*NSLOT+3;++i').replace('mw_d256_full_prefix_dx_ln','mw_d256_saved_stats_prefix_dx_ln')
        self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DDX_FULL_SLOTS={slots}',f'-DDX_FULL_K={k_tile}',f'-DDX_FULL_DEPTH={depth}',f'-DEMIT_DXN={int(emit_dxn)}']
        self.cubin=T.compile_text(body,flags);unit=T.load_unit(str(self.cubin),'mw_d256_saved_stats_prefix_dx_ln')
        self.k=unit.kernel('mw_d256_saved_stats_prefix_dx_ln');self.smem=768*k_tile*slots+1280;self.k.set_max_dynamic_smem(self.smem)
        self.first=unit.kernel('mw_prefix_input_affine_first');self.finish=unit.kernel('mw_prefix_input_weight_finish')
        self.params=T._launch_module().Struct([*self.params.fields,plan.p.input_stats])
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        assert self.registers*384>=128*(32+224*2)
