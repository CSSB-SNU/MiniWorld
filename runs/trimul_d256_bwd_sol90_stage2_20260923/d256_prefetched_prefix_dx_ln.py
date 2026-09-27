"""Prefetch input-LN tiles into retired GEMM slots, with two LN buffers."""
from d256_fast_reduce_prefix_dx_ln import FastReducePrefixDxLN
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


class PrefetchedPrefixDxLN(FastReducePrefixDxLN):
    def __init__(self,plan,emit_dxn=False,ln_rows=16,early=True):
        super().__init__(plan,emit_dxn,4,64,0)
        assert ln_rows in (16,32)
        p=plan.p;body=self.source_text;lnbytes=ln_rows*1024
        body=body.replace('bars+2*NSLOT+2','bars+2*NSLOT+4').replace('i<2*NSLOT+3;','i<2*NSLOT+5;')
        body=body.replace('sm+WG*32768','sm+131072+WG*32768')
        body=body.replace('sm+98304','sm+BAR+1280')
        body=body.replace('uint8_t* scratch=sm+65536+WG*16384;','uint8_t* scratch=nullptr;')
        body=body.replace('s+(c/64)*2048+',f's+(c/64)*{ln_rows*128}+')
        body=body.replace('half<4;',f'half<{64//ln_rows};').replace('half*16',f'half*{ln_rows}').replace('r<16;',f'r<{ln_rows};')
        body=body.replace('scratch+8192',f'scratch+{ln_rows*512}')
        helper=f'''
template<int WG> TMN_DEVI void prefetch_ln(const Params& p,uint8_t* sm,uint64_t* bars,int half){{
 int slot=half%2,row=blockIdx.x*128+WG*64+half*{ln_rows};
 uint8_t* scratch=sm+(WG*2+slot)*{lnbytes};uint64_t* ready=bars+2*NSLOT+WG*2+slot;
 mbar_arrive_expect_tx(ready,{lnbytes});
 tma_load_3d(scratch,&p.x,ready,0,row,0);
 tma_load_3d(scratch+{ln_rows*512},&p.res,ready,0,row,0);
}}
'''
        marker='template<int WG> TMN_DEVI void consume('
        assert body.count(marker)==1;body=body.replace(marker,helper+'\n'+marker)
        begin=body.index('   if(tid==0){\n    mbar_arrive_expect_tx(bars+2*NSLOT+WG,16384);')
        end=body.index('   for(int r=warp;',begin)
        body=body[:begin]+f'''   int lnslot=half%2;scratch=sm+(WG*2+lnslot)*{lnbytes};
   if(tid==0&&half+1<{64//ln_rows})prefetch_ln<WG>(p,sm,bars,half+1);
   mbar_wait(bars+2*NSLOT+WG*2+lnslot,(half/2)&1);named_bar_sync(WG+1,128);
'''+body[end:]
        if early:
            # Final accesses to slots 0/1 retire at step 32/33, respectively.
            # The consumer join protects shared operands of both warpgroups.
            step=32 if ln_rows==16 else 33
            marker='if(tid==0)mbar_arrive(bars+NSLOT+(step-DX_FULL_DEPTH)%NSLOT);}'
            assert body.count(marker)==1
            body=body.replace(marker,marker+f'\n  if(step=={step}){{named_bar_sync(3,256);if(tid==0)prefetch_ln<WG>(p,sm,bars,0);}}')
        else:
            marker='  float gg[8]={},bb[8]={};'
            assert body.count(marker)==1
            body=body.replace(marker,marker+'\n  if(tid==0)prefetch_ln<WG>(p,sm,bars,0);')
        body=body.replace('mw_d256_fast_reduce_prefix_dx_ln','mw_d256_prefetched_prefix_dx_ln');self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),'-DDX_FULL_SLOTS=4','-DDX_FULL_K=64','-DDX_FULL_DEPTH=0',f'-DEMIT_DXN={int(emit_dxn)}']
        self.cubin=T.compile_text(body,flags);unit=T.load_unit(str(self.cubin),'mw_d256_prefetched_prefix_dx_ln')
        self.k=unit.kernel('mw_d256_prefetched_prefix_dx_ln');self.smem=196608+2304;self.k.set_max_dynamic_smem(self.smem)
        self.first=unit.kernel('mw_prefix_input_affine_first');self.finish=unit.kernel('mw_prefix_input_weight_finish')
        L=T._launch_module();fields=self.params.fields.copy()
        for index,t in ((3,p.x),(4,p.dy)):
            fields[index]=L.tensor_map(t,[64,ln_rows,4],dims=[64,p.M,4],strides_bytes=[512,128],swizzle='128B',l2='128B')
        self.params=L.Struct(fields)
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,384,self.smem)))
        assert self.registers*384>=128*(32+224*2)
