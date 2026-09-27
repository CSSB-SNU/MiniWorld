"""Reuse retired GEMM storage for fewer, larger input-LN TMA transfers."""
from d256_full_prefix_dx_ln import FullPrefixDxLN
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

class FullRowsPrefixDxLN(FullPrefixDxLN):
    def __init__(self,plan,emit_dxn=False,slots=4,k_tile=64,depth=0,ln_rows=64):
        super().__init__(plan,emit_dxn,slots,k_tile,depth)
        assert ln_rows in (32,64)
        p=plan.p;body=self.source_text
        gamma=max(65536+ln_rows*2048,768*k_tile*slots+128)
        replacements={
            'sm+98304':f'sm+{gamma}',
            'sm+65536+WG*16384':f'sm+65536+WG*{ln_rows*1024}',
            's+(c/64)*2048+':f's+(c/64)*{ln_rows*128}+',
            'half<4;':f'half<{64//ln_rows};',
            'half*16':f'half*{ln_rows}',
            '2*NSLOT+WG,16384)':f'2*NSLOT+WG,{ln_rows*1024})',
            'scratch+8192':f'scratch+{ln_rows*512}',
            'r<16;':f'r<{ln_rows};',
        }
        for old,new in replacements.items():
            assert old in body,old
            body=body.replace(old,new)
        body=body.replace('mw_d256_full_prefix_dx_ln','mw_d256_full_rows_prefix_dx_ln');self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DDX_FULL_SLOTS={slots}',f'-DDX_FULL_K={k_tile}',f'-DDX_FULL_DEPTH={depth}',f'-DEMIT_DXN={int(emit_dxn)}']
        self.cubin=T.compile_text(body,flags);unit=T.load_unit(str(self.cubin),'mw_d256_full_rows_prefix_dx_ln')
        self.k=unit.kernel('mw_d256_full_rows_prefix_dx_ln');self.smem=max(self.smem,gamma+1024);self.k.set_max_dynamic_smem(self.smem)
        self.first=unit.kernel('mw_prefix_input_affine_first');self.finish=unit.kernel('mw_prefix_input_weight_finish')
        L=T._launch_module();fields=self.params.fields.copy()
        for index,t in ((3,p.x),(4,p.dy)):
            fields[index]=L.tensor_map(t,[64,ln_rows,4],dims=[64,p.M,4],strides_bytes=[512,128],swizzle='128B',l2='128B')
        self.params=L.Struct(fields)
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        assert self.registers*384>=128*(32+224*2)
