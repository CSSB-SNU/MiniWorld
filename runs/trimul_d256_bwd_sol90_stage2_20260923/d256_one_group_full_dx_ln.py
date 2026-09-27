"""One 64x256 dX compute group retains its full row for exact input LN."""
from pathlib import Path
from d256_full_prefix_dx_ln import FullPrefixDxLN
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


class OneGroupFullDxLN(FullPrefixDxLN):
    def __init__(self,plan,emit_dxn=False,slots=2,k_tile=64,depth=1):
        super().__init__(plan,emit_dxn,4,64,depth)
        p=plan.p;body=self.source_text
        body=body.replace('STAGE=768*KD','STAGE=640*KD')
        body=body.replace('if(threadIdx.x)return;','if(threadIdx.x!=128)return;')
        body=body.replace('step*KD,blockIdx.x*2','step*KD,blockIdx.x')
        body=body.replace('256*KD','128*KD')
        body=body.replace('blockIdx.x*128+WG*64','blockIdx.x*64')
        body=body.replace('(blockIdx.x*2+WG)*512','blockIdx.x*512')
        body=body.replace('sm+65536+WG*16384','sm+32768')
        body=body.replace('sm+98304','sm+49152')
        marker='int ct=WG*128+tid;reinterpret_cast<float*>(sm+49152)[ct]=p.gamma[ct];'
        assert body.count(marker)==1
        body=body.replace(marker,'for(int ct=tid;ct<256;ct+=128)reinterpret_cast<float*>(sm+49152)[ct]=p.gamma[ct];')
        body=body.replace('named_bar_sync(3,256);','named_bar_sync(1,128);')
        body=body.replace('mbar_init(bars+i,(i<NSLOT||i>=2*NSLOT)?1:2)','mbar_init(bars+i,1)')
        marker=' if(tid<128){setmaxnreg_dec<32>();produce(p,sm,bars);}\n else{setmaxnreg_inc<224>();if(tid<256)consume<0>(p,sm,bars);else consume<1>(p,sm,bars);}'
        assert body.count(marker)==1
        body=body.replace(marker,' if(tid>=128)produce(p,sm,bars);else consume<0>(p,sm,bars);')
        body=body.replace('__launch_bounds__(384,1)','__launch_bounds__(160,2)')
        body=body.replace('mw_d256_full_prefix_dx_ln','mw_d256_one_group_full_dx_ln')
        self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DDX_FULL_SLOTS={slots}',f'-DDX_FULL_K={k_tile}',f'-DDX_FULL_DEPTH={depth}',f'-DEMIT_DXN={int(emit_dxn)}']
        self.cubin=T.compile_text(body,flags);unit=T.load_unit(str(self.cubin),'mw_d256_one_group_full_dx_ln')
        self.k=unit.kernel('mw_d256_one_group_full_dx_ln')
        self.first=unit.kernel('mw_prefix_input_affine_first');self.finish=unit.kernel('mw_prefix_input_weight_finish')
        self.smem=640*k_tile*slots+128;assert self.smem>=50176+128
        self.grid=p.M//64;self.k.set_max_dynamic_smem(self.smem)
        L=T._launch_module();fields=self.params.fields.copy()
        fields[0]=L.tensor_map(plan.dx.input,[64,k_tile,1],dims=[64,2304,p.M//64],strides_bytes=[p.M*2,128],swizzle='128B',l2='256B')
        fields[1]=L.tensor_map(plan.dx.weights,[64,k_tile,4],dims=[64,2304,4],strides_bytes=[512,128],swizzle='128B',l2='256B')
        self.params=L.Struct(fields)
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,160,self.smem)))

    def __call__(self):
        self.k.launch((self.grid,1,1),(160,1,1),[self.params],self.smem)
        self.first.launch((self.chunks,1,1),(256,1,1),[self.rp],0)
        self.finish.launch((64,1,1),(128,1,1),[self.rp],0)
