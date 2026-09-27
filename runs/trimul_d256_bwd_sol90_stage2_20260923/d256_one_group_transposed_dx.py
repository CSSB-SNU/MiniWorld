"""One N256 compute warpgroup and a single loader warp; two CTA residency."""
from d256_transposed_prefix_dx import TransposedPrefixDX
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


class OneGroupTransposedDX(TransposedPrefixDX):
    def __init__(self,plan,slots=2,depth=1,k_tile=64):
        super().__init__(plan,slots,depth,k_tile)
        p=plan.p;body=self.source_text
        body=body.replace('STAGE=768*KD','STAGE=640*KD')
        body=body.replace('if(threadIdx.x)return;','if(threadIdx.x!=128)return;')
        body=body.replace('(blockIdx.x%2)*2','blockIdx.x%4').replace('(blockIdx.x/2)*4','(blockIdx.x/4)*4')
        body=body.replace('256*KD','128*KD')
        body=body.replace('named_bar_sync(3,256);','')
        body=body.replace('(blockIdx.x/2)*256,col=(blockIdx.x%2)*128+WG*64','(blockIdx.x/4)*256,col=(blockIdx.x%4)*64')
        body=body.replace('mbar_init(bars+i,i<NSLOT?1:2)','mbar_init(bars+i,1)')
        marker=' if(tid<128){setmaxnreg_dec<32>();produce(p,sm,bars);}\n else{setmaxnreg_inc<224>();if(tid<256)consume<0>(p,sm,bars);else consume<1>(p,sm,bars);}'
        assert body.count(marker)==1
        body=body.replace(marker,' if(tid>=128)produce(p,sm,bars);else consume<0>(p,sm,bars);')
        resident=2 if 640*k_tile*slots<115000 else 1
        body=body.replace('__launch_bounds__(384,1)',f'__launch_bounds__(160,{resident})')
        body=body.replace('mw_d256_transposed_prefix_dx','mw_d256_one_group_transposed_dx');self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DDX_FULL_SLOTS={slots}',f'-DDX_FULL_K={k_tile}',f'-DDX_FULL_DEPTH={depth}']
        self.cubin=T.compile_text(body,flags);self.k=T.load_unit(str(self.cubin),'mw_d256_one_group_transposed_dx').kernel('mw_d256_one_group_transposed_dx')
        self.smem=640*k_tile*slots+128;self.grid=p.M//256*4;self.k.set_max_dynamic_smem(self.smem)
        L=T._launch_module()
        a=L.tensor_map(plan.dx.weights,[64,k_tile,1],dims=[64,2304,4],strides_bytes=[512,128],swizzle='128B',l2='256B')
        b=L.tensor_map(plan.dx.input,[64,k_tile,4],dims=[64,2304,p.M//64],strides_bytes=[p.M*2,128],swizzle='128B',l2='256B')
        y=L.tensor_map(p.tensors[10],[64,256],dims=[256,p.M],strides_bytes=[512],swizzle='128B',l2='128B')
        self.params=L.Struct([a,b,y]);drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,160,self.smem)))

    def __call__(self):self.k.launch((self.grid,1,1),(160,1,1),[self.params],self.smem)
