"""Ordered wide dX: exact N192/N256 column tiles and whole-map TMA."""
from pathlib import Path
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


class WideNativePrefixDX:
    def __init__(self,plan,slots=4,depth=1):
        p=plan.p;d=p.D;assert d in (384,512) and p.M%128==0
        n=d//2;regs=160 if n==192 else 224
        root=Path(__file__).resolve().parent
        helper=(root/('mma192_offset.cuh' if n==192 else 'mma256.cuh')).read_text()
        body=(root/'d256_full_width_prefix_dx.cu').read_text().replace('// MMA_HELPER',helper)
        body=body.replace('NSLOT','DX_NATIVE_SLOTS')
        body=body.replace('STAGE=768*KD',f'STAGE={256+2*n}*KD').replace('STEPS=2304/KD',f'STEPS={9*d}/KD')
        body=body.replace('blockIdx.x*2','(blockIdx.x/2)*2')
        old='tma_load_3d(sm+slot*STAGE+256*KD,&p.weight,bars+slot,0,step*KD,0);'
        assert body.count(old)==1
        body=body.replace(old,f'tma_load_3d(sm+slot*STAGE+256*KD,&p.weight,bars+slot,0,step*KD,(blockIdx.x%2)*{n//64});')
        body=body.replace('float acc[128]={};',f'float acc[{n//2}]={{}};')
        if n==192:body=body.replace('mma256<1,1>','mma192_off<0,0,1,1>')
        body=body.replace('sm+WG*32768',f'sm+WG*{n*128}')
        body=body.replace('static_for<64>([&](auto jj)',f'static_for<{n//4}>([&](auto jj)')
        body=body.replace('int row=blockIdx.x*128+WG*64;',f'int row=(blockIdx.x/2)*128+WG*64,col=(blockIdx.x%2)*{n//64};')
        old='[%0,{0,%2,0}],[%1];"::"l"(&p.output),"r"(smem_u32(out)),"r"(row)'
        assert body.count(old)==1
        body=body.replace(old,'[%0,{0,%2,%3}],[%1];"::"l"(&p.output),"r"(smem_u32(out)),"r"(row),"r"(col)')
        body=body.replace('setmaxnreg_inc<224>()',f'setmaxnreg_inc<{regs}>()')
        body=body.replace('mw_d256_full_width_prefix_dx','mw_wide_native_prefix_dx')
        self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DDX_FULL_SLOTS={slots}','-DDX_FULL_K=64',f'-DDX_FULL_DEPTH={depth}']
        self.cubin=T.compile_text(body,flags)
        self.k=T.load_unit(str(self.cubin),'mw_wide_native_prefix_dx').kernel('mw_wide_native_prefix_dx')
        self.smem=(256+2*n)*64*slots+((16*slots+16+127)//128)*128
        assert self.smem<=232448
        self.k.set_max_dynamic_smem(self.smem);self.grid=p.M//128*2
        L=T._launch_module()
        a=L.tensor_map(plan.dx.input,[64,64,2],dims=[64,9*d,p.M//64],strides_bytes=[p.M*2,128],swizzle='128B',l2='128B')
        w=L.tensor_map(plan.dx.weights,[64,64,n//64],dims=[64,9*d,d//64],strides_bytes=[d*2,128],swizzle='128B',l2='128B')
        y=L.tensor_map(p.tensors[10],[64,64,n//64],dims=[64,p.M,d//64],strides_bytes=[d*2,128],swizzle='128B',l2='128B')
        self.params=L.Struct([a,w,y])
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,384,self.smem)))
        assert self.registers*384>=128*(32+regs*2),(self.registers,'insufficient dynamic register pool')

    def __call__(self):
        self.k.launch((self.grid,1,1),(384,1,1),[self.params],self.smem)
