"""Read four adjacent feature slabs in a single TMA command."""
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from wide_mask_transform import mask_stage

class WholeTmaSource:
    def __init__(self,plan):
        p=plan.p;prior=plan.b7;self.__dict__.update(prior.__dict__)
        body=mask_stage(prior.original.source_text,'bulk').replace('sm+BAR+128+slot*128','sm+114816+slot*128')
        old='for(int c=0;c<4;++c)tma_load_2d(sm+slot*INPUT+c*8192,&p.xn,bar+slot,c*64,row);'
        assert body.count(old)==1
        body=body.replace(old,'tma_load_3d(sm+slot*INPUT,&p.xn,bar+slot,0,row,0);')
        old='for(int c=0;c<4;++c)tma_load_2d(sm+WEIGHT+c*8192,&p.w,bar+2,c*64,rank*64);'
        assert body.count(old)==1
        body=body.replace(old,'tma_load_3d(sm+WEIGHT,&p.w,bar+2,0,rank*64,0);')
        body=body.replace('mw_d256_b7_tma','mw_d256_whole_tma_source')
        L=T._launch_module();fields=prior.params.fields.copy()
        fields[0]=L.tensor_map(p.xn,[64,64,4],dims=[64,p.M,4],strides_bytes=[512,128],swizzle='128B',l2='128B')
        fields[1]=L.tensor_map(p.w1,[64,64,4],dims=[64,2048,4],strides_bytes=[512,128],swizzle='128B',l2='128B')
        self.params=L.Struct(fields)
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWEIGHT_SPLITS={self.splits}']
        self.cubin=T.compile_text(body,flags)
        self.source=T.load_unit(str(self.cubin),'mw_d256_whole_tma_source').kernel('mw_d256_whole_tma_source');self.source.set_max_dynamic_smem(self.smem)
        drv=self.source.unit.drv;fn=drv.d.CUfunction(int(self.source.handle))
        query=lambda n:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,n),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
    def __call__(self):self.source.launch((32*self.splits,1,1),(self.source_threads,1,1),[self.params],self.smem)
