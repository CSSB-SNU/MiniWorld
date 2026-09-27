"""Recompute only one 32-feature preactivation plane and load the other."""
from pathlib import Path
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from wide_mask_transform import mask_stage

class SavedHalfSource:
    def __init__(self,plan,pre,half=1):
        assert half in (0,1)
        p=plan.p;prior=plan.b7;self.__dict__.update(prior.__dict__)
        root=Path(__file__).resolve().parent
        body=mask_stage(prior.original.source_text,'bulk').replace('sm+BAR+128+slot*128','sm+106624+slot*128')
        body=body.replace('INPUT=36864,WEIGHT=73728,DERIV=106496','INPUT=40960,WEIGHT=81920,DERIV=98304').replace('sm+114688','sm+106496')
        old='CUtensorMap xn,w,dl,dr,gmap[4];';assert body.count(old)==1
        body=body.replace(old,'CUtensorMap xn,w,dl,dr,gmap[4],premap;')
        old=' tma_load_2d(sm+slot*INPUT+CH,rank<16?&p.dl:&p.dr,bar+slot,row,(rank%16)*32);'
        assert body.count(old)==1
        body=body.replace(old,old+'\n tma_load_2d(sm+slot*INPUT+CH+4096,&p.premap,bar+slot,row,rank*32);')
        body=body.replace('mbar_arrive_expect_tx(bar+2,CH);','mbar_arrive_expect_tx(bar+2,16384);')
        old='for(int c=0;c<4;++c)tma_load_2d(sm+WEIGHT+c*8192,&p.w,bar+2,c*64,rank*64);'
        assert body.count(old)==1
        body=body.replace(old,f'for(int c=0;c<4;++c)tma_load_2d(sm+WEIGHT+c*4096,&p.w,bar+2,c*64,rank*64+{(1-half)*32});')
        body=body.replace('float pre[32]={};','float pre[16]={};')
        old='mma64_off<(k/4)*8192+(k%4)*32,(k/4)*8192+(k%4)*32,0,0>(pre,'
        assert body.count(old)==1
        body=body.replace(old,'mma32_off<(k/4)*8192+(k%4)*32,(k/4)*4096+(k%4)*32,0,0>(pre,')
        body=body.replace('float (&a)[32]','float (&a)[16]')
        old='for(int q=0;q<2;++q){uint32_t dy[4],dg[4],dp[4];'
        assert body.count(old)==1
        body=body.replace(old,old+'uint32_t saved[4];ldsm_x4_t(saved,smem_u32(s+36864)+swz128(q*16+lane%8+8*(mat>>1),(w*16+8*(mat&1))*2));')
        old='uint32_t gr=pack_bf16(a[q*8+j*2],a[q*8+j*2+1]),pr=pack_bf16(a[(q+2)*8+j*2],a[(q+2)*8+j*2+1]);'
        assert body.count(old)==1
        new=('uint32_t gr=pack_bf16(a[q*8+j*2],a[q*8+j*2+1]),pr=saved[j];' if half==1 else 'uint32_t gr=saved[j],pr=pack_bf16(a[q*8+j*2],a[q*8+j*2+1]);')
        body=body.replace(old,new)
        marker='TMN_DEVI void compute_source('
        assert body.count(marker)==1
        body=body.replace(marker,(root/'mma32_offset.cuh').read_text()+'\n'+marker)
        body=body.replace('mw_d256_b7_tma','mw_d256_saved_half_source')
        L=T._launch_module();fields=prior.params.fields.copy()
        fields[1]=L.tensor_map(p.w1,[64,32],dims=[256,2048],strides_bytes=[512],swizzle='128B',l2='128B')
        fields.insert(8,L.tensor_map(pre,[64,32],dims=[p.M,1024],strides_bytes=[p.M*2],swizzle='128B',l2='128B'))
        self.params=L.Struct(fields);self.smem=106880
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWEIGHT_SPLITS={self.splits}']
        self.cubin=T.compile_text(body,flags)
        self.k=T.load_unit(str(self.cubin),'mw_d256_saved_half_source').kernel('mw_d256_saved_half_source');self.k.set_max_dynamic_smem(self.smem)
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
    def __call__(self):self.k.launch((32*self.splits,1,1),(256,1,1),[self.params],self.smem)
