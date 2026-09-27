"""Compute output-gate dW alongside source dW using its resident XN schedule."""
from pathlib import Path
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from wide_mask_transform import mask_stage

class JointGateSource:
    def __init__(self,plan):
        original=plan.b7;self.__dict__.update(original.__dict__)
        body=mask_stage(original.original.source_text,'bulk').replace('sm+BAR+128+slot*128','sm+114816+slot*128')
        body=body.replace('xn,w,dl,dr,gmap[4];','xn,w,dl,dr,gmap[4],gatemap;')
        body=body.replace('blockIdx.x%32','blockIdx.x%36').replace('blockIdx.x/32','blockIdx.x/36')
        marker='extern "C" __global__ __launch_bounds__(256,2)';assert body.count(marker)==1
        helper=Path(__file__).with_suffix('.cuh').read_text()
        body=body.replace(marker,helper+'\n'+marker)
        marker='   mbar_arrive_expect_tx(bar+2,CH);';assert body.count(marker)==1
        body=body.replace(marker,'   if(rank<32){\n'+marker)
        marker='   for(int tile=begin,it=0;tile<end;++tile,++it){int slot=it%2;if(it>=2)mbar_wait(bar+3+slot,((it/2)-1)&1);load_input(p,sm,bar,tile*64,rank,slot);}'
        assert body.count(marker)==1
        body=body.replace(marker,marker+'''
   }else{
    for(int tile=begin,it=0;tile<end;++tile,++it){int slot=it%2;if(it>=2)mbar_wait(bar+3+slot,((it/2)-1)&1);load_gate_input(p,sm,bar,tile*64,rank,slot);}
   }''')
        marker='setmaxnreg_inc<224>();compute_source(p);';assert body.count(marker)==1
        body=body.replace(marker,'setmaxnreg_inc<224>();if(blockIdx.x%36<32)compute_source(p);else compute_gate_dw(p);')
        body=body.replace('mw_d256_b7_tma','mw_d256_joint_gate_source')
        self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWEIGHT_SPLITS={self.splits}']
        self.cubin=T.compile_text(body,flags);self.k=T.load_unit(str(self.cubin),'mw_d256_joint_gate_source').kernel('mw_d256_joint_gate_source')
        self.smem=115072;self.k.set_max_dynamic_smem(self.smem)
        L=T._launch_module();p=plan.p
        gate=L.tensor_map(plan.dx.input[:p.D],[64,64],dims=[p.M,p.D],strides_bytes=[p.M*2],swizzle='128B',l2='128B')
        fields=original.params.fields.copy();fields.insert(8,gate);self.params=L.Struct(fields)
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
    def __call__(self):self.k.launch((36*self.splits,1,1),(256,1,1),[self.params],self.smem)
