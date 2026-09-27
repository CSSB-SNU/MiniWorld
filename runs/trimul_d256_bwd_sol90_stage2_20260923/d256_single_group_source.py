"""Selected N256 accumulation with input prefetch issued by its compute group."""
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


class SingleGroupSource:
    def __init__(self,plan,cap=224,whole=False):
        prior=plan.pool_source;self.__dict__.update(prior.__dict__)
        body=prior.source_text
        marker='if(tid==0)mbar_arrive(bar+3+slot);';assert body.count(marker)==1
        body=body.replace(marker,'if(tid==0 && tile+2<end)load_input(p,sm,bar,(tile+2)*64,rank,slot);')
        start=body.index(' if(threadIdx.x<128){',body.index('extern "C" __global__'))
        end=body.rfind('\n}')
        body=body[:start]+''' if(threadIdx.x==0){
  int rank=blockIdx.x%32,split=blockIdx.x/32,tiles=p.M/64,begin=tiles*split/WEIGHT_SPLITS,end=tiles*(split+1)/WEIGHT_SPLITS;
  mbar_arrive_expect_tx(bar+2,CH);
  for(int c=0;c<4;++c)tma_load_2d(sm+WEIGHT+c*8192,&p.w,bar+2,c*64,rank*64);
  load_input(p,sm,bar,begin*64,rank,0);
  if(begin+1<end)load_input(p,sm,bar,(begin+1)*64,rank,1);
 }
 compute_source(p);
''' +body[end:]
        body=body.replace('__launch_bounds__(256,1)',f'__maxnreg__({cap})')
        if whole:
            marker='for(int c=0;c<4;++c)tma_load_2d(sm+slot*INPUT+c*8192,&p.xn,bar+slot,c*64,row);';assert body.count(marker)==1
            body=body.replace(marker,'tma_load_3d(sm+slot*INPUT,&p.xn,bar+slot,0,row,0);')
            marker='for(int c=0;c<4;++c)tma_load_2d(sm+WEIGHT+c*8192,&p.w,bar+2,c*64,rank*64);';assert body.count(marker)==1
            body=body.replace(marker,'tma_load_3d(sm+WEIGHT,&p.w,bar+2,0,rank*64,0);')
            L=T._launch_module();p=plan.p;fields=prior.params.fields.copy()
            tm=lambda x,rows:L.tensor_map(x,[64,64,4],dims=[64,rows,4],strides_bytes=[512,128],swizzle='128B',l2='128B')
            fields[:2]=[tm(p.xn,p.M),tm(p.w1,2048)];self.params=L.Struct(fields)
        body=body.replace('mw_d256_register_budget_source','mw_d256_single_group_source');self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWEIGHT_SPLITS={self.splits}']
        self.cubin=T.compile_text(body,flags);self.k=T.load_unit(str(self.cubin),'mw_d256_single_group_source').kernel('mw_d256_single_group_source');self.k.set_max_dynamic_smem(self.smem)
        self.pool_metadata=dict(dynamic_roles=False,metadata_patched=False)
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda name:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,name),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,128,self.smem)))

    def __call__(self):self.k.launch((32*self.splits,1,1),(128,1,1),[self.params],self.smem)
