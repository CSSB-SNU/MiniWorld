"""Producer-only experiment: stable spatial compaction with a dense overflow path.

The GP consumer schedule is deliberately not changed or claimed faster here.
"""
from pathlib import Path
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


class CompactMaskMap:
    def __init__(self,plan):
        p=plan.p;self.p=p;self.mask=plan.f.mask;self.capacity=p.M*7//8
        self.rows=torch.empty(p.M,device=p.x.device,dtype=torch.int32)
        blocks=(p.M+255)//256
        self.counts=torch.empty(blocks,device=p.x.device,dtype=torch.int32)
        self.metadata=torch.empty(2,device=p.x.device,dtype=torch.int32)
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo']
        self.cubin=T.compile_text(Path(__file__).with_name('compact_mask_map.cu').read_text(),flags)
        self.unit=T.load_unit(str(self.cubin),'mw_mask_map')
        self.local=self.unit.kernel('mw_mask_local');self.scan=self.unit.kernel('mw_mask_counts');self.finish=self.unit.kernel('mw_mask_global');self.pad=self.unit.kernel('mw_mask_pad')

    def __call__(self,pad=False):
        p=self.p;blocks=(p.M+255)//256
        self.local.launch((blocks,1,1),(256,1,1),[self.mask,self.rows,self.counts,p.M],0)
        self.scan.launch((1,1,1),(1024,1,1),[self.counts,self.metadata,blocks,self.capacity],0)
        self.finish.launch((blocks,1,1),(256,1,1),[self.rows,self.counts,self.metadata,p.M],0)
        if pad:self.pad.launch((528,1,1),(256,1,1),[p.gp_all,self.metadata,p.M,self.capacity,8*p.D],0)


class CompactMaskGP:
    def __init__(self,plan,mapping):
        prior=plan.contract_gp;self.__dict__.update(prior.__dict__);p=plan.p
        body=prior.source_text
        body=body.replace('outmap[4];int N;','outmap[4];int N;const int* rows;')
        start=body.index('  *reinterpret_cast<uint32_t*>(sm+off)=dp;')
        end=body.index('\n });',start)
        body=body[:start]+'''  constexpr int SIDE=(MODE==1||MODE==3),HALF=(MODE>=2);
  int outch=ch+HALF*D,row=(mi+WG*64+r)*p.N+ni+c;
  int lo=p.rows[row],hi=p.rows[row+1];size_t base=size_t(outch)*p.N*p.N;
  if(lo>=0){reinterpret_cast<uint16_t*>(p.gp[2*SIDE])[base+lo]=uint16_t(dp);reinterpret_cast<uint16_t*>(p.gp[2*SIDE+1])[base+lo]=uint16_t(dg);}
  if(hi>=0){reinterpret_cast<uint16_t*>(p.gp[2*SIDE])[base+hi]=uint16_t(dp>>16);reinterpret_cast<uint16_t*>(p.gp[2*SIDE+1])[base+hi]=uint16_t(dg>>16);}
'''+body[end:]
        start=body.index(' fence_proxy_async();__syncthreads();',body.index('template<int MODE> TMN_DEVI void run('))
        end=body.index('\n}\nextern',start)
        body=body[:start]+body[end:]
        body=body.replace('mw_wide_staged_epilogue_gp','mw_wide_compact_mask_gp');self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWIDTH={p.D}']
        self.cubin=T.compile_text(body,flags);self.k=T.load_unit(str(self.cubin),'mw_wide_compact_mask_gp').kernel('mw_wide_compact_mask_gp');self.k.set_max_dynamic_smem(self.smem)
        self.params=T._launch_module().Struct([*prior.params.fields,mapping.rows])
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda name:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,name),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,256,self.smem)))

    def __call__(self):self.k.launch((4*self.p.D*(self.p.n//128)**2,1,1),(256,1,1),[self.params],self.smem)
