"""Stage compact spatial rows in shared memory for aligned 16-byte stores."""
from pathlib import Path
import torch
from wide_compact_mask_gp import CompactMaskMap
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


class CompactPrefixMap(CompactMaskMap):
    def __init__(self,plan):
        super().__init__(plan);p=plan.p
        self.rows=torch.empty(p.M+1,device=p.x.device,dtype=torch.int32)
        body=Path(__file__).with_name('compact_mask_map.cu').read_text()
        body=body.replace('valid?base+__popc(ballot&((1u<<lane)-1)):-1','base+__popc(ballot&((1u<<lane)-1))')
        body=body.replace('(local<0?-1:local+counts[blockIdx.x])','(local+counts[blockIdx.x])')
        marker='if(row<M){int local=rows[row];'
        assert body.count(marker)==1
        body=body.replace(marker,'if(row==0)rows[M]=metadata[1]?metadata[0]:M;\n '+marker)
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo']
        self.cubin=T.compile_text(body,flags);self.unit=T.load_unit(str(self.cubin),'mw_prefix_mask_map')
        self.local=self.unit.kernel('mw_mask_local');self.scan=self.unit.kernel('mw_mask_counts');self.finish=self.unit.kernel('mw_mask_global');self.pad=self.unit.kernel('mw_mask_pad')


class TiledCompactMaskGP:
    def __init__(self,plan,mapping):
        prior=plan.contract_gp;self.__dict__.update(prior.__dict__);p=plan.p
        body=prior.source_text
        body=body.replace('BAR=INPUT*SLOTS','BAR=100352').replace('outmap[4];int N;','outmap[4];int N;const int* rows;')
        start=body.index(' fence_proxy_async();__syncthreads();',body.index('template<int MODE> TMN_DEVI void run('))
        end=body.index('\n}\nextern',start)
        body=body[:start]+''' __syncthreads();
 constexpr int SIDE=(MODE==1||MODE==3),HALF=(MODE>=2);int outch=ch+HALF*D;
 auto staging=reinterpret_cast<uint16_t*>(sm+65536);
 for(int g=0;g<2;++g){
  // Read the dense per-row GP values only after both compute groups finish.
  // The retired mask buffer holds136 values per row to align compact stores.
  for(int i=threadIdx.x;i<128*128;i+=256){
   int r=i/128,c=i%128,row=(mi+r)*p.N+ni,begin=p.rows[row],offset=p.rows[row+c];
   if(p.rows[row+c+1]>offset){
    uint16_t value=*reinterpret_cast<uint16_t*>(sm+g*32768+(r/64*2+c/64)*8192+swz128(r%64,(c%64)*2));
    staging[r*136+(begin&7)+offset-begin]=value;
   }
  }
  __syncthreads();
  for(int i=threadIdx.x;i<128*17;i+=256){
   int r=i/17,v=i%17,row=(mi+r)*p.N+ni,begin=p.rows[row],end=p.rows[row+128];
   int head=begin&7,tail=head+end-begin,q=v*8;
   size_t base=size_t(outch)*p.N*p.N+(begin&~7)+q;
   if(q>=head && q+8<=tail){
    *reinterpret_cast<uint4*>(reinterpret_cast<uint16_t*>(p.gp[2*SIDE+g])+base)=*reinterpret_cast<const uint4*>(staging+r*136+q);
   }else if(q<tail && q+8>head){
    #pragma unroll
    for(int j=0;j<8;++j)if(q+j>=head && q+j<tail)reinterpret_cast<uint16_t*>(p.gp[2*SIDE+g])[base+j]=staging[r*136+q+j];
   }
  }
  __syncthreads();
 }
'''+body[end:]
        body=body.replace('mw_wide_staged_epilogue_gp','mw_wide_tiled_compact_mask_gp');self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWIDTH={p.D}']
        self.cubin=T.compile_text(body,flags);self.k=T.load_unit(str(self.cubin),'mw_wide_tiled_compact_mask_gp').kernel('mw_wide_tiled_compact_mask_gp')
        self.smem=100352+128;self.k.set_max_dynamic_smem(self.smem)
        self.params=T._launch_module().Struct([*prior.params.fields,mapping.rows])
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda name:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,name),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,256,self.smem)))

    def __call__(self):self.k.launch((4*self.p.D*(self.p.n//128)**2,1,1),(256,1,1),[self.params],self.smem)
