"""Row-major ordered dX operands; retain native local input-weight gradients."""
from pathlib import Path
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from wide_mask_transform import mask_stage
from lt_contract import LtBmm

class RowMajorGP:
    def __init__(self,plan):
        self.plan=plan;self.p=p=plan.p;root=Path(__file__).resolve().parent;L=T._launch_module()
        self.input=p.x.new_empty((p.M,9*p.D));self.splits=plan.b7.splits
        body=mask_stage(plan.b7.source_text,'bulk').replace('sm+BAR+128+slot*128','sm+114816+slot*128')
        body=body.replace('void packed_glu(float (&a)[32],uint8_t* s,uint8_t* sg,uint32_t ma,uint32_t mb)',
                          'void packed_glu(float (&a)[32],uint8_t* s,uint8_t* sg,uint32_t ma,uint32_t mb,uint32_t (&saved_dg)[2][4])')
        body=body.replace('int lane=threadIdx.x%32,w=(threadIdx.x/32)%4,mat=lane/8;',
                          'int lane=threadIdx.x%32,w=(threadIdx.x/32)%4,mat=lane/8;uint32_t saved_dp[2][4];')
        needle='  uint32_t addr=swz128(q*16+lane%8+8*(mat>>1),(w*16+8*(mat&1))*2);'
        assert body.count(needle)==1
        body=body.replace(needle,'''  #pragma unroll
  for(int j=0;j<4;++j){saved_dg[q][j]=dg[j];saved_dp[q][j]=dp[j];}
'''+needle)
        end='stsm_x4_t(smem_u32(sg+4096)+addr,dp[0],dp[1],dp[2],dp[3]);\n }\n}'
        assert body.count(end)==1
        body=body.replace(end,end[:-1]+''' named_bar_sync(1,128);
 #pragma unroll
 for(int q=0;q<2;++q){
  uint32_t rm=sw64((w*16+lane%8+8*(mat&1))*64+(q*16+8*(mat>>1))*2);
  stsm_x4(smem_u32(s+CH)+rm,saved_dp[q][0],saved_dp[q][1],saved_dp[q][2],saved_dp[q][3]);
 }
}''')
        body=body.replace('  packed_glu(pre,xn,sm+DERIV,ma,mb);', '  uint32_t saved_dg[2][4];packed_glu(pre,xn,sm+DERIV,ma,mb,saved_dg);')
        old='if(tid==0){store_gp(p.gmap+(rank/16)*2,sm+DERIV+4096,row,(rank%16)*32);store_gp(p.gmap+(rank/16)*2+1,sm+DERIV,row,(rank%16)*32);tma_store_commit();}'
        new='if(tid==0){store_gp(p.gmap+(rank/16)*2,xn+CH,(rank%16)*32,row);tma_store_commit();}'
        assert body.count(old)==1;body=body.replace(old,new)
        old='wgmma_wait<0>();fence_regs(dw0);fence_regs(dw1);if(tid==0)tma_store_wait_all();named_bar_sync(1,128);if(tid==0)mbar_arrive(bar+3+slot);'
        new='''wgmma_wait<0>();fence_regs(dw0);fence_regs(dw1);named_bar_sync(1,128);
  #pragma unroll
  for(int q=0;q<2;++q){int mat=lane/8;
   uint32_t rm=sw64((warp*16+lane%8+8*(mat&1))*64+(q*16+8*(mat>>1))*2);
   stsm_x4(smem_u32(sm+DERIV)+rm,saved_dg[q][0],saved_dg[q][1],saved_dg[q][2],saved_dg[q][3]);
  }
  fence_proxy_async();named_bar_sync(1,128);
  if(tid==0){store_gp(p.gmap+(rank/16)*2+1,sm+DERIV,(rank%16)*32,row);tma_store_commit();tma_store_wait_all();}
  named_bar_sync(1,128);if(tid==0)mbar_arrive(bar+3+slot);'''
        assert body.count(old)==1;body=body.replace(old,new)
        body=body.replace('mw_d256_b7_tma','mw_d256_rowmajor_gp')
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWEIGHT_SPLITS={self.splits}']
        self.cubin=T.compile_text(body,flags)
        self.source=T.load_unit(str(self.cubin),'mw_d256_rowmajor_gp').kernel('mw_d256_rowmajor_gp');self.source.set_max_dynamic_smem(115072)
        maps=[L.tensor_map(self.input[:,p.D+i*2*p.D:p.D+(i+1)*2*p.D],[32,64],dims=[2*p.D,p.M],strides_bytes=[9*p.D*2],swizzle='64B',l2='128B') for i in range(4)]
        fields=plan.b7.params.fields.copy();fields[4:8]=maps;self.params=L.Struct(fields)
        cp='''
#include "tmn_kernels.cuh"
struct P{const uint32_t* src;uint32_t* dst;int M;};
extern "C" __global__ __launch_bounds__(256,4)
void mw_rowmajor_prefix(__grid_constant__ const P p){
 for(int i=blockIdx.x*256+threadIdx.x;i<p.M*128;i+=gridDim.x*256)p.dst[(i/128)*1152+i%128]=p.src[i];
}
'''
        self.copy_cubin=T.compile_text(cp,flags);self.copy=T.load_unit(str(self.copy_cubin),'mw_rowmajor_prefix').kernel('mw_rowmajor_prefix')
        self.cp=L.Struct([p.dg,self.input,p.M])
        self.workspace=torch.empty(64*1024*1024,device=p.x.device,dtype=torch.uint8)
        self.op=LtBmm(self.input.unsqueeze(0),plan.dx.weights.unsqueeze(0),p.tensors[10].unsqueeze(0),self.workspace)

    def source_only(self):self.source.launch((32*self.splits,1,1),(256,1,1),[self.params],115072)
    def prepare(self):self.source_only();self.copy.launch((1056,1,1),(256,1,1),[self.cp],0)
    def __call__(self):
        self.plan.dx.pack_weights();self.prepare();self.op();self.plan.dx.reduce_only()
