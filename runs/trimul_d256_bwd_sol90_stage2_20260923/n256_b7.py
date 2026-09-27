from pathlib import Path
from b7 import B7
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
R=Path(__file__).resolve().parent;PRE=R.parent/'trimul_d256_bwd_sol90_20260923'
class N256B7(B7):
 def __init__(self,p,leaves):
  super().__init__(p,8 if p.n==384 else 16)
  body=(PRE/'source.cu').read_text().replace('#include "mma.cuh"',(PRE/'mma.cuh').read_text()+'\n'+(R/'mma256.cuh').read_text()).replace('mw_d256_b7_source','mw_d256_b7_n256')
  body=body.replace('float dw0[64]={},dw1[64]={};','float dw[128]={};').replace('fence_regs(dw0);fence_regs(dw1);','fence_regs(dw);')
  body=body.replace('mma128<0,1>(dw0,','mma256<0,1>(dw,').replace('   mma128<0,1>(dw1,a,smem_desc(smem_u32(xn+16384+k*2048),8192,1024,1),it>0||k>0);','')
  body=body.replace('for(int j=0;j<64;++j)','for(int j=0;j<128;++j)').replace('p.part[ix]=dw0[j];p.part[ix+128]=dw1[j];','p.part[ix]=dw[j];')
  flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWEIGHT_SPLITS={self.splits}']
  out=T.compile_text(body,flags);self.source=T.load_unit(str(out),'mw_d256_b7_n256').kernel('mw_d256_b7_n256');self.source.set_max_dynamic_smem(114816)
