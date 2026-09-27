"""Saved preactivation GP producer and immediate native dW consumers."""
from pathlib import Path
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from wide_offset_source import offsets

class SavedPipe:
    def __init__(self,p,original,pre,splits=32):
        root=Path(__file__).resolve().parent;prior=root.parent/'trimul_d256_bwd_sol90_20260923'
        self.p=p;self.splits=splits
        helpers=(root/'wide_source.cu').read_text().split('extern "C" __global__')[0]
        helpers=helpers.replace('INPUT=CH+4096','INPUT=CH+12288')
        helpers=helpers.replace('WEIGHT=2*INPUT,DERIV=WEIGHT+CH,BAR=DERIV+8192','WEIGHT=2*INPUT,DERIV=WEIGHT,BAR=DERIV+16384')
        helpers=helpers.replace('int M;};','int M;CUtensorMap pre;};')
        marker=' mbar_arrive_expect_tx(bar+slot,INPUT);'
        helpers=helpers.replace(marker,marker+'\n tma_load_2d(sm+slot*INPUT+CH+4096,&p.pre,bar+slot,0,rank*p.M+row);')
        helpers=helpers.replace('// MMA_HELPERS',(prior/'mma.cuh').read_text())
        helpers=helpers.replace('// PACKED_GLU',(root/'packed_glu.cuh').read_text().replace('s+32768','s+CH'))
        body=offsets((root/'wide_pipe_source.cu').read_text().replace('// SOURCE_HELPERS',helpers))
        start=body.index(' if(tid==0){mbar_arrive_expect_tx(bar+2,CH);')
        end=body.index(' for(int tile=begin,it=0;',start)
        body=body[:start]+' if(tid==0)load_input(p,sm,bar,begin*64,rank,0);\n'+body[end:]
        marker='  if(it>=2)mbar_wait(bar+5+slot,((it/2)-1)&1);\n  if(tid==0)load_input(p,sm,bar,row,rank,slot);'
        assert body.count(marker)==1;body=body.replace(marker,'')
        start=body.index('  float pre[32]={};');end=body.index('  packed_glu(',start)
        body=body[:start]+'''  int ra=warp*16+lane/4;
  uint32_t ma=uint32_t(__bfloat16_as_ushort(p.mask[row+ra]))*0x10001u;
  uint32_t mb=uint32_t(__bfloat16_as_ushort(p.mask[row+ra+8]))*0x10001u;
  if(tile+1<end){
   if(it>=1)mbar_wait(bar+5+(1-slot),((it-1)/2)&1);
   if(tid==0)load_input(p,sm,bar,(tile+1)*64,rank,1-slot);
  }
  float pre[32];
  #pragma unroll
  for(int j=0;j<32;++j){int r=warp*16+lane/4+8*((j/2)&1),c=(j/8)*16+2*(lane%4)+8*((j/2)%4/2)+(j%2);
   pre[j]=__bfloat162float(reinterpret_cast<bf*>(xn+CH+4096)[swz128(r,c*2)/2]);
  }
''' +body[end:]
        body=body.replace('mw_wide_pipe_source','mw_wide_saved_pipe')
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWIDTH={p.D}',f'-DWEIGHT_SPLITS={splits}']
        self.cubin=T.compile_text(body,flags)
        self.k=T.load_unit(str(self.cubin),'mw_wide_saved_pipe').kernel('mw_wide_saved_pipe')
        self.smem=2*(128*p.D+12288)+16384+128;self.k.set_max_dynamic_smem(self.smem)
        L=T._launch_module();prem=L.tensor_map(pre,[64,64],dims=[64,pre.numel()//64],strides_bytes=[128],swizzle='128B',l2='128B')
        self.params=L.Struct([*original.params.fields,prem])
    def __call__(self):self.k.launch((self.p.D//8*self.splits,1,1),(384,1,1),[self.params],self.smem)
