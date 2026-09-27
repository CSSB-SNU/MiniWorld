"""Exact byte mask with escape loads, retaining two GP CTAs in 80KB each."""
import torch
from wide_two_group_contract_gp import TwoGroupContractGP
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


class ByteMaskGP(TwoGroupContractGP):
    def __init__(self,plan,banded=False):
        super().__init__(plan)
        self.banded=banded;p=plan.p;assert p.n==384
        self.mask=plan.f.mask;self.codes=torch.empty_like(self.mask,dtype=torch.uint8)
        body=self.source_text
        body=body.replace('SLOTS=3,BAR=INPUT*SLOTS','SLOTS=2,BAR=81920')
        body=body.replace('load_input<MODE>(p,sm,bar,ch,mi,ni,0,0);load_input<MODE>(p,sm,bar,ch,mi,ni,64,1);','load_input<MODE>(p,sm,bar,ch,mi,ni,0,0);')
        body=body.replace('ki+128<p.N)load_input<MODE>(p,sm,bar,ch,mi,ni,ki+128,(it+2)%SLOTS)','ki+64<p.N)load_input<MODE>(p,sm,bar,ch,mi,ni,ki+64,(it+1)%SLOTS)')
        body=body.replace('mbar_arrive_expect_tx(bar+6,98304);','mbar_arrive_expect_tx(bar+6,81920);')
        marker='tma_load_2d(sm+65536+q,&p.maskmap,bar+6,ni+64*wn,mi+64*wm);'
        assert body.count(marker)==1
        body=body.replace(marker,'if(wm==0 && wn==0)tma_load_2d(sm+65536,&p.maskmap,bar+6,ni,mi);')
        marker='mask=*reinterpret_cast<uint32_t*>(sm+65536+off);'
        assert body.count(marker)==1
        body=body.replace(marker,'''mask;
  uint16_t codes=*reinterpret_cast<uint16_t*>(sm+65536+swz128(WG*64+r,c));
  uint32_t lo=codes&255,hi=codes>>8;
  size_t row=size_t(mi+WG*64+r)*p.N+ni+c;
  lo=lo==255?reinterpret_cast<const uint16_t*>(p.mask)[row]:lo<<7;
  hi=hi==255?reinterpret_cast<const uint16_t*>(p.mask)[row+1]:hi<<7;
  mask=lo|(hi<<16);''')
        start=body.index(' int tiles=p.N/128,mode=');end=body.index('\n int mi=',start)
        if banded:
            body=body.replace('outmap[4];int N;','outmap[4];int N;int row_band;')
            start=body.index(' int tiles=p.N/128,mode=');end=body.index('\n int mi=',start)
            grid=' int tiles=p.N/128;int half=blockIdx.x/(2*D*tiles),rem=blockIdx.x%(2*D*tiles),ch=rem/(2*tiles),mode=2*half+rem%2,tile=(rem/2)%tiles;'
        else:
            grid=' int tiles=p.N/128;int half=blockIdx.x/(2*D*tiles*tiles),rem=blockIdx.x%(2*D*tiles*tiles),ch=rem/(2*tiles*tiles),mode=2*half+rem%2,tile=(rem/2)%(tiles*tiles);'
        body=body[:start]+grid+body[end:]
        if banded:body=body.replace('int mi=(tile/tiles)*128,ni=(tile%tiles)*128;','int mi=p.row_band*128,ni=tile*128;')
        body=body.replace('mw_wide_two_group_contract_gp','mw_wide_byte_mask_gp')
        body+='''
extern "C" __global__ void mw_encode_byte_mask(const uint16_t* raw,uint8_t* codes,int M){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<M){unsigned v=raw[i];codes[i]=(!(v&0x807f)&&((v>>7)<255))?(v>>7):255;}
}
extern "C" __global__ void mw_decode_byte_mask(const uint16_t* raw,const uint8_t* codes,uint16_t* out,int M){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<M)out[i]=codes[i]==255?raw[i]:unsigned(codes[i])<<7;
}
'''
        self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWIDTH={p.D}']
        self.cubin=T.compile_text(body,flags);unit=T.load_unit(str(self.cubin),'mw_wide_byte_mask_gp')
        self.k=unit.kernel('mw_wide_byte_mask_gp');self.encode_k=unit.kernel('mw_encode_byte_mask');self.decode_k=unit.kernel('mw_decode_byte_mask')
        self.smem=81920+128;self.k.set_max_dynamic_smem(self.smem)
        L=T._launch_module();fields=list(self.params.fields)
        assert len(fields)==23,len(fields)
        fields[-6]=L.tensor_map(self.codes,[128,128],dims=[p.n,p.n],strides_bytes=[p.n],swizzle='128B',l2='128B')
        self.params=L.Struct(fields)
        if banded:self.band_params=[L.Struct([*fields,i]) for i in range(3)]
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,256,self.smem)))

    def encode(self):self.encode_k.launch(((self.p.M+255)//256,1,1),(256,1,1),[self.mask,self.codes,self.p.M],0)
    def band(self,i):self.k.launch((4*self.p.D*3,1,1),(256,1,1),[self.band_params[i]],self.smem)
    def __call__(self):
        self.encode();self.k.launch((4*self.p.D*9,1,1),(256,1,1),[self.params],self.smem)
