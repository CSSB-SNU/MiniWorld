"""Interleave contraction modes sharing the same channel's dTriangle in L2."""
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
class InterleavedContractGP:
    def __init__(self,plan,order='channel_modes'):
        prior=plan.contract_gp;self.__dict__.update(prior.__dict__)
        body=prior.source_text
        start=body.index(' int tiles=p.N/128,mode=')
        end=body.index('\n int mi=',start)
        decodes={
            'channel_modes':'int ch=blockIdx.x/(4*tiles*tiles),mode=(blockIdx.x/(tiles*tiles))%4,tile=blockIdx.x%(tiles*tiles);',
            'tile_modes':'int ch=blockIdx.x/(4*tiles*tiles),mode=blockIdx.x%4,tile=(blockIdx.x/4)%(tiles*tiles);',
            'pair_channels':'int half=blockIdx.x/(2*D*tiles*tiles),rem=blockIdx.x%(2*D*tiles*tiles),ch=rem/(2*tiles*tiles),mode=2*half+(rem/(tiles*tiles))%2,tile=rem%(tiles*tiles);',
            'pair_tiles':'int half=blockIdx.x/(2*D*tiles*tiles),rem=blockIdx.x%(2*D*tiles*tiles),ch=rem/(2*tiles*tiles),mode=2*half+rem%2,tile=(rem/2)%(tiles*tiles);',
        }
        body=body[:start]+' int tiles=p.N/128;'+decodes[order]+body[end:]
        body=body.replace('mw_wide_two_group_contract_gp','mw_wide_interleaved_contract_gp')
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWIDTH={self.p.D}']
        self.cubin=T.compile_text(body,flags)
        self.k=T.load_unit(str(self.cubin),'mw_wide_interleaved_contract_gp').kernel('mw_wide_interleaved_contract_gp');self.k.set_max_dynamic_smem(self.smem)
    def __call__(self):self.k.launch((4*self.p.D*(self.p.n//128)**2,1,1),(256,1,1),[self.params],self.smem)
