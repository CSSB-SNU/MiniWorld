"""MiniWorld wide forward: saved K1 + contractions + streaming fused K3.

Explicit development entry; B=1, BF16, H=2D, L divisible by 64.
The saved tensors retain the existing recomputing backward's layouts.
"""
from pathlib import Path
from functools import lru_cache
import hashlib, subprocess, sys
import torch

R = Path(__file__).resolve().parent
ROOT = R.parent.parent
sys.path.insert(0, str(R.parent / 'trimul_widths_20260922'))
from native import T
from selected import pack_into, normalize_into

def k1_smem(D,cfg):
    bi,bj,slots,sk,mb=cfg[:5];ng=bi*bj//64
    stream=len(cfg)>5 and cfg[5]==2
    if D//64%sk or (not stream and slots<D//64//sk):raise ValueError('K1 slots')
    size=bi*bj*D*2+slots*sk*8192+ng*8192+8*D+((2+2*slots)*8+127)//128*128
    if size*mb>232448:raise ValueError('K1 resources')
    return size

@lru_cache(None)
def headers():
    source = ROOT / '.engine-release-2.0.0/src/miniworld_engine/kernels/trimul_inproj/cuda/h100_sources/inference'
    dest = R / 'headers'
    for name in ('tmn_kernels.cuh', 'tmn_ptx.cuh', 'common/tmn_math.cuh'):
        content = (source/name).read_text()
        if name == 'tmn_kernels.cuh':
            content = content.replace('static constexpr int MINB = NCWG == 1 ? 2 : 1;', 'static constexpr int MINB = MW_MINB;')
            content = content.replace('W_RESIDENT || NSLOT >= 2 * SPB', 'W_RESIDENT || (SCHED == 1 && (MW_K1_STREAM || NSLOT >= SPB)) || NSLOT >= 2 * SPB')
        p = dest/name
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists() or p.read_text() != content:
            p.write_text(content)
    return dest

@lru_cache(None)
def compile_kernel(kind, D, cfg, normalize=True):
    inc = headers()
    if kind == 'k1':
        bi,bj,slots,sk,mb = cfg[:5]
        shared=len(cfg)>5 and cfg[5]
        stream=int(shared==2)
        include='front_shared.cuh' if shared else 'tmn_kernels.cuh'
        function='mw_k1_shared' if shared else 'k1_body'
        body = (f'#include "{include}"\n'
                f'using C=tmn::K1Cfg<{D},{2*D},false,{bi},{bj},{slots},{sk},{1 if stream else -1}>;\n'
                'extern "C" __global__ __launch_bounds__(C::NTHR,C::MINB) '
                'void mw_wide_front(__grid_constant__ const tmn::K1Params p){'
                f'tmn::sm90::{function}<C,true,{int(normalize)},false,{str(normalize).lower()},1>(p);}}')
        smem = k1_smem(D, cfg)
        name = 'mw_wide_front'
        flags = [f'-DMW_MINB={mb}',f'-DMW_K1_STREAM={stream}']
    else:
        g = cfg[0]
        body = (R/('output_pipeline.cu' if len(cfg)>1 and cfg[1] else 'output.cu')).read_text()
        kc=cfg[2] if len(cfg)>2 else 1
        smem = 2*D*128 + 2*8192*(g+1)*kc + (8*(2*D) if cfg[1] else 0) + 512 + 128
        name = 'mw_wide_output'
        flags = [f'-DMW_MINB=1','-DMW_K1_STREAM=0',f'-DWIDTH={D}',f'-DGROUPS={g}',f'-DKCHUNK={kc}']
    flags += ['-DTMN_SIGMOID_TANH=1','-DTMN_WSKIP=1','-DTMN_MASK_TEMPLATE=1',
              '-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(inc),'-I'+str(R)]
    key = hashlib.sha256(body.encode()+str(flags).encode()+(R/'front_shared.cuh').read_bytes()+b''.join((inc/n).read_bytes() for n in ('tmn_kernels.cuh','tmn_ptx.cuh','common/tmn_math.cuh'))).hexdigest()
    out = R/'build'/f'{key}.cubin'
    out.parent.mkdir(exist_ok=True)
    if not out.exists():
        src=out.with_suffix('.cu');src.write_text(body)
        p=subprocess.run(['nvcc',*flags,str(src),'-o',str(out)],text=True,capture_output=True)
        out.with_suffix('.log').write_text(p.stdout+p.stderr)
        if p.returncode: raise RuntimeError(p.stderr)
    L=T._launch_module();drv=L.BlockDriver(device=torch.cuda.current_device());mod=drv.load(out.read_bytes())
    unit=L.Unit(name,'sm_90a',torch.cuda.current_device(),drv.drv,mod,{},str(out))
    k=unit.kernel(name);k.set_max_dynamic_smem(smem)
    return k,smem

def tm(t,box,dims,strides):
    return T._launch_module().tensor_map(t,box,dims=dims,strides_bytes=strides,swizzle='128B',l2='128B')

class Front:
    def __init__(self,x,w,mask,gi,bi,cfg,normalize=True):
        _,N,_,D=x.shape
        self.cfg=cfg;self.normalize=normalize
        self.xn=torch.empty_like(x);self.ab=x.new_empty((4*D,N,N))
        self.x,self.gi,self.bi=x,gi,bi
        self.k,self.smem=compile_kernel('k1',D,tuple(cfg),normalize)
        a,b,slots,sk,mb=cfg[:5]
        operand=x if normalize else self.xn
        maps=[tm(operand,[64,b,a],[D,N,N],[D*2,N*D*2]),tm(w,[64,64],[D,8*D],[D*2]),tm(self.ab,[64,1,32],[N,N,4*D],[N*2,N*N*2])]
        tj=(N+b-1)//b;tiles=((N+a-1)//a)*tj
        self.params=T._launch_module().Struct([*maps,mask,gi,bi,self.ab,None,self.xn if normalize else None,N,N,tj,tiles,1,N,1,1e-5,N*D,D,0,0])
        self.grid=min(tiles,torch.cuda.get_device_properties(x.device).multi_processor_count*mb)
        self.threads=128*(a*b//64+1)
    def __call__(self):
        if not self.normalize: normalize_into(self.xn,self.x,self.gi,self.bi)
        self.k.launch((self.grid,1,1),(self.threads,1,1),[self.params],self.smem)

class Output:
    def __init__(self,tri,xn,x,wp,wg,go,bo,ds,groups=2,grid_mult=1,pipeline=False,kchunk=1):
        _,N,_,D=x.shape;self.y=torch.empty_like(x)
        self.k,self.smem=compile_kernel('k3',D,(groups,pipeline,kchunk))
        maps=[tm(tri,[64,64],[N*N,2*D],[N*N*2]),tm(xn,[64,64],[D,N*N],[D*2]),tm(wp,[64,64],[2*D,D],[4*D]),tm(wg,[64,64],[D,D],[D*2])]
        self.params=T._launch_module().Struct([*maps,x,ds,self.y,go,bo,N*N,N])
        self.threads=128*(groups+int(pipeline))
        self.grid=torch.cuda.get_device_properties(x.device).multi_processor_count*grid_mult
    def __call__(self):
        self.k.launch((self.grid,1,1),(self.threads,1,1),[self.params],self.smem)
        return self.y

class Forward:
    def __init__(self,leaves,mask,ds,k1_cfg,groups=2,normalize=True,kchunk=1):
        x,wl,wlg,wr,wrg,wg,wp,gi,bi,go,bo=leaves
        _,N,_,D=x.shape
        assert D in (256,384,512) and N%64==0
        self.weights=(wl,wlg,wr,wrg);self.w=x.new_empty((8*D,D));self.D=D
        self.mask=mask.bfloat16().contiguous();self.ds=ds
        self.front=Front(x,self.w,self.mask,gi,bi,k1_cfg,normalize)
        self.tri=x.new_empty((2*D,N,N))
        self.output=Output(self.tri,self.front.xn,x,wp,wg,go,bo,ds,groups,kchunk=kchunk)
        self.saved=(self.front.xn,self.front.ab,self.tri)
    def __call__(self):
        pack_into(self.w,*self.weights);self.front();ab=self.front.ab;d=self.D;h=2*d
        torch.bmm(ab[:d],ab[h:h+d].transpose(-1,-2),out=self.tri[:d])
        torch.bmm(ab[d:h].transpose(-1,-2),ab[h+d:],out=self.tri[d:])
        return self.output()
