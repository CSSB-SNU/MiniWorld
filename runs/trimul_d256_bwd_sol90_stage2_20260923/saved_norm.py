"""Experimental K3 normalization saves; preserves our forward arithmetic."""
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_wide_forward as F


class SavedNormOutput:
    def __init__(self, f, p, products=None):
        assert p.x.shape[-1] == 256
        cfg = T.read_config('wide_forward/selection.json')[f'256-{p.n}']
        groups, kc = cfg['groups'], cfg['kchunk']
        original = f.output
        self.y = original.y
        self.grid, self.threads, self.smem = original.grid, original.threads, original.smem
        body = (F.R / 'output.cu').read_text()
        body = body.replace('CUtensorMap tri,xn,wp,wg;', 'CUtensorMap tri,xn,wp,wg,norm;')
        body = body.replace('const float *gamma,*beta; int M,L;', 'const float *gamma,*beta; float *mu,*rs; int M,L;')
        body = body.replace('void mw_wide_output(', 'void mw_d256_output_save_norm(')
        body = body.replace(
            'float rs=rsqrtf(sumwarp(s)/H+1e-5f);',
            'float rs=rsqrtf(sumwarp(s)/H+1e-5f); if(lane==0){p.mu[row+r]=mu;p.rs[row+r]=rs;}',
        )
        hook = '  fence_proxy_async();__syncthreads();\n  for(int col=0;'
        assert body.count(hook) == 1
        body = body.replace(hook, '''  fence_proxy_async();__syncthreads();
  if(tid==0){
   for(int c=0;c<H;c+=64){
    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"::"l"(&p.norm),"r"(smem_u32(sx+c*128)),"r"(c),"r"(row):"memory");
   }
   tma_store_commit();
  }
  for(int col=0;''')
        end = '  __syncthreads();\n }\n}'
        assert body.count(end) == 1
        body = body.replace(end, '  if(tid==0)tma_store_wait_all();\n' + end)
        if products is not None:
            assert groups==4 and kc==1
            body=body.replace('CUtensorMap tri,xn,wp,wg,norm;', 'CUtensorMap tri,xn,wp,wg,norm,proj_save,gate_save;')
            helper='''TMN_DEVI void save_forward_tile(const CUtensorMap* map,uint8_t* sm,int c,int row){
 asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"::"l"(map),"r"(smem_u32(sm)),"r"(c),"r"(row):"memory");
}
'''
            body=body.replace('extern "C" __global__',helper+'\nextern "C" __global__')
            point='size_t ix=size_t(row+r)*D+c;float v='
            assert body.count(point)==1
            body=body.replace(point,'''size_t ix=size_t(row+r)*D+c;
 int lc=c-col-64*(tid/128);
 reinterpret_cast<bf*>(buf+8192*(tid/128))[swz128(r,lc*2)/2]=__float2bfloat16_rn(proj[j]);
 reinterpret_cast<bf*>(buf+8192*(G+tid/128))[swz128(r,lc*2)/2]=__float2bfloat16_rn(gate[j]);
 float v=''')
            point='  }\n  if(tid==0)tma_store_wait_all();'
            assert body.count(point)==1
            body=body.replace(point,'''   __syncthreads();fence_proxy_async();__syncthreads();
   if(tid==0){for(int g=0;g<G;++g){save_forward_tile(&p.proj_save,buf+8192*g,col+64*g,row);save_forward_tile(&p.gate_save,buf+8192*(G+g),col+64*g,row);}tma_store_commit();}
  }
  if(tid==0)tma_store_wait_all();''')
        flags = ['-std=c++17', '-O3', '-arch=sm_90a', '--cubin', '-lineinfo', '-Xptxas=-v',
                 '-I'+str(F.headers()), '-DMW_MINB=1', '-DMW_K1_STREAM=0',
                 '-DTMN_SIGMOID_TANH=1', '-DTMN_WSKIP=1', '-DTMN_MASK_TEMPLATE=1',
                 '-DWIDTH=256', f'-DGROUPS={groups}', f'-DKCHUNK={kc}']
        out = T.compile_text(body, flags)
        self.cubin = out
        self.k = T.load_unit(str(out), 'mw_d256_output_save_norm').kernel('mw_d256_output_save_norm')
        self.k.set_max_dynamic_smem(self.smem)
        x, wl, wlg, wr, wrg, wg, wp, gi, bi, go, bo = f.leaves
        n, m = p.n, p.M
        maps = [F.tm(f.tri, [64, 64], [m, 512], [m*2]),
                F.tm(f.front.xn, [64, 64], [256, m], [512]),
                F.tm(wp, [64, 64], [512, 256], [1024]),
                F.tm(wg, [64, 64], [256, 256], [512]), p.maps[3]]
        if products is not None:
            maps += [F.tm(t,[64,64],[256,m],[512]) for t in products]
        self.params = T._launch_module().Struct(
            [*maps, x, f.ds, self.y, go, bo, p.floats[5], p.floats[6], m, n])

    def __call__(self):
        self.k.launch((self.grid, 1, 1), (self.threads, 1, 1), [self.params], self.smem)
        return self.y


def enable(plan):
    plan.f.output = SavedNormOutput(plan.f, plan.p)
    plan.b1.prepare.normalize = lambda: None
    plan.saved = (*plan.saved, plan.p.tensors[6], plan.p.floats[5], plan.p.floats[6])
