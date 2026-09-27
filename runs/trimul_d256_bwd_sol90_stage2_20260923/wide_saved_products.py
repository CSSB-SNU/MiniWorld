"""Wide K3 saves: D384 norm/projection/gate; D512 gate only."""
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_wide_forward as F


class SavedWideProductsOutput:
    def __init__(self, f, p, products=None):
        d=p.D;assert d in (384,512)
        cfg = T.read_config('wide_forward/selection.json')[f'{d}-{p.n}']
        groups, kc = cfg['groups'], cfg['kchunk']
        original = f.output
        self.y = original.y
        self.grid, self.threads, self.smem = original.grid, original.threads, original.smem
        body = (F.R / 'output.cu').read_text()
        body = body.replace('CUtensorMap tri,xn,wp,wg;', 'CUtensorMap tri,xn,wp,wg,norm;')
        body = body.replace('const float *gamma,*beta; int M,L;', 'const float *gamma,*beta; float *mu,*rs; int M,L;')
        body = body.replace('void mw_wide_output(', 'void mw_wide_output_save_products(')
        body = body.replace(
            'float rs=rsqrtf(sumwarp(s)/H+1e-5f);',
            'float rs=rsqrtf(sumwarp(s)/H+1e-5f); if constexpr(D<512){if(lane==0){p.mu[row+r]=mu;p.rs[row+r]=rs;}}',
        )
        hook = '  fence_proxy_async();__syncthreads();\n  for(int col=0;'
        assert body.count(hook) == 1
        body = body.replace(hook, '''  fence_proxy_async();__syncthreads();
  if(tid==0&&D<512){
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
            assert kc==1
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
 if constexpr(D<512)reinterpret_cast<bf*>(buf+8192*(tid/128))[swz128(r,lc*2)/2]=__float2bfloat16_rn(proj[j]);
 reinterpret_cast<bf*>(buf+8192*(G+tid/128))[swz128(r,lc*2)/2]=__float2bfloat16_rn(gate[j]);
 float v=''')
            point='  }\n  if(tid==0)tma_store_wait_all();'
            assert body.count(point)==1
            body=body.replace(point,'''   __syncthreads();fence_proxy_async();__syncthreads();
   if(tid==0){for(int g=0;g<G;++g){if constexpr(D<512)save_forward_tile(&p.proj_save,buf+8192*g,col+64*g,row);save_forward_tile(&p.gate_save,buf+8192*(G+g),col+64*g,row);}tma_store_commit();tma_store_wait_all();}__syncthreads();
  }
  if(tid==0)tma_store_wait_all();''')
        flags = ['-std=c++17', '-O3', '-arch=sm_90a', '--cubin', '-lineinfo', '-Xptxas=-v',
                 '-I'+str(F.headers()), '-DMW_MINB=1', '-DMW_K1_STREAM=0',
                 '-DTMN_SIGMOID_TANH=1', '-DTMN_WSKIP=1', '-DTMN_MASK_TEMPLATE=1',
                 f'-DWIDTH={d}', f'-DGROUPS={groups}', f'-DKCHUNK={kc}']
        out = T.compile_text(body, flags)
        self.cubin = out
        self.k = T.load_unit(str(out), 'mw_wide_output_save_products').kernel('mw_wide_output_save_products')
        self.k.set_max_dynamic_smem(self.smem)
        x, wl, wlg, wr, wrg, wg, wp, gi, bi, go, bo = f.leaves
        n, m = p.n, p.M
        maps = [F.tm(f.tri, [64, 64], [m, 2*d], [m*2]),
                F.tm(f.front.xn, [64, 64], [d, m], [d*2]),
                F.tm(wp, [64, 64], [2*d, d], [4*d]),
                F.tm(wg, [64, 64], [d, d], [d*2]), p.maps[3]]
        if products is not None:
            maps += [F.tm(t,[64,64],[d,m],[d*2]) for t in products]
        self.params = T._launch_module().Struct(
            [*maps, x, f.ds, self.y, go, bo, p.floats[5], p.floats[6], m, n])

    def __call__(self):
        self.k.launch((self.grid, 1, 1), (self.threads, 1, 1), [self.params], self.smem)
        return self.y
