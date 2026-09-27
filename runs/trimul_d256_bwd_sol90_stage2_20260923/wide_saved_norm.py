"""Save our wide K3 normalized tile and FP32 statistics for B1."""
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_wide_forward as F


class SavedOutput:
    def __init__(self,f,p):
        d=p.D;cfg=T.read_config('wide_forward/selection.json')[f'{d}-{p.n}']
        groups,kc=cfg['groups'],cfg['kchunk'];original=f.output
        self.y=original.y;self.grid=original.grid;self.threads=original.threads;self.smem=original.smem
        body=(F.R/'output.cu').read_text()
        body=body.replace('CUtensorMap tri,xn,wp,wg;','CUtensorMap tri,xn,wp,wg,norm;')
        body=body.replace('const float *gamma,*beta; int M,L;','const float *gamma,*beta; float *mu,*rs; int M,L;')
        body=body.replace('bf *y; const float','bf *y,*norm_out; const float')
        body=body.replace('void mw_wide_output(', 'void mw_wide_output_saved_norm(')
        hook='float rs=rsqrtf(sumwarp(s)/H+1e-5f);'
        assert body.count(hook)==2
        body=body.replace(hook,hook+'if constexpr(D<512){if(lane==0){p.mu[row+r]=mu;p.rs[row+r]=rs;}}')
        # D512 forward has a packed two-channel LN reduction. B1's strict
        # reference uses one channel per lane, so save that exact version
        # before forward overwrites its shared triangle. Forward math is intact.
        body=body.replace('#if WIDTH < 512','''#if WIDTH >= 512
  for(int r=warp;r<64;r+=NT/32){
   float sum=0;
   for(int c=lane;c<H;c+=32)sum+=rd(reinterpret_cast<bf*>(sx+(c/64)*8192),swz128(r,(c%64)*2)/2);
   float mu=sumwarp(sum)/H,var=0;
   for(int c=lane;c<H;c+=32){float v=rd(reinterpret_cast<bf*>(sx+(c/64)*8192),swz128(r,(c%64)*2)/2)-mu;var+=v*v;}
   float rs=rsqrtf(sumwarp(var)/H+1e-5f);
   if(lane==0){p.mu[row+r]=mu;p.rs[row+r]=rs;}
   for(int c=lane;c<H;c+=32){float v=rd(reinterpret_cast<bf*>(sx+(c/64)*8192),swz128(r,(c%64)*2)/2);
    p.norm_out[size_t(row+r)*H+c]=__float2bfloat16_rn(fmaf((v-mu)*rs,p.gamma[c],p.beta[c]));
   }
  }
  __syncthreads();
#endif
#if WIDTH < 512''')
        hook='  fence_proxy_async();__syncthreads();\n  for(int col=0;'
        assert body.count(hook)==1
        body=body.replace(hook,'''  fence_proxy_async();__syncthreads();
  if(tid==0&&D<512){for(int c=0;c<H;c+=64){
   asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"::"l"(&p.norm),"r"(smem_u32(sx+c*128)),"r"(c),"r"(row):"memory");
  }tma_store_commit();}
  for(int col=0;''')
        hook='  __syncthreads();\n }\n}'
        assert body.count(hook)==1
        body=body.replace(hook,'  if(tid==0)tma_store_wait_all();\n'+hook)
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v',
               '-I'+str(F.headers()),'-DMW_MINB=1','-DMW_K1_STREAM=0','-DTMN_SIGMOID_TANH=1',
               '-DTMN_WSKIP=1','-DTMN_MASK_TEMPLATE=1',f'-DWIDTH={d}',f'-DGROUPS={groups}',f'-DKCHUNK={kc}']
        self.cubin=T.compile_text(body,flags);self.k=T.load_unit(str(self.cubin),'mw_wide_output_saved_norm').kernel('mw_wide_output_saved_norm')
        self.k.set_max_dynamic_smem(self.smem)
        x,wl,wlg,wr,wrg,wg,wp,gi,bi,go,bo=f.leaves;m=p.M;n=p.n
        maps=[F.tm(f.tri,[64,64],[m,2*d],[m*2]),F.tm(f.front.xn,[64,64],[d,m],[d*2]),
              F.tm(wp,[64,64],[2*d,d],[4*d]),F.tm(wg,[64,64],[d,d],[2*d]),p.maps[3]]
        self.params=T._launch_module().Struct([*maps,x,f.ds,self.y,p.tensors[6],go,bo,p.floats[5],p.floats[6],m,n])
    def __call__(self):
        self.k.launch((self.grid,1,1),(self.threads,1,1),[self.params],self.smem)
        return self.y
