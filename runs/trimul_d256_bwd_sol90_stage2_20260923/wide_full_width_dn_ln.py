"""Four WGMMA groups cover every dNorm channel before in-CTA exact-order LN."""
from pathlib import Path
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
class FullWidthDnLN:
    def __init__(self,plan,emit_dn=False,ln_rows=16):
        p=plan.p;d=p.D;h=2*d;self.p=p;self.threads=512;root=Path(__file__).resolve().parent
        assert d in (256,384)
        body=(root/'wide_stream_dn_ln.cu').read_text()
        body=body.replace('NT=D,ROWS=16','NT=512,ROWS=16,WC=H/4').replace('INPUT=4096+64*D','INPUT=4096+64*H').replace('BAR=DN+64*H*2;','BAR=DN+64*H*2,STATS=BAR+128;')
        body=body.replace('for(int c=0;c<D;c+=64)','for(int c=0;c<H;c+=64)').replace('for(int col=0;col<H;col+=D)','for(int col=0;col<H;col+=H)')
        body=body.replace('float v[64]={};','float v[WC/2]={};').replace('wg*8192','wg*WC*64').replace('wg*128','wg*WC').replace('static_for<32>([&](auto jj)','static_for<WC/4>([&](auto jj)')
        body=body.replace('mma128_off<q*32,q*2048,0,1>','MMA<q*32,q*2048,0,1>')
        helper=(root/('mma192_offset.cuh' if d==384 else 'mma_offset.cuh')).read_text()
        body=body.replace('MMA<','mma192_off<' if d==384 else 'mma128_off<')
        body=body.replace('// MMA_HELPERS',helper).replace('// TRANSPOSE_HELPERS',(root/'tile_transpose.cuh').read_text())
        body=body.replace('H/NT','((H+NT-1)/NT)')
        body=body.replace('int c=tid+q*NT;','int c=tid+q*NT;if(c>=H)continue;')
        body=body.replace('mbar_arrive_expect_tx(bar+2,SB);','mbar_arrive_expect_tx(bar+2,SB+ROWS*8);')
        a='   for(int c=0;c<H;c+=64)tma_load_2d(sm+(c/64)*(ROWS*128),&p.tri,bar+2,row+half*ROWS,c);'
        assert body.count(a)==1
        body=body.replace(a,'''   tma_load_3d(sm,&p.tri,bar+2,row+half*ROWS,0,0);
   asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1],64,[%2];"::"r"(smem_u32(sm+STATS)),"l"(p.mu+row+half*ROWS),"r"(smem_u32(bar+2)):"memory");
   asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1],64,[%2];"::"r"(smem_u32(sm+STATS+64)),"l"(p.rs+row+half*ROWS),"r"(smem_u32(bar+2)):"memory");''')
        body=body.replace('p.mu[row+half*ROWS+r]','reinterpret_cast<float*>(sm+STATS)[r]').replace('p.rs[row+half*ROWS+r]','reinterpret_cast<float*>(sm+STATS)[ROWS+r]')
        a=body.index('  if(tid==0){for(int c=0;c<H;c+=64){');b=body.index('tma_store_commit();',a)
        body=body[:a]+'''  if(tid==0){
   asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0,{%2,0,0}],[%1];"::"l"(&p.dt),"r"(smem_u32(sm)),"r"(row+half*ROWS):"memory");
  '''+body[b:]
        body=body.replace('mw_wide_stream_dn_ln','mw_wide_full_width_dn_ln')
        assert ln_rows in (16,32,64)
        if ln_rows!=16:
            body=body.replace('ROWS=16',f'ROWS={ln_rows}').replace('sm+STATS+64',f'sm+STATS+{ln_rows*4}')
            body=body.replace('[%0],[%1],64,[%2]',f'[%0],[%1],{ln_rows*4},[%2]')
            body=body.replace('if constexpr(ROWS==32)return sw64(v);','if constexpr(ROWS==64)return swz128(c,r*2);else if constexpr(ROWS==32)return sw64(v);')
        self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWIDTH={d}',f'-DEMIT_DN={int(emit_dn)}']
        self.cubin=T.compile_text(body,flags);unit=T.load_unit(str(self.cubin),'mw_wide_full_width_dn_ln')
        self.k=unit.kernel('mw_wide_full_width_dn_ln');self.smem=8192+512*d+128+ln_rows*8;self.k.set_max_dynamic_smem(self.smem)
        self.reduce_first=unit.kernel('mw_wide_stream_ln_reduce_first');self.reduce_last=unit.kernel('mw_wide_stream_ln_reduce_last')
        self.rows=p.M//64;self.chunks=(self.rows+255)//256
        self.partial=torch.empty((self.rows,2,h),device=p.x.device,dtype=torch.float32)
        self.tmp=torch.empty((self.chunks,2,h),device=p.x.device,dtype=torch.float32)
        L=T._launch_module()
        tri=lambda t:L.tensor_map(t,[ln_rows,64,h//64],dims=[p.M,64,h//64],strides_bytes=[p.M*2,p.M*128],swizzle='128B' if ln_rows==64 else '64B' if ln_rows==32 else '32B',l2='128B')
        dp=L.tensor_map(p.tensors[7],[32,64],dims=[d,p.M],strides_bytes=[d*2],swizzle='64B',l2='128B')
        wp=L.tensor_map(plan.b1.wp if d!=256 else plan.leaves[6],[64,32],dims=[h,d],strides_bytes=[h*2],swizzle='128B',l2='128B')
        self.params=L.Struct([tri(p.tri),tri(p.dt),dp,wp,p.floats[5],p.floats[6],p.floats[2],self.partial,p.tensors[9],p.M])
        self.rp=L.Struct([self.partial,self.tmp,p.floats[10],p.floats[11],self.rows,self.chunks])
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,512,self.smem)))
    def __call__(self):
        self.k.launch((self.rows,1,1),(self.threads,1,1),[self.params],self.smem)
        self.reduce_first.launch((2*self.p.D//128,self.chunks,1),(256,1,1),[self.rp],0)
        self.reduce_last.launch((2*self.p.D//128,1,1),(128,1,1),[self.rp],0)
