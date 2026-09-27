"""Accumulate affine gradients in the exact-order row pass after fused dNorm."""
from pathlib import Path
from wide_full_width_dn_ln import FullWidthDnLN
from d256_half_width_dn_ln import HalfWidthDnLN
from d256_async_full_width_dn_ln import AsyncFullWidthDnLN
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
class IntegratedFullWidthDnLN:
    def __init__(self,plan,emit_dn=False,asynchronous=False,whole_weight=False,half_width=False):
        assert not (half_width and asynchronous)
        base=HalfWidthDnLN(plan,emit_dn) if half_width else (AsyncFullWidthDnLN if asynchronous else FullWidthDnLN)(plan,emit_dn,64)
        self.__dict__.update(base.__dict__);p=plan.p;assert p.D==256
        body=base.source_text
        barrier='named_bar_sync(1,NT);'
        marker='float gg[((H+NT-1)/NT)]={},bb[((H+NT-1)/NT)]={};'
        assert body.count(marker)==1
        body=body.replace(marker,'float gg[H/32]={},bb[H/32]={};')
        begin=body.index('  // Channel ownership');end=body.index('  for(int r=warp;',begin)
        body=body[:begin]+body[end:]
        marker='v=dy*gamma[c];s0+=v;s1+=v*z;'
        assert body.count(marker)==1;body=body.replace(marker,marker+'gg[q]+=dy*z;bb[q]+=dy;')
        marker='s0=sumwarp(s0)/H;s1=sumwarp(s1)/H;'
        assert body.count(marker)==1;body=body.replace(marker,marker+'asm volatile("":::"memory");')
        begin=body.index(' if(tid==0)tma_store_wait_all();');end=body.index('\n}',begin)
        body=body[:begin]+'''
 if(tid==0)tma_store_wait_all();named_bar_sync(1,NT);
 float* scratch=reinterpret_cast<float*>(sm);
 #pragma unroll
 for(int q=0;q<H/32;++q){int c=lane+q*32;
  scratch[warp*H+c]=gg[q];scratch[(NT/32+warp)*H+c]=bb[q];
 }
 named_bar_sync(1,NT);
 for(int c=tid;c<H;c+=NT){float g=0,b=0;
  #pragma unroll
  for(int w=0;w<NT/32;++w){g+=scratch[w*H+c];b+=scratch[(NT/32+w)*H+c];}
  p.partial[size_t(blockIdx.x)*2*H+c]=g;p.partial[size_t(blockIdx.x)*2*H+H+c]=b;
 }
'''+body[end:]
        fields=base.params.fields.copy()
        if whole_weight:
            marker=' for(int c=0;c<H;c+=64)tma_load_2d(sm+slot*INPUT+4096+(c/64)*4096,&p.wp,bar+slot,col+c,ki);'
            if half_width:marker=marker.replace('c<H','c<D')
            assert body.count(marker)==1
            coord='col/64' if half_width else '0'
            body=body.replace(marker,f' tma_load_3d(sm+slot*INPUT+4096,&p.wp,bar+slot,0,ki,{coord});')
            fields[3]=T._launch_module().tensor_map(plan.leaves[6],[64,32,4 if half_width else 8],dims=[64,p.D,8],strides_bytes=[1024,128],swizzle='128B',l2='128B')
        oldname='mw_d256_half_width_dn_ln' if half_width else 'mw_d256_async_full_width_dn_ln' if asynchronous else 'mw_wide_full_width_dn_ln'
        name='mw_d256_integrated_full_width_dn_ln';body=body.replace(oldname,name)
        self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),'-DWIDTH=256',f'-DEMIT_DN={int(emit_dn)}']
        self.cubin=T.compile_text(body,flags);unit=T.load_unit(str(self.cubin),name)
        self.k=unit.kernel(name);self.k.set_max_dynamic_smem(self.smem)
        self.params=T._launch_module().Struct(fields)
        self.reduce_first=unit.kernel('mw_wide_stream_ln_reduce_first');self.reduce_last=unit.kernel('mw_wide_stream_ln_reduce_last')
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,self.threads,self.smem)))
        if asynchronous:assert self.registers>=128
    def __call__(self):
        self.k.launch((self.rows,1,1),(self.threads,1,1),[self.params],self.smem)
        self.reduce_first.launch((4,self.chunks,1),(256,1,1),[self.rp],0)
        self.reduce_last.launch((4,1,1),(128,1,1),[self.rp],0)
