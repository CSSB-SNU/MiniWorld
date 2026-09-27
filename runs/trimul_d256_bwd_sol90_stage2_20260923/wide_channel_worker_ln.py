"""Independent affine worker and two protected input slots for output LN."""
from pathlib import Path
import ctypes
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_wide_forward as F
class ChannelWorkerLN:
    def __init__(self,p,rows=16,minblocks=3,main_regs=112,affine_regs=48,reload=True):
        root=Path(__file__).resolve().parent;h=2*p.D
        self.smem=rows*h*6+128+h*4
        helpers=(root/'dn_slim.cu').read_text().split('TMN_DEVI uint32_t rawpos')[1].split('TMN_DEVI void producer')[0]
        helpers=('TMN_DEVI uint32_t rawpos'+helpers).replace('kc<8','kc<H/64')
        if rows==8:helpers=(root/'tile_transpose8.cuh').read_text()
        body=(root/'wide_affine_output_ln.cu').read_text().replace('// TRANSPOSE_HELPERS',helpers)
        body=body.replace('ROWS=16',f'ROWS={rows}')
        body=body.replace('__launch_bounds__(256,2)',f'__launch_bounds__(256,{minblocks})')
        body=body.replace('setmaxnreg_inc<144>()',f'setmaxnreg_inc<{main_regs}>()').replace('setmaxnreg_dec<112>()',f'setmaxnreg_dec<{affine_regs}>()')
        begin=body.index('TMN_DEVI void affine(');end=body.index('extern "C" __global__',begin)
        body=body[:begin]+(root/'wide_channel_worker_affine.cuh').read_text()+body[end:]
        if reload:
            body=body.replace('__bfloat162float(reinterpret_cast<bf*>(sm)[pos(r,c)])','reload_bf(sm,pos(r,c))')
            body=body.replace('__bfloat162float(reinterpret_cast<bf*>(sm+SB)[pos(r,c)])','reload_bf(sm+SB,pos(r,c))')
            marker='TMN_DEVI void normalizer('
            helper='TMN_DEVI float reload_bf(uint8_t* sm,int i){unsigned v;asm volatile("ld.shared.u16 %0,[%1];":"=r"(v):"r"(smem_u32(sm+2*i)):"memory");return __uint_as_float(v<<16);}'
            body=body.replace(marker,helper+'\n'+marker)
        body=body.replace('mw_wide_affine_output_ln','mw_wide_channel_worker_ln')
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(F.headers()),'-DMW_MINB=1','-DMW_K1_STREAM=0',f'-DWIDTH={p.D}']
        self.cubin=T.compile_text(body,flags)
        self.kernel=T.load_unit(str(self.cubin),'mw_wide_channel_worker_ln').kernel('mw_wide_channel_worker_ln');self.kernel.set_max_dynamic_smem(self.smem)
        drv=self.kernel.unit.drv
        self.registers=int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(drv.d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS,drv.d.CUfunction(int(self.kernel.handle)))))
        assert 2*self.registers>=main_regs+affine_regs,('Insufficient CTA register pool',self.registers,main_regs,affine_regs)
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(drv.d.CUfunction(int(self.kernel.handle)),256,self.smem)))
        assert self.occupancy>0
        self.grid=torch.cuda.get_device_properties(p.x.device).multi_processor_count*self.occupancy
        L=T._launch_module()
        tm=lambda t:L.tensor_map(t,[rows,64],dims=[p.M,h],strides_bytes=[p.M*2],swizzle='none' if rows==8 else '32B',l2='128B')
        dn=L.tensor_map(p.tensors[9],[64,rows],dims=[h,p.M],strides_bytes=[h*2],swizzle='128B',l2='128B')
        self.params=L.Struct([tm(p.tri),tm(p.dt),dn,p.tensors[9],p.floats[5],p.floats[6],p.floats[2],p.floats[10],p.floats[11],p.M])
    def __call__(self):
        L=T._launch_module();drv=self.kernel.unit.drv;args=L._Packed([self.params])
        drv._unwrap('cuLaunchCooperativeKernel',drv.d.cuLaunchCooperativeKernel(drv.d.CUfunction(int(self.kernel.handle)),self.grid,1,1,256,1,1,self.smem,drv.d.CUstream(int(torch.cuda.current_stream().cuda_stream)),ctypes.addressof(args.array)))
