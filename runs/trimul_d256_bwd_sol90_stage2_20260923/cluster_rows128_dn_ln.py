from pathlib import Path
import ctypes,torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

class ClusterRows128DnLN:
    def __init__(self,plan,emit_dn=False):
        p=plan.p;d=p.D;h=2*d;self.p=p;self.cluster=4 if d==256 else 8
        root=Path(__file__).resolve().parent
        base=(root/'cluster_dn_ln.cu').read_text()
        start=base.index('TMN_DEVI void normalize(')
        finish=base.index('extern "C" __global__ __launch_bounds__(256,4)')
        normal=(root/'cluster_rows_dn_ln.cuh').read_text()
        helper=(root/'dn_slim.cu').read_text().split('TMN_DEVI uint32_t rawpos')[1].split('TMN_DEVI void producer')[0]
        helper=('TMN_DEVI uint32_t rawpos'+helper).replace('kc<8','kc<H/64')
        normal=normal.replace('// ROW_TRANSPOSE_HELPERS',helper)
        body=(base[:start]+normal+base[finish:]).replace('// MMA_HELPERS',(root.parent/'trimul_d256_bwd_sol90_20260923/mma.cuh').read_text())
        body=body.replace('mw_cluster_dn_ln_finish','mw_cluster_rows_dn_ln_finish')
        assert d==512
        a=body.index('TMN_DEVI void load_gemm(');b=body.index('// Row ownership replaces',a)
        body=body[:a]+(root/'cluster_rows128_gemm.cuh').read_text()+body[b:]
        body=body.replace('STAGE=24576','STAGE=32768').replace('BAR=TRI+16384','BAR=TRI+32768')
        a=body.index('template<bool INVERSE> TMN_DEVI void transpose(');b=body.index('TMN_DEVI void load_gemm(',a)
        helper=body[a:b].replace('warp=threadIdx.x/32','warp=(threadIdx.x%128)/32').replace('__syncthreads();','named_bar_sync(1+threadIdx.x/128,128);')
        body=body[:a]+helper+body[b:]
        body=body.replace('ROWS=64/CLSIZE,RDN=16384,RTRI=32768','ROWS=128/CLSIZE,RDN=32768,RTRI=65536')
        body=body.replace('src=slab*8192+peer*ROWS*128','src=(peer*ROWS/64)*16384+slab*8192+(peer*ROWS%64)*128')
        body=body.replace('r<ROWS;r+=4','r<ROWS;r+=8')
        body=body.replace('transpose32<true>(sm+RDN);','if(tid<128)transpose32<true>(sm+RDN);__syncthreads();')
        body=body.replace(' // Gather warp-local affine sums',' cooperative_groups::this_cluster().sync();\n // Gather warp-local affine sums')
        body=body.replace('(4+warp)*H','(8+warp)*H').replace('(4+w)*H','(8+w)*H').replace('c<H;c+=128','c<H;c+=256').replace('w<4;++w','w<8;++w')
        body=body.replace('if(rank<ACTIVE){int c=rank*128+tid;','if(rank<ACTIVE && tid<128){int c=rank*128+tid;')
        body=body.replace('size_t(row/64)','size_t(row/128)').replace('(blockIdx.x/CLSIZE)*64','(blockIdx.x/CLSIZE)*128')
        body=body.replace('c=blockIdx.x*128+tid;c<H;c+=gridDim.x*128','c=blockIdx.x*256+tid;c<H;c+=gridDim.x*256')
        body=body.replace('tile<p.M/64','tile<p.M/128').replace('__launch_bounds__(128,2)','__launch_bounds__(256,1)')
        body=body.replace('mw_cluster_rows_dn_ln','mw_cluster_rows128_dn_ln')

        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWIDTH={d}',f'-DEMIT_DN={int(emit_dn)}']
        self.cubin=T.compile_text(body,flags);unit=T.load_unit(str(self.cubin),'mw_cluster_rows128_dn_ln')
        self.k=unit.kernel('mw_cluster_rows128_dn_ln');self.finish=unit.kernel('mw_cluster_rows128_dn_ln_finish');self.smem=131072+256+h*4+(128//self.cluster)*8;self.k.set_max_dynamic_smem(self.smem)
        self.partial=torch.empty((p.M//128,2,h),device=p.x.device,dtype=torch.float32)
        L=T._launch_module()
        tm=lambda t,fast,slow:L.tensor_map(t,[64,64],dims=[fast,slow],strides_bytes=[fast*2],swizzle='128B',l2='128B')
        self.params=L.Struct([tm(p.tensors[7],d,p.M),tm(plan.b1.wp,h,d),tm(p.tri,p.M,h),L.tensor_map(p.dt,[128//self.cluster,64],dims=[p.M,h],strides_bytes=[p.M*2],swizzle='32B',l2='128B'),tm(p.tensors[9],h,p.M),p.floats[5],p.floats[6],p.floats[2],self.partial,p.floats[10],p.floats[11],p.M])
        self.args=L._Packed([self.params]);self.drv=unit.drv;driver=self.drv.d
        attr=driver.CUlaunchAttribute();attr.id=driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        attr.value.clusterDim.x=self.cluster;attr.value.clusterDim.y=1;attr.value.clusterDim.z=1
        cfg=driver.CUlaunchConfig();cfg.gridDimX=p.M//128*self.cluster;cfg.gridDimY=1;cfg.gridDimZ=1;cfg.blockDimX=256;cfg.blockDimY=1;cfg.blockDimZ=1;cfg.sharedMemBytes=self.smem;cfg.attrs=[attr];cfg.numAttrs=1
        self.cfg=cfg;self.attr=attr
        function=driver.CUfunction(int(self.k.handle))
        self.active_clusters=int(self.drv._unwrap('cuOccupancyMaxActiveClusters',driver.cuOccupancyMaxActiveClusters(function,cfg)))
        query=lambda n:int(self.drv._unwrap('cuFuncGetAttribute',driver.cuFuncGetAttribute(getattr(driver.CUfunction_attribute,n),function)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
    def __call__(self):
        self.cfg.hStream=self.drv.d.CUstream(int(torch.cuda.current_stream().cuda_stream))
        self.drv._unwrap('cuLaunchKernelEx',self.drv.d.cuLaunchKernelEx(self.cfg,self.drv.d.CUfunction(int(self.k.handle)),ctypes.addressof(self.args.array),0))
        self.finish.launch((self.p.D//16*8,1,1),(256,1,1),[self.params],0)
