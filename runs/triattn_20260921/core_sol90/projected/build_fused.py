"""Compile a full projected attention prototype, without installing it."""
from pathlib import Path
import os,sys,shutil,json
from torch.utils.cpp_extension import load
from make_fused import generate
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
preg=int(sys.argv[1]) if len(sys.argv)>1 else 64
creg=(496-preg)//2
assert creg%8==0 and preg%8==0
ss=os.environ.get('PROJECTED_SS')=='1'
opaque=os.environ.get('PROJECTED_OPAQUE')=='1'
rolled=os.environ.get('PROJECTED_ROLLED')=='1'
wide=os.environ.get('PROJECTED_WIDE')=='1'
cluster=os.environ.get('PROJECTED_CLUSTER')=='1'
tag='p'+str(preg)+('ss' if ss else '')+('z' if opaque else '')+('loop' if rolled else '')+('m128' if wide else '')+('cl4' if cluster else '');name='triattn_projected_'+tag
build=HERE/('build_'+tag);build.mkdir(exist_ok=True)
proof=json.loads((HERE/'check-stsm-14403.json').read_text())
assert proof['passed']==10
projection=(HERE/'projection_stsm.cu').read_text()
body=generate(projection,preg,creg)
if wide:
 a=body.index('template<class MMA,class WTensor>');b=body.index('template<bool Safe>',a)
 helper=body[a:b]
 helper=helper.replace('Shape<_64,_128>{},make_coord(hh,0)','Shape<_128,_128>{},make_coord(0,0)')
 helper=helper.replace('Shape<_64,NN>','Shape<_128,NN>')
 helper=helper.replace('Shape<_64,_32>{},make_coord(hh,0)','Shape<_128,_32>{},make_coord(0,0)')
 helper=helper.replace('Shape<_64,_64>{},make_coord(hh,0)','Shape<_128,_64>{},make_coord(0,0)')
 body=body[:a]+helper+body[b:]
 body=body.replace('project<MQ>(s,w,r,0,0,pt);project<MQ>(s,w,r,1,0,pt);','project<MQ>(s,w,r,0,0,pt);')
 body=body.replace('project<MKV>(s,w,r,0,st,pt);project<MKV>(s,w,r,1,st,pt);','project<MKV>(s,w,r,0,st,pt);')
if rolled:
 old=' #pragma unroll\n for(int kk=0;kk<size<2>(a);kk++)gemm(mma,a(_,_,kk),b(_,_,kk),acc);'
 assert body.count(old)==1
 body=body.replace(old,old.replace('#pragma unroll','#pragma unroll 1'))
if opaque:
 body=body.replace('ClusterTransactionBarrier xr[Rows],wr,bf[8];','ClusterTransactionBarrier xr[Rows],wr,bf[8];unsigned desc_zero;')
 body=body.replace('s.qr.init(1);s.wr.init(1);','s.qr.init(1);s.wr.init(1);s.desc_zero=0;')
 old=' auto a=mt.partition_fragment_A(x);auto b=mt.partition_fragment_B(weight);'
 assert old in body
 body=body.replace(old,''' auto raw_a=mt.partition_fragment_A(x);auto raw_b=mt.partition_fragment_B(weight);
 // Prevent all projection descriptors being retained across separate calls.
 // This immutable shared zero is initialized before the CTA rendezvous.
 unsigned z=*reinterpret_cast<unsigned volatile*>(&s.desc_zero);
 auto ad=raw_a.data().desc_,bd=raw_b.data().desc_;
 ad.reg32_[0]+=z;bd.reg32_[0]+=z;
 auto a=make_tensor(GMMA::DescriptorIterator{ad},raw_a.layout());
 auto b=make_tensor(GMMA::DescriptorIterator{bd},raw_b.layout());''')
if ss:
 body=body.replace('using QK=decltype(make_tiled_mma(GMMA::rs_op_selector','using QK=decltype(make_tiled_mma(GMMA::ss_op_selector')
 start=body.index(' auto qshared=');end=body.index(' {\n  Score sc',start)
 body=body[:start]+body[end:]
 start=body.index('  #pragma unroll\n  for(int hh=0;hh<2;++hh){\n   auto src=qthr')
 end=body.index('  warpgroup_fence_operand(acc[0]);',start)
 body=body[:start]+body[end:]
 old='auto kb=tq.partition_fragment_B(ks);auto& qa=hh==0?qa0:qa1;'
 assert old in body
 body=body.replace(old,'''auto kb=tq.partition_fragment_B(ks);
   auto qs=local_tile(make_tensor(make_smem_ptr(s.q[wg]),SQ{}),Shape<_64,_32>{},make_coord(hh,0));
   auto qa=tq.partition_fragment_A(qs);''')
 body=body.replace('  warpgroup_fence_operand(qa0);warpgroup_fence_operand(qa1);\n','')
tail=(HERE.parent/'m128_three.cu').read_text().split('__global__ void prepare(',1)[1].split('at::Tensor sol_forward',1)[0]
tail='__global__ void prepare('+tail
tail=tail.replace('if(threadIdx.x==0){*valid=yes;*fix=0;}','if(threadIdx.x==0)*valid=yes;for(int x=threadIdx.x;x<6*(L/Rows)*4;x+=256)fix[x]=0;')
body+=tail+(HERE/'fused_host.cuh').read_text()
if cluster:
 from cluster_heads import transform
 body=transform(body)
body=body.replace('triattn_projected_fused','triattn_projected_fused_'+tag).replace('projected_forward_cuda','projected_'+tag+'_forward_cuda').replace('projected_uniform_cuda','projected_'+tag+'_uniform_cuda')
(HERE/('fused_'+tag+'.cu')).write_text(body)
uniform=projection[:projection.index('__global__')]
uniform=uniform.replace('struct Params {TX x;TW w;TWKV wkv;TO q,k,v;GX xs;GW ws,wkvs;GO os;int L;};',
 'struct Params {TX x;TWKV wkv;TO out;GX xs;GW wkvs;GO os;int L;int const* valid;};')
uniform=uniform.replace('triattn_projected_stsm','triattn_projected_uniform')+(HERE/'uniform_body.cuh').read_text()
uniform=uniform.replace('triattn_projected_uniform','triattn_projected_uniform_'+tag).replace('projected_uniform_cuda','projected_'+tag+'_uniform_cuda')
(HERE/('uniform_'+tag+'.cu')).write_text(uniform)
prepare=(HERE/'prepare.cu').read_text().replace('triattn_projected_prepare','triattn_projected_prepare_'+tag).replace('projected_prepare_cuda','projected_'+tag+'_prepare_cuda')
(HERE/('prepare_'+tag+'.cu')).write_text(prepare)
binding='''#include <torch/extension.h>
void PREP(at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,double,bool);
at::Tensor FORWARD(at::Tensor,at::Tensor,at::Tensor,at::Tensor,c10::optional<at::Tensor>,double);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("prepare",&PREP);m.def("forward",&FORWARD);}
'''.replace('PREP','projected_'+tag+'_prepare_cuda').replace('FORWARD','projected_'+tag+'_forward_cuda')
(HERE/('binding_'+tag+'.cpp')).write_text(binding)
os.environ['MAX_JOBS']='2'
sources=['binding_'+tag+'.cpp','prepare_'+tag+'.cu','uniform_'+tag+'.cu','fused_'+tag+'.cu']
m=load(name=name,sources=[str(HERE/n) for n in sources],
 extra_include_paths=[str(ROOT/'oc/opt_core/kernels/triattn_surround_tma'),str(ROOT.parent/'anthropic_adoption_20260919/cutlass-4.2/include')],
 extra_cflags=['-O3','-std=c++17'],
 extra_cuda_cflags=['-O3','-std=c++17','-gencode=arch=compute_90a,code=sm_90a','--expt-relaxed-constexpr','--expt-extended-lambda','-lineinfo','--ptxas-options=-v'],
 build_directory=str(build),verbose=True)
shutil.copy2(m.__file__,HERE/(name+'.so'))
print('BUILT',name,'producer',preg,'consumer',creg,flush=True)
