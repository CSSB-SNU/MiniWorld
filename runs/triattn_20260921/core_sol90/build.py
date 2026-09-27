from pathlib import Path
import os,sys
from torch.utils.cpp_extension import load
HERE=Path(__file__).resolve().parent
CONFIGS={'tile5n32bs':(5,32,128,1,88,24),'tile5n32bcs':(5,32,128,1,88,24),'tile4n32bu':(4,32,128,1,112,32),'tile4n32bcu':(4,32,128,1,112,32),'tile5n32bu':(5,32,128,1,88,24),'tile5n32bcu':(5,32,128,1,88,24),'tile5n32b2l':(5,32,128,1,88,24),'tile5n32bcl':(5,32,128,1,88,24),'tile5n32bc':(5,32,128,1,96,24),'tile5n32b2':(5,32,128,1,96,24),'tile5n32bp':(5,32,128,1,88,24),'tile3n128ring':(3,128,128,1,160,32),'tile5n32b4':(5,32,128,1,88,24),'tile5n32b1':(5,32,128,1,88,24),'tile3n128':(3,128,128,1,160,32),'tile2n128':(2,128,128,1,224,24),'tile5n32gp':(5,32,128,1,88,24),'tile5n32gd':(5,32,128,1,88,24),'tile5n32x64':(5,32,128,1,88,24),'tile1n64static':(1,64,32,4,0,24),'tile5n32':(5,32,128,1,88,24),'tile1n64':(1,64,32,4,112,24),'r6n32v3':(6,32,128,1,80,24),'r6n32v4':(6,32,128,1,80,24),'r6n32v5':(6,32,128,1,80,24),'r5n32v5':(5,32,128,1,88,24),'r4n32v5':(4,32,128,1,112,32),'r6n32v6':(6,32,128,1,80,24),'r4n32v7':(4,32,128,1,112,32),'r5n32v7':(5,32,128,1,88,24),'r4n32v9':(4,32,128,1,112,32),'r3n32v9':(3,32,128,1,160,32),'split4':(4,32,128,1,112,32),'r6n32v2':(6,32,128,1,80,24),'r4n32v2':(4,32,128,1,112,32),'r2n64':(2,64,32,2,0,24),'r6n32':(6,32,128,1,80,24),'r5n32':(5,32,128,1,88,24),'r4n32':(4,32,128,1,112,32)}
def build(variant=None):
 CONFIGS['m128s2p3qrdepfull']=(3,32,128,1,160,32)
 CONFIGS['m128s2p3qrdepsyncfull']=(3,32,128,1,160,32)
 CONFIGS['m128s2p3qrn64']=(2,64,128,1,224,32)
 CONFIGS['m64s2p3qrr4']=(4,32,128,1,112,32)
 CONFIGS['m128s2p3qrfull']=(3,32,128,1,160,32)
 CONFIGS['m128s2p3qrlatefull']=(3,32,128,1,160,32)
 CONFIGS['m128s2p3sslate']=(3,32,128,1,160,32)
 CONFIGS['m128s2p3qrlate']=(3,32,128,1,160,32)
 CONFIGS['m128s2p3ss']=(3,32,128,1,160,32)
 CONFIGS['m128s2p3qr']=(3,32,128,1,160,32)
 CONFIGS['persistkv2gb']=(2,32,128,1,224,32)
 CONFIGS['persistkv2cp']=(2,32,128,1,224,32)
 CONFIGS['persistkv2b4']=(2,32,128,1,224,32)
 CONFIGS['persistkv2']=(2,32,128,1,224,32)
 CONFIGS['warp6']=(6,32,128,1,80,24)
 CONFIGS['warp5']=(5,32,128,1,88,24)
 CONFIGS['warp6f']=(6,32,128,1,80,24)
 CONFIGS['split4f']=(4,32,128,1,112,32)
 CONFIGS['m128r4']=(4,32,128,1,112,32)
 CONFIGS['m128r4f']=(4,32,128,1,112,32)
 CONFIGS['m128r3']=(3,32,128,1,160,32)
 CONFIGS['m128n16r5']=(5,16,128,1,88,24)
 CONFIGS['m128n16r6']=(6,16,128,1,80,24)
 CONFIGS['m64r6']=(6,32,128,1,80,24)
 CONFIGS['m64r5']=(5,32,128,1,88,24)
 CONFIGS['m64r6p2']=(6,32,128,1,80,24)
 CONFIGS['m128n16r6p2']=(6,16,128,1,80,24)
 CONFIGS['m128n16r6p4']=(6,16,128,1,80,24)
 CONFIGS['m128three']=(3,32,128,1,160,32)
 CONFIGS['m128threeqr']=(3,32,128,1,160,32)
 CONFIGS['m128threeflat']=(3,32,128,1,160,32)
 CONFIGS['m128threeqrflat']=(3,32,128,1,160,32)
 CONFIGS['m128three768']=(3,32,128,1,160,32)
 CONFIGS['m128threeqr768']=(3,32,128,1,160,32)
 CONFIGS['m64f40r4']=(4,32,128,1,112,32)
 CONFIGS['m64f40r5']=(5,32,128,1,88,24)
 CONFIGS['m128wide2']=(2,64,128,1,224,32)
 CONFIGS['m128wide240']=(2,64,128,1,240,24)
 CONFIGS['m128widefull']=(2,64,128,1,224,32)
 CONFIGS['m128scaled3']=(3,32,128,1,160,32)
 CONFIGS['m64pref5']=(5,32,128,1,88,24)
 CONFIGS['m64scaled5']=(5,32,128,1,88,24)
 CONFIGS['m128scaled3drain']=(3,32,128,1,160,32)
 CONFIGS['m64n16twocta']=(2,16,128,2,104,32)
 CONFIGS['m64n16twoctaqr']=(2,16,128,2,104,32)
 CONFIGS['m64twocta']=(2,32,128,2,104,32)
 CONFIGS['m64twoctass']=(2,32,128,2,104,32)
 CONFIGS['m128p3qr']=(3,32,128,1,160,32)
 CONFIGS['m128p3ss']=(3,32,128,1,160,32)
 CONFIGS['m128p3qr24']=(3,32,128,1,160,32)
 CONFIGS['m128p3ss24']=(3,32,128,1,160,32)
 CONFIGS['m128p3ss24row']=(3,32,128,1,160,32)
 CONFIGS['m128p3ss24split']=(3,32,128,1,160,32)
 CONFIGS['m64p3ss24r4']=(4,32,128,1,112,32)
 CONFIGS['m256ss2pref']=(2,32,128,1,224,32)
 CONFIGS['m256qr2pref']=(2,32,128,1,224,32)
 CONFIGS['m256p2ss32r3']=(3,32,128,1,160,32)
 CONFIGS['m256p2ss32ip']=(3,32,128,1,160,32)
 CONFIGS['m128p4ss24']=(3,32,128,1,160,32)
 CONFIGS['m64p4ss24s3']=(3,32,128,1,160,32)
 CONFIGS['m64p2ss8r4']=(4,32,128,1,112,32)
 CONFIGS['m128n64s1p2']=(3,64,128,1,160,32)
 CONFIGS['m128n64s1p1']=(3,64,128,1,160,32)
 variant=variant or os.environ.get('SOL_VARIANT','r6n32')
 artifact=variant
 extra_ptxas=[]
 if variant.endswith('_ru0'):
  variant=variant[:-4]
  extra_ptxas=['-Xptxas=--register-usage-level=0']
 if variant.endswith('_c134'):
  variant=variant[:-5]
  from toolchain import use_ptxas
  use_ptxas()
 rows,n,producer,residency,regs,pregs=CONFIGS[variant]
 ns='ta_sol_'+artifact
 defines={'SOL_NAMESPACE':ns,'SOL_ROWS':rows,'SOL_N':n,'SOL_PRODUCER_THREADS':producer,'SOL_RESIDENCY':residency,'SOL_REG_CONSUMER':regs,'SOL_REG_PRODUCER':pregs}
 template='overlap_'+variant[-2:]+'.cu' if variant[-2:] in ('v2','v3','v4','v5','v6','v7','v9') else 'overlap_template.cu'
 text=(HERE/template).read_text()
 if variant=='m64s2p3qrr4':
  from two_score_three_p import transform
  text=transform((HERE/'m128_three_qr.cu').read_text())
  from m64_probability import transform
  text=transform(text)
 if variant.startswith('m128s2p3'):
  from two_score_three_p import transform,pack_late,force_p_before_qk
  text=transform((HERE/('m128_three_qr.cu' if 'qr' in variant else 'm128_three.cu')).read_text())
  if 'late' in variant:text=pack_late(text)
  if 'dep' in variant:text=force_p_before_qk(text,sync='depsync' in variant)
  if variant.endswith('full'):
   from generic_full_producers import transform
   text=transform(text)
 if variant in ('persistkv2','persistkv2b4','persistkv2gb','persistkv2cp'):
  from persistent_kv import transform,alias_q,global_bias
  text=transform((HERE/'m128_three_qr.cu').read_text())
  if variant=='persistkv2b4':text=alias_q(text)
  if variant in ('persistkv2gb','persistkv2cp'):text=global_bias(text,copy_async=variant.endswith('cp'))
 if variant in ('m128n64s1p2','m128n64s1p1'):
  from one_score_n64 import transform
  text=transform((HERE/'m128_three.cu').read_text())
  if variant.endswith('p1'):
   text=text.replace('auto pp1=make_fragment_like(pp);','').replace('warpgroup_fence_operand(pp1);','').replace('clear(pp1);','')
   text=text.replace('auto& prob=hh==0?pp:pp1;','auto& prob=pp;')
   text=text.replace('warpgroup_wait<1>();warpgroup_fence_operand(score);warpgroup_fence_operand(prob);', 'warpgroup_wait<1>();warpgroup_fence_operand(score);')
   text=text.replace('   pack(score,prob);init_score(score,seq+1);','   warpgroup_wait<0>();warpgroup_fence_operand(prob);\n   pack(score,prob);init_score(score,seq+1);')
 if variant=='m64p2ss8r4':
  from two_probability_m64 import transform
  text=transform((HERE/'m128_three.cu').read_text())
 if variant in ('m128p4ss24','m64p4ss24s3'):
  from four_probability import transform
  text=transform((HERE/'m128_three.cu').read_text())
  if variant=='m64p4ss24s3':
   from m64_probability import transform
   text=transform(text).replace('Stages=2','Stages=3')
 if variant in ('m256ss2pref','m256qr2pref'):
  from m256_two_consumers import transform
  text=transform((HERE/'m128_three.cu').read_text(),qr='qr' in variant)
 if variant in ('m256p2ss32r3','m256p2ss32ip'):
  from four_query_halves import transform
  text=transform((HERE/'m128_three.cu').read_text())
  if variant.endswith('ip'):
   from inplace_exponentials import transform
   text=transform(text)
 if variant in ('m128p3qr','m128p3ss','m128p3qr24','m128p3ss24','m128p3ss24row','m128p3ss24split','m64p3ss24r4'):
  from three_probability import transform
  text=transform((HERE/('m128_three_qr.cu' if 'qr' in variant else 'm128_three.cu')).read_text(),period24='24' in variant)
  if variant.endswith(('row','split','r4')):
   from independent_rows import transform
   text=transform(text,split=variant.endswith(('split','r4')))
  if variant=='m64p3ss24r4':
   from m64_probability import transform
   text=transform(text)
 if variant in ('m64twocta','m64twoctass','m64n16twocta','m64n16twoctaqr'):
  from make_twocta import transform
  text=transform((HERE/'occupancy_b4_two.cu').read_text())
  if variant=='m64n16twoctaqr':
   text=text.replace('using QK=decltype(make_tiled_mma(GMMA::ss_op_selector','using QK=decltype(make_tiled_mma(GMMA::rs_op_selector')
   old='auto qa=tq.partition_fragment_A(sq);'
   assert text.count(old)==1
   text=text.replace(old,'auto qa=qk.get_thread_slice(t).partition_fragment_A(sq);')
   pos=text.index(' auto init_score=')
   text=text[:pos]+' auto qcopy=make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N,Element>{},qk);\n auto qs=qcopy.get_thread_slice(t).partition_S(sq);\n auto qd=qcopy.get_thread_slice(t).retile_D(qa);\n copy(qcopy,qs,qd);warpgroup_fence_operand(qa);\n'+text[pos:]
  if variant=='m64twoctass':
   from scaled_ss_global import generic
   text=generic(text)
 if variant in ('m64pref5','m64scaled5'):
  from make_m64_prefetch import transform
  text=transform((HERE/'occupancy_b4_two.cu').read_text(),scaled=variant=='m64scaled5')
 if variant in ('m128scaled3','m128scaled3drain'):
  from make_scaled_three import transform
  text=transform((HERE/'m128_three_qr.cu').read_text())
  if variant=='m128scaled3drain':text=text.replace('step(seq+5,_1{},_1{});','step(seq+5,_1{},_1{});drain();')
 if variant in ('m128wide2','m128wide240','m128widefull'):
  from make_wide import transform
  text=transform((HERE/'m128_three.cu').read_text())
  if variant=='m128widefull':text=text.replace('#pragma unroll 1\n  for(;seq+5<2*nk;seq+=6)', '#pragma unroll\n  for(;seq+5<2*nk;seq+=6)')
 if variant in ('m64f40r4','m64f40r5'):
  from fuse_m64 import transform
  text=transform((HERE/'occupancy_b4_two.cu').read_text(),rows)
 if variant in ('m128r4','m128r4f','m128r3'):text=(HERE/'m128_pipe.cu').read_text()
 if variant in ('m128n16r5','m128n16r6'):text=(HERE/'m128_two.cu').read_text()
 if variant in ('m64r6','m64r5'):text=(HERE/'m64_pipe.cu').read_text()
 if variant in ('m128three','m128threeqr'):text=(HERE/('m128_three_qr.cu' if variant.endswith('qr') else 'm128_three.cu')).read_text()
 if variant in ('m128three768','m128threeqr768'):
  text=(HERE/('m128_three_qr.cu' if 'qr' in variant else 'm128_three.cu')).read_text()
  text=text.replace('p.L','768').replace('p.scale','0x1.6a09e6p-3f')
  text=text.replace('using namespace SOL_NAMESPACE;int L=q.size(-2);', 'using namespace SOL_NAMESPACE;int L=q.size(-2);TORCH_CHECK(L==768 && float(scale)==0x1.6a09e6p-3f,"L768 standard scale required");')
 if variant in ('m128threeflat','m128threeqrflat'):
  text=(HERE/('m128_three_qr.cu' if 'qr' in variant else 'm128_three.cu')).read_text()
  idx=text.index('using SB=')
  text=text[:idx]+'''using SKR=decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Element>{},Shape<Int<LN>,_32,Int<Stages*Rows>>{}));
using SVR=decltype(composition(SKR{},make_ordered_layout(Shape<_32,Int<LN>,Int<Stages*Rows>>{},Step<_2,_1,_3>{})));
'''+text[idx:]
  idx=text.index(' auto init_score=')
  text=text[:idx]+''' auto kr=make_tensor(make_smem_ptr(s.k[0][wg]),SKR{});
 auto vr=make_tensor(make_smem_ptr(s.v[0][wg]),SVR{});
 auto kdesc=tq.partition_fragment_B(local_tile(kr,Shape<Int<N>,_32>{},make_coord(_,0,_)));
 auto vdesc=tp.partition_fragment_B(local_tile(vr,Shape<_32,Int<N>>{},make_coord(0,_,_)));
'''+text[idx:]
  text=text.replace('  auto kall=make_tensor(make_smem_ptr(s.k[st][wg]),SK{});\n  auto ks=local_tile(kall,Shape<_32,_32>{},make_coord(cx,0));auto kb=tq.partition_fragment_B(ks);','  auto kb=kdesc(_,_,_,cx,st*Rows);')
  text=text.replace('  auto vall=make_tensor(make_smem_ptr(s.v[(seq/(2*Ratio))%Stages][wg]),SV{});\n  auto vs=local_tile(vall,Shape<_32,_32>{},make_coord(0,(seq/2)%Ratio));auto vb=tp.partition_fragment_B(vs);','  auto vb=vdesc(_,_,_,(seq/2)%Ratio,((seq/(2*Ratio))%Stages)*Rows);')
 if variant in ('m64r6p2','m128n16r6p2'):
  text=(HERE/('m64_pipe.cu' if variant=='m64r6p2' else 'm128_two.cu')).read_text()
  a=text.index('  #pragma unroll 1\n  for(;seq+7<')
  b=text.index('  #pragma unroll 1\n  for(;seq<',a)
  text=text[:a]+text[b:]
 if variant=='m128n16r6p4':
  text=(HERE/'m128_two.cu').read_text()
  text=text.replace('seq+7<2*nk;seq+=8','seq+3<2*nk;seq+=4')
  text=text.replace('   step(seq+4,_0{},cute::false_type{});step(seq+5,_1{},cute::false_type{});\n   step(seq+6,_0{},cute::false_type{});step(seq+7,_1{},cute::false_type{});drain();','   drain();')
 if variant in ('warp6','warp5','warp6f'):
  text=(HERE/'warp_mma.cu').read_text()
  if variant=='warp6f':
   text=text.replace('SQ{}(make_coord(', 'as_position_independent_swizzle_layout(SQ{})(make_coord(')
   text=text.replace('SK{}(make_coord(', 'as_position_independent_swizzle_layout(SK{})(make_coord(')
   text=text.replace('extern __shared__ __align__(128)', 'extern __shared__ __align__(1024)')
 if variant=='tile5n32b2l':text=(HERE/'occupancy_b4_two.cu').read_text()
 if variant=='tile5n32bcl':text=(HERE/'occupancy_b4_cluster.cu').read_text()
 if variant=='tile5n32bc':text=(HERE/'occupancy_b4_cluster.cu').read_text()
 if variant=='tile5n32b2':text=(HERE/'occupancy_b4_two.cu').read_text()
 if variant=='tile5n32bp':text=(HERE/'occupancy_b4_pipe.cu').read_text()
 if variant=='tile3n128ring':text=(HERE/'occupancy_ring.cu').read_text()
 if variant=='tile5n32b4':text=(HERE/'occupancy_b4.cu').read_text()
 if variant=='tile5n32b1':text=(HERE/'occupancy_b1.cu').read_text()
 if variant in ('tile3n128','tile2n128'):text=(HERE/'occupancy128.cu').read_text()
 if variant=='tile5n32gp':text=(HERE/'occupancy_global_pipe.cu').read_text()
 if variant=='tile5n32gd':text=(HERE/'occupancy_global.cu').read_text()
 if variant=='tile5n32x64':text=(HERE/'occupancy128.cu').read_text().replace('LN=128','LN=64')
 if variant=='tile1n64static':text=(HERE/'occupancy.cu').read_text()
 if variant=='tile5n32':text=(HERE/'occupancy128.cu').read_text()
 if variant=='tile1n64':text=(HERE/'occupancy.cu').read_text()
 if variant in ('split4','split4f'):
  text=(HERE/'overlap_v7.cu').read_text()
  text=text.replace('at::Tensor sol_forward(', (HERE/'specialized.cuh').read_text()+'\nat::Tensor sol_forward(')
  text=text.replace('cudaFuncSetAttribute(attention<false>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared))', 'cudaFuncSetAttribute(specialized::attention,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(specialized::Shared))')
  text=text.replace('attention<false><<<ct,Threads,sizeof(Shared),stream>>>(p);','specialized::attention<<<ct,specialized::Threads,sizeof(specialized::Shared),stream>>>(p);')
  if variant=='split4f':
   text=text.replace('SP{}(q,k)', 'as_position_independent_swizzle_layout(SP{})(q,k)')
   text=text.replace('extern __shared__ __align__(128)', 'extern __shared__ __align__(1024)')
 if variant in ('tile5n32bu','tile5n32bcu','tile4n32bu','tile4n32bcu'):
  text=(HERE/('occupancy_b4_cluster.cu' if variant.endswith('bcu') else 'occupancy_b4_two.cu')).read_text()
  text=text.replace('int wg=tid/128,t=tid%128;','int wg=int(__reduce_max_sync(0xffffffffu,unsigned(tid)/128u)),t=tid%128;')
  text=text.replace('auto tq=qk.get_slice(t);auto tp=pv.get_slice(t);auto tl=ls.get_slice(t);','auto tq=qk.get_slice(0);auto tp=pv.get_slice(0);auto tl=ls.get_slice(0);')
  text=text.replace('auto id=tp.partition_C(','auto id=pv.get_slice(t).partition_C(')
 if variant in ('tile5n32bs','tile5n32bcs'):
  text=(HERE/('occupancy_b4_cluster.cu' if variant.endswith('bcs') else 'occupancy_b4_two.cu')).read_text()
  text=text.replace('int wg=tid/128,t=tid%128;','int wg=int(__reduce_max_sync(0xffffffffu,unsigned(tid)/128u)),t=tid%128;')
  text=text.replace('auto tq=qk.get_slice(t);auto tp=pv.get_slice(t);auto tl=ls.get_slice(t);','auto tq=qk.get_slice(0);auto tp=pv.get_slice(0);auto tl=ls.get_slice(0);')
  text=text.replace('auto id=tp.partition_C(','auto id=pv.get_slice(t).partition_C(')
  text=text.replace('clear(acc);clear(sum);','clear(acc);if constexpr(Safe)clear(sum);')
  text=text.replace(' bool missing_seed=false;', ' float partial[2]={0.f,0.f};\n bool missing_seed=false;')
  text=text.replace('warpgroup_fence_operand(sum);','if constexpr(Safe)warpgroup_fence_operand(sum);')
  text=text.replace('  #pragma unroll\n  for(int kk=0;kk<N/16;kk++)gemm(ls,pp(_,_,kk),lb(_,_,kk),sum);','  if constexpr(Safe){\n  #pragma unroll\n  for(int kk=0;kk<N/16;kk++)gemm(ls,pp(_,_,kk),lb(_,_,kk),sum);\n  }')
  mark='\n auto pack='
  start=text.index(' auto exponentiate=');end=text.index(' auto issue_pv=',start)
  exp=text[start:end]
  exp=exp.replace('    else sr(row,col)=ex2(fmaf(x,c,-mx[row]));','    else {sr(row,col)=ex2(fmaf(x,c,-mx[row]));partial[row]+=sr(row,col);}')
  text=text[:start]+exp+text[end:]
  text=text.replace('  bool bad=missing_seed;','  for(int row=0;row<2;row++){float val=partial[row];val+=__shfl_xor_sync(0xffffffffu,val,1);val+=__shfl_xor_sync(0xffffffffu,val,2);lr(row,0)=val;lr(row,1)=val;}\n  bool bad=missing_seed;')
 variant=artifact
 for key in sorted(defines,key=len,reverse=True):text=text.replace(key,str(defines[key]))
 text=text.replace('sol_forward','sol_forward_'+variant)
 src=HERE/(variant+'.cu')
 if not src.exists() or src.read_text()!=text:src.write_text(text)
 cpp=HERE/(variant+'.cpp')
 text='#include <torch/extension.h>\nat::Tensor sol_forward_'+variant+'(at::Tensor,at::Tensor,at::Tensor,at::Tensor,c10::optional<at::Tensor>,double);\nPYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("forward",&sol_forward_'+variant+');}\n'
 if not cpp.exists() or cpp.read_text()!=text:cpp.write_text(text)
 directory=HERE/('build_'+variant);directory.mkdir(exist_ok=True)
 os.environ['TORCH_CUDA_ARCH_LIST']='9.0a';os.environ['MAX_JOBS']='2'
 return load(name='triattn_sol_'+variant,sources=[str(cpp),str(src)],extra_include_paths=[str(HERE.parents[1]/'anthropic_adoption_20260919/cutlass-4.2/include')],extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--expt-relaxed-constexpr','--expt-extended-lambda','--use_fast_math','-lineinfo','-Xptxas=-v']+extra_ptxas,extra_ldflags=['-lcuda'],build_directory=str(directory),verbose=True)
if __name__=='__main__':build(sys.argv[1] if len(sys.argv)>1 else None)
