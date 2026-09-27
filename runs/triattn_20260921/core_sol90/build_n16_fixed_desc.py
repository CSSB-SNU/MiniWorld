"""Use one K/V descriptor base and exact constant column/stage offsets."""
from pathlib import Path
import argparse,os
from torch.utils.cpp_extension import load
HERE=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--scores',type=int,choices=(2,4),default=2);ap.add_argument('--q-fragments',type=int,choices=(2,3,4),default=4);args=ap.parse_args()
old='m128n16r4s%dp2qrf40'%args.scores;name=old+'fd'
if args.q_fragments!=4:name+='q%d'%args.q_fragments
s=(HERE/(old+'.cu')).read_text().replace(old,name)
a='''  auto kall=make_tensor(make_smem_ptr(s.k[st][wg]),SK{});
  auto ks=local_tile(kall,Shape<Int<N>,_32>{},make_coord(cx,0));auto kb=tq.partition_fragment_B(ks);'''
b='''  auto kall=make_tensor(make_smem_ptr(s.k[0][wg]),SK{});
  auto ks=local_tile(kall,Shape<Int<N>,_32>{},make_coord(0,0));auto kb0=tq.partition_fragment_B(ks);
  auto kdesc=kb0.data().desc_;
  kdesc.reg32_[0]+=uint32_t((st*Rows*LN*D+cx*N*D)/8);
  auto kb=make_tensor(GMMA::DescriptorIterator{kdesc},kb0.layout());'''
assert s.count(a)==1;s=s.replace(a,b)
a='''  auto vall=make_tensor(make_smem_ptr(s.v[(seq/(Ratio*H))%Stages][wg]),SV{});
  auto vs=local_tile(vall,Shape<_32,Int<N>>{},make_coord(0,(seq/H)%Ratio));PV32 pv32;auto raw=pv32.get_slice(0).partition_fragment_B(vs);
  auto desc=raw.data().desc_;
  uint32_t base=cute::cast_smem_ptr_to_uint(s.v[(seq/(Ratio*H))%Stages][wg]+((seq/H)%Ratio)*N*D);
  desc.bitfield.leading_byte_offset_=(cute::cast_smem_ptr_to_uint(s.ones)-base)>>4;'''
b='''  auto vall=make_tensor(make_smem_ptr(s.v[0][wg]),SV{});
  auto vs=local_tile(vall,Shape<_32,Int<N>>{},make_coord(0,0));PV32 pv32;auto raw=pv32.get_slice(0).partition_fragment_B(vs);
  auto desc=raw.data().desc_;
  uint32_t base=cute::cast_smem_ptr_to_uint(s.v[0][wg]);
  desc.bitfield.leading_byte_offset_=(cute::cast_smem_ptr_to_uint(s.ones)-base)>>4;
  uint32_t delta=uint32_t((((seq/(Ratio*H))%Stages)*Rows*LN*D+((seq/H)%Ratio)*N*D)/8);
  desc.reg32_[0]+=delta-(delta<<16);'''
assert s.count(a)==2;s=s.replace(a,b)
if args.q_fragments!=4:
 a='else gemm(qkr,qr_all[hh](_,_,kk),kb(_,_,kk),score);'
 b='''else {
    if constexpr(hh==0)gemm(qkr,qr_all[hh](_,_,kk),kb(_,_,kk),score);
    else {if(kk<%d)gemm(qkr,qr_all[hh](_,_,kk),kb(_,_,kk),score);else gemm(qk,qa(_,_,kk),kb(_,_,kk),score);}
   }'''%(args.q_fragments-2)
 assert s.count(a)==1;s=s.replace(a,b)
 a='auto qd=ct.retile_D(qr_all[hh]);cute::copy(cp,qs,qd);warpgroup_fence_operand(qr_all[hh]);'
 if args.q_fragments==2:
  b='if(hh==0){'+a+'}'
 else:
  b='auto qd=ct.retile_D(qr_all[hh]);cute::copy(cp,qs,qd);if(hh==0){warpgroup_fence_operand(qr_all[hh]);}else{auto part=qr_all[hh](_,_,Int<0>{});warpgroup_fence_operand(part);}'
 assert s.count(a)==1;s=s.replace(a,b)
source=HERE/(name+'.cu');source.write_text(s)
cpp=HERE/(name+'.cpp');cpp.write_text((HERE/(old+'.cpp')).read_text().replace(old,name))
directory=HERE/('build_'+name);directory.mkdir(exist_ok=True)
os.environ.update(TORCH_CUDA_ARCH_LIST='9.0a',MAX_JOBS='2')
load(name='triattn_sol_'+name,sources=[str(cpp),str(source)],extra_include_paths=[str(HERE.parents[1]/'anthropic_adoption_20260919/cutlass-4.2/include')],extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--expt-relaxed-constexpr','--expt-extended-lambda','--use_fast_math','-lineinfo','-Xptxas=-v','--keep','--keep-dir='+str(directory)],extra_ldflags=['-lcuda'],build_directory=str(directory),verbose=True)
print('BUILT',name,flush=True)
