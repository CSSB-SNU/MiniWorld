"""N48 / M128 / R3, two scores, full Q RS, N40 PV and denominator.

Distinct from R2/N64: retain three consumer WGs, reduce QK instruction
frequency, and fuse the denominator into PV. The two/three-P variants
trade a second wait for 12 registers. This is an isolated numeric experiment.
"""
from pathlib import Path
import hashlib, os, sys
from torch.utils.cpp_extension import load

HERE = Path(__file__).resolve().parent
slots = int(sys.argv[1]); assert slots in (2, 3)
installed = HERE.parent/'oc/opt_core/kernels/triattn_core_broadcast/triattn_broadcast.so'
assert hashlib.sha256(installed.read_bytes()).hexdigest() == '97634810cdba1e91fe4aecb8bdfa610dd050dd39e7504ffb8babee8c46d4077b'
parent = 'm128s2p%dqrn64' % slots
variant = 'm128n48p%df40' % slots
s = (HERE/(parent+'.cu')).read_text().replace(parent,variant)
s = s.replace('M=128,N=64,LN=128,Ratio=LN/N,D=32,Rows=2,Stages=2',
              'M=128,N=48,LN=96,Ratio=LN/N,D=32,Rows=3,Stages=2')
s = s.replace('if constexpr(224>0)', 'if constexpr(160>0)').replace('warpgroup_reg_alloc<224>()','warpgroup_reg_alloc<160>()')
old = 'using PV=decltype(make_tiled_mma(SM90_64x32x16_F32BF16BF16_RS<GMMA::Major::K,GMMA::Major::MN>{}));'
assert s.count(old) == 1
s = s.replace(old,old.replace('using PV=', 'using PVBase=')+'\nusing PV=decltype(make_tiled_mma(SM90::GMMA::MMA_64x40x16_F32BF16BF16_RS<GMMA::Major::K,GMMA::Major::MN>{}));')
s = s.replace('v[Stages][Rows][LN*D],ones[8*N];', 'v[Stages][Rows][LN*D];\n alignas(1024) Element ones[N*D];')
s = s.replace('for(int x=tid;x<8*N;x+=Threads)', 'for(int x=tid;x<N*D;x+=Threads)')
s = s.replace('QK qk;PV pv;LS ls;auto tq=qk.get_slice(0);auto tp=pv.get_slice(0);',
              'QK qk;PV pv;PVBase pvbase;LS ls;auto tq=qk.get_slice(0);auto tp=pvbase.get_slice(0);')
s = s.replace('using Output=decltype(partition_fragment_C(pv,Shape<_64,_32>{}));',
              'using Output=decltype(partition_fragment_C(pv,Shape<_64,_40>{}));')
s = s.replace('Den den[2];clear(den[0]);clear(den[1]);',
              'auto d0=make_tensor(acc[0].data()+16,Den{}.layout());\n decltype(d0) den[2]={d0,make_tensor(acc[1].data()+16,Den{}.layout())};')
s = s.replace(' auto lb=tl.partition_fragment_B(make_tensor(make_smem_ptr(s.ones),S1{}));\n','')
old = 'auto vs=local_tile(vall,Shape<_32,Int<N>>{},make_coord(0,(seq/2)%Ratio));auto vb=tp.partition_fragment_B(vs);'
assert s.count(old) == 2
s = s.replace(old,'''auto vs=local_tile(vall,Shape<_32,Int<N>>{},make_coord(0,(seq/2)%Ratio));
  auto raw=tp.partition_fragment_B(vs);auto desc=raw.data().desc_;
  uint32_t va=cast_smem_ptr_to_uint(s.v[(seq/(2*Ratio))%Stages][wg])+uint32_t((seq/2)%Ratio)*N*D*sizeof(Element);
  uint32_t oa=cast_smem_ptr_to_uint(s.ones);
  desc.bitfield.leading_byte_offset_=(oa-va)>>4;
  auto vb=make_tensor(GMMA::DescriptorIterator{desc},raw.layout());''')
old = '  #pragma unroll\n  for(int kk=0;kk<N/16;kk++)gemm(ls,pp(_,_,kk),lb(_,_,kk),den[hh]);\n'
assert s.count(old) == 2
s = s.replace(old,'')
s = s.replace('auto id=pv.get_slice(t).partition_C(make_identity_tensor(Shape<_64,_32>{}));',
              'auto id=pv.get_slice(t).partition_C(make_identity_tensor(Shape<_64,_40>{}));')
s = s.replace('for(int n=0;n<size(acc[hh]);n+=2)', 'for(int n=0;n<16;n+=2)')
# The existing helper deliberately overlaps both QK and PV. Make the P
# register fence explicit rather than relying on ptxas adding C7519 fences.
needle='  if constexpr(Safe){warpgroup_fence_operand(acc[hh]);warpgroup_fence_operand(den[hh]);}\n  #pragma unroll'
assert s.count(needle) == 1
s = s.replace(needle,'  warpgroup_fence_operand(pp);warpgroup_arrive();\n'+needle)
if slots == 3:
    a = s.index('  #pragma unroll 1\n  for(int seq=0;seq<2*nk;seq+=24)')
    b = s.index('\n }\n\n auto id=',a)
    # L768 has 32 chunks. A 24-step ring period followed by exactly eight
    # steps avoids phantom PVs/stores beyond the end of the stream.
    steps=[]
    for seq in range(32):
        steps.append('  step(%d,Int<%d>{},Int<%d>{},cute::%s_type{});' % (seq,seq%2,seq%3,'true' if seq<2 else 'false'))
        if seq in (23,31): steps.append('  drain();')
    steps.append('  issue_pv(2*nk-1,_1{},pp1);drain();release(2*nk-1);')
    s=s[:a]+'\n'.join(steps)+s[b:]
s=s.replace('struct Shared {','struct Shared {',1)
needle='__device__ __forceinline__ float ex2'
s=s.replace(needle,'static_assert(sizeof(Shared)<=232448,"N48 shared memory exceeds H100 limit");\n'+needle)
src=HERE/(variant+'.cu');src.write_text(s)
cpp=HERE/(variant+'.cpp');cpp.write_text((HERE/(parent+'.cpp')).read_text().replace(parent,variant))
directory=HERE/('build_'+variant);directory.mkdir(exist_ok=True)
os.environ['TORCH_CUDA_ARCH_LIST']='9.0a';os.environ['MAX_JOBS']='2'
load(name='triattn_sol_'+variant,sources=[str(cpp),str(src)],
     extra_include_paths=[str(HERE.parents[1]/'anthropic_adoption_20260919/cutlass-4.2/include')],
     extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--expt-relaxed-constexpr','--expt-extended-lambda','--use_fast_math','-lineinfo','-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED','-Xptxas=-v'],
     extra_ldflags=['-lcuda'],build_directory=str(directory),verbose=True)
print('BUILT',variant,flush=True)
