"""M64/R2 two-score, two-P pipeline with two resident CTAs and N40 PV.

Starts from the correctness-tested M64/R4 two-P prototype, not its older
three-score two-CTA variant. Four bias slots and constant chunk positions
reduce shared storage and address state; full Q is cached in eight registers.
"""
from pathlib import Path
import hashlib, os
from torch.utils.cpp_extension import load

HERE = Path(__file__).resolve().parent
variant = 'm64r2s2p2qrf40'
installed = HERE.parent/'oc/opt_core/kernels/triattn_core_broadcast/triattn_broadcast.so'
assert hashlib.sha256(installed.read_bytes()).hexdigest() == '97634810cdba1e91fe4aecb8bdfa610dd050dd39e7504ffb8babee8c46d4077b'
s = (HERE/'m64p2ss8r4.cu').read_text().replace('m64p2ss8r4', variant)
s = '#define CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED\n'+s
s = s.replace('Rows=4,Stages=2', 'Rows=2,Stages=2')
s = s.replace('__launch_bounds__(Threads,1)', '__launch_bounds__(Threads,2)')
s = s.replace('112>0', '104>0').replace('warpgroup_reg_alloc<112>', 'warpgroup_reg_alloc<104>')
s = s.replace('nk=p.L/N', 'nk=768/N')
s = s.replace('ones[8*N]', 'ones[N*D]').replace('x<8*N', 'x<N*D')
s = s.replace('float bias[8][64*N]', 'float bias[4][64*N]')
s = s.replace('bf[4]', 'bf[2]').replace('be[8]', 'be[4]')
old = 'for(int st=0;st<8;st++){if(st<4)s.bf[st].init(2);s.be[st].init(Rows*4);}'
assert s.count(old) == 1
s = s.replace(old, 'for(int st=0;st<4;st++){if(st<2)s.bf[st].init(2);s.be[st].init(Rows*4);}')
s = s.replace('seq%8', 'seq%4').replace('seq/8', 'seq/4').replace('seq>=8', 'seq>=4')
old = 'using PV=decltype(make_tiled_mma(SM90_64x32x16_F32BF16BF16_RS<GMMA::Major::K,GMMA::Major::MN>{}));'
assert s.count(old) == 1
s = s.replace(old, old.replace('using PV=', 'using PV32=')+'\nusing PV=decltype(make_tiled_mma(SM90::GMMA::MMA_64x40x16_F32BF16BF16_RS<GMMA::Major::K,GMMA::Major::MN>{}));\nusing QKr=decltype(make_tiled_mma(GMMA::rs_op_selector<Element,Element,float,Shape<_64,Int<N>,_32>>()));')
s = s.replace('partition_fragment_C(pv,Shape<_64,_32>{})', 'partition_fragment_C(pv,Shape<_64,_40>{})')
old = ' Den den[1];clear(den[0]);'
assert s.count(old) == 1
s = s.replace(old, ' auto den0=make_tensor(acc[0].data()+16,Den{}.layout());\n decltype(den0) den[1]={den0};')
s = s.replace('auto id=pv.get_slice(t).partition_C(make_identity_tensor(Shape<_64,_32>{}));', 'auto id=pv.get_slice(t).partition_C(make_identity_tensor(Shape<_64,_40>{}));')
s = s.replace('for(int n=0;n<size(acc[hh]);n+=2)', 'for(int n=0;n<16;n+=2)')
old = ' s.qr.wait(0);asm volatile("":::"memory");'
assert s.count(old) == 1
s = s.replace(old, old+'''
 QKr qkr;auto qrthr=qkr.get_thread_slice(t);
 auto qshared=make_tensor(make_smem_ptr(s.q[wg]),SQ{});
 auto qr=qrthr.partition_fragment_A(qshared);
 if constexpr(!Safe){
  auto cp=make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N,Element>{},qkr);
  auto ct=cp.get_thread_slice(t);auto qs=ct.partition_S(qshared);auto qd=ct.retile_D(qr);
  cute::copy(cp,qs,qd);warpgroup_fence_operand(qr);
 }
''')
old = '  for(int kk=0;kk<2;kk++)gemm(qk,qa(_,_,kk),kb(_,_,kk),score);'
assert s.count(old) == 1
s = s.replace(old, '''  for(int kk=0;kk<2;kk++){
   if constexpr(Safe)gemm(qk,qa(_,_,kk),kb(_,_,kk),score);
   else gemm(qkr,qr(_,_,kk),kb(_,_,kk),score);
  }''')
old = 'auto vb=tp.partition_fragment_B(vs);'
assert s.count(old) == 2
s = s.replace(old, '''PV32 pv32;auto raw=pv32.get_slice(0).partition_fragment_B(vs);
  auto desc=raw.data().desc_;
  uint32_t base=cute::cast_smem_ptr_to_uint(s.v[(seq/Ratio)%Stages][wg]+(seq%Ratio)*N*D);
  desc.bitfield.leading_byte_offset_=(cute::cast_smem_ptr_to_uint(s.ones)-base)>>4;
  auto vb=make_tensor(GMMA::DescriptorIterator{desc},raw.layout());''')
old = '  #pragma unroll\n  for(int kk=0;kk<N/16;kk++)gemm(ls,pp(_,_,kk),lb(_,_,kk),den[hh]);'
assert s.count(old) == 2
s = s.replace(old, '')

# Keep original native FMA/EX2 arithmetic but explicitly shorten the temporaries.
old = '''   #pragma unroll
   for(int col=0;col<N/4;col++){
    float x=sr(row,col);
    if constexpr(Safe)sr(row,col)=x==-INFINITY?0.f:ex2(fmaf(x,c,-mx[hh][row]));
    else sr(row,col)=ex2(fmaf(x,c,-mx[hh][row]));
   }'''
assert s.count(old) == 1
s = s.replace(old, '''   if constexpr(!Safe){
    #pragma unroll
    for(int col=0;col<N/4;col+=4){
     asm volatile(
      "fma.rn.ftz.f32 %0,%0,%4,%5;\\n"
      "fma.rn.ftz.f32 %1,%1,%4,%5;\\n"
      "fma.rn.ftz.f32 %2,%2,%4,%5;\\n"
      "fma.rn.ftz.f32 %3,%3,%4,%5;\\n"
      "ex2.approx.ftz.f32 %0,%0;\\n"
      "ex2.approx.ftz.f32 %1,%1;\\n"
      "ex2.approx.ftz.f32 %2,%2;\\n"
      "ex2.approx.ftz.f32 %3,%3;\\n"
      : "+f"(sr(row,col)),"+f"(sr(row,col+1)),"+f"(sr(row,col+2)),"+f"(sr(row,col+3))
      : "f"(c),"f"(-mx[hh][row]));
    }
   }else{
'''+old+'\n   }')
a = s.index('  #pragma unroll 1\n  for(int seq=0;seq<nk;seq+=8){')
b = s.index('  drain();release(nk-1);', a)
calls = []
for seq in range(24):
    calls.append('  step(%d,_%d{},cute::%s_type{});' % (seq, seq%2, 'true' if seq == 0 else 'false'))
    if seq%8 == 7:
        calls.append('  drain();')
s = s[:a]+'\n'.join(calls)+'\n'+s[b:]

old = ' attention<false><<<ct,Threads,sizeof(Shared),stream>>>(p);'
assert s.count(old) == 1
s = s.replace(old, ''' int resident=0;
 C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,attention<false>,Threads,sizeof(Shared)));
 TORCH_CHECK(resident>=2,"two-CTA occupancy required, got ",resident," smem=",sizeof(Shared));
'''+old)
src = HERE/(variant+'.cu')
src.write_text(s)
cpp = HERE/(variant+'.cpp')
cpp.write_text((HERE/'m64p2ss8r4.cpp').read_text().replace('m64p2ss8r4',variant))
directory = HERE/('build_'+variant)
directory.mkdir(exist_ok=True)
os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0a'
os.environ['MAX_JOBS'] = '2'
load(name='triattn_sol_'+variant,sources=[str(cpp),str(src)],
     extra_include_paths=[str(HERE.parents[1]/'anthropic_adoption_20260919/cutlass-4.2/include')],
     extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--expt-relaxed-constexpr','--expt-extended-lambda','--use_fast_math','-lineinfo','-Xptxas=-v'],
     extra_ldflags=['-lcuda'],build_directory=str(directory),verbose=True)
print('BUILT',variant,flush=True)
