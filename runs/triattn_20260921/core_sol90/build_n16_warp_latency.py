"""N16 four-score/full-Q alternatives with more independent consumers.

M64/R5: five88-register consumers, producer32, initial80/768 threads.
M128/R4: four112-register consumers, producer32, initial96/640 threads.
Both preserve query tiling explicitly, first32-key seeds, native BF16 and
fusedN40 PV/den. Independent per-row K/V full and empty barriers.
"""
from pathlib import Path
import argparse,hashlib,os
from torch.utils.cpp_extension import load
HERE=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--m128',action='store_true');ap.add_argument('--rows',type=int);ap.add_argument('--prefenced',action='store_true');args=ap.parse_args()
M=128 if args.m128 else 64
rows=args.rows if args.rows is not None else (4 if args.m128 else 5)
assert rows in (3,4,5)
regs={3:160,4:112,5:88}[rows]
H=M//64
variant='m%dn16r%ds4p2qrf40'%(M,rows)
if args.prefenced:variant+='pf'
oldname='m64n64r3s2p2qrf40'
installed=HERE.parent/'oc/opt_core/kernels/triattn_core_broadcast/triattn_broadcast.so'
assert hashlib.sha256(installed.read_bytes()).hexdigest()=='9365a4f35e018afed4adba21590ae2c1734bbccf98d1e53904ebfe2537b3e225'
s=(HERE/(oldname+'.cu')).read_text().replace(oldname,variant)
s=s.replace('M=64,N=64,LN=128,Ratio=LN/N,D=32,Rows=3,Stages=2;','M=%d,N=16,LN=128,Ratio=LN/N,D=32,Rows=%d,Stages=2;\nconstexpr int H=M/64,BiasSlots=Ratio*H;'%(M,rows))
if M==128:
 s=s.replace('using SQ=decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Element>{},Shape<_64,_32>{}));','using SQ=decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Element>{},Shape<_128,_32>{}));')
 s=s.replace('SQ{},Shape<_64,_32>{},_1{}));','SQ{},Shape<_128,_32>{},_1{}));')
 old='(_,_,min(i0+r,768-1)*4+h),Shape<_64,_32>{},make_coord(qt,0));'
 assert s.count(old)==1;s=s.replace(old,old.replace('Shape<_64,_32>','Shape<_128,_32>'))
 old='auto tq=make_tma_copy(SM90_TMA_LOAD{},g(q),SQ{},Shape<_64,_32>{},_1{});'
 assert s.count(old)==1;s=s.replace(old,old.replace('Shape<_64,_32>','Shape<_128,_32>'))
s=s.replace('160>0',str(regs)+'>0').replace('warpgroup_reg_alloc<160>','warpgroup_reg_alloc<%d>'%regs)
s=s.replace('bias[4][64*N]','bias[BiasSlots][64*N]').replace('bf[2]','bf[BiasSlots/2]').replace('be[4]','be[BiasSlots]')
s=s.replace('st<4;st++','st<BiasSlots;st++').replace('if(st<2)','if(st<BiasSlots/2)')
s=s.replace('nk=768/N,','nk=768/N*H,')
s=s.replace('seq<768/N;seq++','seq<768/N*H;seq++').replace('seq>=4','seq>=BiasSlots')
s=s.replace('seq%4','seq%BiasSlots').replace('seq/4','seq/BiasSlots')
s=s.replace('make_shape(256,N/4,768/N,nq,4)','make_shape(256,N/4,768/N*H,nq,4)')
s=s.replace('Score score;Score next_score;Output acc[1];clear(acc[0]);','Score scores[4];auto& score=scores[0];auto& next_score=scores[1];Output acc[H];\n #pragma unroll\n for(int hh=0;hh<H;++hh)clear(acc[hh]);')
s=s.replace('decltype(den0) den[1]={den0};','decltype(den0) den[2]={den0,make_tensor(acc[H-1].data()+16,Den{}.layout())};')
s=s.replace('float mx[1][2]={{Safe?-INFINITY:0.f,Safe?-INFINITY:0.f}};','float mx[2][2]={{Safe?-INFINITY:0.f,Safe?-INFINITY:0.f},{Safe?-INFINITY:0.f,Safe?-INFINITY:0.f}};')
s=s.replace('warpgroup_fence_operand(acc[0]);warpgroup_fence_operand(den[0]);','\n  #pragma unroll\n  for(int hh=0;hh<H;++hh){warpgroup_fence_operand(acc[hh]);warpgroup_fence_operand(den[hh]);}')
a=s.index(' auto qr=qrthr.partition_fragment_A(qshared);');b=s.index('\n auto init_score=',a)
s=s[:a]+''' auto qrproto=qrthr.partition_fragment_A(local_tile(qshared,Shape<_64,_32>{},make_coord(0,0)));
 decltype(qrproto) qr_all[H];
 if constexpr(!Safe){
  auto cp=make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N,Element>{},qkr);
  auto ct=cp.get_thread_slice(t);
  #pragma unroll
  for(int hh=0;hh<H;++hh){
   auto qs=ct.partition_S(local_tile(qshared,Shape<_64,_32>{},make_coord(hh,0)));
   auto qd=ct.retile_D(qr_all[hh]);cute::copy(cp,qs,qd);warpgroup_fence_operand(qr_all[hh]);
  }
 }
''' +s[b:]
s=s.replace('constexpr int hh=0;','constexpr int hh=decltype(half)::value;')
s=s.replace('else gemm(qkr,qr(_,_,kk),kb(_,_,kk),score);','else gemm(qkr,qr_all[hh](_,_,kk),kb(_,_,kk),score);')
s=s.replace('cx=seq%Ratio','cx=(seq/H)%Ratio').replace('make_coord(0,seq%Ratio)','make_coord(0,(seq/H)%Ratio)').replace('+(seq%Ratio)*N*D','+((seq/H)%Ratio)*N*D')
s=s.replace('seq/(Ratio)','seq/(Ratio*H)').replace('seq/Ratio','seq/(Ratio*H)')
s=s.replace('seq%(Ratio)','seq%(Ratio*H)').replace('seq%Ratio','seq%(Ratio*H)').replace('==Ratio-1','==Ratio*H-1')
s=s.replace('col<(Safe?N/4:8)','col<N/4')
a=s.index(' auto drain=');b=s.index(' if constexpr(Safe){',a)
s=s[:a]+''' auto drain=[&]() __attribute__((always_inline)) {
  warpgroup_wait<0>();asm volatile("":::"memory");
  if constexpr(Safe){warpgroup_fence_operand(score);warpgroup_fence_operand(pp);}
  else {
   #pragma unroll
   for(int j=0;j<4;++j)warpgroup_fence_operand(scores[j]);
   warpgroup_fence_operand(pp);warpgroup_fence_operand(pp1);
  }
  #pragma unroll
  for(int hh=0;hh<H;++hh){warpgroup_fence_operand(acc[hh]);warpgroup_fence_operand(den[hh]);}
 };
''' +s[b:]
s=s.replace('for(int seq=0;seq<nk;seq+=2){step(seq,_0{});step(seq+1,_1{});}','for(int seq=0;seq<nk;seq+=H){step(seq,_0{});if constexpr(H==2)step(seq+1,_1{});}')
a=s.index(' }else{\n  clear(pp);clear(pp1);');b=s.index('\n auto id=pv.get_slice',a)
body=(HERE/'n16_four_score_body.cuh').read_text().replace('  // STATIC_STEPS','\n'.join('  step(Int<%d>{});'%i for i in range(2,768//16*H)))
if args.prefenced:
 old='issue_pv(seq-1,Int<(seq-1)%H>{},prob);'
 assert body.count(old)==1
 body=body.replace(old,'if constexpr(seq+2<768/N*H)issue_pv_prefenced(seq-1,Int<(seq-1)%H>{},prob);\n    else issue_pv(seq-1,Int<(seq-1)%H>{},prob);')
s=s[:a]+' }else{\n'+body+' }\n'+s[b:]
s=s.replace('for(int j=0;j<16;++j)bad|=(__float_as_uint(acc[0](j))&0x7f800000u)==0x7f800000u;','for(int hh=0;hh<H;++hh){\n  #pragma unroll\n  for(int j=0;j<16;++j)bad|=(__float_as_uint(acc[hh](j))&0x7f800000u)==0x7f800000u;\n  }')
s=s.replace('for(int hh=0;hh<1;hh++)','for(int hh=0;hh<H;hh++)')
s=s.replace('u=(idx>>9)%(N/8),hh=0,cc=idx/(M*N);','u=(idx>>9)%(N/8),hh=(idx/(64*N))%H,cc=idx/(M*N);')
# Host descriptor shape must expose each 64-query half in staged order.
s=s.replace('make_shape(256,N/4,L/N,L/M,4)','make_shape(256,N/4,L/N*H,L/M,4)')
s=s.replace('int64_t(M*N),int64_t(M*L)','int64_t(64*N),int64_t(M*L)')
assert 'STATIC_STEPS' not in s and ('Rows=%d'%rows) in s
source=HERE/(variant+'.cu');source.write_text(s)
cpp=HERE/(variant+'.cpp');cpp.write_text((HERE/(oldname+'.cpp')).read_text().replace(oldname,variant))
directory=HERE/('build_'+variant);directory.mkdir(exist_ok=True)
os.environ.update(TORCH_CUDA_ARCH_LIST='9.0a',MAX_JOBS='2')
load(name='triattn_sol_'+variant,sources=[str(cpp),str(source)],extra_include_paths=[str(HERE.parents[1]/'anthropic_adoption_20260919/cutlass-4.2/include')],extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--expt-relaxed-constexpr','--expt-extended-lambda','--use_fast_math','-lineinfo','-Xptxas=-v','--keep','--keep-dir='+str(directory)],extra_ldflags=['-lcuda'],build_directory=str(directory),verbose=True)
print('BUILT',variant,flush=True)
