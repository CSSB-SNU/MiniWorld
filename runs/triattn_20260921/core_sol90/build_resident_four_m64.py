"""Independent resident K/V + four M64 consumer WGs, L768 only."""
from pathlib import Path
import hashlib, os, sys

HERE = Path(__file__).resolve().parent
kind = next((x.split('=',1)[1] for x in sys.argv if x.startswith('--kind=')), 'unroll24')
assert kind in ('unroll24', 'loop6', 'two8', 'loop6dep', 'two8dep', 'loop6ssdep', 'two8ssdep', 'hybrid8')
VARIANT = {'unroll24':'resident4m64s3qr','loop6':'resident4m64s3qrloop6','two8':'resident4m64s2qrloop8','loop6dep':'resident4m64s3qrloop6dep','two8dep':'resident4m64s2qrloop8dep','loop6ssdep':'resident4m64s3ssloop6dep','two8ssdep':'resident4m64s2ssloop8dep','hybrid8':'resident4m64hybrid8'}[kind]
hybrid=kind=='hybrid8'
if hybrid:kind='two8ssdep'
shared_q='ss' in kind
kind=kind.replace('ss','')
ordered_descriptors=kind.endswith('dep')
if ordered_descriptors:kind=kind[:-3]
installed = HERE.parent/'oc/opt_core/kernels/triattn_core_broadcast/triattn_broadcast.so'
assert hashlib.sha256(installed.read_bytes()).hexdigest() == '9365a4f35e018afed4adba21590ae2c1734bbccf98d1e53904ebfe2537b3e225'
old = (HERE/'persistkv2b4.cu').read_text()
m64 = (HERE/'m64coop2s4p2sskv4.cu').read_text()
s = old[:old.index(' cutlass::arch::warpgroup_reg_alloc<224>();')]
s = s.replace('two consumer warpgroups.', 'four M64 consumer warpgroups, Q/bias alias, three scores.')
s = s.replace('Consumers=256,Threads=384', 'Consumers=512,Threads=640')
s = s.replace('using PV=decltype', 'using PVBase=decltype', 1)
s = s.replace('using LS=', 'using PV=decltype(make_tiled_mma(SM90::GMMA::MMA_64x40x16_F32BF16BF16_RS<GMMA::Major::K,GMMA::Major::MN>{}));\nusing LS=', 1)
s = s.replace('int L;float scale; };', 'int L;float scale;int zero; };')
s = s.replace('v[Stages][Rows][LN*D],ones[8*N];', 'v[Stages][Rows][LN*D];\n alignas(1024) Element ones[N*D];')
s = s.replace('s.qe.init(Rows*4);s.qd.init(Rows*4);', 's.qe.init(16);s.qd.init(16);')
s = s.replace('x<8*N', 'x<N*D')
s = s.replace('if(g%48==2)', 'if(g%48==0)')
s += r'''
 // 640*96 == 128*32 + 512*112; all register reallocations are full WGs.
 cutlass::arch::warpgroup_reg_alloc<112>();
 int wg=int(__reduce_max_sync(0xffffffffu,unsigned(tid)/128u)),t=tid%128;
 int pair=wg/2,qhalf=wg%2;
 QK qk;PV pv;PVBase pvbase;LS ls;
 qk.accumulate_=GMMA::ScaleOut::One;pv.accumulate_=GMMA::ScaleOut::One;
 auto tq=qk.get_slice(0);auto tp=pvbase.get_slice(0);
 using Score=decltype(partition_fragment_C(qk,Shape<_64,_32>{}));
 using Output=decltype(partition_fragment_C(pv,Shape<_64,_40>{}));
 using Den=decltype(partition_fragment_C(ls,Shape<_64,_8>{}));
 auto qshared=make_tensor(make_smem_ptr(s.q[pair]),SQ{});
 auto qtile=local_tile(qshared,Shape<_64,_32>{},make_coord(qhalf,0));
 auto qa=qk.get_thread_slice(t).partition_fragment_A(qtile);
 auto qcopy=make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N,Element>{},qk);
 auto qthr=qcopy.get_thread_slice(t);
 #pragma unroll 1
 for(int qt=0;qt<6;++qt){
  Score sc[3];Output acc[1];
  auto d0=make_tensor(acc[0].data()+16,Den{}.layout());
  decltype(d0) den[1]={d0};
  auto pproto=make_tensor_like<Element>(make_tensor(sc[0].data(),flash::convert_layout_acc_Aregs<PV>(sc[0].layout())));
  decltype(pproto) pp[2];
  clear(acc[0]);clear(pp[0]);clear(pp[1]);
  float nm[1][2]={};bool bad=false;float c=p.scale*1.4426950408889634f;
  s.qr.wait(qt&1);asm volatile("":::"memory");
  auto qs=qthr.partition_S(qtile);auto qd=qthr.retile_D(qa);copy(qcopy,qs,qd);
  warpgroup_fence_operand(qa);
  // Hardware dependency from every LDSM result to the release address.
  // Each warp arrives once, and both query halves of both rows must arrive
  // before the producer may overwrite the two Q/bias alias slots.
  auto qw=recast<uint32_t>(qa);uint32_t qdep=0;
  #pragma unroll
  for(int x=0;x<size(qw);++x)qdep+=qw(x);
  uint32_t qeaddr=cast_smem_ptr_to_uint(&s.qe)+(qdep&uint32_t(p.zero));
  __syncwarp();
  if(t%32==0)asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"::"r"(qeaddr):"memory");
  warpgroup_fence_operand(acc[0]);
  auto init=[&](auto& a,int seq,uint32_t ordered_zero=0) __attribute__((always_inline)) {
   if(seq<24){
    int bseq=2*seq+qhalf,st=bseq&3;
    s.bf[st].wait((bseq/4)&1);asm volatile("":::"memory");
    auto bp=(st<2?s.bias[st]:s.qb[st-2])+ordered_zero;
    #pragma unroll
    for(int u=0;u<4;++u){
     float4 b=*reinterpret_cast<float4 const*>(bp+u*512+t*4);
     a(4*u)=b.x;a(4*u+1)=b.y;a(4*u+2)=b.z;a(4*u+3)=b.w;
    }
   }else clear(a);
  };
  auto issue_qk=[&](auto& a,int seq,auto hc) __attribute__((always_inline)) {
   int kt=seq/4;
   if(seq<24 && seq%4==0){s.full[kt].wait(0);asm volatile("":::"memory");}
   auto ks=local_tile(make_tensor(make_smem_ptr(s.k[kt<6?kt:0][pair]),SK{}),Shape<_32,_32>{},make_coord(seq%4,0));
   auto kb=tq.partition_fragment_B(ks);
   warpgroup_fence_operand(a);warpgroup_arrive();
   #pragma unroll
   for(int kk=0;kk<2;++kk)gemm(qk,qa(_,_,kk),kb(_,_,kk),a);
   warpgroup_commit_batch();
   if(seq<24){__syncwarp();if(t%32==0)s.be[(2*seq+qhalf)&3].arrive();}
  };
'''
helpers = m64[m64.index('  auto rowmax='):m64.index('  auto release=')]
helpers = helpers.replace('s.v[(seq/2)%Stages][wg]', 's.v[seq/4][pair]')
helpers = helpers.replace('seq%2', 'seq%4')
helpers = helpers.replace('(Safe?1:4)', '(Safe?1:3)')
s += helpers
s += r'''
  if constexpr(Safe){
   #pragma unroll 1
   for(int seq=0;seq<24;++seq){
    init(sc[0],seq);issue_qk(sc[0],seq,_0{});drain();
    exponentiate(sc[0],_0{},cute::true_type{});pack(sc[0],pp[0]);
    issue_pv(pp[0],seq,_0{},cute::false_type{});drain();
   }
  }else{
   // Two QK groups are complete before the uniform two-group body begins.
   // First dummy PV uses zero P, keeping every dynamic path's group count
   // identical. The ring index has period 3 and P has period 2.
   clear(sc[2]);init(sc[0],0);issue_qk(sc[0],0,_0{});
   init(sc[1],1);issue_qk(sc[1],1,_0{});drain();
   auto step=[&](auto seqc) __attribute__((always_inline)) {
    constexpr int seq=decltype(seqc)::value,si=seq%3,fi=(seq+2)%3,pb=(seq+1)%2;
    warpgroup_wait<2>();warpgroup_fence_operand(sc[si]);warpgroup_fence_operand(pp[pb]);
    pack(sc[fi],pp[pb]);warpgroup_fence_operand(pp[pb]);
    auto words=recast<uint32_t>(pp[pb]);uint32_t dep=0;
    #pragma unroll
    for(int x=0;x<8;++x)dep+=words(x);
    init(sc[fi],seq+2,dep&uint32_t(p.zero));
    issue_qk(sc[fi],seq+2,_0{});
    exponentiate(sc[si],_0{},bool_constant<seq==0>{});
    issue_pv(pp[pb],seq==0?0:seq-1,_0{},cute::true_type{});
    if constexpr(seq%8==7){drain();rescale(sc[si]);}
   };
'''
s += ''.join('   step(Int<%d>{});\n' % x for x in range(24))
s += r'''
   pack(sc[2],pp[1]);issue_pv(pp[1],23,_0{},cute::false_type{});drain();
  }
'''
epilogue=m64[m64.index('  auto id=pv.get_slice(t)'):m64.index('__global__ void prepare(')]
epilogue=epilogue.replace('qt*M+hh*64', 'qt*M+qhalf*64')
epilogue=epilogue.replace('int64_t(i0+wg)', 'int64_t(i0+pair)')
epilogue=epilogue.replace('\n }\n}\n', '\n  __syncwarp();if(t%32==0)s.qd.arrive();\n  warpgroup_fence_operand(qa);\n }\n}\n')
s+=epilogue
tail=old[old.index('__global__ void prepare('):]
tail=tail.replace('L,float(scale)};', 'L,float(scale),0};')
tail=tail.replace(' attention<false><<<ct,Threads,sizeof(Shared),stream>>>(p);', '''
 // Query runtime occupancy before any launch; no invalid register budget.
 static bool checked=false;
 if(!checked){
  int active=0;cudaFuncAttributes attrs{};
  C10_CUDA_CHECK(cudaFuncGetAttributes(&attrs,attention<false>));
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,attention<false>,Threads,sizeof(Shared)));
  TORCH_CHECK(active>=1 && attrs.numRegs<=96,"resident4 resources: ",active," CTAs, ",attrs.numRegs," regs");
  std::cout << "RESOURCES " << active << " CTAs " << attrs.numRegs << " regs " << sizeof(Shared) << " smem" << std::endl;
  checked=true;
 }
 attention<false><<<ct,Threads,sizeof(Shared),stream>>>(p);''')
s+=tail
if kind != 'unroll24':
 start=s.index('   auto step=[&](auto seqc)')
 end=s.index('   pack(sc[2],pp[1]);',start)
 body=s[start:s.index('   step(Int<0>{});',start)]
 if kind == 'loop6':
  body=body.replace('auto seqc)', 'auto seqc,int base)')
  body=body.replace('constexpr int seq=decltype(seqc)::value,si=seq%3,fi=(seq+2)%3,pb=(seq+1)%2;',
   'constexpr int ix=decltype(seqc)::value,si=ix%3,fi=(ix+2)%3,pb=(ix+1)%2;int seq=base+ix;')
  body=body.replace('exponentiate(sc[si],_0{},bool_constant<seq==0>{});',
   'if constexpr(ix==0){if(base==0)exponentiate(sc[si],_0{},cute::true_type{});else exponentiate(sc[si],_0{},cute::false_type{});}else exponentiate(sc[si],_0{},cute::false_type{});')
  body=body.replace('if constexpr(seq%8==7)', 'if constexpr(ix==5)')
  steps='   #pragma unroll 1\n   for(int period=0;period<4;++period){\n'+''.join('    step(Int<%d>{},period*6);\n'%x for x in range(6))+'   }\n'
 else:
  body=body.replace('auto seqc)', 'auto seqc,int base)')
  body=body.replace('constexpr int seq=decltype(seqc)::value,si=seq%3,fi=(seq+2)%3,pb=(seq+1)%2;',
   'constexpr int ix=decltype(seqc)::value,si=ix%2,fi=(ix+1)%2,pb=(ix+1)%2;int seq=base+ix;')
  body=body.replace('warpgroup_wait<2>()', 'warpgroup_wait<1>()')
  body=body.replace('seq+2', 'seq+1')
  body=body.replace('exponentiate(sc[si],_0{},bool_constant<seq==0>{});',
   'if constexpr(ix==0){if(base==0)exponentiate(sc[si],_0{},cute::true_type{});else exponentiate(sc[si],_0{},cute::false_type{});}else exponentiate(sc[si],_0{},cute::false_type{});')
  body=body.replace('if constexpr(seq%8==7)', 'if constexpr(ix==7)')
  steps='   #pragma unroll 1\n   for(int period=0;period<3;++period){\n'+''.join('    step(Int<%d>{},period*8);\n'%x for x in range(8))+'   }\n'
 s=s[:start]+body+steps+s[end:]
 if kind == 'two8':
  s=s.replace('Score sc[3]', 'Score sc[2]').replace('(Safe?1:3)', '(Safe?1:2)')
  s=s.replace('clear(sc[2]);init(sc[0],0);issue_qk(sc[0],0,_0{});\n   init(sc[1],1);issue_qk(sc[1],1,_0{});drain();', 'clear(sc[1]);init(sc[0],0);issue_qk(sc[0],0,_0{});\n   issue_pv(pp[0],0,_0{},cute::false_type{});drain();')
  s=s.replace('Two QK groups are complete before', 'QK0 and a zero-P PV group are complete before')
  s=s.replace('ring index has period 3', 'ring index has period 2')
  s=s.replace('pack(sc[2],pp[1]);', 'pack(sc[1],pp[1]);')
if ordered_descriptors:
 # The opaque zero derived from all packed P words stops descriptor setup
 # from becoming a large bank of loop-invariant, long-lived registers.
 s=s.replace('auto issue_qk=[&](auto& a,int seq,auto hc)', 'auto issue_qk=[&](auto& a,int seq,auto hc,uint32_t ordered_zero=0)')
 s=s.replace('s.k[kt<6?kt:0][pair]', 's.k[kt<6?kt:0][pair]+ordered_zero')
 s=s.replace('auto prefenced,int pseq=0)', 'auto prefenced,int pseq=0,uint32_t ordered_zero=0)')
 s=s.replace('s.v[seq/4][pair]', 's.v[seq/4][pair]+ordered_zero')
 s=s.replace('issue_qk(sc[fi],seq+2,_0{});', 'issue_qk(sc[fi],seq+2,_0{},dep&uint32_t(p.zero));')
 s=s.replace('issue_qk(sc[fi],seq+1,_0{});', 'issue_qk(sc[fi],seq+1,_0{},dep&uint32_t(p.zero));')
 s=s.replace('issue_pv(pp[pb],seq==0?0:seq-1,_0{},cute::true_type{});', 'issue_pv(pp[pb],seq==0?0:seq-1,_0{},cute::true_type{},0,dep&uint32_t(p.zero));')
if shared_q:
 # Keep Q in shared memory for all six tiles, with only two bias half-slots.
 # Query-done retires every SS QK before the next query TMA overwrites Q.
 s=s.replace('GMMA::rs_op_selector<Element,Element,float,Shape<_64,_32,_32>>()', 'GMMA::ss_op_selector<Element,Element,float,Shape<_64,_32,_32>>()')
 s=s.replace('union alignas(128) { Element q[Rows][M*D];float qb[2][64*N]; };', 'alignas(128) Element q[Rows][M*D];')
 s=s.replace('qr,full[Stages],bf[4]', 'qr,full[Stages],bf[2]')
 s=s.replace('qe,qd,be[4]', 'qd,be[2]')
 s=s.replace('s.qe.init(16);', '')
 s=s.replace('for(int st=0;st<4;++st){s.bf', 'for(int st=0;st<2;++st){s.bf')
 s=s.replace('int st=g&3;', 'int st=g&1;')
 s=s.replace('    if(g%48==0){s.qe.wait((g/48)&1);asm volatile("":::"memory");}\n', '')
 s=s.replace('if(g>=4){s.be[st].wait(((g/4)-1)&1);', 'if(g>=2){s.be[st].wait(((g/2)-1)&1);')
 s=s.replace('st<2?s.bias[st]:s.qb[st-2]', 's.bias[st]')
 a=s.index(' auto qa=qk.get_thread_slice(t)')
 b=s.index(' #pragma unroll 1\n for(int qt=',a)
 s=s[:a]+' auto qa=tq.partition_fragment_A(qtile);\n'+s[b:]
 a=s.index('  auto qs=qthr.partition_S(qtile)')
 b=s.index('  warpgroup_fence_operand(acc[0]);',a)
 s=s[:a]+s[b:]
 s=s.replace('int bseq=2*seq+qhalf,st=bseq&3;', 'int bseq=2*seq+qhalf,st=bseq&1;')
 s=s.replace('s.bf[st].wait((bseq/4)&1)', 's.bf[st].wait((bseq/2)&1)')
 s=s.replace('s.be[(2*seq+qhalf)&3]', 's.be[(2*seq+qhalf)&1]')
 s=s.replace('  warpgroup_fence_operand(qa);\n', '')
if hybrid:
 # Resident K96KiB, two-stage V32KiB, Q16KiB, bias64KiB, ones2KiB.
 s=s.replace('v[Stages][Rows][LN*D]', 'v[2][Rows][LN*D]')
 s=s.replace('float bias[2][64*N]', 'float bias[8][64*N]')
 s=s.replace('qr,full[Stages],bf[2]', 'qr,full[Stages],vf[2],bf[8]')
 s=s.replace('qd,be[2]', 'qd,ve[2],be[8]')
 s=s.replace('s.full[st].init(2)', 's.full[st].init(1)')
 s=s.replace('for(int st=0;st<2;++st){s.bf', 'for(int st=0;st<2;++st){s.vf[st].init(1);s.ve[st].init(16);}\n  for(int st=0;st<8;++st){s.bf')
 a=s.index('  if(tid==Consumers+64){')
 b=s.index('  if(tid==Consumers){',a)
 s=s[:a]+r'''
  if(tid==Consumers+64){
   #pragma unroll 1
   for(int g=0;g<36;++g){
    int st=g&1,kt=g%6;
    if(g>=2){s.ve[st].wait(((g/2)-1)&1);asm volatile("":::"memory");}
    s.vf[st].arrive_and_expect_tx(Rows*LN*D*sizeof(Element));
    #pragma unroll
    for(int r=0;r<Rows;++r){
     auto src=local_tile(p.v.get_tma_tensor(make_shape(L,32,L*4))(_,_,(i0+r)*4+h),Shape<_128,_32>{},make_coord(kt,0));
     auto dst=make_tensor(make_smem_ptr(s.v[st][r]),SK{});auto sl=p.v.get_slice(_0{});
     copy(p.v.with(reinterpret_cast<uint64_t&>(s.vf[st])),sl.partition_S(src),sl.partition_D(dst));
    }
   }
  }
'''+s[b:]
 s=s.replace('int st=g&1;\n    if(g>=2){s.be[st].wait(((g/2)-1)&1);', 'int st=g&7;\n    if(g>=8){s.be[st].wait(((g/8)-1)&1);')
 s=s.replace('int bseq=2*seq+qhalf,st=bseq&1;', 'int bseq=2*seq+qhalf,st=bseq&7;')
 s=s.replace('s.bf[st].wait((bseq/2)&1)', 's.bf[st].wait((bseq/8)&1)')
 s=s.replace('s.be[(2*seq+qhalf)&1]', 's.be[(2*seq+qhalf)&7]')
 s=s.replace('s.v[seq/4][pair]', 's.v[(seq/4)&1][pair]')
 a=s.index('  auto issue_pv=')
 b=s.index('   auto vs=',a)
 s=s[:b]+'   if(seq%4==0){s.vf[(seq/4)&1].wait(((qt*6+seq/4)/2)&1);asm volatile("":::"memory");}\n'+s[b:]
 a=s.index('  if constexpr(Safe){\n   #pragma unroll 1\n   for(int seq=0;seq<24;')
 s=s[:a]+r'''
  // Every owning warp releases V only after its final PV read has retired.
  auto release_v=[&](int seq) __attribute__((always_inline)) {
   if(seq>=0 && seq%4==3){__syncwarp();if(t%32==0)s.ve[(seq/4)&1].arrive();}
  };
'''+s[a:]
 s=s.replace('issue_pv(pp[0],seq,_0{},cute::false_type{});drain();', 'issue_pv(pp[0],seq,_0{},cute::false_type{});drain();release_v(seq);')
 s=s.replace('warpgroup_wait<1>();warpgroup_fence_operand(sc[si]);warpgroup_fence_operand(pp[pb]);', 'warpgroup_wait<1>();warpgroup_fence_operand(sc[si]);warpgroup_fence_operand(pp[pb]);\n    if(seq>=3)release_v(seq-3);')
 s=s.replace('pack(sc[1],pp[1]);issue_pv(pp[1],23,_0{},cute::false_type{});drain();', 'pack(sc[1],pp[1]);issue_pv(pp[1],23,_0{},cute::false_type{});drain();release_v(23);')
s=s.replace('persistkv2b4',VARIANT)
src=HERE/(VARIANT+'.cu');src.write_text(s)
cpp=HERE/(VARIANT+'.cpp');cpp.write_text((HERE/'persistkv2b4.cpp').read_text().replace('persistkv2b4',VARIANT))
if '--generate-only' in sys.argv:
 print('GENERATED',VARIANT,len(s));sys.exit(0)
from torch.utils.cpp_extension import load
directory=HERE/('build_'+VARIANT);directory.mkdir(exist_ok=True)
os.environ['TORCH_CUDA_ARCH_LIST']='9.0a';os.environ['MAX_JOBS']='2'
load(name='triattn_sol_'+VARIANT,sources=[str(cpp),str(src)],extra_include_paths=[str(HERE.parents[1]/'anthropic_adoption_20260919/cutlass-4.2/include')],extra_cflags=['-O3'],
 extra_cuda_cflags=['-O3','--expt-relaxed-constexpr','--expt-extended-lambda','--use_fast_math','-lineinfo','--keep','--keep-dir='+str(directory),'-Xptxas=-v','-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED'],extra_ldflags=['-lcuda'],build_directory=str(directory),verbose=True)
print('BUILT',VARIANT,flush=True)
