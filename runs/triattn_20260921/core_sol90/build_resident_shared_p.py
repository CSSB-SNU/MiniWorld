"""K-resident four-consumer pipeline with three scores and double shared P."""
from pathlib import Path
import hashlib,os,sys
HERE=Path(__file__).resolve().parent
variant='resident4m64hybrid3sp'
installed=HERE.parent/'oc/opt_core/kernels/triattn_core_broadcast/triattn_broadcast.so'
assert hashlib.sha256(installed.read_bytes()).hexdigest()=='9365a4f35e018afed4adba21590ae2c1734bbccf98d1e53904ebfe2537b3e225'
s=(HERE/'resident4m64hybrid8.cu').read_text()
three=(HERE/'resident4m64s3ssloop6dep.cu').read_text()
shared=(HERE/'m64coop2s4p3qr.cu').read_text()
start=three.index('  }else{\n   // Two QK groups')
end=three.index('  auto id=pv.get_slice(t)',start)
body=three[start:end]
body=body.replace('warpgroup_wait<2>();warpgroup_fence_operand(sc[si]);warpgroup_fence_operand(pp[pb]);',
 'warpgroup_wait<2>();warpgroup_fence_operand(sc[si]);\n    if(seq>=3)release_v(seq-3);')
body=body.replace('pack(sc[fi],pp[pb]);warpgroup_fence_operand(pp[pb]);\n    auto words=recast<uint32_t>(pp[pb]);uint32_t dep=0;\n    #pragma unroll\n    for(int x=0;x<8;++x)dep+=words(x);',
 'uint32_t dep=pack(sc[fi],pp[pb],seq-1);')
body=body.replace('cute::true_type{},0,dep&uint32_t(p.zero)', 'cute::true_type{},seq-1,dep&uint32_t(p.zero)')
body=body.replace('pack(sc[2],pp[1]);issue_pv(pp[1],23,_0{},cute::false_type{});drain();',
 'pack(sc[2],pp[1],23);issue_pv(pp[1],23,_0{},cute::false_type{},23);drain();release_v(23);')
start=s.index('  }else{\n   // QK0 and a zero-P')
end=s.index('  auto id=pv.get_slice(t)',start)
s=s[:start]+body+s[end:]
s=s.replace('Score sc[2]', 'Score sc[3]').replace('(Safe?1:2)', '(Safe?1:3)')
s=s.replace('float bias[8][64*N]', 'Element probabilities[4][2][64*N];\n alignas(128) float bias[6][64*N]')
s=s.replace('bf[8]', 'bf[6]').replace('be[8]', 'be[6]')
s=s.replace('for(int st=0;st<8;++st){s.bf', 'for(int st=0;st<6;++st){s.bf')
s=s.replace('int st=g&7;', 'int st=g%6;')
s=s.replace('if(g>=8){s.be[st].wait(((g/8)-1)&1)', 'if(g>=6){s.be[st].wait(((g/6)-1)&1)')
s=s.replace('int bseq=2*seq+qhalf,st=bseq&7;', 'int bseq=2*seq+qhalf,st=bseq%6;')
s=s.replace('s.bf[st].wait((bseq/8)&1)', 's.bf[st].wait((bseq/6)&1)')
s=s.replace('s.be[(2*seq+qhalf)&7]', 's.be[(2*seq+qhalf)%6]')
s=s.replace('using LS=', 'using SP=decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Element>{},Shape<_64,_32>{}));\nusing PVS=decltype(make_tiled_mma(SM90::GMMA::MMA_64x40x16_F32BF16BF16_SS<GMMA::Major::K,GMMA::Major::MN>{}));\nusing LS=',1)
s=s.replace('  warpgroup_fence_operand(pp[0]);warpgroup_fence_operand(pp[1]);', '  if constexpr(Safe){warpgroup_fence_operand(pp[0]);warpgroup_fence_operand(pp[1]);}')
start=shared.index('  auto pack=')
end=shared.index('  auto issue_pv=',start)
pack=shared[start:end]
pack=pack.replace(' __attribute__((always_inline)) {', ' __attribute__((always_inline)) -> uint32_t {',1)
pack=pack.replace('    // The preceding publication follows waits retiring the old slot\'s PV.',
 '''    uint32_t dep=0;
    #pragma unroll
    for(int n=0;n<8;++n)dep+=words(n);
    // The preceding publication follows waits retiring the old slot's PV.''')
pack=pack.replace('asm volatile("bar.sync %0,128;"::"r"(uint32_t(wg+8)):"memory");', 'asm volatile("bar.sync %0,128;"::"r"(uint32_t(wg+8)):"memory");\n    return dep;')
pack=pack.replace('dst(n)=reinterpret_cast<uint32_t const&>(x);}\n   }', 'dst(n)=reinterpret_cast<uint32_t const&>(x);}\n    return 0;\n   }')
start=s.index('  auto pack=');end=s.index('  auto issue_pv=',start)
s=s[:start]+'  auto probability_ptr=[&](int seq){return s.probabilities[wg][seq&1];};\n'+pack+s[end:]
s=s.replace('if constexpr(Safe || !decltype(prefenced)::value)warpgroup_fence_operand(prob);', 'if constexpr(Safe)warpgroup_fence_operand(prob);')
old='''   #pragma unroll
   for(int kk=0;kk<2;++kk)gemm(pv,prob(_,_,kk),vb(_,_,kk),acc[hh]);'''
new='''   if constexpr(!Safe){
    PVS pvs;pvs.accumulate_=GMMA::ScaleOut::One;
    auto sp=make_tensor(make_smem_ptr(probability_ptr(pseq)),SP{});
    auto pa=pvs.get_slice(0).partition_fragment_A(sp);
    #pragma unroll
    for(int kk=0;kk<2;++kk)gemm(pvs,pa(_,_,kk),vb(_,_,kk),acc[hh]);
   }else{
    #pragma unroll
    for(int kk=0;kk<2;++kk)gemm(pv,prob(_,_,kk),vb(_,_,kk),acc[hh]);
   }'''
assert s.count(old)==1
s=s.replace(old,new)
s=s.replace('resident4m64hybrid8',variant)
src=HERE/(variant+'.cu');src.write_text(s)
cpp=HERE/(variant+'.cpp');cpp.write_text((HERE/'resident4m64hybrid8.cpp').read_text().replace('resident4m64hybrid8',variant))
if '--generate-only' in sys.argv:print('GENERATED',variant,len(s));sys.exit(0)
from torch.utils.cpp_extension import load
directory=HERE/('build_'+variant);directory.mkdir(exist_ok=True)
os.environ['TORCH_CUDA_ARCH_LIST']='9.0a';os.environ['MAX_JOBS']='2'
load(name='triattn_sol_'+variant,sources=[str(cpp),str(src)],extra_include_paths=[str(HERE.parents[1]/'anthropic_adoption_20260919/cutlass-4.2/include')],extra_cflags=['-O3'],
 extra_cuda_cflags=['-O3','--expt-relaxed-constexpr','--expt-extended-lambda','--use_fast_math','-lineinfo','--keep','--keep-dir='+str(directory),'-Xptxas=-v','-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED'],extra_ldflags=['-lcuda'],build_directory=str(directory),verbose=True)
print('BUILT',variant,flush=True)
