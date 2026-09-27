from pathlib import Path
import os,sys,importlib.util
HERE=Path(__file__).resolve().parent
SRC=HERE.parent/'core_tiles/broadcast'
variant=sys.argv[1] if len(sys.argv)>1 else 'base134'
DST=HERE/variant
for p in SRC.rglob('*'):
 if p.is_file() and p.suffix in ('.py','.cu','.cuh','.h'):
  q=DST/p.relative_to(SRC);q.parent.mkdir(parents=True,exist_ok=True)
  s=p.read_text().replace('ta_core_broadcast','ta_sol_'+variant).replace('triattn_broadcast','triattn_sol_'+variant)
  if q.name=='triattn_m1.py':s=s.replace('FLAGS = [0, 256]','FLAGS = [0]')
  if variant.startswith('basetable'):
   from table_exp import transform
   s=transform(q.name,s,int(variant[-1]))
  if variant=='basescaledproducer':
   from scaled_producer import transform
   s=transform(q.name,s)
  if variant=='basescaledssglobal':
   from scaled_ss_global import transform
   s=transform(q.name,s)
  if variant in ('baseconst768','baseqr768','basefast768'):
   if q.name=='triattn_m1.py':
    marker='    bias_staged = ext.stage_bias('
    pos=s.index(marker)
    s=s[:pos]+'    assert q.shape[-2] == 768 and q.shape[-4] == 768 and q.shape[-3] == 4 and abs(float(scale) - 1.0 / math.sqrt(32)) < 1e-9, "L768 H4 standard-scale specialization"\n'+s[pos:]
   if q.name=='triattn_m1_sm90.cuh':
    if variant=='baseqr768':s=s.replace('kQinRegs = (kFlags_ & 1048576) != 0','kQinRegs = !kSafe')
    import re
    constants={'S':'768','N':'768','H':'4','n_qtiles':'6','n_ktiles':'6','scale':'0x1.6a09e6p-3f'}
    for key,value in constants.items():
     s=re.sub(r'params\.'+key+r'\b', '(kSafe ? params.'+key+' : '+value+')', s)
  if variant=='base24':
   from compress_bias import transform
   s=transform(q.name,s)
  if variant=='basef40':
   from fuse_pv40 import transform
   s=transform(q.name,s)
  if variant in ('basescaledqr','basescaledqrloop','basescaledqr768','basescaledqglobal'):
   from scaled_q_regs import transform
   s=transform(q.name,s)
   if variant=='basescaledqglobal':
    if q.name=='triattn_m1_sm90.cuh':
     a=s.index('    auto wait_kv_ready =');b=s.index('    // register copies of this warpgroup',a)
     s=s[:a]+'''    auto wait_kv_ready = [&](int st,uint32_t phase) __attribute__((always_inline)) { pipe_kv.wait_full(st,phase); };
'''+s[b:]
    if q.name=='m1_binding.cu':
     s=s.replace('    auto bias_hot=torch::empty_like(bias_staged);','    auto k_hot=k.to(torch::kFloat16);\n    auto bias_hot=torch::empty_like(bias_staged);')
     s=s.replace('Args hot_a{q, k, v, bias_hot','Args hot_a{q, k_hot, v, bias_hot')
   if variant=='basescaledqrloop' and q.name=='triattn_m1_sm90.cuh':
    s=s.replace('#pragma unroll\n            for(int j=0;j<T::kStageElemsK/2/128;++j)', '#pragma unroll 1\n            for(int j=0;j<T::kStageElemsK/2/128;++j)')
   if variant=='basescaledqr768':
    if q.name=='triattn_m1.py':s=s.replace('    qkv = []','    assert S == 768, "L768 specialization"\n    qkv = []')
    if q.name=='triattn_m1_sm90.cuh':
     start=s.index('        if constexpr(!kSafe){',s.index('    auto derive ='))
     end=s.index('        }else{',start)
     s=s[:start]+'''        if constexpr(!kSafe){
            i0=rg*R; kind3=0; kc0=0; n_tiles=6;
            dead=params.rowkind!=nullptr && params.rowkind[int64_t(b)*params.N]==2;
            #pragma unroll
            for(int r=0;r<R;r++){jb[r]=0;je[r]=6;f[r]=0;}
'''+s[end:]
     s=s.replace(' : n_tiles;', ' : 6;')
     s=s.replace('#pragma unroll 1\n    for (;; ++p, pp ^= 1u)', '#pragma unroll (kSafe ? 1 : 3)\n    for (;; ++p, pp ^= 1u)')
  if q.name=='triattn_m1_sm90.cuh' and variant=='baseablatenobias':
   s=s.replace('kNoBias = (T::kFlags & 1) != 0', 'kNoBias = !kSafe')
  if q.name=='triattn_m1_sm90.cuh' and variant=='baseablatereplay':
   s=s.replace('kReplay = (T::kFlags & 2) != 0 || kReplayOps', 'kReplay = !kSafe')
  if q.name=='triattn_m1_sm90.cuh' and variant=='basew4':
   s=s.replace('CW = 32, R = 3,','CW = 32, R = 4,')
   s=s.replace('static constexpr bool kW = (kFlags_ & 4194304) != 0;','static constexpr bool kW = true;')
   s=s.replace('warpgroup_reg_alloc<160>()','warpgroup_reg_alloc<112>()')
  if q.name=='triattn_m1_sm90.cuh' and variant in ('baserneint','baseroundint','basefusedroundint'):
   old='''                __nv_bfloat162 const h2 = __floats2bfloat162_rn(acc(2 * pr), acc(2 * pr + 1));
                dst32(pr) = reinterpret_cast<uint32_t const&>(h2);'''
   rounding=('u += 0x7fffu + ((u >> 16) & 1u); v += 0x7fffu + ((v >> 16) & 1u);' if variant=='baserneint' else 'u += 0x8000u; v += 0x8000u;')
   new='''                if constexpr (!kSafe) {
                    // P is nonnegative. Clamp NaNs to a quiet NaN before
                    // rounding so a payload carry cannot turn NaN into -0.
                    uint32_t u=min(__float_as_uint(acc(2*pr)),0x7fc00000u);
                    uint32_t v=min(__float_as_uint(acc(2*pr+1)),0x7fc00000u);
                    ROUNDING
                    asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(dst32(pr)) : "r"(u), "r"(v));
                } else {
'''+old+'''
                }'''
   assert s.count(old)==1
   new=new.replace('ROUNDING',rounding)
   if variant=='basefusedroundint':
    new=new.replace('uint32_t u=min(__float_as_uint(acc(2*pr)),0x7fc00000u);','uint32_t u=__float_as_uint(acc(2*pr));')
    new=new.replace('uint32_t v=min(__float_as_uint(acc(2*pr+1)),0x7fc00000u);','uint32_t v=__float_as_uint(acc(2*pr+1));')
    new=new.replace(rounding,'u=min(u+0x8000u,0x7fc08000u); v=min(v+0x8000u,0x7fc08000u);')
   s=s.replace(old,new)
  if q.name=='triattn_m1_sm90.cuh' and variant in ('baseprepv','baseprefence','baseearlyfence','basefast768'):
   pa=s.index('    auto issue_pv =');pb=s.index('    auto pack_chunk =',pa)
   helper=s[pa:pb].replace('auto issue_pv =','auto issue_pv_prefenced =')
   helper=helper.replace('        warpgroup_fence_operand(tP);\n        warpgroup_arrive();\n','')
   s=s[:pb]+helper+s[pb:]
   a=s.index('    auto body =');b=s.index('    // drained state:',a)
   body=s[a:b]
   pack='        pack_chunk(accC[bp], PCb[pb]);\n'
   pv='        issue_pv(PCb[pb], tV, Int<hp>{}, Int<cp>{}, Int<(tp & 1) * R>{});\n'
   assert body.count(pack)==1 and body.count(pv)==1
   body=body.replace(pack+pv,'        if constexpr (kSafe) {\n'+pack+pv+'        }\n')
   qkline=next(x for x in body.splitlines(True) if '        issue_qk(accC[bn]' in x)
   if variant=='baseprepv':
    body=body.replace(qkline,qkline+'        if constexpr (!kSafe) {\n'+pack+pv+'        }\n')
   else:
    pre='        if constexpr (!kSafe) { pack_chunk(accC[bp], PCb[pb]); warpgroup_fence_operand(PCb[pb]); }\n'
    post='        if constexpr (!kSafe) { '+pv.strip().replace('issue_pv(','issue_pv_prefenced(')+' }\n'
    if variant in ('baseearlyfence','basefast768'):
     # Pack and fence P before QK's hardware fence. Keep PV as the final
     # commit, after E(current), without another fence waiting for E writes.
     body=body.replace(qkline,pre+qkline)
     body=body.replace('        if constexpr (kSafe) {\n'+pack+pv+'        }\n','        if constexpr (kSafe) {\n'+pack+pv+'        }\n'+post)
    else:body=body.replace(qkline,pre+qkline+post)
   s=s[:a]+body+s[b:]
  if q.name=='triattn_m1_sm90.cuh' and variant.startswith('basepoly'):
   degree=int(variant[8]); fraction=int(variant[9])
   helper=r'''
// Range-reduced polynomial, with integer exponent reconstruction. Normal range
// is [-126,128); preserve mask zero and overflow so the existing SAFE list works.
__device__ __forceinline__ float ex2_poly(float x) {
 float z=fminf(fmaxf(x,-126.f),128.f);
 float rounded=__fadd_rn(z,12582912.f);
 float n=__fadd_rn(rounded,-12582912.f);
 float f=__fadd_rn(z,-n);
 POLYNOMIAL
 unsigned exponent=__float_as_uint(rounded)<<23;
 float y=__uint_as_float(__float_as_uint(p)+exponent);
 return x < -126.f ? 0.f : (x >= 128.f ? INFINITY : y);
}
'''
   poly=('float p=fmaf(fmaf(fmaf(fmaf(0.009670763656142072f,f,0.055875505357739746f),f,0.24022211728271922f),f,0.6931272658129415f),f,1.0000000522792032f);' if degree==4 else 'float p=fmaf(fmaf(fmaf(0.05587550535773725f,f,0.24229446522590045f),f,0.6931272658129415f),f,0.9999482425444736f);')
   if degree==2:poly='float p=fmaf(fmaf(0.24229444447823467f,f,0.7015086752699851f),f,0.9999482435818677f);'
   if variant.endswith('n'):helper=helper.replace('float z=fminf(fmaxf(x,-126.f),128.f);','float z=x;')
   marker='template <class T>'
   pos=s.index(marker,s.index('__device__ __forceinline__ float ex2_approx'))
   s=s[:pos]+helper.replace('POLYNOMIAL',poly)+s[pos:]
   old='s_rc(mi, ni) = ex2_approx(fmaf(s_rc(mi, ni), c_l2, nm[hh][mi]));'
   new=f'if constexpr (!kSafe) {{ if (ni % 4 < {fraction}) s_rc(mi, ni) = ex2_poly(fmaf(s_rc(mi, ni), c_l2, nm[hh][mi])); else s_rc(mi, ni) = ex2_approx(fmaf(s_rc(mi, ni), c_l2, nm[hh][mi])); }} else {{ {old} }}'
   assert s.count(old)==1
   s=s.replace(old,new)
  q.write_text(s)
os.environ['TORCH_EXTENSIONS_DIR']=str(HERE/'build')
os.environ['TORCH_CUDA_ARCH_LIST']='9.0a';os.environ['MAX_JOBS']='2'
os.environ['CUTLASS_PATH']=str(HERE.parents[1]/'anthropic_adoption_20260919/cutlass-4.2')
if variant=='base134':
 from toolchain import use_ptxas
 use_ptxas()
spec=importlib.util.spec_from_file_location('baseline_builder',DST/'triattn_m1.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module._build(verbose=True)
