"""Compile guarded L768 fast path beside the existing hot and SAFE kernels."""
from pathlib import Path
import os,importlib.util,sys
HERE=Path(__file__).resolve().parent
variant=sys.argv[1] if len(sys.argv)>1 else 'basefastdispatch'
source=HERE/'basefast768';target=HERE/variant
for p in source.rglob('*'):
 if not p.is_file() or p.suffix not in ('.py','.cu','.cuh','.h'):continue
 q=target/p.relative_to(source);q.parent.mkdir(parents=True,exist_ok=True)
 s=p.read_text().replace('basefast768',variant)
 if p.name=='triattn_m1.py':
  s='\n'.join(line for line in s.split('\n') if 'L768 H4 standard-scale specialization' not in line)
  a=s.index('def _all_flags():');b=s.index('\n\ndef _generate_sources',a)
  s=s[:a]+'def _all_flags():\n    return [0, 1024, 1073741824]\n'+s[b:]
 if p.name=='triattn_m1_sm90.cuh':
  marker='    constexpr bool kSafe = T::kSafe;'
  s=s.replace(marker,'    constexpr bool kFast = !T::kSafe && (T::kFlags & 1073741824) != 0;\n'+marker)
  s=s.replace('(kSafe ? params.', '(!kFast ? params.')
  a=s.index('    auto body =');b=s.index('    // drained state:',a)
  body=s[a:b]
  body=body.replace('if constexpr (!kSafe) { pack_chunk(accC[bp]', 'if constexpr (kFast) { pack_chunk(accC[bp]')
  body=body.replace('if constexpr (!kSafe) { issue_pv_prefenced', 'if constexpr (kFast) { issue_pv_prefenced')
  body=body.replace('if constexpr (kSafe) {\n        pack_chunk(accC[bp]', 'if constexpr (!kFast) {\n        pack_chunk(accC[bp]')
  s=s[:a]+body+s[b:]
  if variant.endswith('syn'):
   s=s.replace('constexpr uint32_t kBarReinit = uint32_t(cutlass::arch::ReservedNamedBarriers::FirstUserBarrier) + 4;',
               'constexpr uint32_t kBarReinit = 4; // user-relative ID; CUTLASS adds its reserved offset once')
   s=s.replace('cutlass::arch::NamedBarrier::sync(T::kNumThreads, kBarReinit);',
               'cutlass::arch::NamedBarrier(T::kNumThreads, kBarReinit).arrive_and_wait_unaligned();')
 if p.name=='m1_binding.cu':
  marker='    TORCH_CHECK(hot != table().end() && safe != table().end(),'
  pos=s.index(marker)
  s=s[:pos]+'''    // Shape constants and the P fence schedule were qualified at this shape.
    if(flags==0 && q.size(1)==768 && q.size(2)==4 && q.size(3)==768 && float(scale)==0x1.6a09e6p-3f)
        hot=table().find(1073741824);
'''+s[pos:]
 q.write_text(s)
os.environ['TORCH_EXTENSIONS_DIR']=str(HERE/'build')
os.environ['TORCH_CUDA_ARCH_LIST']='9.0a';os.environ['MAX_JOBS']='2'
os.environ['CUTLASS_PATH']=str(HERE.parents[1]/'anthropic_adoption_20260919/cutlass-4.2')
spec=importlib.util.spec_from_file_location('fast_dispatch_build',target/'triattn_m1.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m._build(verbose=True)
