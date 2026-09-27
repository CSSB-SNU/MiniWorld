"""Retain Q where it fits; separately test reloading Z to increase projection parallelism."""
from pathlib import Path
r=Path(__file__).resolve().parent
base=(r/'qkv_compact_async4/fused.cu').read_text()
s=base.replace('q[(Consumers>2*Projectors?Consumers:2*Projectors)]',
               'q[(Capacity<=768?Capacity/64:(Consumers>2*Projectors?Consumers:2*Projectors))]')
s=s.replace('launch<384,6,2,3>', 'launch<384,6,1,3>').replace('launch<768,6,2,4>', 'launch<768,6,1,3>')
s=s.replace('pair==0?s.q[c+which*Projectors].data():s.kv[which][qt].data()',
            'pair==0?(which==0?s.q[Capacity<=768?qt:c].data():s.kv[0][qt].data()):s.kv[which][qt].data()')
needle='        flash::gemm<true,0>(mma,za,wb,acc);'
assert needle in s
s=s.replace(needle,needle+'''
        if(pair==1) {
          // Gate's temporary K tile must retire before the real K is stored.
          if(lane==0)tma_store_wait<0>();
          cutlass::arch::NamedBarrier::sync(128,c+1);
        }''',1)
b=s.index('    if(lane==0) {\n      auto qg=');e=s.index('    auto score=',b)
old=s[b:e]
loads='    if(lane==0)for(int b=0;b<Stages && b<nt;++b)load_bias(b);\n'
s=s[:b]+'    if constexpr(Capacity>768) {\n'+old+'    } else {\n'+loads+'    }\n'+s[e:]
b=s.index('  // QKV saves are complete.')
s=s[:b]+s[b:].replace('s.q[c].data()', 's.q[Capacity<=768?qt:c].data()')
d=r/'qkv_compact_qresident';assert not (d/'build-ready.json').exists()
d.mkdir(exist_ok=True);(d/'fused.cu').write_text(s)

s=base.replace('array_aligned<Element,8192,1024> w[2];', 'array_aligned<Element,8192,1024> w[1];')
s=s.replace('launch<1024,6,1,2>', 'launch<1024,6,1,3>').replace('Config<1024,6,1,2>::Shared','Config<1024,6,1,3>::Shared')
b=s.index('  // Fewer projection WGs');e=s.index('  // QKV saves are complete.',b)
phase=s[b:e]
phase=phase[phase.index('  if(c<Projectors) {'):]
phase=phase.replace('  if(c<Projectors) {','  for(int pair=0;pair<2;++pair) {\n  if(c<Projectors) {',1)
phase=phase.replace('s.wfull.arrive_and_expect_tx(4*4096*sizeof(Element));','s.wfull.arrive_and_expect_tx(2*4096*sizeof(Element));')
phase=phase.replace('for(int which=0;which<4;++which) {', 'for(int wi=0;wi<2;++wi) {\n        int which=pair*2+wi;')
phase=phase.replace('s.scratch.proj.w[which/2].data()', 's.scratch.proj.w[0].data()')
phase=phase.replace('s.wfull.wait(0);', 's.wfull.wait(pair);\n    int turns=(nt-c+Projectors-1)/Projectors;')
phase=phase.replace('s.zfull[c].wait((qt/Projectors)%2);','s.zfull[c].wait((pair*turns+qt/Projectors)%2);')
phase=phase.replace('      #pragma unroll\n      for(int pair=0;pair<2;++pair) {', '      {')
phase=phase.replace('s.scratch.proj.w[pair].data()', 's.scratch.proj.w[0].data()')
needle='''    // Unlike source-only retirement, this also completes global Q writes for reload.
    if(lane==0)asm volatile("cp.async.bulk.wait_group 0;":::"memory");
  }
'''
assert phase.endswith(needle)
phase=phase[:-len(needle)]+'''  }
  // Retire every reader before replacing the single QG/KV weight tile.
  __syncthreads();
  }
  if(c<Projectors && lane==0)asm volatile("cp.async.bulk.wait_group 0;":::"memory");
'''
s=s[:b]+phase+s[e:]
d=r/'qkv_compact_serial';assert not (d/'build-ready.json').exists()
d.mkdir(exist_ok=True);(d/'fused.cu').write_text(s)
