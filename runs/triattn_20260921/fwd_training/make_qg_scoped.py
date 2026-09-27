"""Shorten projection live ranges and use compile-time gate/Q ownership."""
from pathlib import Path
r=Path(__file__).resolve().parent
s=(r/'qg_preload3/fused.cu').read_text()
begin=s.index('  auto shape=make_shape(L,_32{},_4{},L);')
end=s.index('  {\n    // Z is loaded once.',begin)
loads=s[begin:end]
s=s[:begin]+s[end:]
s=s.replace('L=p.L, nt=L/64;', 'L=p.L;')
s=s.replace('  {\n    // Z is loaded once.', '  {\n    auto shape=make_shape(L,_32{},_4{},L);\n    // Z is loaded once.',1)
start=s.index('    #pragma unroll\n    for(int which=0;which<2;++which) {')
body_start=s.index('{',start)+1
body_end=s.index('    }\n    if(tid==0) {load(0);if(nt>1)load(1);}',body_start)
body=s[body_start:body_end]
body=body.replace('      // Retire every projection reader and the TMA save before scratch/output reuse.\n      cutlass::arch::NamedBarrier::sync(128,0);',
'''      // The gate save must release the shared tile before Q overwrites it.
      if constexpr(which==0)cutlass::arch::NamedBarrier::sync(128,0);''')
s=s[:start]+''.join('    {\n      constexpr int which=%d;'%i+body+'    }\n' for i in (0,1))+s[body_end+len('    }\n'):]
s=s.replace('    if(tid==0) {load(0);if(nt>1)load(1);}\n','',1)
needle='  {\n    int r=0, lane=tid, row=rg;'
assert needle in s
s=s.replace(needle,'  int nt=L/64;\n'+loads+'  if(tid==0) {load(0);if(nt>1)load(1);}\n'+needle,1)
d=r/'qg_scoped';assert not (d/'build-ready.json').exists()
d.mkdir(exist_ok=True);(d/'fused.cu').write_text(s)
