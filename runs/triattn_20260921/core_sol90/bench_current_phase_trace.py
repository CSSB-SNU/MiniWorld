"""Bitwise/overhead control plus coarse same-CTA clock differences.

The imported harness checks compiler and emitted register-role gates, then
compares complete core/block graphs. This script subsequently reads only
the diagnostic module's private global after each isolated core replay.
"""
import ctypes as C,json,statistics
from pathlib import Path
import bench_ptxas_graph as b

name='phasetrace'
idx=next(i for i,c in enumerate(b.cases) if c['name']==name)
get_global=b.api('cuModuleGetGlobal_v2',[C.POINTER(C.c_uint64),C.POINTER(b.Z),b.P,C.c_char_p])
copy_host=b.api('cuMemcpyDtoH_v2',[b.P,C.c_uint64,b.Z])
ptr=C.c_uint64();size=b.Z()
get_global(C.byref(ptr),C.byref(size),b.modules[idx],b'diag_phase_clock')
assert size.value==3*3*20*8,size.value
raw=(C.c_uint64*(3*3*20))()
phases=['cta_init','register_allocate','consumer_setup','q_wait_and_cache',
        'kv_and_bias_wait','first_qk','bootstrap_tail','consumer_align',
        'period0_body','period0_drain','period1_body','period1_drain',
        'period2_body','final_drain','last_pv','epilogue']
coords=[(1,1,0),(2,128,1),(5,255,3)]
samples=[]
with b.torch.no_grad():
 for rd in range(12):
  b.graphs['core'][name].replay();b.torch.cuda.synchronize()
  copy_host(C.cast(raw,b.P),ptr.value,size.value)
  for c,coord in enumerate(coords):
   smids=[]
   for wg in range(3):
    values=list(raw[(c*3+wg)*20:(c*3+wg+1)*20]);clocks=values[:17]
    assert all(x>0 for x in clocks),(rd,c,wg,'missing marker',clocks)
    assert all(a<=z for a,z in zip(clocks,clocks[1:])),(rd,c,wg,'nonmonotonic',clocks)
    smids.append(values[19])
    durations=dict(zip(phases,[z-a for a,z in zip(clocks,clocks[1:])]))
    total=clocks[-1]-clocks[0];assert sum(durations.values())==total
    samples.append(dict(round=rd,cta=list(coord),consumer=wg,smid=values[19],clocks=clocks,
                        total_cycles=total,phase_cycles=durations,phase_fractions={k:v/total for k,v in durations.items()}))
   assert len(set(smids))==1,(rd,c,'CTA cannot span SMs',smids)
summary=[]
for coord in coords:
 for wg in range(3):
  selected=[s for s in samples if s['cta']==list(coord) and s['consumer']==wg]
  summary.append(dict(cta=list(coord),consumer=wg,total_cycles=statistics.median(s['total_cycles'] for s in selected),
      phase_cycles={p:statistics.median(s['phase_cycles'][p] for s in selected) for p in phases},
      phase_fractions={p:statistics.median(s['phase_fractions'][p] for s in selected) for p in phases}))
out=Path(b.a.output).with_name(Path(b.a.output).stem+'-phases.json')
data=dict(scope='Instrumented kernel; three sampled CTAs, within-CTA clock differences only. Markers have overhead; not an unmodified-kernel latency breakdown.',
          graph_comparison=Path(b.a.output).name,ending=b.ending,summary=summary,samples=samples)
out.write_text(json.dumps(data,indent=2)+'\n')
print('PHASE_RESULT',json.dumps(summary),flush=True)
