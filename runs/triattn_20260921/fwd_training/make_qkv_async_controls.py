"""Overlap training saves, keeping source retirement and final Q visibility explicit."""
from pathlib import Path
r=Path(__file__).resolve().parent
s=(r/'qkv_compact/fused.cu').read_text()
s=s.replace('q[Consumers],kv[2]', 'q[(Consumers>2*Projectors?Consumers:2*Projectors)],kv[2]')
s=s.replace('launch<768,6,2,3>', 'launch<768,6,2,4>')
s=s.replace('''      // Complete global Q saves before the attention phase reloads them.
      if(lane==0)asm volatile("cp.async.bulk.wait_group 0;":::"memory");''','''      // Reused Q/gate sources must retire; global completion can overlap later tiles.
      if(lane==0)tma_store_wait<0>();''')
needle='''    }
  }
  // QKV saves are complete.'''
assert needle in s
s=s.replace(needle,'''    }
    // Unlike source-only retirement, this also completes global Q writes for reload.
    if(lane==0)asm volatile("cp.async.bulk.wait_group 0;":::"memory");
  }
  // QKV saves are complete.''',1)
d=r/'qkv_compact_async4';assert not (d/'build-ready.json').exists()
d.mkdir(exist_ok=True);(d/'fused.cu').write_text(s)

for c in (2,4):
    s=(r/('qkv_stream_wide%db/fused.cu'%c)).read_text()
    s=s.replace('kv[2][Stages]', 'kv[2][3]').replace('ready[Stages],empty[Stages]', 'ready[3],empty[3]')
    s=s.replace('for(int st=0;st<Stages;++st) {', 'for(int st=0;st<3;++st) {',1)
    s=s.replace('      for(int cc=0;cc<Consumers;++cc)s.bfull[cc][st].init(1);',
                '      if(st<Stages)for(int cc=0;cc<Consumers;++cc)s.bfull[cc][st].init(1);',1)
    s=s.replace('int st=kt%Stages,phase=(kt/Stages)%2;', 'int st=kt%3,phase=(kt/3)%2;')
    s=s.replace('s.zfull[Consumers+st].wait', 's.zfull[Consumers+kt%2].wait')
    s=s.replace('s.producer_z[st].data()', 's.producer_z[kt%2].data()')
    b=s.index('  if(c==Consumers) {',s.index('// All Q/gate readers'))
    e=s.index('  } else {',b)
    prod=s[b:e]
    prod=prod.replace('tma_store_arrive();tma_store_wait<0>();','tma_store_arrive();')
    prod=prod.replace('        s.ready[st].arrive();','''        s.ready[st].arrive();
        // Keep at most two source reads pending; the next recycled slot is retired.
        if(qgroup==0)tma_store_wait<2>();''')
    prod=prod.rstrip()+'\n    if(lane==0 && qgroup==0)tma_store_wait<0>();\n'
    s=s[:b]+prod+s[e:]
    s=s.replace('s.ready[stage].wait((kt/Stages)%2)', 's.ready[kt%3].wait((kt/3)%2)')
    s=s.replace('s.kv[0][stage]','s.kv[0][kt%3]').replace('s.kv[1][stage]','s.kv[1][kt%3]')
    s=s.replace('s.empty[(kt-1)%Stages]', 's.empty[(kt-1)%3]').replace('s.empty[(nt-1)%Stages]', 's.empty[(nt-1)%3]')
    d=r/('qkv_stream_slots3_c%d'%c);assert not (d/'build-ready.json').exists()
    d.mkdir(exist_ok=True);(d/'fused.cu').write_text(s)
