"""Three consumers own 3/3/2 rows; preserve R8 bias order and all barriers."""
from pathlib import Path

root = Path(__file__).resolve().parent.parent / 'bias_fusion'
base = (root / 'rs8_producer32/grouped.cu').read_text()
staged = (root / 'rs8_staged_parallel/grouped.cu').read_text()
for staging in (False, True):
    s = base.replace('NWG=2, RP=R/NWG', 'NWG=3, RP=3')
    s = s.replace('__launch_bounds__(384,1)', '__launch_bounds__(512,1)')
    s = s.replace('warpgroup_reg_alloc<232>()', 'warpgroup_reg_alloc<160>()')
    s = s.replace('s.kv_full.arrive_and_expect_tx(C::RP*2*2048*sizeof(Element));',
                  'int rows=min(C::RP,R-producer*C::RP);\n      s.kv_full.arrive_and_expect_tx(rows*2*2048*sizeof(Element));')
    a = s.index('    if(tid%32==0')
    b = s.index('  typename C::ScoreMMA', a)
    producer = s[a:b].replace('rr<C::RP', 'rr<rows').replace('jt*C::RP+rr', 'jt*rows+rr')
    s = s[:a] + producer + s[b:]
    if staging:
        s = s.replace('    // dBias reduction', '    array_aligned<float,2048,1024> probability[NWG];\n    // dBias reduction')
        a = s.index('      auto score=partition_fragment_C')
        b = s.index('      cutlass::arch::NamedBarrier::sync(128,wg+1);', a)
        sa = staged.index('      auto score=partition_fragment_C')
        sb = staged.index('      cutlass::arch::NamedBarrier::sync(128,wg+1);', sa)
        s = s[:a] + staged[sa:sb] + s[b:]
    s = s.replace('      constexpr int rr=decltype(rr_)::value;',
                  '      constexpr int rr=decltype(rr_)::value;\n      if(wg*C::RP+rr<R){')
    s = s.replace('int it=jt*C::RP+rr,slot=', 'int it=jt*min(C::RP,R-wg*C::RP)+rr,slot=')
    s = s.replace('if(lane==0) s.q_empty[wg][slot].arrive();\n    });',
                  'if(lane==0) s.q_empty[wg][slot].arrive();\n      }\n    });')
    s = s.replace('NamedBarrier::sync(256,5)', 'NamedBarrier::sync(384,5)')
    s = s.replace('    { // Consumers split', '    if(wg<2) { // First two consumers split')
    s = s.replace('for(int rr=0;rr<C::RP;++rr){\n            uint32_t',
                  'for(int rr=0;rr<C::RP;++rr){\n            if(w*C::RP+rr<R){\n            uint32_t')
    s = s.replace('sum1+=float(Element::bitcast(uint16_t(pair>>16)));',
                  'sum1+=float(Element::bitcast(uint16_t(pair>>16)));\n            }')
    # Last consumer's unused row must not read/write any output or retired K/V.
    marker = '  for(int rr=0;rr<C::RP;++rr) {\n    int r=wg*C::RP+rr,row=group*R+r;'
    assert s.count(marker) == 1
    s = s.replace(marker, marker + '\n    if(r<R){')
    s = s.replace('tma_store_arrive();tma_store_wait<0>();\n    }\n  }',
                  'tma_store_arrive();tma_store_wait<0>();\n    }\n    }\n  }')
    s = s.replace(',384,sizeof(typename C::Shared)', ',512,sizeof(typename C::Shared)')
    s = s.replace('One producer warpgroup and two consumer warpgroups; each consumer owns four outer rows.',
                  'One producer warpgroup and three consumers owning 3/3/2 outer rows.')
    target = root / ('rs8_three_staged' if staging else 'rs8_three')
    target.mkdir(exist_ok=True)
    (target / 'grouped.cu').write_text(s)
