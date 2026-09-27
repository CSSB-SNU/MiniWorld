"""Keep two dX WGMMA groups in flight without changing accumulation order."""


def pipeline(body, mode=1):
    assert mode in (1, 2, 3, 4, 5)
    init = '  float v0[64]={},v1[64]={};'
    issue = 'mbar_wait(bars+slot,(step/2)&1);named_bar_sync(1,128);fence_regs(v0);fence_regs(v1);wgmma_fence();'
    retire = 'wgmma_commit();wgmma_wait<0>();fence_regs(v0);fence_regs(v1);named_bar_sync(1,128);if(tid==0)mbar_arrive(bars+3+slot);'
    assert body.count(init) == body.count(issue) == body.count(retire) == 1
    if mode not in (3, 5):
        body = body.replace(init, init+'fence_regs(v0);fence_regs(v1);wgmma_fence();')
        body = body.replace(issue, 'mbar_wait(bars+slot,(step/2)&1);named_bar_sync(1,128);')
        body = body.replace(retire, '''wgmma_commit();
   // The preceding group's slot is reusable only after that group retires.
   // Accumulators are touched solely by ordered, same-shape WGMMA until drain.
   if(step){wgmma_wait<1>();named_bar_sync(1,128);if(tid==0)mbar_arrive(bars+3+((step-1)%2));}
   if(step==35){wgmma_wait<0>();fence_regs(v0);fence_regs(v1);named_bar_sync(1,128);if(tid==0)mbar_arrive(bars+3+slot);}
''')
    if mode in (2, 3, 4, 5):
        begin = 'for(int step=0;step<36;++step){int slot=step%2;uint8_t* buf='
        end = '  }\n  static_for<64>'
        assert body.count(begin) == body.count(end) == 1
        body = body.replace(begin, 'static_for<36>([&](auto ss){constexpr int step=decltype(ss)::value;int slot=step%2;uint8_t* buf=')
        body = body.replace(end, '  });\n  static_for<64>')
    if mode in (4, 5):
        start = body.index('static_for<36>')
        end = body.index('  static_for<64>', start)
        # Every consumer thread acquires the TMA ready barrier. The empty
        # signal follows completion of the entire older WGMMA group. Keep
        # all LN barriers, and retain the final WGMMA drain before LN.
        body = body[:start]+body[start:end].replace('named_bar_sync(1,128);','')+body[end:]
    return body
