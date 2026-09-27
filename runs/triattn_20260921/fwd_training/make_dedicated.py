"""Dedicated TMA producers with explicit two-stage consumer release."""
from pathlib import Path

r = Path(__file__).resolve().parent
baseline = (r / 'cooperative_q1_s2/fused.cu').read_text()

for name, threads, min_ctas, registers in (
    ('producer_warp_s2', 160, 5, False),
    ('producer_wg_s2', 256, 4, True),
):
    s = baseline.replace('// Training forward: one score buffer, complete QK groups before scalar score reads.',
                         '// Training forward: dedicated TMA producer, independent consumer warpgroup.')
    s = s.replace('__launch_bounds__(128,5)', '__launch_bounds__(%d,%d)' % (threads, min_ctas))
    s = s.replace('),128,sizeof(Config::Shared)', '),%d,sizeof(Config::Shared)' % threads)
    begin = s.index('  auto shape=make_shape(L,_32{},_4{},L);')
    end = s.index('    Config::Score smma;', begin)
    loader = s[begin:s.index('  if(tid==0) {', begin)]
    qload = '''      s.qfull.arrive_and_expect_tx(2048*sizeof(Element));
      auto qq=local_tile(qg(_,_,h,rg),Shape<_64,_32>{},make_coord(qt,0));
      tma_load(p.q,qq,make_tensor(make_smem_ptr(s.q[0].data()),Config::QL{}),s.qfull);
      for(int kt=0;kt<nt;++kt) {
        int stage=kt%2;
        if(kt>=2) s.empty[stage][0].wait(((kt/2)-1)%2);
        load(kt);
      }
      // Producer may exit only after consumers have retired all uses of the stages.
      for(int kt=(nt>1?nt-2:0);kt<nt;++kt) s.empty[kt%2][0].wait((kt/2)%2);
'''
    dec = '    cutlass::arch::warpgroup_reg_dealloc<32>();\n' if registers else ''
    inc = '    cutlass::arch::warpgroup_reg_alloc<96>();\n' if registers else ''
    replacement = '  if(tid>=128) {\n' + dec + loader + '    if(tid==128) {\n' + qload + '    }\n  } else {\n' + inc + '    int r=0, lane=tid, row=rg;\n'
    s = s[:begin] + replacement + s[end:]
    s = s.replace('''        // QK retirement also completes PV(k-1) in every consuming warp.
        cutlass::arch::NamedBarrier::sync(128,r+1);
        if(tid==0 && kt>0 && kt+1<nt)load(kt+1);
''', '')
    s = s.replace('''        flash::gemm<false,-1>(pmma,pr,vb,out);''', '''        flash::gemm<false,0>(pmma,pr,vb,out);
        // All four consumer warps finish PV and bias reads before stage reuse.
        cutlass::arch::NamedBarrier::sync(128,1);
        if(lane==0) s.empty[stage][0].arrive();''')
    folder = r / name
    folder.mkdir(exist_ok=True)
    (folder / 'fused.cu').write_text(s)

# Control: keep six cooperative CTAs, but issue the refill after all score work.
s = baseline.replace('''        // QK retirement also completes PV(k-1) in every consuming warp.
        cutlass::arch::NamedBarrier::sync(128,r+1);
        if(tid==0 && kt>0 && kt+1<nt)load(kt+1);
''', '')
s = s.replace('''        auto vv=make_tensor''', '''        cutlass::arch::NamedBarrier::sync(128,1);
        if(tid==0 && kt>0 && kt+1<nt)load(kt+1);
        auto vv=make_tensor''')
folder = r / 'cooperative_s2_late'
folder.mkdir(exist_ok=True)
(folder / 'fused.cu').write_text(s)
