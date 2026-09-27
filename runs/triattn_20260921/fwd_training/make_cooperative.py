"""Trade two-row bias sharing for four resident cooperative TMA/WGMMA CTAs."""
from pathlib import Path
root=Path(__file__).resolve().parent
s=(root/'stream_r2_pipe/fused.cu').read_text()
s=s.replace('static constexpr int R=2;', 'static constexpr int R=1;')
s=s.replace('k[2][R], v[2][R]', 'k[3][R], v[3][R]')
s=s.replace('bias[2]', 'bias[3]').replace('full[2][R], bfull[2]', 'full[3][R], bfull[3]')
s=s.replace('empty[2][R], bempty[2]', 'empty[3][R], bempty[3]')
s=s.replace('for(int b=0;b<2;++b)', 'for(int b=0;b<3;++b)')
s=s.replace('__launch_bounds__(384,1)', '__launch_bounds__(128,4)')
begin=s.index('  if(tid<128) {')
end=s.index('    Config::Score smma;',begin)
replacement=r'''
  auto shape=make_shape(L,_32{},_4{},L);
  auto qg=p.q.get_tma_tensor(shape), kg=p.k.get_tma_tensor(shape), vg=p.v.get_tma_tensor(shape);
  auto bg=p.bias.get_tma_tensor(make_shape(L,L,_4{}));
  auto load=[&](int kt) {
    int stage=kt%3;
    s.full[stage][0].arrive_and_expect_tx(2*2048*sizeof(Element));
    s.bfull[stage].arrive_and_expect_tx(4096*sizeof(Element));
    auto kk=local_tile(kg(_,_,h,rg),Shape<_64,_32>{},make_coord(kt,0));
    auto vv=local_tile(vg(_,_,h,rg),Shape<_64,_32>{},make_coord(kt,0));
    auto bb=local_tile(bg(_,_,h),Shape<_64,_64>{},make_coord(qt,kt));
    tma_load(p.k,kk,make_tensor(make_smem_ptr(s.k[stage][0].data()),Config::QL{}),s.full[stage][0]);
    tma_load(p.v,vv,make_tensor(make_smem_ptr(s.v[stage][0].data()),Config::QL{}),s.full[stage][0]);
    tma_load(p.bias,bb,make_tensor(make_smem_ptr(s.bias[stage].data()),Config::BL{}),s.bfull[stage]);
  };
  if(tid==0) {
    s.qfull.arrive_and_expect_tx(2048*sizeof(Element));
    auto qq=local_tile(qg(_,_,h,rg),Shape<_64,_32>{},make_coord(qt,0));
    tma_load(p.q,qq,make_tensor(make_smem_ptr(s.q[0].data()),Config::QL{}),s.qfull);
    load(0);if(nt>1)load(1);if(nt>2)load(2);
  }
  {
    int r=0, lane=tid, row=rg;
'''
s=s[:begin]+replacement+s[end:]
s=s.replace('constexpr bool First=decltype(first_)::value, Last=decltype(last_)::value;',
            'constexpr bool First=decltype(first_)::value, Last=decltype(last_)::value;\n        int stage=kt%3;')
s=s.replace('s.bfull[slot].wait((kt/2)%2)', 's.bfull[stage].wait((kt/3)%2)')
s=s.replace('s.bias[slot].data()', 's.bias[stage].data()')
old='''        if(lane==0) {
          s.bempty[slot].arrive();
          if constexpr (!First) s.empty[slot^1][r].arrive();
        }'''
assert old in s
s=s.replace(old,'''        if constexpr (!First) {
          if(tid==0 && kt+2<nt)load(kt+2);
        }''')
s=s.replace('s.full[slot^1][r].wait(((kt+1)/2)%2)', 's.full[(kt+1)%3][r].wait(((kt+1)/3)%2)')
s=s.replace('s.k[slot^1][r].data()', 's.k[(kt+1)%3][r].data()')
s=s.replace('s.v[slot][r].data()', 's.v[stage][r].data()')
s=s.replace('),384,sizeof(Config::Shared)', '),128,sizeof(Config::Shared)')
out=root/'cooperative_q2';out.mkdir(exist_ok=True)
(out/'fused.cu').write_text(s)
