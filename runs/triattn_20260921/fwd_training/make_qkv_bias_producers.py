"""Six projection WGs become four 104-register consumers and two 32-register producers."""
import json
from pathlib import Path
R = Path(__file__).resolve().parent
s = (R/'qkv_compact_retire6/fused.cu').read_text()
s = s.replace('template<int Capacity,int Consumers,int Stages,int Projectors> struct Config {',
'''template<int Capacity,int Consumers,int Stages,int Projectors> struct Config {
  static constexpr int Attn=Consumers==6?4:Consumers;''', 1)
s = s.replace('q[Consumers];', 'q[Attn];').replace('bias[Consumers][Stages];', 'bias[Attn][Stages];')
s = s.replace('bfull[Consumers][Stages];', 'bfull[Consumers][Stages],bempty[Consumers][Stages];')
s = s.replace('s.bfull[cc][b].init(1);', '{s.bfull[cc][b].init(1);s.bempty[cc][b].init(1);}', 1)
needle = '''  auto bg=p.bias.get_tma_tensor(make_shape(L,L,_4{}));
  typename C::Score smma; typename C::PV pmma;'''
assert needle in s
s = s.replace(needle, '''  auto bg=p.bias.get_tma_tensor(make_shape(L,L,_4{}));
  if constexpr(Consumers==6) {
    if(c>=C::Attn) {
      cutlass::arch::warpgroup_reg_dealloc<32>();
      if(lane==0) {
        for(int qi=0;qi<(nt+C::Attn-1)/C::Attn;++qi) {
          for(int kt=0;kt<nt;++kt) {
            int stage=kt%Stages,cycle=qi*(nt/Stages)+kt/Stages;
            #pragma unroll
            for(int cc=c-C::Attn;cc<C::Attn;cc+=2) {
              int qt=cc+qi*C::Attn;
              if(qt>=nt)continue;
              if(cycle>0)s.bempty[cc][stage].wait((cycle-1)%2);
              s.bfull[cc][stage].arrive_and_expect_tx(4096*sizeof(Element));
              auto bb=local_tile(bg(_,_,h),Shape<_64,_64>{},make_coord(qt,kt));
              tma_load(p.bias,bb,make_tensor(make_smem_ptr(s.scratch.attn.bias[cc][stage].data()),typename C::BL{}),s.bfull[cc][stage]);
            }
          }
        }
      }
      return;
    }
    cutlass::arch::warpgroup_reg_alloc<104>();
  }
  typename C::Score smma; typename C::PV pmma;''', 1)
s = s.replace('for(int qt=c;qt<nt;qt+=Consumers)', 'for(int qt=c;qt<nt;qt+=C::Attn)')
s = s.replace('for(int b=0;b<Stages && b<nt;++b)load_bias(b);',
              'if constexpr(Consumers!=6)for(int b=0;b<Stages && b<nt;++b)load_bias(b);', 1)
s = s.replace('s.qfull[c].wait((qt/Consumers)%2);', 's.qfull[c].wait((qt/C::Attn)%2);', 1)
s = s.replace('if constexpr(Stages==2) {', 'if constexpr(Consumers!=6 && Stages==2) {', 1)
s = s.replace('((qt/Consumers)*(nt/Stages)+kt/Stages)%2', '((qt/C::Attn)*(nt/Stages)+kt/Stages)%2', 1)
needle = '''        if constexpr(Stages==1) {
          cutlass::arch::NamedBarrier::sync(128,c+1);
          if(lane==0 && kt+1<nt)load_bias(kt+1);
        }'''
assert needle in s
s = s.replace(needle, '''        if constexpr(Consumers==6) {
          // Every lane has read the current bias before producer overwrites it.
          cutlass::arch::NamedBarrier::sync(128,c+1);
          if(lane==0)s.bempty[c][stage].arrive();
        } else if constexpr(Stages==1) {
          cutlass::arch::NamedBarrier::sync(128,c+1);
          if(lane==0 && kt+1<nt)load_bias(kt+1);
        }''', 1)
s = s.replace('launch<1024,6,1,4>', 'launch<1024,6,2,4>')
s = s.replace('Config<1024,6,1,4>::Shared', 'Config<1024,6,2,4>::Shared')
for name, code in [('qkv_compact_biasprod', s),
                   ('qkv_compact_biasprod5', s.replace('launch<768,6,2,6>', 'launch<768,6,3,5>'))]:
    target=R/name
    assert not (target/'build-ready.json').exists()
    target.mkdir(exist_ok=True)
    (target/'fused.cu').write_text(code)
    (target/'register-roles.json').write_text(json.dumps({
        '384': {'consumers': 4, 'producers': 2},
        '768': {'consumers': 4, 'producers': 2},
        '1024': {'consumers': 4, 'producers': 2},
    }, indent=2)+'\n')
