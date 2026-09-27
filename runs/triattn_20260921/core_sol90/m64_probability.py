"""M64 form of the clean three-score / three-P row-independent pipeline."""
def transform(s):
 s=s.replace('M=128','M=64').replace('Shape<_128,_32>','Shape<_64,_32>')
 s=s.replace('2*Ratio','Ratio').replace('(seq/2)','seq')
 s=s.replace('2*nk','nk').replace('2*768/N','768/N').replace('2*L/N','L/N')
 s=s.replace('hh=decltype(half)::value','hh=0')
 s=s.replace('if(seq<2)exponentiate(sc,half,cute::true_type{});',
             'if(seq<1)exponentiate(sc,half,cute::true_type{});')
 s=s.replace('for(int hh=0;hh<2;hh++)','for(int hh=0;hh<1;hh++)')
 s=s.replace('hh=(idx/(64*N))%2','hh=0')
 s=s.replace('Output acc[2];clear(acc[0]);clear(acc[1]);','Output acc[1];clear(acc[0]);')
 s=s.replace('Den den[2];clear(den[0]);clear(den[1]);','Den den[1];clear(den[0]);')
 s=s.replace('warpgroup_fence_operand(acc[1]);','').replace('warpgroup_fence_operand(den[1]);','')
 s=s.replace('float mx[2][2]={{Safe?-INFINITY:0.f,Safe?-INFINITY:0.f},{Safe?-INFINITY:0.f,Safe?-INFINITY:0.f}};',
             'float mx[1][2]={{Safe?-INFINITY:0.f,Safe?-INFINITY:0.f}};')
 return s
