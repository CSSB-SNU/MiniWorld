"""Three complete cooperative consumer WGs, four scores and two P slots."""
from pathlib import Path
from r4_warp import transform as base
from r4_cooperative import transform as cooperate
HERE=Path(__file__).resolve().parent

def transform(qregs=False):
 s=cooperate(base())
 s=s.replace('Rows=4,Stages=2','Rows=3,Stages=2')
 s=s.replace('Consumers=512,Threads=512','Consumers=384,Threads=384')
 s=s.replace('Score sc[2];','Score sc[4];')
 s=s.replace('for(int n=0;n<2;++n)warpgroup_fence_operand(sc[n]);','for(int n=0;n<4;++n)warpgroup_fence_operand(sc[n]);')
 original=(HERE/'persistent_kv_body.cuh').read_text()
 a=original.index('  }else{\n   init(sc[0],0);')
 b=original.index('\n  auto id=',a)
 body=original[a:b]
 old='    warpgroup_fence_operand(sc[ci]);warpgroup_fence_operand(pp[ph]);'
 assert old in body
 body=body.replace(old,old+'''
    if(seq>=3)release(seq-3);
    if(tid==0 && seq+6<48)load_bias(seq+6);''')
 body=body.replace('47,_1{},cute::false_type{});drain();','47,_1{},cute::false_type{});drain();release(47);')
 a=s.index('  }else{\n   init(sc[0],0);');b=s.index('\n  auto id=',a)
 s=s[:a]+body+s[b:]
 # Preserve the established prefenced PV schedule: pack is BEFORE QK's WG.AR.
 if qregs:
  s=s.replace('GMMA::ss_op_selector<Element,Element,float,Shape<_64,_32,_32>>','GMMA::rs_op_selector<Element,Element,float,Shape<_64,_32,_32>>')
  a=original.index(' auto qshared=');b=original.index(' #pragma unroll 1\n for(int qt=',a)
  helpers=original[a:b]
  marker=' using Den=decltype(partition_fragment_C(ls,Shape<_64,_8>{}));\n'
  assert marker in s
  s=s.replace(marker,marker+helpers)
  a=original.index('  #pragma unroll\n  for(int hh=0;hh<2;++hh){\n   auto src=qthr')
  b=original.index('  warpgroup_fence_operand(acc[0]);',a)
  marker='  s.qr.wait(0);asm volatile("":::"memory");\n'
  assert marker in s
  s=s.replace(marker,marker+original[a:b])
  old='''auto kb=tq.partition_fragment_B(ks);
   auto qs=local_tile(make_tensor(make_smem_ptr(s.q[wg]),SQ{}),Shape<_64,_32>{},make_coord(hh,0));
   auto qa=tq.partition_fragment_A(qs);'''
  assert old in s
  s=s.replace(old,'auto kb=tq.partition_fragment_B(ks);auto& qa=hh==0?qa0:qa1;')
 return s
